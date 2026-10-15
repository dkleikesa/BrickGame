#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地起一个服务跑去重页面：点「删除选中」直接在数据集上生效，不用导 JSON 再跑命令。

为什么要服务：file:// 打开的页面碰不到文件系统（也写不了 index.csv）。
页面本身还是 review_dataset.py 生成的那套，只是多接两个接口：
  POST /api/delete  {paths:[...]} → 摘 index.csv 的行 + 图片挪进 dataset/deleted/（--purge 才真删）
  POST /api/undo                  → 撤销上一次删除（行插回、图片挪回原位）
  GET  /api/state                 → 有没有可撤销的批次（页面据此显示"撤销上次删除"）
每次删除/撤销后会重新生成去重页面，**并顺手重刷 annotate.html**：
删掉的图片在旧标注界面里还在，img 找不到文件就是一片空白（尤其那些还没标完的帧，图刚被删、
页面又没重生成，看着就像坏了）。annotate.html 要 numpy/cv2，所以用 --py 指定的解释器起子进程跑
（默认自动找 ~/.venvs/brick-train 或本目录 .venv），后台合并刷新，不阻塞删除请求。

**只监听 127.0.0.1**：这接口能删数据，别暴露到局域网。
落地逻辑复用 apply_deletions.py（那里也有纯命令行的入口）。

用法：
  .venv/bin/python review_server.py                 # 起服务并打开浏览器
  .venv/bin/python review_server.py --port 8765 --purge   # --purge：确认即真删（不可撤销）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import apply_corrections as C
import apply_deletions as D
import review_dataset as R

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                     # BrickGame-Tool：静态文件根（dataset/ 与 build-catalog/ 都在下面）
_LOCK = threading.Lock()
LAST: dict = {}                                    # 上一批被删的行，供撤销


def default_python() -> str:
    """重刷 annotate.html 用的解释器：需要 numpy/cv2，系统 python3 通常没有。"""
    if _PY["override"]:
        return _PY["override"]
    for p in (os.path.expanduser("~/.venvs/brick-train/bin/python"),
              os.path.join(_HERE, ".venv", "bin", "python"),
              os.path.join(_ROOT, ".venv", "bin", "python")):
        if os.path.exists(p):
            return p
    return sys.executable


_LOCK_R = threading.Lock()
_REFRESH = {"busy": False, "again": False, "last": ""}
_PY = {"override": ""}
REFRESH_ON = True
ANNOTATE = os.path.join(_HERE, "annotate.html")


def refresh_annotate() -> None:
    """后台重刷标注界面；连续删除会自动合并成一次。"""
    if not REFRESH_ON:
        return
    with _LOCK_R:
        if _REFRESH["busy"]:
            _REFRESH["again"] = True
            return
        _REFRESH["busy"] = True
    threading.Thread(target=_refresh_loop, daemon=True).start()


def _refresh_loop() -> None:
    while True:
        cmd = [default_python(), os.path.join(_HERE, "visualize_groups.py"),
               "--dataset", Handler.dataset, "--catalog", Handler.catalog, "--html", ANNOTATE]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=_HERE)
            tail = [ln for ln in (r.stderr or "").strip().splitlines() if ln]
            msg = ("已重刷 annotate.html" if r.returncode == 0
                   else f"annotate.html 重刷失败：{tail[-1] if tail else r.returncode}")
        except OSError as e:
            msg = f"annotate.html 重刷失败：{e}"
        with _LOCK_R:
            _REFRESH["last"] = msg
            print(f"[刷新] {msg}")
            if _REFRESH["again"]:
                _REFRESH["again"] = False
                continue
            _REFRESH["busy"] = False
            return


class Handler(SimpleHTTPRequestHandler):
    """静态文件走 BrickGame-Tool 根目录，另外接 /api/*。"""

    dataset = os.path.join(_HERE, "dataset")
    catalog = os.path.join(_ROOT, "build-catalog", "catalog")
    html = os.path.join(_HERE, "review.html")
    purge = False

    def log_message(self, fmt, *args):             # 别把每个图片请求都刷屏
        if "/api/" in self.path or self.path in ("/", "/review.html"):
            super().log_message(fmt, *args)

    def end_headers(self):
        if "review.html" in self.path or self.path == "/":
            self.send_header("Cache-Control", "no-store")   # 删完刷新必须拿到新页面
        super().end_headers()

    def do_GET(self):
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/build-dataset/review.html")
            self.end_headers()
            return
        # 页面每次加载都按当前 index.csv 重新生成：标注写回后一刷新就是新分类，
        # 不会像静态文件那样"刷新了分类还是错的"（生成一页约 0.1s）。
        if os.path.abspath(os.path.join(_ROOT, self.path.lstrip("/"))) == os.path.abspath(self.html):
            with _LOCK:
                R.build_review(self.dataset, self.catalog, self.html, quiet=True,
                               base=_HERE, serve=True, purge=self.purge)
            return self._send_file(self.html)
        if self.path == "/api/state":
            with _LOCK:
                n = len(LAST.get("rows", []))
            with _LOCK_R:
                note = _REFRESH["last"]
            # rev：index.csv 的指纹。页面轮询它就能发现"标注页刚写回了/别处改了数据"，
            # 提示（或自动）刷新 —— 两个页面各开一个标签时不用自己盯着。
            return self._json({"undo_n": n, "purge": self.purge, "annotate": note, "rev": self._rev()})
        super().do_GET()

    def do_POST(self):
        # 去重页的抽屉里改标注 → 直接写回 index.csv（与标注页同一个实现）
        if self.path.rstrip("/").endswith("/api/corrections"):
            return self._corrections()
        if self.path not in ("/api/delete", "/api/undo"):
            return self.send_error(404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            return self._json({"error": f"请求体不是 JSON：{e}"}, 400)
        if self.path == "/api/delete":
            return self._delete(body)
        return self._undo()

    def _delete(self, body: dict):
        paths = [p for p in (body.get("paths") or []) if isinstance(p, str) and p]
        if not paths:
            return self._json({"error": "没有要删的路径"}, 400)
        with _LOCK:
            res = D.apply_paths(self.dataset, paths, purge=self.purge)
            if res["matched"]:
                # 上一批被顶掉就当它已经确认过，不再给撤销入口
                LAST.clear()
                LAST.update({"rows": res["rows"], "purge": res["purged"],
                             "keep_files": res["keep_files"]})
            R.build_review(self.dataset, self.catalog, self.html, quiet=True,
                           base=_HERE, serve=True, purge=self.purge)
        out = {k: v for k, v in res.items() if k != "rows"}
        print(f"[删除] 命中 {res['matched']} / 剩余 {res['total_after']}，"
              f"{'真删' if res['purged'] else '挪走'} {res['files']} 个文件"
              f"（无效 {len(res['missing'])}）")
        if res["matched"]:
            refresh_annotate()               # 标注界面得跟着重刷，否则删掉的图在那里是空白
            out["annotate"] = "refreshing"
        return self._json(out)

    def _undo(self):
        with _LOCK:
            rows = LAST.get("rows", [])
            if not rows:
                return self._json({"error": "没有可撤销的删除（重启服务或已被新的删除顶掉）"}, 400)
            res = D.restore_rows(self.dataset, rows, move_back=not LAST.get("purge"))
            LAST.clear()
            R.build_review(self.dataset, self.catalog, self.html, quiet=True,
                           base=_HERE, serve=True, purge=self.purge)
        print(f"[撤销] 恢复 {res['restored']} 行，图片挪回 {res['files']} 个")
        refresh_annotate()
        return self._json(res)

    def _rev(self) -> str:
        try:
            st = os.stat(os.path.join(self.dataset, "index.csv"))
            return f"{st.st_mtime_ns}-{st.st_size}"
        except OSError:
            return ""

    def _send_file(self, path: str):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _corrections(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            corr = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            return self._json({"error": f"请求体不是 JSON：{e}"}, 400)
        try:
            with _LOCK:
                res = C.apply(self.dataset, corr, inplace=True)
                R.build_review(self.dataset, self.catalog, self.html, quiet=True,
                               base=_HERE, serve=True, purge=self.purge)
        except Exception as e:                      # 任何异常都回给页面，别静默失败
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        print(f"[改标注] 指派 {res['n_set']} / 标未知 {res['n_unk']} / 无变化 {res['n_same']} / "
              f"无效 {res['n_missing']}")
        refresh_annotate()                          # 标注页也要跟着一致
        return self._json({"ok": True, **res})

    def _json(self, obj: dict, code: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="去重页面 + 直接删除接口（只监听本机）")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--html", default=os.path.join(_HERE, "review.html"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--purge", action="store_true",
                    help="确认即真删图片文件（不可撤销，默认挪进 dataset/deleted/）")
    ap.add_argument("--py", default="", help="重刷 annotate.html 用的解释器（默认自动找 brick-train / .venv）")
    ap.add_argument("--no-annotate", action="store_true", help="删完不重刷 annotate.html（省一次子进程）")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()
    global REFRESH_ON
    _PY["override"] = args.py
    REFRESH_ON = not args.no_annotate

    Handler.dataset, Handler.catalog, Handler.html, Handler.purge = (
        args.dataset, args.catalog, args.html, args.purge)
    # base=_HERE 而不是静态根 _ROOT：同一份 review.html 在 file:// 和 http:// 下都能显示图。
    # 页面在 /build-dataset/review.html 时，"dataset/..." → /build-dataset/dataset/...（=_HERE 下的 dataset），
    # "../build-catalog/..." → /build-catalog/...（自动落到静态根里的 build-catalog）。
    # 反过来若按 _ROOT 生成（"build-dataset/dataset/..."），file:// 打开就会多套一层 → 图全裂。
    R.build_review(args.dataset, args.catalog, args.html,
                   base=_HERE, serve=True, purge=args.purge)

    url = f"http://127.0.0.1:{args.port}/build-dataset/review.html"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=_ROOT))
    print(f"去重页面: {url}\n"
          f"  删除方式: {'真删文件（--purge）' if args.purge else '图片挪进 dataset/deleted/（可撤销）'}\n"
          f"  标注界面: {ANNOTATE}（删完自动重刷"
          f"{'，已关 --no-annotate' if not REFRESH_ON else '，用 ' + default_python() + '）'}\n"
          f"  数据集: {args.dataset}\n  Ctrl+C 退出", flush=True)
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出（页面请用 Ctrl+C 之外的方式关闭；数据改动都已落盘）")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
