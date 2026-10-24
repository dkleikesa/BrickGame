#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标注页的本地服务：页面里点「本帧标完」直接写回 index.csv / frames.csv，不用再导出 JSON。

为什么要有它：annotate.html 是静态页面，浏览器不许它写本地文件 —— 老流程因此是
"导出 corrections.json → 手动跑 apply_corrections.py"，中间那一步纯属多余。这里起一个
只监听 127.0.0.1 的 http.server：
  ① 静态：把 BrickGame-Tool/ 当根目录（页面里的相对路径 ../build-catalog/catalog/tiles、
     dataset/images/...、samples/... 原样可用）；
  ② POST /api/corrections：直接调 apply_corrections.apply() 落库 —— **和手跑脚本同一份实现**；
  ③ 落库后后台重出 annotate.html（临时文件 + os.replace，避免半截页面被浏览器读到），
     于是已经标完的帧下次刷新就不再出现，增量闭环不用手点。

安全：只绑 127.0.0.1（外部访问不到），只改 dataset/index.csv 与 dataset/frames.csv；
请求体里出现 index.csv 里没有的 path 一律按"无效"跳过并回报条数。

用法：
  .venvs/brick-train/bin/python annotate_server.py                  # 重出页面 → 起服务 → 开浏览器
  .venvs/brick-train/bin/python annotate_server.py --port 8770 --no-open
  # 页面地址：http://127.0.0.1:8770/build-dataset/annotate.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # BrickGame-Tool：静态根目录
sys.path.insert(0, _HERE)
import apply_corrections                                    # noqa: E402  落库的唯一实现
import visualize_groups                                     # noqa: E402  重出页面

REGEN_LOCK = threading.Lock()
_regen_timer: threading.Timer | None = None
_timer_lock = threading.Lock()


def schedule_regen(html: str, dataset: str, catalog: str, samples: str, quiet: bool) -> None:
    """防抖 1.2s 后重出页面。

    页面现在每改一格就落库一次，直接每次都重出会把 CPU 打满；合并成一次即可，
    重出的目的只是让"刷新后能看到最新台账"。
    """
    global _regen_timer
    with _timer_lock:
        if _regen_timer is not None:
            _regen_timer.cancel()
        _regen_timer = threading.Timer(1.2, regen,
                                       args=(html, dataset, catalog, samples, quiet))
        _regen_timer.daemon = True
        _regen_timer.start()


class Handler(SimpleHTTPRequestHandler):
    """静态文件 + 一个写回接口。"""

    dataset = os.path.join(_HERE, "dataset")
    catalog = os.path.join(_ROOT, "build-catalog", "catalog")
    samples = os.path.join(_HERE, "samples")
    html = os.path.join(_HERE, "annotate.html")
    quiet = False

    def do_GET(self) -> None:                      # noqa: N802
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/build-dataset/annotate.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:                     # noqa: N802
        # 页面用相对路径 'api/corrections' 发请求，从 /build-dataset/ 看就是
        # /build-dataset/api/corrections —— 所以按后缀匹配，换个挂载点也不会失效。
        if not self.path.rstrip("/").endswith("/api/corrections"):
            self.send_error(404, "unknown endpoint")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            corr = json.loads(self.rfile.read(n) or b"{}")
            res = apply_corrections.apply(self.dataset, corr, inplace=True)
        except Exception as e:                     # 任何异常都回给页面，别静默失败
            body = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                              ensure_ascii=False).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps({"ok": True, **res}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if not self.quiet:
            print(f"  写回: 指派 {res['n_set']} / 标未知 {res['n_unk']} / 无变化 {res['n_same']} / "
                  f"无效 {res['n_missing']}；标完 {len(res['done'])} 帧"
                  f"{'（' + '、'.join(res['done']) + '）' if res['done'] else ''}")
        # 防抖重出：连续改格子只会触发一次重建，刷新页面时才是最新台账
        schedule_regen(self.html, self.dataset, self.catalog, self.samples, self.quiet)

    def log_message(self, fmt: str, *args) -> None:
        if not self.quiet:
            super().log_message(fmt, *args)


def regen(html: str, dataset: str, catalog: str, samples: str, quiet: bool) -> None:
    """重出 annotate.html。先写临时文件再原子替换：别让浏览器读到写了一半的页面。"""
    with REGEN_LOCK:
        tmp = html + ".tmp"
        try:
            visualize_groups.build_html(dataset, catalog, samples, tmp,
                                        only_pending=True, quiet=True)
            os.replace(tmp, html)
            if not quiet:
                print(f"  已重出 {os.path.basename(html)}（标完的帧不再出现）")
        except Exception as e:
            print(f"  重出标注界面失败（不影响已写回的标注）：{type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="标注页本地服务（点『本帧标完』直接写回）")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--samples", default=os.path.join(_HERE, "samples"))
    ap.add_argument("--html", default=os.path.join(_HERE, "annotate.html"))
    ap.add_argument("--no-regen", action="store_true", help="不重出页面（直接用现有 annotate.html）")
    ap.add_argument("--no-open", action="store_true", help="不自动开浏览器")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    Handler.dataset, Handler.catalog, Handler.samples = args.dataset, args.catalog, args.samples
    Handler.html, Handler.quiet = args.html, args.quiet
    if not args.no_regen:
        print("重出标注界面…")
        regen(args.html, args.dataset, args.catalog, args.samples, args.quiet)

    url = f"http://127.0.0.1:{args.port}/build-dataset/annotate.html"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              partial(Handler, directory=_ROOT))
    print(f"标注页: {url}\n（点『本帧标完』直接写回 dataset/index.csv + frames.csv；Ctrl-C 结束）")
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
