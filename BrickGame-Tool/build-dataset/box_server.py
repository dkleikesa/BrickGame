#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
框标注服务：页面点「保存」直接写回 yolo/labels，不用再导出 JSON。

与分类标注的 annotate_server 同一套路（本地 HTTP + 页面 POST），差别只在写回格式。

幂等：第一次保存前把原始 labels 备份成 *.orig，之后每次保存都从 .orig 重算并叠加修改，
所以反复点「保存」不会把补的框加两遍。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/box_server.py
  # 自动打开 http://127.0.0.1:8771
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import webbrowser
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
YOLO = os.path.join(HERE, "yolo")
HTML = os.path.join(HERE, "box_tool.html")
PORT = 8771
REVIEWED = os.path.join(YOLO, "reviewed.json")


def label_path(base: str) -> str:
    """帧的框文件在 train 还是 val 里都可能（split 会变），按实际存在的那个改。"""
    for split in ("train", "val"):
        p = os.path.join(YOLO, "labels", split, base + ".txt")
        if os.path.exists(p):
            return p
    return os.path.join(YOLO, "labels", "train", base + ".txt")


def mark_reviewed(on, off) -> None:
    """记下"人看过了"，取消勾选时移除。

    以页面的 ok 为准：改了框但没点「确认无误」不算确认，
    否则未过目的框会跟着 reviewed 一起进训练集。
    """
    frames = set()
    if os.path.exists(REVIEWED):
        try:
            frames = set(json.load(open(REVIEWED, encoding="utf-8")).get("frames", []))
        except (json.JSONDecodeError, OSError):
            frames = set()
    frames.update(on)
    frames.difference_update(off)
    with open(REVIEWED, "w", encoding="utf-8") as fh:
        json.dump({"frames": sorted(frames)}, fh, ensure_ascii=False, indent=1)


def restore(bases) -> None:
    """把 .orig 还原回 labels（首次编辑时才有 .orig）。"""
    for base in bases:
        p, orig = label_path(base), label_path(base) + ".orig"
        if os.path.exists(orig):
            with open(orig, "rb") as f, open(p, "wb") as g:
                g.write(f.read())


def snapshot(bases) -> None:
    for base in bases:
        p, orig = label_path(base), label_path(base) + ".orig"
        if os.path.exists(p) and not os.path.exists(orig):
            with open(p, "rb") as f, open(orig, "wb") as g:
                g.write(f.read())


def apply(fixes: dict) -> tuple[int, int]:
    """按 fixes 重写 labels，返回（删除数, 新增数）。"""
    deleted = added = 0
    for base, fix in fixes.items():
        # 只是"确认无误"（没改框）的帧不重写文件，也不留 .orig
        if not (fix.get("deleted") or fix.get("added") or fix.get("board")):
            continue
        p = label_path(base)
        if not os.path.exists(p):
            continue
        rows = [l for l in open(p, encoding="utf-8") if l.strip()]
        drop = set(fix.get("deleted", []))
        keep = [l for i, l in enumerate(rows) if i not in drop]
        new = []
        for x1, y1, x2, y2 in fix.get("added", []):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            new.append(f"1 {cx:.6f} {cy:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
        board = fix.get("board")
        replaced_board = 0
        if board:
            # 重画棋盘：整帧只有 1 个 board，先剔掉旧的 0 类框再写新的。
            # 旧 board 是"被替换"不是"被删除"，别算进 deleted 的提示里。
            x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in board]
            replaced_board = sum(1 for l in keep if l.lstrip().startswith("0 "))
            keep = [l for l in keep if not l.lstrip().startswith("0 ")]
            new.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                       f"{x2 - x1:.6f} {y2 - y1:.6f}\n")
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(keep + new)
        deleted += len(rows) - len(keep) - replaced_board
        added += len(new)
    return deleted, added


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静音，别把终端刷满

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/box_tool.html"):
            self._send(200, open(HTML, "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/data":
            # 每次请求现算：补了新数据只要刷新页面，不用重新生成 HTML
            import make_box_tool
            body = json.dumps(make_box_tool.build_data(), ensure_ascii=False).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            fixes = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json")
            return
        bases = list(fixes.keys())
        changed = [b for b in bases
                   if fixes[b].get("deleted") or fixes[b].get("added") or fixes[b].get("board")]
        snapshot(changed)   # 首次保存先留底（只对真的改了框的帧）
        restore(changed)    # 每次从原始版本重算，避免重复叠加
        deleted, added = apply(fixes)
        mark_reviewed([b for b in bases if fixes[b].get("ok")],
                      [b for b in bases if not fixes[b].get("ok")])
        body = json.dumps({"ok": True, "frames": len(bases), "deleted": deleted, "added": added},
                          ensure_ascii=False).encode()
        self._send(200, body, "application/json")


def main() -> None:
    if not os.path.exists(HTML):
        raise SystemExit("先跑 make_box_tool.py 生成页面")
    url = f"http://127.0.0.1:{PORT}/"
    print(f"标注服务：{url}（Ctrl+C 结束）")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
