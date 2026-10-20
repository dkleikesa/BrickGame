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
SPLIT = "train"


def label_path(base: str) -> str:
    return os.path.join(YOLO, "labels", SPLIT, base + ".txt")


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
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(keep + new)
        deleted += len(rows) - len(keep)
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
        snapshot(bases)   # 首次保存先留底
        restore(bases)    # 每次从原始版本重算，避免重复叠加
        deleted, added = apply(fixes)
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
