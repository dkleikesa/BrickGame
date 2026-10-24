#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用当前模型复检标签，找出疑似漏标（模型能检出、标签里没有的）。

覆盖率只能看"框数占格子比例"，残局帧本来就低，会产生大量误报。
真正的漏标判据是：同一帧上模型检出的砖明显多于标签 —— 说明当时没标全。

产物 yolo/audit.json 供 make_box_tool.py 决定哪些已确认帧要拉回复查。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/audit_labels.py [--gap 10]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(HERE, "yolo")
SAMPLES = os.path.join(HERE, "samples")
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(HERE), "runs", "brick", "weights", "best.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--gap", type=int, default=10, help="模型比标签多这么多块砖才算漏标")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f"找不到模型：{args.weights}")
    meta = json.load(open(os.path.join(YOLO_DIR, "meta.json"), encoding="utf-8"))
    model = YOLO(args.weights)

    out, suspects = {}, []
    for m in meta:
        img = next((os.path.join(SAMPLES, m["src"] + e) for e in (".png", ".jpg", ".jpeg")
                    if os.path.exists(os.path.join(SAMPLES, m["src"] + e))), None)
        if not img:
            continue
        lab = os.path.join(YOLO_DIR, "labels", m["split"], m["file"] + ".txt")
        if not os.path.exists(lab):
            continue
        rows_raw = [line.split() for line in open(lab, encoding="utf-8") if line.split()]
        lab_t = [list(map(float, q[1:5])) for q in rows_raw if q[0] == "1"]
        lab_b = [[list(map(float, q[1:5])) for q in rows_raw if q[0] == "0"]]
        res = model.predict(img, imgsz=640, conf=args.conf, iou=0.45, verbose=False)[0]
        H, W = res.orig_shape
        mod_t = [[(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
                 for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                                res.boxes.cls.cpu().numpy()) if int(c) == 1]

        # 一格最多一块砖：先把框映射成格子集合再比，否则模型对同一格重复输出
        # 会被当成"多检出"，把正常标签误判成漏标（实测 12 帧全是这么来的）。
        extra = None
        if m["rows"] and m["cols"] and lab_b:
            def cells(boxes):
                bx, by, bw, bh = lab_b[0][0]
                x0, y0 = bx - bw / 2, by - bh / 2
                got = set()
                for cx, cy, _, _ in boxes:
                    c = int((cx - x0) / (bw / m["cols"])); r = int((cy - y0) / (bh / m["rows"]))
                    if 0 <= r < m["rows"] and 0 <= c < m["cols"]:
                        got.add((r, c))
                return got
            extra = len(cells(mod_t) - cells(lab_t))

        cells = (m["rows"] or 0) * (m["cols"] or 0)
        out[m["file"]] = {"label": len(lab_t), "model": len(mod_t), "cells": cells,
                          "extra_cells": extra,
                          "rows": m["rows"], "cols": m["cols"], "reviewed": m["reviewed"]}
        if extra is not None:
            if extra >= args.gap:
                suspects.append((m["src"], len(lab_t), len(mod_t), extra, cells))
        elif len(mod_t) - len(lab_t) >= args.gap:
            suspects.append((m["src"], len(lab_t), len(mod_t), len(mod_t) - len(lab_t), cells))

    json.dump({"weights": os.path.basename(args.weights),
               "at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "gap": args.gap, "frames": out},
              open(os.path.join(YOLO_DIR, "audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"复检 {len(out)} 帧，疑似漏标 {len(suspects)} 帧"
          f"（按格子比，模型多出 ≥{args.gap} 格才报）")
    for src, a, b, d, cells in sorted(suspects, key=lambda x: -x[3])[:12]:
        print(f"  标签 {a:3d} 框 / {cells or '?'} 格，模型多占 {d:3d} 格  {src}")
    print(f"结果 → {os.path.join(YOLO_DIR, 'audit.json')}")


if __name__ == "__main__":
    main()
