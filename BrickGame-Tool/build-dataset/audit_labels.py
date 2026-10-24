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
        n_lab = sum(1 for line in open(lab, encoding="utf-8")
                    if line.split() and line.split()[0] == "1")
        res = model.predict(img, imgsz=640, conf=args.conf, verbose=False)[0]
        n_mod = sum(1 for c in res.boxes.cls.cpu().numpy() if int(c) == 1)
        cells = (m["rows"] or 0) * (m["cols"] or 0)
        out[m["file"]] = {"label": n_lab, "model": n_mod, "cells": cells,
                          "rows": m["rows"], "cols": m["cols"], "reviewed": m["reviewed"]}
        # 标签已经接近满盘时，模型多出来的只能是误检（一格最多一块砖），不算漏标
        if cells and n_lab >= cells * 0.95:
            continue
        if n_mod - n_lab >= args.gap:
            suspects.append((m["src"], n_lab, n_mod, n_mod - n_lab, cells))

    json.dump({"weights": os.path.basename(args.weights),
               "at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "gap": args.gap, "frames": out},
              open(os.path.join(YOLO_DIR, "audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"复检 {len(out)} 帧，疑似漏标 {len(suspects)} 帧"
          f"（模型比标签多 ≥{args.gap} 块，且标签未满盘）")
    for src, a, b, d, cells in sorted(suspects, key=lambda x: -x[3])[:12]:
        print(f"  标签 {a:3d}/{cells or '?'} 模型 {b:3d} 差 {d:3d}  {src}")
    print(f"结果 → {os.path.join(YOLO_DIR, 'audit.json')}")


if __name__ == "__main__":
    main()
