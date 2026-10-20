#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 box_tool.html 导出的 box_fixes.json 写回 YOLO 标签。

deleted 是按当时页面里那个顺序的索引（0 起），added 是归一化 xyxy —— 转成 YOLO 的 cxcywh 追加。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/apply_box_fixes.py --fixes ~/Downloads/box_fixes.json
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO = os.path.join(HERE, "yolo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    fixes = json.load(open(args.fixes, encoding="utf-8"))
    if not fixes:
        print("没有修改，什么都没做")
        return
    total_del = total_add = 0
    for base, fix in fixes.items():
        path = os.path.join(YOLO, "labels", args.split, base + ".txt")
        if not os.path.exists(path):
            print(f"[skip] 找不到 {base}.txt")
            continue
        rows = [l for l in open(path) if l.strip()]
        keep = [l for i, l in enumerate(rows) if i not in set(fix.get("deleted", []))]
        add = []
        for x1, y1, x2, y2 in fix.get("added", []):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            add.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        with open(path, "w") as f:
            f.writelines(keep + add)
        total_del += len(rows) - len(keep)
        total_add += len(add)
        print(f"{base}: 删 {len(rows) - len(keep)}，补 {len(add)}，剩 {len(keep) + len(add)}")
    print(f"合计：删 {total_del} 个框，补 {total_add} 个")
    print("下一步：train_yolo.py --resume runs/brick/weights/best.pt --epochs 40")


if __name__ == "__main__":
    main()
