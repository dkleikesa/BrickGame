#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给"没有逐格标注"的新帧生成 YOLO 预标注（board + tile）。

工业化补数据的核心一步：新截图进来时没人愿意逐格点 140 个框，所以先用当前模型跑一遍，
人工只做"删掉明显错的框 / 补漏掉的框"。产物落在 yolo/prelabel/<帧名>.txt，
由 make_yolo_data.py 装配进数据集 —— 这样预标注只算一次，重建数据集不会重跑模型。

只处理数据集里还没有的帧（可按 --all 强制重跑）。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/prelabel_yolo.py
  ... --all --conf 0.3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRE = os.path.join(HERE, "yolo", "prelabel")


def labeled_frames() -> set[str]:
    """index.csv 里出现过的帧 = 已有逐格标注，不需要预标注。"""
    out = set()
    path = os.path.join(HERE, "dataset", "index.csv")
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.add(r["frame"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "brick", "weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    os.makedirs(PRE, exist_ok=True)

    done = labeled_frames()
    names = sorted(f for f in os.listdir(args.samples)
                   if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"})
    todo = [n for n in names if args.all or os.path.splitext(n)[0] not in done]
    if not todo:
        print(f"没有需要预标注的帧（共 {len(names)} 帧，全部已有标注）")
        return

    model = YOLO(args.weights)
    total_tile = 0
    for name in todo:
        stem = os.path.splitext(name)[0]
        res = model.predict(os.path.join(args.samples, name), imgsz=640,
                            conf=args.conf, verbose=False)[0]
        lines, n_tile, n_board = [], 0, 0
        for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            h, w = res.orig_shape
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            c = int(cls)
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(cf):.3f}")
            n_tile += c == 1
            n_board += c == 0
        with open(os.path.join(PRE, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        with open(os.path.join(PRE, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"frame": stem, "board": n_board, "tile": n_tile,
                       "weights": os.path.basename(args.weights), "conf": args.conf},
                      fh, ensure_ascii=False, indent=1)
        total_tile += n_tile
        print(f"  {stem}: board {n_board} / tile {n_tile}")

    print(f"预标注 {len(todo)} 帧 → {PRE}（tile 框共 {total_tile} 个）")
    print("下一步：人工抽查 yolo/prelabel 里的框（或先直接训练看效果），再跑 make_yolo_data.py 装配")


if __name__ == "__main__":
    main()
