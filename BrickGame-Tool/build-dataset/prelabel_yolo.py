#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给还没框标注的帧生成 YOLO 预标注（board + tile）。

工业化补数据的核心一步：新截图进来时没人愿意逐格点 140 个框，所以先用当前模型跑一遍，
人工只做"删掉明显错的框 / 补漏掉的框"。产物落在 yolo/prelabel/<帧名>.txt，
由 make_yolo_data.py 装配进数据集 —— 这样预标注只算一次，重建数据集不会重跑模型。

默认只处理"还没有预标注文件"的帧；`--all` 强制重跑（换模型后想刷一遍时用）。
**注意**：预标注只是草稿，必须人工在 box_server.py 里过一遍，否则漏检会被当成
"这里没有砖"写进训练集。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/prelabel_yolo.py
  ... --all --conf 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_grid import resolve_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "yolo", "prelabel")


def prelabeled() -> set[str]:
    """已经有预标注文件的帧。"""
    if not os.path.isdir(PRE):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(PRE) if f.endswith(".txt")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--weights", default="", help="留空 = runs/brick/weights/best.pt，缺失时用 APK 资产")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    os.makedirs(PRE, exist_ok=True)

    done = prelabeled()
    names = sorted(f for f in os.listdir(args.samples)
                   if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"})
    todo = [n for n in names if args.all or os.path.splitext(n)[0] not in done]
    if not todo:
        print(f"没有需要预标注的帧（共 {len(names)} 帧，全部已有预标注）")
        return

    weights = resolve_weights(args.weights)
    model = YOLO(weights)
    total_tile = 0
    for name in todo:
        stem = os.path.splitext(name)[0]
        res = model.predict(os.path.join(args.samples, name), imgsz=640,
                            conf=args.conf, verbose=False)[0]
        h, w = res.orig_shape
        boxes = []
        for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            boxes.append((int(cls), float(cf),
                          (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                          (x2 - x1) / w, (y2 - y1) / h))
        # 一帧只有一个棋盘：多出来的 board 框是重复检测，按置信度只留一个，
        # 否则训练目标自相矛盾（真值里同一张图有两个 "整块棋盘"）。
        boards = [b for b in boxes if b[0] == 0]
        if len(boards) > 1:
            best = max(boards, key=lambda b: b[1])
            boxes = [b for b in boxes if b[0] != 0 or b is best]
        n_tile = sum(1 for b in boxes if b[0] == 1)
        n_board = sum(1 for b in boxes if b[0] == 0)
        lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {cf:.3f}"
                 for c, cf, cx, cy, bw, bh in boxes]
        with open(os.path.join(PRE, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        with open(os.path.join(PRE, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"frame": stem, "board": n_board, "tile": n_tile,
                       "weights": os.path.basename(weights), "conf": args.conf},
                      fh, ensure_ascii=False, indent=1)
        total_tile += n_tile
        print(f"  {stem}: board {n_board} / tile {n_tile}")

    print(f"预标注 {len(todo)} 帧 → {PRE}（tile 框共 {total_tile} 个）")
    print("下一步：make_yolo_data.py 装配 → box_server.py 人工删/补 → 重新装配 → train_yolo.py")


if __name__ == "__main__":
    main()
