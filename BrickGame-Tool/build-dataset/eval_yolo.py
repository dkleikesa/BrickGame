#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 36 张图上逐帧评估 board/tile 检测（train 与 val 分开报，避免自证）。"""
from __future__ import annotations

import json
import os

import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(HERE, "yolo")
WEIGHTS = os.path.join(os.path.dirname(HERE), "runs", "brick", "weights", "best.pt")


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def xywh2xyxy(b):
    cx, cy, w, h = b
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def main() -> None:
    model = YOLO(WEIGHTS)
    meta = json.load(open(os.path.join(YOLO_DIR, "meta.json"), encoding="utf-8"))
    stats, rows = {}, []

    for m in meta:
        split, base = m["split"], m["file"]
        img = next(os.path.join(YOLO_DIR, "images", split, base + e)
                   for e in (".jpg", ".png", ".jpeg")
                   if os.path.exists(os.path.join(YOLO_DIR, "images", split, base + e)))
        gt = []
        for line in open(os.path.join(YOLO_DIR, "labels", split, base + ".txt")):
            c, cx, cy, w, h = line.split()
            gt.append((int(c), xywh2xyxy([float(cx), float(cy), float(w), float(h)])))

        res = model.predict(img, imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]  # 与端侧 NMS 一致
        h, w = res.orig_shape
        preds = []
        for box, cls, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            preds.append((int(cls), [x1 / w, y1 / h, x2 / w, y2 / h], float(cf)))

        gt_board = [b for c, b in gt if c == 0]
        gt_tile = [b for c, b in gt if c == 1]
        pr_board = [b for c, b, _ in preds if c == 0]
        pr_tile = [(b, cf) for c, b, cf in preds if c == 1]

        board_iou = max((iou(gt_board[0], b) for b in pr_board), default=0.0) if gt_board and pr_board else 0.0
        used, tp = set(), 0
        for g in gt_tile:
            best, bi = 0.0, -1
            for i, (p, _cf) in enumerate(pr_tile):
                if i in used:
                    continue
                v = iou(g, p)
                if v > best:
                    best, bi = v, i
            if best > 0.5:
                used.add(bi)
                tp += 1
        fp_confs = [cf for i, (_b, cf) in enumerate(pr_tile) if i not in used]

        s = stats.setdefault(split, {"n": 0, "board": 0, "board_iou": [], "tp": 0, "fn": 0,
                                     "fp": 0, "fpc": []})
        s["n"] += 1
        s["board"] += int(board_iou > 0.7)
        s["board_iou"].append(board_iou)
        s["tp"] += tp
        s["fn"] += len(gt_tile) - tp
        s["fp"] += len(fp_confs)
        s["fpc"].extend(fp_confs)
        rows.append((split, m["src"], len(gt_tile), len(pr_tile), tp, len(fp_confs), board_iou))

    print()
    for k in ("train", "val"):
        s = stats.get(k)
        if not s:
            continue
        rec = s["tp"] / max(1, s["tp"] + s["fn"])
        pre = s["tp"] / max(1, s["tp"] + s["fp"])
        fc = np.array(s["fpc"] or [0.0])
        print(f"{k}  {s['n']} 帧  board {s['board']}/{s['n']}（IoU 中位 {np.median(s['board_iou']):.3f}）"
              f"  tile 召回 {rec:.1%}  精确率 {pre:.1%}")
        print(f"     误检 {len(fc)} 个，置信度中位 {np.median(fc):.2f}，"
              f"<0.35 占 {(fc < 0.35).mean():.0%}，<0.5 占 {(fc < 0.5).mean():.0%}")

    print("\n逐帧最差：")
    rows.sort(key=lambda r: (r[4] - r[2]))
    for split, src, ng, npr, tp, fp, bi in rows[:8]:
        print(f"  [{split}] {src[:26]:<26} 真值 {ng:>3} 检出 {npr:>3} TP {tp:>3} 误 {fp:>3} boardIoU {bi:.2f}")


if __name__ == "__main__":
    main()
