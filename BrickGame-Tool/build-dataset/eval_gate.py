#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练后的放行闸门：在固定 val 集上算 board/tile 指标，与上一版对比，不达标就拒绝。

为什么需要它：补数据重训很容易"这一批好了、上一批坏了"，尤其是新皮肤进来把旧皮肤带偏。
没有闸门就只能靠人肉翻截图，最后谁也不敢换模型。

口径（与 eval_yolo.py 一致）：
  · board 命中 = 有 board 框且与真值 IoU > 0.7；
  · tile  命中 = 同类 IoU > 0.5（贪心一对一）；
  · 召回 = TP / 真值框数，精确率 = TP / 检出框数。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/eval_gate.py
  ... --weights runs/brick/weights/best.pt --update   # 通过后把这版写成新基线
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
YOLO_DIR = os.path.join(HERE, "yolo")
METRICS = os.path.join(ROOT, "runs", "brick", "metrics.json")

MIN_BOARD = 0.95
MIN_RECALL = 0.95


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def xywh2xyxy(b):
    cx, cy, w, h = b
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def labels_fp() -> str:
    """标注内容指纹：labels 改过之后，旧基线的精确率就不可比了。"""
    import hashlib
    h = hashlib.md5()
    # 只看 val：train 的框怎么改都不影响这套指标，拿它算指纹会让基线白白作废
    for p in sorted(glob.glob(os.path.join(YOLO_DIR, "labels", "val", "*.txt"))):
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def evaluate(weights: str, conf: float) -> dict:
    model = YOLO(weights)
    meta = json.load(open(os.path.join(YOLO_DIR, "meta.json"), encoding="utf-8"))
    val = [m for m in meta if m["split"] == "val"]
    tp = fn = fp = board_ok = 0
    for m in val:
        split, base = m["split"], m["file"]
        img = next(os.path.join(YOLO_DIR, "images", split, base + e)
                   for e in (".jpg", ".png", ".jpeg")
                   if os.path.exists(os.path.join(YOLO_DIR, "images", split, base + e)))
        gt = []
        for line in open(os.path.join(YOLO_DIR, "labels", split, base + ".txt")):
            c, cx, cy, w, h = line.split()[:5]
            gt.append((int(c), xywh2xyxy([float(cx), float(cy), float(w), float(h)])))
        res = model.predict(img, imgsz=640, conf=conf, iou=0.45, verbose=False)[0]  # 与端侧 NMS 一致
        h_, w_ = res.orig_shape
        preds = []
        for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box
            preds.append((int(cls), [x1 / w_, y1 / h_, x2 / w_, y2 / h_]))

        gb = [b for c, b in gt if c == 0]
        gt_t = [b for c, b in gt if c == 1]
        pb = [b for c, b in preds if c == 0]
        pt = [b for c, b in preds if c == 1]
        board_ok += int(bool(gb and pb and max(iou(gb[0], b) for b in pb) > 0.7))

        used, hit = set(), 0
        for g in gt_t:
            best, bi = 0.0, -1
            for i, p in enumerate(pt):
                if i in used:
                    continue
                v = iou(g, p)
                if v > best:
                    best, bi = v, i
            if best > 0.5:
                used.add(bi)
                hit += 1
        tp += hit
        fn += len(gt_t) - hit
        fp += len(pt) - hit

    return {
        "frames": len(val),
        "board": board_ok / max(1, len(val)),
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "tp": tp, "fn": fn, "fp": fp,
        "weights": os.path.relpath(weights, ROOT),
        "labels_fp": labels_fp(),
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def fmt(m: dict) -> str:
    return (f"board {m['board']:.1%}  召回 {m['recall']:.1%}  精确率 {m['precision']:.1%}  "
            f"（TP {m['tp']} 漏 {m['fn']} 误 {m['fp']}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "brick", "weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--update", action="store_true", help="通过后把这版写成新基线")
    ap.add_argument("--force", action="store_true", help="不达标也更新基线")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="val 里有没人工确认的预标注时也照常评估/写基线（指标会虚高）")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(YOLO_DIR, "meta.json"), encoding="utf-8"))
    unreviewed = [m["src"] for m in meta if m["split"] == "val" and not m.get("reviewed")]
    if unreviewed:
        # 预标注是当前模型自己出的框：拿它当 val 真值，等于让模型给自己判卷。
        print(f"⚠ val 里有 {len(unreviewed)} 帧仍是模型预标注（未人工确认），"
              f"指标会虚高：{', '.join(unreviewed[:3])}{' …' if len(unreviewed) > 3 else ''}")
        if (args.update or args.force) and not args.allow_unreviewed:
            sys.exit("先跑 box_server.py 把 val 帧过一遍，再 --update；"
                     "确实要写基线就加 --allow-unreviewed")

    cur = evaluate(args.weights, args.conf)
    print(f"候选  {fmt(cur)}")
    base = None
    if os.path.exists(METRICS):
        base = json.load(open(METRICS, encoding="utf-8"))
        print(f"基线  {fmt(base)}    （{base.get('at')} {base.get('weights')}）")

    reasons = []
    if cur["board"] < MIN_BOARD:
        reasons.append(f"board 命中 {cur['board']:.1%} < {MIN_BOARD:.0%}")
    if cur["recall"] < MIN_RECALL:
        reasons.append(f"tile 召回 {cur['recall']:.1%} < {MIN_RECALL:.0%}")
    comparable = True
    if base and base.get("labels_fp") and base["labels_fp"] != cur["labels_fp"]:
        # 标注改过，旧指标是拿另一份真值算的，比下去只会误判
        print("⚠ labels 已变更（基线指纹 "
              f"{base['labels_fp']} → {cur['labels_fp']}），精确率不与旧基线对比")
        comparable = False
    elif base and cur["precision"] < base["precision"] - 0.03:
        reasons.append(f"精确率比基线低 {base['precision'] - cur['precision']:.1%}（>3 个点）")

    if reasons and not args.force:
        print("✗ 不放行：" + "；".join(reasons))
        sys.exit(1)
    print("✓ 通过" + ("（--force 忽略以下问题：" + "；".join(reasons) + "）" if reasons else "")
          + ("" if comparable else "（精确率基线已作废，需重建）"))
    if args.update or args.force:
        json.dump(cur, open(METRICS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"基线已更新 → {METRICS}")


if __name__ == "__main__":
    main()
