#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从现有整帧 + 逐格标注生成 YOLO 弱标注数据集（两类：0=board，1=tile）。

为什么先做弱标注而不是纯人工标：满盘 14x10 一张就是 140 个框，36 张接近 5000 框。
现有逐格 class_id 已经告诉"这一格有没有砖、是什么"，网格又是已知的，砖块框可以
直接反推出来；人工只需要在训练前抽查修正偏框。

规则：
  · tile 框 = 格子内缩 6%（砖块底托 ≈ 0.94 格，内缩后不碰格缝）；
  · board 框 = 整张棋盘外框（残局帧同样标 —— 这正是要模型学会的东西）；
  · 网格自洽校验：格子边长与 pitch 偏差 >12% 的帧整帧跳过（标注错的源头是网格错）。

用法：
  uv run --no-project --with numpy --with opencv-python-headless python make_yolo_data.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tile_recognizer import INSET, detect_grid  # noqa: E402

OUT = os.path.join(HERE, "yolo")
INSET_TILE = 0.06


def load_cells() -> dict[str, dict[tuple[int, int], tuple[int, str]]]:
    out: dict[str, dict[tuple[int, int], tuple[int, str]]] = {}
    with open(os.path.join(HERE, "dataset", "index.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = os.path.basename(r["path"])
            if not name.startswith("r") or len(name) < 8:
                continue
            rc = (int(name[1:3]), int(name[4:6]))
            out.setdefault(r["frame"], {})[rc] = (int(r["class_id"]), r.get("cls_src", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--rebalance", action="store_true", help="重新划分 train/val（会让指标失去可比性）")
    args = ap.parse_args()

    cells = load_cells()
    frames = []
    skipped = []
    for name in sorted(os.listdir(args.samples)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imdecode(np.fromfile(os.path.join(args.samples, name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            g = detect_grid(img, 12, INSET, verbose=False)
        except (SystemExit, Exception) as e:
            skipped.append((stem, f"定位失败:{str(e)[:20]}"))
            continue
        xl, yl = np.asarray(g["xlines"]), np.asarray(g["ylines"])
        rows, cols = len(yl) - 1, len(xl) - 1
        dw = np.abs(np.diff(xl)); dh = np.abs(np.diff(yl))
        if dw.max() > dw.mean() * 1.12 or dh.max() > dh.mean() * 1.12:
            skipped.append((stem, f"格距不齐 {dw.max():.0f}/{dw.mean():.0f}"))
            continue
        if stem not in cells:
            # 没有逐格标注的帧：用 prelabel_yolo.py 跑出来的框（模型预标注，人工只做删改）
            pre = os.path.join(OUT, "prelabel", stem + ".txt")
            if not os.path.exists(pre):
                skipped.append((stem, "无标注且未预标注"))
                continue
            labels = []
            for line in open(pre):
                p = line.split()
                if len(p) >= 5:
                    labels.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            if not labels:
                skipped.append((stem, "预标注为空"))
                continue
            frames.append({"stem": stem, "file": name, "rows": 0, "cols": 0,
                           "tiles": sum(1 for l in labels if l[0] == 1),
                           "labels": labels, "src": "prelabel"})
            continue

        h, w = img.shape[:2]
        lines = []
        # board：整块棋盘
        bx0, by0, bx1, by1 = xl[0], yl[0], xl[-1], yl[-1]
        # YOLO 要的是中心点 + 宽高；写成左上角会让 board 框整体偏移半个棋盘
        lines.append((0, (bx0 + bx1) / 2 / w, (by0 + by1) / 2 / h, (bx1 - bx0) / w, (by1 - by0) / h))
        n_tile = 0
        for (r, c), (cid, src) in cells[stem].items():
            if cid <= 0 or src == "empty" or r >= rows or c >= cols:
                continue
            x0, y0, x1, y1 = xl[c], yl[r], xl[c + 1], yl[r + 1]
            mx, my = (x1 - x0) * INSET_TILE, (y1 - y0) * INSET_TILE
            tx0, ty0, tx1, ty1 = x0 + mx, y0 + my, x1 - mx, y1 - my
            lines.append((1, (tx0 + tx1) / 2 / w, (ty0 + ty1) / 2 / h,
                          (tx1 - tx0) / w, (ty1 - ty0) / h))
            n_tile += 1
        frames.append({"stem": stem, "file": name, "rows": rows, "cols": cols,
                       "tiles": n_tile, "labels": lines, "src": "weak"})

    # 划分必须**持久化**：每次重新抽 val 会让指标不可比，也等于偷偷把评测集喂进训练。
    # 规则：split.json 里已有的帧沿用；新帧一律进 train；只有首次运行才按布局分层抽 val。
    frames.sort(key=lambda f: (f["rows"] * f["cols"], f["tiles"]))
    split_path = os.path.join(OUT, "split.json")
    split: dict[str, str] = {}
    if os.path.exists(split_path):
        split = json.load(open(split_path, encoding="utf-8"))
    if args.rebalance or not split:
        split = {}
        seen: dict[str, list[int]] = {}
        for i, f in enumerate(frames):
            seen.setdefault(f"{f['rows']}x{f['cols']}", []).append(i)
        val_idx = set()
        for key, ids in seen.items():
            if len(ids) >= 3:
                val_idx.add(ids[len(ids) // 2])
        while len(val_idx) < args.val:
            for i in range(len(frames) - 1, -1, -1):
                if i not in val_idx:
                    val_idx.add(i)
                    break
            else:
                break
        for i, f in enumerate(frames):
            split[f["stem"]] = "val" if i in val_idx else "train"
        json.dump(split, open(split_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for f in frames:
        split.setdefault(f["stem"], "train")
    val_idx = {i for i, f in enumerate(frames) if split.get(f["stem"]) == "val"}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = os.path.join(OUT, sub)
        os.makedirs(p, exist_ok=True)
        for old in os.listdir(p):
            os.remove(os.path.join(p, old))

    meta = []
    for i, f in enumerate(frames):
        split = "val" if i in val_idx else "train"
        base = f"f{i:03d}"
        shutil.copyfile(os.path.join(args.samples, f["file"]),
                        os.path.join(OUT, "images", split, base + os.path.splitext(f["file"])[1]))
        with open(os.path.join(OUT, "labels", split, base + ".txt"), "w") as fh:
            for cls, cx, cy, bw, bh in f["labels"]:
                fh.write(f"{cls} {max(0.0, min(1.0, cx)):.6f} {max(0.0, min(1.0, cy)):.6f} "
                         f"{bw:.6f} {bh:.6f}\n")
        meta.append({"file": base, "src": f["stem"], "split": split, "rows": f["rows"],
                     "cols": f["cols"], "tiles": f["tiles"], "labels_from": f.get("src", "weak")})

    with open(os.path.join(OUT, "data.yaml"), "w") as fh:
        fh.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n")
        fh.write("names:\n  0: board\n  1: tile\n")
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    n_train = sum(1 for m in meta if m["split"] == "train")
    n_tile = sum(m["tiles"] for m in meta)
    print(f"帧 {len(meta)}（train {n_train} / val {len(meta) - n_train}），tile 框 {n_tile}，board 框 {len(meta)}")
    n_pre = sum(1 for m in meta if m.get("labels_from") == "prelabel")
    by_layout = {}
    for m in meta:
        if not m["rows"]:
            continue
        by_layout.setdefault(f"{m['rows']}x{m['cols']}", []).append(m["tiles"])
    for k, v in sorted(by_layout.items()):
        print(f"  {k}: {len(v)} 帧，tile 框 {sum(v)}（每帧 {min(v)}~{max(v)}）")
    if n_pre:
        print(f"其中 {n_pre} 帧用的是模型预标注（无逐格标注），训练前建议抽查")
    if skipped:
        print(f"跳过 {len(skipped)} 帧：" + ", ".join(f"{a}({b})" for a, b in skipped))
    print(f"输出 → {OUT}")


if __name__ == "__main__":
    main()
