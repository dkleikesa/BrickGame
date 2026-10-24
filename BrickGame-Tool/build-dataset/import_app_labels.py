#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 App 落盘的 `lowconf_*.txt` 当作**预填标签**写回 dataset/index.csv。

为什么需要它：留档里的矩阵就是"当时线上模型逐格判出来的 class_id"，比 build_dataset 的
自动识别（未微调的 ImageNet 特征最近邻，见 auto_classify 文件头）准得多 —— 直接拿它当标注
起点，人工只需要盯"模型自己没把握"的那些格子。

分两档写，绝不把猜测当真值：
  · 留档**没列进"没把握"清单**的格子 → 线上相似度 ≥ `--thr`（默认 85，与
    `TfliteTileClassifier.DUMP_SIMILARITY` 一致），标签直接采信 → class_id + cls_src="app"，
    cls_dist 记 0.15（低于标注界面默认阈值 0.30，不会被当成低置信）。
  · 留档**列进清单**的格子 → 只保留它的 top-1 当"起点提示"，cls_src="unknown"
    （训练时一律忽略，见 train_classifier.Real），cls_dist = 1 - 相似度/100 → 在
    annotate.html 里天然落在低置信筛选里，等人改。
  · 已是 manual 的行不碰（人工标注永远赢）；检测器已判空位（cls_src="empty"）的格子，
    留档说 0 就保持空位；留档在空位上给了高置信的类，则以留档为准（判空判错的修正）。

按 (行,列) 对齐：留档矩阵与 dataset 的 r{行}c{列} 同一套坐标 —— 前提是 App 与
YOLO 定位（`build-dataset/yolo_grid.py`）标定到同一个棋盘（两边逐格一致，实测 15×10 帧
一致率 98%、14×10 帧 100%）。对不上就是标定没对齐，脚本会按"行列数不符"整帧跳过。

用法：
  .venvs/brick-train/bin/python import_app_labels.py --dump ~/debug/brickgame-lowconf          # 预览
  .venvs/brick-train/bin/python import_app_labels.py --dump ~/debug/brickgame-lowconf --inplace
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

THR = 85            # 与 TfliteTileClassifier.DUMP_SIMILARITY 一致：低于它才进"没把握"清单
# 采信格子的 cls_dist：低于标注界面默认阈值 0.30，不落进低置信筛选。
# 注意：这是**常量**，app 组的 cls_dist 没有分布，不能拿它做低置信排序/筛选。
DIST_OK = 0.15
DIST_LOW = 0.31     # 待人工格子的 cls_dist 下限：留档相似度 78 换算出来只有 0.22，
#                     会从界面的低置信筛选里漏掉 —— 这些格子必须被看见，所以压一个下限

_HEAD = re.compile(r"(\d+)x(\d+)\s+pitch")
_LOW = re.compile(r"\((\d+),(\d+)\)=(\d+)")


def parse_dump(path: str) -> dict | None:
    """`lowconf_*.txt` → {rows, cols, labels(二维), low{(r,c): 相似度}}。

    矩阵与"没把握"清单是同一个行优先坐标系；清单只在低置信时才出现，所以缺的格子
    一律按"线上当时有把握"处理（阈值见 THR）。
    """
    head, matrix = "", []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            head += line
            continue
        toks = line.split()
        if toks and all(t.lstrip("-").isdigit() for t in toks):
            matrix.append([int(t) for t in toks])
    m = _HEAD.search(head)
    if not m or not matrix:
        return None
    low = {(int(r), int(c)): int(s) for r, c, s in _LOW.findall(head)}
    return {"rows": int(m.group(1)), "cols": int(m.group(2)),
            "labels": matrix, "low": low}


def main() -> None:
    ap = argparse.ArgumentParser(description="App 留档标签 → dataset/index.csv（预填）")
    ap.add_argument("--dump", required=True, help="lowconf_*.txt 所在目录（截图与 txt 同名）")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--thr", type=int, default=THR,
                    help=f"线上相似度低于它算「没把握」（默认 {THR}，与 App 的落盘阈值一致）")
    ap.add_argument("--inplace", action="store_true", help="写回 index.csv（默认只预览）")
    args = ap.parse_args()

    index_path = os.path.join(args.dataset, "index.csv")
    with open(index_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    by_path = {r["path"]: r for r in rows}

    n_app = n_unknown = n_manual = n_kept_empty = n_bg_taken = n_skip = 0
    per_frame = []
    for name in sorted(os.listdir(args.dump)):
        if not name.endswith(".txt"):
            continue
        stem = name[:-4]
        d = parse_dump(os.path.join(args.dump, name))
        if d is None:
            print(f"  跳过 {name}（读不出网格或矩阵）")
            continue
        cells = [(r, c) for r in range(d["rows"]) for c in range(d["cols"])]
        paths = [os.path.join("images", "frame", stem, f"r{r:02d}c{c:02d}.png") for r, c in cells]
        if not all(p in by_path for p in paths):
            n_skip += 1
            print(f"  跳过 {stem}（{d['rows']}x{d['cols']} 的格子对不上 dataset，先跑 build_dataset.py）")
            continue
        n_frame_app = n_frame_unk = n_frame_low = 0
        for (r, c), p in zip(cells, paths):
            row = by_path[p]
            if row.get("cls_src") == "manual":          # 人工标注永远赢
                n_manual += 1
                continue
            sim = d["low"].get((r, c))
            if sim is None:
                sim = THR                               # 没列进清单 = 线上有把握
            else:
                n_frame_low += 1
            cid = int(d["labels"][r][c])
            if row.get("cls_src") == "empty" and cid == 0:
                n_kept_empty += 1                       # 检测器与留档都判空位：不动
                continue
            if sim < args.thr:
                # 没把握的格子：只给起点，训练时必须人工过一遍（cls_src=unknown 会被丢掉）
                dist = max(round(1 - sim / 100, 3), DIST_LOW)
                row["class_id"], row["cls_dist"], row["cls_src"] = cid, dist, "unknown"
                n_unknown += 1
                n_frame_unk += 1
            else:
                if cid == 0 and row.get("cls_src") != "empty":
                    n_bg_taken += 1                     # 留档高置信判成背景：按背景类收
                row["class_id"], row["cls_dist"], row["cls_src"] = cid, DIST_OK, "app"
                n_app += 1
                n_frame_app += 1
        per_frame.append((stem, n_frame_app, n_frame_unk, n_frame_low))

    print(f"留档目录: {args.dump}")
    for stem, a, u, low in per_frame:
        print(f"  {stem}: 采信 {a} 格 / 待人工 {u} 格（留档低置信 {low} 格）")
    print(f"\n采信(高置信) {n_app} 格；待人工(unknown，训练时忽略) {n_unknown} 格；"
          f"跳过人工标注 {n_manual} 格；维持空位 {n_kept_empty} 格；"
          f"留档判背景 {n_bg_taken} 格；整帧跳过 {n_skip} 帧")
    if not args.inplace:
        print("\n预览模式：没有写文件。加 --inplace 落地。")
        return
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写回 {index_path}（接着跑 visualize_groups.py 重出标注界面）")


if __name__ == "__main__":
    main()
