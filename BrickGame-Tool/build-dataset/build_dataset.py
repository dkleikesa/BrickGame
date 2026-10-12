#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一条命令跑完"裁图 → 模型识别 → 人工标注界面"：

  ① 裁图：游戏截图 → 棋盘标定 → 每格裁切（**保留底色**）；顺带把图鉴 tiles 收成带真实标签的参考图
  ② 识别：预训练 CNN（默认 resnet18）提特征 → 最近邻匹配图鉴 → 每格得到 class_id + top-N 候选
  ③ 标注：生成 annotate.html（按"视觉一样"分组）；人工改完导出 JSON，再用 apply_corrections.py 写回

产出：
  dataset/images/frame/<截图名>/r03c05.png  截图每格 64x64，含底色（状态信息就在底色里）
  dataset/images/ref/<entry>.png           图鉴参考图（catalog/tiles），唯一带真实标签的数据
  dataset/index.csv    每格 5 列：path（图片）、frame（哪张截图）、class_id（标签）、
                       cls_dist（识别距离，仅供"只看低置信"筛选）、cls_src（auto/manual/unknown）
                       行列已编码在文件名里（r03c05 = 第3行第5列）；图鉴参考图不在这个表里，
                       它直接用 build-catalog/catalog 的 tiles
  dataset/pairs/<帧>.csv  帧内配对（仅 --pairs 导出：训练按帧内 label 采样，用不到它）
  cache/emb_<骨干>.npz    预训练特征缓存（识别与分组都基于它）
  cache/auto_labels.csv   每格 top-N 候选 class_id（标注界面点选用）
  annotate.html           人工标注界面（--html 可改名；记得用 --keep-labels 保住改动）
  （cache/ 是中间产物，随时可删，重跑自动重建；棋盘行列由文件名推、原文件名由 samples/ 找，不额外落盘）

空位格默认不落盘（训练/评测都按 hole==0 过滤，落盘只是占空间）；要保留加 --keep-empty。
帧目录名 = 截图原文件名（中文名当命令行参数时记得加引号），所以帧名本身就能对回 samples/ 里的原图。

为什么要"保留底色"：底色 = 砖块状态（冻结/选中）的唯一信息源。
状态要作为独立任务学（多任务里的分类头），身份头才能顺带学会"忽略底色"。
图鉴参考图同样保留底板：训练靠底色重染造状态多样性，不用去底掩码
（实测去底掩码的边缘残留会伤训练，且"只减底板色"会让同类距离涨到 0.12~0.21）。

用法：
  python build_dataset.py --samples samples --catalog catalog --out dataset
  python build_dataset.py ... --skip-embed     # 图片没变，复用特征缓存（省约 10 秒）
  python build_dataset.py ... --keep-labels    # 重跑时保住人工标注
  python build_dataset.py ... --no-classify    # 只裁图，不识别、不出界面
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
sys.path.insert(0, os.path.dirname(_HERE))   # 项目根：tile_recognizer / build_catalog
sys.path.insert(0, _HERE)                    # 本目录：同流水线的兄弟模块
import tile_recognizer as T  # noqa: E402
import auto_classify  # noqa: E402
import embed_features  # noqa: E402
import visualize_groups  # noqa: E402

TILE = 64          # 数据集里每格存成多大
OUT_INSET = 0.02   # 裁切时几乎不内缩：要把整块砖（含边框和底色）都保留


def cut_tile(img: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    a, c = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    a, c = max(0, a), max(0, c)
    b, d = min(img.shape[1], a + side), min(img.shape[0], c + side)
    patch = img[c:d, a:b]
    if patch.size == 0:
        patch = img[:TILE, :TILE]
    return cv2.resize(patch, (TILE, TILE), interpolation=cv2.INTER_AREA if patch.shape[0] > TILE
                      else cv2.INTER_CUBIC)


def frame_cells(img: np.ndarray, min_pitch: int) -> dict | None:
    """一帧 → 每格的裁切图 + 伪标签 + 元信息。"""
    try:
        g = T.detect_grid(img, min_pitch, T.INSET, verbose=False)
    except SystemExit:
        return None
    holes = (g["score"] < g["score_thr"]).ravel()
    idx = np.where(~holes)[0]
    labels = np.zeros(len(holes), dtype=np.int32)
    dist = None
    if idx.size >= 4:
        D = T.aligned_distance_matrix(g["dev"][idx])
        gap = T.gap_threshold(D)
        sub, _, _ = T.cluster_by_parity(D, gap)
        sub = T.merge_similar(sub, D, gap)
        uniq = sorted(set(sub.tolist()))
        remap = {u: i + 1 for i, u in enumerate(uniq)}
        labels[idx] = np.array([remap[int(v)] for v in sub], dtype=np.int32)
        dist = np.zeros((idx.size, idx.size), dtype=np.float32)
        dist[:] = D
    else:
        gap = 0.0

    xs, ys = g["xlines"], g["ylines"]
    pitch = float(np.median(np.concatenate([np.diff(xs), np.diff(ys)])))
    side = max(8, int(round(pitch * (1 - 2 * OUT_INSET))))
    ring = T._border_mask(TILE, max(2, TILE // 16))

    cells = []
    for j in range(g["rows"]):
        cy = (ys[j] + ys[j + 1]) / 2.0
        for i in range(g["cols"]):
            cx = (xs[i] + xs[i + 1]) / 2.0
            tile = cut_tile(img, cx, cy, side)
            bg = np.median(tile[ring], axis=0)          # 底板/边框色 = 状态代理
            cells.append({
                "row": j, "col": i,
                "label": int(labels[j * g["cols"] + i]),
                "hole": bool(holes[j * g["cols"] + i]),
                "ratio": float(g["score"][j, i]),
                "bg_bgr": bg.round(1).tolist(),
                "tile": tile,
            })
    return {"cells": cells, "rows": g["rows"], "cols": g["cols"],
            "dist": dist, "idx": idx, "gap": gap, "pitch": pitch}


def main() -> None:
    ap = argparse.ArgumentParser(description="游戏截图 → 训练数据集（含底色）+ 底板库")
    ap.add_argument("--samples", default=os.path.join(_HERE, "samples"), help="截图目录")
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"),
                    help="图鉴素材库（取 tiles 当真实身份标签样本）；留空则不纳入")
    ap.add_argument("--out", default=os.path.join(_HERE, "dataset"), help="输出目录")
    ap.add_argument("--min-pitch", type=int, default=12)
    ap.add_argument("--keep-empty", action="store_true", help="也落盘空位格（默认跳过）")
    ap.add_argument("--keep-labels", action="store_true",
                    help="重跑时保留旧 index.csv 里的 class_id 标注（人工改过的不会被清掉）")
    # ---- 后两步：模型识别 + 标注界面（都可单独关掉）----
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "mobilenet_v3_small"],
                    help="预训练骨干（识别与分组的特征来源）")
    ap.add_argument("--cache", default="", help="中间产物目录（默认 数据集同级 cache/）")
    ap.add_argument("--skip-embed", action="store_true",
                    help="复用已有的 cache/emb_*.npz（图片没变时省 10 秒）")
    ap.add_argument("--no-classify", action="store_true", help="只裁图，不做模型识别")
    ap.add_argument("--no-html", action="store_true", help="不生成标注界面")
    ap.add_argument("--html", default=os.path.join(_HERE, "annotate.html"), help="标注界面输出路径")
    ap.add_argument("--auto-top", type=int, default=12, help="每格保存几个候选 class_id")
    ap.add_argument("--same-thr", type=float, default=0.10, help="分组：同款 embedding 距离阈值")
    ap.add_argument("--color-thr", type=float, default=0.35, help="分组：同款色相距离阈值")
    ap.add_argument("--pairs", action="store_true", help="额外导出帧内配对 pairs/*.csv（默认不导出）")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    frame_dir = os.path.join(img_dir, "frame")
    os.makedirs(frame_dir, exist_ok=True)
    index_rows, stats, frame_rows = [], [], []

    # ---------- 1) 游戏截图：自动标定 + 裁格 + 伪标签 ----------
    # 帧名 = 截图原文件名（去掉扩展名）：目录/索引/命令行都能直接对上原图
    names = sorted(f for f in os.listdir(args.samples) if f.lower().endswith((".jpg", ".png")))
    for name in names:
        img = T.load_image(os.path.join(args.samples, name))
        r = frame_cells(img, args.min_pitch)
        if r is None:
            print(f"  跳过 {name}（标定失败）")
            continue
        frame = os.path.splitext(name)[0]
        os.makedirs(os.path.join(frame_dir, frame), exist_ok=True)
        cells, idx, D, gap = r["cells"], r["idx"], r["dist"], r["gap"]
        for c in cells:
            if c["hole"] and not args.keep_empty:
                continue                      # 空位：训练与评测都按 hole==0 过滤，不落盘（要留加 --keep-empty）
            rel = os.path.join("images", "frame", frame, f"r{c['row']:02d}c{c['col']:02d}.png")
            cv2.imwrite(os.path.join(args.out, rel), c["tile"])
            index_rows.append({
                # 只要"图片 + class_id"这条线要的东西：path 指图、frame 用于分组/留出/回原图、
                # class_id 是标签（auto_classify 填 auto，人工在 annotate.html 改的记 manual）、
                # cls_dist 只用于"只看低置信"筛选、cls_src 用来保护人工标注不被覆盖
                "path": rel, "frame": frame,
                "class_id": "", "cls_dist": "", "cls_src": "",
            })
        # 同一帧内的配对：同类=正、异类且距离大=负（灰区不采信）。
        # 默认不导出：训练按帧内 label 采样，这份清单没人读，只用于人工核对。
        n_pairs = n_pos = 0
        if args.pairs and D is not None and idx.size:
            pairs = []
            for a in range(idx.size):
                for b in range(a + 1, idx.size):
                    ca, cb = cells[idx[a]], cells[idx[b]]
                    same = int(ca["label"] == cb["label"])
                    d = float(D[a, b])
                    conf = 1 if (same or d > max(gap, 0.15)) else 0
                    pairs.append((frame, ca["row"], ca["col"], frame, cb["row"], cb["col"],
                                  same, round(d, 4), conf))
            n_pairs, n_pos = len(pairs), sum(1 for p in pairs if p[6])
            os.makedirs(os.path.join(args.out, "pairs"), exist_ok=True)
            with open(os.path.join(args.out, "pairs", f"{frame}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_a", "ra", "ca", "frame_b", "rb", "cb",
                            "same", "distance", "confident"])
                w.writerows(pairs)
        n_cells = r["rows"] * r["cols"]
        frame_rows.append({
            "frame": frame, "source_file": name, "rows": r["rows"], "cols": r["cols"],
            "cells": n_cells, "bricks": int(idx.size),
            "classes": len(set(cells[i]["label"] for i in idx)),
            "holes": n_cells - int(idx.size), "pairs": n_pairs, "pairs_pos": n_pos,
        })
        stats.append(f"{frame} {r['rows']:2d}x{r['cols']:<2d} 砖块{idx.size:3d} "
                     f"类数{frame_rows[-1]['classes']:3d} 空位{frame_rows[-1]['holes']:3d}")
        print("  " + stats[-1])

    if not index_rows:
        raise SystemExit("没产出任何数据：检查 --samples 里是否有截图、--catalog 是否存在")

    # 重跑不丢标注：按 path 把旧 index.csv 的 class_id/cls_dist/cls_src 回填
    old_index = os.path.join(args.out, "index.csv")
    if args.keep_labels and os.path.exists(old_index):
        with open(old_index) as f:
            old = {r["path"]: r for r in csv.DictReader(f)}
        kept = 0
        for r in index_rows:
            o = old.get(r["path"])
            if o and o.get("cls_src"):
                r["class_id"], r["cls_dist"], r["cls_src"] = (
                    o.get("class_id", ""), o.get("cls_dist", ""), o.get("cls_src", ""))
                kept += 1
        print(f"  保留旧标注 {kept} 行（--keep-labels）")
        refreshed = {r["path"] for r in index_rows}
        for p, o in old.items():
            if p not in refreshed and o.get("cls_src") == "manual":
                print(f"  警告: {p} 已不在新数据里，人工标注未保留")

    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)
    print(f"\n① 裁图: {args.out}/images 下 {len(index_rows)} 张截图格 ({TILE}x{TILE})，"
          f"{len(frame_rows)} 帧")
    print(f"  索引 {args.out}/index.csv{'；配对 ' + args.out + '/pairs/*.csv' if args.pairs else ''}")

    if args.no_classify:
        print("（--no-classify：跳过模型识别与标注界面）")
        return

    # ---------- 2) 模型识别：预训练特征最近邻 → class_id ----------
    cache = args.cache or auto_classify.cache_dir_of(args.out)
    emb_path = auto_classify.pick_emb(cache)
    if args.skip_embed and emb_path:
        print(f"② 识别: 复用已有特征 {emb_path}")
    else:
        print(f"② 识别: 提取特征（{args.backbone}）")
        emb_path = embed_features.extract(args.out, args.catalog, args.backbone)
    auto_classify.classify(args.out, args.catalog, top=args.auto_top, emb=emb_path, cache=cache)

    # ---------- 3) 标注界面 ----------
    if args.no_html:
        print("（--no-html：未生成标注界面）")
        return
    visualize_groups.build_html(args.out, args.catalog, args.samples, args.html,
                                top=args.auto_top, emb=emb_path, cache=cache,
                                same_thr=args.same_thr, color_thr=args.color_thr)
    print(f"\n③ 标注: 打开 {args.html}（人工改完导出 JSON，再跑 apply_corrections.py 写回）")


if __name__ == "__main__":
    main()
