#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给参考库加"背景"类（class_id = 0）—— 素材取自游戏截图里"已消除、只剩底色"的格子。

为什么要有这一类
  棋盘上被消除的格子仍然是一个格子，它必须能作为一个**类别**被识别出来，
  而不是靠"切图时特判 + 丢掉"（之前那样一旦判错，砖块就凭空消失）。
  class_id = 0 是保留号：图鉴物品是 1..2368，0 专指"背景/空位"（dataset 里沿用的同一约定）。

背景不是单一颜色
  不同关卡皮肤不同（实测现有 202 个空位格是 2 种木色；以后还会有薄荷绿/灰绿/橙…），
  所以按颜色**去重后放多张参考图**，每张一类里的一个成员。

可以反复运行
  每次按当前 dataset 的空位格重新生成：先删掉旧的 bg* 素材与行，再写新的。
  加新关卡后重跑一次即可，不用手工维护。

用法：
  python add_bg_class.py --dataset ../build-dataset/dataset --catalog .
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # 项目根：tile_recognizer
import tile_recognizer as T  # noqa: E402

BG_CLASS_ID = 0
BG_PREFIX = "bg"
COLOR_TOL = 30.0        # 底色 BGR 距离小于它 = 同一种背景
TILE_SIZE = 96          # 与图鉴 tiles 保持一致（背景是平的，放大不损失信息）
MAX_TILES = 12          # 最多保留几种背景


def pick_representatives(colors: np.ndarray, tol: float, max_n: int) -> list[int]:
    """按颜色去重：每种底色挑一张代表（离该簇中心最近的那张）。

    做法：每轮先在"还没归类的"里找出最主流的底色（32 灰阶量化后取众数），
    把与它距离 < tol 的格子整簇收走、记下代表，然后接着找下一种。
    这样 192 张木色 + 10 张深木色会得到 2 张参考图，而不是 1 张或 202 张。
    """
    left = np.ones(len(colors), bool)
    reps: list[int] = []
    while left.any() and len(reps) < max_n:
        left_idx = np.where(left)[0]
        q = (colors[left_idx] // 32).astype(np.int64)
        key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
        vals, cnt = np.unique(key, return_counts=True)
        seed = int(left_idx[key == vals[int(cnt.argmax())]][0])
        cluster = [int(k) for k in left_idx
                   if float(np.linalg.norm(colors[k] - colors[seed])) < tol]
        centre = colors[cluster].mean(0)
        reps.append(min(cluster, key=lambda k: float(np.linalg.norm(colors[k] - centre))))
        left[cluster] = False
    return sorted(reps)


def main() -> None:
    ap = argparse.ArgumentParser(description="把'背景'类（class_id=0）写进参考库")
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(_HERE), "build-dataset", "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_HERE, "catalog"))
    ap.add_argument("--tol", type=float, default=COLOR_TOL, help="背景色去重阈值")
    args = ap.parse_args()

    index_path = os.path.join(args.catalog, "index.csv")
    tile_dir = os.path.join(args.catalog, "tiles")
    with open(index_path) as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        rows = list(rd)

    empties = [r for r in csv.DictReader(open(os.path.join(args.dataset, "index.csv")))
               if r.get("cls_src") == "empty" or str(r.get("class_id")) == "0"]
    if not empties:
        raise SystemExit("dataset 里没有空位格（cls_src=empty），先跑 build_dataset.py")

    crops, colors = [], []
    for r in empties:
        im = cv2.imread(os.path.join(args.dataset, r["path"]))
        if im is None:
            continue
        crops.append(im)
        colors.append(np.median(im.reshape(-1, 3), axis=0).astype(np.float32))
    colors = np.stack(colors)
    reps = pick_representatives(colors, args.tol, MAX_TILES)
    print(f"空位格 {len(crops)} 张 → 去重后 {len(reps)} 种背景（阈值 {args.tol}）")

    # 清掉旧的 bg 素材（幂等重跑）
    kept = []
    for r in rows:
        if str(r.get("class_id")) == str(BG_CLASS_ID) or os.path.basename(r.get("tile", "")).startswith(BG_PREFIX):
            p = os.path.join(args.catalog, r.get("tile") or "")
            if p and os.path.exists(p):
                os.remove(p)
        else:
            kept.append(r)

    new_rows = []
    for k, i in enumerate(reps, start=1):
        entry = f"{BG_PREFIX}{k:02d}"
        rel = os.path.join("tiles", entry + ".png")
        cv2.imwrite(os.path.join(args.catalog, rel),
                    cv2.resize(crops[i], (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA))
        row = {c: "" for c in fields}
        row.update({"entry": entry, "page": 0,
                    # 颜色写进 page_file（该列只作备注、没程序读），避免新增列改动表结构
                    "page_file": "game_screenshots(已消除格) 底色BGR"
                                 + str([int(v) for v in colors[i].round(0)]),
                    "block": 0, "sub_row": 0, "row": -1, "col": -1,
                    "ratio": 1.0, "contrast": 0.0, "sat": 0.0, "faded": 0,
                    "icon": "", "tile": rel.replace(os.sep, "/"), "class_id": BG_CLASS_ID})
        new_rows.append(row)
        print(f"  {entry}: 代表 {os.path.basename(empties[i]['path'])}  底色 BGR {colors[i].round(0)}")

    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)      # 列结构保持不变
        w.writeheader()
        w.writerows(kept + new_rows)

    n_cls = len({r["class_id"] for r in kept + new_rows if r.get("class_id") != ""})
    print(f"\n参考库已更新：{index_path}")
    print(f"  物品类 {n_cls - 1} 个（1..2368）+ 背景类 1 个（class_id=0，{len(new_rows)} 张参考图）")
    print(f"  行数 {len(kept)} → {len(kept) + len(new_rows)}")


if __name__ == "__main__":
    main()
