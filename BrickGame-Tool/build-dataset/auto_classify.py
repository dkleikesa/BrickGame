#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 dataset 的截图格自动识别 class_id（逐张匹配 catalog 图鉴参考图）。

为什么只是"自动识别一部分"：现在用的是**未训练的去底像素特征 + ±3px 平移对齐**，
色系和轮廓相近的不同物品会互相抢（实测：菠萝的最近邻是胡萝卜，柠檬最近邻是黄色圆盘），
所以它给的是**候选 + 距离**，不是真值 —— 距离大的必须人工在 annotate.html 里改，
改过的行记 cls_src=manual，与自动结果区分开。

产出：
  dataset/index.csv  就地更新 class_id / cls_dist / cls_src 三列（已是 manual 的行不覆盖）
  cache/auto_labels.csv  每张截图格的 top-N 候选（class_id + 距离），给 annotate.html 点选
                         cache/ = 数据集同级的中间产物目录，删了会重跑重建

用法：
  .venv/bin/python auto_classify.py --dataset dataset --catalog catalog
  .venv/bin/python auto_classify.py --thr 0.30    # 只统计"多少张低于该距离"，不改判定
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
sys.path.insert(0, os.path.dirname(_HERE))   # 项目根：tile_recognizer / build_catalog
sys.path.insert(0, _HERE)                    # 本目录：同流水线的兄弟模块
import tile_recognizer as T  # noqa: E402

# ---- 项目选定的自动打标签模型（2026-10 定稿）----
# 五个候选（resnet18/resnet50/mobilenet_v3_small/large/dinov2_vits14）在 1212 格上比过：
#   mobilenet_v3_large 自洽错误率最低档（11.3%）、换底板 top1 69%/加描边 96%、
#   关键难例（高亮西瓜、白花菜）都认对、标签最稳（与 resnet18 只差 59 格）。
# 改这一个常量即可整体切换；embed_features/build_dataset 的 --backbone 默认值都读它。
DEFAULT_BACKBONE = "mobilenet_v3_large"


def dev_feature(paths: list[str]) -> np.ndarray:
    """一批格子 → 去底图标特征 (N,32,32,3)，与 tile_recognizer 帧内聚类用的是同一套。"""
    patches = np.zeros((len(paths), T.CELL, T.CELL, 3), np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        patches[i] = cv2.resize(img, (T.CELL, T.CELL), interpolation=cv2.INTER_AREA)
    return T.icon_features(patches)[0]


def top_matches(dev_q: np.ndarray, dev_r: np.ndarray, k: int = 3, prefilter: int = 12
                ) -> list[list[tuple[float, int]]]:
    """查询 → 参考图 top-k：余弦粗筛 + ±3px 平移精算（与 tile_recognizer 同思路）。

    网格线有亚像素残差，同一图标裁出来会差 1~2 像素，不搜平移会让同款距离虚高。
    """
    q = dev_q.reshape(len(dev_q), -1)
    r = dev_r.reshape(len(dev_r), -1)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
    cand = np.argsort(1.0 - qn @ rn.T, axis=1)[:, :prefilter]
    out = []
    for i in range(len(dev_q)):
        pad = cv2.copyMakeBorder(dev_q[i].astype(np.float32), 3, 3, 3, 3, cv2.BORDER_REPLICATE)
        res = [(float(cv2.matchTemplate(pad, dev_r[j].astype(np.float32),
                                       cv2.TM_SQDIFF_NORMED).min()), int(j)) for j in cand[i]]
        res.sort()
        out.append(res[:k])
    return out


def cache_dir_of(dataset: str) -> str:
    """中间产物目录：数据集同级的 cache/（特征、候选）。可随时删，重跑会自动重建。"""
    return os.path.join(os.path.dirname(os.path.abspath(dataset)), "cache")


def pick_emb(cache_dir: str) -> str:
    """找特征缓存（embed_features.py 的产物）。

    优先用项目选定的骨干（DEFAULT_BACKBONE），**不能只按文件名排序取第一个** ——
    cache/ 里可能留着多个骨干做对比实验的缓存（实测同时有 dinov2 / mobilenet / resnet
    五个），按名字排序会静默挑到 dinov2，标签就悄悄换模型了。
    """
    if not os.path.isdir(cache_dir):
        return ""
    files = sorted(f for f in os.listdir(cache_dir)
                   if f.startswith("emb_") and f.endswith(".npz"))
    want = f"emb_{DEFAULT_BACKBONE}.npz"
    if want in files:
        return os.path.join(cache_dir, want)
    return os.path.join(cache_dir, files[0]) if files else ""


def classify(dataset: str, catalog: str, top: int = 12, thr: float = 0.30, force: bool = False,
             emb: str = "", feature: str = "auto", quiet: bool = False, cache: str = "",
             frames: list[str] | None = None) -> None:
    """给截图格识别 class_id（写回 index.csv 的 class_id/cls_dist/cls_src + cache/auto_labels.csv）。"""
    cache = cache or cache_dir_of(dataset)
    os.makedirs(cache, exist_ok=True)
    index_path = os.path.join(dataset, "index.csv")
    with open(index_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("class_id", "cls_dist", "cls_src"):
        if col not in fieldnames:
            fieldnames.append(col)
            for r in rows:
                r.setdefault(col, "")

    # 空位格也一起识别："背景"在参考库里是一个正式类别（class_id=0，见 build-catalog/add_bg_class.py）。
    # 判空是检测器给的判断，让模型独立判一遍有两个好处：
    #   1) 正常情况下空位格会稳定命中 class 0（实测 202/202），等于复核了检测器；
    #   2) 万一检测器把砖块误判成空位，这里模型会给出真实类别 → 变成可见的异常而非静默丢图。
    # cls_src=empty 的标记保留（不覆盖成 auto），方便事后筛出"检测器与模型不一致"的格子。
    # frames=None 表示"全部帧"；frames=[] 表示"一个都不识别"（增量模式下没有新帧时就是这个）
    targets = [i for i, r in enumerate(rows)
               if (force or r.get("cls_src") != "manual")
               and (frames is None or r.get("frame") in frames)]
    if frames is not None:
        print(f"  只识别指定的 {len(frames)} 个帧（其余帧的标签原样不动）")
    if not targets:
        print("  识别: 没有需要识别的截图格（都已是人工标注）；要重算加 --force")
        return

    refs = list(csv.DictReader(open(os.path.join(catalog, "index.csv"))))
    emb_path = emb or pick_emb(cache)
    use_emb = feature == "emb" or (feature == "auto" and emb_path)
    if use_emb and not os.path.exists(emb_path):
        raise SystemExit(f"找不到特征缓存 {emb_path}，先跑：python embed_features.py")
    if use_emb:
        z = np.load(emb_path)
        Fr = z["ref"]
        pos = {p: i for i, p in enumerate(z["cell_path"].tolist())}
        Fc = np.stack([z["cell"][pos[rows[i]["path"]]] for i in targets])
        D = np.clip(1.0 - Fc @ Fr.T, 0.0, 2.0)              # 余弦距离
        order = np.argsort(D, axis=1)[:, :max(1, top)]
        hits = [[(float(D[t, j]), int(j)) for j in order[t]] for t in range(len(targets))]
        how = f"特征 {z['backbone']}"
    else:
        dev_r = dev_feature([os.path.join(catalog, r["tile"]) for r in refs])
        dev_q = dev_feature([os.path.join(dataset, rows[i]["path"]) for i in targets])
        hits = top_matches(dev_q, dev_r, max(1, top))
        how = "像素特征"

    dists = []
    for t, i in enumerate(targets):
        d0, j0 = hits[t][0]
        rows[i]["class_id"] = int(refs[j0]["class_id"])
        rows[i]["cls_dist"] = round(d0, 3)
        # 保留建库时给的 empty 标记（"检测器判成空位"），其余标 auto
        rows[i]["cls_src"] = rows[i].get("cls_src") or "auto"
        dists.append(d0)
    dists = np.array(dists)

    with open(os.path.join(cache, "auto_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"] + sum(([f"cls{k + 1}", f"d{k + 1}"] for k in range(max(1, top))), []))
        for t, i in enumerate(targets):
            w.writerow([rows[i]["path"]]
                       + sum(([refs[j]["class_id"], round(d, 3)] for d, j in hits[t]), []))

    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_ok = int((dists < thr).sum())
    if not quiet:
        print(f"  识别 {len(targets)} 张（{how}）：距离 p50 {np.percentile(dists, 50):.3f}，"
              f"<{thr} 的 {n_ok}/{len(dists)} 张（{n_ok / len(dists):.0%}）")
        print(f"  → index.csv 的 class_id 三列 + cache/auto_labels.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="截图格 → class_id（自动识别 + 候选）")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--top", type=int, default=12, help="保存几个候选到 auto_labels.csv")
    ap.add_argument("--thr", type=float, default=0.30, help="只用于统计：距离低于它算'较可信'")
    ap.add_argument("--force", action="store_true", help="连人工标注过的行(manual)也重算")
    ap.add_argument("--emb", default="", help="预训练特征缓存（默认自动找 cache/emb_*.npz）")
    ap.add_argument("--cache", default="", help="中间产物目录（默认 dataset 同级的 cache/）")
    ap.add_argument("--feature", default="auto", choices=["auto", "emb", "pixel"],
                    help="auto=有特征缓存就用它，没有退回像素特征")
    args = ap.parse_args()
    classify(args.dataset, args.catalog, args.top, args.thr, args.force, args.emb,
             args.feature, cache=args.cache)


if __name__ == "__main__":
    main()
