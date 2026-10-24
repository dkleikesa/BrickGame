#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 dataset 的截图格自动识别 class_id（模型预标注，给人工确认用）。

**默认用上一版 V2 识别模型**（`androidApp/src/main/assets/brick_embedder_v2.tflite`）：
它就是把要替换掉的那份线上模型，融合检索后直接输出 2369 类的余弦相似度。
用它预标注有两个好处：
  1. 与端上同一套预处理、同一个度量空间 —— 候选就是 App 会给出的答案；
  2. 不再依赖通用 ImageNet 骨干（原来 v3-large 的最近邻会把菠萝认成胡萝卜）。

没有 tflite / 没装 `ai_edge_litert` 时退回旧路径（特征缓存 → 像素特征），只是候选质量差些。

产出：
  dataset/index.csv  就地更新 class_id / cls_dist / cls_src（manual 行不覆盖）
  cache/auto_labels.csv  每格 top-N 候选（class_id + 距离），给 annotate.html 点选
  cache/v2_top32.npz     V2 打分缓存（按 cell 路径 + 模型 mtime），增量只算新格子

用法：
  ~/.venvs/brick-train/bin/python build-dataset/auto_classify.py
  ... --feature pixel        # 强制走旧像素特征（对比用）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
sys.path.insert(0, os.path.dirname(_HERE))   # 项目根：tile_recognizer / build_catalog
sys.path.insert(0, _HERE)                    # 本目录：同流水线的兄弟模块
import tile_recognizer as T  # noqa: E402

# ---- 通用骨干（V2 模型不可用时的退路，2026-10 定稿）----
# 五个候选在 1212 格上比过：mobilenet_v3_large 自洽错误率最低（11.3%）。
DEFAULT_BACKBONE = "mobilenet_v3_large"

# ---- 上一版 V2 识别模型（默认预标注器）----
ASSET_DIR = os.path.join(os.path.dirname(_ROOT), "androidApp", "src", "main", "assets")
V2_TFLITE = os.path.join(ASSET_DIR, "brick_embedder_v2.tflite")
V2_META = os.path.join(ASSET_DIR, "model_meta_v2.json")
V2_BATCH = 32
V2_TOPK = 32       # 缓存里每格保留多少候选（够 UI 点选，也够切 --top）


class V2Scorer:
    """上一版 V2 模型的前向：64×64 格图 → (nClass,) 余弦相似度。

    预处理必须与端上 `CellCropper` 一致：64 → 160 双线性、RGB、0~1（mean/std 已烘进模型）。
    """

    def __init__(self) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as e:  # pragma: no cover - 环境问题，直接说清楚
            raise SystemExit(
                "V2 预标注需要 ai_edge_litert：\n"
                "  uv pip install --python ~/.venvs/brick-train/bin/python ai-edge-litert",
            ) from e
        if not os.path.exists(V2_TFLITE) or not os.path.exists(V2_META):
            raise SystemExit(f"找不到 V2 资产：{V2_TFLITE} / {V2_META}")
        meta = json.load(open(V2_META, encoding="utf-8"))
        if not meta.get("search_fused"):
            raise SystemExit("model_meta_v2.json 里 search_fused=false：本脚本只支持融合检索模型")
        self.class_ids = np.asarray([int(c) for c in meta["class_ids"]], np.int32)
        self.size = int(meta["input"]["size"][-1])          # 契约是 NCHW，取最后一维
        self.stamp = f"{os.path.getmtime(V2_TFLITE):.0f}"
        self.it = Interpreter(model_path=V2_TFLITE, num_threads=4)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        # onnx2tf 会把 NCHW 转成 NHWC，以解释器实际形状为准
        shape = tuple(int(v) for v in self.inp["shape"])
        self.nhwc = shape[-1] == 3
        if shape[0] != V2_BATCH:
            raise SystemExit(f"V2 模型 batch 是 {shape[0]}，脚本按 {V2_BATCH} 喂")

    def _batch(self, paths: list[str]) -> np.ndarray:
        arr = np.zeros((V2_BATCH, self.size, self.size, 3), np.float32)
        for i, p in enumerate(paths):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != self.size:
                img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            arr[i] = img[:, :, ::-1].astype(np.float32) / 255.0
        return arr

    def scores(self, paths: list[str]) -> np.ndarray:
        batch = self._batch(paths)
        if not self.nhwc:
            batch = np.transpose(batch, (0, 3, 1, 2))
        self.it.set_tensor(self.inp["index"], np.ascontiguousarray(batch))
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])


def v2_topk(paths: list[str], k: int, cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (ids, dists)，各 (N, k)；距离 = 1 - 余弦相似度。带增量缓存。"""
    k = max(1, min(k, V2_TOPK))
    scorer = V2Scorer()
    ids = np.zeros((len(paths), k), np.int32)
    dists = np.zeros((len(paths), k), np.float32)
    pos: dict[str, int] = {}
    old_ids = old_scores = None
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        if str(z["stamp"]) == scorer.stamp:
            old_ids, old_scores = z["ids"], z["scores"]
            pos = {p: i for i, p in enumerate(z["path"].tolist())}

    todo = [(i, p) for i, p in enumerate(paths) if p not in pos]
    if todo:
        for s in range(0, len(todo), V2_BATCH):
            chunk = todo[s:s + V2_BATCH]
            sc = scorer.scores([p for _, p in chunk])          # (B, nClass)
            for row, (i, _) in enumerate(chunk):
                order = np.argsort(-sc[row])[:k]
                ids[i] = scorer.class_ids[order]
                dists[i] = 1.0 - sc[row][order]
        # 缓存里已有的行直接搬过来
        for i, p in enumerate(paths):
            if p in pos and old_ids is not None:
                keep = min(k, old_ids.shape[1])
                ids[i, :keep] = old_ids[pos[p], :keep]
                dists[i, :keep] = old_scores[pos[p], :keep]
            elif p in pos and old_ids is None:
                pass
        np.savez_compressed(cache_path, path=np.asarray(paths),
                            ids=ids, scores=(1.0 - dists).astype(np.float32),
                            stamp=scorer.stamp)
    else:
        for i, p in enumerate(paths):
            keep = min(k, old_ids.shape[1])
            ids[i, :keep] = old_ids[pos[p], :keep]
            dists[i, :keep] = old_scores[pos[p], :keep]
    return ids, dists


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

    # 空位格也一起识别："背景"在参考库里是一个正式类别（class_id=0）。
    # 判空是检测器给的判断，让模型独立判一遍可以复核，也避免判空错时砖块静默消失。
    # 只重算"模型自己写的"标签：manual（人工改过）和 app（App 上报，相似度已达标）
    # 都必须保住，否则少跑一次 import_app_labels.py，App 标签就被自动识别顶掉。
    # empty 格允许复核（判空可能错），但结论在下面按模型结果修正 cls_src，不会自相矛盾。
    PROTECT = {"manual", "app"}
    targets = [i for i, r in enumerate(rows)
               if (force or r.get("cls_src") not in PROTECT)
               and (frames is None or r.get("frame") in frames)]
    if frames is not None:
        print(f"  只识别指定的 {len(frames)} 个帧（其余帧的标签原样不动）")
    if not targets:
        print("  识别: 没有需要识别的截图格（都已是人工标注）；要重算加 --force")
        return

    refs = list(csv.DictReader(open(os.path.join(catalog, "index.csv"))))
    emb_path = emb or pick_emb(cache)
    use = feature
    if use == "auto":
        use = "v2" if (os.path.exists(V2_TFLITE) and os.path.exists(V2_META)) else (
            "emb" if emb_path else "pixel")

    do_v2 = use == "v2"
    if do_v2:
        try:
            ids, ds = v2_topk([os.path.join(dataset, rows[i]["path"]) for i in targets],
                              max(1, top), os.path.join(cache, "v2_top32.npz"))
            hits = [[(float(ds[t, j]), int(ids[t, j])) for j in range(ds.shape[1])]
                    for t in range(len(targets))]
            how = f"V2 检索（{V2_TFLITE.split(os.sep)[-1]}）"
        except SystemExit as e:
            if feature == "v2":
                raise
            print(f"  V2 预标注不可用（{e}），退回通用骨干")
            do_v2 = False

    if not do_v2 and use == "emb":
        if not os.path.exists(emb_path):
            raise SystemExit(f"找不到特征缓存 {emb_path}，先跑：python embed_features.py")
        z = np.load(emb_path)
        Fr = z["ref"]
        pos = {p: i for i, p in enumerate(z["cell_path"].tolist())}
        Fc = np.stack([z["cell"][pos[rows[i]["path"]]] for i in targets])
        D = np.clip(1.0 - Fc @ Fr.T, 0.0, 2.0)              # 余弦距离
        order = np.argsort(D, axis=1)[:, :max(1, top)]
        hits = [[(float(D[t, j]), int(refs[j]["class_id"])) for j in order[t]]
                for t in range(len(targets))]
        how = f"特征 {z['backbone']}"
    elif not do_v2:
        dev_r = dev_feature([os.path.join(catalog, r["tile"]) for r in refs])
        dev_q = dev_feature([os.path.join(dataset, rows[i]["path"]) for i in targets])
        hits = top_matches(dev_q, dev_r, max(1, top))
        hits = [[(d, int(refs[j]["class_id"])) for d, j in h] for h in hits]
        how = "像素特征"

    dists = []
    for t, i in enumerate(targets):
        d0, cid0 = hits[t][0]
        prev = (rows[i].get("cls_src") or "").strip()
        rows[i]["class_id"] = int(cid0)
        rows[i]["cls_dist"] = round(d0, 3)
        if prev == "empty" and int(cid0) != 0:
            # 检测器判空、模型却认出砖块：改标 auto，让它进低置信筛选被人看到，
            # 否则 class_id 非 0 + cls_src=empty 这种自相矛盾的行会一直躺在库里。
            rows[i]["cls_src"] = "auto"
        else:
            rows[i]["cls_src"] = prev or "auto"
        dists.append(d0)
    dists = np.array(dists)

    with open(os.path.join(cache, "auto_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"] + sum(([f"cls{k + 1}", f"d{k + 1}"] for k in range(max(1, top))), []))
        for t, i in enumerate(targets):
            w.writerow([rows[i]["path"]]
                       + sum(([cid, round(d, 3)] for d, cid in hits[t]), []))

    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_ok = int((dists < thr).sum())
    if not quiet:
        print(f"  识别 {len(targets)} 张（{how}）：距离 p50 {np.percentile(dists, 50):.3f}，"
              f"<{thr} 的 {n_ok}/{len(dists)} 张（{n_ok / len(dists):.0%}）")
        print("  → index.csv 的 class_id 三列 + cache/auto_labels.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="截图格 → class_id（模型预标注 + 候选）")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--top", type=int, default=12, help="保存几个候选到 auto_labels.csv（≤32）")
    ap.add_argument("--thr", type=float, default=0.30, help="只用于统计：距离低于它算'较可信'")
    ap.add_argument("--force", action="store_true", help="连人工标注过的行(manual)也重算")
    ap.add_argument("--emb", default="", help="预训练特征缓存（默认自动找 cache/emb_*.npz）")
    ap.add_argument("--cache", default="", help="中间产物目录（默认 dataset 同级的 cache/）")
    ap.add_argument("--feature", default="auto", choices=["auto", "v2", "emb", "pixel"],
                    help="auto=有 V2 模型就用 V2，否则退回特征/像素")
    args = ap.parse_args()
    classify(args.dataset, args.catalog, args.top, args.thr, args.force, args.emb,
             args.feature, cache=args.cache)


if __name__ == "__main__":
    main()
