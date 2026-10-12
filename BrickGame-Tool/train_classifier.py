#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物品识别模型训练：素材库(图鉴) → 跨域身份 embedding + 原型库。

设计要点（与方案 v2 对应）：

1. **每类只有 1 张参考图**（图鉴的一个格子），所以不能做 N 路 softmax（2667 类会
   直接背下那 1 张）。用"度量学习 + 原型最近邻"：正样本 = 同一素材的不同增强视图。
2. **跨类之间全是负样本**：这一点很关键 —— 二萬/三萬 这种"只差一个字"的物品会被
   强行推开，不需要额外设计。实测像素基线对这类几乎无解。
3. **域合成/底色重染**：图鉴是美术参考域（统一亮绿底板、无压缩），游戏里是截图域
   （底板色随状态变、图标更大、有 JPEG 压缩）。这里把**整格图**的底板像素换成从真实截图
   采样的底板（底色重染 = 状态增广），再叠加缩放/平移/亮度/色彩/噪声/JPEG/模糊。
   不用去底掩码：底板每次都变，模型只能学图标本身；去底掩码的边缘残留反而会伤训练
   （实测），所以 catalog/icons 默认不生成。
4. **评测必须有真值**：留出 10% 的类完全不参与训练，用它们的图鉴图当原型、用困难
   增强视图当查询，算 top-1 / 同类异类间隔 —— 这正是"上线后遇到新物品"的场景。
   另附:真实截图的帧内一致性检查（无真值，只能证明不比像素基线差）。

全项目**唯一的训练脚本**：模型（`Embedder`）与数据增强都定义在本文件里，不依赖其它
训练脚本。上游数据由 build-catalog/（图鉴素材）与 build-dataset/（截图标注）产出。

用法：
  python train_classifier.py --catalog catalog --samples samples --out out/clf --epochs 30
  python train_classifier.py --eval-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-catalog"))
import tile_recognizer as T                 # noqa: E402
from build_catalog import bg_surface        # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))

CROP = 96      # 合成画布尺寸
IMG = 64       # 模型输入尺寸


# ---------------------------------------------------------------- 模型
def to_tensor(batch: np.ndarray) -> torch.Tensor:
    """BGR uint8 (N,64,64,3) → RGB float32 (N,3,64,64)，ImageNet 归一化。"""
    x = torch.from_numpy(batch[..., ::-1].copy()).float().permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


class Embedder(nn.Module):
    """小 CNN：64x64 → 128 维特征 → 64 维身份向量（L2 归一化）。状态头挂在特征上。

    用 GroupNorm 而非 BatchNorm：batch 里的样本来自不同设备/状态，
    统计量本来就混（实测 BN 下同图标距离抖动明显），GroupNorm 不吃 batch 统计。

    n_state > 0 时额外输出一个"底板状态"分类头（冻结/选中），当前训练用 0（关闭）；
    开启后需要同时给状态标签，见 game_visual_recognition_mobile_optimized_solution.md。
    """

    def __init__(self, embed_dim: int = 64, n_state: int = 0, width: int = 32):
        super().__init__()

        def block(cin: int, cout: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
                nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
            )

        self.stem = block(3, width, 1)          # 64
        self.b1 = block(width, width * 2, 2)    # 32
        self.b2 = block(width * 2, width * 4, 2)  # 16
        self.b3 = block(width * 4, width * 6, 2)  # 8
        self.feat_dim = width * 6
        self.embed = nn.Linear(self.feat_dim, embed_dim)
        self.state = nn.Linear(self.feat_dim, n_state) if n_state > 0 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.b3(self.b2(self.b1(self.stem(x))))
        f = F.adaptive_avg_pool2d(h, 1).flatten(1)
        e = F.normalize(self.embed(f), dim=1)
        return e, (self.state(f) if self.state is not None else None)


# ---------------------------------------------------------------- 数据
class Icons:
    """图鉴素材库：每个 class_id 一条参考图。

    默认用 catalog/tiles 的**整格图**（含底板），A 通道全部为不透明 —— 让模型自己学"忽略底板"，
    底板状态交给"底色重染增广"来制造多样性。catalog/icons 的去底 BGRA 只在显式生成过
    （build_catalog.py --with-icons）时才用；实测去底掩码的边缘残留会伤训练，故不作为默认。
    """

    def __init__(self, catalog: str, max_classes: int = 0, seed: int = 0):
        rows = list(csv.DictReader(open(os.path.join(catalog, "index.csv"))))
        by_class: dict[str, list[dict]] = {}
        for r in rows:
            by_class.setdefault(r["class_id"], []).append(r)
        ids = sorted(by_class, key=lambda c: int(c))
        if max_classes and len(ids) > max_classes:
            rng = np.random.default_rng(seed)
            ids = sorted(rng.choice(ids, max_classes, replace=False).tolist(), key=lambda c: int(c))
        use_icon = bool(rows and rows[0].get("icon")) and os.path.exists(
            os.path.join(catalog, rows[0]["icon"]))
        self.class_ids = [int(c) for c in ids]
        self.entries: list[str] = []
        self.faded = np.zeros(len(ids), np.int8)
        self.icon = np.zeros((len(ids), CROP, CROP, 4), np.uint8)
        for i, c in enumerate(ids):
            r = by_class[c][0]                     # 去重后每类基本只有一条
            if use_icon:
                img = cv2.imread(os.path.join(catalog, r["icon"]), cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 3:
                    img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
            else:
                bgr = cv2.imread(os.path.join(catalog, r["tile"]), cv2.IMREAD_COLOR)
                img = np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)])
            if img is None:
                raise SystemExit(f"读不到 {r.get('icon') or r['tile']}，先跑 build_catalog.py")
            if img.shape[:2] != (CROP, CROP):
                img = cv2.resize(img, (CROP, CROP), interpolation=cv2.INTER_AREA)
            self.icon[i] = img
            self.entries.append(r["entry"])
            self.faded[i] = int(r["faded"])
        self.n = len(ids)
        print(f"素材库: {self.n} 类（其中疑似未解锁 {int(self.faded.sum())} 条），"
              f"每类参考图 1 张，来源 {'去底图标 BGRA' if use_icon else '整格图 tiles(含底板)'}")

    def canvas_from_catalog(self, i: int) -> np.ndarray:
        """图鉴底板 = 该素材自己的底板（拟合外推得到，不含图标）。"""
        crop = self.icon[i][:, :, :3].astype(np.float32)
        return bg_surface(crop)


def real_canvases(samples: str, limit: int = 400, seed: int = 0) -> np.ndarray:
    """从真实截图里取"游戏格底板"当合成画布 —— 域合成的关键素材。"""
    out = []
    for f in sorted(x for x in os.listdir(samples) if x.lower().endswith((".jpg", ".png"))):
        img = cv2.imread(os.path.join(samples, f))
        if img is None:
            continue
        try:
            g = T.detect_grid(img, 12, T.INSET, verbose=False)
        except SystemExit:
            continue
        xs, ys = g["xlines"], g["ylines"]
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                c = img[int(ys[j]):int(ys[j + 1]), int(xs[i]):int(xs[i + 1])]
                if c.size == 0 or min(c.shape[:2]) < 24:
                    continue
                out.append(bg_surface(cv2.resize(c, (CROP, CROP))))
        if len(out) >= limit:
            break
    arr = np.stack(out[:limit]) if out else np.zeros((1, CROP, CROP, 3), np.float32)
    print(f"真实底板画布: {len(arr)} 张（来自 {samples}）")
    return arr


# ---------------------------------------------------------------- 合成
def _cell_bg(img: np.ndarray, w: int = 6) -> np.ndarray:
    """整格图的"当前底板色" = 外圈中位色（图标居中，够不到 6px 外圈）。"""
    ring = np.zeros(img.shape[:2], dtype=bool)
    ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
    return np.median(img[ring].astype(np.float32), axis=0)


def _cell_mask(img: np.ndarray, bg: np.ndarray, tol: float = 14.0, frac: float = 0.30
               ) -> np.ndarray:
    """整格图的图标掩码（自适应阈值）。

    自适应是必须的：未解锁的格子是降饱和渲染，与底板差异很小，固定阈值会把整格判成底板，
    连图标一起被重染成底板色 → 身份信息直接没了。
    """
    mag = np.linalg.norm(img - bg[None, None, :], axis=2)
    contrast = float(np.percentile(mag, 99.5))
    m = (mag > max(tol, frac * contrast)).astype(np.uint8)
    return cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)


def render(icon_bgra: np.ndarray, canvas: np.ndarray, rng: np.random.Generator,
           hard: bool = False) -> np.ndarray:
    """素材 → 训练样本，并叠加与真实截图同源的退化。hard=True 用于评测。

    两条路径按素材类型分开：
    - 去底 BGRA（有 alpha）：图标贴到画布（真实底板）上 —— 跨域合成。
    - 整格图（alpha 全 255）：图标保持原样，**只把底板像素换成画布色**（底色重染）。
      这才是"让模型自己忽略底板"的做法：底板每次换个颜色，它就没法拿底板当身份特征。
    """
    c = canvas.astype(np.float32).copy()
    ic = icon_bgra[:, :, :3].astype(np.float32)
    a = (icon_bgra[:, :, 3:4].astype(np.float32) / 255.0)
    opaque = bool(a.min() > 0.999)

    # 1) 底板染色（模拟冻结/选中/冰雾等状态改变底板色）
    if rng.random() < (0.5 if not hard else 0.7):
        tint = rng.integers(0, 256, 3).astype(np.float32)
        c = c * (1 - 0.25) + tint[None, None, :] * 0.25

    if opaque:
        m = _cell_mask(ic, _cell_bg(ic.astype(np.uint8)))
        c = np.where(m[..., None], ic, c)

    # 2) 缩放 + 平移（亚像素网格误差与机型缩放的等价增广）
    s = float(rng.uniform(0.80, 1.25) if hard else rng.uniform(0.88, 1.15))
    M = np.array([[s, 0, CROP / 2 * (1 - s) + rng.uniform(-3, 3)],
                  [0, s, CROP / 2 * (1 - s) + rng.uniform(-3, 3)]], np.float32)
    if opaque:
        out = cv2.warpAffine(c, M, (CROP, CROP), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    else:
        ic_s = cv2.warpAffine(ic, M, (CROP, CROP), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        a_s = cv2.warpAffine(a, M, (CROP, CROP), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)[..., None]
        out = ic_s * a_s + c * (1 - a_s)

    # 3) 整体降饱和/褪色（模拟未解锁渲染，以及低对比度截图）
    if rng.random() < (0.25 if hard else 0.12):
        k = float(rng.uniform(0.45, 0.85) if hard else rng.uniform(0.6, 0.9))
        gray = out.mean(axis=2, keepdims=True)
        out = out * k + gray * (1 - k)

    # 4) 光度抖动
    out = out * (1.0 + rng.uniform(-0.15, 0.15)) + rng.uniform(-12, 12)
    if rng.random() < 0.3:
        blk = rng.integers(0, 3, 2)
        if blk[0]:
            out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), 0) if hard else out

    # 5) 内存里模拟缩放差异（不同机型格子像素数不同）
    if rng.random() < 0.5:
        k = float(rng.uniform(1.0, 1.4) if hard else rng.uniform(1.0, 1.25))
        small = cv2.resize(out, (int(CROP / k), int(CROP / k)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (CROP, CROP), interpolation=cv2.INTER_LINEAR)

    out = np.clip(out, 0, 255)
    # 6) JPEG 压缩伪影（真实截图都是 JPEG）
    q = int(rng.integers(45, 92) if hard else rng.integers(60, 96))
    ok, buf = cv2.imencode(".jpg", out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
    out += rng.normal(0, 4.0 if hard else 2.5, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (IMG, IMG), interpolation=cv2.INTER_AREA)


def to_tensor_batch(batch: np.ndarray) -> torch.Tensor:
    return to_tensor(np.stack([cv2.resize(b, (IMG, IMG)) for b in batch]))


def supcon(emb: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """每个锚点只有同类的另一视图是正样本（每类只有 1 张参考图）。"""
    n = emb.shape[0]
    if n < 3:
        return emb.new_zeros(())
    logits = emb @ emb.T / tau
    logits = logits - logits.max(1, keepdim=True).values.detach()
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    if not bool(pos.any()):
        return emb.new_zeros(())
    lp = logits - torch.log((logits.exp() * ~eye).sum(1, keepdim=True) + 1e-12)
    return -(lp * pos).sum(1)[pos.sum(1) > 0] / pos.sum(1)[pos.sum(1) > 0].clamp(min=1)


# ---------------------------------------------------------------- 评测
@torch.no_grad()
def embed(models: torch.nn.Module, imgs: np.ndarray, device: str, bs: int = 256) -> np.ndarray:
    models.eval()
    out = []
    for i in range(0, len(imgs), bs):
        e, _ = models(to_tensor_batch(imgs[i:i + bs]).to(device))
        out.append(e.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 64), np.float32)


def eval_holdout(model, icons: Icons, bgs: np.ndarray, hold: np.ndarray, device: str,
                 rng: np.random.Generator, n_view: int = 4) -> dict:
    """留出类：图鉴图当原型，困难增强视图当查询。这是"新物品上线"的场景。"""
    gal = embed(model, np.stack([render(icons.icon[i], icons.canvas_from_catalog(i), rng)
                                 for i in range(icons.n)]), device)
    proto = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)
    hq = np.where(hold)[0]
    if hq.size == 0:
        return {}
    q_imgs, q_lab = [], []
    for i in hq.tolist():
        for v in range(n_view):
            q_imgs.append(render(icons.icon[i], bgs[rng.integers(len(bgs))], rng, hard=True))
            q_lab.append(i)
    q = embed(model, np.stack(q_imgs), device)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    D = np.clip(1.0 - qn @ proto.T, 0, 2)                 # (nq, n_class)
    pred = D.argmin(1)
    top1 = float((pred == np.array(q_lab)).mean())
    same = D[np.arange(len(q_lab)), np.array(q_lab)]
    D2 = D.copy()
    D2[np.arange(len(q_lab)), np.array(q_lab)] = 9.0      # 最近的异类
    other = D2.min(1)
    unknown = embed(model, np.stack([b for b in bgs[:64]]), device)   # 纯底板=无物品
    un = np.clip(1.0 - (unknown / (np.linalg.norm(unknown, axis=1, keepdims=True) + 1e-8)) @ proto.T, 0, 2).min(1)
    return {"n_query": len(q_lab), "n_class": icons.n, "top1": top1,
            "same_mean": float(same.mean()), "same_p99": float(np.percentile(same, 99)),
            "other_min_p1": float(np.percentile(other, 1)), "other_mean": float(other.mean()),
            "margin": float(other.mean() - same.mean()),
            "bg_min_p5": float(np.percentile(un, 5)), "bg_mean": float(un.mean())}


def eval_pairs(model, icons: Icons, gray_pairs, rng, device) -> dict:
    """灰区近重复对（同一素材的两次渲染）在 embedding 里是否靠近 —— 有真值的同类检查。"""
    if not gray_pairs:
        return {}
    d = []
    for i, j in gray_pairs:
        a = render(icons.icon[i], icons.canvas_from_catalog(i), rng)
        b = render(icons.icon[j], icons.canvas_from_catalog(j), rng)
        e = embed(model, np.stack([a, b]), device)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        d.append(float(1.0 - float(e[0] @ e[1])))
    return {"n": len(d), "mean": float(np.mean(d)), "max": float(np.max(d))}


def eval_frames(model, samples: str, device: str) -> dict:
    """真实截图的帧内一致性（无身份真值，只能与像素基线对比）。"""
    rows = []
    for f in sorted(x for x in os.listdir(samples) if x.lower().endswith((".jpg", ".png"))):
        img = cv2.imread(os.path.join(samples, f))
        try:
            g = T.detect_grid(img, 12, T.INSET, verbose=False)
        except SystemExit:
            continue
        holes = (g["score"] < g["score_thr"]).ravel()
        idx = np.where(~holes)[0]
        if idx.size < 4:
            continue
        raw = g["patches"][idx].astype(np.uint8)                     # 32x32 原样裁切
        pix = g["dev"][idx]                                          # 32x32 去底板
        # 像素基线
        Db = T.aligned_distance_matrix(pix)
        sub_b, _, ok_b = T.cluster_by_parity(Db, T.gap_threshold(Db))
        # 模型
        e = embed(model, raw, device)
        Dm = np.clip(1.0 - e @ e.T, 0, 2).astype(np.float32)
        np.fill_diagonal(Dm, 0.0)
        sub_m, _, ok_m = T.cluster_by_parity(Dm, T.gap_threshold(Dm))
        rows.append({"frame": f, "cells": int(idx.size),
                     "pix_clusters": int(len(set(sub_b.tolist()))), "pix_parity_ok": bool(ok_b),
                     "mdl_clusters": int(len(set(sub_m.tolist()))), "mdl_parity_ok": bool(ok_m),
                     "pix_max_intra_gap": float(np.max(np.sort(Db[np.triu_indices_from(Db, 1)])[:max(1, idx.size)]))})
    if not rows:
        return {}
    return {"frames": len(rows),
            "pix_parity_ok": float(np.mean([r["pix_parity_ok"] for r in rows])),
            "mdl_parity_ok": float(np.mean([r["mdl_parity_ok"] for r in rows])),
            "pix_clusters": float(np.mean([r["pix_clusters"] for r in rows])),
            "mdl_clusters": float(np.mean([r["mdl_clusters"] for r in rows])), "detail": rows}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="图鉴素材库 → 物品 embedding + 原型库")
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--samples", default=os.path.join(_ROOT, "build-dataset", "samples"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "out/clf"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--classes-per-batch", type=int, default=64)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--max-classes", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    icons = Icons(args.catalog, args.max_classes, args.seed)
    bgs = real_canvases(args.samples, 400, args.seed)
    cat_bgs = np.stack([icons.canvas_from_catalog(i) for i in range(min(icons.n, 300))])
    bank = np.concatenate([bgs, cat_bgs], 0)

    hold = (np.arange(icons.n) % args.holdout_every == args.holdout_every - 1)
    train_idx = np.where(~hold)[0]
    print(f"训练类 {train_idx.size} / 留出类 {int(hold.sum())}（留出类完全不参与训练）")

    # 有真值的同类对：去重组内成员（同素材） + 灰区近重复对（同素材的两次渲染）
    gray_pairs = []
    name2i = {e: i for i, e in enumerate(icons.entries)}
    for fn, key_a, key_b in (("classes.csv", "entries", None), ("gray_pairs.csv", "entry_a", "entry_b")):
        p = os.path.join(args.catalog, fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if key_b is None:
                if int(r.get("n", 1)) <= 1:
                    continue
                ms = [name2i.get(x) for x in r[key_a].split("|")]
            else:
                ms = [name2i.get(r[key_a]), name2i.get(r[key_b])]
            ms = [m for m in ms if m is not None]
            for a in range(len(ms)):
                for b in range(a + 1, len(ms)):
                    gray_pairs.append((ms[a], ms[b]))
    print(f"有真值的同素材对: {len(gray_pairs)} 对（去重组成员 + 灰区近重复）")

    model = Embedder(args.embed_dim, 0, args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        start = ck.get("epoch", 0)
        print(f"续训: 从 epoch {start} 恢复")

    hist = []
    if not args.eval_only:
        for ep in range(start, args.epochs):
            model.train()
            t0, run, ns = time.time(), 0.0, 0
            for _ in range(args.steps_per_epoch):
                cls = rng.choice(train_idx, min(args.classes_per_batch, train_idx.size), replace=False)
                imgs, labs = [], []
                for c in cls.tolist():
                    canvas = bank[rng.integers(len(bank))]
                    for _v in range(args.views):
                        imgs.append(render(icons.icon[c], canvas, rng))
                        labs.append(c)
                x = to_tensor_batch(np.stack(imgs)).to(dev)
                e, _ = model(x)
                loss = supcon(e, torch.from_numpy(np.array(labs)).to(dev), args.tau).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                run += float(loss)
                ns += len(imgs)
            sched.step()
            msg = f"epoch {ep + 1:3d}/{args.epochs} loss {run / args.steps_per_epoch:.4f} " \
                  f"{time.time() - t0:.1f}s"
            if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
                ev = eval_holdout(model, icons, bank, hold, dev, rng)
                msg += (f" | 留出类 top1 {ev['top1']:.3f} 同类均值 {ev['same_mean']:.3f} / "
                        f"异类均值 {ev['other_mean']:.3f} (间隔 {ev['margin']:.3f})")
                hist.append({"epoch": ep + 1, "loss": run / args.steps_per_epoch, **ev})
                with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                    w.writeheader(); w.writerows(hist)
                torch.save({"model": model.state_dict(), "epoch": ep + 1, "args": vars(args),
                            "embed_dim": args.embed_dim, "width": args.width,
                            "class_ids": icons.class_ids, "entries": icons.entries}, 
                           os.path.join(args.out, "last.pt"))
                if ev["top1"] >= max([h["top1"] for h in hist[:-1]] + [0.0]):
                    torch.save({"model": model.state_dict(), "epoch": ep + 1, "args": vars(args),
                                "embed_dim": args.embed_dim, "width": args.width,
                                "class_ids": icons.class_ids, "entries": icons.entries},
                               os.path.join(args.out, "best.pt"))
                    msg += " → best.pt"
            print(msg, flush=True)

    ck_path = args.resume or os.path.join(
        args.out, "best.pt" if os.path.exists(os.path.join(args.out, "best.pt")) else "last.pt")
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"\n=== 评测（权重 {ck_path}，epoch {ck.get('epoch')}）")
    ev = eval_holdout(model, icons, bank, hold, dev, rng)
    pr = eval_pairs(model, icons, gray_pairs, rng, dev)
    fr = eval_frames(model, args.samples, dev)

    gal = embed(model, np.stack([render(icons.icon[i], icons.canvas_from_catalog(i), rng)
                                 for i in range(icons.n)]), dev)
    proto = (gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    np.savez(os.path.join(args.out, "prototypes.npz"),
             proto=proto, class_ids=np.array(icons.class_ids),
             entries=np.array(icons.entries), faded=icons.faded)

    print(f"留出类({ev.get('n_query', 0)} 查询/{ev.get('n_class', 0)} 类): top1 {ev.get('top1', 0):.3f}，"
          f"同类均值 {ev.get('same_mean', 0):.3f}(p99 {ev.get('same_p99', 0):.3f}) / "
          f"异类均值 {ev.get('other_mean', 0):.3f}(最近异类 p1 {ev.get('other_min_p1', 0):.3f})，"
          f"间隔 {ev.get('margin', 0):.3f}")
    print(f"纯底板(无物品)到最近原型的距离: 均值 {ev.get('bg_mean', 0):.3f}，p5 {ev.get('bg_min_p5', 0):.3f}"
          f"（越大越安全：可用于 unknown 拒识）")
    if pr:
        print(f"同素材异渲染对: {pr['n']} 对，距离均值 {pr['mean']:.3f}，最大 {pr['max']:.3f}")
    if fr:
        print(f"真实截图帧内一致性: {fr['frames']} 帧；像素基线 偶数约束通过 {fr['pix_parity_ok']:.2f} "
              f"平均簇数 {fr['pix_clusters']:.1f}；模型 {fr['mdl_parity_ok']:.2f} 平均簇数 {fr['mdl_clusters']:.1f}")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"holdout": ev, "dup_pairs": pr, "frames": fr, "history": hist}, f, indent=2,
                  ensure_ascii=False)
    print(f"输出: {args.out}/best.pt(权重) / prototypes.npz({icons.n} 类原型) / metrics.json / history.csv")


if __name__ == "__main__":
    main()
