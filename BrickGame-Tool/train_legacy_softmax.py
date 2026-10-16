#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""legacy 配方（闭集 softmax 分类头）套到 Tool 的真实数据 + 2369 类上，用来和现役 embedding 做同尺子对比。

配方照抄 BrickGame-Algorithm/image_classification.py 的结论性部分：
  预训练 MobileNetV3-Large 当**冻结**特征提取器 → GAP → Dense(1024, ReLU) → Dense(2369, softmax)，
  交叉熵训练，推理 = argmax。唯一改动是数据换成 Tool 的（图鉴 1 张/类 + 截图真实格）与
  类别数换成 2369 —— 这正是"如果把 legacy 的方案直接搬到现在这个游戏上会怎样"。

**数据与评测完全复用 `train_classifier.py`**（`Catalog`/`Real`/`to_tensor`/`aug_light`/`eval_real`
的划分逻辑），否则两边尺子不同、结论不可比。所以本文件 import train_classifier，不重新实现数据层。

两个 flag 决定实验臂：
  --freeze-backbone --aug none   → L1：严格 legacy（冻结骨干、无增广）
  去掉 freeze、--aug light       → L2：闭集 softmax + 骨干微调 + 增广（把"目标函数"和"配方细节"拆开）

用法：
  python train_legacy_softmax.py --freeze-backbone --aug none --out out/legacy_softmax/L1
  python train_legacy_softmax.py --aug light --out out/legacy_softmax/L2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import train_classifier as TC  # noqa: E402  数据层/预处理/评测口径的唯一来源


# ---------------------------------------------------------------- 模型
class SoftmaxNet(nn.Module):
    """legacy 的闭集分类头：预训练骨干（可冻结）+ Dense(1024)+ReLU + Dense(n_class)。

    与 `image_classification.py` 的 Sequential(base, GAP, Dense(1024), Dense(N, softmax)) 等价，
    只把 Keras 的 softmax 换成原始 logits（softmax 与 loss 在 F.cross_entropy 里做，等价且数值更稳）。
    """

    def __init__(self, backbone: str, n_class: int, hidden: int = 1024, drop: float = 0.0):
        super().__init__()
        self.backbone_name = backbone
        self.body = TC.build_body(backbone)              # 已含 AdaptiveAvgPool+Flatten
        self.feat_dim = TC._BODY_DIM[backbone]
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_class),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.body(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------- 数据（复用 TC 的划分）
def build_samples(cat: "TC.Catalog", real: "TC.Real") -> tuple[list[np.ndarray], np.ndarray]:
    """训练集 = 每类的图鉴图（整格，96x96）+ 该类的真实截格（64x64），全部原样。

    留出类的图鉴图**不进**（与 TC.real_batch 一致），这样"新物品"在训练期完全没见过 ——
    这正是评测里 top1_unseen_cls 要问的事。
    """
    imgs: list[np.ndarray] = []
    labs: list[int] = []
    for i, c in enumerate(cat.class_ids):
        if c in real.holdout_classes:
            continue
        imgs.append(cat.img[i])
        labs.append(i)
    for ci, p in real.train:
        imgs.append(real.cache[p])
        labs.append(ci)
    return imgs, np.array(labs, np.int64)


def balanced_p(y: np.ndarray, n_class: int) -> np.ndarray:
    """按 1/类频 采样 → 每类被抽到的概率相等，抵消"2369 类各 1 张 vs 77 类几百张"的失衡。"""
    cnt = np.bincount(y, minlength=n_class).astype(np.float64)
    w = 1.0 / cnt[y]
    return (w / w.sum()).astype(np.float64)


# ---------------------------------------------------------------- 训练
def train(args, dev: str, cat: "TC.Catalog", real: "TC.Real", rng: np.random.Generator
          ) -> SoftmaxNet:
    imgs, y = build_samples(cat, real)
    n_class = cat.n
    model = SoftmaxNet(args.backbone, n_class, args.hidden).to(dev)
    body_params = [p for n, p in model.named_parameters() if n.startswith("body.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("body.")]
    if args.freeze_backbone:
        for p in body_params:
            p.requires_grad = False

    groups = [{"params": head_params, "lr": args.lr}]
    if body_params and not args.freeze_backbone:
        groups.append({"params": body_params, "lr": args.lr * args.backbone_lr_scale})
    # 冻结时只训 MLP 头：单独用大一点的 lr，否则 15 epoch 收敛不完（头是随机初始化的）
    if args.freeze_backbone:
        groups[0]["lr"] = args.head_lr
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"臂 {'L1 legacy(冻结)' if args.freeze_backbone else 'L2 softmax(微调)'}："
          f"骨干 {args.backbone}，{n_class} 路 softmax，输入 {TC.IMG}，"
          f"可训参数 {n_tr / 1e6:.2f}M，增广 {args.aug}，"
          f"训练样本 {len(y)}（{len(np.unique(y))} 类有样本）")

    # 冻结骨干：特征只算一次 → 严格等价于"冻结"，但快一个数量级
    feats = None
    if args.freeze_backbone:
        model.eval()
        buf = []
        with torch.no_grad():
            for i in range(0, len(imgs), args.eval_batch):
                x = TC.to_tensor_batch(imgs[i:i + args.eval_batch]).to(dev)
                buf.append(model.features(x).float().cpu())
        feats = torch.cat(buf)                       # (N, feat_dim)
        print(f"  已缓存骨干特征 {tuple(feats.shape)}（冻结路径）")

    p = balanced_p(y, n_class)
    yt = torch.from_numpy(y)
    hist = []
    for ep in range(args.epochs):
        model.train() if not args.freeze_backbone else model.head.train()
        t0, run = time.time(), 0.0
        for _ in range(args.steps_per_epoch):
            idx = rng.choice(len(y), size=args.batch_size, replace=True, p=p)
            if feats is not None:
                x = feats[idx].to(dev)
                logits = model.head(x)
            else:
                batch = [TC.aug_light(imgs[int(i)], rng) if args.aug == "light" else imgs[int(i)]
                         for i in idx]
                x = TC.to_tensor_batch(batch).to(dev)
                logits = model(x)
            loss = F.cross_entropy(logits, yt[idx].to(dev), label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not args.freeze_backbone:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss.detach())
        sched.step()
        hist.append({"epoch": ep + 1, "loss": run / args.steps_per_epoch})
        print(f"epoch {ep + 1:3d}/{args.epochs} loss {run / args.steps_per_epoch:.4f} "
              f"{time.time() - t0:.1f}s", flush=True)
    return model


# ---------------------------------------------------------------- 评测（镜像 TC.eval_real）
@torch.no_grad()
def logits_of(model: SoftmaxNet, imgs: np.ndarray, dev: str, bs: int) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = TC.to_tensor_batch(imgs[i:i + bs]).to(dev)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def eval_softmax(model, cat: "TC.Catalog", real: "TC.Real", dev: str, bs: int) -> dict:
    """与 `TC.eval_real` 同一把尺子：留出帧的真实格当查询，在 n_class 类上算 top1/top5。

    差异只有一处：这里是 argmax logits（闭集判别），TC 那边是最近原型（开集检索）。
    额外报一列**闭集口径**（候选限制在验证集里真实出现过的类）—— legacy 的 best-case 尺子。
    """
    if not real.val:
        return {}
    logits = logits_of(model, real.imgs(real.val), dev, bs)
    lab = np.array([c for c, _ in real.val])
    D = -logits                                       # 越小越可能（与 TC 的距离矩阵方向统一）
    order = np.argsort(D, axis=1)
    hit1 = order[:, 0] == lab
    hit5 = (order[:, :5] == lab[:, None]).any(1)
    per_cls: dict[int, list[float]] = {}
    for i, c in enumerate(lab):
        per_cls.setdefault(int(c), []).append(float(hit1[i]))
    seen = np.array([c in real.by_class for c in lab])

    # 闭集口径：候选 = 验证集里真实出现过的类（约 78 个）。
    # D = -logits（越小越可能），所以取 argmin —— 用 argmax 会挑到最不可能的类。
    obs = np.array(sorted(set(lab.tolist())), np.int64)
    sub = D[:, obs]
    o1 = obs[np.argmin(sub, axis=1)]
    o5 = obs[np.argsort(sub, axis=1)[:, :5]]
    hit1c = o1 == lab
    hit5c = (o5 == lab[:, None]).any(1)

    res = {
        "n_query": len(lab), "n_class": cat.n, "n_query_class": len(per_cls),
        "top1": float(hit1.mean()), "top5": float(hit5.mean()),
        "top1_macro": float(np.mean([np.mean(v) for v in per_cls.values()])),
        "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum()),
        "top1_seen_cls": float(hit1[seen].mean()) if seen.any() else 0.0,
        "top1_unseen_cls": float(hit1[~seen].mean()) if (~seen).any() else 0.0,
        "closed_n_class": int(obs.size),
        "top1_closed": float(hit1c.mean()), "top5_closed": float(hit5c.mean()),
        "top1_closed_seen": float(hit1c[seen].mean()) if seen.any() else 0.0,
        "top1_closed_unseen": float(hit1c[~seen].mean()) if (~seen).any() else 0.0,
    }
    return res


# ---------------------------------------------------------------- 主流程
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="legacy 闭集 softmax 头 + Tool 数据/类别数的对照实验")
    ap.add_argument("--catalog", default=os.path.join(_HERE, "build-catalog", "catalog"))
    ap.add_argument("--real", default=os.path.join(_HERE, "build-dataset", "dataset"))
    ap.add_argument("--samples", default=os.path.join(_HERE, "build-dataset", "samples"))
    ap.add_argument("--out", default=os.path.join(_HERE, "out/legacy_softmax/L1"))
    # 与 train_classifier.py 同名同义，保证两边划分完全一致
    ap.add_argument("--val-frames", type=int, default=0)
    ap.add_argument("--val-cls-ratio", type=float, default=0.15)
    ap.add_argument("--holdout-cls-ratio", type=float, default=0.0)
    ap.add_argument("--aug", default="none", choices=["none", "light"])
    # 模型
    ap.add_argument("--backbone", default="mobilenet_v3_large", choices=sorted(TC._BACKBONES))
    ap.add_argument("--img", type=int, default=224, help="模型输入边长（三臂必须一致）")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--backbone-lr-scale", type=float, default=0.1)
    # 训练
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=3e-3, help="冻结骨干时只训头的 lr")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # 尺寸是 TC 的模块级全局，to_tensor_batch / Catalog 都读它
    TC.IMG = args.img
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cat = TC.Catalog(args.catalog, TC.CROP)
    real = TC.Real(args.real, cat, args.val_frames, args.seed,
                   args.holdout_cls_ratio, args.val_cls_ratio)
    print(f"设备 {dev}；留出类 {len(real.holdout_classes)} 个（其图鉴图与真实格都不进训练）")

    model = train(args, dev, cat, real, rng)
    ev = eval_softmax(model, cat, real, dev, args.eval_batch)

    torch.save({"model": model.state_dict(), "backbone": args.backbone,
                "n_class": cat.n, "img": args.img, "hidden": args.hidden,
                "freeze_backbone": args.freeze_backbone, "args": vars(args)},
               os.path.join(args.out, "softmax.pt"))
    cfg = {k: getattr(args, k) for k in
           ("backbone", "freeze_backbone", "aug", "img", "hidden", "epochs",
            "steps_per_epoch", "batch_size", "lr", "head_lr", "seed",
            "val_cls_ratio", "holdout_cls_ratio", "val_frames")}
    if ev:
        print(f"\n=== 评测（{ev['n_query']} 格 / {ev['n_query_class']} 类 / {cat.n} 类原型）")
        print(f"  [全类] top1 {ev['top1']:.4f}  top5 {ev['top5']:.4f}  逐类平均 {ev['top1_macro']:.4f}"
              f"（随机基线 {1 / cat.n:.4f}）")
        print(f"  有训练格 {ev['top1_seen_cls']:.4f}（{ev['n_seen']} 格）"
              f" / 只在留出帧出现 {ev['top1_unseen_cls']:.4f}（{ev['n_unseen']} 格）")
        print(f"  [闭集 {ev['closed_n_class']} 类] top1 {ev['top1_closed']:.4f}  top5 {ev['top5_closed']:.4f}"
              f"；seen {ev['top1_closed_seen']:.4f} / unseen {ev['top1_closed_unseen']:.4f}")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"config": cfg, "softmax": ev,
                   "holdout_classes": sorted(real.holdout_classes)}, f, indent=2, ensure_ascii=False)
    print(f"输出: {args.out}/softmax.pt / metrics.json")


if __name__ == "__main__":
    main()
