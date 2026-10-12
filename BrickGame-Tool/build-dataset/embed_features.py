#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用预训练 CNN 提取"物品身份"特征并缓存（替代手写的去底像素特征）。

为什么换：实测 400 类（图鉴原图 vs 截图域增强视图，增强含缩放/平移/亮度/JPEG/底色重染）
  去底像素特征（+±3px 对齐）  AUC 0.78，同款 p95 距离 1.00 —— 大量同款完全失效
  resnet18(ImageNet 预训练)   AUC 0.997，同款 p95 0.19 / 异款 p5 0.24 —— 阈值 0.21 几乎不重叠
像素特征失效的典型症状就是"篝火被并进谷仓""麻袋和木桶同一组"。骨干换掉后这两个诉求同时满足：
组内任意两张 ≤ 同款阈值（保证一样），而同一款的不同状态（冻结/选中）又能留在同一组。

产出：cache/emb_<骨干>.npz（cache/ 与数据集同级；项目选定 mobilenet_v3_large）
  ref: (N_ref, 512) float32 L2 归一化；ref_entry: entry 名
  cell: (N_cell, 512)；cell_path: dataset 相对路径（与 index.csv 的 path 对齐）

用法：
  .venv/bin/python embed_features.py --dataset dataset --catalog catalog
  .venv/bin/python embed_features.py --backbone mobilenet_v3_small   # 更快的备选
"""
from __future__ import annotations

import argparse
import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
import sys
import time

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, _HERE)                     # 本目录：兄弟模块 auto_classify
from auto_classify import DEFAULT_BACKBONE    # noqa: E402 项目选定骨干，改那一个常量即整体切换


def _ensure_dinov2_py39() -> None:
    """DINOv2 官方代码用了 `X | None` 注解（Python 3.10+），3.9 下 import 阶段就崩。

    这里在加载前给缓存副本补一行 `from __future__ import annotations`（幂等；
    3.10+ 下也合法，所以不用判断版本）。只动 ~/.cache/torch/hub 里的第三方副本，
    不碰本项目代码。首行插入是安全的：这两个文件开头都是注释，没有 module docstring。
    """
    root = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    for rel in ("dinov2/layers/attention.py", "dinov2/layers/block.py"):
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        src = open(p).read()
        if "from __future__ import annotations" in src:
            continue
        with open(p, "w") as f:
            f.write("from __future__ import annotations\n" + src)


def build_backbone(name: str) -> torch.nn.Module:
    if name == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        body = torch.nn.Sequential(*list(m.children())[:-2])
        return torch.nn.Sequential(body, torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten()).eval()
    if name == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        body = torch.nn.Sequential(*list(m.children())[:-2])
        return torch.nn.Sequential(body, torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten()).eval()
    if name == "mobilenet_v3_small":
        body = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1").features
        return torch.nn.Sequential(body, torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten()).eval()
    if name == "mobilenet_v3_large":
        # V2 权重是更好的训练配方（同参数量，精度更高）
        body = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2").features
        return torch.nn.Sequential(body, torch.nn.AdaptiveAvgPool2d(1),
                                   torch.nn.Flatten()).eval()
    if name.startswith("dinov2_"):
        # 自监督预训练（无标签），对比度/背景变化下的检索比 ImageNet 监督模型稳。
        # 这个模型 forward 直接返回 CLS 向量（vits14 是 384 维），不需要再接池化。
        _ensure_dinov2_py39()
        return torch.hub.load("facebookresearch/dinov2", name, verbose=False).eval()
    raise SystemExit(f"不支持的骨干: {name}")


@torch.no_grad()
def embed(model: torch.nn.Module, paths: list[str], size: int, bs: int, dev: str) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((size, size, 3), np.uint8)
            batch.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)[..., ::-1].copy())
        x = torch.from_numpy(np.stack(batch)).float().permute(0, 3, 1, 2).to(dev) / 255.0
        f = model((x - mean.to(dev)) / std.to(dev)).float().cpu().numpy()
        out.append(f)
        if (i // bs) % 10 == 0:
            print(f"    {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    F = np.concatenate(out) if out else np.zeros((0, 1), np.float32)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def extract(dataset: str, catalog: str, backbone: str = "resnet18", size: int = 224,
            batch: int = 32, out: str = "", device: str = "auto", quiet: bool = False) -> str:
    """给图鉴图 + 截图格提特征并缓存，返回 npz 路径。"""
    if backbone.startswith("dinov2_") and size % 14:
        size = int(round(size / 14)) * 14      # ViT patch=14，边长必须是 14 的整数倍
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
    model = build_backbone(backbone).to(dev)
    cache = os.path.join(os.path.dirname(os.path.abspath(dataset)), "cache")
    os.makedirs(cache, exist_ok=True)
    out_path = out or os.path.join(cache, f"emb_{backbone}.npz")

    refs = list(csv.DictReader(open(os.path.join(catalog, "index.csv"))))
    cells = list(csv.DictReader(open(os.path.join(dataset, "index.csv"))))
    if not quiet:
        print(f"  提特征: {backbone}（{dev}），图鉴 {len(refs)} 张 + 截图格 {len(cells)} 张")
    t0 = time.time()
    Fr = embed(model, [os.path.join(catalog, r["tile"]) for r in refs], size, batch, dev)
    Fc = embed(model, [os.path.join(dataset, r["path"]) for r in cells], size, batch, dev)
    np.savez_compressed(out_path, ref=Fr, ref_entry=np.array([r["entry"] for r in refs]),
                        ref_class=np.array([int(r["class_id"]) for r in refs]),
                        cell=Fc, cell_path=np.array([r["path"] for r in cells]),
                        backbone=backbone)
    if not quiet:
        print(f"  → {out_path}（ref {Fr.shape} / cell {Fc.shape}，{time.time() - t0:.0f}s）")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="预训练 CNN → 身份特征缓存")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    choices=["resnet18", "resnet50", "mobilenet_v3_small",
                             "mobilenet_v3_large", "dinov2_vits14", "dinov2_vitb14"],
                    help=f"默认 {DEFAULT_BACKBONE}（项目选定）；dinov2_* 是自监督预训练，"
                         f"首次会自动下载权重")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="", help="默认写到 cache/emb_<backbone>.npz")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    extract(args.dataset, args.catalog, args.backbone, args.size, args.batch,
            args.out, args.device)


if __name__ == "__main__":
    main()
