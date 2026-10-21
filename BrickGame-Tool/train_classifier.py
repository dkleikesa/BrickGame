#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物品识别模型训练：图鉴素材 + 真实截图 → 跨域身份 embedding + 2369 类原型库。

**数据（`--data real`，默认）**：只用磁盘上现成的真实图片，不渲染、不重染、不生成新图 ——
  ① 图鉴 build-catalog/catalog/tiles/*.png：每类 1 张，原样
  ② 截图 build-dataset/dataset/images/frame/**：所有格子原样收进来（含空位 = class 0）
正样本对 = 同类（图鉴图 ↔ 该类的真实格），跨域对齐就是域适配的全部信号。
验证集分两把尺子（按格随机划分必虚高：同帧内同款砖块出现十几次）：
  **整类留出**（`--holdout-cls-ratio`）= 新物品/换皮肤，该类图鉴图与真实格都不进训练；
  **整帧留出**（`--val-frames`）= 同类换一张截图的稳定性。

**骨干**：默认从 ImageNet 预训练的 mobilenet_v3_large 微调（与自动识别用的骨干同一个，
见 auto_classify.DEFAULT_BACKBONE）；`--backbone scratch` 回到"从零训小 CNN"的老行为。
预训练骨干走 `--img 224`（ImageNet 原分辨率），scratch 走 64。

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
  python train_classifier.py                       # 默认 --data real：只用真实图，零合成
  python train_classifier.py --data synth          # 旧的合成增广流程（图鉴图重染造视图）
  python train_classifier.py --freeze-backbone     # 只训投影头（数据少时更稳）
  python train_classifier.py --eval-only           # 只评测已有权重
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
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-catalog"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-dataset"))
import tile_recognizer as T                 # noqa: E402
from build_catalog import bg_surface        # noqa: E402
import auto_classify                        # noqa: E402  DEFAULT_BACKBONE：与自动识别保持同一骨干

_ROOT = os.path.dirname(os.path.abspath(__file__))

CROP = 96      # 合成画布尺寸
IMG = 64       # 模型输入尺寸


# ---------------------------------------------------------------- 模型
def to_tensor(batch: np.ndarray) -> torch.Tensor:
    """BGR uint8 (N,H,W,3) → RGB float32 (N,3,H,W)，ImageNet 归一化。

    `.contiguous()` 不能省：`from_numpy(...).permute()` 得到的是非连续张量，
    送进 mobilenet 的 depthwise / Squeeze-Excitation 分支后，**MPS 上的反向**会直接炸
    「view size is not compatible with input tensor's size and stride」（实测 CPU 不炸、MPS 炸）。
    """
    x = torch.from_numpy(batch[..., ::-1].copy()).float().permute(0, 3, 1, 2).contiguous() / 255.0
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


# ---------------------------------------------------------------- 预训练骨干
# 与 embed_features.py 用**同一套骨干定义和同一档权重**（mobilenet_v3_large 用 V2），
# 这样"自动识别用的特征"和"训练起点"是同一个域，微调是在同一个度量空间里往前推。
_BACKBONES = {
    "mobilenet_v2": torchvision.models.mobilenet_v2,
    "mobilenet_v3_large": torchvision.models.mobilenet_v3_large,
    "mobilenet_v3_small": torchvision.models.mobilenet_v3_small,
    "resnet18": torchvision.models.resnet18,
    "resnet50": torchvision.models.resnet50,
}
_BODY_DIM = {"mobilenet_v2": 1280, "mobilenet_v3_large": 960, "mobilenet_v3_small": 576,
             "resnet18": 512, "resnet50": 2048}
# 权重档按骨干逐个写死：mobilenet_v3_small 官方只发过 V1，写 V2 会 KeyError（实测）。
# large 仍用 V2 —— 与 embed_features.py / auto_classify 的选型保持一致。
_WEIGHTS = {"mobilenet_v2": "IMAGENET1K_V1",
            "mobilenet_v3_large": "IMAGENET1K_V2", "mobilenet_v3_small": "IMAGENET1K_V1",
            "resnet18": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V1"}


def build_body(name: str) -> nn.Module:
    """ImageNet 预训练骨干：图像 → 全局池化特征（不含分类头）。

    类别数是 2370 而每类只有 1 张参考图，绝不能用骨干自带的 1000 类头，
    也不能接 2370 路 softmax（会直接背下那一张）。骨干只当特征提取器用。
    """
    if name not in _BACKBONES:
        raise SystemExit(f"不支持的骨干: {name}（可选 scratch / " + " / ".join(_BACKBONES) + "）")
    m = _BACKBONES[name](weights=_WEIGHTS[name])
    body = nn.Sequential(*list(m.children())[:-2]) if name.startswith("resnet") else m.features
    return nn.Sequential(body, nn.AdaptiveAvgPool2d(1), nn.Flatten())


class PretrainedEmbedder(nn.Module):
    """预训练骨干 + 投影头 → 128/256 维 L2 归一化身份向量。

    投影头不是可有可无：ImageNet 特征只负责"通用物体"，对"二萬 / 三萬"这种只差一个字的
    图标区分度不够（实测最近邻会互相抢）。SupCon 把骨干拉进"图鉴域 ↔ 截图域"的共同度量空间，
    投影头就是这段适配的落点 —— 原型库用的也是同一个头的输出，训练/推理才一致。
    """

    def __init__(self, backbone: str = "mobilenet_v3_large", embed_dim: int = 256,
                 n_state: int = 0, drop: float = 0.0):
        super().__init__()
        self.backbone_name = backbone
        self.body = build_body(backbone)
        self.feat_dim = _BODY_DIM[backbone]
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.embed = nn.Linear(self.feat_dim, embed_dim)
        self.state = nn.Linear(self.feat_dim, n_state) if n_state > 0 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        f = self.dropout(self.body(x))
        e = F.normalize(self.embed(f), dim=1)
        return e, (self.state(f) if self.state is not None else None)


def make_model(backbone: str, embed_dim: int, width: int, n_state: int = 0) -> nn.Module:
    if backbone == "scratch":
        return Embedder(embed_dim, n_state, width)
    return PretrainedEmbedder(backbone, embed_dim, n_state)


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


def real_canvases(real_root: str, limit: int = 400, seed: int = 0) -> np.ndarray:
    """从真实截图已裁好的格图里取"游戏格底板"当合成画布 —— 域合成的关键素材。

    读 `build-dataset/dataset/images/frame/<帧>/r*c*.png`（YOLO 网格裁出来的 64x64 格），
    不再对整帧跑一遍 OpenCV 定位：那套已删除，而且两套网格必然不一致。
    """
    root = os.path.join(real_root, "images", "frame")
    paths = []
    if os.path.isdir(root):
        for frame in sorted(os.listdir(root)):
            d = os.path.join(root, frame)
            if os.path.isdir(d):
                paths.extend(os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png"))
    out = []
    for p in paths[:limit] if len(paths) > limit else paths:
        img = cv2.imread(p)
        if img is None or min(img.shape[:2]) < 24:
            continue
        out.append(bg_surface(cv2.resize(img, (CROP, CROP))))
    arr = np.stack(out[:limit]) if out else np.zeros((1, CROP, CROP, 3), np.float32)
    print(f"真实底板画布: {len(arr)} 张（来自 {root}）")
    return arr


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
                 rng: np.random.Generator, n_view: int = 4, bs: int = 256) -> dict:
    """留出类：图鉴图当原型，困难增强视图当查询。这是"新物品上线"的场景。"""
    gal = embed(model, np.stack([render(icons.icon[i], icons.canvas_from_catalog(i), rng)
                                 for i in range(icons.n)]), device, bs)
    proto = gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)
    hq = np.where(hold)[0]
    if hq.size == 0:
        return {}
    q_imgs, q_lab = [], []
    for i in hq.tolist():
        for v in range(n_view):
            q_imgs.append(render(icons.icon[i], bgs[rng.integers(len(bgs))], rng, hard=True))
            q_lab.append(i)
    q = embed(model, np.stack(q_imgs), device, bs)
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



def load_weights(model: nn.Module, ck_path: str, dev: str, args) -> dict:
    """载入权重前先核对骨干/输入尺寸 —— 对不上就直说，别让 load_state_dict 抛一堆 shape 错。"""
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    bb, im = ck.get("backbone"), ck.get("img")
    if bb and bb != args.backbone:
        raise SystemExit(f"{ck_path} 是 {bb} 训出来的，当前 --backbone={args.backbone}；"
                         f"加 --backbone {bb} 重跑")
    if im and int(im) != IMG:
        raise SystemExit(f"{ck_path} 的输入尺寸是 {im}，当前 {IMG}；加 --img {im} 重跑")
    model.load_state_dict(ck["model"])
    return ck


def trainable_groups(model: nn.Module, args) -> list[dict]:
    """分两组学习率：投影头（新初始化的）用 --lr，预训练骨干用 --lr×--backbone-lr-scale。

    骨干一上来就用大学习率会把 ImageNet 特征冲掉（每类只有 1 张参考图，救不回来）。
    """
    head = [p for n, p in model.named_parameters() if not n.startswith("body.")]
    body = [p for n, p in model.named_parameters() if n.startswith("body.")]
    if args.freeze_backbone:
        for p in body:
            p.requires_grad = False
    groups = [{"params": head, "lr": args.lr}]
    if body and not args.freeze_backbone:
        groups.append({"params": body, "lr": args.lr * args.backbone_lr_scale})
    return groups


def ckpt_dict(model: nn.Module, args, ep: int, cat: "Catalog" | None = None,
              icons=None) -> dict:
    """权重 + 复现所需的全部信息（推理端靠 backbone/img 才能重建同一个模型）。"""
    ids = cat.class_ids if cat is not None else icons.class_ids
    entries = cat.entries if cat is not None else icons.entries
    return {"model": model.state_dict(), "epoch": ep, "args": vars(args),
            "arch": "scratch" if args.backbone == "scratch" else "pretrained",
            "backbone": args.backbone, "embed_dim": args.embed_dim, "width": args.width,
            "img": IMG, "crop": CROP, "data": args.data,
            "class_ids": ids, "entries": entries}


def config_dict(args) -> dict:
    return {"data": args.data, "backbone": args.backbone,
            "freeze_backbone": args.freeze_backbone, "img": IMG, "crop": CROP,
            "embed_dim": args.embed_dim, "epochs": args.epochs, "lr": args.lr,
            "backbone_lr_scale": args.backbone_lr_scale, "seed": args.seed, "aug": args.aug,
            "val_frames": args.val_frames, "val_cls_ratio": args.val_cls_ratio,
            "holdout_cls_ratio": args.holdout_cls_ratio}


# ---------------------------------------------------------------- 真实数据（零合成）
# 训练只用磁盘上现成的两类图片，不渲染、不重染、不生成任何新图：
#   ① 图鉴 build-catalog/catalog/tiles/*.png —— 每类 1 张，原样
#   ② 截图 build-dataset/dataset/images/frame/** —— 标注过的格子，原样
# 正样本对来自"同类"：图鉴图 ↔ 该类的真实格，即跨域对齐 —— 这就是域适配的全部信号。
class Catalog:
    """图鉴参考图（每类 1 张真实图，原样保留底色）。"""

    def __init__(self, catalog: str, crop: int):
        by_class: dict[int, dict] = {}
        for r in csv.DictReader(open(os.path.join(catalog, "index.csv"))):
            by_class.setdefault(int(r["class_id"]), r)
        ids = sorted(by_class)
        self.class_ids = ids
        self.index = {c: i for i, c in enumerate(ids)}
        self.entries = [by_class[c]["entry"] for c in ids]
        self.faded = np.array([int(by_class[c]["faded"]) for c in ids], np.int8)
        self.img = np.zeros((len(ids), crop, crop, 3), np.uint8)
        for i, c in enumerate(ids):
            bgr = cv2.imread(os.path.join(catalog, by_class[c]["tile"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise SystemExit(f"读不到 {by_class[c]['tile']}，先跑 build_catalog.py")
            self.img[i] = cv2.resize(bgr, (crop, crop), interpolation=cv2.INTER_AREA)
        self.n = len(ids)
        print(f"图鉴: {self.n} 类，每类 1 张真实参考图（{crop}x{crop}，无合成/无重染）")


class Real:
    """dataset 真实截图格。全部收：空位格是 class 0（图鉴里就有这个类），不做任何筛选。

    为什么不按识别距离筛：每帧都在标注页人工复核过（frames.csv status=done），标签即真值；
    再拿自动识别的距离过滤一遍，等于按"模型当时有多自信"扔掉真实样本（实测扔过半）。
    唯一还会丢的是 unknown（人工判为"认不出"）和不在图鉴里的 class_id —— 后者没有原型可用。

    评测集有三种留出口径，可叠加（同一 seed，同参数结果可复现；哪种都不留 = 没有评测）：

    1. **每类按比例留出**（[val_cls_ratio]，`--val-cls-ratio`，默认 0.15，格级随机）：每个类
       随机抽 15% 的格进评测，**最少 1 张**（比例小于 1，所以不会把多张的类留光；只有 1 张
       样本的类会被整类留出 —— 它本来也没有第二个同款可训）。默认口径，保证每个类都有评测
       样本；**代价是同帧近重复会虚高**：同一帧里同款砖块的底色/缩放/压缩完全一致，留出的
       那 1 格几乎就是训练集里那几格的复制品。
    2. **整类留出**（[holdout_classes]，`--holdout-cls-ratio`，默认关）：整个类的图鉴图与
       真实格都不进训练，测"只给一张图鉴图的新物品 / 换皮肤"。该类**图鉴图也不作负样本**
       （见 real_batch），否则模型表征仍见过它、指标照样偏乐观。
    3. **整帧留出**（[val_frames]，`--val-frames`，默认关）：整帧不训练，测"同类换一张截图"。

    三种都留时，eval_real 按"该类在训练集里有没有格"自动拆成 seen / unseen 两块报。
    """

    def __init__(self, dataset: str, cat: "Catalog", val_frames: int, seed: int,
                 holdout_cls_ratio: float = 0.0, val_cls_ratio: float = 0.0):
        rows = list(csv.DictReader(open(os.path.join(dataset, "index.csv"))))
        kept, drop = [], {"未知": 0, "不在图鉴": 0}
        for r in rows:
            cid = r.get("class_id", "")
            if r.get("cls_src") == "unknown" or not cid:
                drop["未知"] += 1
                continue
            c = int(cid)
            if c not in cat.index:
                drop["不在图鉴"] += 1
                continue
            kept.append((cat.index[c], r["path"], r["frame"]))

        rng = np.random.default_rng(seed)

        # ① 整类留出（默认关）：该类**全部**真实格进验证集，且不进 by_class —— 因此不融合进
        #    原型、也不进 real_batch 的负样本池。至少留 1 类，但必须给训练留下至少 1 类。
        all_cls = np.array(sorted({c for c, _, _ in kept}), np.int64)
        n_hold = max(1, int(round(all_cls.size * holdout_cls_ratio))) if holdout_cls_ratio > 0 else 0
        n_hold = min(n_hold, max(0, all_cls.size - 1))
        self.holdout_classes = set(rng.choice(all_cls, n_hold, replace=False).tolist()) if n_hold else set()

        # ② 每类按比例留出（默认口径）：每类抽 val_cls_ratio，最少 1 张、最多留到给训练剩 1 张
        #    —— 只有 1 张样本的类不留，否则它就没有训练格了。
        per_cls_val: set[str] = set()
        if val_cls_ratio > 0:
            by_cls: dict[int, list[str]] = {}
            for ci, p, _ in kept:
                if ci not in self.holdout_classes:
                    by_cls.setdefault(ci, []).append(p)
            for ps in by_cls.values():
                # 最少 1 张；ratio<1 时抽不满，唯一会"留光"的是全类只有 1 张样本的类。
                n = min(max(1, int(round(len(ps) * val_cls_ratio))), len(ps))
                for k in rng.choice(len(ps), n, replace=False):
                    per_cls_val.add(ps[int(k)])

        # ③ 整帧留出（默认关）：候选帧只从"没被上面挑走"的格里数，别和格级留出重叠
        rest_frames = sorted({fr for ci, p, fr in kept
                              if ci not in self.holdout_classes and p not in per_cls_val})
        n_val = min(val_frames, max(0, len(rest_frames) - 1))
        val = set(rng.choice(rest_frames, n_val, replace=False).tolist()) if n_val else set()

        self.cache: dict[str, np.ndarray] = {}
        self.train: list[tuple[int, str]] = []
        self.val: list[tuple[int, str]] = []
        for ci, p, fr in kept:
            img = cv2.imread(os.path.join(dataset, p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            self.cache[p] = img
            to_val = ci in self.holdout_classes or fr in val or p in per_cls_val
            (self.val if to_val else self.train).append((ci, p))
        self.val_frames = sorted(val)
        self.by_class: dict[int, list[str]] = {}
        for ci, p in self.train:
            self.by_class.setdefault(ci, []).append(p)
        self.train_classes = sorted(self.by_class)
        self.val_holdout = sum(1 for c, _ in self.val if c in self.holdout_classes)
        self.val_per_class = len(per_cls_val)
        dropped = "、".join(f"{k} {v}" for k, v in drop.items() if v) or "无"
        print(f"真实格: 收 {len(kept)}/{len(rows)}（丢弃 {dropped}）→ "
              f"训练 {len(self.train)} 格 / 验证 {len(self.val)} 格"
              f"（每类抽样 {self.val_per_class} 格；留出帧 {self.val_frames or '无'}；"
              f"留出类 {len(self.holdout_classes)} 个 / {self.val_holdout} 格查询）")
        print(f"  涉及 {len({c for c, _ in self.train + self.val})} 类；其中 {len(self.train_classes)} 类"
              f"有训练格（占 {len(self.train_classes) / cat.n:.1%}），其余类只当负样本")

    def imgs(self, items: list[tuple[int, str]]) -> np.ndarray:
        return np.stack([self.cache[p] for _, p in items])


def aug_light(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """真实图的轻量增广：平移 / 亮度对比度 / JPEG。

    刻意不做的事：**不镜像**（图标里有文字、数字，镜像会变成另一个物品）、不改底色、
    不生成任何图鉴合成图。默认 `--aug none` 完全关掉。
    """
    out = img.astype(np.float32)
    if rng.random() < 0.5:
        M = np.float32([[1, 0, rng.uniform(-2, 2)], [0, 1, rng.uniform(-2, 2)]])
        out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    out = out * (1 + rng.uniform(-0.15, 0.15)) + rng.uniform(-10, 10)
    if rng.random() < 0.3:
        g = out.mean(axis=2, keepdims=True)
        out = out * 0.8 + g * 0.2
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def real_batch(cat: "Catalog", real: "Real", rng, n_class: int, views: int, aug: bool,
               real_ratio: float) -> tuple[list[np.ndarray], np.ndarray]:
    """一批 = 若干类 ×（图鉴图 + 该类的真实格）。

    采样**偏向有真实格的类**：2369 类里只有 78 类有真实格，均匀抽 64 个平均只命中 2 个，
    跨域正样本对太稀，等于没在适配。其余类补进来只当负样本 —— 把空间撑开，
    避免模型只顾这 78 类、把别的物品混到一起。

    返回**列表**而不是堆好的数组：图鉴图是 96x96、真实格是 64x64，尺寸不同堆不到一起，
    统一在 to_tensor_batch 里 resize 到 IMG（与 embed() 的预处理保持同一条路）。
    """
    n_real = min(len(real.train_classes), max(1, int(n_class * real_ratio)))
    cls = [int(c) for c in rng.choice(real.train_classes, n_real, replace=False)] if n_real else []
    if len(cls) < n_class:
        # 留出类一个都不许进 batch —— 连图鉴图也不行：让模型在训练期完全没见过它，
        # 评测里"只给一张图鉴图"才是真的新物品场景（见 Real 的注释）。
        banned = set(cls) | real.holdout_classes
        rest = np.array([c for c in range(cat.n) if c not in banned], np.int64)
        if rest.size:
            cls += [int(c) for c in rng.choice(rest, min(n_class - len(cls), rest.size), replace=False)]
    imgs: list[np.ndarray] = []
    labs: list[int] = []
    for c in cls:
        imgs.append(cat.img[c])
        labs.append(c)
        ps = real.by_class.get(c)
        if ps:
            for k in rng.choice(len(ps), min(views, len(ps)), replace=False):
                imgs.append(real.cache[ps[int(k)]])
                labs.append(c)
    if aug:                                   # 每个样本再补一个增广视图 → 每类都有正样本对
        n0 = len(imgs)
        for i in range(n0):
            imgs.append(aug_light(imgs[i], rng))
            labs.append(labs[i])
    return imgs, np.array(labs, np.int64)


def build_prototypes(model, cat: "Catalog", real: "Real", dev: str, bs: int, fuse: bool
                     ) -> np.ndarray:
    """2369 类原型。fuse=True 时把"该类训练帧真实格"的 embedding 均值也融进来 ——
    原型中心因此偏向截图域，检索不必只靠图鉴那一张图。原型只用**训练帧**的真实格，
    不含验证帧，否则评测就泄漏了。"""
    gal = embed(model, cat.img, dev, bs)
    if fuse:
        for c in real.train_classes:
            ps = real.by_class[c]
            e = embed(model, np.stack([real.cache[p] for p in ps]), dev, bs).mean(0)
            g = gal[c] / (np.linalg.norm(gal[c]) + 1e-8)
            gal[c] = 0.5 * g + 0.5 * e / (np.linalg.norm(e) + 1e-8)
    return (gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def eval_real(model, cat: "Catalog", real: "Real", proto: np.ndarray, dev: str, bs: int) -> dict:
    """真实域评测：留出帧的真实格当查询，在 2369 类原型上算 top1/top5。

    除整体（micro）外还报**逐类平均（macro）**：你关心的是"这 78 类准不准"，
    而不是"格子多的类准不准"—— 出现次数多的类会把 micro 拉高。
    另外把"该类有训练格 / 没有训练格"分开报，后者才是"新物品只给一张图鉴图"的场景。
    """
    if not real.val:
        return {}
    q = embed(model, real.imgs(real.val), dev, bs)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    D = np.clip(1.0 - qn @ proto.T, 0.0, 2.0)
    lab = np.array([c for c, _ in real.val])
    order = np.argsort(D, axis=1)
    hit1 = (order[:, 0] == lab)
    hit5 = (order[:, :5] == lab[:, None]).any(1)
    per_cls: dict[int, list[float]] = {}
    for i, c in enumerate(lab):
        per_cls.setdefault(int(c), []).append(float(hit1[i]))
    same = D[np.arange(len(lab)), lab]
    D2 = D.copy()
    D2[np.arange(len(lab)), lab] = 9.0
    other = D2.min(1)
    seen = np.array([c in real.by_class for c in lab])
    return {
        "n_query": len(lab), "n_class": cat.n, "n_query_class": len(per_cls),
        "top1": float(hit1.mean()), "top5": float(hit5.mean()),
        "top1_macro": float(np.mean([np.mean(v) for v in per_cls.values()])),
        "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum()),
        "top1_seen_cls": float(hit1[seen].mean()) if seen.any() else 0.0,
        "top1_unseen_cls": float(hit1[~seen].mean()) if (~seen).any() else 0.0,
        "same_mean": float(same.mean()), "same_p95": float(np.percentile(same, 95)),
        "wrong_dist_p5": float(np.percentile(other, 5)),
        "hit_dist_p95": float(np.percentile(same[hit1], 95)) if hit1.any() else 0.0,
    }


def run_real(args, dev: str) -> None:
    """只用真实图（图鉴 + 截图）训练，零合成。"""
    rng = np.random.default_rng(args.seed)
    cat = Catalog(args.catalog, CROP)
    real = Real(args.real, cat, args.val_frames, args.seed, args.holdout_cls_ratio, args.val_cls_ratio)
    if not real.train:
        raise SystemExit("没有可用的真实训练格：检查 dataset/index.csv 的 cls_src / class_id")

    model = make_model(args.backbone, args.embed_dim, args.width).to(dev)
    opt = torch.optim.AdamW(trainable_groups(model, args), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"骨干 {args.backbone}（{'冻结' if args.freeze_backbone else '微调'}）输入 {IMG}x{IMG}，"
          f"身份向量 {args.embed_dim} 维，可训参数 {n_tr / 1e6:.2f}M，"
          f"增广 {'light' if args.aug == 'light' else '无'}")

    def save_ckpt(path: str, ep: int) -> None:
        torch.save(ckpt_dict(model, args, ep, cat), path)

    start = 0
    if args.resume:
        ck = load_weights(model, args.resume, dev, args)
        start = ck.get("epoch", 0)
        print(f"续训: 从 epoch {start} 恢复（{args.resume}）")

    hist = []
    if not args.eval_only:
        for ep in range(start, args.epochs):
            model.train()
            t0, run = time.time(), 0.0
            for _ in range(args.steps_per_epoch):
                imgs, labs = real_batch(cat, real, rng, args.classes_per_batch, args.views,
                                        args.aug == "light", args.real_ratio)
                x = to_tensor_batch(imgs).to(dev)
                e, _ = model(x)
                loss = supcon(e, torch.from_numpy(labs).to(dev), args.tau).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                run += float(loss.detach())
            sched.step()
            msg = f"epoch {ep + 1:3d}/{args.epochs} loss {run / args.steps_per_epoch:.4f} " \
                  f"{time.time() - t0:.1f}s"
            if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
                proto = build_prototypes(model, cat, real, dev, args.eval_batch, args.fuse_proto)
                ev = eval_real(model, cat, real, proto, dev, args.eval_batch)
                msg += (f" | 留出帧 top1 {ev['top1']:.3f}/top5 {ev['top5']:.3f}"
                        f"（逐类均 {ev['top1_macro']:.3f}，有训练格 {ev['top1_seen_cls']:.3f}"
                        f" / 无 {ev['top1_unseen_cls']:.3f}）")
                hist.append({"epoch": ep + 1, "loss": run / args.steps_per_epoch, **ev})
                with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                    w.writeheader()
                    w.writerows(hist)
                save_ckpt(os.path.join(args.out, "last.pt"), ep + 1)
                if ev["top1"] >= max([h["top1"] for h in hist[:-1]] + [0.0]):
                    save_ckpt(os.path.join(args.out, "best.pt"), ep + 1)
                    msg += " → best.pt"
            print(msg, flush=True)

    ck_path = args.resume or os.path.join(
        args.out, "best.pt" if os.path.exists(os.path.join(args.out, "best.pt")) else "last.pt")
    ck = load_weights(model, ck_path, dev, args)
    print(f"\n=== 评测（权重 {ck_path}，epoch {ck.get('epoch')}，骨干 {ck.get('backbone', '?')}）")
    proto = build_prototypes(model, cat, real, dev, args.eval_batch, args.fuse_proto)
    ev = eval_real(model, cat, real, proto, dev, args.eval_batch)
    np.savez(os.path.join(args.out, "prototypes.npz"),
             proto=proto, class_ids=np.array(cat.class_ids), entries=np.array(cat.entries),
             faded=cat.faded, backbone=np.array(args.backbone), img=np.array(IMG),
             embed_dim=np.array(args.embed_dim), data=np.array(args.data))
    if ev:
        print(f"留出帧真实评测（{ev['n_query']} 格 / {ev['n_query_class']} 类 / 共 {ev['n_class']} 类原型）:")
        print(f"  top1 {ev['top1']:.3f}  top5 {ev['top5']:.3f}  逐类平均 top1 {ev['top1_macro']:.3f}")
        print(f"  有训练格的类 {ev['top1_seen_cls']:.3f}（{ev['n_seen']} 格）"
              f" / 只在留出帧出现的类 {ev['top1_unseen_cls']:.3f}（{ev['n_unseen']} 格）")
        print(f"  命中距离均值 {ev['same_mean']:.3f}(p95 {ev['same_p95']:.3f})，"
              f"最近异类距离 p5 {ev['wrong_dist_p5']:.3f}，命中 p95 {ev['hit_dist_p95']:.3f}"
              f"（可当 low-confidence 阈值参考）")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"config": config_dict(args), "real": ev, "history": hist}, f,
                  indent=2, ensure_ascii=False)
    print(f"输出: {args.out}/best.pt / prototypes.npz({cat.n} 类原型) / metrics.json / history.csv")


# ---------------------------------------------------------------- 主流程
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="图鉴素材 + 真实截图 → 物品 embedding + 原型库")
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--samples", default=os.path.join(_ROOT, "build-dataset", "samples"))
    ap.add_argument("--out", default="",
                    help="输出目录，留空= out/clf_v2（--data real）/ out/clf（--data synth）")
    # ---- 数据来源 ----
    ap.add_argument("--data", default="real", choices=["real", "synth"],
                    help="real（默认）=只用真实图：图鉴参考图 + 截图标注，零合成；"
                         "synth=旧的合成增广流程（图鉴图按底色重染造视图）")
    ap.add_argument("--real", default=os.path.join(_ROOT, "build-dataset", "dataset"),
                    help="--data real 用的截图数据集目录（含 index.csv 与 images/）")
    ap.add_argument("--real-ratio", type=float, default=0.6,
                    help="每个 batch 从\"有真实格的类\"里抽的比例，其余从全类抽（只当负样本）")
    ap.add_argument("--val-frames", type=int, default=0,
                    help="--data real 用：整帧留出做验证，这些帧的格子完全不参与训练（默认 0=关）。"
                         "与 --val-cls-ratio 叠加时，评测集是两者的并集")
    ap.add_argument("--val-cls-ratio", type=float, default=0.15,
                    help="--data real 用：每类随机抽这个比例的格进评测，**最少 1 张**"
                         "（默认 0.15，保证每类都有评测样本）。同帧近重复会让指标偏高，"
                         "要干净的\"新物品\"口径就改用 --holdout-cls-ratio")
    ap.add_argument("--holdout-cls-ratio", type=float, default=0.0,
                    help="--data real 用：按比例**整类**留出 —— 该类的真实格只当查询、图鉴图也不作"
                         "负样本，测\"只给一张图鉴图的新物品 / 换皮肤\"（默认 0=关）")
    ap.add_argument("--aug", default="none", choices=["none", "light"],
                    help="none（默认）=原图直接喂，零增广；light=真实图上做平移/亮度/JPEG")
    ap.add_argument("--fuse-proto", action=argparse.BooleanOptionalAction, default=True,
                    help="原型是否融合\"该类训练帧真实格\"的 embedding 均值（默认融合）")
    # ---- 模型 ----
    ap.add_argument("--backbone", default=auto_classify.DEFAULT_BACKBONE,
                    choices=["scratch"] + sorted(_BACKBONES),
                    help=f"默认 {auto_classify.DEFAULT_BACKBONE}（与自动识别同骨干，读 auto_classify 常量）；"
                         f"scratch = 从零训小 CNN（老行为）")
    ap.add_argument("--img", type=int, default=0,
                    help="模型输入边长，0=自动（预训练骨干 224 / scratch 64）")
    ap.add_argument("--crop", type=int, default=0, help="内存里存的图边长，0=自动（96，与图鉴 tiles 同尺寸）")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="冻住骨干只训投影头（每类只有 1 张参考图时更稳，也是最快的对照）")
    ap.add_argument("--backbone-lr-scale", type=float, default=0.1,
                    help="微调时骨干的学习率 = --lr × 该系数（投影头用 --lr）")
    ap.add_argument("--embed-dim", type=int, default=0,
                    help="身份向量维度，0=自动（预训练骨干 256 / scratch 64）")
    ap.add_argument("--width", type=int, default=32, help="scratch 骨干的宽度")
    # ---- 训练 ----
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--classes-per-batch", type=int, default=64)
    ap.add_argument("--views", type=int, default=2, help="每个类抽几张真实格进 batch")
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--holdout-every", type=int, default=10, help="--data synth 用：每 N 类留出 1 类")
    ap.add_argument("--max-classes", type=int, default=0, help="--data synth 用：只用这么多类")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # 尺寸是模块级全局（to_tensor_batch 都在用），这里按骨干定下来
    global IMG, CROP
    if args.crop:
        CROP = args.crop
    pretrained = args.backbone != "scratch"
    IMG = args.img or (224 if pretrained else 64)
    if not args.embed_dim:
        args.embed_dim = 256 if pretrained else 64
    if not args.out:
        args.out = os.path.join(_ROOT, "out/clf" if args.data == "synth" else "out/clf_v2")

    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.data == "synth":
        run_synth(args, dev)
    else:
        run_real(args, dev)


def run_synth(args, dev: str) -> None:
    """旧的合成增广流程（图鉴图 + 底色重染造视图）。--data synth 才走这里。"""
    rng = np.random.default_rng(args.seed)
    icons = Icons(args.catalog, args.max_classes, args.seed)
    bgs = real_canvases(args.real, 400, args.seed)
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

    model = make_model(args.backbone, args.embed_dim, args.width).to(dev)
    # 分两组学习率：投影头（新初始化的）用 --lr，预训练骨干用 --lr×--backbone-lr-scale。
    # 骨干一上来就用大学习率会把 ImageNet 特征冲掉（每类只有 1 张参考图，救不回来）。
    head = [(n, p) for n, p in model.named_parameters() if not n.startswith("body.")]
    body = [(n, p) for n, p in model.named_parameters() if n.startswith("body.")]
    if args.freeze_backbone:
        for _, p in body:
            p.requires_grad = False
    groups = [{"params": [p for _, p in head], "lr": args.lr}]
    if body and not args.freeze_backbone:
        groups.append({"params": [p for _, p in body], "lr": args.lr * args.backbone_lr_scale})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"骨干 {args.backbone}（{'冻结' if args.freeze_backbone else '微调'}）"
          f"输入 {IMG}x{IMG}，身份向量 {args.embed_dim} 维，可训参数 {n_train / 1e6:.2f}M")

    def save_ckpt(path: str, ep: int) -> None:
        """权重 + 复现所需的全部信息（推理端要靠 backbone/img 才能重建同一个模型）。"""
        torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args),
                    "arch": "scratch" if args.backbone == "scratch" else "pretrained",
                    "backbone": args.backbone, "embed_dim": args.embed_dim,
                    "width": args.width, "img": IMG, "crop": CROP,
                    "class_ids": icons.class_ids, "entries": icons.entries}, path)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start = 0
    if args.resume:
        ck = load_weights(model, args.resume, dev, args)
        start = ck.get("epoch", 0)
        print(f"续训: 从 epoch {start} 恢复（{args.resume}）")

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
                ev = eval_holdout(model, icons, bank, hold, dev, rng, bs=args.eval_batch)
                msg += (f" | 留出类 top1 {ev['top1']:.3f} 同类均值 {ev['same_mean']:.3f} / "
                        f"异类均值 {ev['other_mean']:.3f} (间隔 {ev['margin']:.3f})")
                hist.append({"epoch": ep + 1, "loss": run / args.steps_per_epoch, **ev})
                with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
                    w.writeheader(); w.writerows(hist)
                save_ckpt(os.path.join(args.out, "last.pt"), ep + 1)
                if ev["top1"] >= max([h["top1"] for h in hist[:-1]] + [0.0]):
                    save_ckpt(os.path.join(args.out, "best.pt"), ep + 1)
                    msg += " → best.pt"
            print(msg, flush=True)

    ck_path = args.resume or os.path.join(
        args.out, "best.pt" if os.path.exists(os.path.join(args.out, "best.pt")) else "last.pt")
    ck = load_weights(model, ck_path, dev, args)
    print(f"\n=== 评测（权重 {ck_path}，epoch {ck.get('epoch')}，骨干 {ck.get('backbone', '?')}）")
    ev = eval_holdout(model, icons, bank, hold, dev, rng, bs=args.eval_batch)
    pr = eval_pairs(model, icons, gray_pairs, rng, dev)

    gal = embed(model, np.stack([render(icons.icon[i], icons.canvas_from_catalog(i), rng)
                                 for i in range(icons.n)]), dev, args.eval_batch)
    proto = (gal / (np.linalg.norm(gal, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    np.savez(os.path.join(args.out, "prototypes.npz"),
             proto=proto, class_ids=np.array(icons.class_ids),
             entries=np.array(icons.entries), faded=icons.faded,
             backbone=np.array(args.backbone), img=np.array(IMG), embed_dim=np.array(args.embed_dim))

    print(f"留出类({ev.get('n_query', 0)} 查询/{ev.get('n_class', 0)} 类): top1 {ev.get('top1', 0):.3f}，"
          f"同类均值 {ev.get('same_mean', 0):.3f}(p99 {ev.get('same_p99', 0):.3f}) / "
          f"异类均值 {ev.get('other_mean', 0):.3f}(最近异类 p1 {ev.get('other_min_p1', 0):.3f})，"
          f"间隔 {ev.get('margin', 0):.3f}")
    print(f"纯底板(无物品)到最近原型的距离: 均值 {ev.get('bg_mean', 0):.3f}，p5 {ev.get('bg_min_p5', 0):.3f}"
          f"（越大越安全：可用于 unknown 拒识）")
    if pr:
        print(f"同素材异渲染对: {pr['n']} 对，距离均值 {pr['mean']:.3f}，最大 {pr['max']:.3f}")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"config": {"backbone": args.backbone, "freeze_backbone": args.freeze_backbone,
                              "img": IMG, "crop": CROP, "embed_dim": args.embed_dim,
                              "epochs": args.epochs, "lr": args.lr,
                              "backbone_lr_scale": args.backbone_lr_scale, "seed": args.seed},
                   "holdout": ev, "dup_pairs": pr, "history": hist}, f, indent=2,
                  ensure_ascii=False)
    print(f"输出: {args.out}/best.pt(权重) / prototypes.npz({icons.n} 类原型) / metrics.json / history.csv")


if __name__ == "__main__":
    main()
