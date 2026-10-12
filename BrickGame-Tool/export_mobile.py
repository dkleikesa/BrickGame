#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把训练好的身份 embedding 模型导出成 Android 能直接加载的那三个文件。

产出（默认写进 androidApp/src/main/assets/）：
  brick_embedder.onnx  ONNX 模型。输入 RGB NCHW float32，值域 [0,1]（ImageNet 归一化已烘进图里，
                       调用方不要再做归一化）；输出 L2 归一化的 embed_dim 维身份向量。
  prototypes.bin       (n_class, embed_dim) float32，行优先小端。**第 i 行对应 model_meta.json
                       里 class_ids[i]** —— 与输出向量做点积即是相似度（已归一化，点积=余弦）。
  model_meta.json      接口契约 + 类目表 + 阈值 + 训练配置/指标。Android 端加载原型做最近邻时读它。

为什么走 ONNX 而不是 TorchScript：PyTorch Android 的 libtorch 版本远落后于训练用的 torch 2.8，
跨版本加载 TorchScript 很容易直接失败；onnxruntime 只要和训练侧版本对齐就能稳加载
（本仓库用 onnxruntime-android:1.19.2，与校验用的 python onnxruntime 1.19.2 同版）。
`--torchscript` 可以额外导一份 .ptl 备用。

导出后会**用 onnxruntime 现场加载并和 PyTorch 对拍**，数值不一致就报错退出 —— 不把没验证过的
模型塞进 APK。

用法：
  .venv/bin/python export_mobile.py                      # 用 out/clf_real/best.pt
  .venv/bin/python export_mobile.py --ckpt out/clf/best.pt --out /tmp/assets
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                      # 项目根（androidApp 在那边）
sys.path.insert(0, _HERE)
import train_classifier as TC                       # noqa: E402 复用模型定义，避免两份实现漂移

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Wrapped(nn.Module):
    """把 ImageNet 归一化烘进计算图：Android 端只喂 RGB [0,1]，不用知道 mean/std。"""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e, _ = self.m((x - self.mean) / self.std)
        return e                                    # 已是 L2 归一化


def main() -> None:
    ap = argparse.ArgumentParser(description="训练权重 → Android 可加载的模型包")
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "out/clf_real/best.pt"))
    ap.add_argument("--prototypes", default="", help="默认取 ckpt 同目录的 prototypes.npz")
    ap.add_argument("--metrics", default="", help="默认取 ckpt 同目录的 metrics.json")
    ap.add_argument("--out", default=os.path.join(_ROOT, "androidApp", "src", "main", "assets"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--torchscript", action="store_true", help="额外导一份 .ptl（备用，APK 里用不到）")
    args = ap.parse_args()

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    proto_src = args.prototypes or os.path.join(ckpt_dir, "prototypes.npz")
    metrics_src = args.metrics or os.path.join(ckpt_dir, "metrics.json")
    os.makedirs(args.out, exist_ok=True)

    # ---- 1) 重建模型并载入权重 ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone, img, dim = ck["backbone"], int(ck["img"]), int(ck["embed_dim"])
    model = TC.make_model(backbone, dim, int(ck.get("width", 32)))
    model.load_state_dict(ck["model"])
    net = Wrapped(model).eval()
    dummy = torch.rand(1, 3, img, img)
    with torch.no_grad():
        ref = net(dummy).numpy()
    print(f"载入 {args.ckpt}：骨干 {backbone}，输入 {img}x{img}，身份向量 {dim} 维"
          f"（epoch {ck.get('epoch')}，数据 {ck.get('data', '?')}）")

    # ---- 2) 导出 ONNX ----
    onnx_path = os.path.join(args.out, "brick_embedder.onnx")
    torch.onnx.export(
        net, dummy, onnx_path, input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    # ---- 3) 现场加载 + 与 PyTorch 对拍（证明"可加载"且数值一致）----
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    print(f"ONNX 校验: 输入 {sess.get_inputs()[0].shape} → 输出 {got.shape}，"
          f"与 PyTorch 最大差 {diff:.2e}，文件 {os.path.getsize(onnx_path) / 1e6:.1f} MB")
    if diff > 1e-4:
        raise SystemExit(f"ONNX 与 PyTorch 数值不一致（{diff:.2e}），不要接 Android")

    if args.torchscript:
        ts_path = os.path.join(args.out, "brick_embedder.ptl")
        torch.jit.trace(net, dummy).save(ts_path)
        print(f"TorchScript 备用: {ts_path}（{os.path.getsize(ts_path) / 1e6:.1f} MB）")

    # ---- 4) 原型库 + 契约 ----
    z = np.load(proto_src)
    proto = z["proto"].astype(np.float32)
    class_ids = [int(c) for c in z["class_ids"]]
    entries = [str(e) for e in z["entries"]]
    if proto.shape[1] != dim:
        raise SystemExit(f"原型维度 {proto.shape[1]} 与模型输出 {dim} 不符")
    proto_path = os.path.join(args.out, "prototypes.bin")
    proto.tofile(proto_path)                        # float32 行优先小端（macOS/Android 都是 LE）
    print(f"原型: {proto.shape}（{proto.shape[0]} 类）→ {os.path.getsize(proto_path) / 1e6:.1f} MB")

    th = {}
    if os.path.exists(metrics_src):
        with open(metrics_src) as f:
            m = json.load(f)
        r = m.get("real") or {}
        th = {"dist_hit_p95": round(float(r.get("hit_dist_p95", 0)), 4),
              "dist_wrong_p5": round(float(r.get("wrong_dist_p5", 0)), 4),
              "top1": round(float(r.get("top1", 0)), 4),
              "top1_macro": round(float(r.get("top1_macro", 0)), 4),
              "n_query": int(r.get("n_query", 0))}
    meta = {
        "name": "brick_embedder",
        "exported_from": os.path.relpath(args.ckpt, _ROOT),
        "epoch": int(ck.get("epoch", 0)),
        "input": {"name": "input", "layout": "NCHW", "dtype": "float32",
                  "size": [1, 3, img, img], "color": "RGB", "range": [0.0, 1.0],
                  "note": "ImageNet 归一化已烘进模型，调用方不要再归一化；batch 维为动态"},
        "output": {"name": "embedding", "dtype": "float32", "size": [1, dim],
                   "note": "已 L2 归一化；与 prototypes.bin 各行做点积即为余弦相似度"},
        "prototypes": {"file": "prototypes.bin", "shape": [len(class_ids), dim],
                       "dtype": "float32", "order": "row-major",
                       "note": "第 i 行对应 class_ids[i]"},
        "class_ids": class_ids,
        "entries": entries,
        "thresholds": {**th, "distance_def": "余弦距离 = 1 - 点积",
                       "hint": "距离 > dist_hit_p95 → 低置信；> dist_wrong_p5 → 基本可判为未知物品"},
        "train": {"ckpt": os.path.basename(args.ckpt), "backbone": backbone,
                  "embed_dim": dim, "img": img, "class_ids_count": len(class_ids)},
    }
    meta_path = os.path.join(args.out, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"契约: {meta_path}（{os.path.getsize(meta_path) / 1e3:.1f} KB，"
          f"{len(class_ids)} 类；阈值 hit_p95 {th.get('dist_hit_p95')} / wrong_p5 {th.get('dist_wrong_p5')}）")
    print(f"\n全部写入 {args.out}")


if __name__ == "__main__":
    main()
