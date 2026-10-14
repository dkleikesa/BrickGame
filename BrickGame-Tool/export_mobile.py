#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把训练好的身份 embedding 模型导出成 Android 能直接加载的那三个文件。

产出（默认写进 androidApp/src/main/assets/）：
  brick_embedder.onnx  ONNX 模型。输入 RGB NCHW float32，值域 [0,1]（ImageNet 归一化已烘进图里，
                       调用方不要再做归一化）；输出 L2 归一化的 embed_dim 维身份向量。
  prototypes.bin       (n_class, embed_dim) float32，行优先小端。**第 i 行对应 model_meta.json
                       里 class_ids[i]** —— 与输出向量做点积即是相似度（已归一化，点积=余弦）。
  model_meta.json      接口契约 + 类目表 + 阈值 + 训练配置/指标。Android 端加载原型做最近邻时读它。

为什么走 ONNX 而不是 TorchScript：训练侧 torch 迭代快，跨版本加载 TorchScript 很容易直接失败；
ONNX 有版本对齐的运行时（onnxruntime）兜底，而且 `export_tflite.py` 拿它当 ONNX → TFLite 的转换源。
它只是中间格式：Android 侧只加载 `brick_embedder.tflite`。
`--torchscript` 可以额外导一份 .ptl 备用。

**batch 固定**（`--batch`，默认 32，必须与 Android `TfliteTileClassifier.BATCH` 一致）：
留成动态维时，`F.normalize` 的广播会被导成 `Shape(input) → Concat → Expand`，onnx2tf 搬进 TFLite
后那个 `FILL` 的输出是"运行时才定形状"的动态张量，GPU delegate 拒收整图
（真机报 `Error applying delegate`）。固定 batch 后形状在导出期就是常量，这条链直接消失。

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
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                      # 项目根（androidApp 在那边）
sys.path.insert(0, _HERE)
import train_classifier as TC                       # noqa: E402 复用模型定义，避免两份实现漂移
from variants import VARIANTS                       # noqa: E402 变体 → 资产文件名

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 可选模型 → 资产文件名里的后缀。映射本身在 variants.py（与 export_tflite.py 共用，
# 那边跑在另一个 venv 里，import 本文件会连带拉进 torch）。


class Wrapped(nn.Module):
    """把 ImageNet 归一化烘进计算图：Android 端只喂 RGB [0,1]，不用知道 mean/std。

    再可选地把**原型检索**也并进来（`proto` 非空时）：末尾加一次 `e @ Pᵀ`，
    输出从 256 维身份向量变成 2369 维余弦相似度。

    为什么要把检索搬进图里：那一步是 136 格 × 2369 类 × 256 维 = **8250 万次乘加**，
    在真机上是纯 Kotlin 单线程标量循环，实测 **1728ms**（占整帧 62%，比模型推理还贵 6 倍）。
    而它本质就是一次矩阵乘 —— 交给 GPU delegate 就那么几十毫秒的事，
    顺带把 Android 侧那两层循环和"扫描 2.4MB 原型矩阵"一起消掉。
    原型矩阵本来就要以 prototypes.bin 的形式进 APK（2.4MB），融进模型后**净增 0**。
    """

    def __init__(self, m: nn.Module, proto: "np.ndarray | None" = None):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        if proto is None:
            self.proto = None
        else:
            # 传进来的原型已经 L2 归一化过（与 embedding 同域），所以点积即余弦
            self.register_buffer("proto", torch.from_numpy(np.ascontiguousarray(proto)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e, _ = self.m((x - self.mean) / self.std)   # 已是 L2 归一化
        if self.proto is None:
            return e
        return e @ self.proto.t()                   # (N, dim) @ (dim, nClass) → (N, nClass)


def main() -> None:
    ap = argparse.ArgumentParser(description="训练权重 → Android 可加载的模型包")
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "out/clf_real/best.pt"))
    ap.add_argument("--prototypes", default="", help="默认取 ckpt 同目录的 prototypes.npz")
    ap.add_argument("--metrics", default="", help="默认取 ckpt 同目录的 metrics.json")
    ap.add_argument("--out", default=os.path.join(_ROOT, "androidApp", "src", "main", "assets"))
    ap.add_argument("--variant", default="large", choices=sorted(VARIANTS),
                    help="导出哪一个模型（决定文件名的后缀，对应 Android 的选择开关）")
    ap.add_argument("--install-assets", default="",
                    help="把原型库与契约同时装到该 assets 目录（.onnx 留在 --out，不进 APK）")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=32,
                    help="固定 batch，必须与 Android 的 BATCH 一致（动态 batch 会带出动态张量，GPU delegate 挂不上）")
    ap.add_argument("--fuse-search", action=argparse.BooleanOptionalAction, default=True,
                    help="把原型检索并进模型尾部（输出 2369 维相似度），Android 侧就不用再跑那两层循环；"
                         "关掉则输出 256 维身份向量 + 单独的 prototypes.bin（老契约）")
    ap.add_argument("--torchscript", action="store_true", help="额外导一份 .ptl（备用，APK 里用不到）")
    args = ap.parse_args()

    tag = VARIANTS[args.variant]
    stem = f"brick_embedder{tag}"
    proto_name = f"prototypes{tag}.bin"
    meta_name = f"model_meta{tag}.json"

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    proto_src = args.prototypes or os.path.join(ckpt_dir, "prototypes.npz")
    metrics_src = args.metrics or os.path.join(ckpt_dir, "metrics.json")
    os.makedirs(args.out, exist_ok=True)

    # ---- 0) 原型库（融合检索时要先拿到，作为模型的常量权重）----
    z = np.load(proto_src)
    proto = z["proto"].astype(np.float32)
    class_ids = [int(c) for c in z["class_ids"]]
    entries = [str(e) for e in z["entries"]]

    # ---- 1) 重建模型并载入权重 ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone, img, dim = ck["backbone"], int(ck["img"]), int(ck["embed_dim"])
    model = TC.make_model(backbone, dim, int(ck.get("width", 32)))
    model.load_state_dict(ck["model"])
    if args.fuse_search and proto.shape[1] != dim:
        raise SystemExit(f"原型维度 {proto.shape[1]} 与模型输出 {dim} 不符，不能融合检索")
    net = Wrapped(model, proto if args.fuse_search else None).eval()
    dummy = torch.rand(args.batch, 3, img, img)
    with torch.no_grad():
        ref = net(dummy).numpy()
    print(f"载入 {args.ckpt}：骨干 {backbone}，输入 {img}x{img}，身份向量 {dim} 维"
          f"（epoch {ck.get('epoch')}，数据 {ck.get('data', '?')}）；"
          f"检索{'已并进模型 → 输出 ' + str(len(class_ids)) + ' 维相似度' if args.fuse_search else '留在 Android 侧'}")

    # ---- 2) 导出 ONNX（形状全固定：不加 dynamic_axes，见文件头的 batch 说明）----
    onnx_path = os.path.join(args.out, f"{stem}.onnx")
    torch.onnx.export(
        net, dummy, onnx_path, input_names=["input"], output_names=["embedding"],
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
        ts_path = os.path.join(args.out, f"{stem}.ptl")
        torch.jit.trace(net, dummy).save(ts_path)
        print(f"TorchScript 备用: {ts_path}（{os.path.getsize(ts_path) / 1e6:.1f} MB）")

    # ---- 4) 契约（原型库要不要单独出一份，取决于检索是否已经并进模型）----
    proto_path = ""
    if args.fuse_search:
        # 融合后原型是模型的常量权重，Android 侧不再需要单独一份 —— 省 2.4MB 和两层循环
        print(f"原型: {proto.shape}（{proto.shape[0]} 类）已作为权重并入模型，不再单独导出")
    else:
        proto_path = os.path.join(args.out, proto_name)
        proto.tofile(proto_path)                    # float32 行优先小端（macOS/Android 都是 LE）
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
    out_dim = len(class_ids) if args.fuse_search else dim
    meta = {
        "name": f"brick_embedder{tag}",
        "variant": args.variant,
        "exported_from": os.path.relpath(args.ckpt, _ROOT),
        "epoch": int(ck.get("epoch", 0)),
        "input": {"name": "input", "layout": "NCHW", "dtype": "float32",
                  "size": [args.batch, 3, img, img], "color": "RGB", "range": [0.0, 1.0],
                  "note": "ImageNet 归一化已烘进模型，调用方不要再归一化；batch 固定，"
                          "必须与 Android 的 BATCH 一致（动态 batch 会带出动态张量，GPU delegate 挂不上）"},
        "output": {"name": "scores" if args.fuse_search else "embedding",
                   "dtype": "float32", "size": [args.batch, out_dim],
                   "note": ("检索已并进模型：每行是 2369 类的余弦相似度，直接取 argmax 即可（×100 即百分制）；"
                            "顺序与 class_ids 一致" if args.fuse_search
                            else f"已 L2 归一化；与 {proto_name} 各行做点积即为余弦相似度")},
        "search_fused": bool(args.fuse_search),
        "class_ids": class_ids,
        "entries": entries,
        "thresholds": {**th, "distance_def": "余弦距离 = 1 - 点积",
                       "hint": "距离 > dist_hit_p95 → 低置信；> dist_wrong_p5 → 基本可判为未知物品"},
        "train": {"ckpt": os.path.basename(args.ckpt), "backbone": backbone,
                  "embed_dim": dim, "img": img, "class_ids_count": len(class_ids)},
    }
    if not args.fuse_search:
        meta["prototypes"] = {"file": proto_name, "shape": [len(class_ids), dim],
                              "dtype": "float32", "order": "row-major",
                              "note": "第 i 行对应 class_ids[i]"}
    meta_path = os.path.join(args.out, meta_name)
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"契约: {meta_path}（{os.path.getsize(meta_path) / 1e3:.1f} KB，"
          f"{len(class_ids)} 类；阈值 hit_p95 {th.get('dist_hit_p95')} / wrong_p5 {th.get('dist_wrong_p5')}）")

    # ---- 5) 装进 APK：只搬"分类头"要用的那两份（.onnx 只是转换源，不进包）----
    if args.install_assets:
        os.makedirs(args.install_assets, exist_ok=True)
        for src in (proto_path, meta_path):
            if src:
                shutil.copyfile(src, os.path.join(args.install_assets, os.path.basename(src)))
        print(f"已装进 assets: {args.install_assets}/"
              f"{{{meta_name}{', ' + proto_name if proto_path else ''}}}"
              f"（{args.variant} 的 .tflite 由 export_tflite.py 补）")
    print(f"\n全部写入 {args.out}")


if __name__ == "__main__":
    main()
