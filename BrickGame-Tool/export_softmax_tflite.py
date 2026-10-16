#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 train_legacy_softmax.py 训出的闭集 softmax 模型导成 tflite，用来和现役 embedder 做**同机速度对比**。

和 `export_mobile.py` 的 embedder 资产保持同一套接口，否则测出来的不是同一个东西：
  - 固定 batch（默认 32，与 Android `TfliteTileClassifier.BATCH` 一致）；
  - 输入 NHWC、RGB、值域 [0,1]，ImageNet 归一化烘进计算图（调用方不再归一化）；
  - 输出 [batch, n_class] float32 logits（现役 embedder 输出的是 [batch, n_class] 余弦，形状一致）。
大小对齐：输入边长默认 160，与 androidApp 里三档 embedder 资产一致。

导出后用 LiteRT 解释器与 PyTorch 对拍，不一致就报错 —— 不拿没验证过的模型去测速。

用法:
  python export_softmax_tflite.py --ckpt out/stress/L2/softmax.pt --out out/bench/softmax_L2.tflite
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from train_legacy_softmax import SoftmaxNet  # noqa: E402

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Wrapped(nn.Module):
    """NHWC RGB[0,1] 入 → 图内 permute 到 NCHW + ImageNet 归一化 → logits 出。

    与 `export_mobile.Wrapped` 同一个套路：归一化烘进图，Android 侧只喂 RGB [0,1]。
    接受的布局是 NHWC，和 embedder 资产一致（XNNPACK / GPU delegate 的原生布局）。
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)
        return self.net((x - self.mean) / self.std)


def main() -> None:
    ap = argparse.ArgumentParser(description="闭集 softmax 模型 → tflite")
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "out/stress/L2/softmax.pt"))
    ap.add_argument("--out", default=os.path.join(_HERE, "out/bench/softmax_L2.tflite"))
    ap.add_argument("--img", type=int, default=160, help="输入边长（对齐 embedder 资产的 160）")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = SoftmaxNet(ck["backbone"], int(ck["n_class"]), int(ck["hidden"]))
    net.load_state_dict(ck["model"])
    net.eval()
    w = Wrapped(net).eval()
    print(f"载入 {args.ckpt}：骨干 {ck['backbone']}，隐藏层 {ck['hidden']}，"
          f"{ck['n_class']} 路 softmax，训练输入 {ck['img']}")

    example = torch.rand(args.batch, args.img, args.img, 3)
    import litert_torch
    edge = litert_torch.convert(w, (example,))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    edge.export(args.out)
    print(f"写出 {args.out}（{os.path.getsize(args.out) / 1e6:.1f} MB）")

    # ---- 对拍：LiteRT 解释器 vs PyTorch ----
    from ai_edge_litert.interpreter import Interpreter
    with torch.no_grad():
        ref = w(example).numpy()
    interp = Interpreter(model_path=args.out, num_threads=4)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    print(f"TFLite 输入 {list(inp['shape'])} {inp['dtype'].__name__} / 输出 {list(out['shape'])}")
    # 输入是 NHWC 就原样喂；万一被搬成 NCHW 就转置过去
    feed = example.numpy()
    if list(inp["shape"])[1:3] != [args.img, args.img]:
        feed = np.transpose(feed, (0, 3, 1, 2))
    interp.set_tensor(inp["index"], np.ascontiguousarray(feed, dtype=np.float32))
    interp.invoke()
    got = interp.get_tensor(out["index"]).reshape(ref.shape)
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-9)
    print(f"对拍：余弦 min {cos.min():.6f} / mean {cos.mean():.6f}，最大绝对差 {np.abs(ref - got).max():.3e}")
    if cos.min() < 0.999:
        raise SystemExit("TFLite 与 PyTorch 输出差太多，先别拿去测速")


if __name__ == "__main__":
    main()
