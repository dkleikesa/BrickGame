#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可选识别模型 → APK 资产文件名后缀。

这是**跨语言、跨虚拟环境的唯一约定处**，改这里必须同时改 Android 的
`androidApp/.../ml/ModelVariant.kt`（那边按 `brick_embedder$tag.tflite` /
`prototypes$tag.bin` / `model_meta$tag.json` 三个名字取文件）。

为什么单独一个文件：`export_mobile.py`（torch 环境）和 `export_tflite.py`
（onnx2tf 环境）用的是两个不同的 venv，互相 import 会连带拉进 torch/onnx2tf；
这个文件只放一个字典，两边都能安全 import。

large 的后缀是空串 —— 线上那份资产的名字早已定下（`brick_embedder.tflite` 等），
加第二个模型时不动它，只新增 small 那一套。
"""
from __future__ import annotations

VARIANTS = {"large": "", "small": "_small", "v2": "_v2"}


def asset_names(variant: str) -> tuple[str, str, str]:
    """@return (模型 graph, 原型库, 契约) 三个资产的文件名。"""
    tag = VARIANTS[variant]
    return f"brick_embedder{tag}.tflite", f"prototypes{tag}.bin", f"model_meta{tag}.json"
