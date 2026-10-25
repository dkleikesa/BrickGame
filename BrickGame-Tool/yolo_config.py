#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 定位链路的唯一尺寸/阈值契约（改这里 = 改整条链）。

这条链上的脚本过去各写一份 640，改分辨率要满仓库搜常量，漏一处就出现
"训练 1024 / 评估 640 / 装配图还是 640"的静默错位。现在只认这一个文件：

  装配      build-dataset/make_yolo_data.py   MAX_SIDE（存图长边）
  预标注    build-dataset/prelabel_yolo.py    IMG_SIZE / CONF / NMS_IOU
  训练导出  train_yolo.py                     IMG_SIZE
  评估      build-dataset/eval_gate.py        IMG_SIZE / CONF / NMS_IOU
            build-dataset/eval_yolo.py
            build-dataset/audit_labels.py
            build-dataset/yolo_grid.py

Android 端**不** import 这个文件：`YoloBoardDetector` 在构造时从 .tflite 的张量
形状读输入高宽和 anchor 数。所以换尺寸 = 改这里 → 重装配 → 重训 → 重导 tflite →
cp 进 assets，Kotlin 一行都不用动。端侧有长边 1024 下限，误装 640 会直接报错。

尺寸链（以本机 1080x2400 截图为例）：

  手机原图 1080x2400（samples/ 不压，唯一原图来源）
    → yolo/images 长边 MAX_SIDE = 1024      461x1024
    → letterbox 到 1024x480（高x宽）       461x1024 内容 + 左右各 ~9px 灰边
    → 模型输入 / TFLite 契约                [1,3,1024,480]
    → 检测头输出                            [1,6,10080]（128x60 + 64x30 + 32x15）

为什么是 1024 高而不是继续 640：截图里一格砖在 640 下只剩 ~20px 边长，
小砖/半块砖/被悬浮窗压住的边缘在特征图上不到一个 stride-8 单元，
模型根本"看不见"，补多少标注都没用。高度放到 1024 后一格约 32px，
P3 特征图上从 2~3 个像素变成 4~5 个，才是能学出来的尺度。

为什么宽只留 480 而不是方图：截图是竖屏 1080x2400，内容按 1024 高缩放后只有
461 宽。做成 1024x1024 方图时 55% 的像素是灰边，算力翻倍、信息不变；裁到 480
宽正好包住内容，像素量回到 640 方图的水平（0.49M vs 0.41M），推理/训练都省一半。
注意 `IMG_W` 要能盖住最宽的一张截图（当前 1080/2400*1024≈461，留 480）；
换平板/横屏截图时要重新评估。
"""

from __future__ import annotations

# ---- 尺寸（高 x 宽，ultralytics 的 imgsz 顺序也是 (h, w)）----
#: 模型输入高度。同时是装配存图的长边：内容高度 1:1，不放大也不浪费像素。
IMG_H = 1024
#: 模型输入宽度。只需覆盖竖屏截图在 1024 高时的内容宽度（当前 461）。
IMG_W = 480
#: ultralytics 的 imgsz 写法。
IMG_SIZE = (IMG_H, IMG_W)
#: 装配时帧图保留的长边，超过就按比例缩；旧图（尺寸不符）会被
#: make_yolo_data.py 从 samples/ 自动重做。
MAX_SIDE = IMG_H

if IMG_H % 32 or IMG_W % 32:
    # detect 头 stride 8/16/32，输入不是 32 的倍数时特征图尺寸会被 floor，
    # anchors_for() 和 TFLite 实际输出就会对不上（导出口径直接错）。
    raise ValueError(
        f"IMG_H/IMG_W 必须是 32 的倍数（检测头 stride 8/16/32），当前 {IMG_H}x{IMG_W}"
    )

# ---- 推理口径（Python 侧与 Android 端必须完全一致）----
#: 置信度阈值。训练不看它，只有预标注/评估/端侧解码用。
CONF = 0.25
#: NMS IoU。ultralytics 默认 0.7 会给同一块砖留两个重叠框，端侧是 0.45。
NMS_IOU = 0.45
#: 训练时 tile 框每边内缩 6% → 框宽 = 0.88 格；端侧定格距时要除回去。
TILE_INSET = 0.88
#: letterbox 灰边值（Ultralytics 默认 114）。
PAD = 114


def anchors_for(img_h: int = IMG_H, img_w: int = IMG_W) -> int:
    """detect 头展平后的 anchor 数：stride 8/16/32 三层特征图之和。

    1024x480 → 10080，1024x1024 → 21504，640x640 → 8400。
    导出后用它核对 TFLite 输出形状。
    """
    return sum((img_h // stride) * (img_w // stride) for stride in (8, 16, 32))


def predict_kwargs(conf: float = CONF) -> dict:
    """Python 侧推理的统一参数（必须与 TFLite / Android 同口径）。

    `rect=False` 不能省：ultralytics 8.4 的 `Model.predict()` 内部把 rect 默认成
    True（见 engine/model.py 的 custom dict），会按图片长宽比做矩形 letterbox；
    我们导出的是**固定** 1024x480，推理必须用同一个形状，否则预标注、评估就和
    线上不是同一口径。640 时代因为 288+352 恰好凑成 640 方图，这个问题一直没暴露。

    `max_det=1000`：ultralytics 默认 300，Android 端不设上限。未完全收敛的模型
    低分框多，300 可能在 NMS 后把真框挤掉。
    """
    return {
        "imgsz": IMG_SIZE,   # (h, w) = (1024, 480)
        "conf": conf,
        "iou": NMS_IOU,
        "rect": False,       # 固定 1024x480，与 TFLite/Android 一致
        "max_det": 1000,
        "verbose": False,
    }
