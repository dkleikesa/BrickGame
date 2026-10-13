#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 shared 侧纯数学实现的对照基准（`shared/src/commonTest/.../VisionGolden.kt`）。

为什么要这么一份基准：`shared` 的 `LatticeMath` 是把 `tile_recognizer.py` 的数值逻辑
照搬到 Kotlin 的，光"能编译"不能说明没搬错。所以把 Python 的真实输出固化成常量，
让 JVM 单测逐值比对。

只覆盖投影这一组：真实截图 `build-dataset/samples/screenshot_01.jpg` 的行/列投影
→ 周期、相位、精调后的网格线。

（图标特征 / 对齐距离 / 聚类那几组基准随像素聚类识别一起删了：识别现在已经全走模型。
`tile_recognizer.py` 里对应函数留着当参考实现，哪天真要用回来，从这里加回生成逻辑即可。）

用法（用项目自带的 venv，它装了 numpy + opencv）：
  .venv/bin/python gen_vision_golden.py
改完 `tile_recognizer.py` 的数值逻辑后重跑本脚本，再跑 `:shared:testAndroidHostTest`。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import tile_recognizer as T  # noqa: E402

DEFAULT_IMAGE = os.path.join(HERE, "build-dataset", "samples", "screenshot_01.jpg")
DEFAULT_OUT = os.path.join(
    ROOT, "shared", "src", "commonTest", "kotlin", "com", "qcmian", "brickgame", "vision",
    "VisionGolden.kt",
)
MIN_PITCH = 12


def fmt(arr, nd: int = 2) -> str:
    return ",".join(("%g" % round(float(v), nd)) for v in np.asarray(arr).ravel())


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 VisionGolden.kt")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="用于投影基准的截图")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 Kotlin 文件")
    args = ap.parse_args()

    # 真实截图的投影 → 网格
    img = T.load_image(args.image)
    g = T.edge_profile(img)
    row_prof = g.sum(axis=1)
    col_prof = g.sum(axis=0)
    h, w = g.shape
    pitch_y, _ = T.fundamental_pitch(row_prof, MIN_PITCH, int(h * 0.25))
    pitch_x, _ = T.fundamental_pitch(col_prof, MIN_PITCH, int(w * 0.25))
    phase_y = T.lattice_phase(row_prof, pitch_y)
    phase_x = T.lattice_phase(col_prof, pitch_x)
    ylines = T.refine_lines(row_prof, np.arange(phase_y, h, pitch_y), pitch_y)
    xlines = T.refine_lines(col_prof, np.arange(phase_x, w, pitch_x), pitch_x)

    lines = [
        "package com.qcmian.brickgame.vision",
        "",
        "// 本文件由 BrickGame-Tool/gen_vision_golden.py 生成，请勿手改。",
        "// 用途：把 Kotlin 的重写钉在 tile_recognizer.py 的数值结果上（改了 Python 逻辑就重跑生成器）。",
        f"// 数据来源：{os.path.relpath(args.image, ROOT)} 的行/列投影。",
        "internal object VisionGolden {",
        f'    const val IMAGE = "{os.path.basename(args.image)}"',
        f"    const val MIN_PITCH = {MIN_PITCH}",
        f"    const val PITCH_X = {pitch_x}",
        f"    const val PITCH_Y = {pitch_y}",
        f"    const val PHASE_X = {phase_x}",
        f"    const val PHASE_Y = {phase_y}",
        f'    const val ROW_PROF = "{fmt(row_prof)}"',
        f'    const val COL_PROF = "{fmt(col_prof)}"',
        f'    const val YLINES = "{fmt(ylines, 4)}"',
        f'    const val XLINES = "{fmt(xlines, 4)}"',
        "}",
        "",
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"图片 {w}x{h}：pitch=({pitch_x},{pitch_y}) phase=({phase_x},{phase_y}) "
          f"网格 {len(xlines) - 1}列×{len(ylines) - 1}行")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
