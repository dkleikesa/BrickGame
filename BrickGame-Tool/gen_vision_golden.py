#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 shared 侧纯数学实现的对照基准（`shared/src/commonTest/.../VisionGolden.kt`）。

为什么要这么一份基准：`shared` 的 `LatticeMath` / `FeatureMath` 是把 `tile_recognizer.py`
的数值逻辑照搬到 Kotlin 的，光"能编译"不能说明没搬错（实测栽过一次：
OpenCV 的 `TM_SQDIFF_NORMED` 会把结果截断在 1.0，照公式写就会偏）。所以把 Python 的
真实输出固化成常量，让 JVM 单测逐值比对。

三组基准：
  1. 真实截图 `build-dataset/samples/screenshot_01.jpg` 的行/列投影 → 周期、相位、精调后的网格线；
  2. 公式化合成 patch（Kotlin 侧用同一公式重算像素）→ `icon_features` 的 ratio/dev；
  3. 显式构造的距离矩阵 + 合成 dev → `aligned_distance_matrix` / `gap_threshold` /
     `cluster_by_parity` / `merge_similar`。

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
    ROOT, "shared", "src", "commonTest", "kotlin", "com", "example", "brickgame", "vision",
    "VisionGolden.kt",
)
CELL = 32
MIN_PITCH = 12


def fmt(arr, nd: int = 2) -> str:
    return ",".join(("%g" % round(float(v), nd)) for v in np.asarray(arr).ravel())


def synth_patch(dy: int, dx: int, checker: bool = False) -> np.ndarray:
    """背景 200；`checker=False` 时是中心 12×12 的方块，整体平移 (dy, dx)。

    Kotlin 侧 `LatticeMathTest.synthPatch` 必须逐字重写这个公式。
    """
    yy, xx = np.mgrid[0:CELL, 0:CELL]
    p = np.full((CELL, CELL, 3), 200.0, np.float32)
    if checker:
        p[((yy // 4 + xx // 4) % 2 == 0)] = (10, 200, 200)
        return p
    sy, sx = yy - dy, xx - dx
    p[(sy >= 10) & (sy < 22) & (sx >= 10) & (sx < 22)] = (20, 30, 40)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 VisionGolden.kt")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="用于投影基准的截图")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 Kotlin 文件")
    args = ap.parse_args()

    # ---- 1) 真实截图的投影 → 网格 ----
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

    # ---- 2) 合成 patch → 图标特征 ----
    icon_patches = np.stack([synth_patch(0, 0), synth_patch(3, 0), synth_patch(0, 0, True)])
    dev_icon, ratio = T.icon_features(icon_patches.astype(np.float32))
    dev_sum = dev_icon.reshape(3, -1).sum(axis=1)
    dev_nnz = (dev_icon.reshape(3, -1) != 0).sum(axis=1)

    # ---- 3) 合成 dev → 对齐距离 ----
    # 平移到 ±rad 内 / 超出 ±rad 各来一组，两条分支都覆盖到
    align_patches = np.stack([synth_patch(0, 0), synth_patch(3, 0), synth_patch(0, 7),
                              synth_patch(0, 0, True)])
    dev_align, _ = T.icon_features(align_patches.astype(np.float32))
    dist_align = T.aligned_distance_matrix(dev_align, rad=3, prefilter=1.0)

    # ---- 4) 显式距离矩阵 → 聚类 ----
    # 4 个 A + 2 个 B：A 内部会被贪心拆成 2+2，靠 merge_similar 并回去
    classes = [0, 0, 0, 0, 1, 1]
    pairs = {(0, 1): 0.04, (2, 3): 0.05, (0, 2): 0.18, (0, 3): 0.19, (1, 2): 0.17,
             (1, 3): 0.18, (4, 5): 0.03}
    n = len(classes)
    matrix = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = pairs.get((i, j), 0.04 if classes[i] == classes[j] else 0.90)
            matrix[i, j] = matrix[j, i] = d
    gap = T.gap_threshold(matrix)
    labels, thr, ok = T.cluster_by_parity(matrix, gap)
    merged = T.merge_similar(labels, matrix, gap)

    lines = [
        "package com.qcmian.brickgame.vision",
        "",
        "// 本文件由 BrickGame-Tool/gen_vision_golden.py 生成，请勿手改。",
        "// 用途：把 Kotlin 的重写钉在 tile_recognizer.py 的数值结果上（改了 Python 逻辑就重跑生成器）。",
        f"// 数据来源：{os.path.relpath(args.image, ROOT)} 的投影 + 公式化合成 patch。",
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
        "",
        f"    const val CELL = {CELL}",
        "    val ICON_RATIO = doubleArrayOf(%s)" % ", ".join("%.8f" % v for v in ratio),
        "    val ICON_DEV_SUM = doubleArrayOf(%s)" % ", ".join("%.4f" % v for v in dev_sum),
        "    val ICON_DEV_NNZ = intArrayOf(%s)" % ", ".join(str(int(v)) for v in dev_nnz),
        "",
        "    val ALIGNED = arrayOf(",
    ]
    for i in range(dist_align.shape[0]):
        lines.append("        doubleArrayOf(%s)," % ", ".join("%.9f" % v for v in dist_align[i]))
    lines += [
        "    )",
        "",
        "    val CLUSTER_CLASSES = intArrayOf(%s)" % ", ".join(str(c) for c in classes),
        "    val CLUSTER_DIST = arrayOf(",
    ]
    for i in range(matrix.shape[0]):
        lines.append("        doubleArrayOf(%s)," % ", ".join("%.9f" % v for v in matrix[i]))
    lines += [
        "    )",
        "    const val CLUSTER_GAP = %.9f" % gap,
        "    const val CLUSTER_THR = %.9f" % thr,
        "    const val CLUSTER_OK = %s" % ("true" if ok else "false"),
        "    val CLUSTER_LABELS = intArrayOf(%s)" % ", ".join(str(int(v)) for v in labels),
        "    val MERGED_LABELS = intArrayOf(%s)" % ", ".join(str(int(v)) for v in merged),
        "}",
        "",
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"图片 {w}x{h}：pitch=({pitch_x},{pitch_y}) phase=({phase_x},{phase_y}) "
          f"网格 {len(xlines) - 1}列×{len(ylines) - 1}行")
    print(f"聚类：gap={gap:.3f} thr={thr:.3f} 偶数约束={ok} labels={labels.tolist()} "
          f"merged={merged.tolist()}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
