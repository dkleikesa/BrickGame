#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 YOLO 检测器定棋盘网格（**唯一的定位实现**；OpenCV 的 detect_grid 已删除）。

为什么不用投影/自相关：残局/空盘时没有周期可数，那套算法会把 14 行算成 13 行、
格线整体错一格；而 board/tile 检测在残局仍然稳。

算法与 Android 端 YoloBoardDetector 一致：board 框定范围、砖块尺寸定格距、
砖块中心做最小二乘对齐，行列数各试 ±1 取残差最小的一组。
"""
from __future__ import annotations

import os
import sys

import numpy as np
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)                       # 为了 import yolo_config
from yolo_config import CONF, TILE_INSET, predict_kwargs  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(_ROOT, "runs", "brick", "weights", "best.pt")
# 新机器/新克隆还没有 runs/ 时，用仓库里已经打进 APK 的那份模型兜底 ——
# 不然"切新截图"这一步会因为找不到 .pt 直接卡死。
ASSET_WEIGHTS = os.path.join(os.path.dirname(_ROOT), "androidApp", "src", "main", "assets",
                             "brick_yolo.tflite")

_model = None


def resolve_weights(weights: str = "") -> str:
    """按 .pt → APK 资产 .tflite 的顺序找可用模型；都没有就明确报错。"""
    if weights:
        if not os.path.exists(weights):
            raise SystemExit(f"模型不存在：{weights}")
        return weights
    for path in (DEFAULT_WEIGHTS, ASSET_WEIGHTS):
        if os.path.exists(path):
            if path != DEFAULT_WEIGHTS:
                print(f"没有 {DEFAULT_WEIGHTS}，改用 APK 里的 {path}")
            return path
    raise SystemExit(f"找不到定位模型：{DEFAULT_WEIGHTS} 或 {ASSET_WEIGHTS}；先跑 train_yolo.py")


def _load(weights: str):
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def _fit_axis(centers, start, step0, count):
    """一维最小二乘：返回（第 0 格左/上边, 格距）。"""
    if len(centers) < 2:
        return start, step0
    idx = np.round((np.array(centers) - start) / step0 - 0.5).astype(int)
    keep = [(c, i) for c, i in zip(centers, idx) if 0 <= i < count]
    if len({i for _, i in keep}) < 2:
        return start, step0
    cs = np.array([c for c, _ in keep]); ids = np.array([i for _, i in keep], float)
    den = len(cs) * (ids ** 2).sum() - ids.sum() ** 2
    if den == 0:
        return start, step0
    step = (len(cs) * (ids * cs).sum() - ids.sum() * cs.sum()) / den
    if step < step0 * 0.8 or step > step0 * 1.2:
        return start, step0
    center0 = (cs.sum() - step * ids.sum()) / len(cs)
    return center0 - step / 2.0, step


def grid_of(img: np.ndarray, weights: str = "", conf: float = CONF):
    """
    @return (xlines, ylines, rows, cols) 或 None（砖太少/残差过大，判不出可信网格）。
    """
    model = _load(resolve_weights(weights))
    res = model.predict(img, **predict_kwargs(conf))[0]
    board = None
    tiles = []
    for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            if board is None:
                board = box
        else:
            tiles.append(box)
    if board is None or len(tiles) < 2:
        return None

    x1, y1, x2, y2 = board
    bw, bh = x2 - x1, y2 - y1
    # 只认棋盘框内的砖：模型偶尔在标题/状态栏上误检一个框，一个离群点就能把最小二乘
    # 整个带偏。端侧 YoloBoardDetector 一直有这道 10% 外扩过滤，这里补齐，两边对同一帧
    # 的判断才一致（实测 3 帧定位失败就是因为棋盘上方 2 个误检框里漏掉了这一个）。
    mx, my = bw * 0.10, bh * 0.10
    tiles = [b for b in tiles
             if x1 - mx <= (b[0] + b[2]) / 2 <= x2 + mx
             and y1 - my <= (b[1] + b[3]) / 2 <= y2 + my]
    if len(tiles) < 2:
        return None

    side = float(np.median([((b[2] - b[0]) + (b[3] - b[1])) / 2 for b in tiles]))
    pitch = side / TILE_INSET
    if pitch < 12:
        return None
    base_cols, base_rows = int(round(bw / pitch)), int(round(bh / pitch))

    cxs = np.array([float((b[0] + b[2]) / 2) for b in tiles])
    cys = np.array([float((b[1] + b[3]) / 2) for b in tiles])
    need = max(3, int(round(0.6 * len(cxs))))     # 内点至少六成，否则这组行列不成立
    best = None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rows, cols = base_rows + dr, base_cols + dc
            if not (6 <= rows <= 20 and 6 <= cols <= 16):
                continue
            # 两轮：先按全量拟合，剔掉残差超过半个格距的离群框，再拟合一次。
            # 不用 max 残差当判据：一张满盘里混进 1 个离群框，max 直接爆表，
            # 会让正确网格输给贴合离群点的畸形网格（实测 14x10 的 p95 只有 2px，
            # max 却是 220px，于是被迫选 13x11）。
            mask = np.ones(len(cxs), bool)
            res = None
            for _ in range(2):
                lx, sx = _fit_axis(list(cxs[mask]), float(x1), bw / cols, cols)
                ty, sy = _fit_axis(list(cys[mask]), float(y1), bh / rows, rows)
                res = np.array([float(np.hypot(
                    cx - (lx + (min(max(int((cx - lx) / sx), 0), cols - 1) + 0.5) * sx),
                    cy - (ty + (min(max(int((cy - ty) / sy), 0), rows - 1) + 0.5) * sy)))
                    for cx, cy in zip(cxs, cys)])
                keep = res <= pitch * 0.5
                if keep.sum() < need or keep.sum() == mask.sum():
                    mask = keep
                    break
                mask = keep
            if mask.sum() < need:
                continue
            score = float(res[mask].max())
            if best is None or score < best[0]:
                best = (score, lx, sx, ty, sy, rows, cols)
    if best is None or best[0] > pitch * 0.35:
        return None
    res_max, lx, sx, ty, sy, rows, cols = best
    xl = np.linspace(lx, lx + cols * sx, cols + 1)
    yl = np.linspace(ty, ty + rows * sy, rows + 1)
    return xl, yl, rows, cols


def locate(img: np.ndarray, inset: float | None = None) -> dict | None:
    """YOLO 定网格 + 裁格/判空，返回 `tile_recognizer.grid_from_lines` 那份 dict。

    额外补 `pitch_x` / `pitch_y` / `src` 三个字段，兼容旧调用点读的 key。
    定位失败（砖太少/残差过大）返回 None。
    """
    parent = os.path.dirname(_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import tile_recognizer as T  # noqa: PLC0415

    got = grid_of(img)
    if got is None:
        return None
    xl, yl, _, _ = got
    g = T.grid_from_lines(img, xl, yl, T.INSET if inset is None else inset)
    g["pitch_x"] = float(np.median(np.diff(xl)))
    g["pitch_y"] = float(np.median(np.diff(yl)))
    g["src"] = "yolo"
    return g
