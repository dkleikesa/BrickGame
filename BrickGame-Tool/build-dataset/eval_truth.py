#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用人工四角真值评估定位（corner_tool.html 导出的 ground_truth.csv）。

判定口径：
  · verdict=ok   —— 人工确认算法网格正确，不产生独立几何真值，只计入"认可率"；
  · verdict=quad —— 人工点了四角，是唯一独立的真值，用它算角点误差；
  · 算法失败但该帧被标了四角 —— 记为漏检。

误差用 pitch 归一化（单位＝一格），跨分辨率/布局可比。

用法：
  uv run --no-project --with numpy --with opencv-python-headless python eval_truth.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import yolo_grid  # noqa: E402


def read_truth(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.path.join(HERE, "ground_truth.csv"))
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--min-pitch", type=int, default=12)
    args = ap.parse_args()

    if not os.path.exists(args.truth):
        print(f"没有真值文件：{args.truth}")
        print("先打开 corner_tool.html 标注并导出，改名放到这个位置。")
        return

    rows_all = read_truth(args.truth)
    n_ok = n_quad = n_skip = 0
    misses, errs = [], []

    for r in rows_all:
        frame, verdict = r["frame"], (r["verdict"] or "").strip()
        if verdict == "ok":
            n_ok += 1
            continue
        if verdict != "quad" or not r.get("x0"):
            n_skip += 1
            continue
        n_quad += 1
        gt = np.array([[float(r[f"x{i}"]), float(r[f"y{i}"])] for i in range(4)], np.float64)

        path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(args.samples, frame + ext)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            print(f"[skip] 找不到原图 {frame}")
            continue
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        g = yolo_grid.locate(img)
        if g is None:
            misses.append(frame)
            continue
        xl, yl = np.asarray(g["xlines"]), np.asarray(g["ylines"])
        det = np.array([[xl[0], yl[0]], [xl[-1], yl[0]], [xl[-1], yl[-1]], [xl[0], yl[-1]]])
        pitch = float(g["pitch_x"])
        d = np.linalg.norm(det - gt, axis=1) / max(1.0, pitch)
        errs.append((frame, d.max(), d.mean(), float(g["score"].shape[0]), float(g["score"].shape[1])))

    judged = n_ok + n_quad
    print(f"\n有判断 {judged} 帧：人工认可 {n_ok}（{n_ok / max(1, judged):.0%}）"
          f"，标了四角 {n_quad}，未标 {n_skip}")
    if misses:
        print(f"人工认为可定位、算法却没成功：{len(misses)} 帧 → {', '.join(misses[:8])}")
    if errs:
        mx = np.array([e[1] for e in errs])
        print(f"四角最大误差（格）：中位 {np.median(mx):.2f} / P95 {np.percentile(mx, 95):.2f}"
              f" / 最差 {mx.max():.2f}")
        print("误差最大的 5 帧：")
        for frame, m, mean, rows, cols in sorted(errs, key=lambda x: -x[1])[:5]:
            print(f"  {frame} {int(rows)}x{int(cols)} 最大 {m:.2f} 格 / 平均 {mean:.2f} 格")
    if not errs and not misses:
        print("没有被标四角的帧 —— 若算法网格确实全对，这本身就是结论（认可率 100%）。")


if __name__ == "__main__":
    main()
