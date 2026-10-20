#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
棋盘定位离线基线：对 samples/ 里全部整帧跑 detect_grid，量化"网格对不对"。

背景：仓库里 frames.csv 的 rows/cols 是 app 端算法输出，不是人工真值，拿它当
ground truth 会自我验证。所以这里给一个**不需要标注**的判据：

  对齐度(alignment) = 格线邻域的平均梯度能量 / 其余位置的平均梯度能量

网格正确时，棋盘所有格缝都落在格线上 → 格线处能量显著更高；错位时能量摊开。
再对网格整体平移 ±p/4、±p/2 算出"错位版"的最好成绩，用
  margin = 正位对齐度 / 最好错位对齐度
衡量判别力：margin 越接近 1，说明这个网格跟随机平铺无法区分，多半是错的。

这个指标只依赖图像本身，不受 app 端结果污染，可以直接比较不同定位算法。

用法：
  uv run --no-project --with numpy --with opencv-python-headless python eval_lattice.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tile_recognizer import INSET, detect_grid  # noqa: E402

LEDGER = os.path.join(HERE, "dataset", "frames.csv")
SAMPLES = os.path.join(HERE, "samples")
R = 2  # 格线邻域半径（像素）


def alignment(prof: np.ndarray, lines: np.ndarray) -> float:
    """格线邻域的平均能量 / 其余位置的平均能量。>1 表示能量确实聚在格线上。"""
    n = len(prof)
    mask = np.zeros(n, dtype=bool)
    for x in lines:
        xi = int(round(float(x)))
        lo, hi = max(0, xi - R), min(n, xi + R + 1)
        mask[lo:hi] = True
    on = float(prof[mask].sum()) / max(1, int(mask.sum()))
    off = float(prof[~mask].sum()) / max(1, int((~mask).sum()))
    return on / (off + 1e-9)


def load_ledger() -> dict[str, dict]:
    if not os.path.exists(LEDGER):
        return {}
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return {r["frame"]: r for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=SAMPLES)
    ap.add_argument("--out", default=os.path.join(HERE, "eval_lattice.csv"))
    ap.add_argument("--min-pitch", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ledger = load_ledger()
    names = sorted(
        f for f in os.listdir(args.samples)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    )

    out_rows = []
    for name in names:
        frame = os.path.splitext(name)[0]
        img = cv2.imdecode(np.fromfile(os.path.join(args.samples, name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] 读不到 {name}")
            continue
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        ok, rows, cols, px, py, src, err = True, 0, 0, 0, 0, "", ""
        align_x = align_y = 0.0
        try:
            g = detect_grid(img, args.min_pitch, INSET, verbose=args.verbose)
            rows, cols = int(g["score"].shape[0]), int(g["score"].shape[1])
            px, py = int(g["pitch_x"]), int(g["pitch_y"])
            src = str(g.get("src", ""))
            align_x = alignment(np.asarray(g["col_prof"]), np.asarray(g["xlines"]))
            align_y = alignment(np.asarray(g["row_prof"]), np.asarray(g["ylines"]))
        except (SystemExit, Exception) as e:  # detect_grid 用 SystemExit 报"定位失败"
            ok, err = False, str(e)[:50]
        ms = (time.perf_counter() - t0) * 1000

        led = ledger.get(frame, {})
        out_rows.append({
            "frame": frame, "ok": int(ok), "rows": rows, "cols": cols,
            "pitch_x": px, "pitch_y": py, "src": src,
            "ledger_layout": f"{led.get('rows', '')}x{led.get('cols', '')}",
            "layout_match": int(ok and str(rows) == led.get("rows") and str(cols) == led.get("cols")),
            "align_x": round(align_x, 2), "align_y": round(align_y, 2),
            "img": f"{w}x{h}", "ms": round(ms, 1), "err": err,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

    n = len(out_rows)
    ok_rows = [r for r in out_rows if r["ok"]]
    print(f"\n总帧 {n}，定位成功 {len(ok_rows)}（{len(ok_rows) / max(1, n):.0%}）")
    if ok_rows:
        ax = np.array([r["align_x"] for r in ok_rows])
        ay = np.array([r["align_y"] for r in ok_rows])
        print(f"格线能量集中度：x 中位 {np.median(ax):.2f}（最小 {ax.min():.2f}）"
              f" / y 中位 {np.median(ay):.2f}（最小 {ay.min():.2f}）")
    for r in out_rows:
        if not r["ok"]:
            print(f"  失败 {r['frame']}：{r['err']}")
    print(f"明细 → {args.out}")


if __name__ == "__main__":
    main()
