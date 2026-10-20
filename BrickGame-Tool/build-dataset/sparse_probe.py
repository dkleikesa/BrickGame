#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
残局压力测试：把满盘图"擦"掉大部分砖块，看定位从剩多少块开始崩。

为什么这么测：手机上没有失败帧、"很多失败"也没法复现。但残局失败的本质是
**棋盘上周期结构不足**，完全可以拿满盘图合成出来 —— 抹掉格子用 inpaint 填成底色，
几何真值不变（还是那张图的网格），于是"剩 k 块时算法对不对"就能精确量化。

产出：samples-sparse/ 下的合成图（可复用做算法 A/B）、sparse_result.csv、控制台曲线。

用法：
  uv run --no-project --with numpy --with opencv-python-headless python sparse_probe.py
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
from tile_recognizer import INSET, detect_grid  # noqa: E402

OUT_DIR = os.path.join(HERE, "samples-sparse")


def erase(img: np.ndarray, g: dict, keep: set[tuple[int, int]]) -> np.ndarray:
    """把 keep 之外的格子抹成棋盘底色。inpaint 从格缝取色，避免留下硬边。"""
    xl, yl = np.asarray(g["xlines"]), np.asarray(g["ylines"])
    mask = np.zeros(img.shape[:2], np.uint8)
    for r in range(len(yl) - 1):
        for c in range(len(xl) - 1):
            if (r, c) in keep:
                continue
            x0, x1 = int(xl[c]), int(xl[c + 1])
            y0, y1 = int(yl[r]), int(yl[r + 1])
            mx, my = int((x1 - x0) * 0.10), int((y1 - y0) * 0.10)
            mask[y0 + my:y1 - my, x0 + mx:x1 - mx] = 255
    return cv2.inpaint(img, mask, 12, cv2.INPAINT_TELEA)


def corner_error(g: dict, truth: dict) -> float:
    xl, yl = np.asarray(g["xlines"]), np.asarray(g["ylines"])
    det = np.array([[xl[0], yl[0]], [xl[-1], yl[0]], [xl[-1], yl[-1]], [xl[0], yl[-1]]])
    return float(np.linalg.norm(det - truth["quad"], axis=1).max() / truth["pitch"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--limit", type=int, default=4, help="用几张满盘图做底图")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 底图只取满盘（空位少）的帧，避免擦之前就已经是残局
    bases = []
    for name in sorted(os.listdir(args.samples)):
        p = os.path.join(args.samples, name)
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            g = detect_grid(img, 12, INSET, verbose=False)
        except (SystemExit, Exception):
            continue
        rows, cols = g["score"].shape
        empty = int((g["score"] < g["score_thr"]).sum())
        if rows * cols - empty < rows * cols * 0.9:
            continue
        bases.append((os.path.splitext(name)[0], img, g, rows, cols))
        if len(bases) >= args.limit:
            break
    if not bases:
        print("没有满盘底图")
        return

    ks = [3, 5, 8, 12, 20, 30, 50]
    recs = []
    for stem, img, g, rows, cols in bases:
        truth = {
            "rows": rows, "cols": cols, "pitch": float(g["pitch_x"]),
            "quad": np.array([[g["xlines"][0], g["ylines"][0]], [g["xlines"][-1], g["ylines"][0]],
                              [g["xlines"][-1], g["ylines"][-1]], [g["xlines"][0], g["ylines"][-1]]], float),
        }
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        for k in ks:
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 1000 + k)
                idx = rng.choice(len(cells), size=min(k, len(cells)), replace=False)
                keep = {cells[i] for i in idx}
                sparse = erase(img, g, keep)
                name = f"{stem}_k{k}_s{seed}"
                cv2.imencode(".png", sparse)[1].tofile(os.path.join(OUT_DIR, name + ".png"))

                try:
                    got = detect_grid(sparse, 12, INSET, verbose=False)
                    ok = True
                    got_rows, got_cols = got["score"].shape
                    got_px = int(got["pitch_x"])
                    err = corner_error(got, truth)
                    exact = (got_rows == rows and got_cols == cols and abs(got_px - truth["pitch"]) <= 5
                             and err < 0.35)
                    src, note = str(got.get("src", "")), ""
                except (SystemExit, Exception) as e:
                    ok = exact = False
                    got_rows = got_cols = got_px = 0
                    err = float("nan")
                    src, note = "", str(e)[:40]
                recs.append({"name": name, "base": stem, "k": k, "seed": seed,
                             "ok": int(ok), "exact": int(exact),
                             "rows": got_rows, "cols": got_cols, "pitch": got_px,
                             "corner_err": round(err, 2), "src": src, "note": note})

    with open(os.path.join(HERE, "sparse_result.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        wr.writeheader()
        wr.writerows(recs)

    print(f"\n合成 {len(recs)} 张残局图（底图 {len(bases)} 张）")
    print("剩砖  定位成功   网格完全正确   典型错法")
    for k in ks:
        grp = [r for r in recs if r["k"] == k]
        ok = sum(r["ok"] for r in grp)
        ex = sum(r["exact"] for r in grp)
        bad = [r for r in grp if not r["exact"]]
        tip = ""
        if bad:
            b = bad[0]
            tip = (f"{b['rows']}x{b['cols']} pitch={b['pitch']} {b['src']}"
                   f"{(' ' + b['note']) if b['note'] else ''}")
        print(f"{k:>4}   {ok:>2}/{len(grp):<2}      {ex:>2}/{len(grp):<2}         {tip}")
    print(f"合成图 → {OUT_DIR}；明细 → sparse_result.csv")


if __name__ == "__main__":
    main()
