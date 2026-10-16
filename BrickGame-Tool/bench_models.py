#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同机对比两个 tflite 的推理速度：现役 embedder（检索已融合） vs 闭集 softmax。

用 `ai_edge_litert` 解释器 —— 与手机上跑的是同一族运行时、同一套内核（XNNPACK）。
**注意这只排在同一个 CPU 上的相对快慢**：真机的 CPU 型号、线程数、GPU delegate(Metal/OpenCL) 都不同，
绝对毫秒数不能直接搬到手机；能搬的是"两者谁更快"。

一次推理处理 batch 个格子；一帧最多 144 格（12x12），所以帧耗时 ≈ ceil(144/batch) 次推理。
Android 侧 BATCH=32（固定形状，GPU delegate 要求），这里默认也测 32，另测 1 作参考。

用法:
  python bench_models.py --a ../androidApp/src/main/assets/brick_embedder.tflite \
                         --b out/bench/softmax_L2.tflite
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np


def bench(path: str, threads: int, iters: int, warmup: int) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    t_load = time.perf_counter()
    interp = Interpreter(model_path=path, num_threads=threads)
    interp.allocate_tensors()
    load_ms = (time.perf_counter() - t_load) * 1000
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    shape = [int(s) for s in inp["shape"]]
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"path": os.path.basename(path), "in": shape, "out": [int(s) for s in out["shape"]],
            "mb": os.path.getsize(path) / 1e6, "load_ms": load_ms,
            "median": statistics.median(ts), "mean": statistics.fmean(ts), "min": min(ts),
            "p90": sorted(ts)[int(len(ts) * 0.9)], "batch": shape[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description="多个 tflite 的同机延迟对比")
    ap.add_argument("--models", nargs="+", default=[
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "androidApp/src/main/assets/brick_embedder.tflite"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/bench/softmax_L2.tflite"),
    ], help="要测的 tflite 路径（默认 E 的 large 与 L2）")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--frames", type=int, default=144, help="一帧格子数（12x12），用于折算帧耗时")
    args = ap.parse_args()

    rows = []
    for p in args.models:
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}")
        r = bench(p, args.threads, args.iters, args.warmup)
        rows.append(r)

    print(f"线程 {args.threads}，预热 {args.warmup}，计时 {args.iters} 次；"
          f"一帧按 {args.frames} 格折算\n")
    hdr = (f"{'模型':<28}{'输入':<20}{'MB':>6}{'加载ms':>8}"
           f"{'中位ms':>9}{'均值ms':>9}{'最小ms':>9}{'帧ms':>9}{'相对A':>8}")
    print(hdr)
    print("-" * len(hdr))
    base = rows[0]["median"]
    for r in rows:
        batch = r["batch"]
        n_batches = -(-args.frames // batch)                 # ceil
        frame_ms = r["median"] * n_batches
        print(f"{r['path']:<28}{str(r['in']):<20}{r['mb']:>6.1f}{r['load_ms']:>8.1f}"
              f"{r['median']:>9.2f}{r['mean']:>9.2f}{r['min']:>9.2f}{frame_ms:>9.1f}"
              f"{r['median'] / base:>7.2f}x")
    print("\n相对A = 中位延迟 / 第一行中位延迟（<1 更快）")
    print("注意：这只是同一台 Mac 上的相对快慢；真机 CPU/线程数/GPU delegate 不同，绝对毫秒不能直接搬。")


if __name__ == "__main__":
    main()
