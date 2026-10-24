#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""识别模型放行闸门：新模型 vs 基线，**在当前数据集上重新评测**后比较。

为什么不能直接比 `metrics.json`：那份文件是训练当时的快照，数据集（新帧、人工改标）
已经变了，拿旧分数比新分数等于拿两把尺子量。这里每次都重跑 `train_classifier --eval-only`，
保证两个模型看的是同一批查询格（同 seed / 同 val 划分）。

比较口径（`real` 块，越小越好的是距离）：
  · top1 —— 不能比基线低超过 0.3 个点；
  · hit_dist_p95 —— 命中距离 p95 不能比基线高超过 0.005；
  · wrong_dist_p5 —— 最近异类距离 p5 不能比基线低超过 0.01（间隔塌了，阈值就得跟着降）；
  · 绝对下限 top1 ≥ 0.95。
任一项不过 → exit 1，`gate.json` 写 pass=false，导出脚本会拒绝装 assets。

基线来源：默认用滚动基线 `out/clf_baseline`（含 best.pt + baseline.json）；`--base` 只用于临时
指定别处。通过后加 `--update` 会把这一版的 best.pt 和指标写进滚动基线目录，下一轮自动与它比
（旧实现写在 `<new>/baseline.json`，每轮又拿写死的 `--base` 比，基线永远停在第一次那版）。

用法：
  ~/.venvs/brick-train/bin/python eval_clf_gate.py --new out/clf_v2 --base out/clf_real --update
  ... --new out/clf_v2            # 只跟上一版通过的基线比
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))

MIN_TOP1 = 0.95
TOP1_TOL = 0.003
HIT_TOL = 0.005
WRONG_TOL = 0.01


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_info(ckpt: str) -> dict:
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {"backbone": ck.get("backbone"), "img": int(ck.get("img") or 0)}


def eval_ckpt(ckpt: str, out_dir: str) -> dict:
    """在当前数据上跑一次 `--eval-only`，返回 metrics["real"]。"""
    info = ckpt_info(ckpt)
    if not info["backbone"] or not info["img"]:
        raise SystemExit(f"{ckpt} 里没有 backbone/img，无法复现评测")
    cmd = [
        sys.executable, os.path.join(_ROOT, "train_classifier.py"),
        "--eval-only", "--data", "real",
        "--resume", ckpt, "--backbone", info["backbone"], "--img", str(info["img"]),
        "--out", out_dir,
    ]
    print("  评测：", " ".join(cmd[2:]))
    env = dict(os.environ, PYTHONPATH=_ROOT)
    subprocess.run(cmd, check=True, env=env, cwd=_ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)["real"]


def fmt(m: dict | None) -> str:
    if not m:
        return "（无基线）"
    return (f"top1 {m['top1']:.3f}  命中p95 {m['hit_dist_p95']:.3f}  "
            f"异类p5 {m['wrong_dist_p5']:.3f}  类 {m['n_class']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="识别模型放行闸门（新 vs 基线，同一批查询）")
    ap.add_argument("--new", required=True, help="新模型目录（含 best.pt）")
    ap.add_argument("--base", default="", help="临时指定基线目录（含 best.pt）；一般不用给")
    ap.add_argument("--baseline-dir", default=os.path.join(_ROOT, "out", "clf_baseline"),
                    help="滚动基线目录：--update 通过后把这一版写进这里，下次默认与它比")
    ap.add_argument("--update", action="store_true", help="通过后把这次结果写成新基线")
    ap.add_argument("--force", action="store_true", help="不达标也写 gate.json 为 pass（慎用）")
    args = ap.parse_args()

    ckpt = os.path.join(args.new, "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"找不到 {ckpt}")
    print(f"新模型 {ckpt}（{ckpt_info(ckpt)['backbone']} img={ckpt_info(ckpt)['img']}）")
    with tempfile.TemporaryDirectory() as td:
        new_m = eval_ckpt(ckpt, os.path.join(td, "new"))

        # 基线来源：--base（临时）→ --baseline-dir（滚动基线）→ 无。
        # 曾经默认落在 --new 目录，导致每轮都跟一个写死的旧目录比，基线永远不前进。
        base_dir = args.base or args.baseline_dir
        base_ckpt = os.path.join(base_dir, "best.pt")
        base_m = None
        if base_ckpt and os.path.exists(base_ckpt):
            print(f"基线   {base_ckpt}")
            base_m = eval_ckpt(base_ckpt, os.path.join(td, "base"))
        elif os.path.exists(os.path.join(base_dir, "baseline.json")):
            saved = json.load(open(os.path.join(base_dir, "baseline.json"), encoding="utf-8"))
            base_m = saved["real"]
            print(f"基线   沿用 {base_dir}/baseline.json（{saved.get('at', '')}）")
        else:
            print("基线   无（首次运行，只查绝对下限；通过后 --update 建基线）")

    print(f"新     {fmt(new_m)}")
    print(f"基线   {fmt(base_m)}")

    reasons = []
    if new_m["n_class"] != (base_m or new_m)["n_class"]:
        reasons.append(f"类数变了：{new_m['n_class']} vs {base_m['n_class']}（图鉴改过要重建基线）")
    if new_m["top1"] < MIN_TOP1:
        reasons.append(f"top1 {new_m['top1']:.3f} < {MIN_TOP1}")
    if base_m:
        if new_m["top1"] < base_m["top1"] - TOP1_TOL:
            reasons.append(f"top1 比基线低 {base_m['top1'] - new_m['top1']:.3f}（> {TOP1_TOL}）")
        if new_m["hit_dist_p95"] > base_m["hit_dist_p95"] + HIT_TOL:
            reasons.append(f"命中 p95 距离上浮 {new_m['hit_dist_p95'] - base_m['hit_dist_p95']:.3f}"
                           f"（> {HIT_TOL}）")
        if new_m["wrong_dist_p5"] < base_m["wrong_dist_p5"] - WRONG_TOL:
            reasons.append(f"异类 p5 距离下降 {base_m['wrong_dist_p5'] - new_m['wrong_dist_p5']:.3f}"
                           f"（> {WRONG_TOL}）")

    passed = not reasons or args.force
    gate = {
        "pass": passed,
        "reasons": reasons,
        "new": new_m,
        "base": base_m,
        "ckpt": os.path.relpath(ckpt, _ROOT),
        "ckpt_md5": md5_of(ckpt),
        "thresholds": {"min_top1": MIN_TOP1, "top1_tol": TOP1_TOL,
                       "hit_p95_tol": HIT_TOL, "wrong_p5_tol": WRONG_TOL},
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    gate_path = os.path.join(args.new, "gate.json")
    json.dump(gate, open(gate_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {gate_path}（pass={passed}）")

    if not passed:
        print("✗ 不放行：" + "；".join(reasons))
        sys.exit(1)
    print("✓ 通过" + ("（--force）" if reasons else ""))
    if args.update:
        os.makedirs(args.baseline_dir, exist_ok=True)
        shutil.copyfile(ckpt, os.path.join(args.baseline_dir, "best.pt"))
        base_path = os.path.join(args.baseline_dir, "baseline.json")
        json.dump({"real": new_m, "ckpt": gate["ckpt"], "ckpt_md5": gate["ckpt_md5"],
                   "at": gate["at"]},
                  open(base_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"基线已更新 → {base_path}")


if __name__ == "__main__":
    main()
