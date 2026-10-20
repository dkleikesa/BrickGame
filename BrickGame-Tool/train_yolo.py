#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 board/tile 两类检测器并导出端侧模型。

数据集由 make_yolo_data.py 生成（弱标注：tile 框从逐格标签反推，board 框是棋盘外框）。
36 张图训 nano 属于"跑通链路"，不是最终精度 —— 目标是先拿到能装进 APK、能和 OpenCV
逐帧对比的东西，再按实际失败的帧补数据重训（见 --resume）。

用法：
  uv run --no-project --python 3.12 --with ultralytics python train_yolo.py
  uv run ... python train_yolo.py --resume runs/brick/weights/last.pt --epochs 100
"""
from __future__ import annotations

import argparse
import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "build-dataset", "yolo", "data.yaml")
PROJECT = os.path.join(HERE, "runs")


def pick_device() -> str:
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--resume", default="", help="从某个 last.pt 继续训练（补数据后用）")
    ap.add_argument("--export", action="store_true", help="训练完导出 TFLite")
    args = ap.parse_args()

    model = YOLO(args.resume or args.model)
    device = pick_device()
    print(f"训练设备 {device}，数据 {DATA}")
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=PROJECT,
        name="brick",
        exist_ok=True,
        # 36 张图 + 强增强：不加增强会直接把 36 张背下来
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.0,      # 棋盘左右对称，但砖块图案有方向，左右翻会造出不存在的样本
        flipud=0.0,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.35,
        mosaic=1.0,
        close_mosaic=10,
        patience=30,
        workers=4,
    )

    best = os.path.join(PROJECT, "brick", "weights", "best.pt")
    metrics = model.val(data=DATA, imgsz=args.imgsz, device=device)
    print(f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")
    print(f"best.pt → {best}")

    if args.export:
        out = YOLO(best).export(format="tflite", imgsz=args.imgsz, int8=False)
        dst = os.path.join(HERE, "yolo", "brickgame_yolo.tflite")
        shutil.copyfile(str(out), dst)
        print(f"TFLite → {dst}")


if __name__ == "__main__":
    main()
