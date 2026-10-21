#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 board/tile 两类检测器并导出端侧模型。

数据集由 make_yolo_data.py 装配，框只有一个来源：prelabel_yolo.py 的模型预标注 +
box_server.py 的人工删/补。**不要**再从数据集逐格标签反推框（那条路要重算像素网格，
残局帧会被算成 191×48，实测白训一轮）。流程见 skills/brickgame-yolo-locator。

数据量小时（<200 帧）用 --resume 接着上一版练，从头训会让旧皮肤漂掉。

用法：
  .venv-train/bin/python train_yolo.py --epochs 80
  .venv-train/bin/python train_yolo.py --resume runs/brick/weights/last.pt --epochs 40
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
