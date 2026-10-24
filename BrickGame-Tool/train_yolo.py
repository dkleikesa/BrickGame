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


def ensure_data_yaml() -> None:
    """按当前仓库位置重写 data.yaml 的 path。

    yaml 里存的是绝对路径，换机器或换克隆目录后 train/val 会指到不存在的地方；
    装配阶段写的那份只对生成它的机器有效。
    """
    yolo = os.path.dirname(DATA)
    if not os.path.isdir(os.path.join(yolo, "images", "train")):
        raise SystemExit(f"没找到装配好的数据集：{yolo}\n先跑 build-dataset/make_yolo_data.py")
    with open(DATA, "w", encoding="utf-8") as fh:
        fh.write(f"path: {yolo}\ntrain: images/train\nval: images/val\n"
                 "names:\n  0: board\n  1: tile\n")


def check_labels_ready(allow_unreviewed: bool) -> None:
    """开训前拦一次：未人工确认的帧进训练 = 模型学自己猜的框。

    预标注的漏检会被写成"这里没有砖"，训完再拿去预标注下一批，错误只累积不自愈。
    """
    import json
    ydir = os.path.join(HERE, "build-dataset", "yolo")
    meta_path = os.path.join(ydir, "meta.json")
    rev_path = os.path.join(ydir, "reviewed.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"找不到 {meta_path}，先跑 make_yolo_data.py")
    meta = json.load(open(meta_path, encoding="utf-8"))
    # 以 reviewed.json 为准：meta 里的 reviewed 是上次装配时的快照，
    # 用户之后点的确认只有重跑装配才会进去，拿它判断会误拒。
    reviewed = set()
    if os.path.exists(rev_path):
        reviewed = set(json.load(open(rev_path, encoding="utf-8")).get("frames", []))
    bad = [m for m in meta if m["file"] not in reviewed]
    if not bad:
        return
    print(f"⚠ {len(bad)}/{len(meta)} 帧未人工确认："
          f"{', '.join(m['src'] for m in bad[:3])}{' …' if len(bad) > 3 else ''}")
    if allow_unreviewed:
        print("  --allow-unreviewed 已指定，按预标注继续训练")
        return
    raise SystemExit(
        "拒绝开训：先把未确认帧过一遍 box_server.py，再跑 make_yolo_data.py。\n"
        "确实要用预标注硬训，加 --allow-unreviewed（指标不可信，别拿来发版）。"
    )


def backup_weights() -> None:
    """从 runs/brick 从头训会原地覆盖 best.pt/last.pt，先留一份带时间戳的。"""
    import time
    wdir = os.path.join(PROJECT, "brick", "weights")
    best = os.path.join(wdir, "best.pt")
    if not os.path.exists(best):
        return
    dst = os.path.join(wdir, f"best_{time.strftime('%Y%m%d-%H%M')}.pt")
    if not os.path.exists(dst):
        shutil.copyfile(best, dst)
        print(f"旧权重已备份 → {os.path.basename(dst)}")


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
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="未确认帧也允许开训（只用于排查，指标不可信）")
    args = ap.parse_args()

    ensure_data_yaml()
    check_labels_ready(args.allow_unreviewed)
    backup_weights()
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
        # 与 skill 的部署路径一致：nms=False 把 NMS 留在端侧，模型只出原始框
        out = YOLO(best).export(format="tflite", imgsz=args.imgsz, int8=False, nms=False)
        dst = os.path.join(PROJECT, "brick", "weights", "best.tflite")
        if os.path.abspath(str(out)) != os.path.abspath(dst):
            shutil.copyfile(str(out), dst)
        print(f"TFLite → {dst}（再 cp 到 androidApp/src/main/assets/brick_yolo.tflite）")


if __name__ == "__main__":
    main()
