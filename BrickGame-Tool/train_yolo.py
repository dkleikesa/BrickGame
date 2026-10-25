#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 board/tile 两类检测器并导出端侧模型。

数据集由 make_yolo_data.py 装配，框只有一个来源：prelabel_yolo.py 的模型预标注 +
box_server.py 的人工删/补。**不要**再从数据集逐格标签反推框（那条路要重算像素网格，
残局帧会被算成 191×48，实测白训一轮）。流程见 skills/brickgame-yolo-locator。

输入尺寸只认 yolo_config.py（当前 **1024×480，高×宽**）。矩形输入只占方形 1024 的
47% 像素，内容分辨率完全不变。ultralytics 的 train/val 只收整数 imgsz，矩形训练要用
官方的 `rect=True`：它按图片长宽比生成矩形 batch（竖屏正好 1024×480，train 模式
pad=0），同时自动关掉只支持方形的 Mosaic。其余 scale/translate/hsv 增强保留，
249 帧的规模关掉 mosaic 仍能防过拟合。

接续训练有两种，别混：
  `--model runs/brick/weights/best.pt`  从**已完成**的权重继续（新优化器，补数据/换尺寸用）
  `--resume runs/brick/weights/last.pt` 只有**没跑完**的 run 才能 resume（恢复 epoch/优化器）；
                                         已完成的 run 会报 "training to N epochs is finished"

用法：
  .venv-train/bin/python train_yolo.py --epochs 80
  .venv-train/bin/python train_yolo.py --model runs/brick/weights/best.pt --epochs 40
  .venv-train/bin/python train_yolo.py --name brick_smoke --epochs 1   # 只验证链路，别发版
"""
from __future__ import annotations

import argparse
import os
import shutil

from PIL import Image
from ultralytics import YOLO

from yolo_config import IMG_H, IMG_W, IMG_SIZE, anchors_for

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "build-dataset", "yolo", "data.yaml")
PROJECT = os.path.join(HERE, "runs")


def parse_imgsz(text: str) -> tuple[int, int]:
    """'1024x480' / '1024,480' / '1024' → (高, 宽)；宽高必须是 32 的倍数。"""
    raw = str(text).lower().replace(",", "x").replace("*", "x").strip()
    parts = raw.split("x")
    try:
        if len(parts) == 1:
            h = w = int(parts[0])
        elif len(parts) == 2:
            h, w = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"imgsz 格式应为 高x宽（如 1024x480）：{text}")
    if h <= 0 or w <= 0 or h % 32 or w % 32:
        raise argparse.ArgumentTypeError(f"imgsz 宽高必须是 32 的倍数：{text}")
    if h < w:
        # rect=True 的 batch 形状由数据集图片长宽比决定（本项目是竖屏），导出宽高
        # 必须与之一致；写成 480x1024 会导出一个和训练形状转置的模型。
        raise argparse.ArgumentTypeError(f"高应不小于宽（竖屏截图），收到 {text}")
    return h, w


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


def check_dataset_resolution(imgsz: tuple[int, int], allow_upscale: bool) -> None:
    """开训前确认装配图的像素够 imgsz —— 否则就是在放大旧图。

    装配时长边被缩到 yolo_config.MAX_SIDE（= IMG_H）。如果这里读到的长边小于模型
    输入的长边，说明数据集还是上一代尺寸：算力全花在插值上，小砖块该糊还是糊，
    训完指标还可能"看起来正常"（因为验证集也是同一批放大图）。默认直接拦。
    """
    need_h, need_w = imgsz
    need = max(need_h, need_w)
    img_dir = os.path.join(HERE, "build-dataset", "yolo", "images", "train")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"没找到 {img_dir}，先跑 build-dataset/make_yolo_data.py")
    names = sorted(n for n in os.listdir(img_dir)
                   if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"})[:8]
    if not names:
        raise SystemExit(f"{img_dir} 是空的，先跑 build-dataset/make_yolo_data.py")
    longest = 0
    for name in names:
        with Image.open(os.path.join(img_dir, name)) as im:
            longest = max(longest, max(im.size))
    print(f"装配图长边 {longest}px（抽检 {len(names)} 张），训练 imgsz={need_h}x{need_w}")
    if longest >= need:
        return
    msg = (f"装配图长边只有 {longest}px，小于 imgsz 长边 {need}：训练时会放大，"
           f"小砖块的细节不会因此变多。\n"
           f"  先把数据集升到当前尺寸（框是归一化的，不会错位）：\n"
           f"  .venv-train/bin/python build-dataset/make_yolo_data.py")
    if not allow_upscale:
        raise SystemExit(msg + "\n确实要做放大对比实验，加 --allow-upscale。")
    print("⚠ " + msg)


def run_dir(resume: str, name: str) -> str:
    """这次训练真正的输出目录。

    `--resume` 时 ultralytics 用 checkpoint 里的 project/name 决定保存位置，
    传进来的 --name 会被忽略；所以优先从 `.../weights/last.pt` 反推 run 目录，
    否则备份和导出会落到另一个 run 里。
    """
    if resume:
        path = os.path.abspath(resume)
        if os.path.basename(os.path.dirname(path)) == "weights":
            return os.path.dirname(os.path.dirname(path))
    return os.path.join(PROJECT, name)


def backup_weights(wdir: str) -> None:
    """从同一个 run 目录从头训会原地覆盖 best.pt/last.pt，先留一份带时间戳的。"""
    import time
    best = os.path.join(wdir, "weights", "best.pt")
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


def verify_tflite(path: str, imgsz: tuple[int, int]) -> None:
    """导出后核对 IO 形状。

    Android 端从 .tflite 的张量形状自适应，所以这里不是为了让端侧能跑，而是拦住
    "导出没吃 --imgsz / 输出层数变了"这类静默跑偏：形状不对就别往 assets 里拷。
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("（没装 ai_edge_litert，跳过 IO 形状校验）")
        return
    h, w = imgsz
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    in_shape = list(interp.get_input_details()[0]["shape"])
    out_shape = list(interp.get_output_details()[0]["shape"])
    print(f"TFLite IO：输入 {in_shape} 输出 {out_shape}")
    want_anchors = anchors_for(h, w)
    problems = []
    if in_shape != [1, 3, h, w]:
        problems.append(f"输入应为 [1,3,{h},{w}]")
    if len(out_shape) != 3 or out_shape[0] != 1 or out_shape[1] != 6 or out_shape[2] != want_anchors:
        problems.append(f"输出应为 [1,6,{want_anchors}]（{h}x{w} 三层特征图之和）")
    if problems:
        raise SystemExit("；".join(problems) + "，不要拷进 assets")
    print(f"✓ IO 与 yolo_config 一致（{h}x{w} → {want_anchors} anchors）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=parse_imgsz, default=IMG_SIZE,
                    help=f"导出/推理的高x宽，默认 {IMG_H}x{IMG_W}；训练用其长边 + rect=True "
                         f"生成同尺寸矩形 batch。也接受 1024 这种方形写法")
    ap.add_argument("--batch", type=int, default=8,
                    help="默认 8：1024x480 的像素量与 640 方图接近")
    ap.add_argument("--name", default="brick",
                    help="输出到 runs/<name>/；冒烟测试用别的名字，别覆盖正式权重")
    ap.add_argument("--resume", default="", help="从**没跑完**的 last.pt 恢复训练（补数据后用 --model）")
    ap.add_argument("--export", action="store_true", help="训练完导出 TFLite")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="未确认帧也允许开训（只用于排查，指标不可信）")
    ap.add_argument("--allow-upscale", action="store_true",
                    help="装配图小于 imgsz 也硬训（只用于对比实验）")
    args = ap.parse_args()

    ensure_data_yaml()
    check_labels_ready(args.allow_unreviewed)
    check_dataset_resolution(args.imgsz, args.allow_upscale)
    out_dir = run_dir(args.resume, args.name)
    backup_weights(out_dir)
    model = YOLO(args.resume or args.model)
    device = pick_device()

    ## 训练几何：ultralytics 的 train/val 只收整数 imgsz（传 tuple 会被强制回方图），
    # 矩形要用 rect=True：dataset 按图片长宽比生成 batch_shapes（train 模式 pad=0，
    # 竖屏 1024 高 → 1024x480），并自动把 mosaic 置 0（Mosaic 只支持方形）。
    train_imgsz = max(args.imgsz)
    square = args.imgsz[0] == args.imgsz[1]
    train_rect = not square
    print(f"训练设备 {device}，imgsz={train_imgsz} rect={train_rect} batch={args.batch}，数据 {DATA}")
    print(f"训练几何：{'方形' if square else '矩形'} batch≈{args.imgsz[0]}x{args.imgsz[1]}，"
          f"导出/推理 imgsz={args.imgsz[0]}x{args.imgsz[1]}，"
          f"mosaic={'1.0' if square else '0（rect=True 自动关闭）'}")
    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=train_imgsz,
        batch=args.batch,
        device=device,
        rect=train_rect,  # 矩形 batch；导出/推理是固定 1024x480
        project=PROJECT,
        name=args.name,
        exist_ok=True,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.0,      # 棋盘左右对称，但砖块图案有方向，左右翻会造出不存在的样本
        flipud=0.0,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.35,
        mosaic=1.0 if square else 0.0,
        close_mosaic=10 if square else 0,
        patience=30,
        workers=4,
    )

    # 以 trainer 的 best 为准：--resume 时保存目录来自 checkpoint，不是 --name。
    trainer_best = getattr(getattr(model, "trainer", None), "best", None)
    best = str(trainer_best) if trainer_best and os.path.exists(str(trainer_best)) \
        else os.path.join(out_dir, "weights", "best.pt")
    # val 也用 rect=True（与训练同口径）。注意 val 模式 pad=0.5，矩形 batch 会略大
    # （1056x480）；最终部署指标以 eval_gate.py 的 1024x480 predict 为准。
    metrics = model.val(data=DATA, imgsz=train_imgsz, device=device, rect=train_rect)
    print(f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")
    print(f"best.pt → {best}")

    if args.export:
        # 与 skill 的部署路径一致：nms=False 把 NMS 留在端侧，模型只出原始框
        out = YOLO(best).export(format="tflite", imgsz=args.imgsz, int8=False, nms=False)
        dst = os.path.join(os.path.dirname(best), "best.tflite")
        if os.path.abspath(str(out)) != os.path.abspath(dst):
            shutil.copyfile(str(out), dst)
        print(f"TFLite → {dst}（再 cp 到 androidApp/src/main/assets/brick_yolo.tflite）")
        verify_tflite(dst, args.imgsz)


if __name__ == "__main__":
    main()
