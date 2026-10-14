#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""brick_embedder.onnx → brick_embedder.tflite，并现场与 ONNX 输出对拍。

Android 侧的识别模型就是这个 tflite（`ml/TfliteTileClassifier`，跑 GPU delegate），
ONNX 只留在这边当转换源和数值基准 —— 12.8MB 的 .onnx 不往 APK 里塞。

为什么从 ONNX 转而不是从 `best.pt` 重导：模型包里已经有校验过的 ONNX，
再装一套 torch 只会多一个版本漂移的来源。

转换完有两道闸，都不过就别接 Android：
  1. 数值：TFLite 与 ONNX 同一批输入，余弦 ≥ 0.999；
  2. GPU delegate：用 LiteRT 的 GPU 加速器试编译一次。GPU delegate 只吃**静态形状**，
     模型里只要有一个"运行时才定形状"的张量就挂不上，真机上表现为
     `Interpreter` 构造抛 `Error applying delegate`（且只有真机才发现，所以本地先拦）。

用法：
  ~/.venvs/brick-tflite/bin/python export_tflite.py
  ~/.venvs/brick-tflite/bin/python export_tflite.py --onnx out/brick_embedder.onnx
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(os.path.dirname(_HERE), "androidApp", "src", "main", "assets")

# ONNX 的两个可能位置：从 APK 目录挪出来的那份，或 export_mobile.py 的默认输出位置
_ONNX_CANDIDATES = (
    os.path.join(_HERE, "out", "brick_embedder.onnx"),
    os.path.join(_ASSETS, "brick_embedder.onnx"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX → TFLite（Android 侧速度对照用）")
    ap.add_argument(
        "--onnx",
        default=next((p for p in _ONNX_CANDIDATES if os.path.exists(p)), _ONNX_CANDIDATES[0]),
    )
    ap.add_argument("--out", default=os.path.join(_ASSETS, "brick_embedder.tflite"))
    ap.add_argument("--batch", type=int, default=0,
                    help="固定成这个 batch（只对还留着动态维的老 ONNX 有用）；0 = 按 ONNX 原样")
    ap.add_argument("--workdir", default=os.path.join(_HERE, "out", "tflite"))
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(
            f"找不到 {args.onnx}\n（先跑 export_mobile.py，或把它产出的 .onnx 放到 BrickGame-Tool/out/）"
        )

    import onnx2tf

    # onnx2tf 不做增量，直接清掉上次的产物，避免拿到旧文件
    if os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)
    os.makedirs(args.workdir, exist_ok=True)

    src = args.onnx
    if args.batch > 0:
        src = os.path.join(args.workdir, f"brick_embedder_b{args.batch}.onnx")
        fix_batch(args.onnx, src, args.batch)

    onnx2tf.convert(
        input_onnx_file_path=src,
        output_folder_path=args.workdir,
        batch_size=args.batch if args.batch > 0 else None,
        non_verbose=True,
    )

    all_tflite = sorted(glob.glob(os.path.join(args.workdir, "**", "*.tflite"), recursive=True))
    # onnx2tf 会同时吐 float32 和 float16 两份。默认取 float32：float16 那份是权重量化，
    # 算子行为和精度都不一样，塞进 APK 会让"谁更快"这个问题变成"量化后谁更快"。
    produced = [p for p in all_tflite if "float16" not in os.path.basename(p)] or all_tflite
    if not produced:
        raise SystemExit(f"{args.workdir} 里没有产出 .tflite")
    shutil.copyfile(produced[0], args.out)
    print(f"写入 {args.out}（{os.path.getsize(args.out) / 1e6:.1f} MB，源 {os.path.basename(produced[0])}）")

    verify(args.onnx, args.out)


def fix_batch(onnx_path: str, out_path: str, batch: int) -> None:
    """把输入/输出的 batch 维从符号固定成常量。

    为什么不直接靠 onnx2tf 的 `batch_size`：实测这个版本对动态输入不生效，导出来还是
    动态 batch —— 而动态 batch 会让 XNNPACK 在 prepare 阶段就失败
    （`failed to reshape runtimeNode ...`），TFLite 只能退回未优化的 CPU kernel，
    那时候量出来的速度就不是 TFLite 的真实水平了。
    """
    import onnx

    model = onnx.load(onnx_path)
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) > 0 and dims[0].dim_param:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = batch
    onnx.save(model, out_path)
    print(f"固定 batch={batch} → {os.path.basename(out_path)}")


def verify(onnx_path: str, tflite_path: str) -> None:
    """同一批随机输入跑两边，比余弦相似度 —— 不一致就别塞进 APK。"""
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_dims = sess.get_inputs()[0].shape            # [batch, 3, side, side]
    out_dims = sess.get_outputs()[0].shape          # [batch, dim]
    if not isinstance(in_dims[0], int) or not isinstance(out_dims[0], int):
        raise SystemExit(
            "ONNX 的 batch 还是动态维：用 export_mobile.py 重导（默认 --batch 32 固定形状），"
            "动态 batch 会在 TFLite 里留下运行时 shape 的张量，GPU delegate 挂不上"
        )
    n, side, dim = int(in_dims[0]), int(in_dims[2]), int(out_dims[1])

    rng = np.random.default_rng(0)
    x = rng.random((n, 3, side, side), dtype=np.float32)

    ref = sess.run(None, {"input": x})[0]

    interp = Interpreter(model_path=tflite_path, num_threads=4)
    in_shape = list(interp.get_input_details()[0]["shape"])
    print(f"TFLite 输入 shape: {in_shape}")
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    # onnx2tf 会把输入换成 TFLite 惯用的 NHWC —— 那正是 XNNPACK 的原生布局，
    # 所以保留它（Android 侧按 NHWC 喂，不做 NCHW→NHWC 的图内 transpose）
    shape = list(inp["shape"])
    feed = np.transpose(x, (0, 2, 3, 1)) if shape[1:3] == [side, side] else x
    interp.set_tensor(inp["index"], np.ascontiguousarray(feed, dtype=np.float32))
    interp.invoke()
    got = interp.get_tensor(out["index"]).reshape(n, dim)
    print(f"TFLite 输出 shape: {list(out['shape'])}")

    cos = np.sum(ref * got, axis=1) / (
        np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-9
    )
    print(
        f"对拍：输入 {inp['shape'].tolist()} 输出 {list(got.shape)}；"
        f"与 ONNX 余弦 min {cos.min():.6f} / mean {cos.mean():.6f}，最大绝对差 {np.abs(ref - got).max():.3e}"
    )
    if cos.min() < 0.999:
        raise SystemExit("TFLite 与 ONNX 输出差太多，先别接 Android")
    verify_gpu_delegate(tflite_path)


def verify_gpu_delegate(tflite_path: str) -> None:
    """本地用 LiteRT 的 GPU 加速器试编译一次：挂不上就直接退出。

    复刻的就是真机上 `TfliteTileClassifier.load` 那一步 —— `Interpreter` 挂 delegate 失败时
    抛 `Error applying delegate`。GPU delegate 只接受静态形状，模型里只要有一个"运行时才定形状"
    的张量（典型来源：动态 batch 导出的 ONNX 里 `F.normalize` 的广播链 `Shape→Concat→Expand`），
    整张图都会被拒，所以这道闸必须在出厂前跑。
    """
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    try:
        CompiledModel.from_file(tflite_path, hardware_accel=HardwareAccelerator.GPU | HardwareAccelerator.CPU)
    except Exception as e:  # noqa: BLE001 —— 任何失败都不能放行
        raise SystemExit(
            f"GPU delegate 挂不上这个 tflite（真机上会崩在 Interpreter 构造）：{e}\n"
            f"多半是模型里有运行时才定形状的张量，回去用 export_mobile.py 固定 batch 重导"
        )
    print("GPU delegate 校验: 可挂载（静态形状，真机不会报 Error applying delegate）")


if __name__ == "__main__":
    main()
