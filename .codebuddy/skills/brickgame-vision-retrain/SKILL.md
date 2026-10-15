---
name: brickgame-vision-retrain
description: 用新增游戏截图（含 App 低置信度留档 lowconf_*.png/.txt）补训 BrickGame 的物品识别模型：截图入库 → 人工标注 → 按类去重 → 训练 → 导出 TFLite 进 APK。当用户说"我又加了低置信度截图/新截图，加训练数据重训模型"、"重新训练识别模型"、"更新 V2/small/large 模型"、"认不准/低置信度太多"、"把模型导进 APK"、"标注页/去重页"时使用。This skill should be used when retraining or exporting the on-device tile classifier of the BrickGame project.
---

# BrickGame 识别模型重训

各步骤的**参数含义与原理见对应脚本的文件头**（都写得很细，别在这里重复）。本文件只给顺序、
命令和非显然的坑。工作目录一律 `<repo>/BrickGame-Tool`。

环境：训练/数据集用 `~/.venvs/brick-train/bin/python`（torch/cv2/onnx），
ONNX→TFLite 用 `~/.venvs/brick-tflite/bin/python`（onnx2tf）。

## ① 入库：留档/截图 → dataset

```bash
cd BrickGame-Tool/build-dataset
cp <留档目录>/*.png samples/ && ~/.venvs/brick-train/bin/python build_dataset.py   # 增量切图+自动识别
~/.venvs/brick-train/bin/python import_app_labels.py --dump <留档目录> --inplace    # 必须在上一条之后
~/.venvs/brick-train/bin/python visualize_groups.py                                 # 重出 annotate.html
```

## ② 标注、③ 按类去重 —— 必须用户自己点，Agent 只起服务

```bash
cd BrickGame-Tool/build-dataset
nohup ~/.venvs/brick-train/bin/python annotate_server.py >/tmp/annotate.log 2>&1 &   # :8770
nohup ~/.venvs/brick-train/bin/python review_server.py   >/tmp/review.log   2>&1 &   # :8765
open http://127.0.0.1:8770/build-dataset/annotate.html && open http://127.0.0.1:8765/review.html
grep -c ',pending,' dataset/frames.csv          # 0 = 标完；去重删完还要 embed_features.py 重刷特征缓存
```

## ④ 训练

| 变体 | `--backbone` | `--out` | `--epochs` |
|------|--------------|---------|-----------|
| `v2`（线上默认） | mobilenet_v2 | out/clf_v2 | 20 |
| `small` | mobilenet_v3_small | out/clf_small | 30 |
| `large` | mobilenet_v3_large | out/clf_real | 30 |

```bash
cd BrickGame-Tool && ~/.venvs/brick-train/bin/python train_classifier.py \
  --data real --img 160 --backbone mobilenet_v2 --epochs 20 --out out/clf_v2
```
`--img 160` 必须显式给（默认 224，会跟线上输入尺寸不一致）。

## ⑤ 导出：ONNX → TFLite → assets

```bash
~/.venvs/brick-train/bin/python export_mobile.py  --ckpt out/clf_v2/best.pt --variant v2 \
  --out out/v2_pkg --install-assets ../androidApp/src/main/assets
~/.venvs/brick-tflite/bin/python export_tflite.py --variant v2 --onnx out/v2_pkg/brick_embedder_v2.onnx
```
文件名后缀以 `variants.py` 为准（与 Android 的 `ModelVariant.kt` 是同一约定）；两道数值对拍
任一不过就停下修，别绕。

## 几个不写就会错的点

1. **顺序**：`build_dataset.py` 必须在 `import_app_labels.py` 之前 —— `auto_classify` 只保护
   `cls_src=manual`，反过来留档标签会被自动识别的结果冲掉。
2. **标定谐波**：`detect_grid` 的自相关会取到 2/3 倍谐波，把 15×10 的棋盘切成 5×10。切出来的
   行列数若与留档头部的 `RxC` 不一致，先查 `tile_recognizer.reconcile_pitches`，别急着标数据。
3. **15 行关卡的最后一整行（r14）会被 App 自己的悬浮窗盖住**（"正在识别盘面"+网格/关闭那块），
   真值是空位。在 App 改成"截图时连悬浮窗一起藏"之前，不要把这些带黑边的格当训练样本收，
   会污染 class 0 原型；标空位用界面里的「标为空位」。
4. **评测别自欺**：看 `out/clf_<v>/metrics.json` 的 `real.top1` 与 `dist_wrong_p5`（后者决定线上
   留档阈值，塌了等于阈值被拉低），再用 `compare_models.py` 在真实留档截图上跟上一版对比。
   刚训进去的帧不能当评测集。
