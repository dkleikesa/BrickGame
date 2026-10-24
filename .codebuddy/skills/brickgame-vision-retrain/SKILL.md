---
name: brickgame-vision-retrain
description: 用新增游戏截图（含 App 低置信度留档 lowconf_*.png/.txt）补训 BrickGame 的物品识别模型：截图入库 → 上一版 V2 预标注 → 人工确认 → 按类去重 → 训练 → 放行闸门 → 导出 TFLite 进 APK。当用户说"我又加了低置信度截图/新截图，加训练数据重训模型"、"重新训练识别模型"、"更新 V2 模型"、"认不准/低置信度太多"、"把模型导进 APK"、"标注页/去重页"时使用。This skill should be used when retraining or exporting the on-device tile classifier of the BrickGame project.
---

# BrickGame 识别模型重训

识别链**只有一个模型：MobileNetV2（V2）**。large / small 的资产与训练入口已删除
（精度和端上性能都不如 V2）。各步骤参数见对应脚本文件头，这里只给顺序、命令和非显然的坑。

工作目录一律 `<repo>/BrickGame-Tool`。环境：

| 用途 | Python | 依赖 |
|---|---|---|
| 数据/训练/闸门 | `~/.venvs/brick-train/bin/python` | torch、cv2、onnx、ai-edge-litert |
| ONNX→TFLite | `~/.venvs/brick-tflite/bin/python` | tensorflow、onnx2tf、ai-edge-litert |

V2 预标注要读 APK 里的 tflite，缺了就装：
`uv pip install --python ~/.venvs/brick-train/bin/python ai-edge-litert`

## ① 入库：留档/截图 → dataset

```bash
cd BrickGame-Tool/build-dataset
cp <留档目录>/*.png samples/ && ~/.venvs/brick-train/bin/python build_dataset.py   # 增量切图 + 自动识别
~/.venvs/brick-train/bin/python import_app_labels.py --dump <留档目录> --inplace    # 必须在上一条之后
~/.venvs/brick-train/bin/python visualize_groups.py                                 # 重出 annotate.html
```

**新增帧只保留"模型拿不准"的格子**：`build_dataset.py` 识别完会把 top-1 距离
< `--highconf-thr`（默认 0.30）的格子丢掉（记进 `deleted.csv`，reason=highconf-skip）。
理由是现有数据已经覆盖高置信样本，重复收只会拖慢训练、加重类别不平衡。
想全收加 `--keep-highconf`；阈值调 `--highconf-thr 0` 等于关闭。只作用于本次新增帧，
人工改过的（manual）和 App 上报的（app）一律不动。

`build_dataset.py` 里的自动识别**默认已经换成上一版 V2**（`assets/brick_embedder_v2.tflite`），
与端上同一套预处理、同一个度量空间；没有 tflite 才退回通用骨干（v3-large 特征）或像素特征。
显式指定用 `auto_classify.py --feature v2|emb|pixel`。

## ② 标注、③ 按类去重 —— 必须用户自己点，Agent 只起服务

```bash
cd BrickGame-Tool/build-dataset
nohup ~/.venvs/brick-train/bin/python annotate_server.py >/tmp/annotate.log 2>&1 &   # :8770
nohup ~/.venvs/brick-train/bin/python review_server.py   >/tmp/review.log   2>&1 &   # :8765
open http://127.0.0.1:8770/build-dataset/annotate.html && open http://127.0.0.1:8765/review.html
grep -c ',pending,' dataset/frames.csv          # 0 = 标完；去重删完还要 embed_features.py 重刷特征缓存
```

服务要用后台任务托管启动，别用 `(cmd &)`（父 shell 退出会把子进程带走）。

## ④ 训练（只有 V2 一个变体）

```bash
cd BrickGame-Tool && ~/.venvs/brick-train/bin/python train_classifier.py \
  --data real --img 160 --backbone mobilenet_v2 --epochs 20 --out out/clf_v2
```

`--img 160` 必须显式给（默认 224，和端上输入尺寸不一致就白训）。

## ⑤ 放行闸门（不许跳过）

```bash
~/.venvs/brick-train/bin/python eval_clf_gate.py --new out/clf_v2 --update
```

- 闸门会**在当前数据集上重跑两个模型的 `--eval-only`**，保证同一批查询格；直接比旧的
  `metrics.json` 等于拿两把尺子量（数据集已经加过帧/改过标）。
- 硬规则：`top1 ≥ 0.95` 且不比基线低超过 0.3 个点；`hit_dist_p95` 不比基线高超过 0.005；
  `wrong_dist_p5` 不比基线低超过 0.01。不过就 exit 1，`gate.json` 写 `pass=false`。
- 基线是**滚动**的：固定放在 `out/clf_baseline/`（best.pt + baseline.json）。`--update` 通过后
  会把这一版拷进去，下一轮自动与它比。**不要**再传 `--base out/clf_real` —— 那是 10 月的旧目录，
  会让基线永远停在那一版（旧实现把 baseline.json 写进 `<new>/`，配合写死的 --base 就是这个结果）。
- `--update` 只会在通过时更新基线；没通过连 best.pt 都不会写进去。
- 留档阈值 `dumpSimilarity` 来自 `dist_wrong_p5`，这一项塌了等于线上误留档暴增。

## ⑥ 导出：ONNX → TFLite → assets

```bash
~/.venvs/brick-train/bin/python export_mobile.py --ckpt out/clf_v2/best.pt --variant v2 \
  --out out/v2_pkg --install-assets ../androidApp/src/main/assets
~/.venvs/brick-tflite/bin/python export_tflite.py --variant v2 --onnx out/v2_pkg/brick_embedder_v2.onnx
```

- `export_mobile.py` 装 assets 前会核对 `out/clf_v2/gate.json`：`pass` 必须为 true，
  且 `ckpt_md5` 必须与当前 `best.pt` 一致 —— 重训后忘了过闸门会被直接拒掉（排障才加 `--force`）。
- 两道数值对拍（ONNX↔PyTorch、TFLite↔ONNX）任一不过就停下修，别绕。

## ⑦ 构建 + 实机

```bash
cd <repo>
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :androidApp:assembleDebug
adb install -r androidApp/build/outputs/apk/debug/androidApp-debug.apk
```

看 logcat 里的识别距离与「裁格」耗时；留档量突然变多先查阈值而不是模型。

## 几个不写就会错的点

1. **顺序**：`build_dataset.py` 必须在 `import_app_labels.py` 之前 —— `auto_classify` 只保护
   `cls_src=manual`，反过来留档标签会被自动识别的结果冲掉。
2. **裁格只走 YOLO**：`build_dataset.py` 已经没有 OpenCV 兜底（那条路径已删除）。切出来的
   行列数若与留档头部的 `RxC` 不一致，就是定位模型没认对盘面 —— 这一帧先跳过，补进
   `brickgame-yolo-locator` 重训定位模型，**不要**改用投影法硬切。
3. **15 行关卡的最后一整行（r14）会被 App 悬浮窗盖住**（"正在识别盘面"+网格/关闭那块），
   真值是空位。别把带黑边的格当训练样本收，会污染 class 0 原型；标空位用界面里的「标为空位」。
4. **预标注别混用两套模型**：默认就是上一版 V2；只有在 V2 tflite 缺失时才会退回
   v3-large 特征缓存。两套的 `cls_dist` 量纲不同，混着看阈值会误判。
5. **基线必须滚动**：每轮都用 `--new <新目录> --update`，基线在 `out/clf_baseline` 自己前移；
   不要每轮复制同样的 `--base` 命令，否则永远比同一版。`out/` 不入库，基线丢了就重新
   `--update` 一次（首次会只查绝对下限）。
6. **闸门与权重绑定**：`gate.json` 记的是 `best.pt` 的 md5。重新训练后必须重跑闸门，
   旧 gate 不会自动失效成"通过"——导出脚本会拦住。
7. **刚训进去的帧不能当评测集**：评估用 `--eval-only` 的类划分，但人工刚改过的帧仍在训练集里，
   真正的外部检验靠 `compare_models.py` 在留档图上逐格对比。
