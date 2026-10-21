---
name: brickgame-yolo-locator
description: BrickGame 棋盘定位（board + tile 检测器）的补数据闭环：App 留档 → 去重入库 → 模型预标注 → 人工确认框 → 装配 → 增量训练 → 放行闸门 → 导出 TFLite → 打进 APK → 实机回归。当用户说"又攒了一批截图/失败帧，重训定位模型"、"补数据优化 YOLO/定位"、"定位不准/残局切错"、"把新模型打进 APK 对比"时使用。This skill covers the on-device board localizer (board/tile YOLO), not the 2369-class tile classifier (that is brickgame-vision-retrain).
---

# BrickGame 定位模型：补数据闭环

定位只回答三件事：**几行几列、每格在哪、哪格是空的**。识别（2369 类）是另一条链，见
`brickgame-vision-retrain`。两条链共用截图与 APK，但训练互不影响。

工作目录一律 `<repo>/BrickGame-Tool`，YOLO 相关一律用仓库内 `.venv-train/bin/python`。

## 铁律（违反就是白训一轮）

1. **标注唯一来源 = 模型预标注 + 人工确认**。禁止用 `dataset/index.csv` 的逐格标签反推框。
2. **OpenCV 定位已经从代码里删干净**（Python 和 Android 都没有了）。残局/空盘实测会被算成
   191×48、15×10 算成 16×10；Python 侧定位只走 `build-dataset/yolo_grid.py`，
   端侧只走 `ml/YoloBoardDetector`，判不出就跳过这一帧。
3. **没有人工确认的帧不许进训练**。`make_yolo_data.py` 会打印未确认帧数，必须为 0。
4. **val 划分一次定终身**（`yolo/split.json`）。新帧只进 train，只有换代才 `--rebalance`。

## 环境（新机器 / venv 丢了先做）

```bash
cd BrickGame-Tool
uv venv .venv-train --python 3.12
uv pip install --python .venv-train/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  ultralytics 'litert-torch>=0.9.0' 'ai-edge-litert>=2.1.4'
```

- 导出 TFLite 走 LiteRT：缺 `litert_torch` 会直接报 `No module named 'litert_torch'`。
- 识别链是两个独立 venv（`~/.venvs/brick-train` 训练、`~/.venvs/brick-tflite` 转 TFLite），
  转 TFLite 需要 `tensorflow + onnx2tf + ai-edge-litert`；`export_mobile.py` 在 torch≥2.6 上
  还需要 `onnxscript`（requirements.txt 没写，单独装）。
- 没有 `runs/brick/weights/best.pt` 时，`yolo_grid` 会自动用
  `androidApp/src/main/assets/brick_yolo.tflite` 兜底，所以新克隆也能切新截图。

## 0. 采集：App 自己留档，不要手动截图

| 目录（`Android/data/com.qcmian.brickgame/files/debug-frames/`） | 触发条件 | 内容 |
|---|---|---|
| `low-conf/` | 定位成功但识别器没把握 | `*.png` + 同名 `*.txt`（整盘矩阵、可疑格、pitch、行列） |
| `scan-fail/` | **定位失败**（返回 null） | 只有 `*.png` |

```bash
export PATH=$PATH:~/Library/Android/sdk/platform-tools
adb pull /sdcard/Android/data/com.qcmian.brickgame/files/debug-frames ./dumps
```

**采集判据**：每种皮肤 × 每种布局 ≥ 20 帧，其中至少 5 帧残局（剩 <15 块砖）。
一个皮肤没凑够就不要训 —— 样本进得去，泛化出不来。

## 1. 入库：去重

```bash
cd BrickGame-Tool/build-dataset
cp dumps/low-conf/*.png dumps/scan-fail/*.png samples/
```

去重按内容 md5（手机时间戳会重）：

```bash
~/.venvs/brick-train/bin/python - <<'PY'
import hashlib, os
seen = {}
for f in sorted(os.listdir("samples")):
    h = hashlib.md5(open(os.path.join("samples", f), "rb").read()).hexdigest()
    if h in seen: os.remove(os.path.join("samples", f)); print("重复删除", f, "=", seen[h])
    else: seen[h] = f
PY
```

分类标注是另一条链（`build_dataset.py` → `annotate.html`），补进来的帧也要过一遍；
它现在**只走 YOLO 定位**，判不出就跳过，不会退回 OpenCV。

## 2. 预标注：模型先出框

```bash
cd BrickGame-Tool
.venv-train/bin/python build-dataset/prelabel_yolo.py          # 只补没有预标注的帧
.venv-train/bin/python build-dataset/prelabel_yolo.py --all    # 换了模型，全部重刷
```

产物 `build-dataset/yolo/prelabel/<帧名>.txt`。脚本保证**一帧只有一个 board 框**
（多出来的按置信度裁掉），tile 框每边内缩 6%（宽 = 0.88 格）。

## 3. 人工确认（不许跳过）

```bash
cd BrickGame-Tool/build-dataset
../.venv-train/bin/python box_server.py      # :8771，自动开浏览器
```

- 页面**只列还没确认的帧**；保存后该帧写入 `yolo/reviewed.json`，刷新即消失。
- **蓝色 = 砖块**（点一下删）、**空白拖拽 = 补漏检**。
- **棋盘框偏了**：点卡片上的「重画棋盘」→ 在图上拖出整个棋盘（青色框）→ 保存。
  底图上烤进去的白框擦不掉，以青色为准；labels 里一帧只会保留一个 board。
- 保存幂等：首次保存前备份 `.orig`，之后从备份重算再叠加，反复保存不会重复加框。
- 只做两件事：删误检、补漏检。漏检在训练里等于"这里没有砖"，会被固化。

确认进度看 `make_yolo_data.py` 的告警，或：

```bash
python3 -c "import json;m=json.load(open('yolo/meta.json'));print(sum(1 for x in m if not x['reviewed']),'/',len(m),'未确认')"
```

## 4. 装配数据集

```bash
cd BrickGame-Tool
.venv-train/bin/python build-dataset/make_yolo_data.py
```

- **只搬框，不生成框**：`yolo/labels/` 优先（人工确认过的），没有才用 `prelabel/`。
- 文件名是 `f + md5(帧名)[:8]`，**不是序号** —— 加新截图不会让旧 labels 串帧。
- 已存在的 labels 一律不覆盖（人工改过的保住）；`--refresh-prelabel` 只覆盖未确认帧。
- 没有任何框的帧会跳过并打印原因（`无预标注` / `缺 board 框` / `没有 tile 框`）。

## 5. 增量训练

```bash
.venv-train/bin/python train_yolo.py --epochs 80
.venv-train/bin/python train_yolo.py --resume runs/brick/weights/last.pt --epochs 40
```

数据少（<200 帧）**必须 `--resume`**，从头训会让旧皮肤漂掉。新数据进 train 后
early-stop 通常在第 30~60 轮就停，属正常。

## 6. 放行闸门（不许跳过）

```bash
.venv-train/bin/python build-dataset/eval_gate.py --update
```

硬门槛：board 命中 ≥ 95%、tile 召回 ≥ 95%、精确率不低于基线 3 个点，不过就 exit 1。
基线在 `runs/brick/metrics.json`，只有 `--update`/`--force` 才覆盖。

闸门会先检查 val 帧是否人工确认过：**val 里还是预标注时指标必然虚高**（模型给自己的框
打分，实测三项全 100%），这时 `--update` 会被拒绝，必须先把 val 过一遍。
换了标注口径也要把旧基线挪走，否则精确率会跟旧口径比：

```bash
mv runs/brick/metrics.json runs/brick/metrics_weak_labels.json
```

**精确率别单看**：标注不完整时它会虚低（没标的砖算误检）。先算覆盖率：
`已标注格 / (rows×cols)`。实测 60% 覆盖时精确率只有 73%，召回仍 100% —— 这是标注问题，
不是模型问题。要提精确率就回去补第 3 步，别调阈值。

## 7. 导出 TFLite 进 assets

```bash
.venv-train/bin/python -c "from ultralytics import YOLO; print(YOLO('runs/brick/weights/best.pt').export(format='tflite', imgsz=640, int8=False, nms=False))"
cp runs/brick/weights/best.tflite ../androidApp/src/main/assets/brick_yolo.tflite
```

导出后先核对 IO，再进 APK：

```bash
.venv-train/bin/python - <<'PY'
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path='runs/brick/weights/best.tflite'); it.allocate_tensors()
print(it.get_input_details()[0]['shape'], it.get_output_details()[0]['shape'])  # 期望 [1,3,640,640] / [1,6,8400]
PY
```

## 8. 构建 + 实机回归

```bash
cd <repo>
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :androidApp:assembleDebug
adb install -r androidApp/build/outputs/apk/debug/androidApp-debug.apk
```

Gradle 报 `Remote host terminated the handshake`（dl.google.com TLS）先原样重试一次，
通常是瞬时网络问题，不是配置错。

定位只有 YOLO 一条路（OpenCV 的实现、开关、依赖都已删除，APK 里不再有 `libopencv_java4.so`）。
判不出可信网格时这一帧直接跳过，不会退回投影法。

```bash
adb logcat -s YoloBoardDetector OpenCvBoardScanner HintSession
# 期望：YOLO 定位 6x6 pitch=164/164 原点=(49,800) tile 6 个 最大残差=2px
```

**残差 > 8px 就是网格偏了**，不要只看行列数对不对。

## 契约（改模型/改代码必须同步）

| 项 | 值 | 不同步的后果 |
|---|---|---|
| 输入 | `[1,3,640,640]` float32，**NCHW** | 数据错位，分数掉到 0.003 |
| 归一化 | letterbox + 灰边 114，值域 0..1（RGB） | 框整体缩水/偏移 |
| 输出 | `[1,6,8400]`，4 框 + 2 类分数，**坐标归一化 0..1** | 按 640 像素解会把格距算成 0.06px |
| 类别 | 0=board，1=tile | 棋盘当砖块 |
| tile 框 | 每边内缩 6% → 宽 = **0.88 格** | 除 0.94 会把 6 列算成 7 列（残局必现） |
| board 框 | **中心点 + 宽高**，一帧只有一个 | 板框偏半个棋盘，IoU 0.45 |

## 已踩过的坑

1. **弱标注 + OpenCV 反推已删除**（2026-10-21）。实测 47 帧里 5 帧网格对不上：2 帧多/少一行，
   3 帧残局被算成 191×48。整轮训练等于部分白做，别再走这条路。
2. **标注覆盖率才是精度上限**：`index.csv` 只覆盖 3047/5050 = 60% 的格子，早期帧甚至只有
   6%（8/140）。精确率低先查这个。
3. **残局/空盘是分水岭**：同一张只剩 6 块砖的图，投影法给 7 行、YOLO 给 6 行。YOLO 判不出
   （砖 < 2）就跳过，**不要退回 OpenCV**。
4. **15 行关卡的最后一整行（r14）会被 App 悬浮窗盖住**，真值是空位；别把带黑边的格当样本。
5. **留档文件名带毫秒**（`MMdd-HHmmss-SSS`）：只到秒同一秒两帧会互相覆盖。
6. **ZIP 同时打 `low-conf/` 和 `scan-fail/`**：定位失败帧以前根本导不出来，补数据缺的就是它。
7. **labels 文件名绝不能按序号**：中途加帧会让编号整体后移，人工改过的框串到别的帧上。
8. **端侧裁格已换纯 Kotlin**（`ml/CellCropper` + `vision/CellResampler`）：区域平均/双线性
   替代 `INTER_AREA/INTER_LINEAR`，逐像素差 ≤1 级灰阶、embedding 余弦 ≥0.9993（150 格实测），
   远小于命中阈值（0.0385）。性能：整帧只 `getPixels` 一次 + 12/8 位定点权重表，
   JVM 单线程 135 格约 14ms/帧（旧浮点实现 23ms）；`CellResamplerTest` 里有对拍和基准，
   改缩放核必须重跑。
