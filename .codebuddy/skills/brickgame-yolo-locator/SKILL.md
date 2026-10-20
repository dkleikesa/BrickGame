---
name: brickgame-yolo-locator
description: BrickGame 棋盘定位（board + tile 检测器）的工业级补数据闭环：App 自动留档 → 导出 → 去重入库 → 模型预标注 → 装配数据集 → 增量训练 → 放行闸门 → 导出 TFLite → 打进 APK → 实机回归。当用户说"又攒了一批截图/失败帧，重训定位模型"、"补数据优化 YOLO/定位"、"定位不准/残局切错"、"把新模型打进 APK 对比"、"OpenCV 和 YOLO 哪个准"时使用。This skill covers the on-device board localizer (board/tile YOLO), not the 2370-class tile classifier (that is brickgame-vision-retrain).
---

# BrickGame 定位模型：补数据闭环

定位只回答三件事：**几行几列、每格在哪、哪格是空的**。识别（2370 类）是另一条链，见
`brickgame-vision-retrain`。两条链共用截图与 APK，但训练互不影响。

工作目录：`<repo>/BrickGame-Tool`。YOLO 相关一律用仓库内 `.venv-train/bin/python`
（torch + ultralytics + tensorflow/litert）；识别模型用 `~/.venvs/brick-train`。

## 0. 采集：App 自己留档，不要手动截图

App 里两个目录，出错就会写：

| 目录（`Android/data/com.qcmian.brickgame/files/debug-frames/`） | 触发条件 | 内容 |
|---|---|---|
| `low-conf/` | 定位成功但识别器对某些格没把握 | `*.png` + 同名 `*.txt`（整盘矩阵、可疑格、pitch、行列） |
| `scan-fail/` | **定位失败**（返回 null） | 只有 `*.png` |

导出（两种都行，ZIP 已包含两个目录）：

```bash
export PATH=$PATH:~/Library/Android/sdk/platform-tools
adb pull /sdcard/Android/data/com.qcmian.brickgame/files/debug-frames ./dumps
# 或 App 主页「分享」按钮，解压后同样是 low-conf/ 与 scan-fail/ 两个子目录
```

**采集判据**：目标是"每种皮肤 × 每种布局 ≥ 20 帧"，其中至少 5 帧是残局（剩 <15 块砖）。
一个皮肤没凑够就不要训——样本进得去，泛化出不来。

## 1. 入库：去重 + 分目录

```bash
cd BrickGame-Tool/build-dataset
cp dumps/low-conf/*.png dumps/scan-fail/*.png samples/     # 同名冲突时保留先到的
```

去重按内容 md5，别按文件名（手机时间戳会重）：

```bash
~/.venvs/brick-train/bin/python - <<'PY'
import hashlib, os
seen = {}
for f in sorted(os.listdir("samples")):
    p = os.path.join("samples", f)
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h in seen:
        os.remove(p); print("重复删除", f, "=", seen[h])
    else:
        seen[h] = f
PY
```

## 2. 预标注：模型先出框，人只删改

```bash
cd BrickGame-Tool
.venv-train/bin/python build-dataset/prelabel_yolo.py            # 只处理 index.csv 里没有的帧
```

产物 `build-dataset/yolo/prelabel/<帧名>.txt`（YOLO 格式，含 board 与 tile）。
**抽查是必须的**：预标注的 tile 框来自当前模型，模型在残局/新皮肤上会漏，漏的框等于告诉下一版"这里不是砖"。
快速抽查：

```bash
.venv-train/bin/python - <<'PY'
from ultralytics import YOLO
YOLO("runs/brick/weights/best.pt").predict("build-dataset/samples", imgsz=640,
      conf=0.25, save=True, project="runs", name="check", exist_ok=True)
PY
open runs/check    # 一张张看框，错得多的那几帧直接删掉 samples 里的图重采
```

## 3. 装配数据集（划分持久化）

```bash
.venv-train/bin/python build-dataset/make_yolo_data.py
```

- 有 `dataset/index.csv` 逐格标注的帧 → **弱标注**（tile 框由格子和类别反推，最准）
- 没有的帧 → 用第 2 步的**预标注**
- `yolo/split.json` 锁定 train/val：**新帧一律进 train**，val 永远不重新抽
  （重抽 = 指标失去可比性 + 悄悄把评测集喂进训练）。真要重划才加 `--rebalance`。

## 4. 增量训练

```bash
.venv-train/bin/python train_yolo.py --epochs 80                        # 从头训
.venv-train/bin/python train_yolo.py --resume runs/brick/weights/last.pt --epochs 40   # 接着练
```

数据少（<200 帧）时用 `--resume`，别从头训——36 帧从头训出来的模型对旧皮肤会漂。

## 5. 放行闸门（不许跳过）

```bash
.venv-train/bin/python build-dataset/eval_gate.py --update
```

硬门槛：board 命中 ≥ 95%、tile 召回 ≥ 95%、精确率不低于基线 3 个点。不达标就 exit 1。
基线存在 `runs/brick/metrics.json`，只有 `--update` 或 `--force` 才覆盖。
**精确率别单看**：标注不完整时它会虚低（没标的砖被算成误检），先看逐帧明细再决定。

## 6. 导出 TFLite 并进 assets

```bash
cd BrickGame-Tool
.venv-train/bin/python -c "from ultralytics import YOLO; print(YOLO('runs/brick/weights/best.pt').export(format='tflite', imgsz=640, int8=False, nms=False))"
cp runs/brick/weights/best.tflite ../androidApp/src/main/assets/brick_yolo.tflite
```

导出依赖 `litert-torch` + `ai-edge-litert`（ultralytics 8.4 起走 LiteRT 后端）。缺了会报
`No module named 'litert_torch'`，用清华源补：

```bash
uv pip install --python .venv-train/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'litert-torch>=0.9.0' 'ai-edge-litert>=2.1.4' pip
```

## 7. 构建 + 实机回归

```bash
cd <repo>
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :androidApp:assembleDebug
adb install -r androidApp/build/outputs/apk/debug/androidApp-debug.apk
```

控制条最左边是 `YOLO` 开关（选择存 `SharedPreferences` 的 `board_locator`），切完点「刷新」。
验证看 logcat：

```bash
adb logcat -s YoloBoardDetector OpenCvBoardScanner HintSession
# 期望：YOLO 定位 6x6 pitch=164/164 原点=(49,800) tile 6 个 最大残差=2px
```

**残差 > 8px 就是网格偏了**，不要只看行列数对不对。

## 关键契约（改模型/改代码必须同步）

| 项 | 值 | 不同步的后果 |
|---|---|---|
| 输入 | `[1,3,640,640]` float32，**NCHW** | 数据错位，分数掉到 0.003 |
| 归一化 | letterbox + 灰边 114，值域 0..1（RGB） | 框整体缩水/偏移 |
| 输出 | `[1,6,8400]`，4 框 + 2 类分数，**坐标归一化 0..1** | 按 640 像素解会把格距算成 0.06px |
| 类别 | 0=board，1=tile | 棋盘当砖块 |
| tile 框 | 每边内缩 6% → 宽 = **0.88 格** | 除 0.94 会把 6 列算成 7 列（残局必现） |
| board 框 | **中心点 + 宽高**（不是左上角） | 板框整体偏半个棋盘，IoU 0.45 |

## 已踩过的坑

1. **别用 board 框直接铺网格**：它是整片区域的回归，边缘常差 15~20px。用 tile 中心做一次最小二乘
   （`fitAxis`），残差能从 20px 降到 2px。
2. **标注不完整会让指标失真**：`index.csv` 每帧只覆盖 24~135 格（棋盘 90~150 格），没标的砖
   在训练里是"背景"。精确率 60%~70% 先怀疑标注，别急着调阈值。
3. **残局是两条路的分水岭**：同一张只剩 6 块砖的图，OpenCV 给 7 行、YOLO 给 6 行。
   回归测试必须包含残局帧，否则等于没测。
4. **留档文件名带毫秒**（`MMdd-HHmmss-SSS`）：只到秒时同一秒两帧会互相覆盖。
5. **ZIP 现在同时打 `low-conf/` 和 `scan-fail/`**：定位失败帧以前根本导不出来，补数据时缺的就是它。
