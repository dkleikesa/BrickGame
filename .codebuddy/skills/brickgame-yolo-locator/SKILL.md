---
name: brickgame-yolo-locator
description: BrickGame 棋盘定位（board + tile 检测器）的补数据闭环：收图 → 预标注 → 装配 → 人工确认框 → 再装配 → 训练 → 定版校验 → 回灌下轮预标注 → 导出 TFLite 进 APK → 实机回归。当用户说"又攒了一批截图/失败帧，重训定位模型"、"补数据优化 YOLO/定位"、"定位不准/残局切错"、"把新模型打进 APK 对比"时使用。This skill covers the on-device board localizer (board/tile YOLO), not the 2369-class tile classifier (that is brickgame-vision-retrain).
---

# BrickGame 定位模型：补数据闭环

定位只回答三件事：**几行几列、每格在哪、哪格是空的**。认颜色/图案是 2369 类识别器，
另一条链（`brickgame-vision-retrain`），两条链共用截图和 APK，训练互不影响。

工作目录一律 `<repo>/BrickGame-Tool`，Python 一律用仓库内 `.venv-train/bin/python`。

## 九步一览

| # | 步骤 | 命令 | 产物 | 通过判据 |
|---|---|---|---|---|
| 1 | 收图 | adb pull + cp | `samples/` 原图 | 每皮肤×布局 ≥20 帧，≥5 残局 |
| 2 | 预标注 | `prelabel_yolo.py` | `yolo/prelabel/*.txt` | 每帧 1 个 board |
| 3 | 装配① | `make_yolo_data.py` | `yolo/images`+`labels`+`meta.json` | 帧数对得上，新帧出现在标注页 |
| 4 | 标注 | `box_server.py` | `labels/*.txt`+`reviewed.json` | 页面显示"已确认，服务端已记录" |
| 5 | 装配② | `make_yolo_data.py` | 刷新的 `meta.json` | **未确认数 = 0** |
| 6 | 训练 | `train_yolo.py` | `runs/brick/weights/best.pt` | 正常收敛，不漏皮肤 |
| 7 | 闸门 | `eval_gate.py --update` | `runs/brick/metrics.json` | board≥95%、tile 召回≥95%、精确率不掉 3 点 |
| 8 | 回灌 | 无需命令 | 下轮预标注自动用新 `best.pt` | — |
| 9 | 导出上线 | export + gradlew | `brick_yolo.tflite` 进 assets | IO 形状对、logcat 残差 ≤8px |

3 和 5 是同一个脚本，跑两趟。**标注页的帧清单来自 `meta.json`，只有装配会生成它**——
跳过第 3 步，新截图根本不会出现在页面上。

---

## 1. 收图

App 自己留档，别手动截图：

```bash
cd <repo>/BrickGame-Tool
export PATH=$PATH:~/Library/Android/sdk/platform-tools
adb pull /sdcard/Android/data/com.qcmian.brickgame/files/debug-frames ./dumps
cd build-dataset                       # 以下去重、冷备都在这里
cp dumps/low-conf/*.png dumps/scan-fail/*.png samples/
```

| 目录 | 触发 | 内容 |
|---|---|---|
| `low-conf/` | 定位成功但识别没把握 | png + 同名 txt（矩阵/可疑格/pitch/行列） |
| `scan-fail/` | 定位失败 | 只有 png（补数据最缺的就是它） |

按内容 md5 去重（手机时间戳会重）：

```bash
../.venv-train/bin/python - <<'PY'
import hashlib, os
seen = {}
for f in sorted(os.listdir("samples")):
    h = hashlib.md5(open(os.path.join("samples", f), "rb").read()).hexdigest()
    if h in seen: os.remove(os.path.join("samples", f)); print("重复删除", f, "=", seen[h])
    else: seen[h] = f
PY
```

- `samples/` 是**原图唯一来源**：1080×2400 PNG，单张 1~3.6MB，不入库
- 冷备（cwd 在 `build-dataset`）：`./backup_samples.sh`，产物在 `~/BrickGame-backups/`

## 2. 预标注（让模型先猜框）

```bash
cd <repo>/BrickGame-Tool
.venv-train/bin/python build-dataset/prelabel_yolo.py          # 只补没有预标注的帧
.venv-train/bin/python build-dataset/prelabel_yolo.py --all    # 换模型后全量重刷
```

- 权重顺序：`runs/brick/weights/best.pt` → `androidApp/.../brick_yolo.tflite`
- 产物：`yolo/prelabel/<帧名>.txt`（类别 + 中心 xy + 宽高 + 置信度）和同名 `.json`
- 一帧只保留 1 个 board 框（多个按置信度裁掉），否则训练目标自相矛盾
- **不改原图、不产生新图**，模型内部缩到 640 发生在内存

## 3. 装配①（唯一压图的地方）

```bash
cd <repo>/BrickGame-Tool
.venv-train/bin/python build-dataset/make_yolo_data.py
```

做的事：

1. 读 `samples/` 原图 → 缩到长边 **640**（1080×2400 → 288×640）→ `yolo/images/<split>/fXXXXXXXX.png`
2. 框 → `yolo/labels/<split>/fXXXXXXXX.txt`（归一化，标签 `labels/` 优先于 `prelabel/`）
3. 写 `meta.json`（file↔src 映射、rows/cols、reviewed）、`split.json`、`data.yaml`
4. 文件名 = `f` + md5(帧名)[:8]，**不是序号**（序号会让新帧把旧框串位）
5. `split.json` 每次都全量落盘，新帧补登为 train；只有 `--rebalance` 才重抽 val
6. 打印**标注覆盖率报表**：按格数（有 `rows/cols`）和按面积（台账没覆盖的帧）两套，
   列出 <50% 的帧和边长 >50% 的异常 tile 框

压缩只在这里发生：240M → 63M。已装配过的图直接跳过，不重编码（否则 JPG 每跑一次掉一代质量）。

`samples/` 里没有原图的帧：**人工确认过的 label 原地保留并告警**，未确认的移进 `yolo/_trash/`。
不会 `os.remove`（旧版会连确认过的框一起删）。

## 4. 人工确认

```bash
cd BrickGame-Tool/build-dataset
../.venv-train/bin/python box_server.py     # → http://127.0.0.1:8771
```

页面只列未确认帧。底图取 `samples/` **原图**缩到 460 宽（不是 640 训练图，放大看会糊）。

只做两件事：**点蓝色框 = 删误检；拖空白 = 补漏检**。棋盘偏了点「重画棋盘」再拖一个整框。

| 动作 | 落盘行为 |
|---|---|
| 点「确认无误」 | **立即** `POST /save` → `reviewed.json`，回执显示"已确认，服务端已记录" |
| 再点一次 | 从 `reviewed.json` 移除（取消确认） |
| 改了框但没点确认 | 框写入 `labels/`，但**不算确认**，下次仍在列表里 |
| 保存失败 | 按钮回滚 + 明确报错，不静默 |
| 页面加载 | 自动补交 localStorage 里积压的确认（旧版遗留），补完刷新 |

写入是加锁 + 原子替换：连点确认不会丢。顶部显示「已确认 N/总帧数」。

**复查帧**（红框、卡片标题带「⚠ 复查」）来自模型复检：

```bash
.venv-train/bin/python build-dataset/audit_labels.py     # 写 yolo/audit.json
```

它对每帧重跑当前模型，挑出「模型能检出、标签里没有」的帧：默认差 ≥10 块砖，
且**标签未满盘**（`label < rows×cols×95%`）。满盘的帧模型多检出只可能是误检 ——
一格最多一块砖。这些帧即使已确认也会重新出现在页面最前面。

**别用覆盖率判断漏标**：残局帧覆盖率天然只有 1%~5%，模型也只检出 2 块，那是正确标注。

## 5. 装配②（未确认归零）

```bash
.venv-train/bin/python build-dataset/make_yolo_data.py
.venv-train/bin/python -c "import json;m=json.load(open('build-dataset/yolo/meta.json'));print(sum(1 for x in m if not x['reviewed']),'/',len(m),'未确认')"
```

**输出必须是 `0 / N 未确认`**，否则不许进第 6 步。

原因：没确认的帧，框是模型自己猜的。直接训 = 模型学自己的漏检（漏掉的砖被标成"没有砖"），
loss 会降但定位能力不涨。装配只是把 `reviewed.json` 刷进 `meta.json`，不会替你确认。

## 6. 训练

```bash
.venv-train/bin/python train_yolo.py --epochs 80
.venv-train/bin/python train_yolo.py --resume runs/brick/weights/last.pt --epochs 40
```

- **开训前会拦一次**：`meta.json` 里有未确认帧直接退出，逃生门是 `--allow-unreviewed`
  （只用于排查，指标不可信）
- 旧 `best.pt` 自动备份成 `best_<时间>.pt`，不会再被 `exist_ok=True` 静默覆盖
- `yolo11n`，`imgsz=640`，`batch=8`，device 自动选 mps
- 数据 <200 帧**必须 `--resume`**，从头训会让旧皮肤漂掉
- 训练前自动重写 `data.yaml` 的绝对路径，换机器/换克隆目录不用重新装配
- 增强：`degrees=3, translate=0.08, scale=0.35, mosaic=1.0, close_mosaic=10`，
  `fliplr/flipud=0`（砖块图案有方向），`patience=30`
- 产物：`runs/brick/weights/best.pt`

## 7. 定版校验

```bash
.venv-train/bin/python build-dataset/eval_gate.py --update
```

硬门槛：board 命中 ≥95%、tile 召回 ≥95%、精确率不低于基线 3 个点，不过就 exit 1。

- 口径：board IoU>0.7 算命中；tile 同类 IoU>0.5 贪心一对一；召回=TP/真值，精确率=TP/检出
- val 里有未确认帧时 `--update` 会被拒绝（模型给自己判卷，指标必然虚高）
- 基线里存了 `labels_fp`（labels 内容指纹）。**改过标注后旧基线自动作废**：指纹不一致时跳过
  「精确率掉 3 点」这条，只判 board/召回，避免拿另一份真值的数字误杀
- 想让新口径成为基线，重跑一次 `eval_gate.py --update` 即可（会自动写入新指纹）
- 精确率低先算标注覆盖率（已标注格 / rows×cols），60% 覆盖时精确率天然偏低

## 8. 回灌：新模型接管下一轮预标注

没有单独命令。`prelabel_yolo.py` 的权重顺序第一项就是 `runs/brick/weights/best.pt`，
**训练完这一版自动成为下轮预标注用的模型**（没有 `runs/` 时才退回 APK 里的 tflite）。

想重标的范围：

| 需求 | 做法 |
|---|---|
| 只标新截图 | 什么都不用做，第 2 步本来就是增量 |
| 旧帧也想用新模型重标 | `prelabel_yolo.py --all` 重刷 `prelabel/` |
| 只重刷未确认帧 | `make_yolo_data.py --refresh-prelabel` |

两种都不会覆盖人工确认过的 `labels/`（装配时 labels 优先）。

## 9. 导出 + 上线

```bash
.venv-train/bin/python -c "from ultralytics import YOLO; print(YOLO('runs/brick/weights/best.pt').export(format='tflite', imgsz=640, int8=False, nms=False))"
cp runs/brick/weights/best.tflite ../androidApp/src/main/assets/brick_yolo.tflite
```

核对 IO 再进 APK：

```bash
.venv-train/bin/python - <<'PY'
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path='runs/brick/weights/best.tflite'); it.allocate_tensors()
print(it.get_input_details()[0]['shape'], it.get_output_details()[0]['shape'])  # [1,3,640,640] / [1,6,8400]
PY
```

```bash
cd <repo>
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :androidApp:assembleDebug
adb install -r androidApp/build/outputs/apk/debug/androidApp-debug.apk
adb logcat -s YoloBoardDetector
# 期望：YOLO 定位 6x6 pitch=164/164 原点=(49,800) tile 6 个 最大残差=2px
```

**残差 > 8px 就是网格偏了**，别只看行列数对。Gradle 报 `Remote host terminated the handshake`
是 dl.google.com TLS 瞬时问题，原样重试即可。

---

## 铁律

1. **框只来自「预标注 + 人工确认」**，禁止用 `dataset/index.csv` 逐格标签反推（那条路实测把残局算成 191×48）。
2. **未确认帧不许进训练**，第 5 步的未确认数必须为 0。
3. **val 划分一次定终身**（`split.json`），新帧只进 train，只有换代才 `--rebalance`。
4. **`samples/` 是原图唯一来源**，不入库；删了高分辨率就回不来了。
5. **图已按 640 压好**，想提高 `imgsz` 必须改 `make_yolo_data.py` 的 `MAX_SIDE` 再从 `samples/` 重跑。

## 尺寸在链路里怎么变

| 环节 | 尺寸 | 说明 |
|---|---|---|
| 手机 / `samples/` | 1080×2400 | 原图，不压 |
| 标注页显示 | 460 宽 | 原图下采样，1:1 显示 |
| `yolo/images/` | 288×640 | 装配时压，唯一压缩点 |
| 模型输入 | 640×640 | letterbox 补灰边 114 |
| App 推理 | 1080×2400 → 640×640 | 与训练同口径 |

框坐标全程归一化 0~1，所以换尺寸不会错位。

## 契约（改模型/改代码必须同步两端）

| 项 | 值 | 不同步的后果 |
|---|---|---|
| 输入 | `[1,3,640,640]` float32，**NCHW** | 排布错分数掉到 0.003 |
| 归一化 | letterbox + 灰边 114，RGB 0..1 | 框整体缩水/偏移 |
| 输出 | `[1,6,8400]`，4 框 + 2 类分数，无 NMS，坐标 0..1 | 按 640 像素解，格距算成 0.06px |
| 类别 | 0=board，1=tile | — |
| tile 框 | 宽 = **0.88 格**（每边内缩 6%） | 除 0.94 会把 6 列算成 7 列（残局必现） |
| board 框 | 中心点 + 宽高，一帧只有 1 个 | 板框偏半个棋盘 |
| 网格拟合 | 行列各试 ±1，残差 >0.35×pitch 放弃 | 14 行算成 13 行，格线整体错一格 |

## 已踩过的坑

1. **标注页确认按钮曾只写 localStorage**，不落盘，导致反复重标。现已改为点即写 + 服务端回执 + 启动补交。
2. **装配曾直接 `os.remove` 旧 label**，samples 缺帧时连人工确认的框一起删（实测丢 47 个）。现改为归档 `_trash` + 保留告警。
3. **装配曾每次重编码全部图**，JPG 反复压缩。现在目标已 ≤640 就跳过。
4. **标注页曾用 640 缩图当底图**，放大 1.6 倍去框选，tile 边缘糊。现在优先取 `samples/` 原图。
5. **`meta.json` 的 `labels_from` 曾二次装配后被洗白成 `labels`**，看不出哪些是模型猜的。现在按 `reviewed` 判真实来源。
6. **弱标注反推网格已删除**（2026-10-21）：47 帧里 5 帧对不上，2 帧多/少一行，3 帧残局算成 191×48。
7. **15 行关卡最后一整行常被 App 悬浮窗盖住**，真值是空位，别把带黑边的格当样本。
8. **留档文件名带毫秒**，只到秒同秒两帧会互相覆盖。
9. **ZIP 要同时打 `low-conf/` 和 `scan-fail/`**，定位失败帧以前根本导不出来。
10. **覆盖率低 ≠ 漏标**：残局帧本来就只有 2~3 块砖，覆盖率 1% 是正常的。
    判漏标要用 `audit_labels.py` 的模型复检（模型检出比标签多 ≥10 块），
    实测 255 帧里只有 12 帧真漏标，覆盖率告警会误报 70 多帧。
11. **半个棋盘被框成一块砖**：`scanfail_1022-004943` 出现 w=0.97 的 tile 框。
    这种框会把格距撑大一倍，装配会告警边长 >50% 的框。
12. **改过 labels 必须重建基线**，否则 `metrics.json` 里的指标是对旧真值算的，
    闸门比的是两套口径。现在有 `labels_fp` 指纹自动识别。

## 仓库入库策略

| 入库 | 不入库 |
|---|---|
| `yolo/images`、`labels`、`meta.json`、`split.json`、`reviewed.json` | `samples/`、`yolo/prelabel*`、`yolo/_trash`、`data.yaml` |
| `dataset/images/frame`、`dataset/*.csv` | `dataset/deleted/`、`cache/`、`dumps-new/`、`*.bak-*.csv`、`__pycache__` |

`data.yaml` 存绝对路径但已入库忽略，`train_yolo.py` 每次训练前重写，不需要手动维护。

## 环境（新机器）

```bash
cd BrickGame-Tool
uv venv .venv-train --python 3.12
uv pip install --python .venv-train/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  ultralytics 'litert-torch>=0.9.0' 'ai-edge-litert>=2.1.4'
```

缺 `litert_torch` 会直接报 `No module named 'litert_torch'`。没有 `runs/brick/weights/best.pt` 时，
`yolo_grid` 和 `prelabel_yolo` 会自动用 APK 资产里的 `brick_yolo.tflite` 兜底。
