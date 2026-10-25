package com.qcmian.brickgame.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.CellRect
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YOLO 定位：一次推理同时出 board 框和 tile 框，再用"棋盘框 + 砖块尺寸"铺网格。
 *
 * 和 OpenCV 那条链的区别：残局时棋盘上的周期结构几乎为零，投影/自相关的依据全塌了；
 * 但"这里有一块砖""这整片是棋盘"是局部外观特征，检出 3 块砖就够反推格距，板框还能
 * 独立给出范围 —— 两件事都不依赖砖块密度。
 *
 * 模型契约（导出时定死，改模型必须同步改这里）：
 *   · 输入  [1, 3, H, W] float32，**NCHW**（LiteRT 导出的默认排布，不是 NHWC）
 *   · 输出  [1, 6, N]，4 个框 + 2 类分数（0=board，1=tile），anchor-free、不含 NMS
 *           N = (H/8·W/8) + (H/16·W/16) + (H/32·W/32)：1024×480 → 10080
 *   · 训练用 letterbox（等比缩放 + 灰边 114），推理必须同样处理，否则框会整体缩水
 *
 * H/W 和 N **不写死**：构造时从 .tflite 的张量形状读。换尺寸（长边 ≥1024）只要
 * 重新导出模型、把 brick_yolo.tflite 拷进 assets，这里不用改。
 *
 * 为什么是 1024 高：1080×2400 的截图缩到 640 后，一格砖只剩 ~20px，到了 stride-8
 * 的 P3 特征图上就 2~3 个像素，小砖/边缘/被悬浮窗压住的部分直接学不出来。1024 高
 * 下一格 ~32px，才够检测头分辨。
 * 为什么宽只有 480：竖屏内容按 1024 高缩放后只有 461 宽，方图 1024 有 55% 是灰边；
 * 裁到 480 宽后像素量与 640 方图相当，算力省一半，内容分辨率完全不变。
 */
class YoloBoardDetector(context: Context) : Closeable {

    companion object {
        private const val TAG = "YoloBoardDetector"
        // 下面三个是手抄自 BrickGame-Tool/yolo_config.py 的常量（端侧不读 Python 配置）。
        // 改任何一条都要两边同时改，否则预标注/评估和线上就不是同一口径。
        private const val PAD = 114
        private const val CONF = 0.25f
        private const val NMS_IOU = 0.45f

        /**
         * 训练时 tile 框**每边**内缩 6%（见 yolo_config.TILE_INSET），
         * 所以框宽 = 0.88 格。这里除回去得到格距。
         *
         * 曾经写成 0.94（只算了一边），残局时 board 宽 1002 / 154 = 6.5 → 取整成 7 列，
         * 满盘时却因为框好巧不巧偏大而凑对 —— 属于错误一直没暴露的那类 bug。
         */
        private const val TILE_INSET = 0.88f

        private const val CLS_BOARD = 0
        private const val CLS_TILE = 1
        private const val NUM_CLASSES = 2
        /** 长边下限：低于它的 .tflite 是 640 时代旧模型，直接拒绝，避免静默降级。 */
        private const val MIN_LONG_SIDE = 1024
    }

    private class Box(
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val conf: Float, val cls: Int,
    ) {
        val left get() = cx - w / 2f
        val top get() = cy - h / 2f
        val right get() = cx + w / 2f
        val bottom get() = cy + h / 2f

        /**
         * 点是否落在这个框里（[margin] 是按框宽高的外扩比例）。
         *
         * 用来把"棋盘外"的 tile 误检挡掉：实测 0.52 分的框会出现在关卡标题上
         * （棋盘上方 260px），它的中心会把网格的最小二乘和残差整个带偏。
         */
        fun contains(x: Float, y: Float, margin: Float = 0f): Boolean {
            val mx = w * margin
            val my = h * margin
            return x >= left - mx && x <= right + mx && y >= top - my && y <= bottom + my
        }
    }

    /** 有 GPU 就用 GPU：YOLO 全是标准卷积，delegate 支持度比识别模型高得多。 */
    private val gpuDelegate: GpuDelegate? = runCatching { GpuDelegate() }
        .onFailure { Log.w(TAG, "GPU delegate 初始化失败，回退 CPU", it) }
        .getOrNull()

    /**
     * 解释器构造：GPU delegate 挂不上时**必须**回退纯 CPU。
     *
     * delegate 在部分老机型/驱动上会编译失败或 OOM（换输入尺寸后尤其明显），而这条
     * 链只剩 YOLO 一个定位器（OpenCV 已删），构造一失败就整帧定位不了。CPU 慢，但
     * 至少还能用，也能把 `Error applying delegate` 留在 logcat 里而不是静默变成
     * "定位失败"。
     */
    private val interp: Interpreter = runCatching {
        Interpreter(
            loadModel(context, "brick_yolo.tflite"),
            Interpreter.Options().apply {
                gpuDelegate?.let { addDelegate(it) }
                setNumThreads(4)
            },
        )
    }.onFailure {
        Log.w(TAG, "GPU delegate 挂不上 YOLO 模型，回退 CPU", it)
    }.getOrNull() ?: Interpreter(
        loadModel(context, "brick_yolo.tflite"),
        Interpreter.Options().apply { setNumThreads(4) },
    )

    /**
     * 输入/输出形状只在构造时读一次。`allocateTensors()` 顺带把解释器准备好，
     * 后面 `run` 不再重新分配（每帧重新分配是几 MB 的抖动）。
     */
    private val io: Pair<IntArray, IntArray> = interp.allocateTensors().let {
        interp.getInputTensor(0).shape() to interp.getOutputTensor(0).shape()
    }
    private val inShape: IntArray = io.first
    private val outShape: IntArray = io.second

    /** Ultralytics 导出的 TFLite 是 NCHW；万一哪天换成 NHWC 导出也认。 */
    private val nchw: Boolean = inShape.size == 4 && inShape[1] == 3

    /** 模型输入尺寸（高, 宽）。矩形输入，不是正方形。 */
    private val dims: Pair<Int, Int> = run {
        require(inShape.size == 4) {
            "YOLO 输入应为 4 维 [1,3,H,W]，实际 ${inShape.contentToString()}"
        }
        if (nchw) {
            require(inShape[1] == 3) { "NCHW 输入第二维应为 3：${inShape.contentToString()}" }
            inShape[2] to inShape[3]
        } else {
            require(inShape[3] == 3) { "NHWC 输入最后一维应为 3：${inShape.contentToString()}" }
            inShape[1] to inShape[2]
        }
    }
    private val inH: Int = dims.first
    private val inW: Int = dims.second

    /** 检测头 anchor 数：三层特征图之和（1024×480 → 10080）。 */
    private val anchors: Int = run {
        require(outShape.size == 3 && outShape[0] == 1) {
            "YOLO 输出应为 [1,${4 + NUM_CLASSES},N]，实际 ${outShape.contentToString()}；" +
                "导出时别加 NMS（nms=False）"
        }
        require(outShape[1] == 4 + NUM_CLASSES) {
            "YOLO 输出第二维应为 ${4 + NUM_CLASSES}（4 框 + $NUM_CLASSES 类），" +
                "实际 ${outShape.contentToString()}"
        }
        val expected = (inH / 8) * (inW / 8) + (inH / 16) * (inW / 16) + (inH / 32) * (inW / 32)
        require(outShape[2] == expected) {
            "输出 anchor 数 ${outShape[2]} 与输入 ${inW}x${inH} 不符（应为 $expected）"
        }
        outShape[2]
    }

    private val inputBuf: ByteBuffer =
        ByteBuffer.allocateDirect(inW * inH * 3 * 4).order(ByteOrder.nativeOrder())

    private val output = Array(1) { Array(4 + NUM_CLASSES) { FloatArray(anchors) } }

    private val letterboxed = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(letterboxed)
    private val padPaint = Paint().apply { color = Color.rgb(PAD, PAD, PAD) }
    private val drawPaint = Paint().apply { isFilterBitmap = true }

    /** 每帧复用，避免每帧新分配几 MB 触发 GC。 */
    private val px = IntArray(inW * inH)
    private val planeBuf = if (nchw) FloatArray(inW * inH) else FloatArray(0)

    init {
        require(inH > 0 && inW > 0 && inH % 32 == 0 && inW % 32 == 0) {
            "YOLO 输入宽高必须是 32 的倍数：${inW}x$inH"
        }
        require(maxOf(inH, inW) >= MIN_LONG_SIDE) {
            "brick_yolo.tflite 输入 ${inW}x${inH}，长边小于 $MIN_LONG_SIDE：" +
                "这是旧 640 模型，重新导出 1024x480 后再装"
        }
        Log.i(
            TAG,
            "模型 IO：输入 ${inShape.contentToString()} 输出 ${outShape.contentToString()}" +
                "（${if (nchw) "NCHW" else "NHWC"}，${inW}x${inH}，${anchors} anchors）",
        )
    }

    /** 模型版本与耗时，控制条上要能看出这一帧到底是谁算的。 */
    val label: String = "YOLO nano ${inW}x${inH}"

    fun scan(frame: Bitmap): BoardScan? {
        val t0 = System.nanoTime()
        // letterbox：等比缩到模型输入框内，余量填灰（与训练一致）
        val scale = min(inW.toFloat() / frame.width, inH.toFloat() / frame.height)
        val dw = (frame.width * scale).roundToInt()
        val dh = (frame.height * scale).roundToInt()
        val ox = (inW - dw) / 2
        val oy = (inH - dh) / 2
        canvas.drawColor(Color.rgb(PAD, PAD, PAD))
        canvas.drawBitmap(frame, null, RectF(ox.toFloat(), oy.toFloat(),
            (ox + dw).toFloat(), (oy + dh).toFloat()), drawPaint)

        val tLb = System.nanoTime()

        fillInput(letterboxed)
        val tFill = System.nanoTime()

        interp.run(inputBuf, output)
        val tRun = System.nanoTime()

        val boxes = decode(ox, oy, scale)
        val tDec = System.nanoTime()
        Log.i(TAG, "${inW}x${inH} 原始输出最高分 board=${maxBoardScore} tile=${maxTileScore}，" +
            "过阈值后 board=${boxes.count { it.cls == CLS_BOARD }} tile=${boxes.count { it.cls == CLS_TILE }}" +
            "｜分段(ms) letterbox=${(tLb - t0) / 1_000_000} fill=${(tFill - tLb) / 1_000_000} " +
            "推理=${(tRun - tFill) / 1_000_000} 解码=${(tDec - tRun) / 1_000_000} " +
            "网格=${(System.nanoTime() - tDec) / 1_000_000}")

        val board = boxes.filter { it.cls == CLS_BOARD }.maxByOrNull { it.conf } ?: run {
            Log.w(TAG, "没有 board 框 → 这一帧放弃")
            return null
        }
        // 只认棋盘框内的 tile：模型偶尔会在关卡标题/状态栏上误检一个低分框
        // （实测 0.52 分、在棋盘上方 260px），它的中心会把最小二乘和残差整个带偏，
        // 让本来正确的 14x10 也算出 300px 残差、整帧被判"网格不可信"。
        // 外扩 10% 是为了容纳"board 框略小于实际棋盘"的情况；标题离得远，扩不到。
        val allTiles = boxes.filter { it.cls == CLS_TILE }
        val onBoard = allTiles.filter { board.contains(it.cx, it.cy, margin = 0.10f) }
        if (onBoard.size != allTiles.size) {
            Log.i(TAG, "丢弃 ${allTiles.size - onBoard.size} 个棋盘外的 tile 框（共 ${allTiles.size} 个）")
        }
        // 尺寸/长宽比异常的框：两块相邻砖被模型合并成一个高框（实测 h≈2×中位、conf 0.4），
        // 它的中心落在两格之间，单独一个就能把最大残差顶到 44px、整帧被判不可信。
        // 处理：沿长边拆成 N 个单块框（保留原 conf，分类仍按格子重新裁），
        // 拆不了的明显异常框才丢。
        val sizes = onBoard.map { (it.w + it.h) / 2f }.sorted()
        val medianSide = if (sizes.isEmpty()) 0f else sizes[sizes.size / 2]
        val tiles = ArrayList<Box>(onBoard.size + 2)
        var split = 0
        var dropped = 0
        if (medianSide > 0f) {
            for (t in onBoard) {
                val side = (t.w + t.h) / 2f
                val aspect = max(t.w, t.h) / min(t.w, t.h)
                when {
                    side >= medianSide * 0.6f && side <= medianSide * 1.5f && aspect <= 1.6f ->
                        tiles.add(t)

                    t.h > medianSide * 1.5f && t.w <= medianSide * 1.5f -> {
                        val n = (t.h / medianSide).roundToInt().coerceIn(2, 4)
                        for (i in 0 until n) {
                            val cy = t.cy - t.h / 2f + t.h * (i + 0.5f) / n
                            tiles.add(Box(t.cx, cy, t.w, t.h / n, t.conf, t.cls))
                        }
                        split++
                    }

                    t.w > medianSide * 1.5f && t.h <= medianSide * 1.5f -> {
                        val n = (t.w / medianSide).roundToInt().coerceIn(2, 4)
                        for (i in 0 until n) {
                            val cx = t.cx - t.w / 2f + t.w * (i + 0.5f) / n
                            tiles.add(Box(cx, t.cy, t.w / n, t.h, t.conf, t.cls))
                        }
                        split++
                    }

                    else -> dropped++
                }
            }
        } else {
            tiles.addAll(onBoard)
        }
        if (split > 0 || dropped > 0) {
            Log.i(
                TAG,
                "合并框拆分 $split 个、丢弃尺寸异常 $dropped 个" +
                    "（中位 ${medianSide.roundToInt()}px，共 ${onBoard.size} 个）",
            )
        }

        // 残局只剩一两块砖时，砖的尺寸仍然可测（一块就够），缺的是交叉验证的样本数。
        // 所以：砖够多 → 现算并缓存；砖很少 → 复用上一帧的网格，前提是棋盘框还在原位
        // （换关卡/换模式时框会明显跳变，缓存随之作废）。
        val cached = lastGrid?.takeIf { sameBoard(it, board) }
        val grid = when {
            tiles.size >= 3 -> buildGrid(board, tiles, cached)?.also { lastGrid = it }
            cached != null -> cached.also {
                Log.i(TAG, "只检出 ${tiles.size} 块砖，复用上一帧网格（${it.rows}x${it.cols}）")
            }
            // 1~2 块砖：砖框尺寸照样能测，只是没有多余样本交叉验证，所以不写缓存
            tiles.isNotEmpty() -> buildGrid(board, tiles)?.also {
                Log.i(TAG, "只检出 ${tiles.size} 块砖，直接用砖框尺寸定格距（${it.rows}x${it.cols}）")
            }
            else -> null
        }
        if (grid == null) {
            Log.w(TAG, "tile ${tiles.size} 个且没有可复用的网格 → 放弃（${boxText(board)}）")
            return null
        }
        val cols = grid.cols
        val rows = grid.rows
        val leftC = grid.left
        val topC = grid.top
        val stepX = grid.stepX
        val stepY = grid.stepY
        val score = FloatArray(rows * cols)
        for (r in 0 until rows) {
            val cy = (topC + (r + 0.5) * stepY).toFloat()
            for (c in 0 until cols) {
                val cx = (leftC + (c + 0.5) * stepX).toFloat()
                // 判据：空格中心不会被任何 tile 框覆盖（框都不含中心说明这块砖在隔壁）
                val hit = tiles.any {
                    cx >= it.left && cx <= it.right && cy >= it.top && cy <= it.bottom
                }
                score[r * cols + c] = if (hit) 1f else 0f
            }
        }
        val rects = Array(rows * cols) { i ->
            val r = i / cols
            val c = i % cols
            val l = (leftC + c * stepX).roundToInt()
            val t = (topC + r * stepY).roundToInt()
            val rt = (leftC + (c + 1) * stepX).roundToInt()
            val b = (topC + (r + 1) * stepY).roundToInt()
            CellRect(
                l.coerceIn(0, frame.width), t.coerceIn(0, frame.height),
                rt.coerceIn(0, frame.width), b.coerceIn(0, frame.height),
            )
        }
        // 网格置信度：残差偏大 / board 分数低 / 砖太少，都算"能用但可疑"，交给上层留档
        val suspect = buildList {
            if (grid.residual > stepX * 0.15f) add("网格残差 ${grid.residual.roundToInt()}px")
            if (board.conf < 0.6f) add("board 分数 ${"%.2f".format(board.conf)}")
            if (tiles.size < 3) add("只检出 ${tiles.size} 块砖")
        }.takeIf { it.isNotEmpty() }?.joinToString("、")
        val tileSide = if (tiles.isNotEmpty()) {
            tiles.map { (it.w + it.h) / 2f }.sorted()[tiles.size / 2]
        } else 0f
        Log.i(TAG, "YOLO 定位 ${rows}x$cols pitch=${stepX.roundToInt()}/${stepY.roundToInt()}" +
            " 原点=(${leftC.roundToInt()},${topC.roundToInt()})" +
            " ${boxText(board)}" +
            " tile=${tiles.size} 砖框中位=${tileSide.roundToInt()}px" +
            " 最大残差=${grid.residual.roundToInt()}px" +
            (suspect?.let { " ⚠$it" } ?: "") +
            "，耗时 ${(System.nanoTime() - t0) / 1_000_000}ms")
        return BoardScan(
            rows = rows, cols = cols,
            xlines = DoubleArray(cols + 1) { leftC + it * stepX.toDouble() },
            ylines = DoubleArray(rows + 1) { topC + it * stepY.toDouble() },
            score = score, scoreThreshold = 0.5f,
            rects = rects, pitchX = stepX.roundToInt(), pitchY = stepY.roundToInt(),
            suspect = suspect,
        )
    }

    /** 一帧的网格几何。缓存它是为了在残局（砖少到无法独立定格距）时继续可用。 */
    private class Grid(
        val left: Double, val top: Double, val stepX: Float, val stepY: Float,
        val cols: Int, val rows: Int,
        val boardCx: Float, val boardCy: Float, val boardW: Float, val boardH: Float,
        /** 砖块中心到最近格中心的最大偏差（像素）。网格错一格时这个值≈一个 pitch。 */
        val residual: Float,
    )

    private fun Grid.copyWithResidual(res: Float) = Grid(
        left = left, top = top, stepX = stepX, stepY = stepY, cols = cols, rows = rows,
        boardCx = boardCx, boardCy = boardCy, boardW = boardW, boardH = boardH, residual = res,
    )

    private var lastGrid: Grid? = null

    /** 用 board 框定范围、砖块尺寸定格距，再用砖心最小二乘把网格对齐到砖上。 */
    private fun buildGrid(board: Box, tiles: List<Box>, expected: Grid? = null): Grid? {
        val sides = tiles.map { (it.w + it.h) / 2f }.sorted()
        val tileSide = sides[sides.size / 2]
        val pitch = tileSide / TILE_INSET
        if (pitch < 12f) {
            Log.w(
                TAG,
                "格距只有 ${pitch}px（tile 框太小）→ 放弃；${boxText(board)}" +
                    " tile=${tiles.size} 砖框=${sideText(sides)}",
            )
            return null
        }
        val baseCols = (board.w / pitch).roundToInt()
        val baseRows = (board.h / pitch).roundToInt()

        // 行列数各试 ±1：残局时 board 框会缩一点，14 行就被算成 13 行，格线整体错一格
        // （实测残差 104px ≈ 一个 pitch，随后执行必然"坐标越界"）。
        // 用"砖块中心到最近格中心"的残差挑最优，残差过大说明怎么排都对不上，这一帧直接放弃。
        var best: Grid? = null
        for (dr in -1..1) {
            for (dc in -1..1) {
                val rows = baseRows + dr
                val cols = baseCols + dc
                if (rows !in 6..20 || cols !in 6..16) continue
                val g = fitGrid(board, tiles, rows, cols)
                if (best == null || g.residual < best.residual) best = g
            }
        }
        var pick = best
        if (pick == null) {
            Log.w(
                TAG,
                "没有候选网格：${boxText(board)} tile=${tiles.size} 砖框=${sideText(sides)}" +
                    " pitch=${pitch.roundToInt()} base=${baseRows}x${baseCols}",
            )
            return null
        }

        // 残局里整行/整列空掉时，±1 搜索会把空行裁掉（14 行 → 13 行），坐标系就变了，
        // 正在执行的计划会"坐标越界"。上一帧网格还在、残差又差不多时，沿用它的行列数：
        // 换关卡那种真的变了行列数的情况，旧行列数拟合残差会明显更大，不会误锁。
        if (expected != null && (pick.rows != expected.rows || pick.cols != expected.cols)) {
            val locked = fitGrid(board, tiles, expected.rows, expected.cols)
            val tolerance = max(pick.residual * 1.5f, pick.residual + pitch * 0.08f)
            if (locked.residual <= tolerance && locked.residual <= pitch * 0.35f) {
                Log.i(
                    TAG,
                    "行列数 ${pick.rows}x${pick.cols} → 沿用上一帧 ${expected.rows}x${expected.cols}" +
                        "（残差 ${pick.residual.roundToInt()} / ${locked.residual.roundToInt()}）",
                )
                pick = locked
            }
        }

        if (pick.residual > pitch * 0.35f) {
            val over = tiles.count { distanceTo(pick, it) > pitch * 0.35f }
            Log.w(
                TAG,
                "网格残差 ${pick.residual.roundToInt()}px（上限 ${(pitch * 0.35f).roundToInt()}）→ 放弃；" +
                    "${boxText(board)} tile=${tiles.size} 砖框=${sideText(sides)}" +
                    " pitch=${pitch.roundToInt()} base=${baseRows}x${baseCols}" +
                    " best=${pick.rows}x${pick.cols} step=${pick.stepX.roundToInt()}/${pick.stepY.roundToInt()}" +
                    " 原点=(${pick.left.roundToInt()},${pick.top.roundToInt()})" +
                    " 超限砖=$over/${tiles.size}",
            )
            return null
        }
        return pick
    }

    /** board 框和置信度，网格失败/成功日志共用。 */
    private fun boxText(board: Box): String =
        "board=(${board.left.roundToInt()},${board.top.roundToInt()}," +
            "${board.w.roundToInt()}x${board.h.roundToInt()}) conf=${"%.2f".format(board.conf)}"

    /** 砖框边长区间（最小~最大），判断是否混进了半块砖/异常框。 */
    private fun sideText(sides: List<Float>): String =
        if (sides.isEmpty()) "-" else "${sides.first().roundToInt()}~${sides.last().roundToInt()}"

    /** 按给定行列数拟合一套网格；行列数不在这个函数里改。 */
    private fun fitGrid(board: Box, tiles: List<Box>, rows: Int, cols: Int): Grid {
        val fitX = fitAxis(
            tiles.map { it.cx }.toFloatArray(),
            board.left.toDouble(), board.w.toDouble() / cols, cols,
        )
        val fitY = fitAxis(
            tiles.map { it.cy }.toFloatArray(),
            board.top.toDouble(), board.h.toDouble() / rows, rows,
        )
        val g = Grid(
            left = fitX.first, top = fitY.first,
            stepX = fitX.second.toFloat(), stepY = fitY.second.toFloat(),
            cols = cols, rows = rows,
            boardCx = (board.left + board.right) / 2f,
            boardCy = (board.top + board.bottom) / 2f,
            boardW = board.w, boardH = board.h,
            residual = 0f,
        )
        return g.copyWithResidual(residualOf(g, tiles))
    }

    /** 砖块中心到"它所在格中心"的最大距离。网格错一格时等于一个 pitch。 */
    private fun residualOf(g: Grid, tiles: List<Box>): Float {
        var worst = 0f
        tiles.forEach { worst = maxOf(worst, distanceTo(g, it)) }
        return worst
    }

    /** 一块砖的中心到它所在格中心的距离。 */
    private fun distanceTo(g: Grid, box: Box): Float {
        val c = ((box.cx - g.left) / g.stepX).toInt().coerceIn(0, g.cols - 1)
        val r = ((box.cy - g.top) / g.stepY).toInt().coerceIn(0, g.rows - 1)
        val dx = box.cx - (g.left + (c + 0.5) * g.stepX)
        val dy = box.cy - (g.top + (r + 0.5) * g.stepY)
        return kotlin.math.hypot(dx, dy).toFloat()
    }

    /** 缓存的网格还能不能用：棋盘框中心和大小都没明显变化（换关卡/换模式时会跳）。 */
    private fun sameBoard(g: Grid, board: Box): Boolean {
        val dx = (board.left + board.right) / 2f - g.boardCx
        val dy = (board.top + board.bottom) / 2f - g.boardCy
        val tol = g.stepX * 0.5f
        val dw = kotlin.math.abs(board.w - g.boardW) / g.boardW
        val dh = kotlin.math.abs(board.h - g.boardH) / g.boardH
        return dx * dx + dy * dy <= tol * tol && dw < 0.08f && dh < 0.08f
    }

    /**
     * 一条轴上的网格拟合：输入是砖块中心坐标，输出「第 0 格左/上边」和「格距」。
     *
     * 做法是"先按 board 框分格 → 用分到的格号做最小二乘 → 再回代"。砖心是逐块回归出来的，
     * 比整块 board 框准得多；残局只剩几块砖时也够用（只要跨两格就能定斜率）。
     * 拟合出的格距偏离初值超过 20% 就判定不可信，保留 board 框的结果。
     */
    private fun fitAxis(centers: FloatArray, start: Double, step0: Double, count: Int): Pair<Double, Double> {
        if (centers.size < 2) return start to step0
        val idx = centers.map { ((it - start) / step0 - 0.5).roundToInt() }
        val keep = idx.indices.filter { idx[it] in 0 until count }
        if (keep.map { idx[it] }.distinct().size < 2) return start to step0
        val m = keep.size.toDouble()
        val sumI = keep.sumOf { idx[it].toDouble() }
        val sumC = keep.sumOf { centers[it].toDouble() }
        val sumIC = keep.sumOf { idx[it].toDouble() * centers[it] }
        val sumII = keep.sumOf { (idx[it] * idx[it]).toDouble() }
        val den = m * sumII - sumI * sumI
        if (den == 0.0) return start to step0
        val step = (m * sumIC - sumI * sumC) / den
        if (step < step0 * 0.8 || step > step0 * 1.2) return start to step0
        val center0 = (sumC - step * sumI) / m
        return (center0 - step / 2.0) to step
    }

    /** Bitmap → float32，值域 0..1（训练时 Ultralytics 就是这么归一化的）。 */
    private fun fillInput(bmp: Bitmap) {
        bmp.getPixels(px, 0, inW, 0, 0, inW, inH)
        val plane = inW * inH
        val fb = inputBuf.asFloatBuffer()
        if (nchw) {
            for (ch in 0..2) {
                val shift = when (ch) { 0 -> 16; 1 -> 8; else -> 0 }
                for (i in 0 until plane) {
                    planeBuf[i] = ((px[i] shr shift) and 0xFF) / 255f
                }
                fb.position(ch * plane)
                fb.put(planeBuf)
            }
        } else {
            // NHWC：每个像素的 RGB 连续写入
            var o = 0
            for (i in 0 until plane) {
                val p = px[i]
                fb.put(o++, ((p shr 16) and 0xFF) / 255f)
                fb.put(o++, ((p shr 8) and 0xFF) / 255f)
                fb.put(o++, (p and 0xFF) / 255f)
            }
        }
        inputBuf.rewind()
    }

    /** 最近一帧的原始最高分，日志用。 */
    private var maxBoardScore = 0f
    private var maxTileScore = 0f

    /** 输出 [1,4+类别数,N] → 框列表（坐标已还原到原图）。 */
    private fun decode(ox: Int, oy: Int, scale: Float): List<Box> {
        val out = output[0]
        val n = out[0].size
        val cand = ArrayList<Box>(256)
        maxBoardScore = out[4].max()
        maxTileScore = out[5].max()
        // 模型输出是**归一化 0..1**（LiteRT 导出的 YOLO 就是这种），不是输入像素坐标。
        // 少了这一步，框的边长只有 0.06 → 格距算出来 0.06px → 静默放弃。
        // 矩形输入下 x/w 按宽归一化、y/h 按高归一化，不能共用一个 k。
        val normalized = out[0].max() <= 2f
        val kx = if (normalized) inW.toFloat() else 1f
        val ky = if (normalized) inH.toFloat() else 1f
        for (i in 0 until n) {
            val sBoard = out[4][i]
            val sTile = out[5][i]
            val (cls, conf) = if (sTile >= sBoard) CLS_TILE to sTile else CLS_BOARD to sBoard
            if (conf < CONF) continue
            val cx = (out[0][i] * kx - ox) / scale
            val cy = (out[1][i] * ky - oy) / scale
            val w = out[2][i] * kx / scale
            val h = out[3][i] * ky / scale
            cand.add(Box(cx, cy, w, h, conf, cls))
        }
        return nms(cand, NMS_IOU)
    }

    private fun nms(boxes: List<Box>, iouThr: Float): List<Box> {
        val sorted = boxes.sortedByDescending { it.conf }
        val keep = ArrayList<Box>()
        val dropped = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (dropped[i]) continue
            val a = sorted[i]
            keep.add(a)
            for (j in i + 1 until sorted.size) {
                if (dropped[j] || sorted[j].cls != a.cls) continue
                if (iou(a, sorted[j]) > iouThr) dropped[j] = true
            }
        }
        return keep
    }

    private fun iou(a: Box, b: Box): Float {
        val iw = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val ih = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val inter = iw * ih
        val uni = a.w * a.h + b.w * b.h - inter
        return if (uni <= 0f) 0f else inter / uni
    }

    private fun loadModel(context: Context, name: String): ByteBuffer {
        val bytes = context.assets.open(name).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }
    }

    override fun close() {
        runCatching { interp.close() }
        gpuDelegate?.let { runCatching { it.close() } }
        letterboxed.recycle()
    }
}
