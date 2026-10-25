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
 *   · 输入  [1, 3, 640, 640] float32，**NCHW**（LiteRT 导出的默认排布，不是 NHWC）
 *   · 输出  [1, 6, 8400]，4 个框 + 2 类分数（0=board，1=tile），anchor-free、不含 NMS
 *   · 训练用 letterbox（等比缩放 + 灰边 114），推理必须同样处理，否则框会整体缩水
 */
class YoloBoardDetector(context: Context) : Closeable {

    companion object {
        private const val TAG = "YoloBoardDetector"
        private const val INPUT = 640
        private const val PAD = 114
        private const val CONF = 0.25f
        private const val NMS_IOU = 0.45f

        /**
         * 训练时 tile 框**每边**内缩 6%（见 make_yolo_data 的 INSET_TILE），
         * 所以框宽 = 0.88 格。这里除回去得到格距。
         *
         * 曾经写成 0.94（只算了一边），残局时 board 宽 1002 / 154 = 6.5 → 取整成 7 列，
         * 满盘时却因为框好巧不巧偏大而凑对 —— 属于错误一直没暴露的那类 bug。
         */
        private const val TILE_INSET = 0.88f

        private const val CLS_BOARD = 0
        private const val CLS_TILE = 1
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

    private val interp = Interpreter(
        loadModel(context, "brick_yolo.tflite"),
        Interpreter.Options().apply {
            gpuDelegate?.let { addDelegate(it) }
            setNumThreads(4)
        },
    )

    private val inputBuf: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT * INPUT * 3 * 4).order(ByteOrder.nativeOrder())

    private val output = Array(1) { Array(6) { FloatArray(8400) } }

    private val letterboxed = Bitmap.createBitmap(INPUT, INPUT, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(letterboxed)
    private val padPaint = Paint().apply { color = Color.rgb(PAD, PAD, PAD) }
    private val drawPaint = Paint().apply { isFilterBitmap = true }

    /** 模型版本与耗时，控制条上要能看出这一帧到底是谁算的。 */
    val label: String = "YOLO nano"

    fun scan(frame: Bitmap): BoardScan? {
        val t0 = System.nanoTime()
        // letterbox：等比缩到 640 内，余量填灰（与训练一致）
        val scale = min(INPUT.toFloat() / frame.width, INPUT.toFloat() / frame.height)
        val dw = (frame.width * scale).roundToInt()
        val dh = (frame.height * scale).roundToInt()
        val ox = (INPUT - dw) / 2
        val oy = (INPUT - dh) / 2
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
        Log.i(TAG, "原始输出最高分 board=${maxBoardScore} tile=${maxTileScore}，" +
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
        val tiles = allTiles.filter { board.contains(it.cx, it.cy, margin = 0.10f) }
        if (tiles.size != allTiles.size) {
            Log.i(TAG, "丢弃 ${allTiles.size - tiles.size} 个棋盘外的 tile 框（共 ${allTiles.size} 个）")
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
            Log.w(TAG, "tile ${tiles.size} 个且没有可复用的网格 → 放弃")
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
        Log.i(TAG, "YOLO 定位 ${rows}x$cols pitch=${stepX.roundToInt()}/${stepY.roundToInt()}" +
            " 原点=(${leftC.roundToInt()},${topC.roundToInt()}) tile ${tiles.size} 个" +
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
        val tileSide = tiles.map { (it.w + it.h) / 2f }.sorted()[tiles.size / 2]
        val pitch = tileSide / TILE_INSET
        if (pitch < 12f) {
            Log.w(TAG, "格距只有 ${pitch}px（tile 框太小）→ 放弃")
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
        var pick = best ?: return null

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
            Log.w(TAG, "网格残差 ${pick.residual.roundToInt()}px（上限 ${(pitch * 0.35f).roundToInt()}）" +
                " → 这一帧不可信，放弃")
            return null
        }
        return pick
    }

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
        tiles.forEach {
            val c = ((it.cx - g.left) / g.stepX).toInt().coerceIn(0, g.cols - 1)
            val r = ((it.cy - g.top) / g.stepY).toInt().coerceIn(0, g.rows - 1)
            val dx = it.cx - (g.left + (c + 0.5) * g.stepX)
            val dy = it.cy - (g.top + (r + 0.5) * g.stepY)
            worst = maxOf(worst, kotlin.math.hypot(dx, dy).toFloat())
        }
        return worst
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

    /** Bitmap → NCHW float32，值域 0..1（训练时 Ultralytics 就是这么归一化的）。 */
    private fun fillInput(bmp: Bitmap) {
        val px = IntArray(INPUT * INPUT)
        bmp.getPixels(px, 0, INPUT, 0, 0, INPUT, INPUT)
        inputBuf.rewind()
        val plane = INPUT * INPUT
        val buf = FloatArray(plane)
        for (ch in 0..2) {
            val shift = when (ch) { 0 -> 16; 1 -> 8; else -> 0 }
            for (i in 0 until plane) {
                buf[i] = ((px[i] shr shift) and 0xFF) / 255f
            }
            val fb = inputBuf.asFloatBuffer()
            fb.position(ch * plane)
            fb.put(buf)
        }
        inputBuf.rewind()
    }

    /** 输出 [1,6,8400] → 框列表（坐标已还原到原图）。 */
    private var maxBoardScore = 0f
    private var maxTileScore = 0f

    private fun decode(ox: Int, oy: Int, scale: Float): List<Box> {
        val out = output[0]
        val n = out[0].size
        val cand = ArrayList<Box>(256)
        maxBoardScore = out[4].max()
        maxTileScore = out[5].max()
        // 模型输出是**归一化 0..1**（LiteRT 导出的 YOLO 就是这种），不是 640 像素坐标。
        // 少了这一步，框的边长只有 0.06 → 格距算出来 0.06px → 静默放弃。
        val k = if (out[0].max() <= 2f) INPUT.toFloat() else 1f
        for (i in 0 until n) {
            val sBoard = out[4][i]
            val sTile = out[5][i]
            val (cls, conf) = if (sTile >= sBoard) CLS_TILE to sTile else CLS_BOARD to sBoard
            if (conf < CONF) continue
            val cx = (out[0][i] * k - ox) / scale
            val cy = (out[1][i] * k - oy) / scale
            val w = out[2][i] * k / scale
            val h = out[3][i] * k / scale
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
