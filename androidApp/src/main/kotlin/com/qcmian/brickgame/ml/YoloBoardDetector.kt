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
    }

    private val interp = Interpreter(
        loadModel(context, "brick_yolo.tflite"),
        Interpreter.Options().apply { setNumThreads(4) },
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

        fillInput(letterboxed)
        interp.run(inputBuf, output)
        val boxes = decode(ox, oy, scale)
        Log.i(TAG, "原始输出最高分 board=${maxBoardScore} tile=${maxTileScore}，" +
            "过阈值后 board=${boxes.count { it.cls == CLS_BOARD }} tile=${boxes.count { it.cls == CLS_TILE }}")

        val board = boxes.filter { it.cls == CLS_BOARD }.maxByOrNull { it.conf } ?: run {
            Log.w(TAG, "没有 board 框 → 这一帧放弃")
            return null
        }
        val tiles = boxes.filter { it.cls == CLS_TILE }
        if (tiles.size < 3) {
            Log.w(TAG, "tile 只有 ${tiles.size} 个，不足以定格距 → 放弃")
            return null
        }
        val tileSide = tiles.map { (it.w + it.h) / 2f }.sorted()[tiles.size / 2]
        val pitch = tileSide / TILE_INSET
        if (pitch < 12f) {
            Log.w(TAG, "格距只有 ${pitch}px（tile 框太小）→ 放弃")
            return null
        }
        val cols = (board.w / pitch).roundToInt()
        val rows = (board.h / pitch).roundToInt()
        if (cols !in 6..16 || rows !in 6..20) {
            Log.w(TAG, "网格尺寸不合理：${rows}x$cols（pitch=${pitch.roundToInt()}）→ 放弃")
            return null
        }

        // board 框是"整片区域"的回归，边缘常差十几像素；每块砖的中心准得多。
        // 先按 board 框定初值，再用砖心做一次最小二乘，把网格对齐到砖块上。
        val fitX = fitAxis(tiles.map { it.cx }.toFloatArray(), board.left.toDouble(), board.w.toDouble() / cols, cols)
        val fitY = fitAxis(tiles.map { it.cy }.toFloatArray(), board.top.toDouble(), board.h.toDouble() / rows, rows)
        val leftC = fitX.first
        val stepX = fitX.second.toFloat()
        val topC = fitY.first
        val stepY = fitY.second.toFloat()
        val cellW = stepX
        val cellH = stepY
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
        // 拟合残差：砖块中心到"它所在格中心"的距离，正常应该只有几像素
        var res = 0.0
        tiles.forEach {
            val c = ((it.cx - leftC) / stepX).toInt().coerceIn(0, cols - 1)
            val r = ((it.cy - topC) / stepY).toInt().coerceIn(0, rows - 1)
            val dx = it.cx - (leftC + (c + 0.5) * stepX)
            val dy = it.cy - (topC + (r + 0.5) * stepY)
            res = maxOf(res, kotlin.math.hypot(dx, dy))
        }
        Log.i(TAG, "YOLO 定位 ${rows}x$cols pitch=${stepX.roundToInt()}/${stepY.roundToInt()}" +
            " 原点=(${leftC.roundToInt()},${topC.roundToInt()}) tile ${tiles.size} 个" +
            " 最大残差=${res.roundToInt()}px，耗时 ${(System.nanoTime() - t0) / 1_000_000}ms")
        return BoardScan(
            rows = rows, cols = cols,
            xlines = DoubleArray(cols + 1) { leftC + it * stepX.toDouble() },
            ylines = DoubleArray(rows + 1) { topC + it * stepY.toDouble() },
            score = score, scoreThreshold = 0.5f,
            rects = rects, pitchX = stepX.roundToInt(), pitchY = stepY.roundToInt(),
        )
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
        letterboxed.recycle()
    }
}
