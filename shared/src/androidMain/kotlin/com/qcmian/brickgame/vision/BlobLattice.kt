package com.qcmian.brickgame.vision

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 残局兜底：自相关失效（棋盘只剩几块砖）时，用"棋盘底色 + 砖块色块"反推网格。
 *
 * 对应 `tile_recognizer.py` 的 `_board_blobs` / `_pitch_from_centers` / `estimate_lattice_from_blobs`。
 *
 * 实测触发场景：棋盘只剩十几块砖时，边沿投影被天空/马路/UI 的边缘带偏，
 * 算出的 x 周期 23px、y 周期 94px（真值 110/110），网格画到房子上去了，
 * 于是切出一堆"半块砖"和背景。这里用棋盘外框 + 砖块间距重建等距网格。
 */
internal object BlobLattice {

    class Lattice(
        val xlines: DoubleArray,
        val ylines: DoubleArray,
        val pitchX: Int,
        val pitchY: Int,
    )

    private const val BG_TOLERANCE = 45.0
    private const val MIN_BOX = 12

    fun estimate(img: Mat): Lattice? {
        val h = img.rows()
        val w = img.cols()
        if (h < 32 || w < 32 || img.channels() != 3) return null

        val bytes = ByteArray(h * w * 3)
        img.get(0, 0, bytes)

        val bg = boardColor(bytes, w, h) ?: return null
        val isBg = BooleanArray(h * w)
        for (p in 0 until h * w) {
            val i = p * 3
            val db = bytes[i].toInt() - bg[0]
            val dg = bytes[i + 1].toInt() - bg[1]
            val dr = bytes[i + 2].toInt() - bg[2]
            isBg[p] = sqrt((db * db + dg * dg + dr * dr).toDouble()) < BG_TOLERANCE
        }

        // 棋盘底色的最大连通域 = 棋盘外框（残局时它反而是画面里最大的一片均匀色）
        val board = components(isBg, w, h)?.maxByOrNull { it.area } ?: return null
        if (board.width < w * 0.3 || board.height < h * 0.2) return null

        // 域内的非底色像素 = 残余砖块
        val bricks = BooleanArray(board.width * board.height)
        for (y in 0 until board.height) {
            for (x in 0 until board.width) {
                bricks[y * board.width + x] = !isBg[(board.top + y) * w + board.left + x]
            }
        }

        val boxes = components(bricks, board.width, board.height)
            .orEmpty()
            .filter { it.width > MIN_BOX && it.height > MIN_BOX }
        if (boxes.isEmpty()) return null

        val wMed = medianOf(boxes.map { it.width.toDouble() })
        val hMed = medianOf(boxes.map { it.height.toDouble() })
        val kept = boxes.filter {
            it.width >= 0.5 * wMed && it.width <= 2.0 * wMed &&
                it.height >= 0.5 * hMed && it.height <= 2.0 * hMed
        }
        if (kept.isEmpty()) return null

        val cx = kept.map { it.left + it.width / 2.0 }.toDoubleArray()
        val cy = kept.map { it.top + it.height / 2.0 }.toDoubleArray()
        val px = pitchFromCenters(cx, wMed)
        val py = pitchFromCenters(cy, hMed)
        val cand = listOf(px, py).filter { it > 0 }
        if (cand.isEmpty()) return null

        val pitch = medianOf(cand)
        if (pitch <= 0) return null
        val cols = (board.width / pitch).roundToInt()
        val rows = (board.height / pitch).roundToInt()
        if (cols !in 3..30 || rows !in 3..30) return null

        return Lattice(
            xlines = linspace(board.left.toDouble(), (board.left + board.width).toDouble(), cols + 1),
            ylines = linspace(board.top.toDouble(), (board.top + board.height).toDouble(), rows + 1),
            pitchX = (board.width.toDouble() / cols).roundToInt(),
            pitchY = (board.height.toDouble() / rows).roundToInt(),
        )
    }

    /** 棋盘底色：ROI 内出现最多的量化颜色（`//16` 分桶）的均值。 */
    private fun boardColor(bytes: ByteArray, w: Int, h: Int): IntArray? {
        val y0 = (h * 0.12).toInt()
        val y1 = (h * 0.92).toInt()
        val x0 = (w * 0.05).toInt()
        val x1 = (w * 0.95).toInt()
        if (y1 <= y0 || x1 <= x0) return null

        val histogram = HashMap<Int, Int>()
        for (y in y0 until y1) {
            for (x in x0 until x1) {
                val i = (y * w + x) * 3
                val b = (bytes[i].toInt() and 0xFF) / 16
                val g = (bytes[i + 1].toInt() and 0xFF) / 16
                val r = (bytes[i + 2].toInt() and 0xFF) / 16
                val key = b * 10000 + g * 100 + r
                histogram[key] = (histogram[key] ?: 0) + 1
            }
        }
        var bestKey = -1
        var bestCount = -1
        for ((key, count) in histogram) {
            if (count > bestCount) {
                bestCount = count
                bestKey = key
            }
        }
        if (bestKey < 0) return null

        var n = 0
        val sum = LongArray(3)
        for (y in y0 until y1) {
            for (x in x0 until x1) {
                val i = (y * w + x) * 3
                val b = (bytes[i].toInt() and 0xFF) / 16
                val g = (bytes[i + 1].toInt() and 0xFF) / 16
                val r = (bytes[i + 2].toInt() and 0xFF) / 16
                if (b * 10000 + g * 100 + r == bestKey) {
                    sum[0] += bytes[i].toInt() and 0xFF
                    sum[1] += bytes[i + 1].toInt() and 0xFF
                    sum[2] += bytes[i + 2].toInt() and 0xFF
                    n++
                }
            }
        }
        if (n == 0) return null
        return intArrayOf((sum[0] / n).toInt(), (sum[1] / n).toInt(), (sum[2] / n).toInt())
    }

    /** 连通域外框（8 邻域）。OpenCV 的 stats 列顺序是 left/top/width/height/area。 */
    class Box(val left: Int, val top: Int, val width: Int, val height: Int, val area: Int)

    private fun components(mask: BooleanArray, w: Int, h: Int): List<Box>? {
        val mat = Mat(h, w, CvType.CV_8UC1)
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        try {
            val data = ByteArray(mask.size)
            for (i in mask.indices) data[i] = if (mask[i]) 1.toByte() else 0.toByte()
            mat.put(0, 0, data)
            val n = Imgproc.connectedComponentsWithStats(mat, labels, stats, centroids, 8, CvType.CV_32S)
            if (n <= 1) return emptyList()
            val raw = IntArray(n * 5)
            stats.get(0, 0, raw)
            val out = ArrayList<Box>(n - 1)
            for (label in 1 until n) {
                val base = label * 5
                out.add(Box(raw[base], raw[base + 1], raw[base + 2], raw[base + 3], raw[base + 4]))
            }
            return out
        } catch (e: Throwable) {
            return null
        } finally {
            mat.release()
            labels.release()
            stats.release()
            centroids.release()
        }
    }

    /**
     * 一串色块中心 → 格距。
     *
     * 不能用"色块宽度"当格距：砖块底托比格子小一圈（实测底托 103px、真实格距 110px），
     * 按宽度算 1107/103 会得出 11 列（真值 10 列）。改用**相邻中心的最小间距**，
     * 它只可能是格距的整数倍；再用"格子必定装得下底托"这个下限反推倍了几格。
     */
    private fun pitchFromCenters(centers: DoubleArray, plate: Double): Double {
        val c = centers.distinct().sorted()
        if (c.size >= 2 && plate > 0) {
            var d = Double.MAX_VALUE
            for (i in 1 until c.size) d = minOf(d, c[i] - c[i - 1])
            if (d >= plate * 0.85) {
                return d / max(1, (d / (plate * 0.85)).toInt())
            }
        }
        return if (plate > 0) plate else 0.0
    }

    private fun medianOf(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val v = values.sorted()
        return if (v.size % 2 == 1) v[v.size / 2] else 0.5 * (v[v.size / 2 - 1] + v[v.size / 2])
    }

    private fun linspace(start: Double, stop: Double, count: Int): DoubleArray {
        if (count <= 1) return doubleArrayOf(start)
        val step = (stop - start) / (count - 1)
        return DoubleArray(count) { start + it * step }
    }
}
