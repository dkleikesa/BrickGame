package com.qcmian.brickgame.vision

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.TimeSource

/**
 * `CellResampler` 的两件事：
 *  1. 数值对齐 —— 与"按定义直接算"的浮点实现比，逐通道差 ≤2（8 位灰阶）；
 *  2. 性能 —— 一帧 135 格的裁格耗时（JVM 数据，只用来对比新旧实现的量级）。
 */
class CellResamplerTest {

    private val cropSide = 98
    private val size = 64
    private val outSize = 160

    private fun frame(w: Int = 1080, h: Int = 2400): IntArray =
        IntArray(w * h) { i -> (0xFF shl 24) or ((i * 2654435761L).toInt() and 0xFFFFFF) }

    @Test
    fun areaMatchesReference() {
        val src = frame()
        val x0 = 100
        val y0 = 600
        val ax = CellResampler.axis(cropSide, size)
        val ay = CellResampler.axis(cropSide, size)
        val tmp = IntArray(cropSide * size)
        val got = IntArray(size * size)
        CellResampler.area(src, 1080, x0, y0, cropSide, cropSide, ax, ay, tmp, got, size)

        val want = refArea(src, 1080, x0, y0, cropSide, cropSide, size)
        var maxDiff = 0
        for (i in got.indices) {
            for (shift in intArrayOf(16, 8, 0)) {
                maxDiff = maxOf(maxDiff, abs(((got[i] shr shift) and 0xFF) - ((want[i] shr shift) and 0xFF)))
            }
        }
        assertTrue(maxDiff <= 2, "区域平均与浮点参考差过大：$maxDiff")
    }

    @Test
    fun bilinearMatchesReference() {
        val src = IntArray(size * size) { i -> (0xFF shl 24) or ((i * 40503L).toInt() and 0xFFFFFF) }
        val ax = CellResampler.linear(size, outSize)
        val out = FloatArray(outSize * outSize * 3)
        CellResampler.bilinearToFloat(src, size, outSize, ax, out, 0)

        val want = refLinear(src, size, outSize)
        var maxDiff = 0f
        for (i in out.indices) maxDiff = maxOf(maxDiff, abs(out[i] - want[i]))
        assertTrue(maxDiff <= 2f / 255f, "双线性与浮点参考差过大：$maxDiff")
    }

    /**
     * 一帧 135 格：新实现 vs 旧实现（每格拷一次 + 浮点二维区域平均 + 浮点双线性）。
     * 只打印耗时；断言放得很松，只拦异常退化。
     */
    @Test
    fun benchmark135Cells() {
        val fw = 1080
        val fh = 2400
        val src = frame(fw, fh)
        val side = cropSide
        val xs = IntArray(135)
        val ys = IntArray(135)
        for (k in 0 until 135) {
            xs[k] = 40 + (k % 10) * 102
            ys[k] = 560 + (k / 10) * 102
        }

        val ax = CellResampler.axis(side, size)
        val ay = CellResampler.axis(side, size)
        val lin = CellResampler.linear(size, outSize)
        val tmp = IntArray(side * size)
        val mid = IntArray(size * size)
        val out = FloatArray(outSize * outSize * 3)

        fun runNew(): Long {
            val t0 = TimeSource.Monotonic.markNow()
            for (k in 0 until 135) {
                CellResampler.area(src, fw, xs[k], ys[k], side, side, ax, ay, tmp, mid, size)
                CellResampler.bilinearToFloat(mid, size, outSize, lin, out, 0)
            }
            return t0.elapsedNow().inWholeMilliseconds
        }

        val cell = IntArray(side * side)
        fun runOld(): Long {
            val t0 = TimeSource.Monotonic.markNow()
            for (k in 0 until 135) {
                for (row in 0 until side) {
                    var i = 0
                    while (i < side) {
                        cell[row * side + i] = src[(ys[k] + row) * fw + xs[k] + i]
                        i++
                    }
                }
                val m = refArea(cell, side, 0, 0, side, side, size)
                refLinear(m, size, outSize)
            }
            return t0.elapsedNow().inWholeMilliseconds
        }

        repeat(3) { runNew() }
        repeat(2) { runOld() }
        val n = 5
        var newMs = 0L
        var oldMs = 0L
        repeat(n) {
            newMs += runNew()
            oldMs += runOld()
        }
        val perFrameNew = newMs.toDouble() / n
        val perFrameOld = oldMs.toDouble() / n
        println(
            "裁格 135 格：新 " + round1(perFrameNew) + " ms/帧（" + round1(perFrameNew / 135) + " ms/格）" +
                " / 旧 " + round1(perFrameOld) + " ms/帧（" + round1(perFrameOld / 135) + " ms/格）",
        )
        assertTrue(perFrameNew < 500.0, "裁格异常退化：$perFrameNew ms/帧")
    }

    private fun round1(v: Double): Double = (v * 10).roundToInt() / 10.0

    // ---------------------------------------------------------------- 浮点参考实现

    private fun refArea(src: IntArray, stride: Int, x0: Int, y0: Int, w: Int, h: Int, dst: Int): IntArray {
        val out = IntArray(dst * dst)
        val sx = w.toDouble() / dst
        val sy = h.toDouble() / dst
        for (dy in 0 until dst) {
            val sy0 = dy * sy
            val sy1 = min(sy0 + sy, h.toDouble())
            for (dx in 0 until dst) {
                val sx0 = dx * sx
                val sx1 = min(sx0 + sx, w.toDouble())
                var r = 0.0
                var g = 0.0
                var b = 0.0
                var sum = 0.0
                var yy = floor(sy0).toInt()
                while (yy < ceil(sy1).toInt()) {
                    val wy = min(sy1, (yy + 1).toDouble()) - maxOf(sy0, yy.toDouble())
                    if (wy > 0) {
                        var xx = floor(sx0).toInt()
                        while (xx < ceil(sx1).toInt()) {
                            val wx = min(sx1, (xx + 1).toDouble()) - maxOf(sx0, xx.toDouble())
                            if (wx > 0) {
                                val p = src[(y0 + min(yy, h - 1)) * stride + x0 + min(xx, w - 1)]
                                val wgt = wx * wy
                                r += ((p shr 16) and 0xFF) * wgt
                                g += ((p shr 8) and 0xFF) * wgt
                                b += (p and 0xFF) * wgt
                                sum += wgt
                            }
                            xx++
                        }
                    }
                    yy++
                }
                out[dy * dst + dx] = (0xFF shl 24) or
                    ((r / sum + 0.5).toInt().coerceIn(0, 255) shl 16) or
                    ((g / sum + 0.5).toInt().coerceIn(0, 255) shl 8) or
                    (b / sum + 0.5).toInt().coerceIn(0, 255)
            }
        }
        return out
    }

    private fun refLinear(src: IntArray, srcSize: Int, dst: Int): FloatArray {
        val out = FloatArray(dst * dst * 3)
        val scale = srcSize.toDouble() / dst
        var o = 0
        for (y in 0 until dst) {
            val fy = (y + 0.5) * scale - 0.5
            val y0 = floor(fy).toInt().coerceIn(0, srcSize - 1)
            val y1 = min(y0 + 1, srcSize - 1)
            val wy = (fy - y0).coerceIn(0.0, 1.0)
            for (x in 0 until dst) {
                val fx = (x + 0.5) * scale - 0.5
                val x0 = floor(fx).toInt().coerceIn(0, srcSize - 1)
                val x1 = min(x0 + 1, srcSize - 1)
                val wx = (fx - x0).coerceIn(0.0, 1.0)
                val p00 = src[y0 * srcSize + x0]
                val p01 = src[y0 * srcSize + x1]
                val p10 = src[y1 * srcSize + x0]
                val p11 = src[y1 * srcSize + x1]
                val w00 = (1 - wx) * (1 - wy)
                val w01 = wx * (1 - wy)
                val w10 = (1 - wx) * wy
                val w11 = wx * wy
                out[o++] = ((p00 shr 16 and 0xFF) * w00 + (p01 shr 16 and 0xFF) * w01 +
                    (p10 shr 16 and 0xFF) * w10 + (p11 shr 16 and 0xFF) * w11).toFloat() / 255f
                out[o++] = ((p00 shr 8 and 0xFF) * w00 + (p01 shr 8 and 0xFF) * w01 +
                    (p10 shr 8 and 0xFF) * w10 + (p11 shr 8 and 0xFF) * w11).toFloat() / 255f
                out[o++] = ((p00 and 0xFF) * w00 + (p01 and 0xFF) * w01 +
                    (p10 and 0xFF) * w10 + (p11 and 0xFF) * w11).toFloat() / 255f
            }
        }
        return out
    }
}
