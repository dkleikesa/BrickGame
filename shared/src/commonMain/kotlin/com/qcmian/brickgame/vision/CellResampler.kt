package com.qcmian.brickgame.vision

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 纯 Kotlin 重采样：区域平均（缩小）+ 双线性（放大）。
 *
 * 只复刻了识别裁格真正用到的那两个核（OpenCV 的 `INTER_AREA` / `INTER_LINEAR`），
 * 移植到端侧时逐像素差 ≤1 级灰阶、embedding 余弦 ≥0.9993（见 `CellResamplerTest`）。
 *
 * 性能上的三个取舍（老实现是逐格 `getPixels` + 每像素 `Double` 累加）：
 *  1. **不碰浮点**：权重在查表阶段算好，热点循环全是 `Int` 乘加；面积权重 12 位、
 *     双线性权重 8 位，累加上限 255×4096×taps 远小于 `Int.MAX`。
 *  2. **区域平均拆成两趟**（横→竖）：98→64 的 2.4×2.4 窗口从"每格 6 个源像素"降到
 *     每趟只有 2~3 个 tap，乘加次数约少一半。
 *  3. **权重表按尺寸缓存**：一帧里所有格子的边长都一样，表只算一次；只有贴边被裁的
 *     那几格会多出一两个尺寸。
 *
 * 缓冲区由调用方持有（多线程各一份），本对象无状态。
 */
object CellResampler {

    private const val AREA_BITS = 12
    private const val AREA_ONE = 1 shl AREA_BITS
    private const val LIN_BITS = 8
    private const val LIN_ONE = 1 shl LIN_BITS

    /** 一维区域平均权重表：第 i 个目标像素取源 `[start[i], start[i]+count[i])`。 */
    class Axis internal constructor(
        val cells: Int,
        val maxTaps: Int,
        val start: IntArray,
        val count: IntArray,
        val weight: IntArray,
    )

    /** 一维双线性采样表。 */
    class Linear internal constructor(
        val cells: Int,
        val i0: IntArray,
        val i1: IntArray,
        val w: IntArray,
    )

    fun axis(src: Int, dst: Int): Axis {
        require(src > 0 && dst > 0)
        val scale = src.toDouble() / dst
        val maxTaps = ceil(scale).toInt() + 1
        val start = IntArray(dst)
        val count = IntArray(dst)
        val weight = IntArray(dst * maxTaps)
        for (i in 0 until dst) {
            val s0 = i * scale
            val s1 = min(s0 + scale, src.toDouble())
            val i0 = floor(s0).toInt().coerceIn(0, src - 1)
            val i1 = min(ceil(s1).toInt().coerceIn(i0 + 1, src), i0 + maxTaps)
            var sum = 0.0
            for (k in i0 until i1) {
                sum += min(s1, (k + 1).toDouble()) - maxOf(s0, k.toDouble())
            }
            start[i] = i0
            count[i] = i1 - i0
            var acc = 0
            for (k in i0 until i1) {
                val overlap = min(s1, (k + 1).toDouble()) - maxOf(s0, k.toDouble())
                val v = (overlap / sum * AREA_ONE).roundToInt()
                weight[i * maxTaps + (k - i0)] = v
                acc += v
            }
            // 舍入残差补到第一项，保证每一列权重和恰好是 AREA_ONE
            weight[i * maxTaps] += AREA_ONE - acc
        }
        return Axis(dst, maxTaps, start, count, weight)
    }

    fun linear(src: Int, dst: Int): Linear {
        require(src > 0 && dst > 0)
        val scale = src.toDouble() / dst
        val i0 = IntArray(dst)
        val i1 = IntArray(dst)
        val w = IntArray(dst)
        for (i in 0 until dst) {
            val f = (i + 0.5) * scale - 0.5
            val a = floor(f).toInt().coerceIn(0, src - 1)
            i0[i] = a
            i1[i] = min(a + 1, src - 1)
            w[i] = ((f - a).coerceIn(0.0, 1.0) * LIN_ONE).roundToInt()
        }
        return Linear(dst, i0, i1, w)
    }

    /**
     * 区域平均：源矩形 `(x0,y0,w,h)`（`src` 的 stride = [stride]）→ `dst` 的 `dstSize²`。
     */
    fun area(
        src: IntArray,
        stride: Int,
        x0: Int,
        y0: Int,
        w: Int,
        h: Int,
        ax: Axis,
        ay: Axis,
        tmp: IntArray,
        dst: IntArray,
        dstSize: Int,
    ) {
        // 横向：h 行 → tmp[h, dstSize]
        for (y in 0 until h) {
            val row = (y0 + y) * stride + x0
            val trow = y * dstSize
            for (dx in 0 until dstSize) {
                val s = ax.start[dx]
                val n = ax.count[dx]
                val wb = dx * ax.maxTaps
                var r = 0
                var g = 0
                var b = 0
                for (k in 0 until n) {
                    val p = src[row + s + k]
                    val wgt = ax.weight[wb + k]
                    r += ((p shr 16) and 0xFF) * wgt
                    g += ((p shr 8) and 0xFF) * wgt
                    b += (p and 0xFF) * wgt
                }
                tmp[trow + dx] = pack(
                    (r + (AREA_ONE shr 1)) shr AREA_BITS,
                    (g + (AREA_ONE shr 1)) shr AREA_BITS,
                    (b + (AREA_ONE shr 1)) shr AREA_BITS,
                )
            }
        }
        // 纵向：tmp[h, dstSize] → dst[dstSize, dstSize]
        for (dy in 0 until dstSize) {
            val s = ay.start[dy]
            val n = ay.count[dy]
            val wb = dy * ay.maxTaps
            val drow = dy * dstSize
            for (x in 0 until dstSize) {
                var r = 0
                var g = 0
                var b = 0
                for (k in 0 until n) {
                    val p = tmp[(s + k) * dstSize + x]
                    val wgt = ay.weight[wb + k]
                    r += ((p shr 16) and 0xFF) * wgt
                    g += ((p shr 8) and 0xFF) * wgt
                    b += (p and 0xFF) * wgt
                }
                dst[drow + x] = pack(
                    (r + (AREA_ONE shr 1)) shr AREA_BITS,
                    (g + (AREA_ONE shr 1)) shr AREA_BITS,
                    (b + (AREA_ONE shr 1)) shr AREA_BITS,
                )
            }
        }
    }

    /** 双线性放大：源矩形 → `dstSize²` 的 ARGB（用于格子比 [64] 还小的低分辨率机器）。 */
    fun bilinearInt(
        src: IntArray,
        stride: Int,
        x0: Int,
        y0: Int,
        w: Int,
        h: Int,
        ax: Linear,
        ay: Linear,
        dst: IntArray,
        dstSize: Int,
    ) {
        for (dy in 0 until dstSize) {
            val wy = ay.w[dy]
            val iwy = LIN_ONE - wy
            val r0 = (y0 + ay.i0[dy]) * stride
            val r1 = (y0 + ay.i1[dy]) * stride
            val drow = dy * dstSize
            for (dx in 0 until dstSize) {
                val wx = ax.w[dx]
                val iwx = LIN_ONE - wx
                val w00 = iwx * iwy
                val w01 = wx * iwy
                val w10 = iwx * wy
                val w11 = wx * wy
                val p00 = src[r0 + x0 + ax.i0[dx]]
                val p01 = src[r0 + x0 + ax.i1[dx]]
                val p10 = src[r1 + x0 + ax.i0[dx]]
                val p11 = src[r1 + x0 + ax.i1[dx]]
                dst[drow + dx] = pack(
                    ((p00 shr 16 and 0xFF) * w00 + (p01 shr 16 and 0xFF) * w01 +
                        (p10 shr 16 and 0xFF) * w10 + (p11 shr 16 and 0xFF) * w11 + 32768) shr 16,
                    ((p00 shr 8 and 0xFF) * w00 + (p01 shr 8 and 0xFF) * w01 +
                        (p10 shr 8 and 0xFF) * w10 + (p11 shr 8 and 0xFF) * w11 + 32768) shr 16,
                    ((p00 and 0xFF) * w00 + (p01 and 0xFF) * w01 +
                        (p10 and 0xFF) * w10 + (p11 and 0xFF) * w11 + 32768) shr 16,
                )
            }
        }
    }

    /**
     * 双线性放大 + 通道转 RGB + `/255`，直接写进模型输入（HWC）。
     *
     * @param src 边长 [srcSize] 的方图（ARGB），`srcSize < dstSize`。
     */
    fun bilinearToFloat(
        src: IntArray,
        srcSize: Int,
        dstSize: Int,
        ax: Linear,
        out: FloatArray,
        offset: Int,
    ) {
        var o = offset
        for (dy in 0 until dstSize) {
            val wy = ax.w[dy]   // 方图：x/y 共用一张表
            val iwy = LIN_ONE - wy
            val r0 = ax.i0[dy] * srcSize
            val r1 = ax.i1[dy] * srcSize
            for (dx in 0 until dstSize) {
                val wx = ax.w[dx]
                val iwx = LIN_ONE - wx
                val w00 = iwx * iwy
                val w01 = wx * iwy
                val w10 = iwx * wy
                val w11 = wx * wy
                val p00 = src[r0 + ax.i0[dx]]
                val p01 = src[r0 + ax.i1[dx]]
                val p10 = src[r1 + ax.i0[dx]]
                val p11 = src[r1 + ax.i1[dx]]
                out[o++] = (((p00 shr 16 and 0xFF) * w00 + (p01 shr 16 and 0xFF) * w01 +
                    (p10 shr 16 and 0xFF) * w10 + (p11 shr 16 and 0xFF) * w11 + 32768) shr 16) /
                    255f
                out[o++] = (((p00 shr 8 and 0xFF) * w00 + (p01 shr 8 and 0xFF) * w01 +
                    (p10 shr 8 and 0xFF) * w10 + (p11 shr 8 and 0xFF) * w11 + 32768) shr 16) /
                    255f
                out[o++] = (((p00 and 0xFF) * w00 + (p01 and 0xFF) * w01 +
                    (p10 and 0xFF) * w10 + (p11 and 0xFF) * w11 + 32768) shr 16) / 255f
            }
        }
    }

    private fun pack(r: Int, g: Int, b: Int): Int =
        (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or
            b.coerceIn(0, 255)
}
