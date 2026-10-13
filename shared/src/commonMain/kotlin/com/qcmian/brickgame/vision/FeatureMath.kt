package com.qcmian.brickgame.vision

import kotlin.math.sqrt

/**
 * 一帧所有格子的裁切图，行优先排列。
 *
 * [pixels] 是 `rows*cols*cell*cell*3` 的浮点像素，取值 0..255，**通道顺序固定为 BGR**
 * （与 OpenCV / `tile_recognizer.py` 的 `cv2.imread` 一致，[FeatureMath.cellFlatness] 的灰度权重依赖它）。
 */
class TileFrame(
    val rows: Int,
    val cols: Int,
    val cell: Int,
    val pixels: FloatArray,
) {
    val count: Int get() = rows * cols
    val valuesPerPatch: Int get() = cell * cell * 3

    fun patchOffset(index: Int): Int = index * valuesPerPatch

    init {
        require(pixels.size == count * valuesPerPatch) {
            "像素数量不符：期望 ${count * valuesPerPatch}，实际 ${pixels.size}"
        }
    }
}

/**
 * `tile_recognizer.py` 里"判空"那一步的 Kotlin 等价实现。
 *
 * 历史上这里还有一整套"图标特征 + 帧内聚类"（给没接模型时的像素聚类识别用）。
 * 识别已经全部交给 GPU 上的模型，那条路连同它的 golden 对拍测试一起删了；
 * 哪天真要退回聚类，从 git 历史里捞比在这里养着死代码强。
 */
object FeatureMath {

    /**
     * 每格"内部有没有东西"：格子内部（去掉外圈 margin）的 Sobel 边缘能量均值。
     *
     * 为什么不用"图标对比度"判空：已消除的格子只剩棋盘底色 → 内部完全平坦
     * （实测边缘能量 0.0~0.8）；任何真砖块内部都有结构（实测最低 31）。
     * 两者相差 40 倍且无重叠，所以用"平不平"判空稳得多。
     */
    fun cellFlatness(frame: TileFrame, margin: Double = 0.18): FloatArray {
        val cell = frame.cell
        val out = FloatArray(frame.count)
        val gray = DoubleArray(cell * cell)

        val lo = (cell * margin).toInt()
        val hi = (cell * (1 - margin)).toInt()
        val iw = hi - lo
        if (iw <= 0) return out

        for (idx in 0 until frame.count) {
            val off = frame.patchOffset(idx)
            for (p in 0 until cell * cell) {
                val d = off + p * 3
                // OpenCV COLOR_BGR2GRAY 权重；Python 是 `p.astype(np.uint8)`（截断）后再转灰度，
                // 所以这里同样截断而不是四舍五入。
                gray[p] = (
                    0.114 * frame.pixels[d] +
                        0.587 * frame.pixels[d + 1] +
                        0.299 * frame.pixels[d + 2]
                    ).toInt().toDouble()
            }

            var sum = 0.0
            for (y in 0 until iw) {
                for (x in 0 until iw) {
                    val gx = sobelX(gray, cell, lo + x, lo + y)
                    val gy = sobelY(gray, cell, lo + x, lo + y)
                    sum += sqrt(gx * gx + gy * gy)
                }
            }
            out[idx] = (sum / (iw * iw)).toFloat()
        }
        return out
    }

    // Sobel ksize=3：x 方向列核 [-1,0,1] × 行权 [1,2,1]；y 方向行核 [-1,0,1] × 列权 [1,2,1]。
    // 边界按 BORDER_REFLECT_101 取。只取模长，所以整体符号不影响结果。
    private val SOBEL_COL = intArrayOf(-1, 0, 1)
    private val SOBEL_ROW = intArrayOf(-1, 0, 1)
    private val SOBEL_W = intArrayOf(1, 2, 1)

    private fun sobelX(g: DoubleArray, cell: Int, x: Int, y: Int): Double {
        var s = 0.0
        for (j in -1..1) {
            for (i in -1..1) {
                s += SOBEL_COL[i + 1] * SOBEL_W[j + 1] * at(g, cell, x + i, y + j)
            }
        }
        return s
    }

    private fun sobelY(g: DoubleArray, cell: Int, x: Int, y: Int): Double {
        var s = 0.0
        for (j in -1..1) {
            for (i in -1..1) {
                s += SOBEL_W[i + 1] * SOBEL_ROW[j + 1] * at(g, cell, x + i, y + j)
            }
        }
        return s
    }

    /** BORDER_REFLECT_101：`fedcba|abcdefgh|hgfedcb`。 */
    private fun at(g: DoubleArray, cell: Int, x: Int, y: Int): Double {
        return g[reflect101(y, cell) * cell + reflect101(x, cell)]
    }

    private fun reflect101(v: Int, n: Int): Int {
        if (n == 1) return 0
        var r = v
        while (r < 0 || r >= n) {
            if (r < 0) r = -r else r = 2 * n - 2 - r
        }
        return r
    }
}
