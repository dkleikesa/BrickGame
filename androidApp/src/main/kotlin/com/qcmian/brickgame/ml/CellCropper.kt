package com.qcmian.brickgame.ml

import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.CellResampler
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 格子 → 模型输入的预处理。**纯 Kotlin，不依赖 OpenCV。**
 *
 * 规格必须与训练严格对齐（三处分别抄自 Python 侧）：
 *  1. **裁切**：`build_dataset.frame_cells` —— 以格心为中心裁
 *     `round(pitch * (1 - 2*OUT_INSET))` 的正方形，`OUT_INSET = 0.02`；
 *  2. **缩放**：先缩到 [CROP]（`build_dataset.cut_tile`：缩小用区域平均、放大用三次插值），
 *     再缩到模型输入边长（`train_classifier.to_tensor_batch` 的 `cv2.resize` 默认线性插值）；
 *  3. **通道与值域**：BGR→RGB、0~1，HWC 交错（TFLite 的 NHWC 输入）。
 *     ImageNet 的 mean/std 已经烘进模型，**这里不要再做归一化**。
 *
 * 性能（相对第一版纯 Kotlin 实现）：
 *  - 整帧只 `getPixels` 一次（原来每格一次：135 次 JNI + 每格一次拷贝），
 *    裁格直接从整帧缓冲按 stride 取；
 *  - 重采样交给 `CellResampler` 的定点权重表，热点循环里没有 `Double`；
 *  - 权重表按 `(宽, 高)` 缓存，一帧只算一次（贴边被裁的格子最多多一两个尺寸）。
 *
 * 非线程安全（复用 `mid`/`tmp` 缓冲）：调用方每线程各持一份。
 */
internal class CellCropper(val inputSize: Int) {

    /** 一格的 float 数量（HWC 交错，3 通道）。 */
    val floatsPerTile = inputSize * inputSize * 3

    /** 缩到 64×64 的中间图，训练时真实存在的那一跳。 */
    private val mid = IntArray(CROP * CROP)

    /** 区域平均横向那一趟的中间产物（h × CROP），按需扩容。 */
    private var tmp = IntArray(0)

    private val areaW = HashMap<Int, CellResampler.Axis>()
    private val areaH = HashMap<Int, CellResampler.Axis>()
    private val linW = HashMap<Int, CellResampler.Linear>()
    private val linH = HashMap<Int, CellResampler.Linear>()

    /** 64 → 模型输入边长（方图，x/y 共用一张表）。 */
    private val lin64 = CellResampler.linear(CROP, inputSize)

    /** 与 `build_dataset.frame_cells` 同式：格距中位数 × (1 - 2*OUT_INSET)。 */
    fun cropSide(scan: BoardScan): Int {
        val pitches = DoubleArray(scan.cols + scan.rows)
        for (i in 0 until scan.cols) pitches[i] = scan.xlines[i + 1] - scan.xlines[i]
        for (j in 0 until scan.rows) pitches[scan.cols + j] = scan.ylines[j + 1] - scan.ylines[j]
        pitches.sort()
        val median = if (pitches.size % 2 == 1) {
            pitches[pitches.size / 2]
        } else {
            0.5 * (pitches[pitches.size / 2 - 1] + pitches[pitches.size / 2])
        }
        return max(8, (median * (1 - 2 * OUT_INSET)).roundToInt())
    }

    /**
     * 把 [scan] 的第 [index] 格裁好、缩放、转成 RGB 0~1 后，
     * 写进 `out[offset until offset + floatsPerTile]`。
     *
     * @param pixels 整帧 ARGB（`frameWidth × frameHeight`，由调用方每帧取一次）
     */
    fun fill(
        pixels: IntArray,
        frameWidth: Int,
        frameHeight: Int,
        scan: BoardScan,
        index: Int,
        side: Int,
        out: FloatArray,
        offset: Int,
    ) {
        val point = scan.pointAt(index)
        val cx = ((scan.xlines[point.col] + scan.xlines[point.col + 1]) / 2.0).roundToInt()
        val cy = ((scan.ylines[point.row] + scan.ylines[point.row + 1]) / 2.0).roundToInt()
        val a = (cx - side / 2.0).roundToInt().coerceIn(0, frameWidth - 2)
        val c = (cy - side / 2.0).roundToInt().coerceIn(0, frameHeight - 2)
        val w = minOf(frameWidth, a + side) - a
        val h = minOf(frameHeight, c + side) - c
        if (w <= 1 || h <= 1) {
            out.fill(0f, offset, offset + floatsPerTile)
            return
        }

        if (h > CROP) {
            val ax = areaW.getOrPut(w) { CellResampler.axis(w, CROP) }
            val ay = areaH.getOrPut(h) { CellResampler.axis(h, CROP) }
            if (tmp.size < h * CROP) tmp = IntArray(h * CROP)
            CellResampler.area(pixels, frameWidth, a, c, w, h, ax, ay, tmp, mid, CROP)
        } else {
            val ax = linW.getOrPut(w) { CellResampler.linear(w, CROP) }
            val ay = linH.getOrPut(h) { CellResampler.linear(h, CROP) }
            CellResampler.bilinearInt(pixels, frameWidth, a, c, w, h, ax, ay, mid, CROP)
        }
        CellResampler.bilinearToFloat(mid, CROP, inputSize, lin64, out, offset)
    }

    companion object {
        /** 与 `build_dataset.TILE` 一致。 */
        private const val CROP = 64
        private const val OUT_INSET = 0.02
    }
}
