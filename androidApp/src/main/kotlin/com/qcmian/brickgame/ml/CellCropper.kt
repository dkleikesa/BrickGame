package com.qcmian.brickgame.ml

import android.graphics.Bitmap
import com.qcmian.brickgame.vision.BoardScan
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 格子 → 模型输入的预处理。
 *
 * 规格必须与训练严格对齐（三处分别抄自 Python 侧），错了模型再好也白搭：
 *  1. **裁切**：`build_dataset.frame_cells` —— 以格心为中心裁
 *     `round(pitch * (1 - 2*OUT_INSET))` 的正方形，`OUT_INSET = 0.02`
 *     （几乎不内缩，保留底板/边框：砖块状态就在底色里）；
 *  2. **缩放**：先缩到 [CROP]（`build_dataset.cut_tile`：缩小用 INTER_AREA、放大用 INTER_CUBIC），
 *     再缩到模型输入边长（`train_classifier.to_tensor_batch` 的 `cv2.resize` 默认 INTER_LINEAR）
 *     —— 两步都要做，中间那次 64 的量化是训练时真实发生的；
 *  3. **通道与值域**：`train_classifier.to_tensor` 是 `batch[..., ::-1] / 255`，即 BGR→RGB、0~1，
 *     且按 HWC 交错摆放（TFLite 的 NHWC 输入，也是 XNNPACK / GPU delegate 的原生布局）；
 *     ImageNet 的 mean/std 已经烘进模型，**这里不要再做归一化**。
 *
 * 非线程安全（内部复用 OpenCV Mat），调用方要保证串行使用。
 */
internal class CellCropper(val inputSize: Int) {

    /** 一格的 float 数量（HWC 交错，3 通道）。 */
    val floatsPerTile = inputSize * inputSize * 3

    private val tmp64 = Mat()
    private val tmpInput = Mat()
    private val tmpRgb = Mat()
    private val bytes = ByteArray(inputSize * inputSize * 3)

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
     * 把 `scan` 的第 [index] 格裁好、缩放、转成 RGB 0~1 后，
     * 写进 `out[offset until offset + floatsPerTile]`。
     */
    fun fill(bgr: Mat, scan: BoardScan, index: Int, side: Int, out: FloatArray, offset: Int) {
        val point = scan.pointAt(index)
        val cx = ((scan.xlines[point.col] + scan.xlines[point.col + 1]) / 2.0).roundToInt()
        val cy = ((scan.ylines[point.row] + scan.ylines[point.row + 1]) / 2.0).roundToInt()
        val a = (cx - side / 2.0).roundToInt().coerceIn(0, bgr.cols() - 2)
        val c = (cy - side / 2.0).roundToInt().coerceIn(0, bgr.rows() - 2)
        val b = minOf(bgr.cols(), a + side)
        val d = minOf(bgr.rows(), c + side)

        val cell = bgr.submat(c, d, a, b)
        val downscale = if (cell.rows() > CROP) Imgproc.INTER_AREA else Imgproc.INTER_CUBIC
        Imgproc.resize(cell, tmp64, Size(CROP.toDouble(), CROP.toDouble()), 0.0, 0.0, downscale)
        Imgproc.resize(
            tmp64, tmpInput, Size(inputSize.toDouble(), inputSize.toDouble()), 0.0, 0.0,
            Imgproc.INTER_LINEAR,
        )
        Imgproc.cvtColor(tmpInput, tmpRgb, Imgproc.COLOR_BGR2RGB)
        tmpRgb.get(0, 0, bytes)

        // bytes 本身就是 HWC 交错的 RGB，顺序拷一遍即可
        for (i in bytes.indices) out[offset + i] = (bytes[i].toInt() and 0xFF) / 255f
    }

    fun release() {
        tmp64.release()
        tmpInput.release()
        tmpRgb.release()
    }

    companion object {
        /** 与 `build_dataset.TILE` 一致。 */
        private const val CROP = 64
        private const val OUT_INSET = 0.02

        /**
         * 整帧截图 → BGR。输入任意位图配置，输出 3 通道 BGR（[fill] 只接受 BGR）。
         *
         * @return BGR Mat，**调用方负责 release**（它不是内部复用的对象）。
         */
        fun bitmapToBgr(frame: Bitmap): Mat {
            val rgba = Mat()
            val bgr = Mat()
            try {
                Utils.bitmapToMat(frame, rgba)
                when (rgba.channels()) {
                    4 -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
                    3 -> rgba.copyTo(bgr)
                    else -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_GRAY2BGR)
                }
                return bgr
            } finally {
                rgba.release()
            }
        }
    }
}
