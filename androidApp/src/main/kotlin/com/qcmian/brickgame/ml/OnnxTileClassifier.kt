package com.qcmian.brickgame.ml

import android.content.Context
import android.graphics.Bitmap
import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.TileClassifier
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 接训练好的身份模型：每格 → 256 维身份向量 → 与 2369 条原型比余弦 → 最近的那类。
 *
 * 预处理必须与训练严格对齐，否则模型再好也白搭（三处规格分别抄自 Python 侧）：
 *  1. **裁切**：`build_dataset.frame_cells` —— 以格心为中心裁 `round(pitch * (1 - 2*OUT_INSET))`
 *     的正方形，`OUT_INSET = 0.02`（几乎不内缩，保留底板/边框：砖块状态就在底色里）；
 *  2. **缩放**：先缩到 `CROP = 64`（`build_dataset.cut_tile`：缩小用 INTER_AREA、放大用 INTER_CUBIC），
 *     再缩到模型输入边长（`train_classifier.to_tensor_batch` 的 `cv2.resize` 默认 INTER_LINEAR）
 *     —— 两步都要做，中间那次 64 的量化是训练时真实发生的；
 *  3. **通道与值域**：`train_classifier.to_tensor` 是 `batch[..., ::-1] / 255`，即 BGR→RGB、0~1；
 *     ImageNet 的 mean/std 已经烘进 ONNX，**这里不要再做归一化**。
 *
 * 空位格子直接给 0，不送模型：`meta.class_ids[0] = 0` 且 entry 就是 `bg01`，
 * 与 [com.qcmian.brickgame.vision.BoardState]"0 = 空位"的约定天然一致，还能省掉一批推理。
 */
class OnnxTileClassifier private constructor(
    private val model: BrickEmbedderModel,
    private val prototypes: FloatArray,
) : TileClassifier, Closeable {

    private val dim = model.meta.embedDim
    private val nClass = model.meta.nClass
    private val img = model.meta.inputSize

    override val name: String = "ONNX 身份向量（$nClass 类，${model.meta.embedDim} 维）"

    private val tmp64 = Mat()
    private val tmpInput = Mat()
    private val tmpRgb = Mat()
    private val bytes = ByteArray(img * img * 3)
    private val plane = img * img

    override fun classify(frame: Bitmap, scan: BoardScan): IntArray {
        val labels = IntArray(scan.cellCount)
        val targets = (0 until scan.cellCount).filter { !scan.empty[it] }
        if (targets.isEmpty()) return labels

        val side = cropSide(scan)
        val rgba = Mat()
        val bgr = Mat()
        try {
            Utils.bitmapToMat(frame, rgba)
            when (rgba.channels()) {
                4 -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
                3 -> rgba.copyTo(bgr)
                else -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_GRAY2BGR)
            }

            for (chunk in targets.chunked(BATCH)) {
                val nchw = FloatArray(chunk.size * 3 * plane)
                for ((k, index) in chunk.withIndex()) {
                    fillCell(bgr, scan, index, side, nchw, k * 3 * plane)
                }
                val embeddings = model.embedBatch(nchw, chunk.size)
                for ((k, index) in chunk.withIndex()) {
                    labels[index] = nearestClass(embeddings, k)
                }
            }
        } finally {
            rgba.release()
            bgr.release()
        }
        return labels
    }

    /** 与 `build_dataset.frame_cells` 同式：格距中位数 × (1 - 2*0.02)。 */
    private fun cropSide(scan: BoardScan): Int {
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

    private fun fillCell(bgr: Mat, scan: BoardScan, index: Int, side: Int, out: FloatArray, offset: Int) {
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
            tmp64, tmpInput, Size(img.toDouble(), img.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR,
        )
        Imgproc.cvtColor(tmpInput, tmpRgb, Imgproc.COLOR_BGR2RGB)
        tmpRgb.get(0, 0, bytes)

        // HWC(RGB, 0~255) → NCHW(0~1)，与训练侧 permute(0,3,1,2)/255 等价
        for (y in 0 until img) {
            for (x in 0 until img) {
                val base = (y * img + x) * 3
                val at = y * img + x
                out[offset + at] = (bytes[base].toInt() and 0xFF) / 255f
                out[offset + plane + at] = (bytes[base + 1].toInt() and 0xFF) / 255f
                out[offset + 2 * plane + at] = (bytes[base + 2].toInt() and 0xFF) / 255f
            }
        }
    }

    /** 与原型做点积（两边都已 L2 归一化 → 点积即余弦相似度），取最大。 */
    private fun nearestClass(embeddings: FloatArray, row: Int): Int {
        val base = row * dim
        var best = 0
        var bestScore = Float.NEGATIVE_INFINITY
        for (c in 0 until nClass) {
            val protoBase = c * dim
            var dot = 0f
            for (k in 0 until dim) dot += embeddings[base + k] * prototypes[protoBase + k]
            if (dot > bestScore) {
                bestScore = dot
                best = c
            }
        }
        return model.meta.classIds[best]
    }

    override fun close() {
        tmp64.release()
        tmpInput.release()
        tmpRgb.release()
        model.close()
    }

    companion object {
        private const val TAG = "OnnxTileClassifier"

        /** 与 `build_dataset.TILE` / `OUT_INSET` 一致。 */
        const val CROP = 64
        const val OUT_INSET = 0.02

        /** 一帧 100+ 格，分块跑既能吃满 batch，又不会一次要 40MB 以上的张量。 */
        const val BATCH = 32

        fun load(context: Context): OnnxTileClassifier {
            val model = BrickEmbedderModel.load(context)
            val prototypes = readPrototypes(context, model.meta.nClass, model.meta.embedDim)
            return OnnxTileClassifier(model, prototypes)
        }

        private fun readPrototypes(context: Context, nClass: Int, dim: Int): FloatArray {
            val expected = nClass * dim * 4
            val raw = context.assets.open(BrickEmbedderModel.PROTOTYPES_ASSET).use { it.readBytes() }
            require(raw.size == expected) {
                "prototypes.bin 大小 ${raw.size} 与 (nClass=$nClass x dim=$dim x 4) = $expected 不符"
            }
            val out = FloatArray(nClass * dim)
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
            return out
        }
    }
}
