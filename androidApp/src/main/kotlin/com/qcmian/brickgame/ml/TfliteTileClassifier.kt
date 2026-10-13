package com.qcmian.brickgame.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.TileClassifier
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * 身份识别：每格裁图 → MobileNetV3 出 256 维身份向量 → 与 [ModelMeta.nClass] 条原型比余弦 → 最近的那类。
 *
 * 真机实测（骁龙 865，批 32 张）GPU delegate 229ms 是最快的一条路（ONNX 371ms、XNNPACK 252ms），
 * 所以这里只走 GPU：**挂不上 GPU 才退回 CPU（XNNPACK）**，并在日志和名字里写明，
 * 免得在别的机型上"静默变慢"却查不出原因。
 *
 * 模型形状是硬约定：固定 batch=[BATCH]、输入 NHWC。GPU delegate 不支持动态 shape 的 resize，
 * 与其留一段"偷偷 resize 再退回 CPU"的代码，不如直接校验失败 —— 换 shape 就重导模型
 * （见 `BrickGame-Tool/export_tflite.py --batch`）。
 *
 * 空位格子直接给 0，不送模型（`class_ids[0]` 就是背景），省一批推理。
 */
class TfliteTileClassifier private constructor(
    private val interpreter: Interpreter,
    private val prototypes: FloatArray,
    private val meta: ModelMeta,
    private val threads: Int,
    /** 非空表示这次真的挂在 GPU 上；空 = 退回 CPU（XNNPACK）。 */
    private val gpuDelegate: GpuDelegate?,
) : TileClassifier, Closeable {

    private val dim = meta.embedDim
    private val nClass = meta.nClass
    private val img = meta.inputSize

    private val cropper = CellCropper(img)
    private val per = cropper.floatsPerTile

    val usingGpu: Boolean get() = gpuDelegate != null

    override val name: String =
        "TFLite 身份向量（$nClass 类，$dim 维，${if (usingGpu) "GPU" else "CPU ${threads} 线程"}）"

    /** 一批的输入（HWC 交错）。复用同一块，避免每批 9.8MB 的新分配。 */
    private val batchInput = FloatArray(BATCH * per)

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(BATCH * per * 4).order(ByteOrder.nativeOrder())

    private val output = Array(BATCH) { FloatArray(dim) }

    /** 上一帧每格最高余弦相似度（百分制，空位记 0），只给日志用。 */
    private var lastSimilarity: IntArray? = null
    private var lastDims: Pair<Int, Int>? = null

    override fun classify(frame: Bitmap, scan: BoardScan): IntArray {
        val labels = IntArray(scan.cellCount)
        val similarity = IntArray(scan.cellCount)
        val targets = (0 until scan.cellCount).filter { !scan.empty[it] }
        if (targets.isEmpty()) {
            lastSimilarity = similarity
            lastDims = scan.rows to scan.cols
            return labels
        }

        val side = cropper.cropSide(scan)
        val bgr = CellCropper.bitmapToBgr(frame)
        try {
            for (chunk in targets.chunked(BATCH)) {
                for ((slot, index) in chunk.withIndex()) {
                    cropper.fill(bgr, scan, index, side, batchInput, slot * per)
                }
                // 补满：直接复制最后一张的有效段，比重新裁一次便宜；多出来的结果反正要丢掉
                if (chunk.size < BATCH) {
                    val src = (chunk.size - 1) * per
                    for (slot in chunk.size until BATCH) {
                        System.arraycopy(batchInput, src, batchInput, slot * per, per)
                    }
                }

                inputBuffer.rewind()
                inputBuffer.asFloatBuffer().put(batchInput)
                inputBuffer.rewind()
                interpreter.run(inputBuffer, output)

                for ((k, index) in chunk.withIndex()) {
                    labels[index] = nearestClass(output, k, similarity, index)
                }
            }
        } finally {
            bgr.release()
        }
        lastSimilarity = similarity
        lastDims = scan.rows to scan.cols
        return labels
    }

    /**
     * 上一帧的相似度矩阵 + 汇总，用来区分"格线切歪"和"模型/原型有问题"：
     * 分数普遍低 = 裁出来的是模型没见过的东西（几何错），分数高但类别乱 = 识别本身的问题。
     */
    override fun diagnostics(): List<String> {
        val sim = lastSimilarity ?: return emptyList()
        val (rows, cols) = lastDims ?: return emptyList()
        if (rows <= 0 || cols <= 0 || sim.size < rows * cols) return emptyList()
        val out = ArrayList<String>(rows + 1)
        var min = 100
        var low = 0
        var counted = 0
        for (r in 0 until rows) {
            val from = r * cols
            out += "sim r${r.toString().padStart(2, '0')} " + (0 until cols).joinToString(" ") { sim[from + it].toString() }
            for (c in 0 until cols) {
                val v = sim[from + c]
                if (v <= 0) continue      // 空位没送模型
                counted++
                if (v < min) min = v
                if (v < LOW_SIMILARITY) low++
            }
        }
        out += "sim 汇总 非空=$counted 最低=$min 低于 ${LOW_SIMILARITY} 的格子=$low"
        return out
    }

    /** 与原型做点积（两边都已 L2 归一化 → 点积即余弦相似度），取最大。 */
    private fun nearestClass(embeddings: Array<FloatArray>, row: Int, similarity: IntArray, index: Int): Int {
        val vec = embeddings[row]
        var best = 0
        var bestScore = Float.NEGATIVE_INFINITY
        for (c in 0 until nClass) {
            val protoBase = c * dim
            var dot = 0f
            for (k in 0 until dim) dot += vec[k] * prototypes[protoBase + k]
            if (dot > bestScore) {
                bestScore = dot
                best = c
            }
        }
        similarity[index] = (bestScore * 100).roundToInt()
        return meta.classIds[best]
    }

    override fun close() {
        cropper.release()
        interpreter.close()
        // delegate 要晚于 interpreter 释放：Interpreter 还引用着它
        gpuDelegate?.let { runCatching { it.close() } }
    }

    companion object {
        private const val TAG = "TfliteTileClassifier"

        private const val MODEL_ASSET = "brick_embedder.tflite"

        /** 固定批大小，与 `export_tflite.py --batch` 的默认值必须一致。 */
        private const val BATCH = 32

        /** 退回 CPU 时的线程数。 */
        private const val DEFAULT_THREADS = 4

        /** 相似度（百分制）低于它就算"模型拿不准"：正常同款能到 90+ 甚至 99。 */
        private const val LOW_SIMILARITY = 70

        /** 本机能不能用 GPU delegate（界面据此显示"GPU"还是"CPU 回退"）。 */
        fun gpuSupported(): Boolean = try {
            CompatibilityList().isDelegateSupportedOnThisDevice
        } catch (t: Throwable) {
            Log.w(TAG, "查询 GPU delegate 支持失败", t)
            false
        }

        fun load(context: Context): TfliteTileClassifier {
            val meta = ModelAssets.readMeta(context)
            val prototypes = ModelAssets.readPrototypes(context, meta.nClass, meta.embedDim)

            val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            val modelBuffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            modelBuffer.put(bytes).rewind()

            val gpu = createGpuDelegate()
            val options = Interpreter.Options().apply {
                setNumThreads(DEFAULT_THREADS)
                gpu?.let { addDelegate(it) }
            }
            val interpreter = Interpreter(modelBuffer, options)

            try {
                val shape = interpreter.getInputTensor(0).shape()
                require(
                    shape.size == 4 && shape[0] == BATCH &&
                        shape[1] == meta.inputSize && shape[2] == meta.inputSize && shape[3] == 3,
                ) {
                    "TFLite 输入 shape ${shape.toList()} 不是固定 " +
                        "[$BATCH, ${meta.inputSize}, ${meta.inputSize}, 3]（用 export_tflite.py 重导）"
                }
                val outShape = interpreter.getOutputTensor(0).shape()
                require(outShape.size == 2 && outShape[1] == meta.embedDim) {
                    "TFLite 输出 shape ${outShape.toList()} 与 embed_dim=${meta.embedDim} 不符"
                }
            } catch (t: Throwable) {
                runCatching { interpreter.close() }
                gpu?.let { runCatching { it.close() } }
                throw t
            }

            Log.i(
                TAG,
                "加载完成：输入 ${interpreter.getInputTensor(0).shape().toList()} " +
                    "输出 ${interpreter.getOutputTensor(0).shape().toList()} " +
                    (if (gpu != null) "GPU delegate" else "CPU $DEFAULT_THREADS 线程"),
            )
            return TfliteTileClassifier(interpreter, prototypes, meta, DEFAULT_THREADS, gpu)
        }

        /** @return null = 本机不支持或创建失败，照 CPU 跑（这是唯一的兜底，别再往下退）。 */
        private fun createGpuDelegate(): GpuDelegate? = try {
            val compat = CompatibilityList()
            if (!compat.isDelegateSupportedOnThisDevice) {
                Log.w(TAG, "本机不支持 GPU delegate → 用 CPU")
                null
            } else {
                val opts = compat.bestOptionsForThisDevice ?: GpuDelegate.Options()
                // 允许用 fp16 算：MobileNet 这类模型掉精度可忽略，但显存带宽省一半
                opts.setPrecisionLossAllowed(true)
                GpuDelegate(opts)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate 创建失败 → 用 CPU", t)
            null
        }
    }
}
