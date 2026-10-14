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
 * 免得在别的机型上"静默变慢"却查不出原因。"挂不上"包括兼容列表说支持、但 `Interpreter`
 * 构造时抛 `Error applying delegate` 的情况（设备支持 ≠ 这个模型能跑在它的 delegate 上）。
 *
 * 模型形状是硬约定：固定 batch=[BATCH]、输入 NHWC。GPU delegate 不支持动态 shape 的 resize，
 * 与其留一段"偷偷 resize 再退回 CPU"的代码，不如直接校验失败 —— 换 shape 就重导模型
 * （见 `BrickGame-Tool/export_tflite.py --batch`）。
 *
 * 空位格子直接给 0，不送模型（`class_ids[0]` 就是背景），省一批推理。
 *
 * **置信度只用来留档，不用来改判**：每格一律给 top-1 原型（相似度算出来只为诊断），
 * 哪一帧有低于 [DUMP_SIMILARITY] 的格子由 [lowConfidenceCells] 报上去，上层把整帧截图存下来
 * （见 `HintSession.dumpLowConfidence`），事后补标注重训。
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

    /** 上一帧没把握的格子（序号 → 相似度），见 [lowConfidenceCells]。 */
    private var lastLowConfidence: List<Pair<Int, Int>> = emptyList()

    override fun classify(frame: Bitmap, scan: BoardScan): IntArray {
        val labels = IntArray(scan.cellCount)
        val similarity = IntArray(scan.cellCount)
        val targets = (0 until scan.cellCount).filter { !scan.empty[it] }
        if (targets.isEmpty()) {
            lastLowConfidence = emptyList()
            return labels
        }

        val n = targets.size
        val bestClass = IntArray(n)

        val side = cropper.cropSide(scan)
        val bgr = CellCropper.bitmapToBgr(frame)
        var slot = 0
        try {
            for (chunk in targets.chunked(BATCH)) {
                for ((s, index) in chunk.withIndex()) {
                    cropper.fill(bgr, scan, index, side, batchInput, s * per)
                }
                // 补满：直接复制最后一张的有效段，比重新裁一次便宜；多出来的结果反正要丢掉
                if (chunk.size < BATCH) {
                    val src = (chunk.size - 1) * per
                    for (s in chunk.size until BATCH) {
                        System.arraycopy(batchInput, src, batchInput, s * per, per)
                    }
                }

                inputBuffer.rewind()
                inputBuffer.asFloatBuffer().put(batchInput)
                inputBuffer.rewind()
                interpreter.run(inputBuffer, output)

                for (k in chunk.indices) {
                    val vec = output[k]
                    // 与原型做点积（两边都已 L2 归一化 → 点积即余弦相似度），取最大
                    var best = 0
                    var bestScore = Float.NEGATIVE_INFINITY
                    for (c in 0 until nClass) {
                        val protoBase = c * dim
                        var dot = 0f
                        for (d in 0 until dim) dot += vec[d] * prototypes[protoBase + d]
                        if (dot > bestScore) {
                            bestScore = dot
                            best = c
                        }
                    }
                    bestClass[slot] = meta.classIds[best]
                    similarity[targets[slot]] = (bestScore * 100).roundToInt()
                    slot++
                }
            }
        } finally {
            bgr.release()
        }
        writeLabels(labels, targets, bestClass)
        lastLowConfidence = targets.filter { similarity[it] < DUMP_SIMILARITY }.map { it to similarity[it] }
        return labels
    }

    /**
     * 把识别结果写回格子：**一律采信 top-1 原型**，运行时不做置信度判断。
     *
     * 曾经对低分格子做过"帧内找最像的格子纠错 / 否则给唯一未知号"的处理，结果是把真结果藏了起来
     * —— 低分格照样有 top-1，直接给出来更透明；哪一帧没把握由 [lowConfidenceCells] 报给上层，
     * 由上层把整帧截图留档（事后补标注重训），而不是在运行时改判。
     */
    private fun writeLabels(labels: IntArray, targets: List<Int>, bestClass: IntArray) {
        for (i in targets.indices) labels[targets[i]] = bestClass[i]
    }

    /**
     * 上一帧"模型没把握"的格子（行优先序号 → 相似度百分制，只含送过模型的格子）。
     *
     * 阈值取 [DUMP_SIMILARITY]（85）：正常同款能到 90+，低于它的多半是模型没见过的样子。
     * 运行时不改判，只用来决定"这一帧要不要留档"，见 `HintSession.dumpLowConfidence`。
     */
    override fun lowConfidenceCells(): List<Pair<Int, Int>> = lastLowConfidence

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

        /**
         * 相似度（百分制）低于它就把**整帧截图留档**（补标注重训用）。
         *
         * 取值偏灵敏：正常同款实测 90+，85 以下就有可能是同一款砖的另一种样子，
         * 宁可多留几张，也别漏掉要补标注的那种没见过的情况。
         */
        private const val DUMP_SIMILARITY = 85

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

            var gpu = createGpuDelegate()
            val interpreter = try {
                Interpreter(modelBufferOf(bytes), optionsFor(gpu))
            } catch (t: Throwable) {
                if (gpu == null) throw t
                // "本机支持"只管设备，不管这个模型：真机上出现过兼容列表说支持、挂上去却
                // 抛 IllegalArgumentException: Error applying delegate（驱动/算子不支持）。
                // 这种情况必须退回 CPU，否则每帧都在识别器初始化上崩。
                Log.w(TAG, "GPU delegate 挂载失败 → 退回 CPU $DEFAULT_THREADS 线程", t)
                runCatching { gpu.close() }
                gpu = null
                // 缓冲要重新建：失败的 Interpreter 可能已经动过它
                Interpreter(modelBufferOf(bytes), optionsFor(null))
            }

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

        /** 每次 `Interpreter` 构造都用一块新的直接缓冲：构造失败时旧缓冲可能已被 native 动过。 */
        private fun modelBufferOf(bytes: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                put(bytes)
                rewind()
            }

        private fun optionsFor(gpu: GpuDelegate?): Interpreter.Options = Interpreter.Options().apply {
            setNumThreads(DEFAULT_THREADS)
            gpu?.let { addDelegate(it) }
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
