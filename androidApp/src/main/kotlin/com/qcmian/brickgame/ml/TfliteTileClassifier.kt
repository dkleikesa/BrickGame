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
 * **当前跑纯 CPU（XNNPACK）**：挂上 GPU delegate 后这个模型只有 24% 的算子能上 GPU，
 * 整图被切成 3 个 GPU 分区 + 25 个 CPU 分区，来回搬张量反而把推理拖到一批 1.6 秒
 * （一帧 8.1s，占整帧 72%）。判定依据与回退方法见 [PREFER_GPU] 的注释，那段有真机日志。
 *
 * 模型形状是硬约定：固定 batch=[BATCH]、输入 NHWC。GPU delegate 不支持动态 shape 的 resize，
 * 与其留一段"偷偷 resize 再退回 CPU"的代码，不如直接校验失败 —— 换 shape 就重导模型
 * （见 `BrickGame-Tool/export_tflite.py --batch`）。
 *
 * 空位格子直接给 0，不送模型（`class_ids[0]` 就是背景），省一批推理。
 *
 * **置信度只用来留档，不用来改判**：每格一律给 top-1 原型（相似度算出来只为诊断），
 * 哪一帧有低于 [ModelMeta.dumpSimilarity] 的格子由 [lowConfidenceCells] 报上去，
 * 上层把整帧截图存下来（见 `LowConfStore`），事后补标注重训。
 *
 * 内置多个模型（[ModelVariant]），每个变体各有一套资产与自己的留档阈值；换模型时
 * 上是**重建一份解释器**（见 `HintSession.acquireClassifier`），所以这个类本身是无状态的单份实例。
 */
class TfliteTileClassifier private constructor(
    private val interpreter: Interpreter,
    private val prototypes: FloatArray,
    private val meta: ModelMeta,
    private val threads: Int,
    /** 非空表示这次真的挂在 GPU 上；空 = 退回 CPU（XNNPACK）。 */
    private val gpuDelegate: GpuDelegate?,
    /** 挂在哪个变体上：名字里要写出来，否则两个模型的语言日志分不清。 */
    private var variant: ModelVariant,
) : TileClassifier, Closeable {

    private val dim = meta.embedDim
    private val nClass = meta.nClass
    private val img = meta.inputSize

    /** 检索是否已并进模型（见 [ModelMeta.fusedSearch]）：决定了输出宽度与要不要跑点积。 */
    private val fused = meta.fusedSearch

    /** 模型输出宽度：融合时是类数（每行就是各类相似度），否则是身份向量维度。 */
    private val outDim = meta.outDim

    private val cropper = CellCropper(img)

    /**
     * 裁格并行：135 格串行要 200ms，主要开销是一次次 JNI + resize。
     *
     * CellCropper 内部复用像素缓冲，不能跨线程共享，所以每线程各持一份（ThreadLocal）。
     * 目标数组按格号分片写入，互不重叠，不需要额外同步。
     */
    private val cropThreads = 4
    private val cropPool = java.util.concurrent.Executors.newFixedThreadPool(cropThreads) { r ->
        Thread(r, "brick-crop").apply { isDaemon = true }
    }
    private val threadCropper = ThreadLocal.withInitial { CellCropper(img) }
    private val per = cropper.floatsPerTile

    val usingGpu: Boolean get() = gpuDelegate != null

    override val name: String =
        "TFLite ${variant.label}（$nClass 类，${if (fused) "检索已融合" else "$dim 维"}，" +
            "${if (usingGpu) "GPU" else "CPU ${threads} 线程"}）"

    override val label: String = variant.label

    /** 留档阈值：这个模型契约里的 `thresholds.dist_wrong_p5` 换算出来的相似度下限。 */
    private val dumpSimilarity = meta.dumpSimilarity

    /** 整帧 ARGB：每帧只 getPixels 一次，裁格按 stride 直接取（原来每格一次 JNI）。 */
    private var framePixels = IntArray(0)

    /** 一批的输入（HWC 交错）。复用同一块，避免每批 9.8MB 的新分配。 */
    private val batchInput = FloatArray(BATCH * per)

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(BATCH * per * 4).order(ByteOrder.nativeOrder())

    private val output = Array(BATCH) { FloatArray(outDim) }

    /** 上一帧没把握的格子（序号 → 相似度），见 [lowConfidenceCells]。 */
    private var lastLowConfidence: List<Pair<Int, Int>> = emptyList()

    /** 上一次分类的耗时分解，见 [phaseTimings]。 */
    private var lastPhases: List<Pair<String, Long>> = emptyList()

    override fun classify(frame: Bitmap, scan: BoardScan): IntArray {
        val labels = IntArray(scan.cellCount)
        val similarity = IntArray(scan.cellCount)
        val targets = (0 until scan.cellCount).filter { !scan.empty[it] }
        if (targets.isEmpty()) {
            lastLowConfidence = emptyList()
            lastPhases = emptyList()
            return labels
        }

        val n = targets.size
        val bestClass = IntArray(n)

        val tConvert = System.nanoTime()
        val side = cropper.cropSide(scan)
        val fw = frame.width
        val fh = frame.height
        if (framePixels.size < fw * fh) framePixels = IntArray(fw * fh)
        frame.getPixels(framePixels, 0, fw, 0, 0, fw, fh)
        val convertNs = System.nanoTime() - tConvert

        var cropNs = 0L
        var inferNs = 0L
        var searchNs = 0L
        var slot = 0
        for (chunk in targets.chunked(BATCH)) {
            val tCrop = System.nanoTime()
            val jobs = chunk.mapIndexed { s, index ->
                cropPool.submit { threadCropper.get()!!.fill(framePixels, fw, fh, scan, index, side, batchInput, s * per) }
            }
            jobs.forEach { it.get() }
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
            cropNs += System.nanoTime() - tCrop

            val tInfer = System.nanoTime()
            interpreter.run(inputBuffer, output)
            inferNs += System.nanoTime() - tInfer

            val tSearch = System.nanoTime()
            for (k in chunk.indices) {
                val vec = output[k]
                // 与原型比余弦（两边都已 L2 归一化 → 点积即余弦），取最大。
                // 融合后这一步在模型里已经做完了，这里只是在一行 2369 个分数里取 argmax。
                var best = 0
                var bestScore = Float.NEGATIVE_INFINITY
                if (fused) {
                    for (c in 0 until nClass) {
                        val s = vec[c]
                        if (s > bestScore) {
                            bestScore = s
                            best = c
                        }
                    }
                } else {
                    for (c in 0 until nClass) {
                        val protoBase = c * dim
                        var dot = 0f
                        for (d in 0 until dim) dot += vec[d] * prototypes[protoBase + d]
                        if (dot > bestScore) {
                            bestScore = dot
                            best = c
                        }
                    }
                }
                bestClass[slot] = meta.classIds[best]
                similarity[targets[slot]] = (bestScore * 100).roundToInt()
                slot++
            }
            searchNs += System.nanoTime() - tSearch
        }
        writeLabels(labels, targets, bestClass)
        // 阈值来自这个模型自己的契约（见 ModelMeta.dumpSimilarity）：两个模型的相似度分布不同，
        // 共用一个常量会让其中一个的留档量跑偏。
        lastLowConfidence = targets.filter { similarity[it] < dumpSimilarity }.map { it to similarity[it] }
        val searchName = if (fused) "取最大" else "检索"
        lastPhases = listOf(
            "准备" to convertNs / 1_000_000,
            "裁格" to cropNs / 1_000_000,
            "推理" to inferNs / 1_000_000,
            searchName to searchNs / 1_000_000,
        )
        return labels
    }

    override fun phaseTimings(): List<Pair<String, Long>> = lastPhases

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
     * 阈值是**这个模型自己契约里的** `dumpSimilarity`（见 [ModelMeta]）：large 上约 84、
     * small 上约 85 —— 数值接近但含义不同，各自对应自己距离分布里"基本可判为未知"的那条线。
     * 运行时不改判，只用来决定"这一帧要不要留档"，见 `HintSession.dumpLowConfidence`。
     */
    override fun lowConfidenceCells(): List<Pair<Int, Int>> = lastLowConfidence

    override fun close() {
        cropPool.shutdownNow()
        interpreter.close()
        // delegate 要晚于 interpreter 释放：Interpreter 还引用着它
        gpuDelegate?.let { runCatching { it.close() } }
    }

    companion object {
        private const val TAG = "TfliteTileClassifier"

        /** 固定批大小，与 `export_tflite.py --batch` 的默认值必须一致。 */
        private const val BATCH = 32

        /** 退回 CPU 时的线程数。 */
        private const val DEFAULT_THREADS = 4

        /**
         * 是否优先挂 GPU delegate。**当前是关的**，原因是一次真机实测（见下），不是猜的。
         *
         * 2026-10-15，Redmi K30 Pro（骁龙 865 / Adreno 650）、`brick_embedder.tflite` 160x160 batch32：
         * ```
         * Replacing 39 out of 162 node(s) with delegate (TfLiteGpuDelegateV2), yielding 3 partitions
         * Replacing 110 out of 124 node(s) with delegate (TfLiteXNNPackDelegate), yielding 25 partitions
         * ```
         * 也就是说 GPU 只吃下 **24% 的算子**，整图被切成 3 个 GPU 分区 + 25 个 CPU 分区，
         * 每次 `run` 都要在 CPU↔GPU 之间来回搬中间张量。实测推理 8121ms/帧
         * （占整帧 11.3s 的 72%）。连续三帧都是 9.4s 量级，**不是首次执行的预热**。
         *
         * 关掉 GPU 走纯 CPU 更慢（推理 9486ms），因为这台机器上我们的进程只拿到
         * **约 1 个核**的算力：整帧 12s 挂钟里进程只消耗 12.9s CPU（`Cpus_allowed_list: 0-3`
         * 允许 4 个大核，`setNumThreads(4)` 也设了，但图被切成 29 段后并行度全丢）。
         * 对照：同一个 tflite 在 Mac 上 4 线程只花 47.9ms 挂钟 / 161.7ms CPU = **3.38 核**。
         *
         * 结论：**必须靠 GPU**（CPU 这条路算力不够，物理下限就有 0.5~0.7s/帧）。
         * 但这个模型在 2.16.1 的 GPU delegate 上只覆盖 24%，两者都慢的根因是同一个
         * ——"图被切碎"。下一步是换 GPU 友好的图（升级 LiteRT / 换骨干），不是调这里。
         */
        private const val PREFER_GPU = true

        /** 本机能不能用 GPU delegate。 */
        fun gpuSupported(): Boolean = try {
            CompatibilityList().isDelegateSupportedOnThisDevice
        } catch (t: Throwable) {
            Log.w(TAG, "查询 GPU delegate 支持失败", t)
            false
        }

        /** 界面显示用：实际会跑在哪条后端上。 */
        fun backendDescription(): String =
            if (PREFER_GPU && gpuSupported()) "GPU delegate" else "CPU（XNNPACK $DEFAULT_THREADS 线程）"

        /**
         * 加载一个变体。每次调用都新建一份解释器与缓冲 —— 切换模型时**先把旧的关掉**，
         * 两个模型同时在内存里要多占一份图 + 2.4MB 原型，没必要。
         *
         * @param variant 用哪套资产（见 [ModelVariant] 与 `BrickGame-Tool/variants.py` 的约定）。
         */
        fun load(context: Context, variant: ModelVariant = ModelVariant.DEFAULT): TfliteTileClassifier {
            val meta = ModelAssets.readMeta(context, variant)
            // 融合了检索的模型不需要单独的原型库（省 2.4MB 的读盘与解析）
            val prototypes = if (meta.fusedSearch) FloatArray(0)
            else ModelAssets.readPrototypes(context, variant, meta.nClass, meta.embedDim)

            val bytes = context.assets.open(variant.graphAsset).use { it.readBytes() }

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
                require(outShape.size == 2 && outShape[1] == meta.outDim) {
                    "TFLite 输出 shape ${outShape.toList()} 与契约里的 out_dim=${meta.outDim} 不符"
                }
            } catch (t: Throwable) {
                runCatching { interpreter.close() }
                gpu?.let { runCatching { it.close() } }
                throw t
            }

            Log.i(
                TAG,
                "加载完成（${variant.label}，${variant.graphAsset}）：" +
                    "输入 ${interpreter.getInputTensor(0).shape().toList()} " +
                    "输出 ${interpreter.getOutputTensor(0).shape().toList()} " +
                    (if (gpu != null) "GPU delegate" else "CPU $DEFAULT_THREADS 线程") +
                    "，留档阈值 ${meta.dumpSimilarity}",
            )
            return TfliteTileClassifier(interpreter, prototypes, meta, DEFAULT_THREADS, gpu, variant)
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
            if (!PREFER_GPU) {
                // 见 PREFER_GPU 的说明：这个模型挂 GPU 反而是"CPU 跑大头 + 来回搬张量"
                Log.i(TAG, "按 PREFER_GPU=false 走纯 CPU（XNNPACK $DEFAULT_THREADS 线程）")
                null
            } else if (!compat.isDelegateSupportedOnThisDevice) {
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
