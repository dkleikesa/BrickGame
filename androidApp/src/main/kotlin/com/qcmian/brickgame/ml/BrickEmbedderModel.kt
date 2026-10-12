package com.qcmian.brickgame.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.io.Closeable
import java.nio.FloatBuffer

/**
 * 物品身份 embedding 模型（`BrickGame-Tool/out/clf_real/best.pt` 导出，见 `export_mobile.py`）。
 *
 * 本类**只负责把模型从 assets 加载起来**并暴露接口契约；图片 → 张量、向量 → 最近原型的接线
 * 先不做（应用还在开发中）。
 *
 * assets 三个文件（`androidApp/src/main/assets/`）：
 *  - `brick_embedder.onnx` 输入 RGB NCHW float32，值域 **[0,1]**（ImageNet 归一化已烘进计算图，
 *    调用方不要再做 mean/std），输出 L2 归一化的 [embedDim] 维向量；batch 维动态。
 *  - `prototypes.bin` (2369, embedDim) float32 行优先：第 i 行对应 `meta.classIds[i]`。
 *    与输出向量做点积即为余弦相似度（两边都已归一化）。
 *  - `model_meta.json` 契约 + 类目表 + 阈值（`thresholds`）。
 *
 * 用法：
 * ```
 * val model = BrickEmbedderModel.load(context)          // 加载一次，复用
 * val v = model.embed(rgbFloatArray)                    // 3*160*160 → 256
 * model.close()
 * ```
 */
class BrickEmbedderModel private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val meta: Meta,
) : Closeable {

    /** `model_meta.json` 里 Android 端需要的部分。 */
    class Meta(
        /** 模型输入边长（正方形，当前 160）。 */
        val inputSize: Int,
        /** 身份向量维度（当前 256）。 */
        val embedDim: Int,
        /** 2369 个 class_id，顺序与 prototypes.bin 的行一一对应。 */
        val classIds: IntArray,
        /** 与 classIds 同序的图鉴 entry 名（如 p006_b1_r0_c0）。 */
        val entries: Array<String>,
        /** 命中样本距离的 p95：大于它该判为低置信。 */
        val hitDistP95: Float,
        /** 最近异类距离的 p5：大于它基本可判为未知物品。 */
        val wrongDistP5: Float,
    ) {
        val nClass: Int get() = classIds.size
    }

    /**
     * 单张图推理。
     *
     * @param rgb **RGB** 顺序、值域 [0,1]、长度 `3 * meta.inputSize * meta.inputSize`（NCHW 摊平）。
     * @return L2 归一化的身份向量，长度 `meta.embedDim`。
     */
    fun embed(rgb: FloatArray): FloatArray {
        val side = meta.inputSize
        require(rgb.size == 3 * side * side) {
            "输入长度应为 ${3 * side * side}（3 x $side x $side），实际 ${rgb.size}"
        }
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(rgb), longArrayOf(1, 3, side.toLong(), side.toLong()))
        try {
            val result = session.run(mapOf(INPUT_NAME to tensor))
            try {
                @Suppress("UNCHECKED_CAST")
                return (result.get(0).value as Array<FloatArray>)[0]
            } finally {
                result.close()
            }
        } finally {
            tensor.close()
        }
    }

    /**
     * 批量推理。
     *
     * 一次跑满比逐格跑快一个量级 —— 一帧有 100+ 格，逐格跑 MobileNetV3 要 1~2 秒，整帧就废了。
     * 模型输入 batch 维是动态的，直接堆即可。
     *
     * @param rgb **RGB** 顺序、值域 [0,1]、长度 `batch * 3 * inputSize * inputSize`（NCHW 摊平）。
     * @return 长度 `batch * embedDim`，第 i 段是第 i 张图的 L2 归一化身份向量。
     */
    fun embedBatch(rgb: FloatArray, batch: Int): FloatArray {
        require(batch > 0) { "batch 必须为正，实际 $batch" }
        val side = meta.inputSize
        require(rgb.size == batch * 3 * side * side) {
            "输入长度应为 ${batch * 3 * side * side}（$batch x 3 x $side x $side），实际 ${rgb.size}"
        }
        val tensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(rgb),
            longArrayOf(batch.toLong(), 3, side.toLong(), side.toLong()),
        )
        try {
            val result = session.run(mapOf(INPUT_NAME to tensor))
            try {
                @Suppress("UNCHECKED_CAST")
                val rows = result.get(0).value as Array<FloatArray>
                val out = FloatArray(batch * meta.embedDim)
                for (i in 0 until batch) rows[i].copyInto(out, i * meta.embedDim)
                return out
            } finally {
                result.close()
            }
        } finally {
            tensor.close()
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val MODEL_ASSET = "brick_embedder.onnx"
        const val PROTOTYPES_ASSET = "prototypes.bin"
        const val META_ASSET = "model_meta.json"
        const val INPUT_NAME = "input"

        /** 从 assets 加载。加载失败直接抛异常（开发阶段要能立刻看见，不要静默降级）。 */
        fun load(context: Context, env: OrtEnvironment = OrtEnvironment.getEnvironment()): BrickEmbedderModel {
            val metaText = context.assets.open(META_ASSET).bufferedReader().use { it.readText() }
            val meta = parseMeta(JSONObject(metaText))
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val modelBytes = context.assets.open(MODEL_ASSET).readBytes()
            return BrickEmbedderModel(env, env.createSession(modelBytes, opts), meta)
        }

        private fun parseMeta(json: JSONObject): Meta {
            val input = json.getJSONObject("input").getJSONArray("size")
            val output = json.getJSONObject("output").getJSONArray("size")
            val ids = json.getJSONArray("class_ids")
            val entries = json.getJSONArray("entries")
            val th = json.optJSONObject("thresholds") ?: JSONObject()
            return Meta(
                inputSize = input.getInt(2),                       // [batch, 3, H, W]
                embedDim = output.getInt(1),                       // [batch, dim]
                classIds = IntArray(ids.length()) { ids.getInt(it) },
                entries = Array(entries.length()) { entries.getString(it) },
                hitDistP95 = th.optDouble("dist_hit_p95", 0.0).toFloat(),
                wrongDistP5 = th.optDouble("dist_wrong_p5", 0.0).toFloat(),
            )
        }
    }
}
