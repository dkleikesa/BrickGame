package com.qcmian.brickgame.ml

/**
 * 内置的识别模型：assets 里每个变体各有一套「计算图 + 原型库 + 契约」。
 *
 * [tag] **必须与 `BrickGame-Tool/variants.py` 逐字一致** —— 那是导出侧的唯一约定处，
 * 文件名按 `brick_embedder$tag.tflite` / `prototypes$tag.bin` / `model_meta$tag.json` 拼。
 * large 的 tag 是空串：线上那份资产的名字早已定下（`brick_embedder.tflite` 等），
 * 加 small 时不动它、只新增一套，免得已发出的包和导出脚本对不上。
 *
 * 各变体的输入尺寸与输出维度目前相同（160x160 / 256 维），所以 [CellCropper] 与
 * 原型检索的循环都不用按变体分支；但**不要依赖这一点** —— 这些值一律从各自的
 * `model_meta.json` 读（见 [ModelMeta]），换骨干/换维度时不用改这里。
 */
enum class ModelVariant(val tag: String, val label: String) {
    /** MobileNetV3-large，3.22M 参数，线上一直在跑的那个。 */
    LARGE("", "V3 Large"),

    /**
     * MobileNetV3-small，1.07M 参数。留档阈值与 large 不同（85 vs 84，见 [ModelMeta.dumpSimilarity]）：
     * 它的度量空间更挤，没见过的东西更容易被判"没把握"。
     */
    SMALL("_small", "V3 Small"),

    /**
     * MobileNetV2，2.55M 参数。**进这个变体的唯一理由是它在手机 GPU 上能整图跑**：
     * V3 系列带 SE 模块与 hard-swish，转出来 171 个算子（10 TRANSPOSE / 9 MEAN / 5 PAD），
     * 实测 GPU delegate 只肯接 24%，剩下的在 CPU 上被切碎成 29 个分区 —— 一帧推理 9.4 秒。
     * V2 只有 conv/dwconv/relu6/add，TFLite 官方的 GPU 示例用的就是它。
     */
    V2("_v2", "V2"),
    ;

    val graphAsset: String get() = "brick_embedder$tag.tflite"
    val prototypesAsset: String get() = "prototypes$tag.bin"
    val metaAsset: String get() = "model_meta$tag.json"

    companion object {
        /**
         * 默认用哪个。
         *
         * 2026-10-15 起是 **V2**：三个变体里它精度最好（留出帧 top1 0.987 / top5 0.996 /
         * 帧内一致性 0.95，距离间隔 0.149 vs large 的 0.097），而且只有它能被 GPU delegate
         * **整图吃下**（78/78 算子、1 个分区）—— 实测端到端 0.93s，large 是 11.3s。
         */
        val DEFAULT = V2

        /** 存档里读出来的字符串 → 变体；认不出（旧版本写进去过、或名字改过）就回默认。 */
        fun parse(id: String?): ModelVariant = entries.firstOrNull { it.name == id } ?: DEFAULT
    }
}
