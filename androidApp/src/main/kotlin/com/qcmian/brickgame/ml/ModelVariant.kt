package com.qcmian.brickgame.ml

/**
 * 内置识别模型 —— **现在只有 V2 一个变体**（2026-10 定稿）。
 *
 * large / small 已经删除：精度与端上性能都不如 V2（V2 是唯一能被 GPU delegate
 * 整图吃下的，large 一帧要 9~11s）。训练资产、导出脚本入口、设置页切换都一并移除。
 *
 * [tag] **必须与 `BrickGame-Tool/variants.py` 逐字一致** —— 那是导出侧的唯一约定处，
 * 文件名按 `brick_embedder$tag.tflite` / `prototypes$tag.bin` / `model_meta$tag.json` 拼。
 * 输入尺寸与输出维度一律从各自的 `model_meta.json` 读（见 [ModelMeta]）。
 */
enum class ModelVariant(val tag: String, val label: String) {
    /** MobileNetV2，2.55M 参数。 */
    V2("_v2", "V2"),
    ;

    val graphAsset: String get() = "brick_embedder$tag.tflite"
    val prototypesAsset: String get() = "prototypes$tag.bin"
    val metaAsset: String get() = "model_meta$tag.json"

    companion object {
        val DEFAULT = V2

        /** 存档里读出来的字符串 → 变体；认不出（旧版本写过 LARGE/SMALL）就回默认。 */
        fun parse(id: String?): ModelVariant = entries.firstOrNull { it.name == id } ?: DEFAULT
    }
}
