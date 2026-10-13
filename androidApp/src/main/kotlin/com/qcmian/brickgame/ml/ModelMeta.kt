package com.qcmian.brickgame.ml

import org.json.JSONObject

/**
 * `model_meta.json` 里 Android 端**真正要用**的部分。
 *
 * 契约文件里还有 `entries`（2369 个图鉴名）、`thresholds`（命中/未知距离分位）等字段，
 * 那是给 Python 侧和人工排查看的，端上不读 —— 光 `entries` 就是两千多个字符串对象，
 * 没必要每台机器上都解析一遍。
 */
class ModelMeta(
    /** 模型输入边长（正方形，当前 160）。 */
    val inputSize: Int,
    /** 身份向量维度（当前 256）。 */
    val embedDim: Int,
    /** 2369 个 class_id，顺序与 `prototypes.bin` 的行一一对应。 */
    val classIds: IntArray,
) {
    val nClass: Int get() = classIds.size

    companion object {
        fun parse(json: JSONObject): ModelMeta {
            val input = json.getJSONObject("input").getJSONArray("size")
            val output = json.getJSONObject("output").getJSONArray("size")
            val ids = json.getJSONArray("class_ids")
            return ModelMeta(
                inputSize = input.getInt(2),                       // [batch, 3, H, W]
                embedDim = output.getInt(1),                       // [batch, dim]
                classIds = IntArray(ids.length()) { ids.getInt(it) },
            )
        }
    }
}
