package com.qcmian.brickgame.ml

import org.json.JSONObject
import kotlin.math.roundToInt

/**
 * `model_meta*.json` 里 Android 端**真正要用**的部分。
 *
 * 契约文件里还有 `entries`（2369 个图鉴名）、`train` 等字段，那是给 Python 侧和人工排查看的，
 * 端上不读 —— 光 `entries` 就是两千多个字符串对象，没必要每台机器上都解析一遍。
 *
 * 各模型**必须各读自己那一份**：下面的 [dumpSimilarity] 就是按各自的距离分布算出来的，
 * 照搬另一个模型的值会让留档量跑偏。
 */
class ModelMeta(
    /** 模型输入边长（正方形，当前 160）。 */
    val inputSize: Int,
    /**
     * 模型输出宽度。
     *
     * - [fusedSearch] = true：是**类数**（2369），每行就是各类的余弦相似度，取 argmax 即可；
     * - false：是身份向量维度（256），要拿它跟 `prototypes$tag.bin` 做点积。
     */
    val outDim: Int,
    /** 2369 个 class_id，顺序与 `prototypes.bin` 的行（或融合后输出的各列）一一对应。 */
    val classIds: IntArray,
    /**
     * "没把握"的相似度下限（百分制）：低于它就把整帧截图留档。
     *
     * 取自契约里的 `thresholds.dist_wrong_p5`（"基本可判为未知物品"那条线，= 异类距离的 5 分位），
     * 换算成相似度即 `100 - 100*dist`。为什么不取 `dist_hit_p95`：那条线会把 5% 的**正确命中**
     * 也算成没把握（实测留出帧上会有大量格子越线），留档会变成每帧必存、失去筛选意义。
     * 取不到字段时退回 [FALLBACK_DUMP_SIMILARITY]（老的、没有 thresholds 的契约）。
     */
    val dumpSimilarity: Int,
    /**
     * 原型检索是否已经并进模型（见 `BrickGame-Tool/export_mobile.py --fuse-search`）。
     *
     * 融合后 Android 侧不用再跑那两层循环，也不用读 `prototypes.bin`：真机上那一步是
     * 8250 万次乘加的纯 Kotlin 单线程循环，实测 1728ms —— 比模型推理本身还贵 6 倍。
     */
    val fusedSearch: Boolean,
) {
    val nClass: Int get() = classIds.size

    /** 未融合时才需要：身份向量维度（= 输出宽度），读 `prototypes.bin` 用。 */
    val embedDim: Int get() = outDim

    companion object {

        /** 契约里没有 thresholds 时的兜底（历来手写的那个值）。 */
        const val FALLBACK_DUMP_SIMILARITY = 85

        fun parse(json: JSONObject): ModelMeta {
            val input = json.getJSONObject("input").getJSONArray("size")
            val output = json.getJSONObject("output").getJSONArray("size")
            val ids = json.getJSONArray("class_ids")
            return ModelMeta(
                inputSize = input.getInt(2),                       // [batch, 3, H, W]
                outDim = output.getInt(1),                         // [batch, outDim]
                classIds = IntArray(ids.length()) { ids.getInt(it) },
                dumpSimilarity = dumpSimilarityOf(json),
                // 老契约没有这个字段：缺省 false（输出 256 维身份向量 + 单独的原型库）
                fusedSearch = json.optBoolean("search_fused", false),
            )
        }

        private fun dumpSimilarityOf(json: JSONObject): Int {
            val wrongP5 = json.optJSONObject("thresholds")?.optDouble("dist_wrong_p5", Double.NaN)
            if (wrongP5 == null || !wrongP5.isFinite() || wrongP5 <= 0.0) {
                return FALLBACK_DUMP_SIMILARITY
            }
            // 上下夹一下：解析出 0 或负数只说明契约坏了，宁可退回兜底值也不要"阈值=100 → 每帧都留档"
            return (100 - wrongP5 * 100).roundToInt().coerceIn(50, 99)
        }
    }
}
