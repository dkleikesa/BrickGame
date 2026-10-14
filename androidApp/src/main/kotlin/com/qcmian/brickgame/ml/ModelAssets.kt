package com.qcmian.brickgame.ml

import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * assets 里的模型包读取：原型库 + 契约。
 *
 * 计算图（`brick_embedder*.tflite`）由 [TfliteTileClassifier] 自己加载，
 * 这里只管"分类头"要用的那两份 —— 换运行时、换机型、换变体都不该动它们。
 *
 * 三个文件名都从 [ModelVariant] 拼（`brick_embedder$tag.tflite` / `prototypes$tag.bin` /
 * `model_meta$tag.json`），所以**换模型只需要换一个 variant 参数**，加载路径不出分支。
 */
internal object ModelAssets {

    /** (nClass, dim) 原型库，行优先小端；第 i 行对应 `meta.classIds[i]`。 */
    fun readPrototypes(context: Context, variant: ModelVariant, nClass: Int, dim: Int): FloatArray {
        val expected = nClass * dim * 4
        val raw = context.assets.open(variant.prototypesAsset).use { it.readBytes() }
        require(raw.size == expected) {
            "${variant.prototypesAsset} 大小 ${raw.size} 与 (nClass=$nClass x dim=$dim x 4) = $expected 不符"
        }
        val out = FloatArray(nClass * dim)
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    fun readMeta(context: Context, variant: ModelVariant): ModelMeta {
        val text = context.assets.open(variant.metaAsset).bufferedReader().use { it.readText() }
        return ModelMeta.parse(JSONObject(text))
    }
}
