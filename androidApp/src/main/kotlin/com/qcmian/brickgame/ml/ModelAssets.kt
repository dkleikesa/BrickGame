package com.qcmian.brickgame.ml

import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * assets 里的模型包读取：原型库 + 契约。
 *
 * 计算图（`brick_embedder.tflite`）由 [TfliteTileClassifier] 自己加载，
 * 这里只管"分类头"要用的那两份 —— 换运行时、换机型都不该动它们。
 */
internal object ModelAssets {

    private const val PROTOTYPES_ASSET = "prototypes.bin"
    private const val META_ASSET = "model_meta.json"

    /** (nClass, dim) 原型库，行优先小端；第 i 行对应 `meta.classIds[i]`。 */
    fun readPrototypes(context: Context, nClass: Int, dim: Int): FloatArray {
        val expected = nClass * dim * 4
        val raw = context.assets.open(PROTOTYPES_ASSET).use { it.readBytes() }
        require(raw.size == expected) {
            "prototypes.bin 大小 ${raw.size} 与 (nClass=$nClass x dim=$dim x 4) = $expected 不符"
        }
        val out = FloatArray(nClass * dim)
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    fun readMeta(context: Context): ModelMeta {
        val text = context.assets.open(META_ASSET).bufferedReader().use { it.readText() }
        return ModelMeta.parse(JSONObject(text))
    }
}
