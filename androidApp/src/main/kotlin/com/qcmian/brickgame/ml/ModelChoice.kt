package com.qcmian.brickgame.ml

import android.content.Context

/**
 * "用哪个识别模型"的落盘存档。
 *
 * 为什么不只用 Compose 状态：无障碍服务与界面是**同进程的两个入口**，服务在界面没打开时
 * 也要知道该加载哪个模型。所以真正的来源是 SharedPreferences，界面上的选择器只是它的投影
 * （运行期的即时状态放在 `HintUiBus.variant`，供界面重组）。
 *
 * 读失败（存档不存在、系统不给读）一律回 [ModelVariant.DEFAULT]：加载模型失败会让整帧识别
 * 不可用，宁可退回线上一直在跑的那个。
 */
object ModelChoice {

    private const val PREF = "brickgame"
    private const val KEY_VARIANT = "model_variant"

    fun current(context: Context): ModelVariant = runCatching {
        val pref = context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
        ModelVariant.parse(pref.getString(KEY_VARIANT, null))
    }.getOrDefault(ModelVariant.DEFAULT)

    fun save(context: Context, variant: ModelVariant) {
        runCatching {
            context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
                .edit().putString(KEY_VARIANT, variant.name).apply()
        }
    }
}
