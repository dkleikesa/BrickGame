package com.qcmian.brickgame.capture

import android.content.Context

/**
 * 无解时是否继续自动玩（补数据模式）。
 *
 * 为什么需要：自动通关遇到"已穷举证明无解"会停下，但那恰恰是最该继续的场景 ——
 * 无解往往意味着**识别出的盘面和真实盘面已经不一致**（定位偏了、空格判错、类别错），
 * 这时候继续点、继续留档，才能拿到"模型看到的世界 vs 真实画面"的成对样本。
 *
 * 开启后：无解不再停止，改为贪心消一对；一步都走不了就隔几秒重试。
 * **留档规则不变**：仍然只在低置信度 / 定位失败时存，不因为开了这个开关就每帧存 ——
 * 每帧存一张 1.6MB 的 PNG 既占存储，也会把真正的疑难帧淹掉。
 */
object AutoContinue {

    private const val PREF = "brickgame"
    private const val KEY = "auto_continue_unsolvable"

    fun enabled(context: Context): Boolean = runCatching {
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE).getBoolean(KEY, false)
    }.getOrDefault(false)

    fun save(context: Context, on: Boolean) {
        runCatching {
            context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
                .edit().putBoolean(KEY, on).apply()
        }
    }
}
