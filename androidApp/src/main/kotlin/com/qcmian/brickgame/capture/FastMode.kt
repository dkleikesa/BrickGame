package com.qcmian.brickgame.capture

import android.content.Context

/**
 * 自动通关的**快速模式**开关（落盘，跨会话保持）。
 *
 * 开：连续出手（一步接一步，中间不截屏识别），连续 10 步才识别一帧校准，
 * 手势的滑动/点击时长也压短。
 *
 * 为什么不是默认开：快速模式的前提是"游戏照单全收每一步"，而手势被吞、动画期间不响应这类
 * 情况真机上确实存在。普通模式每步都复检，错了立刻能停；快速模式只靠每 10 步的校准兜底，
 * 中间几步没落地会被后面几步的坐标连带着偏掉。所以由用户按盘面/机型自己选。
 */
object FastMode {

    private const val PREF = "brickgame"
    private const val KEY = "auto_fast_mode"

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
