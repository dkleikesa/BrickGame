package com.qcmian.brickgame.ml

import android.content.Context

/** 定位算法：OpenCV（几何）与 YOLO（检测 + 拟合）。实机对比用。 */
enum class Locator(val label: String) {
    OpenCV("OpenCV"),
    Yolo("YOLO"),
    ;

    companion object {
        /** YOLO 在残局/换肤下明显更稳，默认走它；要对比再在 App 里切回 OpenCV。 */
        val DEFAULT = Yolo
        fun parse(raw: String?): Locator = entries.firstOrNull { it.name == raw } ?: DEFAULT
    }
}

/**
 * "用哪个定位算法"的落盘存档。与 [ModelChoice] 同构：真正的来源是 SharedPreferences，
 * 因为无障碍服务在没有界面时也要知道该走哪条链路。
 */
object LocatorChoice {

    private const val PREF = "brickgame"
    private const val KEY = "board_locator"

    fun current(context: Context): Locator = runCatching {
        val pref = context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
        Locator.parse(pref.getString(KEY, null))
    }.getOrDefault(Locator.DEFAULT)

    fun save(context: Context, locator: Locator) {
        runCatching {
            context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
                .edit().putString(KEY, locator.name).apply()
        }
    }
}
