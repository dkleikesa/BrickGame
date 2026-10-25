package com.qcmian.brickgame.capture

import android.os.Process

/**
 * 跑单次诊断用的日志快照。
 *
 * 为什么要抓 logcat 而不是自己维护环形缓冲：定位失败的结论全在
 * `YoloBoardDetector` / `HintSession` 的 Log 里（没有 board 框、格距太小、残差超限、
 * 第几帧、耗时），自己再记一份等于把日志写两遍，还容易漏。
 *
 * Android 10+ 普通应用执行 logcat 只能看到本进程的日志，加上 `--pid` 更明确。
 */
object Diagnostics {

    /** 最多抓这么多行：一次分享几百 KB 足够定位，再大也只是重复。 */
    private const val MAX_LINES = 4000

    /** 抓本进程最近的 logcat。失败时返回一行说明，不抛异常。 */
    fun logcat(): String = runCatching {
        val pid = Process.myPid()
        val p = Runtime.getRuntime().exec(
            arrayOf("logcat", "-d", "-v", "threadtime", "-t", MAX_LINES.toString(), "--pid=$pid"),
        )
        val text = p.inputStream.bufferedReader().use { it.readText() }
        val err = p.errorStream.bufferedReader().use { it.readText() }
        p.waitFor()
        when {
            text.isNotBlank() -> text
            err.isNotBlank() -> "logcat 无输出：$err"
            else -> "（logcat 为空）"
        }
    }.getOrElse { "logcat 读取失败：${it.javaClass.simpleName} ${it.message}" }
}