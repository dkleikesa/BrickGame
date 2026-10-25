package com.qcmian.brickgame.capture

import android.content.Context
import android.os.Process
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 诊断日志：抓本进程最近的 logcat，并在关键失败点另存一份文件。
 *
 * 为什么要落盘：有些手机没有 adb 调试权限，日志只能靠 App 的分享带出来。
 * 分享 ZIP 里那份是"点分享时现抓"的；这里在自动结束 / 定位失败时先把日志写进
 * `debug-frames/logs/`，即使之后 App 被重启，日志还在，分享 ZIP 会一起打进去。
 *
 * Android 10+ 普通应用执行 logcat 只能看到本进程的日志，加上 `--pid` 更明确。
 */
object Diagnostics {

    private const val TAG = "Diagnostics"

    /** 最多抓这么多行：一次分享几百 KB 足够定位，再大也只是重复。 */
    private const val MAX_LINES = 4000

    /** 日志目录名，挂在 `debug-frames/` 下。 */
    private const val DIR = "logs"

    /** 最多留这么多份，别把手机存储攒满。 */
    private const val KEEP = 10

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

    /**
     * 把当前 logcat 另存一份，[reason] 写在文件头（例如"求解超预算…"）。
     *
     * 写盘是阻塞的（几十到几百毫秒），调用方要放到后台线程。
     * @return 写出的文件；目录建不出来 / 写盘失败返回 null。
     */
    fun save(context: Context, reason: String): File? = runCatching {
        val dir = logDir(context)
        if (!dir.exists() && !dir.mkdirs()) return null
        val now = Date()
        val stamp = SimpleDateFormat("MMdd-HHmmss-SSS", Locale.US).format(now)
        val file = File(dir, "logcat_$stamp.txt")
        file.writeText(
            buildString {
                appendLine("# $reason")
                appendLine("# ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(now)}")
                appendLine("# ${header(context)}")
                appendLine()
                append(logcat())
            },
        )
        trim(dir)
        Log.i(TAG, "日志已留档：${file.name}（${file.length() / 1024}KB）")
        file
    }.getOrElse {
        Log.w(TAG, "日志留档失败", it)
        null
    }

    /** 已留档的日志（按文件名 = 时间序），分享 ZIP 用。 */
    fun files(context: Context): List<File> = filesIn(logDir(context))

    /**
     * 一行环境信息：App 版本、机型、系统、屏幕、pid。
     *
     * 落盘日志的文件头用它；会话开始时也打一条进 logcat —— 用户发来的日志里
     * 第一眼就能看到"哪台机器、哪个版本"，不用再追问。
     */
    fun header(context: Context): String = buildString {
        val info = runCatching {
            context.packageManager.getPackageInfo(context.packageName, 0)
        }.getOrNull()
        val dm = context.resources.displayMetrics
        append("app ${info?.versionName ?: "?"} · ")
        append("${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · ")
        append("Android ${android.os.Build.VERSION.RELEASE}(SDK ${android.os.Build.VERSION.SDK_INT}) · ")
        append("screen ${dm.widthPixels}x${dm.heightPixels}@${"%.2f".format(Locale.US, dm.density)} · ")
        append("pid ${Process.myPid()}")
    }

    private fun logDir(context: Context): File =
        File(context.getExternalFilesDir(LowConfStore.DUMP_DIR) ?: context.filesDir, DIR)

    private fun filesIn(dir: File): List<File> =
        dir.listFiles()
            ?.filter { it.isFile && it.name.endsWith(".txt") }
            ?.sortedBy { it.name }
            ?: emptyList()

    private fun trim(dir: File) {
        filesIn(dir).dropLast(KEEP).forEach { runCatching { it.delete() } }
    }
}
