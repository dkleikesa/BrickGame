package com.qcmian.brickgame.capture

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.qcmian.brickgame.HintUiBus
import com.qcmian.brickgame.vision.HintPipeline
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * 低置信度帧留档：模型对某些格子没把握时，把这一帧**原图 + 识别结果**写进
 * `debug-frames/low-conf/`，事后拿去补标注重训。
 *
 * 为什么连原图一起存：模型优化要的是"当时屏幕上的样子"，只有矩阵文件没法补标注。
 * 一对文件同名（`lowconf_MMdd-HHmmss.png/.txt`），txt 里写明可疑格子、相似度和整盘矩阵。
 *
 * 也管"一共存了多少张"（主页状态行）和"打包分享"（主页按钮，走系统分享）。
 */
object LowConfStore {

    private const val TAG = "LowConfStore"

    /** 留档根目录，`Diagnostics` 的 logs 子目录也挂在它下面。 */
    internal const val DUMP_DIR = "debug-frames"

    private const val LOW_CONF_DIR = "low-conf"

    /** "没定位到棋盘"的帧留在这里：`debug-frames/scan-fail/`。 */
    private const val SCAN_FAIL_DIR = "scan-fail"

    /** 最多留这么多帧，别把手机存储攒满（手动刷新一次最多一张）。 */
    private const val KEEP = 200

    /** 分享 ZIP 里最多带几份落盘日志：一份几百 KB，带太多只是重复。 */
    private const val MAX_SHARE_LOGS = 5

    /** 分享 ZIP 里的说明，免得拿到包的人不知道各目录是什么。 */
    private val ZIP_README = """
        BrickGame 留档与诊断日志
        - low-conf/：低置信度帧，png + 同名 txt（矩阵 / 可疑格 / 耗时）
        - scan-fail/：定位失败帧，只有 png
        - logs/logcat_*.txt：自动结束或手动定位失败时落盘的 logcat，文件头写了原因 / 版本 / 机型
        - logs/logcat.txt：点分享时现抓的 logcat，包含落盘之后的最后几帧
    """.trimIndent()

    private val LOG_ONLY_README = """
        BrickGame 诊断日志
        - logcat_*.txt：自动结束或手动定位失败时落盘的 logcat，文件头写了原因 / 版本 / 机型
        - logcat.txt：点分享时现抓的 logcat
    """.trimIndent()

    /** 分享用的 ZIP 落在这个 cache 子目录（FileProvider 只暴露它）。 */
    private const val SHARE_DIR = "lowconf-share"

    /** 存一帧。[bitmap] 是这一帧的原图，调用方随后就会 recycle，所以在这里先写完再返回。 */
    fun save(context: Context, bitmap: Bitmap, result: HintPipeline.Result, scanSuspect: String? = null) {
        // 两个入档条件：分类器没把握，或定位"能用但可疑"（网格残差大/board 分数低/砖太少）。
        // 后者往往才是"看起来正常、实际错了"的帧，补数据时最值钱。
        if (result.lowConfidence.isEmpty() && scanSuspect == null) return
        val dir = ensureDir(context) ?: return
        val stem = "lowconf_" + SimpleDateFormat("MMdd-HHmmss-SSS", Locale.US).format(Date())
        val scan = result.scan
        val t0 = SystemClock.elapsedRealtime()
        try {
            File(dir, "$stem.png").outputStream().use {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            File(dir, "$stem.txt").writeText(
                buildString {
                    // 记下是哪个模型判的没把握：两个模型的留档阈值不同（见 ModelMeta.dumpSimilarity），
                    // 混在一起补标注时分不清是"模型不行"还是"阈值太灵敏"
                    appendLine("# ${scan.rows}x${scan.cols} pitch=${scan.pitchX}x${scan.pitchY} " +
                        "空位=${scan.empty.count { it }}/${scan.cellCount} 模型=${result.modelLabel} " +
                        result.timingText())
                    if (scanSuspect != null) appendLine("# 定位置信度可疑：$scanSuspect")
                    appendLine("# 没把握的格子（序号=相似度）：" + result.lowConfidence.joinToString(" ") {
                        "(${it.first / scan.cols},${it.first % scan.cols})=${it.second}"
                    })
                    appendLine("# 可消对 " + result.hints.joinToString(" ") { it.pairText() })
                    for (r in 0 until scan.rows) {
                        val from = r * scan.cols
                        appendLine((0 until scan.cols).joinToString(" ") { result.state.labels[from + it].toString() })
                    }
                },
            )
            trim(dir)
            publish(context)
            // 写盘耗时打进日志：这一段不在帧耗时里（见 HintSession 的 dumpExecutor），
            // 但它块头不小（整屏 PNG 无损压缩实测 ~1.6s），排查"点了要等几秒"时必须能看到
            Log.i(
                TAG,
                "低置信度帧已留档：$stem.png（${result.lowConfidence.size} 个可疑格：" +
                    "${result.lowConfidence}），写盘 ${SystemClock.elapsedRealtime() - t0}ms",
            )
        } catch (e: Throwable) {
            Log.w(TAG, "低置信度帧留档失败", e)
        }
    }

    /**
     * 存一帧"没定位到棋盘"的原图（`debug-frames/scan-fail/`）。
     *
     * 为什么要单独留这一份：识别失败以前在磁盘上是**完全看不见**的 —— 只有界面上一句
     * "未识别到棋盘"，调标定只能靠猜。成功帧有低置信度留档，失败帧一张不落，于是每次
     * 都是"用户说 50% 失败、手上没有一张失败图"。文件名前缀同 lowconf_ + 时间戳，
     * 与同刻的 logcat（YoloBoardDetector 那条 定位/残差 诊断）按时间对上即可定位原因。
     */
    fun saveScanFail(context: Context, bitmap: Bitmap) {
        val dir = ensureDir(context, SCAN_FAIL_DIR) ?: return
        val stem = "scanfail_" + SimpleDateFormat("MMdd-HHmmss", Locale.US).format(Date())
        try {
            File(dir, "$stem.png").outputStream().use {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
            trim(dir)
            publish(context)
            Log.i(TAG, "定位失败帧已留档：$stem.png（${dir.absolutePath}）")
        } catch (e: Throwable) {
            Log.w(TAG, "定位失败帧留档失败", e)
        }
    }

    /** 留档目录里现存的截图（按文件名 = 时间序，主页的"相册"直接用）。 */
    fun shots(context: Context): List<File> = pngs(readDir(context))

    /** 列一遍目录并推给界面（界面启动时调，之后存/删都会自动刷新）。 */
    fun refresh(context: Context) {
        publish(context)
    }

    /**
     * 清空留档与日志（分享完顺手删掉，别一直占存储）：
     * `low-conf/`、`scan-fail/`、`logs/`、`debug-frames` 根目录的历史产物，连同上次打包的 ZIP。
     * @return 删掉的截图帧数（低置信 + 定位失败；界面只用来打日志）。
     */
    fun deleteAll(context: Context): Int {
        val low = pngs(readDir(context)).size
        val fail = readFailDir(context)?.let { pngs(it).size } ?: 0
        val logs = Diagnostics.files(context).size
        readDir(context)?.listFiles()?.forEach { runCatching { it.delete() } }
        readFailDir(context)?.listFiles()?.forEach { runCatching { it.delete() } }
        Diagnostics.files(context).forEach { runCatching { it.delete() } }
        // debug-frames 根目录下还有历史调试产物（latest.png/txt、matrix/），一并清掉
        context.getExternalFilesDir(DUMP_DIR)?.listFiles()?.forEach { f ->
            if (f.isFile) {
                runCatching { f.delete() }
            } else if (f.name == "matrix") {
                f.listFiles()?.forEach { runCatching { it.delete() } }
            }
        }
        runCatching { File(context.cacheDir, SHARE_DIR).listFiles()?.forEach { it.delete() } }
        publish(context)
        Log.i(TAG, "已删除留档：截图 ${low + fail} 张（低置信 $low / 定位失败 $fail）+ 日志 $logs 份")
        return low + fail
    }

    /**
     * 把整盘存档（png + txt 成对）打成一个 ZIP 供系统分享。
     *
     * 写在 cache 里、每次覆盖（只留最新那个），分享完不用清 —— 系统回收 cache 就行。
     * @return ZIP 文件；没有可分享的帧或建不出目录时返回 null。
     */
    fun zipAll(context: Context): File? {
        // 两个目录都要打：只打 low-conf 的话，"定位失败"那批帧永远拿不到 ——
        // 而那正是补数据最缺的一类（定位错/没定位到）。
        val groups = listOf(LOW_CONF_DIR to readDir(context), SCAN_FAIL_DIR to readFailDir(context))
            .mapNotNull { (name, dir) -> dir?.takeIf { pngs(it).isNotEmpty() }?.let { name to it } }
        // 落盘的诊断日志也一起打（只带最近几份）；没有截图、只有日志时也要能分享出去
        val logFiles = Diagnostics.files(context).takeLast(MAX_SHARE_LOGS)
        val outDir = File(context.cacheDir, SHARE_DIR)
        if (!outDir.exists() && !outDir.mkdirs()) return null
        outDir.listFiles()?.forEach { runCatching { it.delete() } }
        val zip = File(outDir, "brickgame-dumps.zip")
        var total = 0
        try {
            ZipOutputStream(zip.outputStream().buffered()).use { out ->
                for ((prefix, dir) in groups) {
                    for (png in pngs(dir)) {
                        // 带子目录前缀：解压后 low-conf / scan-fail 各自成目录，入库脚本按目录分流
                        addEntry(out, png, "$prefix/${png.name}")
                        val txt = File(dir, png.name.removeSuffix(".png") + ".txt")
                        if (txt.isFile) addEntry(out, txt, "$prefix/${txt.name}")
                        total++
                    }
                }
                // 先附上落盘的日志（自动结束/定位失败时写的），再补一份"分享时现抓"的：
                // 落盘之后又跑的那几帧也能带上。没 adb 权限的手机就靠这个 ZIP 拿日志。
                for (log in logFiles) addEntry(out, log, "logs/${log.name}")
                addText(out, "logs/logcat.txt", Diagnostics.logcat())
                addText(out, "README.txt", ZIP_README)
            }
        } catch (e: Throwable) {
            Log.w(TAG, "打包失败", e)
            zip.delete()
            return null
        }
        Log.i(TAG, "已打包 $total 帧 + ${logFiles.size} 份日志（${zip.length() / 1024}KB）：$zip")
        return zip
    }

    /**
     * 只打包日志（落盘 + 分享时现抓的 logcat），给"没 adb、只想发日志排查"的场景用。
     *
     * 和 [zipAll] 分开：留档里可能有一百多张 scan-fail 原图，全打包会到上百 MB，
     * 只想排查一次报错时没必要发整包。
     */
    fun zipLogs(context: Context): File? {
        val outDir = File(context.cacheDir, SHARE_DIR)
        if (!outDir.exists() && !outDir.mkdirs()) return null
        val zip = File(outDir, "brickgame-logs.zip")
        val logs = Diagnostics.files(context).takeLast(MAX_SHARE_LOGS)
        try {
            ZipOutputStream(zip.outputStream().buffered()).use { out ->
                for (log in logs) addEntry(out, log, "logs/${log.name}")
                addText(out, "logs/logcat.txt", Diagnostics.logcat())
                addText(out, "README.txt", LOG_ONLY_README)
            }
        } catch (e: Throwable) {
            Log.w(TAG, "打包日志失败", e)
            zip.delete()
            return null
        }
        Log.i(TAG, "已打包 ${logs.size} 份日志 + 实时 logcat（${zip.length() / 1024}KB）：$zip")
        return zip
    }

    /** 把当前的留档/日志列表推给界面（写盘/删除后各一次）。 */
    private fun publish(context: Context) {
        HintUiBus.lowConfShots = shots(context)
        HintUiBus.scanFailCount = readFailDir(context)?.let { pngs(it).size } ?: 0
        HintUiBus.logFiles = Diagnostics.files(context)
    }

    private fun addEntry(out: ZipOutputStream, file: File, entryName: String = file.name) {
        out.putNextEntry(ZipEntry(entryName))
        file.inputStream().use { it.copyTo(out) }
        out.closeEntry()
    }

    private fun addText(out: ZipOutputStream, entryName: String, text: String) {
        out.putNextEntry(ZipEntry(entryName))
        out.write(text.toByteArray(Charsets.UTF_8))
        out.closeEntry()
    }

    /** 只读地找留档目录（数张数/打包用，不建目录）。 */
    private fun readDir(context: Context): File? {
        val dir = File(context.getExternalFilesDir(DUMP_DIR) ?: return null, LOW_CONF_DIR)
        return if (dir.isDirectory) dir else null
    }

    /** 只读地找"定位失败"目录，与 [readDir] 同构。 */
    private fun readFailDir(context: Context): File? {
        val dir = File(context.getExternalFilesDir(DUMP_DIR) ?: return null, SCAN_FAIL_DIR)
        return if (dir.isDirectory) dir else null
    }

    private fun ensureDir(context: Context, sub: String = LOW_CONF_DIR): File? {
        val dir = File(context.getExternalFilesDir(DUMP_DIR) ?: return null, sub)
        if (!dir.exists() && !dir.mkdirs()) return null
        return dir
    }

    /** 按文件名（含时间戳，字典序即时间序）留最近 [KEEP] 帧，png/txt 成对删。 */
    private fun trim(dir: File) {
        for (old in pngs(dir).dropLast(KEEP)) {
            runCatching { old.delete() }
            runCatching { File(dir, old.name.removeSuffix(".png") + ".txt").delete() }
        }
    }

    private fun pngs(dir: File?): List<File> =
        dir?.listFiles { f -> f.name.endsWith(".png") }?.sortedBy { it.name } ?: emptyList()
}
