package com.qcmian.brickgame.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import android.view.View
import com.qcmian.brickgame.HintUiBus
import com.qcmian.brickgame.HintUiState
import com.qcmian.brickgame.overlay.ControlBarView
import com.qcmian.brickgame.overlay.HintOverlayView
import com.qcmian.brickgame.overlay.OverlayController
import com.qcmian.brickgame.vision.HintPipeline
import com.qcmian.brickgame.vision.TileClassifier
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** 取一帧屏幕；无障碍服务（takeScreenshot）与调试用的"喂图片"都实现它。 */
fun interface ScreenshotProvider {
    /** @param onResult (截图, 失败原因)；成功时原因必须为 null。 */
    fun capture(onResult: (Bitmap?, String?) -> Unit)
}

/**
 * 提示会话：**完全手动**的一帧一帧。
 *
 * 按"开始提示"跑一帧，之后只有点控制条上的「刷新」才再跑一帧 —— 不再自动续帧。
 * 为什么不做自动轮询：一帧要好几秒（标定 + 模型推理，真机实测 7~8s），自动跑就是一直占着
 * CPU/GPU、还容易在画面变化时给出过期提示；手动触发反而更可控。
 *
 * 线程分工：
 *  - 主线程：发起截屏（API 要求）、渲染调试图、更新悬浮窗；
 *  - `brick-hint-worker`：OpenCV 标定 + 模型推理是纯 CPU/GPU 活，不能占主线程。
 *
 * 分析**不设超时**：一帧识别本来就要几秒（真机实测 7~8s），卡死上限反而把好帧丢掉，
 * 结果屏幕上留着上一帧的旧提示，"看起来就是提示不对"。所以只保留截屏回调的兜底
 * （[CAPTURE_TIMEOUT_MS]：`takeScreenshot` 的回调有时永远不回来，不兜底就永久卡住），
 * 分析慢就等它慢，算完为止。
 */
class HintSession(
    private val context: Context,
    private val screenshots: ScreenshotProvider,
    private val overlay: OverlayController,
    classifierProvider: () -> TileClassifier,
) {

    /**
     * 识别器**延迟到 worker 线程第一次用时**才构造：`BrickClassifiers.create` 会读 13MB 模型资产
     * 并建 ONNX 会话，放在 `onServiceConnected` 里做就是几百毫秒的主线程卡顿。
     */
    private val classifierLazy: Lazy<TileClassifier> = lazy(classifierProvider)

    private val classifier: TileClassifier get() = classifierLazy.value

    private val main = Handler(Looper.getMainLooper())
    private val workerThread = HandlerThread("brick-hint-worker").apply { start() }
    private val worker = Handler(workerThread.looper)

    /**
     * 帧代号。会话被停掉时，"还在跑"的那条旧线程算完还是会往主线程丢结果 ——
     * 靠它把过期的结果丢掉，不然会把已经作废的一帧画上去。
     */
    private val generation = AtomicInteger(0)

    /** 会话开着（悬浮窗在、按钮显示"停止提示"）。 */
    private val running = AtomicBoolean(false)

    /** 当前是否允许跑一帧（手动模式下只在"这一帧在飞"期间为 true）。 */
    private val active = AtomicBoolean(false)

    /** 一帧是否正在飞（截屏 + 分析）；防止连点"刷新"叠出多个并发帧。 */
    private val capturing = AtomicBoolean(false)

    private val captureTimeout = Runnable {
        if (capturing.compareAndSet(true, false)) {
            active.set(false)
            overlay.setHintLayerVisible(true)      // 截图没回来，别把提示层永久藏掉
            Log.w(TAG, "截屏回调超时：${CAPTURE_TIMEOUT_MS}ms 内既没成功也没失败")
            updateBar("截屏超时", "", "takeScreenshot 回调超时，点「刷新」重试")
        }
    }

    private val previewView = HintOverlayView(context)

    private var frames = 0

    /** 已写出的类别数组文件序号（每成功一帧 +1）。 */
    private var matrixFiles = 0

    init {
        overlay.onRefresh = { refreshNow() }
        overlay.onClose = { stop() }
    }

    /** 开始提示：挂悬浮窗 + 立刻跑一帧，之后等用户点「刷新」。 */
    fun start() {
        if (!running.compareAndSet(false, true)) return
        overlay.show()
        HintUiBus.overlayWindowType = overlay.activeType
        HintUiBus.state = HintUiBus.state.copy(running = true, message = null)
        refreshNow()
    }

    fun stop() {
        if (!running.compareAndSet(true, false)) return
        active.set(false)
        capturing.set(false)
        main.removeCallbacks(captureTimeout)
        generation.incrementAndGet()          // 让还在跑的那一帧的结果失效
        overlay.hide()
        HintUiBus.state = HintUiBus.state.copy(running = false, hintText = "", cleared = false, stuck = false)
    }

    /**
     * 跑一帧。控制条上的「刷新」、以及 [start] 都走这里。
     *
     * 手动模式下的三个"不跑"：会话没开、上一帧还在飞、或者刚被停掉。
     */
    fun refreshNow() {
        if (!running.get()) return
        if (!active.compareAndSet(false, true)) return
        if (capturing.get()) {
            active.set(false)
            return
        }
        updateBar("正在分析…", "")
        captureFrame()
    }

    /** 只跑一次（界面上的"验证一次"），不改运行状态，也不弹控制条。 */
    fun once() {
        if (running.get()) return
        if (!active.compareAndSet(false, true)) return
        overlay.showHint()
        HintUiBus.overlayWindowType = overlay.activeType
        captureFrame()
    }

    fun release() {
        stop()
        workerThread.quitSafely()
        if (classifierLazy.isInitialized()) {
            (classifierLazy.value as? java.io.Closeable)?.let { runCatching { it.close() } }
        }
    }

    private fun captureFrame() {
        if (!active.get()) return
        if (!capturing.compareAndSet(false, true)) return
        val gen = generation.get()
        main.removeCallbacks(captureTimeout)
        main.postDelayed(captureTimeout, CAPTURE_TIMEOUT_MS)
        // 截图前把提示层收掉：上一帧画在屏幕上的提示框会污染本次截图（强边缘带偏标定、
        // 50% 红色改变砖块颜色）。只藏提示层，控制条留着。
        overlay.setHintLayerVisible(false)
        screenshots.capture { bitmap, reason ->
            overlay.setHintLayerVisible(true)
            main.removeCallbacks(captureTimeout)
            if (bitmap == null) {
                val message = reason ?: "截屏失败"
                Log.w(TAG, message)
                report(message)
                updateBar("截屏失败", "", "$message（点「刷新」重试）")
                finishFrame()
                return@capture
            }
            worker.post {
                val result = HintPipeline.analyze(bitmap, classifier)
                main.post {
                    if (gen != generation.get()) {
                        // 会话已经停了、这一帧算作废：丢掉结果，别再往画面上画
                        Log.w(TAG, "丢弃过期帧的结果（gen=$gen）")
                        bitmap.recycle()
                        return@post
                    }
                    // 预览要先把像素画进另一张位图，所以回收放到 handle 之后
                    handle(bitmap, result)
                    bitmap.recycle()
                    finishFrame()
                }
            }
        }
    }

    /** 一帧结束。手动模式：**不排下一帧**，等用户点「刷新」。 */
    private fun finishFrame() {
        capturing.set(false)
        active.set(false)
        // "验证一次"不是会话，跑完就把悬浮窗收掉；会话里则留着等刷新
        if (!running.get()) overlay.hide()
    }

    private fun handle(bitmap: Bitmap, result: HintPipeline.Result?) {
        if (result == null) {
            val message = "定位失败：没找到成排的格子（确认目标游戏是全屏、截图没有黑边）"
            report(message)
            updateBar("未识别到棋盘", "", message)
            return
        }
        frames++

        val scan = result.scan
        val hints = result.hints
        val bricks = scan.empty.count { !it }
        // 一行里能消的对可能有很多对，控制条放不下，只报总数 + 前两对
        val hintText = when {
            result.cleared -> "盘面已清空"
            hints.isNotEmpty() -> buildString {
                append("共 ${hints.size} 对可消除：" )
                append(hints.take(2).joinToString("，") { it.pairText() })
                if (hints.size > 2) append(" …")
            }
            else -> "没有可消除的一对（多半是识别把同类判成了不同类）"
        }
        HintUiBus.state = HintUiState(
            running = running.get(),
            frames = frames,
            scanSummary = "${scan.rows}行×${scan.cols}列，砖块 $bricks，空位 ${scan.cellCount - bricks}，" +
                "格距 ${scan.pitchX}×${scan.pitchY}px，耗时 ${result.elapsedMs}ms（第 $frames 帧）",
            hintText = hintText,
            cleared = result.cleared,
            stuck = result.stuck,
            elapsedMs = result.elapsedMs,
            message = null,
        )

        val frame = HintOverlayView.Frame(
            background = null,
            scan = scan,
            state = result.state,
            hints = hints,
            showGrid = HintUiBus.showGrid,
            showLabels = HintUiBus.showLabels,
        )
        overlay.updateHint(frame)
        updateBar(hintText, "第 $frames 帧 · ${result.elapsedMs}ms")

        // 只在前几帧和每 10 帧打一条：真机排查时靠它判断"链路走通没有"，又不至于刷屏
        if (frames <= 3 || frames % 10 == 0) {
            Log.i(
                TAG,
                "第 $frames 帧：${scan.rows}x${scan.cols} 砖块 $bricks 空位 ${scan.cellCount - bricks} " +
                    "${result.elapsedMs}ms 可消对=${hints.size}" +
                    (if (result.stuck) "（识别疑似有误：有砖块但无解）" else ""),
            )
        }

        dumpMatrix(result)

        if (HintUiBus.previewEnabled) {
            HintUiBus.preview = renderPreview(frame, bitmap)
        }
    }

    /**
     * 每识别成功一帧，就把这一帧的**二维类别数组**单独存一个文件（`debug-frames/matrix/NNNN.txt`）。
     *
     * 0 = 空位；行优先，一行一格行。文件头带上格距/空位数/可消对，方便事后拿真实盘面回放算法。
     */
    private fun dumpMatrix(result: HintPipeline.Result) {
        val dir = context.getExternalFilesDir(DUMP_DIR) ?: return
        val sub = File(dir, MATRIX_DIR)
        if (!sub.exists() && !sub.mkdirs()) return
        matrixFiles++
        val scan = result.scan
        val labels = result.state.labels
        val file = File(sub, String.format("%04d.txt", matrixFiles))
        try {
            file.writeText(
                buildString {
                    appendLine("# ${scan.rows}x${scan.cols} pitch=${scan.pitchX}x${scan.pitchY} " +
                        "空位=${scan.empty.count { it }}/${scan.cellCount} 可消对=${result.hints.size} " +
                        "耗时=${result.elapsedMs}ms")
                    appendLine("# " + result.hints.joinToString(" ") { it.pairText() })
                    for (r in 0 until scan.rows) {
                        val from = r * scan.cols
                        appendLine((0 until scan.cols).joinToString(" ") { labels[from + it].toString() })
                    }
                },
            )
            Log.i(TAG, "第 $matrixFiles 帧的类别数组已写出：$file")
        } catch (e: Throwable) {
            Log.w(TAG, "类别数组写出失败", e)
        }
    }

    /** 把同一套绘制逻辑渲成调试图（截图 → 网格 → 类别 → 提示框），供界面核对标定对不对。 */
    private fun renderPreview(frame: HintOverlayView.Frame, background: Bitmap): Bitmap {
        val w = (background.width * PREVIEW_SCALE).toInt().coerceAtLeast(1)
        val h = (background.height * PREVIEW_SCALE).toInt().coerceAtLeast(1)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.scale(PREVIEW_SCALE, PREVIEW_SCALE)
        previewView.frame = frame.copy(background = background)
        previewView.measure(
            View.MeasureSpec.makeMeasureSpec(background.width, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(background.height, View.MeasureSpec.EXACTLY),
        )
        previewView.layout(0, 0, background.width, background.height)
        previewView.draw(canvas)
        return out
    }

    private fun updateBar(hintText: String, statusText: String, message: String? = null) {
        overlay.updateBar(ControlBarView.State(hintText = hintText, statusText = statusText, message = message))
    }

    private fun report(message: String) {
        HintUiBus.state = HintUiBus.state.copy(message = message)
    }

    private companion object {
        const val TAG = "HintSession"

        /** 截屏回调迟迟不回来时的兜底时限（分析那一段不设限，见类注释）。 */
        const val CAPTURE_TIMEOUT_MS = 3000L

        const val PREVIEW_SCALE = 0.5f

        /**
         * 调试图与体检报告的落盘目录（app 外部私有目录下）。
         * 取出来核对：`adb pull /sdcard/Android/data/com.qcmian.brickgame/files/debug-frames/`。
         */
        const val DUMP_DIR = "debug-frames"

        /** 每帧一个二维类别数组文件，存在 [DUMP_DIR] 下这个子目录里。 */
        const val MATRIX_DIR = "matrix"
    }
}
