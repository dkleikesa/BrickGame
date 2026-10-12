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
 * 为什么不做自动轮询：一帧要 1~2.5 秒（标定 + ONNX），自动跑就是一直占着 CPU、
 * 还容易在画面变化时给出过期提示；手动触发反而更可控。
 *
 * 线程分工：
 *  - 主线程：发起截屏（API 要求）、渲染调试图、更新悬浮窗；
 *  - `brick-hint-worker`：OpenCV 标定 + ONNX 推理是纯 CPU 活，不能占主线程。
 *
 * 两道兜底（都在真机上踩过）：
 *  - [CAPTURE_TIMEOUT_MS]：`takeScreenshot` 的回调有时永远不回来，不兜底就永久卡住；
 *  - [ANALYSIS_TIMEOUT_MS]：棋盘被识别歪时格子数会离谱（实测 199×89），
 *    下游 O(n²) 能把一帧拖到 25 秒以上，超时就丢弃该帧并重建 worker。
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
    private var workerThread = HandlerThread("brick-hint-worker").apply { start() }
    private var worker = Handler(workerThread.looper)

    /**
     * 帧代号。分析被判超时后，"还在跑"的那条旧线程算完还是会往主线程丢结果 ——
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
            Log.w(TAG, "截屏回调超时：${CAPTURE_TIMEOUT_MS}ms 内既没成功也没失败")
            updateBar("截屏超时", "", "takeScreenshot 回调超时，点「刷新」重试")
        }
    }

    private val analysisTimeout = Runnable {
        Log.w(TAG, "分析超过 ${ANALYSIS_TIMEOUT_MS}ms 还没回来 → 丢弃这一帧并重建 worker")
        generation.incrementAndGet()
        recreateWorker()
        active.set(false)
        if (capturing.compareAndSet(true, false)) {
            updateBar("分析超时，已跳过这一帧", "", "点「刷新」重试")
        }
    }

    private val previewView = HintOverlayView(context)

    private var frames = 0

    init {
        overlay.onRefresh = { refreshNow() }
        overlay.onClose = { stop() }
    }

    val isRunning: Boolean get() = running.get()

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
        main.removeCallbacks(analysisTimeout)
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

    private fun recreateWorker() {
        workerThread.quitSafely()
        workerThread = HandlerThread("brick-hint-worker").apply { start() }
        worker = Handler(workerThread.looper)
    }

    private fun captureFrame() {
        if (!active.get()) return
        if (!capturing.compareAndSet(false, true)) return
        val gen = generation.get()
        main.removeCallbacks(captureTimeout)
        main.removeCallbacks(analysisTimeout)
        main.postDelayed(captureTimeout, CAPTURE_TIMEOUT_MS)
        screenshots.capture { bitmap, reason ->
            main.removeCallbacks(captureTimeout)
            if (bitmap == null) {
                val message = reason ?: "截屏失败"
                Log.w(TAG, message)
                report(message)
                updateBar("截屏失败", "", "$message（点「刷新」重试）")
                finishFrame()
                return@capture
            }
            main.postDelayed(analysisTimeout, ANALYSIS_TIMEOUT_MS)
            worker.post {
                val result = HintPipeline.analyze(bitmap, classifier)
                main.post {
                    if (gen != generation.get()) {
                        // 这一帧已经被判超时、算作废了：丢掉结果，别再往画面上画
                        Log.w(TAG, "丢弃过期帧的结果（gen=$gen）")
                        bitmap.recycle()
                        return@post
                    }
                    main.removeCallbacks(analysisTimeout)
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
        val hint = result.hint
        val bricks = scan.empty.count { !it }
        val hintText = when {
            result.cleared -> "盘面已清空"
            hint != null -> hint.describe()
            else -> "没有可消除的一对（多半是识别把同类判成了不同类）"
        }
        HintUiBus.state = HintUiState(
            running = running.get(),
            frames = frames,
            scanSummary = "${scan.rows}行×${scan.cols}列，砖块 $bricks，空位 ${scan.cellCount - bricks}，" +
                "格距 ${scan.pitchX}×${scan.pitchY}px，耗时 ${result.elapsedMs}ms（第 $frames 帧）",
            classifier = result.classifierName,
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
            hint = hint,
            showGrid = HintUiBus.showGrid,
            showLabels = HintUiBus.showLabels,
        )
        overlay.updateHint(frame)
        updateBar(hintText, "第 $frames 帧 · ${result.elapsedMs}ms · ${result.classifierName}")

        // 只在前几帧和每 10 帧打一条：真机排查时靠它判断"链路走通没有"，又不至于刷屏
        if (frames <= 3 || frames % 10 == 0) {
            Log.i(
                TAG,
                "第 $frames 帧：${scan.rows}x${scan.cols} 砖块 $bricks 空位 ${scan.cellCount - bricks} " +
                    "${result.elapsedMs}ms 提示=${hint?.describe() ?: "无"}" +
                    (if (result.stuck) "（识别疑似有误：有砖块但无解）" else ""),
            )
        }

        if (HintUiBus.previewEnabled) {
            HintUiBus.preview = renderPreview(frame, bitmap)
        }
    }

    /** 把同一套绘制逻辑渲成调试图（截图 → 网格 → 类别 → 提示箭头），供界面核对标定对不对。 */
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

        /** 截屏回调迟迟不回来时的兜底时限。 */
        const val CAPTURE_TIMEOUT_MS = 3000L

        /**
         * 一帧分析（标定 + 识别 + 求解）的时限。正常情况下真机 14×10 约 1.5~2.5s，
         * 8s 已经远超合理范围 —— 再等下去就是白等。
         */
        const val ANALYSIS_TIMEOUT_MS = 8000L

        const val PREVIEW_SCALE = 0.5f
    }
}
