package com.qcmian.brickgame.capture

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
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
 * 为什么不做自动轮询：一帧要好几秒（标定 + 模型推理，真机实测 7~8s），自动跑就是一直占着
 * CPU/GPU、还容易在画面变化时给出过期提示；手动触发反而更可控。
 *
 * 线程分工：
 *  - 主线程：发起截屏（API 要求）、更新悬浮窗；
 *  - `brick-hint-worker`：OpenCV 标定 + 模型推理是纯 CPU/GPU 活，不能占主线程。
 *
 * 分析**不设超时**：一帧识别本来就要几秒（真机实测 7~8s），卡死上限反而把好帧丢掉。
 * 只保留截屏回调的兜底（[CAPTURE_TIMEOUT_MS]：`takeScreenshot` 的回调有时永远不回来，
 * 不兜底就永久卡住），分析慢就等它慢，算完为止。
 *
 * 刷新期间**提示层整块藏起来**：从点「刷新」到算出新结果，屏幕上不画任何提示
 * （截屏要干净、分析中的旧提示又容易看错），只在换好新结果那一刻放出来；中途失败就
 * 保持空白 + 控制条上报错，绝不把上一帧的过期提示留在屏幕上（见 [captureFrame]）。
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
            Log.w(TAG, "截屏回调超时：${CAPTURE_TIMEOUT_MS}ms 内既没成功也没失败")
            updateBar("截屏超时", "", "takeScreenshot 回调超时，点「刷新」重试")
        }
    }

    private var frames = 0

    /**
     * 上一帧已经画出去的调试图层。
     *
     * 控制条上的两个调试开关（网格 / 类别号）是**即时生效**的：点一下直接拿这一帧重画，
     * 不用再截屏分析一遍 —— 否则点了开关要等下一次刷新才变，看着就像没点。
     */
    private var lastFrame: HintOverlayView.Frame? = null

    init {
        overlay.onRefresh = { refreshNow() }
        overlay.onClose = { stop() }
        overlay.onToggleGrid = { on ->
            HintUiBus.showGrid = on
            redrawHints()
        }
        overlay.onToggleLabels = { on ->
            HintUiBus.showLabels = on
            redrawHints()
        }
    }

    /** 开始提示：挂悬浮窗 + 立刻跑一帧，之后等用户点「刷新」。 */
    fun start() {
        if (!running.compareAndSet(false, true)) return
        overlay.show()
        HintUiBus.overlayWindowType = overlay.activeType
        HintUiBus.state = HintUiBus.state.copy(running = true)
        // 控制条上的开关跟着 App 界面里的开关走（两处共用 HintUiBus 这一个状态）
        overlay.setBarToggles(HintUiBus.showGrid, HintUiBus.showLabels)
        refreshNow()
    }

    fun stop() {
        if (!running.compareAndSet(true, false)) return
        active.set(false)
        capturing.set(false)
        main.removeCallbacks(captureTimeout)
        generation.incrementAndGet()          // 让还在跑的那一帧的结果失效
        overlay.hide()
        lastFrame = null
        HintUiBus.state = HintUiBus.state.copy(running = false)
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
        // 刷新一开始就把提示层收掉，直到算出新结果才放回来：
        //  1. 截图必须干净：上一帧画在屏幕上的提示框会污染本次截图（强边缘带偏标定、
        //     50% 红色改变砖块颜色）；
        //  2. 分析要好几秒，期间挂着的是**上一帧的旧提示**，玩家看着就是"提示不对"。
        // 只藏提示层，控制条留着（用户要靠它点「刷新」）。
        overlay.setHintLayerVisible(false)
        screenshots.capture { bitmap, reason ->
            // 这里**不**恢复可见性：整帧（截屏 + 分析）都保持隐藏，见上面第 2 条。
            // 失败分支也不恢复 —— 旧提示是过期信息，展示出来只会误导。
            main.removeCallbacks(captureTimeout)
            if (bitmap == null) {
                val message = reason ?: "截屏失败"
                Log.w(TAG, message)
                updateBar("截屏失败", "", "$message（点「刷新」重试）")
                finishFrame()
                return@capture
            }
            worker.post {
                // analyze 自己会把异常吞成 null；这里兜的是**识别器初始化**（懒加载模型/GPU delegate）
                // 抛出来的那类异常 —— 工作线程上没人接就是整个进程崩掉。失败不缓存，点「刷新」会重试。
                val analysis = runCatching { HintPipeline.analyze(bitmap, classifier) }
                // 模型没把握就趁原图还在手上留档（在 worker 线程写盘，别占主线程）
                analysis.getOrNull()?.let { LowConfStore.save(context, bitmap, it) }
                main.post {
                    if (gen != generation.get()) {
                        // 会话已经停了、这一帧算作废：丢掉结果，别再往画面上画
                        Log.w(TAG, "丢弃过期帧的结果（gen=$gen）")
                        bitmap.recycle()
                        return@post
                    }
                    val failure = analysis.exceptionOrNull()
                    if (failure != null) {
                        val message = "识别器不可用：${failure.message ?: failure.javaClass.simpleName}"
                        Log.w(TAG, message, failure)
                        updateBar("识别器不可用", "", "$message（点「刷新」重试）")
                    } else {
                        handle(analysis.getOrNull())
                    }
                    bitmap.recycle()
                    finishFrame()
                }
            }
        }
    }

    /**
     * 按当前两个调试开关重画上一帧（控制条上点了开关就走这里）。
     *
     * 没有上一帧（刚开会话、还没算出结果）就什么也不做 —— 下一帧自然会带上新开关。
     */
    private fun redrawHints() {
        val frame = lastFrame ?: return
        overlay.updateHint(
            frame.copy(showGrid = HintUiBus.showGrid, showLabels = HintUiBus.showLabels),
        )
    }

    /** 一帧结束。手动模式：**不排下一帧**，等用户点「刷新」。 */
    private fun finishFrame() {
        capturing.set(false)
        active.set(false)
    }

    private fun handle(result: HintPipeline.Result?) {
        if (result == null) {
            val message = "定位失败：没找到成排的格子（确认目标游戏是全屏、截图没有黑边）"
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
        HintUiBus.state = HintUiState(running = running.get(), frames = frames)

        val frame = HintOverlayView.Frame(
            scan = scan,
            state = result.state,
            hints = hints,
            showGrid = HintUiBus.showGrid,
            showLabels = HintUiBus.showLabels,
        )
        overlay.updateHint(frame)
        lastFrame = frame
        // 先换好新提示再放出来，避免"旧提示一闪 → 新提示"的中间态（见 captureFrame 的隐藏说明）
        overlay.setHintLayerVisible(true)
        // App 界面里也能改这两个开关，每帧顺手让控制条跟上，别让两边显示不一致
        overlay.setBarToggles(HintUiBus.showGrid, HintUiBus.showLabels)
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

    }

    private fun updateBar(hintText: String, statusText: String, message: String? = null) {
        overlay.updateBar(ControlBarView.State(hintText = hintText, statusText = statusText, message = message))
    }

    private companion object {
        const val TAG = "HintSession"

        /** 截屏回调迟迟不回来时的兜底时限（分析那一段不设限，见类注释）。 */
        const val CAPTURE_TIMEOUT_MS = 3000L
    }
}
