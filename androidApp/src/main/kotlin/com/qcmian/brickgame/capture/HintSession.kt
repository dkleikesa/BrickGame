package com.qcmian.brickgame.capture

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.qcmian.brickgame.HintUiBus
import com.qcmian.brickgame.HintUiState
import com.qcmian.brickgame.algorithm.AutoSolver
import com.qcmian.brickgame.algorithm.BrickGameTipHelper
import com.qcmian.brickgame.ml.YoloBoardDetector
import com.qcmian.brickgame.ml.ModelVariant
import com.qcmian.brickgame.overlay.ControlBarView
import com.qcmian.brickgame.overlay.HintOverlayView
import com.qcmian.brickgame.overlay.OverlayController
import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.CellPoint
import com.qcmian.brickgame.vision.CellRect
import com.qcmian.brickgame.vision.HintEngine
import com.qcmian.brickgame.vision.BoardState
import com.qcmian.brickgame.vision.HintPipeline
import com.qcmian.brickgame.vision.TileClassifier
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** 取一帧屏幕；无障碍服务（takeScreenshot）与调试用的"喂图片"都实现它。 */
fun interface ScreenshotProvider {
    /** @param onResult (截图, 失败原因)；成功时原因必须为 null。 */
    fun capture(onResult: (Bitmap?, String?) -> Unit)
}

/**
 * 提示会话：默认**手动**的一帧一帧，可切到**自动通关**闭环。
 *
 * 手动模式：按"开始提示"跑一帧，之后只有点控制条上的「刷新」才再跑一帧 —— 不自动续帧。
 * 为什么不做自动轮询：一帧要好几秒（标定 + 模型推理，真机实测 7~8s），一直跑就是占着
 * CPU/GPU、还容易在画面变化时给出过期提示。
 *
 * 自动模式（[startAuto]）：**先整盘算一次，再一步一步点**，不是每帧重算。
 *
 * 每帧的 [handle] 结尾交给 [autoAdvance]：先用判空复检上一步该消的两格有没有真的消掉，
 * 没消掉就把计划作废重算；有计划就直接走下一步（不搜索）；计划走完才重新求一次整盘。
 * 逐格复检用的是"空没空"，比图标分类可靠得多，所以识别抖一帧不会把整局带崩。
 *
 * 四种终止都有明确报错，不会静默死循环：
 *  - 盘面清空 → "已通关"；
 *  - 穷举证明无解 → "本关无解，要洗牌"；
 *  - 连续 [BAD_CHECK_LIMIT] 次复检不符 → "滑动没生效"；
 *  - 重新求解超过 [PLAN_ROUND_LIMIT] 次 → "盘面一直对不上"。
 *
 * 线程分工：
 *  - 主线程：发起截屏（API 要求）、更新悬浮窗；
 *  - `brick-hint-worker`：YOLO 定位 + 模型推理是纯 CPU/GPU 活，不能占主线程。
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
    /** 自动运行"出手"用的手势通道（无障碍服务实现）。 */
    private val gestures: GestureExecutor,
    private val overlay: OverlayController,
    /** 按变体造一份识别器。**只在 worker 线程调用**（见 [acquireClassifier]）。 */
    private val classifierProvider: (ModelVariant) -> TileClassifier,
) {

    /**
     * 识别器**延迟到 worker 线程第一次用时**才构造：加载一个 4.4~13MB 的模型资产
     * 并建 TFLite 解释器，放在 `onServiceConnected` 里做就是主线程卡顿。
     *
     * 这两个字段**只在 worker 线程读写**（[acquireClassifier] / [closeClassifier] 都只被 worker
     * 上的任务调用）：换模型与"跑一帧"排队在同一条 HandlerThread 上，于是"关掉旧解释器"永远
     * 发生在当前这一帧算完之后 —— 不会 close 一个正在推理的解释器。
     */
    private var loaded: TileClassifier? = null

    /** YOLO 定位器（10MB 模型 + 解释器）。同 [loaded] 一样只在 worker 线程构造/关闭。 */
    private var yolo: YoloBoardDetector? = null

    private val main = Handler(Looper.getMainLooper())
    private val workerThread = HandlerThread("brick-hint-worker").apply { start() }
    private val worker = Handler(workerThread.looper)

    /**
     * 低置信度留档的写盘线程（单线程队列）。
     *
     * 单独开一条，而不是复用 [worker]：一整屏 1080x2400 的 PNG 无损压缩真机实测要 1.6 秒，
     * 比整帧识别（1 秒）还久。排在同一条队列上，用户连点两次刷新时下一帧的 analyze
     * 会被上一次写盘挡住 —— 照样是两三秒的体感。
     */
    private val dumpExecutor: ExecutorService =
        Executors.newSingleThreadExecutor { r -> Thread(r, "brick-lowconf-dump") }

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

    /**
     * 端到端计时的两个量（只在主线程读写）。
     *
     * 为什么要在 `elapsedMs` 之外单独记：[HintPipeline] 只量"分析"那一段，
     * 而用户感知的是"点刷新 → 看到提示"，中间还夹着**截屏**（`takeScreenshot` 是异步的，
     * 回调前的等待完全没被计量）。之前排查"体感三秒"就是在这段上卡住的。
     */
    private var frameStartMs = 0L
    private var captureMs = 0L

    private val captureTimeout = Runnable {
        if (capturing.compareAndSet(true, false)) {
            active.set(false)
            Log.w(TAG, "截屏回调超时：${CAPTURE_TIMEOUT_MS}ms 内既没成功也没失败")
            updateBar("截屏超时", "", "takeScreenshot 回调超时，点「刷新」重试")
        }
    }

    private var frames = 0

    /**
     * 自动通关开关。
     *
     * 自动与手动共用同一条"截图 → 识别 → 展示"链路，差别只在 [handle] 结尾交给 [autoAdvance]。
     */
    private val auto = AtomicBoolean(false)

    /** 自动运行已确认消掉的**对数**（一步一对）。 */
    private var autoSteps = 0

    /**
     * **整盘计划**：一次 [AutoSolver.solve] 算出来的完整解，之后逐步执行，不再每步重算。
     *
     * 为什么必须"一次算完再逐步点"（第一版是每帧重算，实测很快就失败）：
     *  - 每帧重算要连着成功 70 次，识别只要抖一帧（同类判成不同类）就会得到"无解"，
     *    整局就此中断 —— 失败概率随步数线性叠加；
     *  - 每帧重算真机上就是 70 次搜索，又慢又费电。
     * 于是改成"算一次、执行一整条"，只在**观察到盘面跟计划对不上**时才重新求解。
     */
    private var plan: List<AutoSolver.Step> = emptyList()

    /** 当前执行到 [plan] 的第几步。 */
    private var planIndex = 0

    /** 当前计划是按几行几列算的；执行时和当前 scan 不一致就说明网格换了坐标系。 */
    private var planRows = 0
    private var planCols = 0

    /** 连续几次因为网格尺寸变化而作废计划；防止"每帧都在变"时空转。 */
    private var gridMismatch = 0

    /**
     * 上一步**配对砖所在的格子**，下一帧拿它复检这一步有没有真的落地。
     *
     * 只看这一格的空没空，理由：
     *  - 判空（格子平坦度）远比图标分类可靠，整盘逐格比对会被识别噪声打爆；
     *  - 这一格在滑动前**必定是占着的**，它变空只可能是"这对砖被消掉了" ——
     *    既不会误判成成功，也不会因为落点位置跟模型有偏差而误判成失败。
     */
    private var pendingPair: CellPoint? = null

    /** 连续几次复检不符（上一步没落地）。 */
    private var badChecks = 0

    /** 一共重新求了几次解。它一直涨说明盘面/识别在飘，得停下来报错。 */
    private var planRounds = 0

    /**
     * 本次自动运行是否已经做过第一次完整求解。
     *
     * 只用来区分控制条文案（首帧 / 重算）；预算已经统一，两者都是
     * [SOLVE_CONFIG] / [SOLVE_TIME_BUDGET_MS]。
     */
    private var deepSolveDone = false

    /** 自动模式下连续几帧"根本没定位到棋盘"。 */
    private var missStreak = 0

    /**
     * 判无解之后又换了几帧复核，以及复核时看到的盘面。
     *
     * 一帧说无解不算数：判空靠 YOLO 的 tile 框，误检会把空格认成砖（tile 精确率只有
     * 66~69%），盘面凭空多出几块就"无解"了 —— 用户看到的就是"先显示无解、点刷新又有解"。
     * 所以换一帧再判：**同一个盘面读两遍都无解**才认，盘面变了就一直复核到上限。
     */
    private var unsolvableChecks = 0
    private var unsolvableGrid: String? = null

    /**
     * "无解 + 一对都消不掉"连续重试了几次。
     *
     * 这个状态下程序什么也做不了，只能等外部变化；但游戏是静止的，不设上限就是无限空转
     * （每 3 秒一次截屏 + 识别）。所以到 [UNSOLVABLE_RETRY_LIMIT] 次就停，让人去洗牌。
     */
    private var unsolvableRetries = 0

    /**
     * 上一次手势是否还没收到回调。
     *
     * `dispatchGesture` 的回调**可能永远不来**（游戏抢焦点、系统手势冲突），
     * 只等回调的话闭环会永久停住。所以派发的同时挂一个看门狗：回调先到就按回调走，
     * 看门狗先到就按"没收到"继续往下走 —— 用 CAS 保证两条路只走一条。
     */
    private val gesturePending = AtomicBoolean(false)

    /**
     * 上一帧已经画出去的调试图层。
     *
     * 控制条上的两个调试开关（网格 / 类别号）是**即时生效**的：点一下直接拿这一帧重画，
     * 不用再截屏分析一遍 —— 否则点了开关要等下一次刷新才变，看着就像没点。
     */
    private var lastFrame: HintOverlayView.Frame? = null

    init {
        overlay.onRefresh = { refreshNow() }
        overlay.onToggleAuto = { on -> if (on) startAuto() else stopAuto() }
        // 切定位算法只改落盘的选择，下一帧自然生效；不自动重跑 —— 此刻前台可能是 App 自己，
        // 截图必然定位失败，看起来就像"一切就坏"。
        overlay.onToggleGrid = { on ->
            HintUiBus.showGrid = on
            redrawHints()
        }
        overlay.onToggleLabels = { on ->
            HintUiBus.showLabels = on
            redrawHints()
        }
        // 无解继续：只改落盘开关，下一帧自动流程读它决定"退出还是继续贪心"
        overlay.onToggleContinue = { on ->
            AutoContinue.save(context, on)
            updateBar(if (on) "无解时继续自动玩：开" else "无解时继续自动玩：关",
                "点「刷新」重跑一帧")
        }
    }

    /**
     * 开始提示：**只挂悬浮窗，不跑帧**。
     *
     * 点「显示悬浮窗」时前台通常还是本 App，截到的不是游戏 —— 这一帧必然定位失败，
     * 白付几秒截屏+分析的代价。手动模式由用户切到游戏后点「刷新」；
     * 自动模式由 [startAuto] 自己起第一帧。
     */
    fun start() {
        if (!running.compareAndSet(false, true)) return
        overlay.show()
        HintUiBus.overlayWindowType = overlay.activeType
        // 顺手把上一次自动留下的报错清掉：重新开一次会话不该还挂着旧错误
        HintUiBus.state = HintUiBus.state.copy(running = true, autoMessage = null)
        // 控制条上的开关跟着 App 界面里的开关走（两处共用 HintUiBus 这一个状态）
        overlay.setBarAuto(auto.get())
        overlay.setBarToggles(HintUiBus.showGrid, HintUiBus.showLabels)
        overlay.setBarContinue(AutoContinue.enabled(context))
        updateBar("点「刷新」分析", "")
        // 首次把两个模型加载掉（约 12 秒），别让它压在第一帧上
        warmUpModels()
    }

    fun stop() {
        if (!running.compareAndSet(true, false)) return
        auto.set(false)
        active.set(false)
        capturing.set(false)
        main.removeCallbacks(captureTimeout)
        main.removeCallbacks(autoTick)
        generation.incrementAndGet()          // 让还在跑的那一帧的结果失效
        overlay.hide()
        overlay.setBarAuto(false)
        lastFrame = null
        HintUiBus.state = HintUiBus.state.copy(running = false, auto = false)
    }

    /**
     * 开始自动通关：清空进度，跑第一帧，之后由 [handle] → [autoAdvance] → 手势 → 下一帧闭环推进。
     *
     * 注意 [auto] **必须在起帧之前置位**：本方法在最后自己调 [refreshNow]，那一帧的 [handle]
     * 结尾要看到 auto=true 才会往下走，反过来的话第一帧就白跑了。
     */
    fun startAuto() {
        if (!auto.getAndSet(true)) {
            autoSteps = 0
            plan = emptyList()
            planIndex = 0
            planRows = 0
            planCols = 0
            gridMismatch = 0
            pendingPair = null
            badChecks = 0
            planRounds = 0
            missStreak = 0
            unsolvableRetries = 0
            unsolvableChecks = 0
            unsolvableGrid = null
            deepSolveDone = false
            HintUiBus.state = HintUiBus.state.copy(auto = true, autoSteps = 0, autoMessage = null)
            overlay.setBarAuto(true)
        }
        if (!running.get()) {
            start()
        }
        // start() 不再代为起帧；自动模式必须自己跑第一帧
        if (!capturing.get()) refreshNow()
        updateBar("识别中…", "")
    }

    /** 停止自动，回到手动提示（会话与悬浮窗都留着）。 */
    fun stopAuto() {
        if (!auto.getAndSet(false)) return
        main.removeCallbacks(autoTick)
        overlay.setBarAuto(false)
        HintUiBus.state = HintUiBus.state.copy(auto = false)
        updateBar("已停止", "已消 $autoSteps 对")
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
        // 一帧要好几秒（标定 + 推理），这段时间用户只能看这条状态行，自动模式下要说清在干什么
        if (auto.get()) {
            updateBar("识别中…", "")
        } else {
            updateBar("正在分析…", "")
        }
        captureFrame()
    }

    fun release() {
        stop()
        // 关闭也要排在 worker 上：主线程直接关会打断正在跑的那一帧。
        // quitSafely 会把已入队的这条消息处理完再退出。
        worker.post { closeClassifier() }
        workerThread.quitSafely()
        // shutdown（不是 shutdownNow）：让手上那张图写完，别留下半个文件
        dumpExecutor.shutdown()
    }

    /** 取识别器（只有一个变体，加载过就直接复用）。 */
    private fun acquireClassifier(): TileClassifier {
        loaded?.let { return it }
        closeClassifier()
        val want = ModelVariant.DEFAULT
        Log.i(TAG, "加载识别模型：${want.label}（${want.graphAsset}）")
        return classifierProvider(want).also { loaded = it }
    }

    /**
     * 预加载两个模型（YOLO 10.7MB + 识别 12.6MB）。
     *
     * 首次构造解释器实测约 12 秒（GPU delegate 初始化 ~8 秒、识别模型加载 ~3.5 秒），
     * 以前全压在第一帧上，用户看到的就是"首帧 13 秒"。这里在无障碍服务连上、
     * 悬浮窗刚打开时先在 worker 线程加载好，第一帧只剩 ~0.5 秒的推理。
     *
     * 重复调用是安全的：两个字段已经存在时直接复用，不再加载。
     */
    fun warmUpModels() {
        worker.post {
            val t0 = SystemClock.elapsedRealtime()
            val warm = loaded != null && yolo != null
            runCatching { acquireClassifier() }
                .onFailure { Log.w(TAG, "预加载识别模型失败", it) }
            runCatching { if (yolo == null) yolo = YoloBoardDetector(context) }
                .onFailure { Log.w(TAG, "预加载 YOLO 失败", it) }
            if (!warm) Log.i(TAG, "模型预加载完成，耗时 ${SystemClock.elapsedRealtime() - t0}ms")
        }
    }

    /** 只走 YOLO：模型加载失败或判不出可信网格就返回 null，这一帧直接跳过。 */
    private fun localize(frame: Bitmap): BoardScan? {
        val detector = yolo ?: runCatching { YoloBoardDetector(context).also { yolo = it } }
            .onFailure { Log.w(TAG, "YOLO 模型加载失败", it) }
            .getOrNull()
            ?: return null
        return runCatching { detector.scan(frame) }
            .onFailure { Log.w(TAG, "YOLO 定位异常", it) }
            .getOrNull()
    }

    private fun closeClassifier() {
        (loaded as? java.io.Closeable)?.let { runCatching { it.close() } }
        yolo?.let { runCatching { it.close() } }
        yolo = null
        loaded = null
    }

    private fun captureFrame() {
        if (!active.get()) return
        if (!capturing.compareAndSet(false, true)) return
        val gen = generation.get()
        // 刷新一开始就把提示层收掉，直到算出新结果才放回来：
        //  1. 截图必须干净：上一帧画在屏幕上的提示框会污染本次截图（强边缘带偏标定、
        //     50% 红色改变砖块颜色），**而且低置信度留档存的就是这张图** —— 红框烤进去，
        //     事后拿它补标注就把提示框一起训进去了；
        //  2. 分析要好几秒，期间挂着的是**上一帧的旧提示**，玩家看着就是"提示不对"。
        // 只藏提示层，控制条留着（用户要靠它点「刷新」）。
        overlay.setHintLayerVisible(false)
        // 关键时序：改可见性只是**排了一次重绘**，当帧的显示内容里还留着上一帧画上去的提示框。
        // 所以必须等它真的从画面上消失再截 —— 否则截到的就是"带着红框的那一帧"。
        main.postDelayed({ beginCapture(gen) }, HINT_HIDE_SETTLE_MS)
    }

    /** 提示层已经藏好之后才真正发起截屏（时序原因见 [captureFrame]）。 */
    private fun beginCapture(gen: Int) {
        // 等待期间会话被停掉/被新一代取代：这张图不要了，把在飞标记放掉
        if (!active.get() || gen != generation.get()) {
            capturing.set(false)
            return
        }
        main.removeCallbacks(captureTimeout)
        main.postDelayed(captureTimeout, CAPTURE_TIMEOUT_MS)
        frameStartMs = SystemClock.elapsedRealtime()
        screenshots.capture { bitmap, reason ->
            captureMs = SystemClock.elapsedRealtime() - frameStartMs
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
                val analysis = runCatching {
                    HintPipeline.analyze(bitmap, acquireClassifier()) { localize(it) }
                }
                val result = analysis.getOrNull()

                // 先把结果推给主线程，**留档写盘必须排在它后面**（见下面）。
                main.post {
                    if (gen != generation.get()) {
                        // 会话已经停了、这一帧算作废：丢掉结果，别再往画面上画。
                        // 这里**不回收位图** —— 回收统一交给写盘线程那个分支，
                        // 否则会和还在用它的 save 抢同一张 bitmap。
                        Log.w(TAG, "丢弃过期帧的结果（gen=$gen）")
                        return@post
                    }
                    val failure = analysis.exceptionOrNull()
                    if (failure != null) {
                        val message = "识别器不可用：${failure.message ?: failure.javaClass.simpleName}"
                        Log.w(TAG, message, failure)
                        updateBar("识别器不可用", "", "$message（点「刷新」重试）")
                    } else {
                        handle(result)
                    }
                    finishFrame()
                }

                // 模型没把握就把整屏原图留档（补标注重训用）。写盘走**独立线程**，两个理由：
                //  1. 一张 1080x2400 的 PNG 无损压缩真机实测要 1.6 秒（比整帧识别还久），
                //     串在 handle 前面就是用户感知到的"点了要等三秒"；
                //  2. 也不能排在 worker 队列里 —— 那样用户连点刷新时，下一帧的 analyze
                //     会被上一次写盘挡住，照样是两三秒。
                // 位图所有权一并移交：谁最后用谁回收。
                if (result != null) {
                    dumpExecutor.execute {
                        runCatching { LowConfStore.save(context, bitmap, result, result.scanSuspect) }
                            .onFailure { Log.w(TAG, "留档失败", it) }
                        if (!bitmap.isRecycled) bitmap.recycle()
                    }
                } else {
                    // 定位失败也留档：失败帧以前磁盘上一张都没有，标定挂了只能盲调（见 saveScanFail）
                    dumpExecutor.execute {
                        runCatching { LowConfStore.saveScanFail(context, bitmap) }
                            .onFailure { Log.w(TAG, "失败帧留档失败", it) }
                        if (!bitmap.isRecycled) bitmap.recycle()
                    }
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
            // 自动模式下这一帧不算数：连续几次都定位不到（多半是切走了应用/游戏不在前台）就停，
            // 否则会一直空转。定位到之后下面会把计数清零。
            if (auto.get()) {
                missStreak++
                if (missStreak >= MISS_LIMIT) {
                    finishAuto("连续 $MISS_LIMIT 帧没识别到棋盘：$message")
                } else {
                    main.postDelayed(autoTick, STEP_SETTLE_MS)
                }
            }
            return
        }
        missStreak = 0
        frames++

        val scan = result.scan
        val hints = result.hints
        val bricks = scan.empty.count { !it }
        // 坐标由悬浮窗画，文字只报"现在就能点"了几对；全被挡住时说清原因
        val direct = hints.count { it.direct }
        val hintText = when {
            result.cleared -> "盘面已清空"
            direct > 0 -> "$direct 对可消"
            hints.isNotEmpty() -> "${hints.size} 对同款，都被挡住（要先挪开挡路的砖）"
            else -> "没有同款可消的砖"
        }
        // copy 而不是新建：自动那三个字段（auto/autoSteps/autoMessage）也要跟着走，
        // 每帧 `HintUiState(...)` 会把它们冲回默认值，界面上"自动通关"就永远显示未启用。
        HintUiBus.state = HintUiBus.state.copy(running = running.get(), frames = frames)

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
        // 状态栏报**端到端**（点刷新 → 看到提示，含截屏），那才是用户感知的那个数；
        // 括号里补上识别段，换模型时看它。
        val totalMs = SystemClock.elapsedRealtime() - frameStartMs
        // 自动模式下这两行归 [autoAdvance] 管（识别中 / 求解中 / 第 k/N 步），
        // 这里再写一遍只会被同帧的下一刻覆盖，干脆不写；帧信息照旧进日志。
        // 总耗时单独写进控制条右侧：自动模式的状态行会被"求解中/第 k 步"反复改写，
        // 拼在状态行里一转眼就没了。分段明细仍只在 logcat 里（控制条放不下那么多字）。
        lastTiming = "总耗时 ${totalMs}ms"
        overlay.setBarTiming(lastTiming)
        if (!auto.get()) {
            updateBar(hintText, "剩 $bricks 块")
        }

        // 每帧一条：刷新是用户手动点的，一帧最多一行，不会刷屏。
        // 分段耗时写在里面 —— 线上"一帧十秒"这类问题只能靠这一行定位（见 Result.timingText）。
        // 前面单列"截屏"，后面单列"端到端"：这两个数都不在 result.timingText 里，
        // 但"体感几秒"的账必须把它们算进去。
        Log.i(
            TAG,
            "第 $frames 帧[${result.modelLabel}]：${scan.rows}x${scan.cols} 砖块 $bricks " +
                "空位 ${scan.cellCount - bricks} 可消对=${hints.size}" +
                (if (result.stuck) "（识别疑似有误：有砖块但无解）" else "") +
                " | 截屏 ${captureMs}ms " + result.timingText() + " | 端到端 ${totalMs}ms",
        )

        // 展示做完了，自动模式接着走计划：复检上一步 →（有计划就）点下一步 →（没计划才）求解
        if (auto.get()) autoAdvance(result)
    }

    // ------------------------------------------------------------------
    // 自动通关闭环
    // ------------------------------------------------------------------

    /**
     * 一帧的自动决策：**先复检上一步，再按计划走下一步；只有没计划时才搜索。**
     *
     * 一盘只算一次，之后每帧只有一次"截图 + 识别"的开销，不再有搜索：
     *  1. 盘面清空 → 通关收工；
     *  2. 复检上一步的**配对砖那一格**是否真的空了 —— 空了就把计划向后推一格，
     *     没空说明手势没生效，作废计划重算（连着几次都不符就报错停）；
     *  3. 计划里还有步子 → 直接走下一步（不搜索）；
     *  4. 计划走完 / 被作废 → 才求一次整盘。
     */
    private fun autoAdvance(result: HintPipeline.Result) {
        val state = result.state

        if (HintEngine.isCleared(state)) {
            finishAuto("已通关：盘面清空")
            return
        }

        // 1) 复检上一步：配对砖那一格空了吗
        pendingPair?.let { pair ->
            pendingPair = null
            if (state.labels[pair.row * state.cols + pair.col] == 0) {
                badChecks = 0
                // 有进展就把"重新求解"的配额清零：它拦的应该是"反复重算却没进展"，
                // 不该因为一盘 70 步里零星错几步就把次数耗完、提前收工。
                planRounds = 0
                autoSteps++
                planIndex++
                gridMismatch = 0
                HintUiBus.state = HintUiBus.state.copy(autoSteps = autoSteps)
            } else {
                // 这一步没落地：整条计划作废，重算（连续几次都不行就报错停）
                badChecks++
                plan = emptyList()
                planIndex = 0
                if (badChecks >= BAD_CHECK_LIMIT) {
                    finishAuto("连续 $BAD_CHECK_LIMIT 次复检不符：滑动没生效（手势被吞，或落点不对）")
                    return
                }
            }
        }

        // 2) 有计划就照着走，不搜索
        if (planIndex < plan.size) {
            playStep(result, plan[planIndex])
            return
        }

        // 3) 没计划（首次 / 走完 / 被作废）→ 求一次整盘
        if (planRounds >= PLAN_ROUND_LIMIT) {
            finishAuto("重新求解 $PLAN_ROUND_LIMIT 次仍没走完：盘面或识别一直对不上")
            return
        }
        planRounds++
        solveAndPlan(result)
    }

    /** 求一次整盘，成功就把它当成后续要执行的计划；无解/超预算各有明确出场。 */
    private fun solveAndPlan(result: HintPipeline.Result) {
        val grid = result.state.asGrid()
        val gen = generation.get()
        // 首帧和重算用同一份预算（12 万×16 / 120s）：原来重算只给 8 秒，
        // 8 秒一到就 Unknown、整局停在“已消 0 对”。现在只保留 120 秒这一道挂钟限制。
        val firstSolve = !deepSolveDone
        deepSolveDone = true
        val config = SOLVE_CONFIG
        val budgetMs = SOLVE_TIME_BUDGET_MS
        Log.i(
            TAG,
            "求解开始：${if (firstSolve) "首帧" else "重算"}（第 $planRounds 轮），" +
                "节点 ${config.nodesPerAttempt}×${config.maxAttempts}，挂钟 ${budgetMs}ms",
        )
        updateBar(
            if (firstSolve) "首帧求解中…（最长 ${budgetMs / 1000} 秒）" else "求解中…",
            "剩 ${countBricks(result.state)} 块",
        )

        worker.post {
            // 节点预算拦不住"单次尝试就是慢"的病态盘，再压一道挂钟上限
            val deadline = SystemClock.elapsedRealtime() + budgetMs
            val outcome = AutoSolver.solve(grid, config) {
                // 停止自动时立刻收工：首帧最长可能跑 120s，不该让线程空烧完
                SystemClock.elapsedRealtime() > deadline || !auto.get()
            }
            main.post {
                if (gen != generation.get() || !auto.get()) return@post
                when (outcome) {
                    is AutoSolver.Outcome.Solved -> {
                        if (outcome.steps.isEmpty()) {
                            finishAuto("已通关：盘面清空")
                            return@post
                        }
                        plan = outcome.steps
                        planIndex = 0
                        planRows = result.scan.rows
                        planCols = result.scan.cols
                        // 这一帧判的是"有解"：之前攒的无解复核计数作废
                        unsolvableChecks = 0
                        unsolvableGrid = null
                        Log.i(TAG, "自动计划就绪：${plan.size} 步（第 $planRounds 次求解）")
                        // 先把"一共多少步"亮出来再动手：不给这一下停顿的话，
                        // 紧接着的 [playStep] 会把它覆盖成"第 1/N 步"，用户根本看不到总数。
                        updateBar(
                            "共 ${plan.size} 步",
                            "剩 ${countBricks(result.state)} 块",
                        )
                        main.postDelayed({
                            if (gen == generation.get() && auto.get() && planIndex < plan.size) {
                                playStep(result, plan[planIndex])
                            }
                        }, PLAN_ANNOUNCE_MS)
                    }

                    AutoSolver.Outcome.Unsolvable -> {
                        if (!AutoContinue.enabled(context)) {
                            // 复核：同一盘面读两遍都无解才认。盘面变了说明上一帧的判空有误，
                            // 接着换帧，最多换 UNSOLVABLE_CONFIRM_LIMIT 次（见 [unsolvableChecks]）。
                            val key = result.state.labels.joinToString(",")
                            val sameBoard = key == unsolvableGrid
                            if (!sameBoard) {
                                unsolvableGrid = key
                                unsolvableChecks++
                                if (unsolvableChecks <= UNSOLVABLE_CONFIRM_LIMIT) {
                                    updateBar(
                                        "疑似无解，复核中 $unsolvableChecks/$UNSOLVABLE_CONFIRM_LIMIT",
                                        "剩 ${countBricks(result.state)} 块",
                                    )
                                    main.postDelayed(autoTick, UNSOLVABLE_CONFIRM_MS)
                                    return@post
                                }
                            }
                            finishAuto("本关无解，请在游戏里洗牌，然后点「刷新」")
                            return@post
                        }
                        // 补数据模式：无解说明盘面多半已经识别错了，继续走一步让游戏产生变化，
                        // 每帧强制留档（见 beginCapture），攒够再一次性回训练。
                        val action = BrickGameTipHelper.autoHint(grid)
                        if (action != null) {
                            unsolvableRetries = 0
                            playAction(result, action, "无解，继续消")
                        } else {
                            // 诊断：无解时把"配对存在但落点不通"这件事说清楚，别让人以为程序坏了
                            val moves = BrickGameTipHelper.findMoves(grid)
                            Log.w(TAG, "autoHint=null：找到 ${moves.size} 对同款，逐对检查落点")
                            moves.take(6).forEach { m ->
                                val cA = grid[m[0]][m[3]]   // 落点 (fromRow, pairCol)
                                val cB = grid[m[2]][m[1]]   // 落点 (pairRow, fromCol)
                                val ok = BrickGameTipHelper.resolveAction(grid, m) != null
                                Log.w(TAG, "  (${m[0]},${m[1]})+(${m[2]},${m[3]}) " +
                                    "落点值=$cA/$cB 可执行=$ok")
                            }
                            unsolvableRetries++
                            if (unsolvableRetries >= UNSOLVABLE_RETRY_LIMIT) {
                                finishAuto("本关无解，请在游戏里洗牌，然后点「刷新」")
                                return@post
                            }
                            // 这里只描述"程序自己在做什么"；要用户动手的话放在结束文案里说清楚
                            updateBar(
                                "无解，重试中 $unsolvableRetries/$UNSOLVABLE_RETRY_LIMIT",
                                "剩 ${countBricks(result.state)} 块",
                            )
                            finishFrame()
                            main.postDelayed(autoTick, UNSOLVABLE_RETRY_MS)
                        }
                    }

                    is AutoSolver.Outcome.Unknown -> {
                        // 预算烧完还没结论：**不落子**。
                        // 贪心走一步很可能把有解盘走成真无解 —— 这盘只是"还没算完"，
                        // 不是"不能解"。保留盘面原样，重开自动会重新给满 120 秒；
                        // 但搜索序是固定种子，同一盘面重试大概率还是同样的结果。
                        finishAuto(
                            "求解超预算（${outcome.nodes} 节点），已暂停且未落子。" +
                                "点「自动通关」重试",
                        )
                    }
                }
            }
        }
    }

    /**
     * 走计划里的下一步。
     *
     * 第一行报"第几步 / 一共几步"。总数 = **已确认消掉的对数 + 计划里还剩的步数**，
     * 中途重新求解过（计划变短）也不会跳号，用户看到的始终是"整盘的第几步"。
     */
    private fun playStep(result: HintPipeline.Result, step: AutoSolver.Step, note: String? = null) {
        val remaining = (plan.size - planIndex).coerceAtLeast(1)
        val total = autoSteps + remaining
        val current = autoSteps + 1
        playAction(result, step.toAction(), note ?: "第 $current/$total 步")
    }

    /**
     * 把一步 6 元动作翻成屏幕上的滑动。
     *
     * - `to != from`：滑块从 `from` 格滑到落点格（落点是空的；游戏靠"两块对齐"判定消除）；
     * - `to == from`：两块同款砖**已经同线且中间没有别的砖** —— 游戏里点一下直消，不用滑动。
     *   但**必须点两下**：一块砖可能同时和左右（或上下）两块同款砖同线，只点它一下游戏
     *   不知道要消哪一对，得再点一下指定配对砖（见 [GestureExecutor.tapPair]）；
     * - 其余情况：把 `from` 那块滑到落点格（落点是空的），靠"两块对齐"来消。
     */
    private fun playAction(result: HintPipeline.Result, action: IntArray, note: String? = null) {
        val scan = result.scan
        val fr = action[0]
        val fc = action[1]
        val tr = action[2]
        val tc = action[3]
        val pr = action[4]
        val pc = action[5]
        if (fr !in 0 until scan.rows || fc !in 0 until scan.cols ||
            tr !in 0 until scan.rows || tc !in 0 until scan.cols ||
            pr !in 0 until scan.rows || pc !in 0 until scan.cols
        ) {
            // 残局时 YOLO 会把空的顶行/首列裁掉，网格从 14x10 变成 13x10；
            // 这份计划是按旧坐标系算的，不能硬套到新网格上。作废重算，不要整局停 ——
            // 砖已经很少，重算基本是毫秒级。
            Log.w(
                TAG,
                "计划坐标越界（计划 ${planRows}x${planCols} / 当前 ${scan.rows}x${scan.cols}）" +
                    "→ 作废计划，按当前网格重算",
            )
            plan = emptyList()
            planIndex = 0
            pendingPair = null
            gridMismatch++
            if (gridMismatch >= GRID_MISMATCH_LIMIT) {
                finishAuto("识别网格反复变化（当前 ${scan.rows}x${scan.cols}），没法稳定执行")
                return
            }
            // 下一帧重新识别，autoAdvance 会走"没计划"分支重算
            main.postDelayed(autoTick, STEP_SETTLE_MS)
            return
        }

        val inPlace = fr == tr && fc == tc
        val from = scan.rectOf(CellPoint(fr, fc))
        val pairRect = scan.rectOf(CellPoint(pr, pc))
        val toRect = scan.rectOf(CellPoint(tr, tc))
        // 滑完之后那块砖是否同时对齐两块同款砖：是的话游戏会让玩家二选一，
        // 只在落点松手什么都不会消（真机实测：反复"滑动没生效"）。
        val ambiguous = !inPlace && ambiguousAfterMove(result.state, action)
        // 直消：第一下点起点、第二下点配对砖；滑动：起点滑到落点格中心（再多推一点，见 [overshoot]）。
        val target = if (inPlace) pairRect else toRect
        val x1 = from.centerX.toFloat()
        val y1 = from.centerY.toFloat()
        val swept = if (inPlace) target.centerX.toFloat() to target.centerY.toFloat()
        else overshoot(result.scan, target, tr - fr, tc - fc)
        val x2 = swept.first
        val y2 = swept.second
        val gestureMs = when {
            inPlace -> TAP_MS * 2 + TAP_GAP_MS
            ambiguous -> SWIPE_MS + TAP_GAP_MS + TAP_MS
            else -> SWIPE_MS
        }

        val gen = generation.get()
        // 记下配对砖那一格：下一帧 [autoAdvance] 拿它复检这一步有没有真的落地
        pendingPair = CellPoint(pr, pc)
        // 坐标/滑动方向屏幕上看得到，不再写进状态行；只报"还剩多少块"
        updateBar(note ?: "执行中", "剩 ${countBricks(result.state)} 块")

        // 手势发出去之后要排下一帧。步数不在这里记 —— 要等下一帧复检确认真的消掉了才算
        // （见 [autoAdvance]），否则"没生效的手势"也会被算成战果。
        val advance = {
            if (gen == generation.get() && auto.get()) {
                // 等游戏把消除动画播完再截下一帧，否则会截到"半消"的中间态
                main.postDelayed(autoTick, STEP_SETTLE_MS)
            }
        }
        val settle: (Boolean, String) -> Unit = { done, why ->
            if (gesturePending.compareAndSet(true, false)) {
                if (!done) Log.w(TAG, "手势未正常完成（$why）：($x1,$y1)→($x2,$y2)")
                advance()
            }
        }

        gesturePending.set(true)
        if (inPlace) {
            gestures.tapPair(x1, y1, x2, y2, TAP_MS, TAP_GAP_MS) { ok ->
                settle(ok, if (ok) "已点击" else "点击被取消/派发失败")
            }
        } else if (ambiguous) {
            Log.i(TAG, "二选一：滑到 ($tr,$tc) 后多块同款可配，补点配对砖 ($pr,$pc)")
            gestures.swipeThenTap(
                x1, y1, x2, y2,
                pairRect.centerX.toFloat(), pairRect.centerY.toFloat(),
                SWIPE_MS, TAP_MS, TAP_GAP_MS,
            ) { ok -> settle(ok, if (ok) "已滑动+补点" else "滑动/补点被取消") }
        } else {
            gestures.swipe(x1, y1, x2, y2, SWIPE_MS) { ok ->
                settle(ok, if (ok) "已滑动" else "滑动被取消/派发失败")
            }
        }
        // 看门狗：回调迟迟不来也别把闭环停死（复检不符的兜底会拦住"反复空推"）
        main.postDelayed({ settle(false, "回调超时") }, gestureMs + GESTURE_TIMEOUT_MS)
    }

    /**
     * 落点再往滑动方向多推 0.3 格。
     *
     * 为什么必须过冲：无障碍手势的坐标是**按路径长度均速插值**的，最后一条 MOVE 正好压在
     * 落点格中心，紧接着就抬手。游戏把拖动距离换算成格数是"取整 / 数过界"，终点恰好落在
     * 格边界上时，差几个像素就少算一格 —— 于是整条砖被推过去了，那块砖却没对上，什么也不消
     * （真机表现："还没拖到位就松手，移动了但没消失"）。多推 0.3 格，两种算法都还落在目标格上
     * （取整 1.3→1，过界也没多跨一个格子），同时给游戏的跟手插值留出追上来的时间。
     */
    private fun overshoot(scan: BoardScan, rect: CellRect, dRow: Int, dCol: Int): Pair<Float, Float> {
        val stepX = scan.pitchX * SWIPE_OVERSHOOT
        val stepY = scan.pitchY * SWIPE_OVERSHOOT
        val x = rect.centerX + when {
            dCol > 0 -> stepX
            dCol < 0 -> -stepX
            else -> 0f
        }
        val y = rect.centerY + when {
            dRow > 0 -> stepY
            dRow < 0 -> -stepY
            else -> 0f
        }
        // 贴边的那一列/行推不出去：夹回棋盘里，别让手指滑到盘外（游戏会当成取消）
        val left = scan.xlines.first().toFloat() + EDGE_INSET
        val right = scan.xlines.last().toFloat() - EDGE_INSET
        val top = scan.ylines.first().toFloat() + EDGE_INSET
        val bottom = scan.ylines.last().toFloat() - EDGE_INSET
        return x.coerceIn(left, right) to y.coerceIn(top, bottom)
    }

    /**
     * 这一步滑完之后，落点那块砖是否会同时对齐 ≥2 块同款砖。
     *
     * 判定与算法（[com.qcmian.brickgame.algorithm.BrickGameTipHelper]）一致：同行/同列、
     * 中间没有别的砖。≥2 时游戏要求玩家点一下指定消哪一对，光松手不会消。
     *
     * 推演**借算法自己算**（[BrickGameTipHelper.actionStep]），不再手推"落点换成这块砖"：
     * 拖动时整条砖会被顶着走，手推出来的盘面跟游戏差好几格，二选一就会漏点或多点一下。
     */
    private fun ambiguousAfterMove(state: BoardState, action: IntArray): Boolean {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairRow = action[4]
        val pairCol = action[5]
        val cls = state.labels[fromRow * state.cols + fromCol]
        if (cls <= 0) return false
        val after = state.asGrid()
        when (BrickGameTipHelper.actionStep(after, action)) {
            BrickGameTipHelper.ACTION_SUCCESS,
            BrickGameTipHelper.GAME_SUCCESS,
            BrickGameTipHelper.GAME_FAIL,
            -> Unit

            else -> return false
        }
        // actionStep 已经顺手把这一对消掉了：把这两格放回去，就是"刚落位、还没消"的盘面
        after[toRow][toCol] = cls
        after[pairRow][pairCol] = cls
        return linePartners(after, toRow, toCol, cls) >= 2
    }

    /** 同行/同列、中间无砖的第一个非空格如果是同款，就算一个可配对象。 */
    private fun linePartners(grid: Array<IntArray>, row: Int, col: Int, cls: Int): Int {
        val rows = grid.size
        val cols = grid[0].size
        var n = 0
        var c = col - 1
        while (c >= 0) {
            if (grid[row][c] != 0) { if (grid[row][c] == cls) n++; break }
            c--
        }
        c = col + 1
        while (c < cols) {
            if (grid[row][c] != 0) { if (grid[row][c] == cls) n++; break }
            c++
        }
        var r = row - 1
        while (r >= 0) {
            if (grid[r][col] != 0) { if (grid[r][col] == cls) n++; break }
            r--
        }
        r = row + 1
        while (r < rows) {
            if (grid[r][col] != 0) { if (grid[r][col] == cls) n++; break }
            r++
        }
        return n
    }

    /** 自动运行的心跳：一步走完了再来一帧。 */
    private val autoTick = Runnable {
        if (!auto.get() || !running.get()) return@Runnable
        refreshNow()
    }

    /** 自动收尾：关开关 + 把原因写在控制条上（红字），并说明已消多少对。 */
    private fun finishAuto(message: String) {
        auto.set(false)
        main.removeCallbacks(autoTick)
        overlay.setBarAuto(false)
        HintUiBus.state = HintUiBus.state.copy(auto = false, autoMessage = message, autoSteps = autoSteps)
        // 第一行给标题+统计，第二行才是原因（message 非空时控制条用红色显示它）——
        // 以前两行都塞 message，屏幕上就是同一句话重复两遍
        updateBar("已停止 · 已消 $autoSteps 对", "", message)
        Log.i(TAG, "自动结束：$message（已消 $autoSteps 对）")
    }

    private fun countBricks(state: com.qcmian.brickgame.vision.BoardState): Int =
        state.labels.count { it != 0 }

    /** 本帧分段耗时，自动模式也要看得到，所以挂在 [updateBar] 上统一追加。 */
    private var lastTiming = ""

    private fun updateBar(hintText: String, statusText: String, message: String? = null) {
        overlay.updateBar(ControlBarView.State(hintText = hintText, statusText = statusText, message = message))
    }

    private companion object {
        const val TAG = "HintSession"

        /** 截屏回调迟迟不回来时的兜底时限（分析那一段不设限，见类注释）。 */
        const val CAPTURE_TIMEOUT_MS = 3000L

        /**
         * 藏掉提示层之后、真正发起截屏之前要等的时长。
         *
         * 必须**至少跨一两帧**：`View.setVisibility(GONE)` 只是排了一次重绘，
         * 当帧的显示内容里还留着上一帧画上去的提示框，立刻截就会把红框截进图里
         * （低置信度留档存的就是这张图）。一帧 16ms，这里留足 150ms。
         */
        const val HINT_HIDE_SETTLE_MS = 150L

        /**
         * 求解参数：首帧和重算同一份预算，节点 12 万/次、最多重启 16 次。
         *
         * 原来重算用 4 万×8，32 万节点就先被节点预算掐死，根本轮不到挂钟上限；
         * 统一之后重算也能吃满下面的 120 秒。
         */
        val SOLVE_CONFIG = AutoSolver.Config(nodesPerAttempt = 120_000, maxAttempts = 16)

        /**
         * 求解的挂钟上限：**只保留这一道，120 秒**，首帧和重算一样。
         *
         * 8 秒原来是旧版“每帧都要算”的节流阀，后来被沿用到重算上：8 秒一到就返回
         * `Unknown`，自动层又不落子，整局就停在“已消 0 对”。节点预算仍负责拦内存
         * （置换表 16 B/状态，12 万约 4 MB），单次尝试特别慢的病态盘由这 120 秒兜底。
         */
        const val SOLVE_TIME_BUDGET_MS = 120_000L

        /**
         * 一步滑动的按住时长：太短会被系统当成 fling/滚动丢掉。
         *
         * 从 220ms 提到 320ms：手势是按路径长度均速插值的，220ms 走三四格时每一步只摊到
         * 两三个 MOVE 事件，游戏的跟手插值还没追上就已经抬手了。慢一点只是每步多 0.1 秒。
         */
        const val SWIPE_MS = 320L

        /**
         * 滑动落点往前的过冲比例（格）。0.3 是"取整/数过界"两种换算都不会多算一格的上限。
         * 见 [overshoot]。
         */
        const val SWIPE_OVERSHOOT = 0.3f

        /** 过冲时离棋盘边线留的余量（px），免得手指滑到盘外。 */
        const val EDGE_INSET = 2f

        /** 直消时每次"点一下"的按住时长。 */
        const val TAP_MS = 60L

        /**
         * 直消两次点击之间的间隔。
         *
         * 太短会被游戏当成一次双击/连点，"选中 → 消掉"就来不及走完；太长则拖慢每一步。
         */
        const val TAP_GAP_MS = 140L

        /**
         * 消掉一对之后等消除动画播完再截下一帧。
         *
         * 消除动画本身 0.7 秒，取 1 秒留余量：截早了会截到"半消"的中间态 ——
         * 那一帧的格子既不像原物也不像空位，分类相似度整体被拉低。
         */
        const val STEP_SETTLE_MS = 1000L

        /** 无解继续模式下，一步都走不动时的重试间隔。 */
        const val UNSOLVABLE_RETRY_MS = 3000L

        /** 无解且一对都消不掉时最多重试几次（10 次 ≈ 30 秒），超过就停下来让人洗牌。 */
        const val UNSOLVABLE_RETRY_LIMIT = 10

        /**
         * 判无解后最多换几帧复核。
         *
         * 只在"每帧读出来的盘面都不一样"时才会用满：盘面稳定下来（连着两帧同一张盘面
         * 都无解）就立刻收工，不会让真无解的盘白等。
         */
        const val UNSOLVABLE_CONFIRM_LIMIT = 3

        /** 复核间隔：比 UNSOLVABLE_RETRY_MS 短，只等一帧新截图。 */
        const val UNSOLVABLE_CONFIRM_MS = 600L

        /**
         * 求解完成后，把"一共多少步"亮一会儿再开始动手的时长。
         *
         * 不给这一下停顿的话，紧接着的下一步会立刻把提示覆盖掉，用户看不到总步数。
         */
        const val PLAN_ANNOUNCE_MS = 600L

        /** 复检连续这么多次不符（计划里的格子没被消掉），就停下报错。 */
        const val BAD_CHECK_LIMIT = 3

        /**
         * 网格尺寸连续变化这么多次就不硬撑了。
         *
         * 正常残局只会从 14x10 变一次 13x10（空顶行被裁掉），重算一两次就能收尾；
         * 一直变说明识别在飘，再重算也是空转。
         */
        const val GRID_MISMATCH_LIMIT = 3

        /** 重新求解的次数上限：一直重算说明盘面/识别在飘。 */
        const val PLAN_ROUND_LIMIT = 12

        /** 自动模式下连续这么多帧定位不到棋盘，就停下来报错（多半是切走了应用）。 */
        const val MISS_LIMIT = 3

        /** 手势回调的看门狗时限：超过它没收到回调，就自己去排下一帧。 */
        const val GESTURE_TIMEOUT_MS = 1500L
    }
}
