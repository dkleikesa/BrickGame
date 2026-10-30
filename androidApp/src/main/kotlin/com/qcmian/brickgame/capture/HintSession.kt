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
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicIntegerArray
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

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
     * 并行求解的线程池。
     *
     * 只给"一次要算几十秒"的搜索用：根节点的候选动作分给几个线程各搜一棵子树，
     * 谁先搜到就返回。线程常驻而不是每次新建 —— 一秒内可能连续求解多帧（自动通关），
     * 现开 4 条线程的代价比求解本身还大。
     */
    private val solvePool: ExecutorService =
        Executors.newFixedThreadPool(PARALLEL_SHARDS) { r -> Thread(r, "brick-solve") }

    /**
     * 最近一帧原图（单槽）。
     *
     * 为什么不能分析完就回收：求解超时 / 无解时要拿"当时那一帧"落盘，而求解本身要跑
     * 几秒到几十秒，那时位图早被回收了，再截一张屏幕已经是另一个画面。
     *
     * 读写全部发生在 [dumpExecutor] 这条单线程上（持有、替换回收、写盘），所以不会出现
     * "一边压缩一边被回收"。代价是常驻多一张整屏位图（1080x2400 约 10MB）。
     */
    private val heldFrame = AtomicReference<Bitmap?>(null)

    private fun holdLastFrame(bitmap: Bitmap) {
        heldFrame.getAndSet(bitmap)?.let { if (!it.isRecycled) it.recycle() }
    }

    private fun releaseLastFrame() {
        heldFrame.getAndSet(null)?.let { if (!it.isRecycled) it.recycle() }
    }

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
     * 本次求解已探索的节点数（控制条上的"已尝试了多少步"）。
     *
     * solver 在 worker 线程每 256 个节点写一次，主线程每秒读一次：
     * 求解不设挂钟（见 [SOLVE_TIME_BUDGET_MS]），如果状态行只写"求解中…"，用户对着一条
     * 几分钟不动的字分不清是卡死还是还在算。这个数每秒变一次，本身就是"还活着"的证据。
     */
    private val solveNodes = AtomicLong(0)

    /** 求解进度心跳的状态（只在主线程读写）。 */
    private var solving = false
    private var solveHint = ""
    private var solveRemaining = 0
    private var solveStartMs = 0L
    private var solveBudgetMs = 0L

    /** 每秒把进度写进控制条；求解结束时由 [stopSolveProgress] 摘掉。 */
    private val solveProgressTick = object : Runnable {
        override fun run() {
            if (!solving || !running.get() || !auto.get()) return
            val elapsedS = (SystemClock.elapsedRealtime() - solveStartMs) / 1000
            updateBar(
                solveHint,
                "已尝试 ${formatSteps(solveNodes.get())} 步 · 剩 $solveRemaining 块",
            )
            // 右上角的"总耗时"临时换成求解计时；无挂钟时只显示已用时长
            overlay.setBarTiming(
                if (solveBudgetMs > 0) "${elapsedS}s/${solveBudgetMs / 1000}s"
                else "${elapsedS}s（手动停止）",
            )
            main.postDelayed(this, SOLVE_PROGRESS_MS)
        }
    }

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
     * 上一次求解（以"没得出结论"收场）时的盘面 key 和当时的结论文案。
     *
     * 为什么记这个：搜索是确定性的 —— 同一张盘再算一遍必然还是同一个结论。
     * 所以盘面没变就不要再求解，直接把上次的结论报出来；只有盘面真变了才算新问题，
     * 才值得再花一次完整预算。
     */
    private var lastSolveKey: String? = null
    private var lastSolveMessage: String? = null

    /** 当前计划是基于哪张盘面算出来的。复检不符时用它区分"手势没生效"和"盘面真的变了"。 */
    private var planKey: String? = null

    /** 自动模式下连续几帧"根本没定位到棋盘"。 */
    private var missStreak = 0

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
        Log.i(TAG, "会话开始：${Diagnostics.header(context)}")
        // 首次把两个模型加载掉（约 12 秒），别让它压在第一帧上
        warmUpModels()
    }

    fun stop() {
        if (!running.compareAndSet(true, false)) return
        Log.i(TAG, "会话停止：帧数=$frames 已消=$autoSteps 对 未决帧=$planIndex/${plan.size}")
        auto.set(false)
        active.set(false)
        capturing.set(false)
        main.removeCallbacks(captureTimeout)
        main.removeCallbacks(autoTick)
        stopSolveProgress()
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
            lastSolveKey = null
            lastSolveMessage = null
            planKey = null
            HintUiBus.state = HintUiBus.state.copy(auto = true, autoSteps = 0, autoMessage = null)
            overlay.setBarAuto(true)
            Log.i(TAG, "自动开始")
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
        stopSolveProgress()
        overlay.setBarAuto(false)
        HintUiBus.state = HintUiBus.state.copy(auto = false)
        updateBar("已停止", "已消 $autoSteps 对")
        Log.i(TAG, "自动停止（手动）：已消 $autoSteps 对，未决帧=$planIndex/${plan.size}")
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
        // 单槽里那张原图不再需要了，回收掉（也排在写盘队列后面，不会打断正在写的文件）
        runCatching { dumpExecutor.execute { releaseLastFrame() } }
        // shutdown（不是 shutdownNow）：让手上那张图写完，别留下半个文件
        dumpExecutor.shutdown()
        // solvePool 不在这里关：release 之后可能马上又开会话，池子关了会抛 RejectedExecution；
        // 4 条常驻空闲线程的代价可以忽略，随进程一起结束。
    }

    /** 取识别器（只有一个变体，加载过就直接复用）。 */
    private fun acquireClassifier(): TileClassifier {
        loaded?.let { return it }
        closeClassifier()
        val want = ModelVariant.DEFAULT
        Log.i(TAG, "加载识别模型：${want.label}（${want.graphAsset}）")
        val t0 = SystemClock.elapsedRealtime()
        return classifierProvider(want).also {
            loaded = it
            Log.i(TAG, "识别模型加载完成：${it.name}，耗时 ${SystemClock.elapsedRealtime() - t0}ms")
        }
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
        val detector = yolo ?: runCatching {
            val t0 = SystemClock.elapsedRealtime()
            YoloBoardDetector(context).also {
                yolo = it
                Log.i(TAG, "YOLO 定位器加载完成，耗时 ${SystemClock.elapsedRealtime() - t0}ms")
            }
        }
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
                // 定位失败时把检测器给的具体原因带进留档：产物只有 png 的话，
                // "没找到 board"和"网格判不出"事后分不清，而这两处要改的代码完全不同。
                val scanFailReason = if (result == null) yolo?.lastFailure else null

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
                        // 不立刻回收：求解失败时还要用这一帧落盘，交给单槽持有（见 lastFrame）
                        holdLastFrame(bitmap)
                    }
                } else {
                    // 定位失败也留档：失败帧以前磁盘上一张都没有，标定挂了只能盲调（见 saveScanFail）
                    dumpExecutor.execute {
                        runCatching { LowConfStore.saveScanFail(context, bitmap, scanFailReason) }
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
            Log.w(TAG, "未识别到棋盘：$message（auto=${auto.get()}，已消 $autoSteps 对）")
            // 手动刷新失败也落一份日志：没 adb 权限时用户只能靠分享 ZIP 带出来
            if (!auto.get()) saveLogAsync("定位失败")
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
                (result.scanSuspect?.let { " ⚠$it" } ?: "") +
                (if (result.lowConfidence.isNotEmpty()) " 可疑格=${result.lowConfidence.size}" else "") +
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
                Log.w(
                    TAG,
                    "复检不符：配对格 (${pair.row},${pair.col}) 仍未消，计划作废" +
                        "（badChecks=$badChecks/${BAD_CHECK_LIMIT}）",
                )
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

        // 3) 没计划（首次 / 走完）→ 求一次整盘
        //
        // 但盘面没变就不再求解：搜索是确定性的，同一张盘再算一遍必然还是同一个结论，
        // 只会多等一次完整预算。直接把上次的结论报出来；盘面真变了才算新问题。
        val boardKey = state.labels.joinToString(",")
        if (boardKey == lastSolveKey) {
            finishAuto(lastSolveMessage ?: "这盘刚算过，结论不会变", state.asGrid())
            return
        }
        if (planRounds >= PLAN_ROUND_LIMIT) {
            finishAuto("重新求解 $PLAN_ROUND_LIMIT 次仍没走完：盘面或识别一直对不上")
            return
        }
        planRounds++
        solveAndPlan(result)
    }

    /**
     * 分片并行求解：把根节点的候选动作摊给几个线程，各自从子盘往下搜。
     *
     * 为什么这样分不改结论：
     *  - 每个分片的第一手互不相同，解集合不相交，合起来正好是整盘；
     *  - 某个分片搜到的解，前面补上它自己的第一手就是整盘的解；
     *  - 必须**所有**第一手都被穷举完，才敢说这盘无解。
     *
     * 代价：死表内存按分片摊（4 × 64MB ≈ 原来 1 × 256MB），单表更小、清空更频繁，
     * 所以加速比拿不满线程数，估 2~3 倍。
     */
    private fun solveSharded(
        grid: Array<IntArray>,
        config: AutoSolver.Config,
        abort: () -> Boolean,
        reportNodes: (Long) -> Unit,
    ): AutoSolver.Outcome {
        val roots = BrickGameTipHelper.findActions(grid)

        class Node(val steps: List<IntArray>, val board: Array<IntArray>, val root: Int)

        val rootTotal = IntArray(roots.size)              // 每个第一手最终切出几个任务
        val rootDead = AtomicIntegerArray(roots.size)      // 其中已证无解的任务数
        val rootBroken = BooleanArray(roots.size)          // 第一手本身就走不了（保守：不据此判无解）
        var immediate: AutoSolver.Outcome.Solved? = null

        // 第一层：落第一手
        var frontier = ArrayList<Node>()
        for ((ri, first) in roots.withIndex()) {
            val child = Array(grid.size) { grid[it].copyOf() }
            if (BrickGameTipHelper.actionStep(child, first, checkTerminal = false) <= 0) {
                rootBroken[ri] = true
                continue
            }
            if (child.all { row -> row.all { it == 0 } }) {
                immediate = solvedOf(listOf(first))
                break
            }
            frontier.add(Node(listOf(first), child, ri))
        }
        immediate?.let { return it }

        // 继续往下切，直到任务够多（或到深度上限）。
        // 为什么按需切而不是固定两级：第一手的子树大小能差几万倍（实测 156k / 14 / 422k / 10），
        // 固定两级在这盘上只把最大任务从 422k 降到 ~200k；切到第三、四级才真正摊平。
        var depth = 1
        while (frontier.size < TARGET_TASKS && depth < MAX_SPLIT_DEPTH) {
            val next = ArrayList<Node>(frontier.size * 2)
            for (node in frontier) {
                val moves = BrickGameTipHelper.findActions(node.board)
                if (moves.isEmpty()) {
                    // 走到死胡同：这条前缀本身就是死路，直接记成"已证无解"
                    rootTotal[node.root]++
                    rootDead.incrementAndGet(node.root)
                    continue
                }
                for (move in moves) {
                    // 必须从**父节点的盘**往下落这一手：`node.steps` 是完整前缀，盘也要是完整前缀的盘。
                    // 曾经错写成 copy 根盘，于是 steps 长 d、盘只少 1 对砖 —— 子搜索按更满的盘再解一遍，
                    // 计划长度 = d + (bricks-2)/2（140 块满盘就是 71/72 步），其中 d-1 步是把前缀已消掉
                    // 的那几对又消一次，落到真机就是 1~2 步空转。
                    val child = Array(grid.size) { node.board[it].copyOf() }
                    if (BrickGameTipHelper.actionStep(child, move, checkTerminal = false) <= 0) continue
                    if (child.all { row -> row.all { it == 0 } }) {
                        immediate = solvedOf(node.steps + move)
                        break
                    }
                    next.add(Node(node.steps + move, child, node.root))
                }
                if (immediate != null) break
            }
            frontier = next
            depth++
            immediate?.let { return it }
            if (frontier.isEmpty()) break
        }
        // 剩下的每个节点就是一个待搜索的任务
        for (node in frontier) rootTotal[node.root]++

        if (frontier.isEmpty()) {
            // 切到最后一条活路都没有：要么全是死胡同，要么第一手就走不了
            return if (rootBroken.any { it }) AutoSolver.Outcome.Unknown(0)
            else AutoSolver.Outcome.Unsolvable
        }

        val shards = minOf(frontier.size, PARALLEL_SHARDS, Runtime.getRuntime().availableProcessors())
        if (shards <= 1) return AutoSolver.solve(grid, config, abort, reportNodes)

        // 所有分片**共用一张死表**：不同前缀的后继盘面会重合，各自持表等于重复搜；
        // 内存也不乘份数 —— 同样 256MB 装的是"全盘唯一状态"。
        val shared = if (config.deadSlots > 0) AutoSolver.SharedDeadTable(config.deadSlots) else null
        Log.i(
            TAG,
            "并行求解：${shards} 个分片，任务 ${frontier.size} 个（切到第 $depth 手），" +
                (if (shared != null) "共享死表 ${shared.slots * 8L / 1048576}MB" else "死表按节点预算自动"),
        )

        val tasks = frontier
        val next = AtomicInteger(0)
        val solved = AtomicReference<AutoSolver.Outcome.Solved?>()
        val inconsistent = AtomicReference<AutoSolver.Outcome.Inconsistent?>()
        val totalNodes = AtomicLong(0)
        val latch = CountDownLatch(shards)

        repeat(shards) {
            solvePool.execute {
                try {
                    while (true) {
                        val i = next.getAndIncrement()
                        if (i >= tasks.size) break
                        if (solved.get() != null || inconsistent.get() != null || abort()) break
                        val task = tasks[i]
                        // 每个子搜索的节点数从 0 重新计，这里累加进总量，进度只增不减
                        var last = 0L
                        val outcome = AutoSolver.solve(
                            task.board,
                            config,
                            { solved.get() != null || inconsistent.get() != null || abort() },
                            onProgress = { nodes ->
                                totalNodes.addAndGet(nodes - last)
                                last = nodes
                                reportNodes(totalNodes.get())
                            },
                            sharedDead = shared,
                        )
                        when (outcome) {
                            is AutoSolver.Outcome.Solved -> solved.compareAndSet(
                                null,
                                AutoSolver.Outcome.Solved(task.steps.map { it.toStep() } + outcome.steps),
                            )
                            AutoSolver.Outcome.Unsolvable -> rootDead.incrementAndGet(task.root)
                            is AutoSolver.Outcome.Inconsistent ->
                                inconsistent.compareAndSet(null, outcome)
                            else -> Unit   // 没算完：这个任务不算"已证无解"
                        }
                    }
                } finally {
                    latch.countDown()
                }
            }
        }
        latch.await()

        solved.get()?.let { return it }
        inconsistent.get()?.let { return it }
        // 每个第一手的**所有**分支都被穷举完，才敢说整盘无解
        val allProven = roots.indices.all { ri ->
            !rootBroken[ri] && rootDead.get(ri) == rootTotal[ri]
        }
        if (allProven) return AutoSolver.Outcome.Unsolvable
        return AutoSolver.Outcome.Unknown(totalNodes.get())
    }

    /** 求一次整盘，成功就把它当成后续要执行的计划；无解/超预算各有明确出场。 */
    private fun solveAndPlan(result: HintPipeline.Result) {
        val grid = result.state.asGrid()
        val gen = generation.get()
        // 首帧和重算用**同一套深配置**：两者都是"必须给出结论"的时刻。
        // 历史教训：重算曾经是 12 万节点 ×16 次重启 / 120 秒，对需要连续几十万节点的盘
        // 等于每次都在 12 万处被切碎，16 次重来也拼不出一次完整搜索 —— 最后把
        // "已证无解"拖成"没算完"。慢一点可以接受，给错结论不行。
        val config = SOLVE_CONFIG
        val budgetMs = SOLVE_TIME_BUDGET_MS
        val nodesText = if (config.nodesPerAttempt == Long.MAX_VALUE) "无上限" else "${config.nodesPerAttempt}"
        Log.i(
            TAG,
            "求解开始（第 $planRounds 轮），节点 ${nodesText}×${config.maxAttempts}，" +
                "挂钟 ${if (budgetMs > 0) "${budgetMs}ms" else "无（手动停止）"}",
        )
        // 一进搜索就把进度立起来：第一秒先显示 0 步，之后每秒由 [solveProgressTick] 刷新。
        // 以前只留一句"求解中…（最长 300 秒）"，几分钟不动，分不清是卡死还是还在算。
        startSolveProgress(
            hint = "求解中…",
            remaining = countBricks(result.state),
            budgetMs = budgetMs,
        )

        worker.post {
            // 节点预算拦不住"单次尝试就是慢"的病态盘，再压一道挂钟上限
            val solveStart = SystemClock.elapsedRealtime()
            // budgetMs=0 表示不设挂钟：唯一停止条件是手动关自动/关悬浮窗。
            val deadline = if (budgetMs > 0) solveStart + budgetMs else Long.MAX_VALUE
            val abort = {
                !auto.get() || (budgetMs > 0 && SystemClock.elapsedRealtime() > deadline)
            }
            // solver 每 256 个节点回调一次累计节点数；这里只写原子量，UI 由主线程每秒取一次
            val reportNodes: (Long) -> Unit = { nodes -> solveNodes.set(nodes) }
            val outcome = try {
                // 首帧 256MB 死表；LargeHeap 下仍分配失败就按 10MB 梯子逐级降，绝不让 OOM 崩掉进程。
                solveSharded(grid, config, abort, reportNodes)
            } catch (t: Throwable) {
                if (t !is OutOfMemoryError) {
                    Log.e(TAG, "求解异常", t)
                    main.post {
                        if (gen == generation.get() && auto.get()) {
                            finishAuto("求解异常：${t.message ?: t.javaClass.simpleName}")
                        }
                    }
                    return@post
                }
                var last: Throwable = t
                var retried: AutoSolver.Outcome? = null
                for (slots in SOLVE_DEAD_SLOTS_FALLBACK) {
                    if (slots == config.deadSlots) continue
                    val mb = slots * 16 / (1024 * 1024)
                    try {
                        Log.w(TAG, "死表分配失败，降到 ${mb}MB（${slots} 槽）重试", last)
                        // 降级是重新开一次搜索，进度从 0 重新计数
                        solveNodes.set(0)
                        retried = AutoSolver.solve(grid, config.copy(deadSlots = slots), abort, reportNodes)
                        break
                    } catch (t2: Throwable) {
                        last = t2
                        if (t2 !is OutOfMemoryError) {
                            Log.e(TAG, "求解异常（降级重试）", t2)
                            main.post {
                                if (gen == generation.get() && auto.get()) {
                                    finishAuto("求解异常：${t2.message ?: t2.javaClass.simpleName}")
                                }
                            }
                            return@post
                        }
                    }
                }
                if (retried == null) {
                    Log.e(TAG, "求解内存不足：10MB 梯度全部失败", last)
                    main.post {
                        if (gen == generation.get() && auto.get()) {
                            finishAuto("求解内存不足：256MB→64MB 全部失败，请关闭其他应用或洗牌")
                        }
                    }
                    return@post
                }
                retried
            }
            val solveMs = SystemClock.elapsedRealtime() - solveStart
            main.post {
                // 先摘掉每秒心跳，再把控制条交还给各结局的文案
                stopSolveProgress()
                val tried = solveNodes.get()
                val rate = if (solveMs > 0) tried * 1000 / solveMs else 0
                Log.i(TAG, "求解结束：已尝试 $tried 步，${solveMs}ms，约 $rate 步/秒")
                if (gen != generation.get() || !auto.get()) return@post
                when (outcome) {
                    is AutoSolver.Outcome.Inconsistent -> {
                        Log.w(
                            TAG,
                            "盘面不一致：奇数类 ${outcome.oddClasses}（砖块不可能是单个，识别/重复框有误）",
                        )
                        rememberSolve(
                            result.state,
                            "识别可能出错：${outcome.oddClasses.size} 类砖块数为奇数（常见于重复框/误检），请刷新",
                        )
                        finishAuto(
                            "识别可能出错：${outcome.oddClasses.size} 类砖块数为奇数（常见于重复框/误检），请刷新",
                        )
                    }

                    is AutoSolver.Outcome.Solved -> {
                        if (outcome.steps.isEmpty()) {
                            finishAuto("已通关：盘面清空")
                            return@post
                        }
                        plan = outcome.steps
                        planIndex = 0
                        planKey = result.state.labels.joinToString(",")
                        planRows = result.scan.rows
                        planCols = result.scan.cols
                        // 这一帧有解：清掉"这盘算过"的记账（计划会推着盘面往前走）
                        lastSolveKey = null
                        lastSolveMessage = null
                        Log.i(TAG, "自动计划就绪：${plan.size} 步（第 $planRounds 次求解，${solveMs}ms）")
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
                        Log.w(TAG, "求解结论：无解（${solveMs}ms，剩 ${countBricks(result.state)} 块）")
                        if (!AutoContinue.enabled(context)) {
                            // 结论就是"无解"，直接报。以前这里还要"换一帧复核"，而那一步会
                            // 走通用求解路径、用弱配置重算 120 秒，把确定结论拖成"没算完"。
                            // 复核的意义本就是确认"盘面还是不是这一盘" —— 那件事现在由
                            // [lastSolveKey] 在下一帧用字符串比较完成，不需要再搜一次。
                            rememberSolve(result.state, UNSOLVABLE_MESSAGE)
                            finishAuto(UNSOLVABLE_MESSAGE, grid)
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
                                rememberSolve(result.state, UNSOLVABLE_MESSAGE)
                                finishAuto(UNSOLVABLE_MESSAGE, grid)
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

                    is AutoSolver.Outcome.MemoryLimited -> {
                        // 首帧已改成"表满清空继续搜"，正常走不到这里；留着是为了配置改回
                        // abortWhenFull=true 时仍能明确报出来，而不是表现成"没算完"。
                        Log.w(TAG, "求解死表满：${outcome.nodes} 节点")
                        val message = "求解中止：死表已满（${outcome.nodes} 节点），未落子"
                        rememberSolve(result.state, message)
                        finishAuto(message, grid)
                    }

                    is AutoSolver.Outcome.Unknown -> {
                        // 挂钟烧完还没结论：**不落子**。
                        // 贪心走一步很可能把有解盘走成真无解 —— 这盘只是"还没算完"，
                        // 不是"不能解"。和"已证明无解"必须分开说，否则用户会白白去洗牌。
                        Log.w(TAG, "求解未完成：${outcome.nodes} 节点，${solveMs}ms，剩 ${countBricks(result.state)} 块")
                        val message = "求解未完成（已 ${outcome.nodes} 步 / ${solveMs / 1000}s），未落子。" +
                            "这盘还没算出结论，可以先手动消几步破局"
                        rememberSolve(result.state, message)
                        finishAuto(message, grid)
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
        val stepTag = if (plan.isNotEmpty()) "第 ${planIndex + 1}/${plan.size} 步" else "贪心"
        Log.i(
            TAG,
            "$stepTag：($fr,$fc)→($tr,$tc) 配($pr,$pc) " +
                when {
                    inPlace -> "原地消"
                    ambiguous -> "滑动+补点"
                    else -> "滑动"
                },
        )
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

    /** 把 6 元动作数组转成求解器的 [AutoSolver.Step]。 */
private fun IntArray.toStep(): AutoSolver.Step =
    AutoSolver.Step(
        AutoSolver.Cell(this[0], this[1]),
        AutoSolver.Cell(this[2], this[3]),
        AutoSolver.Cell(this[4], this[5]),
    )

/** 由"已经走完的前缀"拼一个解出来（第一手/第二手就消完的盘用得上）。 */
private fun solvedOf(steps: List<IntArray>): AutoSolver.Outcome.Solved =
    AutoSolver.Outcome.Solved(steps.map { it.toStep() })

/** 记下"这张盘算过、结论是这个"：同一盘面再来就不再求解（见 [lastSolveKey]）。 */
    private fun rememberSolve(state: BoardState, message: String) {
        lastSolveKey = state.labels.joinToString(",")
        lastSolveMessage = message
    }

    /**
     * 自动收尾：关开关 + 把原因写在控制条上（红字），并说明已消多少对。
     *
     * @param grid 求解失败时的盘面。会写进落盘日志的文件头 —— 没有它，事后
     *   只能看到"算不出来"，复现不了到底是哪一盘点不动（留档只为补视觉标注，不存矩阵）。
     */
    private fun finishAuto(message: String, grid: Array<IntArray>? = null) {
        auto.set(false)
        main.removeCallbacks(autoTick)
        stopSolveProgress()
        overlay.setBarAuto(false)
        HintUiBus.state = HintUiBus.state.copy(auto = false, autoMessage = message, autoSteps = autoSteps)
        // 第一行给标题+统计，第二行才是原因（message 非空时控制条用红色显示它）——
        // 以前两行都塞 message，屏幕上就是同一句话重复两遍
        updateBar("已停止 · 已消 $autoSteps 对", "", message)
        Log.i(TAG, "自动结束：$message（已消 $autoSteps 对）")
        // 求解没算完（超预算 / 死表满 / 无解）时，连原图带盘面一起落盘：
        // 日志只回答"算了多久、多少节点"，看不出当时屏幕上是什么样
        if (grid != null) saveSolveFailAsync(message, grid)
        // 把这一轮的 logcat 落一份盘：没 adb 权限的手机靠分享 ZIP 把 logs/ 带出来
        saveLogAsync(message, grid)
    }

    /**
     * 把当前进程的 logcat 另存一份（后台写盘），分享 ZIP 会带上 `logs/` 下的文件。
     *
     * 只在"自动结束 / 手动定位失败"这种值得查的时刻调用；写盘约几十到几百毫秒，
     * 不能放主线程。会话已经 release 时队列已关，这里吞掉拒绝异常。
     */
    /**
     * 求解没算完时把"那一帧原图 + 当时的盘面"落一份（debug-frames/solve-fail/）。
     *
     * 和日志分开存：日志回答"算了多久、多少节点"，图回答"当时屏幕上到底是什么"。
     * 两者对得上，才能分清是盘面太大算不动，还是识别错了才无解。
     */
    private fun saveSolveFailAsync(reason: String, grid: Array<IntArray>) {
        runCatching {
            dumpExecutor.execute {
                LowConfStore.saveSolveFail(context, heldFrame.get(), reason, grid)
            }
        }.onFailure { Log.w(TAG, "求解失败帧留档排队失败", it) }
    }

    private fun saveLogAsync(reason: String, grid: Array<IntArray>? = null) {
        val detail = grid?.let { g ->
            "# 求解失败时的盘面（${g.size}x${g[0].size}，0 = 空位）\n" +
                g.joinToString("\n") { row -> row.joinToString(" ") }
        }
        runCatching {
            dumpExecutor.execute {
                Diagnostics.save(context, reason, detail)
                // 日志份数变了，顺手刷新分享按钮
                LowConfStore.refresh(context)
            }
        }.onFailure { Log.w(TAG, "日志留档排队失败", it) }
    }

    private fun countBricks(state: com.qcmian.brickgame.vision.BoardState): Int =
        state.labels.count { it != 0 }

    /** 本帧分段耗时，自动模式也要看得到，所以挂在 [updateBar] 上统一追加。 */
    private var lastTiming = ""

    private fun updateBar(hintText: String, statusText: String, message: String? = null) {
        overlay.updateBar(ControlBarView.State(hintText = hintText, statusText = statusText, message = message))
    }

    /**
     * 启动求解进度心跳：立刻写一帧"0 步"，之后每秒刷一次。
     *
     * 只在主线程调用；重复调用会先摘掉上一次的心跳（例如 OOM 降级重试前）。
     */
    private fun startSolveProgress(hint: String, remaining: Int, budgetMs: Long) {
        main.removeCallbacks(solveProgressTick)
        solving = true
        solveHint = hint
        solveRemaining = remaining
        solveBudgetMs = budgetMs
        solveStartMs = SystemClock.elapsedRealtime()
        solveNodes.set(0)
        updateBar(hint, "已尝试 0 步 · 剩 $remaining 块")
        overlay.setBarTiming(if (budgetMs > 0) "0s/${budgetMs / 1000}s" else "0s（手动停止）")
        main.postDelayed(solveProgressTick, SOLVE_PROGRESS_MS)
    }

    /**
     * 停止求解进度心跳，并把控制条右上角恢复成这一帧的端到端耗时。
     *
     * 求解结束 / 被叫停都要走这里，否则下一秒心跳会把"共 N 步"、"无解"之类的结论覆盖掉。
     */
    private fun stopSolveProgress() {
        if (!solving) return
        solving = false
        main.removeCallbacks(solveProgressTick)
        overlay.setBarTiming(lastTiming)
    }

    /**
     * 进度里的步数：千分位，读起来不容易看错位。
     *
     * 不折成"万"：慢盘一秒只走几百个节点，折成一位小数的万会十几秒才变一次，
     * 又变成"看着像卡死"；精确到个位才能每秒都看出在动。
     */
    private fun formatSteps(nodes: Long): String = String.format(Locale.US, "%,d", nodes)

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
         * 求解进度刷新间隔：1 秒一次。
         *
         * 太密会跟状态行其他更新抢重绘，太慢又看不出"还在动"；1 秒正好。
         */
        const val SOLVE_PROGRESS_MS = 1000L

        /**
         * 死表总预算（槽数，64-bit 指纹下每槽 8B）。
         *
         * 按堆上限的 40% 取，封顶 256MB：
         *  - 512MB 堆（largeHeap 常见）→ 200MB → 25M 槽 → 约 **1250 万状态**；
         *  - 256MB 堆（没吃到 largeHeap）→ 100MB → 12.5M 槽 → 约 625 万状态。
         * 早先固定 16M 槽 + 128-bit 指纹只有 839 万状态 —— "上千万"那类盘就是在这里被截断的。
         */
        private fun deadSlotBudget(): Int {
            val bytes = minOf(Runtime.getRuntime().maxMemory() * 2 / 5, 256L * 1024 * 1024)
            return (bytes / 8).toInt().coerceIn(1 shl 20, 1 shl 25)
        }

        /**
         * 求解参数：首帧和重算通用（**单次搜索 + 大死表，不再 16 次随机重启**）。
         *
         * - 节点不设限：一次搜索给足深度；
         * - 死表按堆预算走（见 [deadSlotBudget]），表满**清空继续搜**（abortWhenFull=false），
         *   宁可重复探索，也不要在表满处就把搜索掐掉；
         * - 分配失败时按 [SOLVE_DEAD_SLOTS_FALLBACK] 逐级退。
         */
        val SOLVE_CONFIG = AutoSolver.Config(
            nodesPerAttempt = Long.MAX_VALUE,  // 不设节点上限（注意 Int.MAX_VALUE 只是 21.4 亿，会被静默截断）
            maxAttempts = 1,
            deadSlots = deadSlotBudget(),
            abortWhenFull = false,
            // 84% 的死状态 64 个节点内就能重算，存它们只是白占内存（实测条目降到 15%、覆盖 93% 价值）
            deadMinCost = 64,
            // 剩 20 块起按"同线锁死形态"排序：这类盘面的胜负完全由最后几块的几何决定
            endgameBricks = 20,
            // 尾部盘面重合率极高，一律记下（不设阈值，宁多记不漏记）
            tailBricks = 8,
        )

        /**
         * 求解挂钟：**0 = 不设上限**，一直算到出结论（找到解 / 证明无解）或用户手动停。
         *
         * 为什么不要上限：算不动多半是"这盘本身极深"，掐掉只是把"还没算完"变成一次失败，
         * 下次重试还是同一盘同一搜索序。宁可挂着慢慢算，也别在中途说谎。
         * 代价是接口层没有自动退出，只有"关自动/关悬浮窗"能停 —— 控制条右上角会显示
         * 「已用 Ns（手动停止）」。死表满了会自动清空继续（见 SOLVE_CONFIG）。
         */
        const val SOLVE_TIME_BUDGET_MS = 0L

        /** 判无解的结论文案：报错处和"同盘再来不再重算"处共用，避免两处写法不一致。 */
        const val UNSOLVABLE_MESSAGE = "本关无解，请在游戏里洗牌，然后点「刷新」"

        /**
         * 首帧死表 OOM 的降级梯度：从 256MB 开始，每次减 10MB，最低 64MB。
         *
         * 1 槽 16B → 10MB = 655,360 槽。不再 256→128 一刀砍半；
         * 246/236/…/68 都可能刚好塞进当前堆，尽量保住搜索效率。
         */
        val SOLVE_DEAD_SLOTS_FALLBACK: List<Int> = buildList {
            val step = (10 * 1024 * 1024) / 8     // 64-bit 指纹：每槽 8B
            val min = (64 * 1024 * 1024) / 8
            var slots = SOLVE_CONFIG.deadSlots
            while (slots >= min) {
                add(slots)
                slots -= step
            }
        }

        /**
         * 并行求解的线程数上限（实际还会被"任务个数"和 CPU 核数压小）。
         *
         * 6 是折中：本机实测 4→6 线程还有 15% 收益，6→8 只剩 4%；
         * 手机上大核就 1+3 个，再多线程只会挤到小核上发热。
         */
        const val PARALLEL_SHARDS = 6

        /** 切分任务的目标个数：少于这个数就继续往下切一手（深度上限见 [MAX_SPLIT_DEPTH]）。 */
        const val TARGET_TASKS = 32

        /**
         * 切分深度上限（手数）。
         *
         * 实测同一盘：按第一手切（4 个任务）加速 1.05x，第三手 30 个任务 3.2x，第四手 45 个任务 4.7x。
         * 再深就只剩冗余（每个任务都要重建棋盘、共享表也压不住）。
         */
        const val MAX_SPLIT_DEPTH = 4

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
