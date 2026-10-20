package com.qcmian.brickgame.vision

import android.graphics.Bitmap
import android.util.Log

/**
 * 整条链路的门面：截图 → 切图 → 识别 → 接算法 → 可操作提示。
 *
 * 每一步都只依赖上一步的输出，任何一环挂了都返回 null 而不是抛异常
 * （游戏内每秒都要跑，不能因为某一帧没定位到网格就崩掉悬浮窗）。
 */
object HintPipeline {

    private const val TAG = "HintPipeline"

    class Result(
        /** 切图结果：网格几何 + 每格判空 + 32×32 裁切图。 */
        val scan: BoardScan,
        /** 识别结果：每格类别 id。 */
        val state: BoardState,
        /** 当前盘面上全部可消除的对（一行可能同时存在好几对）。 */
        val hints: List<Hint>,
        /** 这一帧模型没把握的格子（序号 → 相似度百分制）；非空就该把截图留档，见 [analyze]。 */
        val lowConfidence: List<Pair<Int, Int>>,
        val elapsedMs: Long,
        /**
         * 这一帧**实际**用的识别器短名（见 [TileClassifier.label]）。
         *
         * 带出来是为了让界面能证明"换模型真的生效了"：报告的是跑这一帧的那个实现，
         * 而不是用户刚点选的那个（两者在加载失败/服务没连上时会不一致）。
         */
        val modelLabel: String,
        /**
         * 识别那一段的耗时（裁格 + 推理 + 原型检索），**不含标定与求解**。
         *
         * 单独计时才有意义：整帧耗时被 OpenCV 标定和博弈求解主导（真机实测一帧 7~10s），
         * 换模型省下的那点比例在 [elapsedMs] 里几乎看不见。
         */
        val classifyMs: Long,
        /** 标定（定位网格 + 裁格 + 判空）耗时。 */
        val scanMs: Long,
        /** 接算法那一段（构造盘面 + 枚举可消对）；实测是微秒级，列出来是为了**排除**它。 */
        val hintsMs: Long,
        /** 识别内部的细分（阶段名 → ms），见 [TileClassifier.phaseTimings]。 */
        val classifyDetail: List<Pair<String, Long>> = emptyList(),
    ) {

        /** 一行分段耗时，给日志与留档 txt 用。 */
        fun timingText(): String = buildString {
            append("标定 ${scanMs}ms / 识别 ${classifyMs}ms")
            if (classifyDetail.isNotEmpty()) {
                append("（").append(classifyDetail.joinToString(" ") { "${it.first} ${it.second}ms" }).append("）")
            }
            append(" / 求解 ${hintsMs}ms / 共 ${elapsedMs}ms")
        }

        val cleared: Boolean get() = HintEngine.isCleared(state)
        val stuck: Boolean get() = HintEngine.isStuck(state)
    }

    /**
     * @param minPitch 格子最小像素间距，分辨率越低越要调大（默认 12，与 Python 一致）。
     * @return 定位失败返回 null。
     */
    fun analyze(
        frame: Bitmap,
        classifier: TileClassifier,
        minPitch: Int = OpenCvBoardScanner.MIN_PITCH,
        /** 定位实现；默认走 OpenCV。换 YOLO 只是换这一个 lambda。 */
        localize: BoardLocalizer = BoardLocalizer { OpenCvBoardScanner.scan(it, minPitch) },
    ): Result? {
        val start = System.nanoTime()
        return try {
            val scanStart = System.nanoTime()
            val scan = localize.locate(frame) ?: return null
            val scanMs = (System.nanoTime() - scanStart) / 1_000_000

            val classifyStart = System.nanoTime()
            val labels = classifier.classify(frame, scan)
            val classifyMs = (System.nanoTime() - classifyStart) / 1_000_000

            val hintsStart = System.nanoTime()
            val state = BoardState.of(scan, labels)
            val hints = HintEngine.hints(scan, state)
            val hintsMs = (System.nanoTime() - hintsStart) / 1_000_000

            Result(
                scan = scan,
                state = state,
                hints = hints,
                lowConfidence = classifier.lowConfidenceCells(),
                elapsedMs = (System.nanoTime() - start) / 1_000_000,
                modelLabel = classifier.label,
                classifyMs = classifyMs,
                scanMs = scanMs,
                hintsMs = hintsMs,
                classifyDetail = classifier.phaseTimings(),
            )
        } catch (e: Throwable) {
            Log.w(TAG, "分析失败", e)
            null
        }
    }
}
