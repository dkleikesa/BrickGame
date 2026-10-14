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
    ) {
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
    ): Result? {
        val start = System.nanoTime()
        return try {
            val scan = OpenCvBoardScanner.scan(frame, minPitch) ?: return null
            val labels = classifier.classify(frame, scan)
            val state = BoardState.of(scan, labels)
            val hints = HintEngine.hints(scan, state)
            Result(
                scan = scan,
                state = state,
                hints = hints,
                lowConfidence = classifier.lowConfidenceCells(),
                elapsedMs = (System.nanoTime() - start) / 1_000_000,
            )
        } catch (e: Throwable) {
            Log.w(TAG, "分析失败", e)
            null
        }
    }
}
