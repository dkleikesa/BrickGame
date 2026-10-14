package com.qcmian.brickgame.vision

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * "识别结果 → 可操作提示"这一环的验证。
 *
 * 这里不需要截图和模型：手工造一个 [BoardScan]（几何 + 判空）+ 标签，看 [HintEngine] 给出的
 * 提示能不能真的被 `BrickGameTipHelper.actionStep` 执行掉。
 */
class HintEngineTest {

    /** 造一个 4×4 的假切图结果：格子边长 10px，四边留 10px 边距。 */
    private fun fakeScan(emptyMask: BooleanArray): BoardScan {
        val rows = 4
        val cols = 4
        val xlines = DoubleArray(cols + 1) { 10.0 + it * 10.0 }
        val ylines = DoubleArray(rows + 1) { 10.0 + it * 10.0 }
        val rects = Array(rows * cols) { index ->
            val r = index / cols
            val c = index % cols
            CellRect(10 + c * 10, 10 + r * 10, 20 + c * 10, 20 + r * 10)
        }
        val score = FloatArray(rows * cols) { if (emptyMask[it]) 0f else 100f }
        return BoardScan(
            rows = rows,
            cols = cols,
            xlines = xlines,
            ylines = ylines,
            score = score,
            scoreThreshold = 5f,
            rects = rects,
            pitchX = 10,
            pitchY = 10,
        )
    }

    @Test
    fun sameLineWithEmptyPathIsPairable() {
        // (0,0) 和 (0,3) 同款，中间 (0,1)(0,2) 空 → 往右滑过去撞上它，可消
        val empty = BooleanArray(16) { true }
        empty[0] = false
        empty[3] = false
        val scan = fakeScan(empty)

        val labels = IntArray(16)
        labels[0] = 7
        labels[3] = 7
        val state = BoardState.of(scan, labels)

        val hint = HintEngine.nextHint(scan, state)
        assertNotNull(hint, "同款两块砖在同一行且中间全空，应当能给出提示")
        assertTrue(hint.from == CellPoint(0, 0) || hint.pair == CellPoint(0, 0))
        assertTrue(hint.from == CellPoint(0, 3) || hint.pair == CellPoint(0, 3))

        // 棋盘矩形要落在扫描给出的格子上
        assertEquals(scan.rectOf(hint.from), hint.fromRect)
        assertEquals(scan.rectOf(hint.pair), hint.pairRect)
    }

    @Test
    fun bricksConnectedOnlyByBendAreNotPairable() {
        // (0,0) 与 (2,1)：周围全是空格，但两者不在一条直线/斜线上 —— 滑行只走直线、不拐弯，移不到
        val empty = BooleanArray(16) { true }
        empty[0] = false
        empty[2 * 4 + 1] = false
        val scan = fakeScan(empty)

        val labels = IntArray(16)
        labels[0] = 7
        labels[2 * 4 + 1] = 7
        val state = BoardState.of(scan, labels)

        assertNull(HintEngine.nextHint(scan, state), "靠拐弯才连上的两块砖在游戏里消不掉")
    }

    @Test
    fun emptyBoardHasNoHint() {
        val empty = BooleanArray(16) { true }
        val scan = fakeScan(empty)
        val state = BoardState.of(scan, IntArray(16) { 5 })

        assertTrue(state.labels.all { it == 0 }, "判空为空的格子必须被强制归零")
        assertTrue(HintEngine.isCleared(state))
        assertNull(HintEngine.nextHint(scan, state))
        assertTrue(HintEngine.hints(scan, state).isEmpty())
    }

    @Test
    fun boardWithUnpairableBricksIsStuck() {
        val empty = BooleanArray(16) { true }
        empty[0] = false
        empty[1] = false
        val scan = fakeScan(empty)
        val labels = IntArray(16)
        labels[0] = 1
        labels[1] = 2                 // 两块不同款，永远消不掉
        val state = BoardState.of(scan, labels)

        assertTrue(HintEngine.isStuck(state), "剩下砖块却无解应当判为 stuck（识别有误的信号）")
        assertNull(HintEngine.nextHint(scan, state))
    }
}
