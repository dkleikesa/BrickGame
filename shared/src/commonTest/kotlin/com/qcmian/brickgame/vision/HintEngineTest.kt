package com.qcmian.brickgame.vision

import com.qcmian.brickgame.algorithm.BrickGameTipHelper
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
    fun hintIsExecutableOnSyntheticBoard() {
        // 只有 (0,0) 和 (2,1) 是砖块，且同款（类别 7）
        val empty = BooleanArray(16) { true }
        empty[0] = false
        empty[2 * 4 + 1] = false
        val scan = fakeScan(empty)

        val labels = IntArray(16)
        labels[0] = 7
        labels[2 * 4 + 1] = 7
        val state = BoardState.of(scan, labels)

        val hint = HintEngine.nextHint(scan, state)
        assertNotNull(hint, "同款两块砖应当能给出提示")
        assertTrue(hint.from == CellPoint(0, 0) || hint.pair == CellPoint(0, 0))
        assertTrue(hint.from == CellPoint(2, 1) || hint.pair == CellPoint(2, 1))
        // 落点必须与 from 同行或同列
        assertTrue(hint.to.row == hint.from.row || hint.to.col == hint.from.col)
        // 提示必须真的能被执行
        val grid = state.asGrid()
        assertEquals(
            BrickGameTipHelper.GAME_SUCCESS,
            BrickGameTipHelper.actionStep(
                grid,
                intArrayOf(hint.from.row, hint.from.col, hint.to.row, hint.to.col, hint.pair.row, hint.pair.col),
            ),
        )

        // 棋盘矩形要落在扫描给出的格子上
        assertEquals(scan.rectOf(hint.from), hint.fromRect)
        assertEquals(scan.rectOf(hint.to), hint.toRect)
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
