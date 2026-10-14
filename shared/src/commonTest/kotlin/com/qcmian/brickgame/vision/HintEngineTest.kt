package com.qcmian.brickgame.vision

import com.qcmian.brickgame.algorithm.BrickGameTipHelper
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * "识别结果 → 可操作提示"这一环的验证。
 *
 * 这里不需要截图和模型：手工造一个 [BoardScan]（几何 + 判空）+ 标签，看 [HintEngine] 给出的
 * 提示是不是 [BrickGameTipHelper.findMoves] 的全量结果。
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

        val hint = HintEngine.hints(scan, state).firstOrNull()
        assertNotNull(hint, "同款两块砖在同一行且中间全空，应当能给出提示")
        assertTrue(hint.from == CellPoint(0, 0) || hint.pair == CellPoint(0, 0))
        assertTrue(hint.from == CellPoint(0, 3) || hint.pair == CellPoint(0, 3))

        // 棋盘矩形要落在扫描给出的格子上
        assertEquals(scan.rectOf(hint.from), hint.fromRect)
        assertEquals(scan.rectOf(hint.pair), hint.pairRect)
    }

    @Test
    fun pairIsFoundBySlidingIntoAlignment() {
        // (0,0) 与 (2,1) 同款：把 (0,0) 沿本行向右滑到 (0,1)，就与正下方的 (2,1) 对齐成对。
        // 提示只报"哪两块"，落点由游戏自己判定（不走 resolveAction）。
        val empty = BooleanArray(16) { true }
        empty[0] = false
        empty[2 * 4 + 1] = false
        val scan = fakeScan(empty)

        val labels = IntArray(16)
        labels[0] = 7
        labels[2 * 4 + 1] = 7
        val state = BoardState.of(scan, labels)

        val hint = HintEngine.hints(scan, state).firstOrNull()
        assertNotNull(hint, "滑到对齐位就能消：两块同款砖在算法枚举出的走法里")
        assertEquals(CellPoint(0, 0), hint.from, "该滑的是 (0,0) 那块")
        assertEquals(CellPoint(2, 1), hint.pair, "配对砖是 (2,1)")
    }

    /** 提示必须与 [BrickGameTipHelper.findMoves] 一一对应：不做任何过滤。 */
    @Test
    fun hintsMatchFindMovesExactly() {
        val boards = listOf(
            intArrayOf(
                0, 0, 7, 0,
                7, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
            ),
            intArrayOf(
                5, 0, 0, 0,
                0, 0, 0, 0,
                0, 5, 0, 0,
                0, 0, 0, 0,
            ),
            intArrayOf(
                1, 1, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
            ),
        )
        for (labels in boards) {
            val empty = BooleanArray(16) { labels[it] == 0 }
            val scan = fakeScan(empty)
            val state = BoardState.of(scan, labels)
            assertEquals(
                BrickGameTipHelper.findMoves(state.asGrid()).size,
                HintEngine.hints(scan, state).size,
                "提示条数应等于 findMoves 的候选数：${labels.toList()}",
            )
        }
    }

    @Test
    fun emptyBoardHasNoHint() {
        val empty = BooleanArray(16) { true }
        val scan = fakeScan(empty)
        val state = BoardState.of(scan, IntArray(16) { 5 })

        assertTrue(state.labels.all { it == 0 }, "判空为空的格子必须被强制归零")
        assertTrue(HintEngine.isCleared(state))
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
        assertTrue(HintEngine.hints(scan, state).isEmpty(), "无解时一条提示都不给")
    }
}
