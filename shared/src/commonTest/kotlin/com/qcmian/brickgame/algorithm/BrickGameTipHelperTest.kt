package com.qcmian.brickgame.algorithm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class BrickGameTipHelperTest {

    /** 复刻 lib2.jar 中 `BrickGame.main` 的用例，校验移植后行为与原 Java 版一致。 */
    @Test
    fun actionStepMatchesOriginalJavaSample() {
        val grid = arrayOf(
            intArrayOf(1, 0, 0, 1, 2, 1, 0, 0),
            intArrayOf(0, 0, 0, 0, 1, 1, 1, 0),
            intArrayOf(1, 0, 0, 0, 1, 4, 1, 1),
            intArrayOf(13, 1, 15, 16, 1, 1, 1, 1),
            intArrayOf(13, 2, 15, 16, 1, 1, 1, 1),
            intArrayOf(13, 1, 15, 16, 1, 1, 1, 1),
        )

        val ret = BrickGameTipHelper.actionStep(grid, intArrayOf(5, 1, 2, 1, 2, 4))

        assertEquals(BrickGameTipHelper.ACTION_SUCCESS, ret)
        val expected = arrayOf(
            intArrayOf(1, 1, 0, 1, 2, 1, 0, 0),
            intArrayOf(0, 2, 0, 0, 1, 1, 1, 0),
            intArrayOf(1, 0, 0, 0, 0, 4, 1, 1),
            intArrayOf(13, 0, 15, 16, 1, 1, 1, 1),
            intArrayOf(13, 0, 15, 16, 1, 1, 1, 1),
            intArrayOf(13, 0, 15, 16, 1, 1, 1, 1),
        )
        for (r in grid.indices) {
            assertTrue(expected[r].contentEquals(grid[r]), "row $r = ${grid[r].toList()}")
        }
    }

    /** 与原始 jar 逐条比对的 findMoves 输出。 */
    @Test
    fun findMovesMatchesOriginalJavaOutput() {
        assertEquals(
            listOf(listOf(0, 2, 1, 0)),
            BrickGameTipHelper.findMoves(
                arrayOf(
                    intArrayOf(0, 0, 7, 0),
                    intArrayOf(7, 0, 0, 0),
                )
            ).map { it.toList() },
        )

        assertEquals(
            listOf(listOf(0, 0, 0, 1)),
            BrickGameTipHelper.findMoves(
                arrayOf(
                    intArrayOf(1, 1, 0, 0),
                    intArrayOf(0, 0, 0, 0),
                )
            ).map { it.toList() },
        )

        assertEquals(
            listOf(listOf(0, 0, 2, 1)),
            BrickGameTipHelper.hints(
                arrayOf(
                    intArrayOf(5, 0, 0, 0),
                    intArrayOf(0, 0, 0, 0),
                    intArrayOf(0, 5, 0, 0),
                )
            ).map { listOf(it.fromRow, it.fromCol, it.pairRow, it.pairCol) },
        )
    }

    /** 自动提示给出的动作必须能被 actionStep 接受，并在清空后判定胜利。 */
    @Test
    fun autoHintYieldsExecutableWinningAction() {
        val cases = listOf(
            arrayOf(
                intArrayOf(5, 0, 0, 0),
                intArrayOf(0, 0, 0, 0),
                intArrayOf(0, 5, 0, 0),
            ),
            arrayOf(
                intArrayOf(0, 0, 7, 0),
                intArrayOf(7, 0, 0, 0),
            ),
            arrayOf(
                intArrayOf(1, 1, 0, 0),
                intArrayOf(0, 0, 0, 0),
            ),
        )

        for (grid in cases) {
            val action = BrickGameTipHelper.autoHint(grid)
            assertNotNull(action, "autoHint should find a move on ${grid.map { it.toList() }}")
            assertEquals(
                BrickGameTipHelper.GAME_SUCCESS,
                BrickGameTipHelper.actionStep(grid, action),
                "action ${action.toList()} should clear the board",
            )
        }
    }

    @Test
    fun noHintOnClearedBoard() {
        val grid = arrayOf(
            intArrayOf(0, 0, 0, 0),
            intArrayOf(0, 0, 0, 0),
        )
        assertTrue(BrickGameTipHelper.findMoves(grid).isEmpty())
        assertTrue(BrickGameTipHelper.hints(grid).isEmpty())
        assertNull(BrickGameTipHelper.firstHint(grid))
        assertNull(BrickGameTipHelper.autoHint(grid))
    }

    @Test
    fun illegalActionIsRejected() {
        val grid = arrayOf(
            intArrayOf(3, 0, 0, 4),
            intArrayOf(0, 0, 0, 0),
        )
        // 取空格作为起点
        assertEquals(
            BrickGameTipHelper.ACTION_FAIL,
            BrickGameTipHelper.actionStep(grid, intArrayOf(1, 1, 1, 1, 0, 0)),
        )
        // 不同色配对
        assertEquals(
            BrickGameTipHelper.ACTION_FAIL,
            BrickGameTipHelper.actionStep(grid, intArrayOf(0, 0, 0, 0, 0, 3)),
        )
    }
}
