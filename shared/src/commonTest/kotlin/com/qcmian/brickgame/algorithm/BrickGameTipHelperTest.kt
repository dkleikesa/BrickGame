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
    fun inPlacePairIsReportedOnlyOnce() {
        // 一对相邻同款砖，原地消从两头各能枚举出一条；两条落到的盘面相同，
        // findActions 只保留 from 行优先序在前的那一条，避免求解器白搜一遍。
        val actions = BrickGameTipHelper.findActions(arrayOf(intArrayOf(7, 7, 0, 0)))
        assertEquals(
            listOf(listOf(0, 0, 0, 0, 0, 1)),
            actions.map { it.toList() },
        )
    }

    /**
     * 落点被顶走的动作不能丢：拖一块砖时，它和空段之间的砖会被一起顶开，
     * 所以 `to` 可以是原本被别的砖占着的格子。目标反查实现如果只盯着"空格"，
     * 这四条就全漏了（这盘是差分测试踩出来的反例）。
     */
    @Test
    fun pushMoveTargetIsEnumerated() {
        val source = arrayOf(
            intArrayOf(0, 22, 3, 18, 19),
            intArrayOf(22, 0, 14, 4, 11),
            intArrayOf(14, 0, 0, 16, 7),
            intArrayOf(2, 0, 0, 6, 0),
            intArrayOf(4, 12, 16, 20, 21),
            intArrayOf(0, 13, 11, 19, 5),
            intArrayOf(13, 8, 0, 0, 20),
            intArrayOf(1, 0, 0, 6, 2),
            intArrayOf(5, 8, 0, 0, 0),
            intArrayOf(0, 17, 15, 9, 0),
            intArrayOf(18, 17, 1, 10, 0),
            intArrayOf(9, 12, 0, 7, 21),
        )
        val expected = setOf(
            listOf(2, 0, 1, 0, 1, 2),
            listOf(5, 1, 6, 1, 6, 0),
            listOf(6, 0, 6, 1, 5, 1),
            listOf(10, 2, 7, 2, 7, 0),
        )
        val actions = BrickGameTipHelper.findActions(source)
        val actual = actions.map { it.toList() }.toSet()
        assertTrue(actual.containsAll(expected), "落点被顶走的动作不能丢，实际动作：$actual")
        for (action in actions) {
            val probe = Array(source.size) { source[it].copyOf() }
            assertTrue(
                BrickGameTipHelper.actionStep(probe, action, checkTerminal = false) > 0,
                "枚举出的动作必须可执行：${action.toList()}",
            )
        }
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
