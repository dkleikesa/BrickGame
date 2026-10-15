package com.qcmian.brickgame.algorithm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * [AutoSolver] 的验证。
 *
 * 两个"真实盘面"是从真机截图（1080×2400，棋盘 14 行 × 10 列 = 140 格）上逐格抄下来的，
 * 每类个数全为偶数（游戏的硬不变量）；抄完还跟求解器算出的"开局可走 6 / 5 步"对过账，
 * 所以这两组数据可以当初值基准用：
 *  - [REAL_BOARD_A]：**无解**（穷举完整张可达图只有 768 个状态，最多只能消 28 块）；
 *  - [REAL_BOARD_B]：**可通关**（完整解能清空 140 格）。
 */
class AutoSolverTest {

    // ---------------------------------------------------------------- 小盘面

    @Test
    fun adjacentPairIsSolvedInPlace() {
        // 同款相邻 → 原地消，一步通关
        val grid = arrayOf(intArrayOf(1, 1))

        val outcome = AutoSolver.solve(grid)
        assertIs<AutoSolver.Outcome.Solved>(outcome)
        assertEquals(1, outcome.steps.size)
        assertEquals(AutoSolver.Cell(0, 0), outcome.steps[0].from)
        assertEquals(AutoSolver.Cell(0, 0), outcome.steps[0].to)
        assertEquals(AutoSolver.Cell(0, 1), outcome.steps[0].paired)
    }

    @Test
    fun blockedBoardIsProvedUnsolvable() {
        // A B A B：任何一块的最近邻都是不同款，一步都走不了 → 确证无解
        val grid = arrayOf(intArrayOf(1, 2, 1, 2))
        assertEquals(AutoSolver.Outcome.Unsolvable, AutoSolver.solve(grid))
    }

    @Test
    fun slidingIntoAlignmentIsSolved() {
        // (0,0) 与 (2,1) 同款：把 (0,0) 沿本行滑到 (0,1) 与它对齐，两步清空
        val grid = arrayOf(
            intArrayOf(7, 0, 0, 0),
            intArrayOf(0, 0, 0, 0),
            intArrayOf(0, 7, 0, 0),
            intArrayOf(0, 0, 0, 0),
        )
        val outcome = AutoSolver.solve(grid)
        assertIs<AutoSolver.Outcome.Solved>(outcome)
        assertEquals(1, outcome.steps.size)
        assertEquals(AutoSolver.Cell(0, 0), outcome.steps[0].from)
        assertEquals(AutoSolver.Cell(0, 1), outcome.steps[0].to)
        assertEquals(AutoSolver.Cell(2, 1), outcome.steps[0].paired)
    }

    @Test
    fun clearedBoardNeedsNoStep() {
        val grid = arrayOf(intArrayOf(0, 0), intArrayOf(0, 0))
        assertEquals(AutoSolver.Outcome.Solved(emptyList()), AutoSolver.solve(grid))
    }

    // ---------------------------------------------------------------- 真机盘面

    @Test
    fun realBoardBIsSolvedAndReplayClearsIt() {
        val grid = REAL_BOARD_B.map { it.copyOf() }.toTypedArray()

        val outcome = AutoSolver.solve(grid, AutoSolver.Config.DEFAULT)
        assertIs<AutoSolver.Outcome.Solved>(outcome, "这盘真机截图可通关，求解器必须给出完整解")

        replayAndAssertCleared(REAL_BOARD_B, outcome.steps)
    }

    @Test
    fun realBoardAIsProvedUnsolvable() {
        val grid = REAL_BOARD_A.map { it.copyOf() }.toTypedArray()

        val outcome = AutoSolver.solve(grid, AutoSolver.Config.THOROUGH)
        assertEquals(
            AutoSolver.Outcome.Unsolvable,
            outcome,
            "这盘真机截图穷举后无解；如果这里变成了 Solved/Unknown，说明算法或数据被改坏了",
        )
    }

    @Test
    fun solveDoesNotMutateCallerGrid() {
        val grid = REAL_BOARD_B.map { it.copyOf() }.toTypedArray()
        val before = grid.map { it.copyOf() }

        AutoSolver.solve(grid, AutoSolver.Config.DEFAULT)

        for (r in grid.indices) {
            assertTrue(before[r].contentEquals(grid[r]), "求解器不允许改动调用方的盘面（第 $r 行变了）")
        }
    }

    // ---------------------------------------------------------------- 工具

    /** 把解逐步喂回 [BrickGameTipHelper.actionStep]，确认最后真的清空、且每一步都合法。 */
    private fun replayAndAssertCleared(start: List<IntArray>, steps: List<AutoSolver.Step>) {
        assertTrue(steps.isNotEmpty(), "解不该是空的")
        val grid = start.map { it.copyOf() }.toTypedArray()
        for ((i, step) in steps.withIndex()) {
            val result = BrickGameTipHelper.actionStep(grid, step.toAction())
            val expected = if (i == steps.lastIndex) {
                BrickGameTipHelper.GAME_SUCCESS
            } else {
                BrickGameTipHelper.ACTION_SUCCESS
            }
            assertEquals(expected, result, "第 ${i + 1} 步 ${step} 执行结果不对")
        }
        assertTrue(AutoSolver.isCleared(grid), "按解执行完必须清空盘面")
    }

    private companion object {

        /**
         * 真机截图 A（第 3/4 层）——**无解**。
         *
         * 37 类：33 类各 4 个、4 类各 2 个（全偶数）；开局只有 6 步可走（3 处横向 + 3 处纵向同类相邻）。
         */
        val REAL_BOARD_A: List<IntArray> = listOf(
            intArrayOf(1, 2, 3, 1, 4, 5, 6, 7, 8, 9),
            intArrayOf(10, 11, 12, 13, 6, 14, 11, 15, 16, 17),
            intArrayOf(18, 19, 20, 21, 22, 12, 23, 3, 12, 8),
            intArrayOf(15, 1, 1, 24, 3, 25, 10, 26, 23, 27),
            intArrayOf(13, 10, 28, 20, 7, 17, 29, 27, 11, 18),
            intArrayOf(16, 13, 19, 30, 9, 29, 19, 27, 28, 31),
            intArrayOf(32, 4, 31, 20, 29, 5, 19, 11, 4, 2),
            intArrayOf(22, 20, 25, 23, 9, 27, 26, 14, 18, 7),
            intArrayOf(24, 5, 33, 13, 16, 14, 34, 15, 24, 31),
            intArrayOf(30, 28, 18, 4, 32, 36, 8, 14, 17, 15),
            intArrayOf(34, 16, 24, 29, 2, 23, 10, 35, 36, 17),
            intArrayOf(32, 37, 37, 26, 12, 6, 35, 30, 34, 31),
            intArrayOf(36, 7, 37, 8, 5, 32, 33, 30, 35, 9),
            intArrayOf(28, 2, 6, 26, 35, 21, 34, 36, 37, 3),
        )

        /** 真机截图 B（第 1 层）——**可通关**。35 类各 4 个，开局 5 步可走。 */
        val REAL_BOARD_B: List<IntArray> = listOf(
            intArrayOf(18, 22, 21, 9, 8, 32, 25, 26, 14, 24),
            intArrayOf(12, 31, 37, 11, 14, 6, 12, 19, 26, 11),
            intArrayOf(31, 29, 22, 11, 35, 4, 17, 10, 11, 32),
            intArrayOf(14, 25, 20, 3, 8, 3, 28, 4, 15, 29),
            intArrayOf(21, 18, 9, 4, 27, 7, 27, 31, 13, 28),
            intArrayOf(15, 21, 6, 13, 33, 24, 27, 7, 15, 25),
            intArrayOf(25, 37, 34, 24, 27, 19, 20, 19, 20, 19),
            intArrayOf(18, 6, 33, 20, 13, 28, 22, 34, 17, 30),
            intArrayOf(29, 33, 16, 13, 22, 12, 10, 17, 16, 8),
            intArrayOf(3, 17, 18, 4, 5, 14, 36, 36, 10, 15),
            intArrayOf(5, 31, 30, 16, 24, 29, 30, 16, 10, 9),
            intArrayOf(2, 5, 37, 32, 12, 2, 35, 35, 7, 36),
            intArrayOf(3, 34, 26, 7, 8, 5, 32, 33, 30, 9),
            intArrayOf(28, 2, 6, 26, 35, 21, 34, 36, 37, 2),
        )
    }
}
