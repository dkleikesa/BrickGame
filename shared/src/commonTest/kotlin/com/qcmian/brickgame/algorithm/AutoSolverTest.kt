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
 *  - [REAL_BOARD_A]：**可通关**。它曾被判"穷举后无解（可达图 768 个状态、最多消 28 块）"，
 *    实际是求解器漏了"把整条砖顶着走"的走法（见 [AutoSolverAgreementTest] 类注释）；
 *  - [REAL_BOARD_B]：**可通关**（完整解能清空 140 格）。
 *
 * 两盘现在都按"有解 + 逐步重放能清空"来校验。
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
        // 两块都在原地：谁当"被拖的那块"都行（走法序随机），只校验配对与原地消除
        val step = outcome.steps[0]
        assertTrue(step.isInPlace, "同款相邻必须原地消，不该产生位移：$step")
        assertEquals(setOf(AutoSolver.Cell(0, 0), AutoSolver.Cell(0, 1)), setOf(step.from, step.paired))
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
        val step = outcome.steps[0]
        // 两种移动方案都合法（拖 (0,0) 或拖 (2,1)），求解器挑哪个都算对
        assertTrue(!step.isInPlace, "两块不同线，必须有一次移动：$step")
        assertEquals(setOf(AutoSolver.Cell(0, 0), AutoSolver.Cell(2, 1)), setOf(step.from, step.paired))
        assertTrue(
            step.paired.row == step.to.row || step.paired.col == step.to.col,
            "落点必须与配对砖同行或同列：$step",
        )
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

    /**
     * 真机盘面 A 以前被判"穷举后无解"，其实是**求解器漏了走法**：它自己重推落点、
     * 还要求落点是空位，而拖动时整条砖会被顶着走。修好之后这盘能解 ——
     * "先报无解、再刷新又有解"就是这个假无解。
     */
    @Test
    fun realBoardAIsSolvableOnceRunPushingMovesAreAllowed() {
        val grid = REAL_BOARD_A.map { it.copyOf() }.toTypedArray()

        val outcome = AutoSolver.solve(grid, AutoSolver.Config.THOROUGH)
        assertIs<AutoSolver.Outcome.Solved>(
            outcome,
            "这盘真机截图有完整解；报无解说明走法枚举又退化了",
        )
        replayAndAssertCleared(REAL_BOARD_A, outcome.steps)
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

    @Test
    fun progressReportsCumulativeNodesAcrossAttempts() {
        // 每轮只给 0 个节点：第一次 dfs 走 1 个节点就被叫停，重启后累计继续涨。
        // 这种"小预算重启"正是进度最容易停在 0 的场景，所以每轮结束也必须报一次。
        val seen = mutableListOf<Long>()

        val outcome = AutoSolver.solve(
            arrayOf(intArrayOf(1, 2, 1, 2)),
            AutoSolver.Config(nodesPerAttempt = 0, maxAttempts = 3),
            onProgress = { seen += it },
        )

        assertIs<AutoSolver.Outcome.Unknown>(outcome)
        assertEquals(listOf(1L, 2L, 3L), seen, "每次尝试结束都要报累计节点数，不能每次从 0 重来")
        assertEquals(seen.last(), outcome.nodes)
    }

    @Test
    fun progressReportsNodeLevelCounts() {
        // 512 节点的预算至少要跨过 256 这个节点级回调点；这盘真机截图要几百个节点才解完。
        val seen = mutableListOf<Long>()

        val outcome = AutoSolver.solve(
            REAL_BOARD_A.map { it.copyOf() }.toTypedArray(),
            AutoSolver.Config(nodesPerAttempt = 512, maxAttempts = 1),
            onProgress = { seen += it },
        )

        assertTrue(seen.isNotEmpty(), "至少要报一次进度")
        assertEquals(seen.sorted(), seen, "累计节点数必须单调不减：$seen")
        assertTrue(seen.count { it % 256 == 0L } >= 1, "节点级进度必须按 256 的倍数报：$seen")
        if (outcome is AutoSolver.Outcome.Unknown) {
            assertEquals(outcome.nodes, seen.last(), "尝试结束时要补报真实总数，不能少掉最后的零头")
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
