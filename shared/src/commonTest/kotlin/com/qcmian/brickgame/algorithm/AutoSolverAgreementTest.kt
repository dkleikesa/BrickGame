package com.qcmian.brickgame.algorithm

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * 求解器与"直接按 findActions 全量枚举"的对账。
 *
 * 为什么需要这一层：[AutoSolver.Search.resolve] **自己重新推落点**（只试矩形两个角），
 * 而提示层走的是 [BrickGameTipHelper.findActions] 给出的 6 元动作。两套推导只要有一处不等价，
 * 求解器就会漏掉真实存在的走法 —— 表现出来就是"盘面明明能消，却报无解"，
 * 而且穷举完还会把这个错误结论当成"确定无解"。
 *
 * 这里用随机盘面（同类砖成对投放）做穷举对账：
 *  - 求解器说 Solved → 按 steps 重放必须真的清空；
 *  - 求解器说 Unsolvable → 独立穷举也不能找到解；
 *  - 两者对"能不能解"的判断必须一致。
 */
class AutoSolverAgreementTest {

    /**
     * 现场抓到的假无解：拖 (0,0) 的 4 到 (0,1)（那块砖被顶着走），
     * 落点当时是 3 占着的 —— 旧实现要求"落点必须是空位"，这一步被判非法，
     * 整盘于是被穷举成"确定无解"。
     */
    @Test
    fun pushRunMoveIsNotLost() {
        val grid = arrayOf(
            intArrayOf(4, 3, 5, 2),
            intArrayOf(5, 2, 0, 0),
            intArrayOf(3, 4, 0, 0),
        )
        val outcome = AutoSolver.solve(copy(grid), AutoSolver.Config.THOROUGH)
        assertTrue(outcome is AutoSolver.Outcome.Solved, "这盘有解（4 步清空），求解器不该报无解：$outcome")
        replayAndAssertCleared(grid, (outcome as AutoSolver.Outcome.Solved).steps, -1)
    }

    @Test
    fun fuzzAgreesWithDirectEnumeration() {
        val rnd = Random(0x2026_1025)
        var solved = 0
        var unsolvable = 0
        var unknown = 0

        repeat(300) { trial ->
            val grid = randomBoard(rnd)
            val outcome = AutoSolver.solve(copy(grid), AutoSolver.Config.THOROUGH)

            when (outcome) {
                is AutoSolver.Outcome.Solved -> {
                    solved++
                    replayAndAssertCleared(grid, outcome.steps, trial)
                }

                AutoSolver.Outcome.Unsolvable -> {
                    unsolvable++
                    if (directlySolvable(copy(grid))) {
                        fail("第 $trial 盘：求解器判无解，但全量枚举找得到解\n${render(grid)}")
                    }
                }

                is AutoSolver.Outcome.Unknown -> unknown++
            }
        }

        assertTrue(solved > 0 && unsolvable > 0, "样本要同时覆盖有解/无解：solved=$solved unsolvable=$unsolvable unknown=$unknown")
    }

    // ---------------------------------------------------------------- 生成盘面

    /** 随机把成对的同类砖投到格子里：保证每类个数为偶数（游戏的硬不变量）。 */
    private fun randomBoard(rnd: Random): Array<IntArray> {
        val rows = 3 + rnd.nextInt(2)        // 3~4 行
        val cols = 4 + rnd.nextInt(2)        // 4~5 列
        val n = rows * cols
        val grid = Array(rows) { IntArray(cols) }
        val order = (0 until n).shuffled(rnd)
        var i = 0
        var cls = 1
        while (i + 1 < n && i < n) {
            // 3 成概率投放一对，剩下的留空
            if (rnd.nextFloat() < 0.28f) { i += 2; continue }
            val a = order[i]
            val b = order[i + 1]
            grid[a / cols][a % cols] = cls
            grid[b / cols][b % cols] = cls
            i += 2
            cls++
        }
        return grid
    }

    private fun copy(grid: Array<IntArray>) = Array(grid.size) { grid[it].copyOf() }

    // ---------------------------------------------------------------- 独立穷举

    /**
     * 直接用 [BrickGameTipHelper.findActions] 的全量动作做 DFS（带记忆化）。
     * 砖数逐步严格递减，状态图是 DAG，所以"访问即记死"是安全的。
     */
    private fun directlySolvable(start: Array<IntArray>): Boolean {
        val dead = HashSet<String>()

        fun dfs(grid: Array<IntArray>): Boolean {
            if (AutoSolver.isCleared(grid)) return true
            val key = grid.joinToString("/") { row -> row.joinToString(",") }
            if (!dead.add(key)) return false
            for (action in BrickGameTipHelper.findActions(grid)) {
                val child = copy(grid)
                val r = BrickGameTipHelper.actionStep(child, action)
                if (r != BrickGameTipHelper.ACTION_SUCCESS &&
                    r != BrickGameTipHelper.GAME_SUCCESS &&
                    r != BrickGameTipHelper.GAME_FAIL
                ) continue
                if (dfs(child)) return true
            }
            return false
        }

        return dfs(copy(start))
    }

    private fun replayAndAssertCleared(start: Array<IntArray>, steps: List<AutoSolver.Step>, trial: Int) {
        val grid = copy(start)
        steps.forEachIndexed { i, step ->
            val r = BrickGameTipHelper.actionStep(grid, step.toAction())
            assertTrue(
                r == BrickGameTipHelper.ACTION_SUCCESS ||
                    r == BrickGameTipHelper.GAME_SUCCESS ||
                    r == BrickGameTipHelper.GAME_FAIL,
                "第 $trial 盘第 $i 步不可执行（码 $r）：${render(start)}\nstep=$step",
            )
        }
        assertTrue(AutoSolver.isCleared(grid), "第 $trial 盘按解重放后没清空：\n${render(start)}")
    }

    private fun render(grid: Array<IntArray>): String =
        grid.joinToString("\n") { row -> row.joinToString(" ") { if (it == 0) "." else it.toString() } }
}
