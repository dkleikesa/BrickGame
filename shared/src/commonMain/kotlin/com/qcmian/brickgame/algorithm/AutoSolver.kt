package com.qcmian.brickgame.algorithm

import kotlin.random.Random

/**
 * 自动通关求解器。
 *
 * ## 为什么是"DFS + 置换表 + 随机走法序 + 重启"
 *
 * 1. **状态图是 DAG**：每一步 [BrickGameTipHelper.actionStep] 必然消掉 2 块砖，砖数严格递减，
 *    永远回不到同一个盘面。所以「从这盘无解」这个结论可以安全缓存（置换表），不会因为环而误判。
 * 2. **穷举能证无解**：实测 14×10 的盘面可达状态大多只有几百到几万个，穷举一遍很便宜。
 *    这个结论是刚需 —— 无解就得洗牌，不能像 Beam Search 那样把"没搜到"当成"无解"。
 * 3. **走法排序必须随机**：曾经按「走完后可行动作数降序」排，结果是个陷阱 ——
 *    它专门把人往"看着能动、实则必死"的大死区里带（51 盘里 3 盘烧到 20 万节点、最慢 88 秒）。
 *    换成随机序 + 小预算重启后，同样这 51 盘全部给出确定结论，平均 504ms、最慢 5.4s。
 * 4. **随机序不能只跑一次**：单次随机序方差极大（有的盘 71 节点就出解，有的要 2 万节点），
 *    重启正好抹平这个方差。而"任一尝试若在预算内穷举完"依旧保留了证无解的能力。
 *
 * ## 用法
 *
 * ```
 * when (val r = AutoSolver.solve(grid, AutoSolver.Config.DEFAULT)) {
 *     is AutoSolver.Outcome.Solved -> r.steps     // 完整动作序列，按顺序执行
 *     AutoSolver.Outcome.Unsolvable -> /* 这盘必死：洗牌 */ Unit
 *     is AutoSolver.Outcome.Unknown -> /* 预算烧完还没结论：退化成贪心走一步 */ Unit
 * }
 * ```
 */
object AutoSolver {

    /** 格子坐标（行、列）。 */
    data class Cell(val row: Int, val col: Int)

    /**
     * 一步动作，与 [BrickGameTipHelper.actionStep] 的 6 元数组一一对应：
     * `from` 滑到 `to`，然后 `to` 与 `paired` 一起被消掉。
     *
     * 注意 [to] 可以等于 [from]（[isInPlace]）：两块同款砖**已经同线且中间没有别的砖**时，
     * 算法记的就是"原地消"，落点没有位移 —— 游戏里这种直消不需要滑动，输入层要把 [isInPlace]
     * 翻译成"**点两下**"（先点 [from] 再点 [paired]，第二下用来消"一块砖可配两块"的歧义），
     * 而不是滑动（见 `HintSession.playAction` / `GestureExecutor.tapPair`）。
     */
    data class Step(val from: Cell, val to: Cell, val paired: Cell) {

        val isInPlace: Boolean get() = from == to

        /** 转成 [BrickGameTipHelper.actionStep] 要的 6 元数组。 */
        fun toAction(): IntArray =
            intArrayOf(from.row, from.col, to.row, to.col, paired.row, paired.col)
    }

    /** 求解结果。三种结局都要能区分开，自动运行才知道"是赢了、要洗牌、还是再等等"。 */
    sealed interface Outcome {

        /** 通关：按顺序执行 [steps] 即可清空盘面（已空盘时为 `Solved(emptyList())`）。 */
        data class Solved(val steps: List<Step>) : Outcome

        /** 穷举完整张可达状态图也找不到解 —— 可以确定这盘必须洗牌，不是搜索不给力。 */
        data object Unsolvable : Outcome

        /** 预算烧完还没结论（大盘）。此时**不能**当成无解，只能退化成贪心继续走。 */
        data class Unknown(val nodes: Long) : Outcome
    }

    /**
     * @param nodesPerAttempt 单次尝试的节点预算。烧完就换一个随机序重来。
     *   实战参数：3 万节点 ≈ 桌面 2s / 真机更久，够覆盖绝大多数盘（多数盘 <1000 节点就出解）。
     * @param maxAttempts 重启次数上限。**每次尝试都用全新的置换表**，内存峰值被
     *   `nodesPerAttempt` 卡住（3 万状态的 key 约 10MB），不会随重启次数线性涨。
     */
    data class Config(
        val nodesPerAttempt: Int = 30_000,
        val maxAttempts: Int = 8,
    ) {
        companion object {
            val DEFAULT = Config()

            /** 离线校验/单测用：给足预算，让病态盘也能跑出结论（宁可慢，也要确定）。 */
            val THOROUGH = Config(nodesPerAttempt = 80_000, maxAttempts = 12)
        }
    }

    /**
     * 求解整盘。
     *
     * @param grid `grid[row][col]`，0 = 空位。**不会被修改**（内部全程在副本上跑）。
     * @param shouldAbort 返回 true 表示"没时间了"。每 1024 个节点问一次，
     *   给真机上的挂钟兜底用（节点预算拦不住"单次尝试就是慢"）。
     */
    fun solve(
        grid: Array<IntArray>,
        config: Config = Config.DEFAULT,
        shouldAbort: () -> Boolean = { false },
    ): Outcome {
        if (grid.isEmpty() || grid[0].isEmpty()) return Outcome.Solved(emptyList())
        val rows = grid.size
        val cols = grid[0].size
        require(grid.all { it.size == cols }) { "盘面每行长度必须一致" }

        if (isCleared(grid)) return Outcome.Solved(emptyList())

        var totalNodes = 0L
        for (attempt in 0 until config.maxAttempts) {
            if (shouldAbort()) return Outcome.Unknown(totalNodes)
            val search = Search(rows, cols, Random(SEED + attempt), config.nodesPerAttempt, shouldAbort)
            val solved = search.run(grid)
            totalNodes += search.nodes
            if (solved) return Outcome.Solved(search.solution())
            // 没超标就说明整张可达状态图已经被穷举完 —— 这盘是真的无解，不是没搜到
            if (!search.aborted) return Outcome.Unsolvable
        }
        return Outcome.Unknown(totalNodes)
    }

    /** 只要下一步：自动运行时每帧调一次就够，不用把整条解都算出来。 */
    fun nextStep(
        grid: Array<IntArray>,
        config: Config = Config.DEFAULT,
        shouldAbort: () -> Boolean = { false },
    ): Pair<Outcome, Step?> = when (val outcome = solve(grid, config, shouldAbort)) {
        is Outcome.Solved -> outcome to outcome.steps.firstOrNull()
        else -> outcome to null
    }

    private const val SEED = 0x51ED1L

    internal fun isCleared(grid: Array<IntArray>): Boolean {
        for (row in grid) for (v in row) if (v != BrickGameTipHelper.BACKGROUND) return false
        return true
    }

    /**
     * 一次尝试（一个固定的随机种子）。
     *
     * 递归深度 ≤ 砖数/2（14×10 满盘 = 70），每条递归层用一块**固定复用**的棋盘，
     * 避免每层都新建二维数组 —— 实测这是单节点耗时里占比最大的分配开销。
     */
    private class Search(
        private val rows: Int,
        private val cols: Int,
        private val random: Random,
        private val budget: Int,
        private val shouldAbort: () -> Boolean,
    ) {

        var nodes: Long = 0
            private set

        /** 预算烧完（或被叫停）而提前收工；false 表示"整张可达图已穷举完"。 */
        var aborted: Boolean = false
            private set

        /** 已证"从这里走不出解"的盘面。只在穷举完一个状态时才写入，所以结论永远可靠。 */
        private val dead = HashSet<String>()

        private val maxDepth = rows * cols / 2 + 1
        private val pool = Array(maxDepth + 1) { Array(rows) { IntArray(cols) } }
        private val chosen = arrayOfNulls<IntArray>(maxDepth)
        private var solutionDepth = -1

        /** resolveAction 试落点用的探针盘，全局复用一块。 */
        private val probe = Array(rows) { IntArray(cols) }

        fun run(start: Array<IntArray>): Boolean {
            copyInto(start, pool[0])
            return dfs(0, pool[0])
        }

        fun solution(): List<Step> {
            val out = ArrayList<Step>(solutionDepth.coerceAtLeast(0))
            for (d in 0 until solutionDepth) {
                val a = chosen[d] ?: break
                out += Step(Cell(a[0], a[1]), Cell(a[2], a[3]), Cell(a[4], a[5]))
            }
            return out
        }

        private fun dfs(depth: Int, grid: Array<IntArray>): Boolean {
            if (aborted) return false
            nodes++
            if (nodes > budget) {
                aborted = true
                return false
            }
            if ((nodes and 0x3FFL) == 0L && shouldAbort()) {
                aborted = true
                return false
            }
            if (isCleared(grid)) {
                solutionDepth = depth
                return true
            }

            val moves = moves(grid)
            if (moves.isEmpty()) return false
            moves.shuffle(random)

            for (action in moves) {
                val child = pool[depth + 1]
                copyInto(grid, child)
                BrickGameTipHelper.actionStep(child, action)
                if (dead.contains(keyOf(child))) continue
                chosen[depth] = action
                if (dfs(depth + 1, child)) return true
                if (aborted) return false
            }

            dead.add(keyOf(grid))
            return false
        }

        /** 全部合法动作：`findMoves` 给候选对，两个角落点各拿 `actionStep` 在探针盘上试一次。 */
        private fun moves(grid: Array<IntArray>): MutableList<IntArray> {
            val out = ArrayList<IntArray>(8)
            for (move in BrickGameTipHelper.findMoves(grid)) {
                resolve(grid, move)?.let { out.add(it) }
            }
            return out
        }

        /**
         * 与 [BrickGameTipHelper.resolveAction] 逐条等价，只是把"每个角都新建一块副本"
         * 换成复用同一块探针盘。落点依然是"让 actionStep 自己判"，不做几何硬猜。
         */
        private fun resolve(grid: Array<IntArray>, move: IntArray): IntArray? {
            val fr = move[0]
            val fc = move[1]
            val pr = move[2]
            val pc = move[3]
            if (tryResolve(grid, fr, fc, fr, pc, pr, pc)) return intArrayOf(fr, fc, fr, pc, pr, pc)
            if (tryResolve(grid, fr, fc, pr, fc, pr, pc)) return intArrayOf(fr, fc, pr, fc, pr, pc)
            return null
        }

        private fun tryResolve(
            grid: Array<IntArray>, fromRow: Int, fromCol: Int,
            toRow: Int, toCol: Int, pairRow: Int, pairCol: Int,
        ): Boolean {
            if (toRow == pairRow && toCol == pairCol) return false
            val isFrom = toRow == fromRow && toCol == fromCol
            if (!isFrom && grid[toRow][toCol] != BrickGameTipHelper.BACKGROUND) return false
            if (toRow != fromRow && toCol != fromCol) return false

            copyInto(grid, probe)
            val action = intArrayOf(fromRow, fromCol, toRow, toCol, pairRow, pairCol)
            return when (BrickGameTipHelper.actionStep(probe, action)) {
                // GAME_FAIL 也算"这一步能走"：它说的是这步消完盘面没救了（全局策略问题），
                // 不是这一步不合法 —— 求解器要自己决定要不要走这一步。
                BrickGameTipHelper.ACTION_SUCCESS,
                BrickGameTipHelper.GAME_SUCCESS,
                BrickGameTipHelper.GAME_FAIL,
                -> true

                else -> false
            }
        }

        private fun keyOf(grid: Array<IntArray>): String = buildString(rows * cols) {
            for (r in 0 until rows) {
                val row = grid[r]
                for (c in 0 until cols) append(row[c].toChar())
            }
        }

        private fun copyInto(src: Array<IntArray>, dst: Array<IntArray>) {
            for (r in 0 until rows) src[r].copyInto(dst[r])
        }
    }
}
