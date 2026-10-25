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
            val search = Search(
                rows, cols, Random(SEED + attempt), config.nodesPerAttempt, shouldAbort,
            )
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

    /**
     * 置换表最多用多少槽。槽数按节点预算的 2 倍取 2 的幂，载入到一半就清空。
     *
     * 单槽 16 B（两个 Long 指纹），12 万节点预算约 4 MB，50 万约 16 MB。
     * 清空只损失剪枝（部分死区被重搜），不影响结论正确性 —— 这是内存的硬保险。
     */
    private const val MAX_DEAD_SLOTS = 1 shl 20

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

        /** 槽数按节点预算的 2 倍取 2 的幂（载入因子 0.5），并受 [MAX_DEAD_SLOTS] 限制。 */
        private val deadCapacity: Int = run {
            var c = 16
            while (c < budget * 2 && c < MAX_DEAD_SLOTS) c = c shl 1
            c
        }

        /**
         * 已证"从这里走不出解"的盘面，存 128-bit 指纹。
         *
         * 每状态 16 B（两个 Long），比 String key 的 ~340 B 省 20 倍。
         * 代价是理论上存在哈希碰撞：碰撞会多剪枝 → 可能把有解判成无解。
         * 128-bit 在百万级状态下的碰撞概率约 1e-28，工程上视为零（见 [FingerprintSet]）。
         */
        private val dead = FingerprintSet(deadCapacity)

        private val maxDepth = rows * cols / 2 + 1
        private val pool = Array(maxDepth + 1) { Array(rows) { IntArray(cols) } }
        private val chosen = arrayOfNulls<IntArray>(maxDepth)
        private var solutionDepth = -1

        /** 校验候选动作用的探针盘，全局复用一块（[moves] 里每个动作都要 actionStep 试一次）。 */
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
                // 只借 actionStep 落子；终局由下一层 moves() 判空，这里不查
                BrickGameTipHelper.actionStep(child, action, checkTerminal = false)
                fingerprint(child, fp)
                if (dead.contains(fp[0], fp[1])) continue
                chosen[depth] = action
                if (dfs(depth + 1, child)) return true
                if (aborted) return false
            }

            // 置换表只用于剪枝：到顶清空只会让部分死区被重搜，不会把“有解”误判成“无解”。
            if (dead.size >= dead.limit) dead.clear()
            fingerprint(grid, fp)
            dead.add(fp[0], fp[1])
            return false
        }

        /**
         * 全部合法动作：**直接用 [BrickGameTipHelper.findActions] 的 6 元动作**，
         * 再把每个动作拿 [BrickGameTipHelper.actionStep] 在探针盘上试一遍。
         *
         * 为什么要改掉"自己推落点"：这里原来只对 `findMoves` 的配对重推矩形两个角，
         * 还额外要求**落点当前是空位**。但拖动时整条砖会被顶着走，落点完全可以是别的砖
         * 此刻占着的格子 —— 这种走法全被判非法，于是可解盘被穷举成"确定无解"
         * （4x3 的最小反例见 AutoSolverAgreementTest）。提示层一直走 findActions，
         * 所以现场表现就是"提示能消、自动却报无解"。现在两边共用同一份动作。
         */
        private fun moves(grid: Array<IntArray>): MutableList<IntArray> {
            val out = ArrayList<IntArray>(8)
            for (action in BrickGameTipHelper.findActions(grid)) {
                copyInto(grid, probe)
                // checkTerminal=false：终局交给下一层 moves() 判空。GAME_FAIL 本来也被
                // 当成"能走"，这里再全盘 findActions 一次纯属浪费 —— 原来每个候选动作
                // 付一次全盘扫描，是求解器 1.27ms/节点的主要来源。
                if (BrickGameTipHelper.actionStep(probe, action, checkTerminal = false) > 0) {
                    out.add(action)
                }
            }
            return out
        }

        /** 复用的指纹输出，避免每节点分配。 */
        private val fp = LongArray(2)

        /**
         * 盘面的 128-bit 指纹：两个独立混合的 64-bit 哈希。
         *
         * 直接扫原盘面，不生成中间数组 —— 这是它比“打包 key”更省时间的地方。
         * 求解只关心“同款同号”，与类别 id 的具体值无关，所以直接拿原值参与混合即可。
         */
        private fun fingerprint(grid: Array<IntArray>, out: LongArray) {
            var h1 = 0x811C9DC5L
            var h2 = 0x27D4EB2FL
            for (r in 0 until rows) {
                val row = grid[r]
                for (c in 0 until cols) {
                    val v = row[c].toLong()
                    h1 = (h1 xor v) * 0x100000001B3L
                    h2 = h2 + v + 0x9E3779B9L
                    h2 = h2 xor (h2 ushr 29)
                    h2 *= -0x40A7B892E31B1A47L
                }
            }
            // 全零留作“空槽”标记；极小概率撞上时挪一下，不参与任何判断
            if (h1 == 0L && h2 == 0L) h1 = 1L
            out[0] = h1
            out[1] = h2
        }

        /**
         * 定长开放寻址的指纹表。
         *
         * 不用 `HashSet<Long>`：HashMap.Node 每条约 32 B，加上装箱对象就省不下来了。
         * 这里两个 Long 连续存，单条 16 B；载入到 [limit]（容量一半）由调用方 clear()。
         */
        private class FingerprintSet(val capacity: Int) {
            private val mask = capacity - 1
            private val slots = LongArray(capacity * 2)

            /** 存到这个条数就提示调用方清空（载入因子 0.5，保证探测链短）。 */
            val limit = capacity ushr 1

            var size = 0
                private set

            fun contains(h1: Long, h2: Long): Boolean {
                if (h1 == 0L && h2 == 0L) return false
                var i = index(h1)
                var probes = 0
                while (probes < capacity) {
                    val a = slots[i * 2]
                    val b = slots[i * 2 + 1]
                    if (a == 0L && b == 0L) return false
                    if (a == h1 && b == h2) return true
                    i = (i + 1) and mask
                    probes++
                }
                return false
            }

            fun add(h1: Long, h2: Long) {
                if (h1 == 0L && h2 == 0L) return
                var i = index(h1)
                var probes = 0
                while (probes < capacity) {
                    val a = slots[i * 2]
                    val b = slots[i * 2 + 1]
                    if (a == 0L && b == 0L) {
                        slots[i * 2] = h1
                        slots[i * 2 + 1] = h2
                        size++
                        return
                    }
                    if (a == h1 && b == h2) return
                    i = (i + 1) and mask
                    probes++
                }
            }

            fun clear() {
                slots.fill(0L)
                size = 0
            }

            private fun index(h: Long): Int = (h xor (h ushr 32)).toInt() and mask
        }

        private fun copyInto(src: Array<IntArray>, dst: Array<IntArray>) {
            for (r in 0 until rows) src[r].copyInto(dst[r])
        }
    }
}
