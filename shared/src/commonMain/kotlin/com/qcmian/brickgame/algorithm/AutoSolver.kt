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
 *     is AutoSolver.Outcome.Inconsistent -> /* 某类砖块奇数个：识别/框重复，先刷新 */ Unit
 *     AutoSolver.Outcome.Unsolvable -> /* 这盘必死：洗牌 */ Unit
 *     is AutoSolver.Outcome.MemoryLimited -> /* 死表到顶：加大内存或洗牌 */ Unit
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

    /** 求解结果。五种结局都要能区分开，自动运行才知道"是赢了、要洗牌、内存不够、还是再等等"。 */
    sealed interface Outcome {

        /** 通关：按顺序执行 [steps] 即可清空盘面（已空盘时为 `Solved(emptyList())`）。 */
        data class Solved(val steps: List<Step>) : Outcome

        /**
         * 盘面本身不可能被清空：某类方块数量是奇数。
         *
         * 游戏每一类砖的数量必然是偶数（成对）；出现奇数只可能是识别错了或框重复了。
         * 这不是"搜索没找到"，再怎么加预算也不会有解 —— 直接让上层提示刷新，
         * 别在错板上烧 120 秒。
         */
        data class Inconsistent(val oddClasses: List<Int>) : Outcome

        /** 穷举完整张可达状态图也找不到解 —— 可以确定这盘必须洗牌，不是搜索不给力。 */
        data object Unsolvable : Outcome

        /** 死表内存到顶（[Config.abortWhenFull]）：已无法继续保证穷尽。 */
        data class MemoryLimited(val nodes: Long) : Outcome

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
        /** 死表槽数；0 = 按节点预算自动。槽数 × 16B = 表内存；允许非 2 的幂（取模哈希）。 */
        val deadSlots: Int = 0,
        /** 死表满时：true 立刻返回 [Outcome.MemoryLimited]；false 清空重搜（省内存、费时间）。 */
        val abortWhenFull: Boolean = false,
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
     * @param onProgress 每探索 256 个节点（以及每次尝试结束时）回调一次**累计**节点数
     *   （含前面几次尝试），给控制条刷"已尝试多少步"用。回调在**求解线程**上执行，
     *   不要在里面做重活。
     */
    fun solve(
        grid: Array<IntArray>,
        config: Config = Config.DEFAULT,
        shouldAbort: () -> Boolean = { false },
        onProgress: (Long) -> Unit = {},
    ): Outcome {
        if (grid.isEmpty() || grid[0].isEmpty()) return Outcome.Solved(emptyList())
        val rows = grid.size
        val cols = grid[0].size
        require(grid.all { it.size == cols }) { "盘面每行长度必须一致" }

        if (isCleared(grid)) return Outcome.Solved(emptyList())

        // 硬不变量：每类砖必须成对。奇数类在数学上不可能被清空，先于搜索直接判掉。
        val counts = HashMap<Int, Int>()
        for (row in grid) for (v in row) {
            if (v != BrickGameTipHelper.BACKGROUND) counts[v] = (counts[v] ?: 0) + 1
        }
        val odd = counts.filterValues { it % 2 != 0 }.keys.sorted()
        if (odd.isNotEmpty()) return Outcome.Inconsistent(odd)

        var totalNodes = 0L
        for (attempt in 0 until config.maxAttempts) {
            if (shouldAbort()) return Outcome.Unknown(totalNodes)
            // 重启会让单次搜索的 nodes 从 0 再来，进度要报"这盘一共试了多少"，所以加上基数
            val baseNodes = totalNodes
            val search = Search(
                rows, cols, Random(SEED + attempt), config.nodesPerAttempt, shouldAbort,
                config.deadSlots, config.abortWhenFull,
                onNodes = { nodes -> onProgress(baseNodes + nodes) },
            )
            val solved = search.run(grid)
            totalNodes += search.nodes
            // 每次尝试结束也报一次：小预算重启时可能还不到 256 个节点，
            // 只靠节点边界的话进度会一直停在 0
            onProgress(totalNodes)
            if (search.memoryFull) return Outcome.MemoryLimited(totalNodes)
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
    private const val MAX_DEAD_SLOTS = 1 shl 24   // 16M 槽 × 16B = 256MB 上限

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
        deadSlots: Int,
        private val abortWhenFull: Boolean,
        /** 每 256 个节点报一次当前尝试已走过的节点数（比 UI 的 1 秒采样更密，慢盘也能看到数在变）。 */
        private val onNodes: (Long) -> Unit,
    ) {

        var nodes: Long = 0
            private set

        /** 预算烧完（或被叫停）而提前收工；false 表示"整张可达图已穷举完"。 */
        var aborted: Boolean = false
            private set

        /** 死表满且 [abortWhenFull]：内存到顶，无法继续保证穷尽。 */
        var memoryFull: Boolean = false
            private set

        /** 死表槽数：显式配置优先（允许非 2 的幂，取模探测）；否则按节点预算取 2 的幂。 */
        private val deadCapacity: Int = if (deadSlots > 0) {
            require(deadSlots in 16..MAX_DEAD_SLOTS) {
                "deadSlots 必须在 16..$MAX_DEAD_SLOTS：$deadSlots"
            }
            deadSlots
        } else {
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

        /** 已选动作按 Long 打包保存（每个 60 bit），不再持有 IntArray。 */
        private val chosen = LongArray(maxDepth)
        private var solutionDepth = -1

        /** 动作枚举的最近邻缓存，整个搜索复用，避免每节点分配四张索引表。 */
        private val near = BrickGameTipHelper.Nearest(rows, cols)

        /**
         * 每层一个动作缓冲：父节点正在遍历自己的动作时，子节点会枚举它自己的动作，
         * 不能共用同一块缓冲（否则父节点的列表会被子节点清掉）。
         */
        private val actionBuffers = Array(maxDepth + 1) { ActionBuffer() }
        private val actionScratch = IntArray(6)

        fun run(start: Array<IntArray>): Boolean {
            copyInto(start, pool[0])
            fingerprint(pool[0], fp)
            return dfs(0, pool[0], fp[0], fp[1])
        }

        fun solution(): List<Step> {
            val out = ArrayList<Step>(solutionDepth.coerceAtLeast(0))
            for (d in 0 until solutionDepth) {
                unpackAction(chosen[d], actionScratch)
                out += Step(
                    Cell(actionScratch[0], actionScratch[1]),
                    Cell(actionScratch[2], actionScratch[3]),
                    Cell(actionScratch[4], actionScratch[5]),
                )
            }
            return out
        }

        private fun dfs(depth: Int, grid: Array<IntArray>, h1: Long, h2: Long): Boolean {
            if (aborted) return false
            nodes++
            if (nodes > budget) {
                aborted = true
                return false
            }
            if ((nodes and 0xFFL) == 0L) onNodes(nodes)
            if ((nodes and 0x3FFL) == 0L && shouldAbort()) {
                aborted = true
                return false
            }
            if (isCleared(grid)) {
                solutionDepth = depth
                return true
            }

            val actions = actionBuffers[depth]
            if (moves(grid, depth) == 0) return false
            actions.shuffle(random)

            for (i in 0 until actions.size) {
                val action = actions[i]
                val child = pool[depth + 1]
                copyInto(grid, child)
                // 落子与合法性校验合并成一次：findActions 枚举出来的动作绝大多数都合法，
                // 以前先在探针盘上试一遍、再把同一个动作在 child 上重放一次，纯属白算。
                // 真遇到不合法的（理论上的枚举漏洞），这里跳过即可，child 下一轮会被覆盖。
                unpackAction(action, actionScratch)
                if (BrickGameTipHelper.actionStep(child, actionScratch, checkTerminal = false) <= 0) continue
                fingerprint(child, fp)
                val childH1 = fp[0]
                val childH2 = fp[1]
                if (dead.contains(childH1, childH2)) continue
                chosen[depth] = action
                if (dfs(depth + 1, child, childH1, childH2)) return true
                if (aborted) return false
            }

            // 死表满：
            //  - abortWhenFull=true：内存到顶，停止并让上层报 MemoryLimited，绝不 OOM；
            //  - false：清空重搜（省内存，但会重复探索，时间不可控）。
            if (dead.size >= dead.limit) {
                if (abortWhenFull) {
                    memoryFull = true
                    aborted = true
                    return false
                }
                dead.clear()
            }
            // 当前盘面的指纹在进 dfs 时就算好并传进来了（子节点算指纹时顺便得出），
            // 这里直接入死表，不再重扫 140 格。
            dead.add(h1, h2)
            return false
        }

        /**
         * 全部合法动作：**直接用 [BrickGameTipHelper.findActionsInto] 的打包动作**。
         *
         * 以前这里还会对每个候选动作 copy 整盘到探针盘、再用 [BrickGameTipHelper.actionStep]
         * 试一遍，确认合法后才进 [dfs]；而 [dfs] 落子时又 copy + actionStep 一次。
         * 实测 findActions 的候选在 10 万张随机盘面上都没有非法动作（约 120 万个候选），
         * 所以把校验挪到 [dfs] 落子那一次：合法就继续，非法就跳过。
         *
         * 为什么要改掉"自己推落点"：这里原来只对 `findMoves` 的配对重推矩形两个角，
         * 还额外要求**落点当前是空位**。但拖动时整条砖会被顶着走，落点完全可以是别的砖
         * 此刻占着的格子 —— 这种走法全被判非法，于是可解盘被穷举成"确定无解"
         * （4x3 的最小反例见 AutoSolverAgreementTest）。提示层一直走 findActions，
         * 所以现场表现就是"提示能消、自动却报无解"。现在两边共用同一份动作。
         */
        /** 动作写进本层复用的缓冲，返回本节点可选动作数。 */
        private fun moves(grid: Array<IntArray>, depth: Int): Int {
            val buffer = actionBuffers[depth]
            BrickGameTipHelper.findActionsInto(grid, near, buffer)
            return buffer.size
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
                    i = (i + 1) % capacity
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
                    i = (i + 1) % capacity
                    probes++
                }
            }

            fun clear() {
                slots.fill(0L)
                size = 0
            }

            private fun index(h: Long): Int =
                ((h xor (h ushr 32)).toInt() and Int.MAX_VALUE) % capacity
        }

        private fun copyInto(src: Array<IntArray>, dst: Array<IntArray>) {
            for (r in 0 until rows) src[r].copyInto(dst[r])
        }
    }
}
