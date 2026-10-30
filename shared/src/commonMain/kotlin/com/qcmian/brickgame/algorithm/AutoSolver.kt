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
     * @param nodesPerAttempt 单次尝试的节点预算（Long，因为大盘单任务会超过 Int 上限 21.4 亿）。烧完就换一个随机序重来。
     *   实战参数：3 万节点 ≈ 桌面 2s / 真机更久，够覆盖绝大多数盘（多数盘 <1000 节点就出解）。
     * @param maxAttempts 重启次数上限。**每次尝试都用全新的置换表**，内存峰值被
     *   `nodesPerAttempt` 卡住（3 万状态的 key 约 10MB），不会随重启次数线性涨。
     */
    data class Config(
        val nodesPerAttempt: Long = 30_000,
        val maxAttempts: Int = 8,
        /** 死表槽数；0 = 按节点预算自动。槽数 × 16B = 表内存；允许非 2 的幂（取模哈希）。 */
        val deadSlots: Int = 0,
        /** 死表满时：true 立刻返回 [Outcome.MemoryLimited]；false 清空重搜（省内存、费时间）。 */
        val abortWhenFull: Boolean = false,
        /**
         * 死表准入阈值：证明这个状态已死**至少花了多少节点**，小于它的不写表。
         *
         * 为什么：绝大多数（实测 84%）死状态 64 个节点以内就能重算一遍，存它们只是把
         * 2621 万槽的表塞满、把每条有意义的记录稀释掉。按 64 过滤后条目只剩 15%，
         * 但这些条目覆盖了 93% 的重算节省量。
         */
        val deadMinCost: Int = 0,
        /**
         * 剩余砖数 ≤ 这个值时开始用"终局感知"排序动作；0 = 不启用（只 shuffle）。
         *
         * 为什么需要：常规启发式（能消对数、空格数）看不出来"再走五步会锁死"。
         * 终局视野里再加一项"同行/同列里同类被异类隔开且中间没有空位"的计数——
         * 这种形态推不动也消不掉，实战里就是最后 4 块卡死的那个形状。
         */
        val endgameBricks: Int = 0,
        /**
         * 剩余砖数 ≤ 这个值时，死表结论不再受 [deadMinCost] 过滤，一律记下来。
         *
         * 越到尾部盘面越像，不同路径撞到同一个尾盘的概率极高，所以尾部的"无解"结论
         * 是全表里复用率最高的那批 —— 它们的代价小（几十个节点），但被记住一次的收益远超成本。
         */
        val tailBricks: Int = 0,
    ) {
        companion object {
            val DEFAULT = Config()

            /** 离线校验/单测用：给足预算，让病态盘也能跑出结论（宁可慢，也要确定）。 */
            val THOROUGH = Config(nodesPerAttempt = 80_000L, maxAttempts = 12)
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
        sharedDead: SharedDeadTable? = null,
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
                config.deadSlots, config.abortWhenFull, config.deadMinCost.toLong(),
                config.endgameBricks, config.tailBricks, sharedDead,
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

    /**
     * 多线程共享的死表。
     *
     * 为什么要共享：按第一手把盘面切成几个任务并行搜索时，不同任务的**后继盘面会重合**
     * （消 A 再消 B 与消 B 再消 A 落到同一张盘）。各自持表等于把这份重合重复算一遍 ——
     * 实测切成 4 个任务后总搜索量是单线程的 1.34 倍，共享一张表能把这份重复吃回去。
     *
     * 并发策略：定长、只增不清。装满了就不再记新状态（读仍然有效）——
     * 别的线程正在用这张表，整表清空会让他们读到"全空"，等于集体失忆。
     * 少了剪枝只是慢一点，结论不受影响。
     */
    class SharedDeadTable(val slots: Int) {

        init {
            require(slots in 16..MAX_DEAD_SLOTS) { "共享死表槽数必须在 16..$MAX_DEAD_SLOTS：$slots" }
        }

        private val table = AtomicLongTable(slots)

        /**
         * 已写入的槽数。用来判断"满了没有"。
         *
         * 为什么必须显式记：开放寻址判定"不在表里"只能靠撞到空槽。表满之后一个空槽都没有，
         * 每次未命中都要把整张表扫一遍（实测 2621 万槽时 21 ms/次，吞吐从 5 万/秒掉到 20/秒）。
         * 有了计数就能在满的瞬间停手，不会退化。
         */
        private val count = java.util.concurrent.atomic.AtomicInteger(0)

        /** 当前已占用槽数，给进度/诊断用。 */
        val size: Int get() = count.get()

        fun contains(h: Long): Boolean {
            var i = index(h)
            var probes = 0
            while (probes < slots) {
                val a = table.get(i)
                if (a == 0L) return false
                if (a == h) return true
                i++
                if (i == slots) i = 0
                probes++
            }
            return false
        }

        fun add(h: Long) {
            var i = index(h)
            var probes = 0
            while (probes < slots) {
                val a = table.get(i)
                if (a == h) return
                if (a == 0L && table.compareAndSet(i, 0L, h)) {
                    count.incrementAndGet()
                    return
                }
                i++
                if (i == slots) i = 0
                probes++
            }
        }

        private fun index(h: Long): Int {
            val mixed = (h xor (h ushr 32)).toInt() and Int.MAX_VALUE
            return mixed % slots
        }
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

    /** [Search.dfs] 的哨兵返回值：找到解；被中止（预算/挂钟/内存）。都是负数，与代价区分开。 */
    private const val SOLVED = -1L
    private const val STOP = -2L

    /**
     * 置换表最多用多少槽。槽数按节点预算的 2 倍取 2 的幂，载入到一半就清空。
     *
     * 单槽 16 B（两个 Long 指纹），12 万节点预算约 4 MB，50 万约 16 MB。
     * 清空只损失剪枝（部分死区被重搜），不影响结论正确性 —— 这是内存的硬保险。
     */
    private const val SEGMENTS = 64

    private const val MAX_DEAD_SLOTS = 1 shl 25   // 32M 槽 × 8B = 256MB 上限（64-bit 指纹）



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
        private val budget: Long,
        private val shouldAbort: () -> Boolean,
        deadSlots: Int,
        private val abortWhenFull: Boolean,
        /** 死表准入阈值，见 [Config.deadMinCost]。 */
        private val deadMinCost: Long,
        /** 剩余砖数 ≤ 此值启用终局感知排序，见 [Config.endgameBricks]。 */
        private val endgameBricks: Int,
        /** 剩余砖数 ≤ 此值时死表不受 [deadMinCost] 过滤，见 [Config.tailBricks]。 */
        private val tailBricks: Int,
        /** 多线程共享的死表；给了就不再用自己的定长表，也不再整表清空。 */
        private val sharedDead: SharedDeadTable?,
        /** 每 256 个节点报一次当前尝试已走过的节点数（比 UI 的 1 秒采样更密，慢盘也能看到数在变）。 */
        private val onNodes: (Long) -> Unit,
    ) {

        var nodes: Long = 0
            private set

        /** 起始盘面的砖数；剩余砖数 = totalBricks - 已消对数 × 2。 */
        private var totalBricks = 0

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
         * 已证"从这里走不出解"的盘面，存 64-bit 指纹。
         *
         * 每状态 8 B，比 String key 的 ~340 B 省 40 倍；同样内存装的状态数是 128-bit 的 2 倍
         * （这是"上千万状态"那类盘唯一装得下的办法）。
         * 代价是哈希碰撞：碰撞只多剪枝 → 理论上可能把有解判成无解。碰撞概率 ≈ n²/2⁶⁵，
         * 一千万状态约 3e-6、一亿状态约 3e-4 —— 对"宁可多算一会儿"的定位足够低。
         */
        /**
         * 自己的死表（非共享时用）。初始按 4M 槽起，之后按装载率自动翻倍，
         * 不再有"满了清空"这条路径 —— 清表等于把已经花掉的搜索量扔掉重算。
         */
        private val dead: FingerprintSet? =
            if (sharedDead == null) FingerprintSet(minOf(deadCapacity, 1 shl 22)) else null

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
            var bricks = 0
            for (row in start) for (v in row) if (v != BrickGameTipHelper.BACKGROUND) bricks++
            totalBricks = bricks
            dfs(0, pool[0], fingerprint(pool[0]))
            return !aborted && solutionDepth >= 0
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

        private fun dfs(depth: Int, grid: Array<IntArray>, h: Long): Long {
            if (aborted) return STOP
            nodes++
            if (nodes > budget) {
                aborted = true
                return STOP
            }
            if ((nodes and 0xFFL) == 0L) onNodes(nodes)
            if ((nodes and 0x3FFL) == 0L && shouldAbort()) {
                aborted = true
                return STOP
            }
            if (isCleared(grid)) {
                solutionDepth = depth
                return SOLVED
            }

            val actions = actionBuffers[depth]
            if (moves(grid, depth) == 0) return 1L
            // 只打乱顺序，不排序。
            //
            // 这里原来会按"某类剩两块 > 原地消 > 其余"再稳定排一次，但满盘上所有候选都是
            // 原地消，三档打分恒等于 1，排序是恒等变换。实测（第 18 关 140 块满盘、45 秒）
            // 开关前后：1665 万节点 / 370k 每秒 vs 1636 万节点 / 364k 每秒，差 1.7%。
            // 保留 shuffle 是为了不同重启之间有差异，而不是为了"排序"。
            actions.shuffle(random)
            // 终局视野：剩下的砖不多时，"能不能收尾"由几何形态决定，不由"现在能消几对"决定。
            // 这时换成按死锁形态打分排序（见 [orderEndgame]），把最后 4 块锁死的路提前排到后面。
            val bricks = totalBricks - (depth shl 1)
            if (endgameBricks > 0 && bricks <= endgameBricks) orderEndgame(grid, actions)

            var cost = 1L
            for (i in 0 until actions.size) {
                val action = actions[i]
                val child = pool[depth + 1]
                copyInto(grid, child)
                // 落子与合法性校验合并成一次：findActions 枚举出来的动作绝大多数都合法，
                // 以前先在探针盘上试一遍、再把同一个动作在 child 上重放一次，纯属白算。
                // 真遇到不合法的（理论上的枚举漏洞），这里跳过即可，child 下一轮会被覆盖。
                unpackAction(action, actionScratch)
                if (BrickGameTipHelper.actionStep(child, actionScratch, checkTerminal = false) <= 0) continue
                val childH = fingerprint(child)
                if (isDead(childH)) {
                    cost++
                    continue
                }
                chosen[depth] = action
                val sub = dfs(depth + 1, child, childH)
                if (sub == SOLVED) return SOLVED
                if (sub == STOP) return STOP
                cost += sub
            }

            // 准入阈值：太便宜的死（重算一次也花不了几个节点）不值得占一格 + 一次探测。
            // 实测 84% 的死状态都在 64 个节点以内证完，它们正是把表塞满、把有意义的记录挤掉的那批。
            val bricksLeft = totalBricks - (depth shl 1)
            val tail = tailBricks > 0 && bricksLeft <= tailBricks
            if (cost < deadMinCost && !tail) return cost

            // 当前盘面的指纹在进 dfs 时就算好并传进来了（子节点算指纹时顺便得出），
            // 这里直接入死表，不再重扫 140 格。表会自己扩容，不截断、不清空。
            val own = dead
            if (own != null) own.add(h) else sharedDead?.add(h)
            return cost
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

        /** 查死表：有共享表就用共享的（跨分片去重），否则用自己那张。 */
        private fun isDead(h: Long): Boolean {
            val s = sharedDead
            return if (s != null) s.contains(h) else dead!!.contains(h)
        }

        /** 终局打分的探针盘（单独一块，不能和 pool 撞）。 */
        private val endgameProbe = Array(rows) { IntArray(cols) }
        private val endgameKeys = IntArray(256)
        private val endgameOrder = LongArray(256)
        private val endgameAction = IntArray(6)

        /** 扫描行/列时记录“每类上一次出现的位置”，初值 -1。 */
        private val rowLast = IntArray(64) { -1 }
        private val colLast = IntArray(64) { -1 }

        /**
         * 终局排序：按“走完这一步之后的死锁形态”给动作打分，降序（稳定）。
         *
         * 只在剩余砖数不多时调用，所以每条动作多复制一次盘面的开销可以接受。
         * 非法动作给最低分，dfs 里会再判一次并跳过。
         */
        private fun orderEndgame(grid: Array<IntArray>, actions: ActionBuffer) {
            val n = actions.size
            if (n <= 1 || n > endgameKeys.size) return
            for (i in 0 until n) {
                unpackAction(actions[i], endgameAction)
                copyInto(grid, endgameProbe)
                endgameKeys[i] =
                    if (BrickGameTipHelper.actionStep(endgameProbe, endgameAction, checkTerminal = false) <= 0) {
                        Int.MIN_VALUE
                    } else {
                        endgameScore(endgameProbe)
                    }
                endgameOrder[i] = actions[i]
            }
            for (i in 1 until n) {
                val k = endgameKeys[i]
                val v = endgameOrder[i]
                var j = i - 1
                while (j >= 0 && endgameKeys[j] < k) {
                    endgameKeys[j + 1] = endgameKeys[j]
                    endgameOrder[j + 1] = endgameOrder[j]
                    j--
                }
                endgameKeys[j + 1] = k
                endgameOrder[j + 1] = v
            }
            actions.overwrite(endgameOrder, n)
        }

        /**
         * 终局形态分：越大越好。
         *
         * - adj * 1000：紧邻同类对数 —— 现在就能消的机会；
         * - -lock * 400：同一条线上同类被异类隔开、且中间没有空位的对数。
         *   这种形态既消不掉、又推不动，实战里就是最后 4 块卡死的那个形状；
         * - + empty * 10：空格越多，越有余地调整。
         */
        private fun endgameScore(g: Array<IntArray>): Int {
            var adj = 0
            var empty = 0
            var lock = 0
            for (r in 0 until rows) {
                val row = g[r]
                var lastEmpty = -1
                for (c in 0 until cols) {
                    val v = row[c]
                    if (v == BrickGameTipHelper.BACKGROUND) {
                        empty++
                        lastEmpty = c
                        continue
                    }
                    if (c + 1 < cols && row[c + 1] == v) adj++
                    if (v in 1 until rowLast.size) {
                        val lp = rowLast[v]
                        if (lp >= 0 && lastEmpty <= lp) lock++
                        rowLast[v] = c
                    }
                }
                for (c in 0 until cols) {
                    val v = row[c]
                    if (v in 1 until rowLast.size) rowLast[v] = -1
                }
            }
            for (c in 0 until cols) {
                var lastEmpty = -1
                for (r in 0 until rows) {
                    val v = g[r][c]
                    if (v == BrickGameTipHelper.BACKGROUND) {
                        lastEmpty = r
                        continue
                    }
                    if (r + 1 < rows && g[r + 1][c] == v) adj++
                    if (v in 1 until colLast.size) {
                        val lp = colLast[v]
                        if (lp >= 0 && lastEmpty <= lp) lock++
                        colLast[v] = r
                    }
                }
                for (r in 0 until rows) {
                    val v = g[r][c]
                    if (v in 1 until colLast.size) colLast[v] = -1
                }
            }
            return adj * 1000 - lock * 400 + empty * 10
        }

        /**
         * 盘面的 64-bit 指纹。
         *
         * 直接扫原盘面，不生成中间数组 —— 这是它比“打包 key”更省时间的地方。
         * 求解只关心“同款同号”，与类别 id 的具体值无关，所以直接拿原值参与混合即可。
         *
         * 每格混一次 FNV 乘法，末尾再做一次 64-bit 雪崩（xor-shift + 乘法），
         * 让低位也参与索引计算 —— 索引只取低位，不雪崩的话相邻盘面会撞在同一段探测链上。
         */
        private fun fingerprint(grid: Array<IntArray>): Long {
            var h = -0x340d631b7bdddcdbL
            for (r in 0 until rows) {
                val row = grid[r]
                for (c in 0 until cols) {
                    h = (h xor row[c].toLong()) * 0x100000001B3L
                }
            }
            h = h xor (h ushr 33)
            h *= -0x40A7B892E31B1A47L
            h = h xor (h ushr 29)
            // 全零留作“空槽”标记；极小概率撞上时挪一下，不参与任何判断
            return if (h == 0L) 1L else h
        }

        /**
         * 分段可扩容的指纹表 —— 不截断、不清空、不丢条目。
         *
         * 为什么分段：单张表扩容要整表翻倍，大表那一刻新旧同时存在（25 GB → 75 GB 峰值），
         * 而且 LongArray 长度上限 2^31-1，超过直接分配不出来。切成 64 段后每段独立扩容，
         * 峰值只有"一段"的 1.5 倍。
         *
         * 段内线性探测，装载 0.9 翻倍（留 10% 空槽，否则查不到时要扫完整个段）；8 B/条。
         */
        private class FingerprintSet(initialCapacity: Int) {

            private val segments = Array(SEGMENTS) { Segment() }

            var size = 0
                private set

            fun contains(h: Long): Boolean = segOf(h).contains(h)

            fun add(h: Long) {
                if (segOf(h).add(h)) size++
            }

            private fun segOf(h: Long): Segment =
                segments[((h ushr 58) and (SEGMENTS - 1).toLong()).toInt()]

            private class Segment {
                private var slots = LongArray(1 shl 14)
                private var mask = slots.size - 1
                private var limit = slots.size * 9 / 10
                private var count = 0

                fun contains(h: Long): Boolean {
                    var i = index(h)
                    while (true) {
                        val a = slots[i]
                        if (a == 0L) return false
                        if (a == h) return true
                        i = (i + 1) and mask
                    }
                }

                /** 返回 true 表示确实是新条目。 */
                fun add(h: Long): Boolean {
                    if (count >= limit) grow()
                    var i = index(h)
                    while (true) {
                        val a = slots[i]
                        if (a == h) return false
                        if (a == 0L) {
                            slots[i] = h
                            count++
                            return true
                        }
                        i = (i + 1) and mask
                    }
                }

                private fun grow() {
                    val old = slots
                    val next = LongArray(old.size shl 1)
                    slots = next
                    mask = next.size - 1
                    limit = next.size * 9 / 10
                    for (v in old) {
                        if (v == 0L) continue
                        var i = index(v)
                        while (next[i] != 0L) i = (i + 1) and mask
                        next[i] = v
                    }
                }

                private fun index(h: Long): Int {
                    val mixed = (h xor (h ushr 32)).toInt()
                    return mixed and mask
                }
            }
        }

        private fun copyInto(src: Array<IntArray>, dst: Array<IntArray>) {
            for (r in 0 until rows) src[r].copyInto(dst[r])
        }
    }
}
