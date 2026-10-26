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
                config.deadSlots, config.abortWhenFull, sharedDead,
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
                if (a == 0L && table.compareAndSet(i, 0L, h)) return
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

    /**
     * 置换表最多用多少槽。槽数按节点预算的 2 倍取 2 的幂，载入到一半就清空。
     *
     * 单槽 16 B（两个 Long 指纹），12 万节点预算约 4 MB，50 万约 16 MB。
     * 清空只损失剪枝（部分死区被重搜），不影响结论正确性 —— 这是内存的硬保险。
     */
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
        private val budget: Int,
        private val shouldAbort: () -> Boolean,
        deadSlots: Int,
        private val abortWhenFull: Boolean,
        /** 多线程共享的死表；给了就不再用自己的定长表，也不再整表清空。 */
        private val sharedDead: SharedDeadTable?,
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
         * 已证"从这里走不出解"的盘面，存 64-bit 指纹。
         *
         * 每状态 8 B，比 String key 的 ~340 B 省 40 倍；同样内存装的状态数是 128-bit 的 2 倍
         * （这是"上千万状态"那类盘唯一装得下的办法）。
         * 代价是哈希碰撞：碰撞只多剪枝 → 理论上可能把有解判成无解。碰撞概率 ≈ n²/2⁶⁵，
         * 一千万状态约 3e-6、一亿状态约 3e-4 —— 对"宁可多算一会儿"的定位足够低。
         */
        private val dead: FingerprintSet? = if (sharedDead == null) FingerprintSet(deadCapacity) else null

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

        /**
         * 每类还剩几块（当前路径上），走法排序用。
         *
         * 一步固定消两块（[BrickGameTipHelper.actionStep]），所以只在这一条路径上前进/回退 ±2，
         * 不用每个节点重扫 140 格。下标越界的类别（识别层理论上给不出）不计数，退化成"不优先"。
         */
        private val remaining = IntArray(4096)

        /** [orderActions] 复用的键缓冲，避免每个节点新建数组。 */
        private var orderKeys = IntArray(64)

        fun run(start: Array<IntArray>): Boolean {
            copyInto(start, pool[0])
            for (row in start) for (v in row) if (v in 1 until remaining.size) remaining[v]++
            return dfs(0, pool[0], fingerprint(pool[0]))
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

        private fun dfs(depth: Int, grid: Array<IntArray>, h: Long): Boolean {
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
            // 先随机（保证不同随机序的尝试仍然有差异），再稳定地按"有用程度"排在前面：
            // 只剩两块的那一类 → 原地消 → 其余。只改顺序、不改动作集合，正确性不受影响。
            actions.shuffle(random)
            orderActions(grid, actions)

            for (i in 0 until actions.size) {
                val action = actions[i]
                val child = pool[depth + 1]
                copyInto(grid, child)
                // 落子与合法性校验合并成一次：findActions 枚举出来的动作绝大多数都合法，
                // 以前先在探针盘上试一遍、再把同一个动作在 child 上重放一次，纯属白算。
                // 真遇到不合法的（理论上的枚举漏洞），这里跳过即可，child 下一轮会被覆盖。
                unpackAction(action, actionScratch)
                val cls = grid[actionScratch[0]][actionScratch[1]]
                if (BrickGameTipHelper.actionStep(child, actionScratch, checkTerminal = false) <= 0) continue
                if (cls in 1 until remaining.size) remaining[cls] -= 2
                val childH = fingerprint(child)
                if (isDead(childH)) {
                    if (cls in 1 until remaining.size) remaining[cls] += 2
                    continue
                }
                chosen[depth] = action
                val solved = dfs(depth + 1, child, childH)
                if (cls in 1 until remaining.size) remaining[cls] += 2
                if (solved) return true
                if (aborted) return false
            }

            // 死表满：
            //  - abortWhenFull=true：内存到顶，停止并让上层报 MemoryLimited，绝不 OOM；
            //  - false：清空重搜（省内存，但会重复探索，时间不可控）。
            val own = dead
            if (own != null && own.size >= own.limit) {
                if (abortWhenFull) {
                    memoryFull = true
                    aborted = true
                    return false
                }
                own.clear()
            }
            // 当前盘面的指纹在进 dfs 时就算好并传进来了（子节点算指纹时顺便得出），
            // 这里直接入死表，不再重扫 140 格。
            if (own != null) own.add(h) else sharedDead?.add(h)
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

        /** 查死表：有共享表就用共享的（跨分片去重），否则用自己那张。 */
        private fun isDead(h: Long): Boolean {
            val s = sharedDead
            return if (s != null) s.contains(h) else dead!!.contains(h)
        }

        /**
         * 按"有用程度"稳定排序（0 最优先）：
         *  0. 这一步消掉的类**只剩这两块** —— 不消掉它，这两块就是全盘唯一的出口；
         *  1. 原地消（不用先滑），少一次位移就少一处被卡住的风险；
         *  2. 其余。
         *
         * 只影响搜索序，不影响可解性判断：全部动作仍然会被枚举到。
         */
        private fun orderActions(grid: Array<IntArray>, actions: ActionBuffer) {
            val n = actions.size
            if (n <= 1) return
            if (orderKeys.size < n) orderKeys = IntArray(n)
            for (i in 0 until n) {
                unpackAction(actions[i], actionScratch)
                val cls = grid[actionScratch[0]][actionScratch[1]]
                orderKeys[i] = when {
                    cls in 1 until remaining.size && remaining[cls] <= 2 -> 0
                    actionScratch[0] == actionScratch[2] && actionScratch[1] == actionScratch[3] -> 1
                    else -> 2
                }
            }
            actions.stableSortByKey(orderKeys, 3)
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
         * 定长开放寻址的指纹表。
         *
         * 不用 `HashSet<Long>`：HashMap.Node 每条约 32 B，加上装箱对象就省不下来了。
         * 这里一条 Long 8 B；载入到 [limit]（容量一半）由调用方 clear()。
         */
        private class FingerprintSet(val capacity: Int) {
            private val slots = LongArray(capacity)

            /** 存到这个条数就提示调用方清空（载入因子 0.5，保证探测链短）。 */
            val limit = capacity ushr 1

            var size = 0
                private set

            fun contains(h: Long): Boolean {
                var i = index(h)
                var probes = 0
                while (probes < capacity) {
                    val a = slots[i]
                    if (a == 0L) return false
                    if (a == h) return true
                    i++
                    if (i == capacity) i = 0
                    probes++
                }
                return false
            }

            fun add(h: Long) {
                var i = index(h)
                var probes = 0
                while (probes < capacity) {
                    val a = slots[i]
                    if (a == 0L) {
                        slots[i] = h
                        size++
                        return
                    }
                    if (a == h) return
                    i++
                    if (i == capacity) i = 0
                    probes++
                }
            }

            fun clear() {
                slots.fill(0L)
                size = 0
            }

            private fun index(h: Long): Int {
                val mixed = (h xor (h ushr 32)).toInt() and Int.MAX_VALUE
                return mixed % capacity
            }
        }

        private fun copyInto(src: Array<IntArray>, dst: Array<IntArray>) {
            for (r in 0 until rows) src[r].copyInto(dst[r])
        }
    }
}
