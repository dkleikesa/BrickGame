package com.qcmian.brickgame.algorithm

import kotlin.random.Random

/**
 * 一组可消除的同款砖。
 *
 * 字段就是 [BrickGameTipHelper.findMoves] 返回的 `[fromRow, fromCol, pairRow, pairCol]`：
 * 前两个是待滑动的砖，后两个是它的配对砖。真正能执行的 6 元动作（含落点）由
 * [BrickGameTipHelper.resolveAction] 补全。
 */
data class BrickMove(
    val fromRow: Int,
    val fromCol: Int,
    val pairRow: Int,
    val pairCol: Int,
)

/** 一个动作的 6 个坐标各占 10 bit（0..1023），合计 60 bit，能塞进一个 Long。 */
private const val ACTION_COORD_BITS = 10
private const val ACTION_COORD_MASK = 0x3FFL

/** 把 [fromRow, fromCol, toRow, toCol, pairRow, pairCol] 打包成一个 Long，不产生数组。 */
internal fun packAction(fr: Int, fc: Int, tr: Int, tc: Int, pr: Int, pc: Int): Long {
    var v = 0L
    v = v or (fr.toLong() and ACTION_COORD_MASK)
    v = v or ((fc.toLong() and ACTION_COORD_MASK) shl ACTION_COORD_BITS)
    v = v or ((tr.toLong() and ACTION_COORD_MASK) shl (ACTION_COORD_BITS * 2))
    v = v or ((tc.toLong() and ACTION_COORD_MASK) shl (ACTION_COORD_BITS * 3))
    v = v or ((pr.toLong() and ACTION_COORD_MASK) shl (ACTION_COORD_BITS * 4))
    v = v or ((pc.toLong() and ACTION_COORD_MASK) shl (ACTION_COORD_BITS * 5))
    return v
}

/** 解包到调用方给的 6 元 IntArray；求解器复用同一块 scratch，避免每动作分配。 */
internal fun unpackAction(packed: Long, out: IntArray) {
    out[0] = (packed and ACTION_COORD_MASK).toInt()
    out[1] = ((packed ushr ACTION_COORD_BITS) and ACTION_COORD_MASK).toInt()
    out[2] = ((packed ushr (ACTION_COORD_BITS * 2)) and ACTION_COORD_MASK).toInt()
    out[3] = ((packed ushr (ACTION_COORD_BITS * 3)) and ACTION_COORD_MASK).toInt()
    out[4] = ((packed ushr (ACTION_COORD_BITS * 4)) and ACTION_COORD_MASK).toInt()
    out[5] = ((packed ushr (ACTION_COORD_BITS * 5)) and ACTION_COORD_MASK).toInt()
}

/**
 * 求解器专用的动作缓冲：节点之间复用，不分配 `IntArray` / `ArrayList`。
 *
 * 动作用 [packAction] 压成 Long；典型容量 32，超过才扩容。
 */
internal class ActionBuffer(initialCapacity: Int = 32) {
    private var data = LongArray(initialCapacity)
    var size = 0
        private set

    operator fun get(index: Int): Long = data[index]

    fun clear() {
        size = 0
    }

    fun add(fr: Int, fc: Int, tr: Int, tc: Int, pr: Int, pc: Int) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = packAction(fr, fc, tr, tc, pr, pc)
    }

    /**
     * 按 [keys]（0..[buckets]-1）稳定重排，用于求解器的走法优先序。
     *
     * 稳定：同键的动作保持进入时的相对顺序，所以调用方可以"先 shuffle 再排序"来兼得
     * 优先序和随机性。计数排序，O(size)。
     */
    fun stableSortByKey(keys: IntArray, buckets: Int) {
        if (size <= 1) return
        if (scratch.size < data.size) scratch = LongArray(data.size)
        val starts = IntArray(buckets)
        for (i in 0 until size) starts[keys[i]]++
        var acc = 0
        for (b in 0 until buckets) {
            val count = starts[b]
            starts[b] = acc
            acc += count
        }
        for (i in 0 until size) scratch[starts[keys[i]]++] = data[i]
        for (i in 0 until size) data[i] = scratch[i]
    }

    /** [stableSortByKey] 的临时缓冲（与 data 同长，按需扩容）。 */
    private var scratch = LongArray(0)

    /** 原地 Fisher-Yates，随机数消耗顺序与 `Collections.shuffle` 一致，搜索序不变。 */
    fun shuffle(random: Random) {
        for (i in size downTo 2) {
            val j = random.nextInt(i)
            val tmp = data[i - 1]
            data[i - 1] = data[j]
            data[j] = tmp
        }
    }
}

/**
 * 方块消除游戏的提示与执行逻辑。
 *
 * 网格 grid[row][col]：BACKGROUND(0) 表示空位，其他值为方块类型。
 *
 * move（findMoves 返回）：[fromRow, fromCol, pairRow, pairCol] —— 前两个是待滑动的砖，
 * 后两个是配对砖（识别层拿到的整盘二维数组直接喂进 [findMoves] 即可）。
 * action（actionStep 入参）：[fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]
 */
object BrickGameTipHelper {

    @JvmField
    var BACKGROUND = 0

    /**
     * 是否打印每一步的消除日志（默认关）。
     *
     * [AutoSolver] 一次求解要跑几万到几十万个节点，每个节点都会试落点、每次成功试落点都会走到
     * [logRemove] —— 一直打就是几十万行 stdout，真机上既刷屏又把求解拖慢一个量级。
     * 移植自原 Java 版的那行格式没动，只是加了开关（要对照原实现的输出时把它打开）。
     */
    @JvmField
    var logRemovals = false

    /** 游戏胜利：网格已清空 */
    const val GAME_SUCCESS = 2

    /** 操作成功 */
    const val ACTION_SUCCESS = 1

    /** 操作无效 */
    const val ACTION_FAIL = -1

    /** 游戏失败：已无可行操作 */
    const val GAME_FAIL = -2

    /** from / to / paired 的位置关系不合法 */
    const val FROM_PAIR_SAME_BRICK = -3

    /** from 到 to 方向上没有空位可移动 */
    const val FROM_TO_SAME_LINE = -4

    /** 空位数量不足以移动到 to */
    const val FROM_TO_SAME_LINE_CAN_MOVE = -5

    /** to 与 paired 之间有阻挡 */
    const val TO_PAIR_SAME_LINE = -6

    // ------------------------------------------------------------------
    // 提示：枚举所有可行操作
    // ------------------------------------------------------------------

    /**
     * 全部可执行动作：[fromRow, fromCol, toRow, toCol, pairRow, pairCol]。
     *
     * 关键点是**带着目标位置**。早先只给 [from, pair]，执行层只能拿"矩形的另一个角"去猜落点，
     * 猜出来跟枚举时的实际目标不一致，于是出现"提示画了框、自动却判无解"。
     *
     * 注意 `to` **不要求当前是空位**：拖动时整条砖会被顶着走，`to` 可以是同一条线上
     * 另一块砖此刻占着的格子（那块砖被顶开后落位）。所以"落点必须是空的"这种推测是错的，
     * 能不能走一律以 [actionStep] 的判定为准。
     */
    fun findActions(grid: Array<IntArray>): List<IntArray> {
        val near = Nearest(grid.size, grid[0].size)
        val buffer = ActionBuffer()
        findActionsInto(grid, near, buffer)
        val out = ArrayList<IntArray>(buffer.size)
        val action = IntArray(6)
        for (i in 0 until buffer.size) {
            unpackAction(buffer[i], action)
            out.add(action.copyOf())
        }
        return out
    }

    /**
     * 求解器专用入口：最近邻索引表 + 动作缓冲都由调用方复用，热点路径零分配。
     *
     * 旧实现每个节点要扫 450 多次"到最近非空"；现在先建表，再按"空格 + 左右/上下
     * 最近砖"反查生成动作，动作直接压成 Long 写进 [out]。
     */
    internal fun findActionsInto(grid: Array<IntArray>, near: Nearest, out: ActionBuffer) {
        val rows = grid.size
        val cols = grid[0].size
        require(near.rows == rows && near.cols == cols) {
            "Nearest 尺寸不匹配：${near.rows}x${near.cols} vs ${rows}x${cols}"
        }
        near.prepare(grid)
        out.clear()
        addInPlaceMoves(grid, near, out)
        addHorizontalMoves(grid, near, out)
        addVerticalMoves(grid, near, out)
    }

    /** 原地消：只看右/下方向，保证一对砖只从行优先序靠前的那一端报一次。 */
    private fun addInPlaceMoves(grid: Array<IntArray>, near: Nearest, moves: ActionBuffer) {
        val cols = near.cols
        for (r in 0 until near.rows) {
            val base = r * cols
            for (c in 0 until cols) {
                val cls = grid[r][c]
                if (cls == BACKGROUND) continue
                val idx = base + c
                val right = near.right[idx]
                if (right >= 0 && grid[r][right] == cls) {
                    moves.add(r, c, r, c, r, right)
                }
                val down = near.down[idx]
                if (down >= 0 && grid[down][c] == cls) {
                    moves.add(r, c, r, c, down, c)
                }
            }
        }
    }

    /**
     * 横向移动：按"空段 + 紧邻砖段"枚举，和原实现的区域语义一致。
     *
     * 关键点（踩过的坑）：落点不一定是空位。把一块砖往空段方向拖，会把它和空段之间的
     * 砖一起顶走，落点可以是砖段里的某一格。所以目标 t 的范围是
     * `[砖段起点+1, 砖段终点+空段长度]`，再从窗口里反查"哪块砖能落到 t"。
     */
    private fun addHorizontalMoves(grid: Array<IntArray>, near: Nearest, moves: ActionBuffer) {
        val rows = near.rows
        val cols = near.cols
        for (r in 0 until rows) {
            var p = 0
            while (p < cols) {
                if (grid[r][p] != BACKGROUND) { p++; continue }
                val emptyStart = p
                while (p < cols && grid[r][p] == BACKGROUND) p++
                val emptyEnd = p - 1
                val emptySize = emptyEnd - emptyStart + 1

                // 空段左边的砖段：向右拖进空段
                if (emptyStart > 0) {
                    var brickStart = emptyStart - 1
                    while (brickStart > 0 && grid[r][brickStart - 1] != BACKGROUND) brickStart--
                    horizontalToRight(grid, near, moves, r, brickStart, emptyStart - 1, emptySize)
                }
                // 空段右边的砖段：向左拖进空段
                if (emptyEnd + 1 < cols) {
                    var brickEnd = emptyEnd + 1
                    while (brickEnd + 1 < cols && grid[r][brickEnd + 1] != BACKGROUND) brickEnd++
                    horizontalToLeft(grid, near, moves, r, emptyEnd + 1, brickEnd, emptySize)
                }
            }
        }
    }

    private fun horizontalToRight(
        grid: Array<IntArray>, near: Nearest, moves: ActionBuffer,
        row: Int, brickStart: Int, brickEnd: Int, emptySize: Int,
    ) {
        val cols = near.cols
        val runMask = classMaskRow(grid, row, brickStart, brickEnd)
        val targetFirst = brickStart + 1
        val targetLast = brickEnd + emptySize
        for (t in targetFirst..targetLast) {
            val lo = maxOf(brickStart, t - emptySize)
            val hi = minOf(brickEnd, t - 1)
            if (lo > hi) continue
            val idx = row * cols + t
            val up = near.up[idx]
            val down = near.down[idx]
            val upClass = if (up >= 0) grid[up][t] else BACKGROUND
            val downClass = if (down >= 0) grid[down][t] else BACKGROUND
            if (upClass == BACKGROUND && downClass == BACKGROUND) continue
            if (!mayContain(runMask, upClass) && !mayContain(runMask, downClass)) continue
            for (b in lo..hi) {
                val cls = grid[row][b]
                if (cls == upClass) moves.add(row, b, row, t, up, t)
                if (cls == downClass) moves.add(row, b, row, t, down, t)
            }
        }
    }

    private fun horizontalToLeft(
        grid: Array<IntArray>, near: Nearest, moves: ActionBuffer,
        row: Int, brickStart: Int, brickEnd: Int, emptySize: Int,
    ) {
        val cols = near.cols
        val runMask = classMaskRow(grid, row, brickStart, brickEnd)
        val targetFirst = brickStart - emptySize
        val targetLast = brickEnd - 1
        for (t in targetFirst..targetLast) {
            val lo = maxOf(brickStart, t + 1)
            val hi = minOf(brickEnd, t + emptySize)
            if (lo > hi) continue
            val idx = row * cols + t
            val up = near.up[idx]
            val down = near.down[idx]
            val upClass = if (up >= 0) grid[up][t] else BACKGROUND
            val downClass = if (down >= 0) grid[down][t] else BACKGROUND
            if (upClass == BACKGROUND && downClass == BACKGROUND) continue
            if (!mayContain(runMask, upClass) && !mayContain(runMask, downClass)) continue
            for (b in lo..hi) {
                val cls = grid[row][b]
                if (cls == upClass) moves.add(row, b, row, t, up, t)
                if (cls == downClass) moves.add(row, b, row, t, down, t)
            }
        }
    }

    /** 纵向移动，逻辑与 [addHorizontalMoves] 对称：空段 + 上/下紧邻砖段。 */
    private fun addVerticalMoves(grid: Array<IntArray>, near: Nearest, moves: ActionBuffer) {
        val rows = near.rows
        val cols = near.cols
        for (c in 0 until cols) {
            var p = 0
            while (p < rows) {
                if (grid[p][c] != BACKGROUND) { p++; continue }
                val emptyStart = p
                while (p < rows && grid[p][c] == BACKGROUND) p++
                val emptyEnd = p - 1
                val emptySize = emptyEnd - emptyStart + 1

                // 空段上方的砖段：向下拖进空段
                if (emptyStart > 0) {
                    var brickStart = emptyStart - 1
                    while (brickStart > 0 && grid[brickStart - 1][c] != BACKGROUND) brickStart--
                    verticalToDown(grid, near, moves, c, brickStart, emptyStart - 1, emptySize)
                }
                // 空段下方的砖段：向上拖进空段
                if (emptyEnd + 1 < rows) {
                    var brickEnd = emptyEnd + 1
                    while (brickEnd + 1 < rows && grid[brickEnd + 1][c] != BACKGROUND) brickEnd++
                    verticalToUp(grid, near, moves, c, emptyEnd + 1, brickEnd, emptySize)
                }
            }
        }
    }

    private fun verticalToDown(
        grid: Array<IntArray>, near: Nearest, moves: ActionBuffer,
        col: Int, brickStart: Int, brickEnd: Int, emptySize: Int,
    ) {
        val cols = near.cols
        val runMask = classMaskCol(grid, col, brickStart, brickEnd)
        val targetFirst = brickStart + 1
        val targetLast = brickEnd + emptySize
        for (t in targetFirst..targetLast) {
            val lo = maxOf(brickStart, t - emptySize)
            val hi = minOf(brickEnd, t - 1)
            if (lo > hi) continue
            val idx = t * cols + col
            val left = near.left[idx]
            val right = near.right[idx]
            val leftClass = if (left >= 0) grid[t][left] else BACKGROUND
            val rightClass = if (right >= 0) grid[t][right] else BACKGROUND
            if (leftClass == BACKGROUND && rightClass == BACKGROUND) continue
            if (!mayContain(runMask, leftClass) && !mayContain(runMask, rightClass)) continue
            for (b in lo..hi) {
                val cls = grid[b][col]
                if (cls == leftClass) moves.add(b, col, t, col, t, left)
                if (cls == rightClass) moves.add(b, col, t, col, t, right)
            }
        }
    }

    private fun verticalToUp(
        grid: Array<IntArray>, near: Nearest, moves: ActionBuffer,
        col: Int, brickStart: Int, brickEnd: Int, emptySize: Int,
    ) {
        val cols = near.cols
        val runMask = classMaskCol(grid, col, brickStart, brickEnd)
        val targetFirst = brickStart - emptySize
        val targetLast = brickEnd - 1
        for (t in targetFirst..targetLast) {
            val lo = maxOf(brickStart, t + 1)
            val hi = minOf(brickEnd, t + emptySize)
            if (lo > hi) continue
            val idx = t * cols + col
            val left = near.left[idx]
            val right = near.right[idx]
            val leftClass = if (left >= 0) grid[t][left] else BACKGROUND
            val rightClass = if (right >= 0) grid[t][right] else BACKGROUND
            if (leftClass == BACKGROUND && rightClass == BACKGROUND) continue
            if (!mayContain(runMask, leftClass) && !mayContain(runMask, rightClass)) continue
            for (b in lo..hi) {
                val cls = grid[b][col]
                if (cls == leftClass) moves.add(b, col, t, col, t, left)
                if (cls == rightClass) moves.add(b, col, t, col, t, right)
            }
        }
    }

    /** 一段砖的类别位图；类别超过 63 时退回 -1（每次都扫窗口，不靠位图剪枝）。 */
    private fun classMaskRow(grid: Array<IntArray>, row: Int, from: Int, to: Int): Long {
        var mask = 0L
        for (i in from..to) {
            val v = grid[row][i]
            if (v !in 1..63) return -1L
            mask = mask or (1L shl v)
        }
        return mask
    }

    private fun classMaskCol(grid: Array<IntArray>, col: Int, from: Int, to: Int): Long {
        var mask = 0L
        for (i in from..to) {
            val v = grid[i][col]
            if (v !in 1..63) return -1L
            mask = mask or (1L shl v)
        }
        return mask
    }

    private fun mayContain(mask: Long, cls: Int): Boolean {
        if (cls <= 0) return false
        if (cls >= 64) return mask == -1L
        return (mask and (1L shl cls)) != 0L
    }

    /**
     * 一张盘面的"四个方向最近非空格"索引表，按尺寸复用。
     *
     * 索引 `i = row * cols + col`；`left` / `right` 存最近非空的**列号**，
     * `up` / `down` 存**行号**，-1 表示该方向没有非空格。
     */
    internal class Nearest(val rows: Int, val cols: Int) {
        val left = IntArray(rows * cols)
        val right = IntArray(rows * cols)
        val up = IntArray(rows * cols)
        val down = IntArray(rows * cols)

        fun prepare(grid: Array<IntArray>) {
            for (r in 0 until rows) {
                val base = r * cols
                var last = -1
                for (c in 0 until cols) {
                    left[base + c] = last
                    if (grid[r][c] != BACKGROUND) last = c
                }
                last = -1
                for (c in cols - 1 downTo 0) {
                    right[base + c] = last
                    if (grid[r][c] != BACKGROUND) last = c
                }
            }
            for (c in 0 until cols) {
                var last = -1
                for (r in 0 until rows) {
                    up[r * cols + c] = last
                    if (grid[r][c] != BACKGROUND) last = r
                }
                last = -1
                for (r in rows - 1 downTo 0) {
                    down[r * cols + c] = last
                    if (grid[r][c] != BACKGROUND) last = r
                }
            }
        }
    }

    /**
     * 兼容旧口径：只给 [fromRow, fromCol, pairRow, pairCol]。
     *
     * **丢掉了落点，所以不能拿来执行**（同一个配对可能有好几种落点），只适合提示层画框。
     * 执行与求解一律走 [findActions]。这里按 (from, pair) 无序去重，免得同一对在提示里画两遍。
     */
    fun findMoves(grid: Array<IntArray>): List<IntArray> {
        val cols = grid[0].size
        val result = ArrayList<IntArray>()
        for (action in findActions(grid)) {
            val from = action[0] * cols + action[1]
            val pair = action[4] * cols + action[5]
            val duplicate = result.any {
                val f = it[0] * cols + it[1]
                val p = it[2] * cols + it[3]
                (f == from && p == pair) || (f == pair && p == from)
            }
            if (!duplicate) result.add(intArrayOf(action[0], action[1], action[4], action[5]))
        }
        return result
    }

    /** [findMoves] 的强类型包装：全部可行走法（识别出来的整盘二维数组直接喂进来）。 */
    fun hints(grid: Array<IntArray>): List<BrickMove> = findMoves(grid).map { move ->
        BrickMove(
            fromRow = move[0],
            fromCol = move[1],
            pairRow = move[2],
            pairCol = move[3],
        )
    }

    /** 第一条可行走法。 */
    fun firstHint(grid: Array<IntArray>): BrickMove? = hints(grid).firstOrNull()

    /**
     * 把 [findMoves] 给出的候选补全为可直接喂给 [actionStep] 的 6 元动作。
     *
     * 落点**不由这里猜**：同一对砖经常有好几种可行的拖动方案（拖其中任意一块、拖多远），
     * 全部由 [findActions] 枚举好 —— 这里只按配对把动作捞出来，挨个用 [actionStep] 试到能行为止。
     * 落点是不是空位也不预设：拖动时整条砖会被顶着走，落点可能是别的砖正占着的格子。
     *
     * @return `[fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]`，一个都试不成返回 null。
     */
    fun resolveAction(grid: Array<IntArray>, move: IntArray): IntArray? {
        val fromRow = move[0]
        val fromCol = move[1]
        val pairRow = move[2]
        val pairCol = move[3]

        for (action in findActions(grid)) {
            if (action[0] != fromRow || action[1] != fromCol ||
                action[4] != pairRow || action[5] != pairCol
            ) continue
            val probe = Array(grid.size) { grid[it].copyOf() }
            // GAME_FAIL 也算"这一步能动"：它表示这步消完盘面没救了（全局策略问题），
            // 不是这一步不合法 —— 提示层要列全部可消对，不能因此把真能消的一对吞掉。
            when (actionStep(probe, action)) {
                ACTION_SUCCESS, GAME_SUCCESS, GAME_FAIL -> return action
            }
        }
        return null
    }

    /**
     * 自动提示 / 自动求解入口：从当前盘面挑出第一个可执行的 6 元动作。
     *
     * @return 可直接交给 [actionStep] 的动作；已无可消除对时返回 null（即游戏失败）。
     */
    fun autoHint(grid: Array<IntArray>): IntArray? {
        for (action in findActions(grid)) {
            val probe = Array(grid.size) { grid[it].copyOf() }
            when (actionStep(probe, action)) {
                ACTION_SUCCESS, GAME_SUCCESS, GAME_FAIL -> return action
            }
        }
        return null
    }

    // ------------------------------------------------------------------
    // 执行一步操作
    // ------------------------------------------------------------------

    /**
     * @param action [fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]
     * @param checkTerminal 是否在最后判"这步之后还有没有动作 / 是否已清空"。
     *   提示与执行层要这个结论（GAME_SUCCESS / GAME_FAIL）；求解器内部**不要**：
     *   它把 GAME_FAIL 也当可走，终局由下一层 `moves()` 自然判空。关掉后每节点从
     *   1+C 次全盘 [findActions] 降到 1 次（C = 候选动作数）。
     * @return GAME_SUCCESS / ACTION_SUCCESS / GAME_FAIL 或各类负数错误码
     */
    fun actionStep(
        grid: Array<IntArray>,
        action: IntArray,
        checkTerminal: Boolean = true,
    ): Int {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairedRow = action[4]
        val pairedCol = action[5]

        if (grid[fromRow][fromCol] == BACKGROUND || grid[pairedRow][pairedCol] == BACKGROUND) {
            return ACTION_FAIL
        }
        if ((fromRow == pairedRow && fromCol == pairedCol) ||
            grid[fromRow][fromCol] != grid[pairedRow][pairedCol]
        ) {
            return ACTION_FAIL
        }
        if (toRow == pairedRow && toCol == pairedCol) {
            return FROM_PAIR_SAME_BRICK
        }
        // to 必须与 from 同行或同列
        if (fromRow != toRow && fromCol != toCol) {
            return FROM_PAIR_SAME_BRICK
        }
        // from 与 paired 同行/同列时，只允许原地消除
        if ((fromRow == pairedRow || fromCol == pairedCol) && (fromRow != toRow || fromCol != toCol)) {
            return FROM_PAIR_SAME_BRICK
        }
        // to 必须与 paired 同行或同列
        if (toRow != pairedRow && toCol != pairedCol) {
            return FROM_PAIR_SAME_BRICK
        }

        if (fromRow == toRow && fromCol == toCol) {
            if (!tryRemoveBrick(grid, fromRow, fromCol, pairedRow, pairedCol)) {
                return FROM_PAIR_SAME_BRICK
            }
        } else if (fromRow == toRow) {
            val result = if (fromCol < toCol) {
                tryMoveRightAndRemoveBrick(grid, action)
            } else {
                tryMoveLeftAndRemoveBrick(grid, action)
            }
            if (result < 0) return result
        } else if (fromCol == toCol) {
            val result = if (fromRow < toRow) {
                tryMoveDownAndRemoveBrick(grid, action)
            } else {
                tryMoveUpAndRemoveBrick(grid, action)
            }
            if (result < 0) return result
        }

        // 求解器只关心"这一步合不合法"，终局由下一层 moves() 判空，这里直接收工
        if (!checkTerminal) return ACTION_SUCCESS
        if (isGameSuccess(grid)) return GAME_SUCCESS
        // 直接用 findActions 判空：findMoves 只是它的去重投影，多一次 O(n²) 没用
        if (findActions(grid).isEmpty()) return GAME_FAIL
        return ACTION_SUCCESS
    }

    /** from 所在列向上平移若干格，再消除 to 与 paired。 */
    private fun tryMoveUpAndRemoveBrick(grid: Array<IntArray>, action: IntArray): Int {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairedRow = action[4]
        val pairedCol = action[5]

        val steps = fromRow - toRow
        var emptySize = 0
        var firstEmptyIndex = -1
        for (i in fromRow - 1 downTo 0) {
            if (grid[i][fromCol] != BACKGROUND) {
                if (emptySize != 0) break
            } else {
                if (firstEmptyIndex < 0) firstEmptyIndex = i
                emptySize++
            }
        }
        if (emptySize == 0) return FROM_TO_SAME_LINE
        if (emptySize < steps) return FROM_TO_SAME_LINE_CAN_MOVE

        val start = minOf(toCol, pairedCol)
        val end = maxOf(toCol, pairedCol)
        for (i in start + 1 until end) {
            if (grid[toRow][i] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (s in 0 until steps) {
            for (j in firstEmptyIndex + 1..fromRow) {
                grid[j - 1 - s][fromCol] = grid[j - s][fromCol]
                grid[j - s][fromCol] = BACKGROUND
            }
        }
        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

    /** from 所在列向下平移若干格，再消除 to 与 paired。 */
    private fun tryMoveDownAndRemoveBrick(grid: Array<IntArray>, action: IntArray): Int {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairedRow = action[4]
        val pairedCol = action[5]

        val steps = toRow - fromRow
        var emptySize = 0
        var firstEmptyIndex = -1
        for (i in fromRow + 1 until grid.size) {
            if (grid[i][fromCol] != BACKGROUND) {
                if (emptySize != 0) break
            } else {
                if (firstEmptyIndex < 0) firstEmptyIndex = i
                emptySize++
            }
        }
        if (emptySize == 0) return FROM_TO_SAME_LINE
        if (emptySize < steps) return FROM_TO_SAME_LINE_CAN_MOVE

        val start = minOf(toCol, pairedCol)
        val end = maxOf(toCol, pairedCol)
        for (i in start + 1 until end) {
            if (grid[toRow][i] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (s in 0 until steps) {
            for (j in firstEmptyIndex - 1 downTo fromRow) {
                grid[j + 1 + s][fromCol] = grid[j + s][fromCol]
                grid[j + s][fromCol] = BACKGROUND
            }
        }
        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

    /** from 所在行向左平移若干格，再消除 to 与 paired。 */
    private fun tryMoveLeftAndRemoveBrick(grid: Array<IntArray>, action: IntArray): Int {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairedRow = action[4]
        val pairedCol = action[5]

        val steps = fromCol - toCol
        val row = grid[fromRow]
        var emptySize = 0
        var firstEmptyIndex = -1
        for (i in fromCol - 1 downTo 0) {
            if (row[i] != BACKGROUND) {
                if (emptySize != 0) break
            } else {
                if (firstEmptyIndex < 0) firstEmptyIndex = i
                emptySize++
            }
        }
        if (emptySize == 0) return FROM_TO_SAME_LINE
        if (emptySize < steps) return FROM_TO_SAME_LINE_CAN_MOVE

        val start = minOf(toRow, pairedRow)
        val end = maxOf(toRow, pairedRow)
        for (i in start + 1 until end) {
            if (grid[i][toCol] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (s in 0 until steps) {
            for (j in firstEmptyIndex + 1..fromCol) {
                row[j - 1 - s] = row[j - s]
                row[j - s] = BACKGROUND
            }
        }
        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

    /** from 所在行向右平移若干格，再消除 to 与 paired。 */
    private fun tryMoveRightAndRemoveBrick(grid: Array<IntArray>, action: IntArray): Int {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairedRow = action[4]
        val pairedCol = action[5]

        val steps = toCol - fromCol
        val row = grid[fromRow]
        var emptySize = 0
        var firstEmptyIndex = -1
        for (i in fromCol + 1 until row.size) {
            if (row[i] != BACKGROUND) {
                if (emptySize != 0) break
            } else {
                if (firstEmptyIndex < 0) firstEmptyIndex = i
                emptySize++
            }
        }
        if (emptySize == 0) return FROM_TO_SAME_LINE
        if (emptySize < steps) return FROM_TO_SAME_LINE_CAN_MOVE

        val start = minOf(toRow, pairedRow)
        val end = maxOf(toRow, pairedRow)
        for (i in start + 1 until end) {
            if (grid[i][toCol] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (s in 0 until steps) {
            for (j in firstEmptyIndex - 1 downTo fromCol) {
                row[j + 1 + s] = row[j + s]
                row[j + s] = BACKGROUND
            }
        }
        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

    /** 原地消除：from 与 paired 同行/同列且中间无阻挡。 */
    private fun tryRemoveBrick(
        grid: Array<IntArray>, fromRow: Int, fromCol: Int, pairedRow: Int, pairedCol: Int
    ): Boolean {
        if (fromRow == pairedRow) {
            val start = minOf(fromCol, pairedCol)
            val end = maxOf(fromCol, pairedCol)
            for (i in start + 1 until end) {
                if (grid[fromRow][i] != BACKGROUND) return false
            }
        }
        if (fromCol == pairedCol) {
            val start = minOf(fromRow, pairedRow)
            val end = maxOf(fromRow, pairedRow)
            for (i in start + 1 until end) {
                if (grid[i][fromCol] != BACKGROUND) return false
            }
        }
        grid[fromRow][fromCol] = 0
        grid[pairedRow][pairedCol] = 0
        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        return true
    }

    private fun isGameSuccess(grid: Array<IntArray>): Boolean =
        grid.all { row -> row.all { it == BACKGROUND } }

    /** 保持与原实现一致的日志输出（from 坐标出现两次）；默认关，见 [logRemovals]。 */
    private fun logRemove(fromRow: Int, fromCol: Int, pairedRow: Int, pairedCol: Int) {
        if (!logRemovals) return
        println("remove brick:$fromRow,$fromCol->$fromRow,$fromCol->$pairedRow,$pairedCol")
    }
}
