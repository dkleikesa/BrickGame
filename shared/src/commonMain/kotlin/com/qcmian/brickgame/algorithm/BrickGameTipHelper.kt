package com.qcmian.brickgame.algorithm

/**
 * 一组可消除的同色砖。
 *
 * 语义：把 [fromRow, fromCol] 处的砖沿直线滑动到 [pairRow, pairCol] 所在行/列的交叉空位，
 * 滑行结束时两砖成对消除（配对砖由游戏引擎在落点自动判定，不需要预知落点）。
 *
 * 注意：字段来自 `findMoves` 返回的 `[fromRow, fromCol, pairRow, pairCol]`，
 * 其中前两个元素是待滑动的砖、后两个是它的配对砖。真正可执行的 6 元动作
 * 由 [BrickGameTipHelper.resolveAction] 计算。
 */
data class BrickMove(
    val fromRow: Int,
    val fromCol: Int,
    val pairRow: Int,
    val pairCol: Int,
)

/**
 * 由 `lib2.jar`(`com.sogou.iot.lib2.BrickGameTipHelper`)反编译后逐行移植的纯 Kotlin 实现，
 * 供游戏自动提示 / 自动求解使用。
 *
 * 坐标系：`grid[row][col]`，row 向下递增、col 向右递增，`BACKGROUND` 表示空格。
 * 依赖点阵/方块图的 `classBitmap_data` 与 Python 侧 `brick_game_env_custom.py` 保持一致。
 */
object BrickGameTipHelper {

    /** 空格（无砖）的取值，与 Python `to_state` 中 `grid != 0` 的判定一致。 */
    const val BACKGROUND = 0

    /** 全部砖块消除，游戏胜利。 */
    const val GAME_SUCCESS = 2

    /** 本次消除成功。 */
    const val ACTION_SUCCESS = 1

    /** 非法操作：越界、取到空格、或落点与终点不构成合法句型。 */
    const val ACTION_FAIL = -1

    /** 当前盘面已无可消除对，游戏失败。 */
    const val GAME_FAIL = -2

    /** from 与 to 无效 / 落点与配对砖/终点不共线等结构性错误。 */
    const val FROM_PAIR_SAME_BRICK = -3

    /** from -> to 方向紧邻无空格，无法移动。 */
    const val FROM_TO_SAME_LINE = -4

    /** from -> to 之间空格数少于所需步数。 */
    const val FROM_TO_SAME_LINE_CAN_MOVE = -5

    /** to 与 paired 之间被其它砖块阻挡。 */
    const val TO_PAIR_SAME_LINE = -6

    /** 调试日志钩子，默认关闭；接入方可用它观察消除过程。 */
    var logger: ((String) -> Unit)? = null

    /**
     * 枚举当前盘面所有可行的消除动作。
     *
     * @return 每个动作为 `[fromRow, fromCol, toRow, toCol]`，顺序不保证稳定。
     */
    fun findMoves(grid: Array<IntArray>): List<IntArray> {
        val moves = mutableListOf<IntArray>()
        val rows = grid.size
        val cols = grid[0].size

        for (i in 0 until rows) {
            moveRight(grid, i, moves)
            moveLeft(grid, i, moves)
        }

        for (i in 0 until cols) {
            moveDown(grid, i, moves)
            moveUp(grid, i, moves)
        }

        return uniqueMove(moves)
    }

    /** 为自动提示准备的强类型包装，返回全部可行解。 */
    fun hints(grid: Array<IntArray>): List<BrickMove> = findMoves(grid).map { move ->
        BrickMove(
            fromRow = move[0],
            fromCol = move[1],
            pairRow = move[2],
            pairCol = move[3],
        )
    }

    /** 为自动提示准备的强类型包装，取首个可行解。 */
    fun firstHint(grid: Array<IntArray>): BrickMove? = hints(grid).firstOrNull()

    /**
     * 把 [findMoves] 给出的候选补全为可直接喂给 [actionStep] 的 6 元动作。
     *
     * 落点只可能是候选砖与配对砖构成的矩形的另一个角：`(fromRow, pairCol)` 或 `(pairRow, fromCol)`；
     * 当两砖本就相邻同线时落点等于起点（不需要滑行）。返回第一个能通过 [actionStep] 校验的方案。
     *
     * @return `[fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]`，无解返回 null。
     */
    fun resolveAction(grid: Array<IntArray>, move: IntArray): IntArray? {
        val fromRow = move[0]
        val fromCol = move[1]
        val pairRow = move[2]
        val pairCol = move[3]

        val candidates = listOf(
            intArrayOf(fromRow, pairCol),
            intArrayOf(pairRow, fromCol),
        )

        for (to in candidates) {
            if (to[0] == pairRow && to[1] == pairCol) continue
            val isFrom = to[0] == fromRow && to[1] == fromCol
            if (!isFrom && grid[to[0]][to[1]] != BACKGROUND) continue
            if (to[0] != fromRow && to[1] != fromCol) continue

            val action = intArrayOf(fromRow, fromCol, to[0], to[1], pairRow, pairCol)
            val probe = Array(grid.size) { grid[it].copyOf() }
            val result = actionStep(probe, action)
            if (result == ACTION_SUCCESS || result == GAME_SUCCESS) {
                return action
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
        for (move in findMoves(grid)) {
            resolveAction(grid, move)?.let { return it }
        }
        return null
    }

    private fun uniqueMove(list: List<IntArray>): List<IntArray> {
        val ret = mutableListOf<IntArray>()

        for (origin in list) {
            var isDuplicate = false

            for (existing in ret) {
                if (existing[0] == origin[0] && existing[1] == origin[1] &&
                    existing[2] == origin[2] && existing[3] == origin[3]
                ) {
                    isDuplicate = true
                    break
                }

                if (existing[0] == origin[2] && existing[1] == origin[3] &&
                    existing[2] == origin[0] && existing[3] == origin[1]
                ) {
                    isDuplicate = true
                    break
                }
            }

            if (!isDuplicate) {
                ret.add(origin)
            }
        }

        return ret
    }

    private fun moveUp(grid: Array<IntArray>, colIndex: Int, moves: MutableList<IntArray>) {
        var startIndex = 0

        var i = 0
        while (i < grid.size && grid[i][colIndex] != BACKGROUND) {
            matchRow(grid, i, colIndex, i, colIndex, moves)
            i++
        }

        while (true) {
            val region = findNextUpMoveRegionInCol(grid, colIndex, startIndex)
            if (!isValidMoveRegion(region)) return

            val emptySize = region[1] - region[0] + 1

            for (brickIndex in region[2]..region[3]) {
                for (offset in 0..emptySize) {
                    matchRow(grid, brickIndex, colIndex, brickIndex - offset, colIndex, moves)
                }
            }

            startIndex = region[3] + 1
        }
    }

    private fun findNextUpMoveRegionInCol(grid: Array<IntArray>, colIndex: Int, startIndex: Int): IntArray {
        var emptyStart = -1
        var emptyEnd = -1
        var brickStart = -1
        var brickEnd = -1
        val rows = grid.size

        if (startIndex >= rows) {
            return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
        }

        for (i in startIndex until rows) {
            if (grid[i][colIndex] != BACKGROUND) {
                if (emptyEnd >= 0) {
                    if (brickEnd < 0) brickEnd = i
                    if (brickStart < 0) brickStart = i
                    if (i == brickEnd + 1) brickEnd = i
                }
            } else {
                if (emptyEnd < 0) emptyEnd = i
                if (emptyStart < 0) emptyStart = i
                if (i == emptyEnd + 1) emptyEnd = i
                if (brickStart >= 0) break
            }
        }

        return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
    }

    private fun moveDown(grid: Array<IntArray>, colIndex: Int, moves: MutableList<IntArray>) {
        var endIndex = grid.size - 1

        var i = endIndex
        while (i >= 0 && grid[i][colIndex] != BACKGROUND) {
            matchRow(grid, i, colIndex, i, colIndex, moves)
            i--
        }

        while (true) {
            val region = findNextDownMoveRegionInCol(grid, colIndex, endIndex)
            if (!isValidMoveRegion(region)) return

            val emptySize = region[1] - region[0] + 1

            for (brickIndex in region[3] downTo region[2]) {
                for (offset in 0..emptySize) {
                    matchRow(grid, brickIndex, colIndex, brickIndex + offset, colIndex, moves)
                }
            }

            endIndex = region[2] - 1
        }
    }

    private fun findNextDownMoveRegionInCol(grid: Array<IntArray>, colIndex: Int, endIndex: Int): IntArray {
        var emptyStart = -1
        var emptyEnd = -1
        var brickStart = -1
        var brickEnd = -1

        if (endIndex <= 0) {
            return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
        }

        for (i in endIndex downTo 0) {
            if (grid[i][colIndex] != BACKGROUND) {
                if (emptyEnd >= 0) {
                    if (brickEnd < 0) brickEnd = i
                    if (brickStart < 0) brickStart = i
                    if (i == brickStart - 1) brickStart = i
                }
            } else {
                if (emptyEnd < 0) emptyEnd = i
                if (emptyStart < 0) emptyStart = i
                if (i == emptyStart - 1) emptyStart = i
                if (brickStart >= 0) break
            }
        }

        return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
    }

    private fun matchRow(
        grid: Array<IntArray>,
        targetRow: Int,
        targetCol: Int,
        rowIndex: Int,
        colIndex: Int,
        moves: MutableList<IntArray>,
    ) {
        val target = grid[targetRow][targetCol]

        if (colIndex > 0) {
            for (i in colIndex - 1 downTo 0) {
                if (grid[rowIndex][i] != BACKGROUND) {
                    if (grid[rowIndex][i] == target) {
                        moves.add(intArrayOf(targetRow, targetCol, rowIndex, i))
                    }
                    break
                }
            }
        }

        if (colIndex < grid[0].size - 1) {
            for (i in colIndex + 1 until grid[0].size) {
                if (grid[rowIndex][i] != BACKGROUND) {
                    if (grid[rowIndex][i] == target) {
                        moves.add(intArrayOf(targetRow, targetCol, rowIndex, i))
                    }
                    break
                }
            }
        }
    }

    private fun moveLeft(grid: Array<IntArray>, rowIndex: Int, moves: MutableList<IntArray>) {
        var startIndex = 0
        val row = grid[rowIndex]

        var i = 0
        while (i < row.size && row[i] != BACKGROUND) {
            matchColumn(grid, rowIndex, i, rowIndex, i, moves)
            i++
        }

        while (true) {
            val region = findNextLeftMoveRegionInRow(grid, rowIndex, startIndex)
            if (!isValidMoveRegion(region)) return

            val emptySize = region[1] - region[0] + 1

            for (brickIndex in region[2]..region[3]) {
                for (offset in 0..emptySize) {
                    matchColumn(grid, rowIndex, brickIndex, rowIndex, brickIndex - offset, moves)
                }
            }

            startIndex = region[3] + 1
        }
    }

    private fun moveRight(grid: Array<IntArray>, rowIndex: Int, moves: MutableList<IntArray>) {
        var endIndex = grid[0].size - 1
        val row = grid[rowIndex]

        var i = endIndex
        while (i >= 0 && row[i] != BACKGROUND) {
            matchColumn(grid, rowIndex, i, rowIndex, i, moves)
            i--
        }

        while (true) {
            val region = findNextRightMoveRegionInRow(grid, rowIndex, endIndex)
            if (!isValidMoveRegion(region)) return

            val emptySize = region[1] - region[0] + 1

            for (brickIndex in region[3] downTo region[2]) {
                for (offset in 0..emptySize) {
                    matchColumn(grid, rowIndex, brickIndex, rowIndex, brickIndex + offset, moves)
                }
            }

            endIndex = region[2] - 1
        }
    }

    private fun findNextLeftMoveRegionInRow(grid: Array<IntArray>, rowIndex: Int, startIndex: Int): IntArray {
        val row = grid[rowIndex]
        var emptyStart = -1
        var emptyEnd = -1
        var brickStart = -1
        var brickEnd = -1
        val cols = grid[0].size

        if (startIndex >= cols) {
            return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
        }

        for (i in startIndex until cols) {
            if (row[i] != BACKGROUND) {
                if (emptyEnd >= 0) {
                    if (brickEnd < 0) brickEnd = i
                    if (brickStart < 0) brickStart = i
                    if (i == brickEnd + 1) brickEnd = i
                }
            } else {
                if (emptyEnd < 0) emptyEnd = i
                if (emptyStart < 0) emptyStart = i
                if (i == emptyEnd + 1) emptyEnd = i
                if (brickStart >= 0) break
            }
        }

        return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
    }

    private fun findNextRightMoveRegionInRow(grid: Array<IntArray>, rowIndex: Int, endIndex: Int): IntArray {
        val row = grid[rowIndex]
        var emptyStart = -1
        var emptyEnd = -1
        var brickStart = -1
        var brickEnd = -1

        if (endIndex <= 0) {
            return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
        }

        for (i in endIndex downTo 0) {
            if (row[i] != BACKGROUND) {
                if (emptyEnd >= 0) {
                    if (brickEnd < 0) brickEnd = i
                    if (brickStart < 0) brickStart = i
                    if (i == brickStart - 1) brickStart = i
                }
            } else {
                if (emptyEnd < 0) emptyEnd = i
                if (emptyStart < 0) emptyStart = i
                if (i == emptyStart - 1) emptyStart = i
                if (brickStart >= 0) break
            }
        }

        return intArrayOf(emptyStart, emptyEnd, brickStart, brickEnd)
    }

    private fun matchColumn(
        grid: Array<IntArray>,
        targetRow: Int,
        targetCol: Int,
        rowIndex: Int,
        colIndex: Int,
        moves: MutableList<IntArray>,
    ) {
        val target = grid[targetRow][targetCol]

        if (rowIndex > 0) {
            for (i in rowIndex - 1 downTo 0) {
                if (grid[i][colIndex] != BACKGROUND) {
                    if (grid[i][colIndex] == target) {
                        moves.add(intArrayOf(targetRow, targetCol, i, colIndex))
                    }
                    break
                }
            }
        }

        if (rowIndex < grid.size - 1) {
            for (i in rowIndex + 1 until grid.size) {
                if (grid[i][colIndex] != BACKGROUND) {
                    if (grid[i][colIndex] == target) {
                        moves.add(intArrayOf(targetRow, targetCol, i, colIndex))
                    }
                    break
                }
            }
        }
    }

    private fun isValidMoveRegion(region: IntArray): Boolean {
        for (value in region) {
            if (value < 0) return false
        }
        return true
    }

    /**
     * 执行一次消除，直接就地修改 [grid]。
     *
     * @param action `[fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]`
     * @return 见本类顶部常量：`ACTION_SUCCESS` / `GAME_SUCCESS` / `GAME_FAIL` 及各类负值错误码。
     */
    fun actionStep(grid: Array<IntArray>, action: IntArray): Int {
        val fromRow = action[0]
        val fromCol = action[1]
        val toRow = action[2]
        val toCol = action[3]
        val pairedRow = action[4]
        val pairedCol = action[5]

        if (grid[fromRow][fromCol] == BACKGROUND || grid[pairedRow][pairedCol] == BACKGROUND) {
            return ACTION_FAIL
        }

        if (fromRow == pairedRow && fromCol == pairedCol) {
            return ACTION_FAIL
        }

        if (grid[fromRow][fromCol] != grid[pairedRow][pairedCol]) {
            return ACTION_FAIL
        }

        if (toRow == pairedRow && toCol == pairedCol) {
            return FROM_PAIR_SAME_BRICK
        }

        if (fromRow != toRow && fromCol != toCol) {
            return FROM_PAIR_SAME_BRICK
        }

        val validLine =
            (fromRow != pairedRow && fromCol != pairedCol) || (fromRow == toRow && fromCol == toCol)
        if (!validLine) {
            return FROM_PAIR_SAME_BRICK
        }

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

        return when {
            isGameSuccess(grid) -> GAME_SUCCESS
            findMoves(grid).isEmpty() -> GAME_FAIL
            else -> ACTION_SUCCESS
        }
    }

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

        if (emptySize == 0) {
            return FROM_TO_SAME_LINE
        }
        if (emptySize < steps) {
            return FROM_TO_SAME_LINE_CAN_MOVE
        }

        val start = minOf(toCol, pairedCol)
        val end = maxOf(toCol, pairedCol)
        for (i in start + 1 until end) {
            if (grid[toRow][i] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (i in 0 until steps) {
            for (j in firstEmptyIndex + 1..fromRow) {
                grid[j - 1 - i][fromCol] = grid[j - i][fromCol]
                grid[j - i][fromCol] = BACKGROUND
            }
        }

        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

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

        if (emptySize == 0) {
            return FROM_TO_SAME_LINE
        }
        if (emptySize < steps) {
            return FROM_TO_SAME_LINE_CAN_MOVE
        }

        val start = minOf(toCol, pairedCol)
        val end = maxOf(toCol, pairedCol)
        for (i in start + 1 until end) {
            if (grid[toRow][i] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (i in 0 until steps) {
            for (j in firstEmptyIndex - 1 downTo fromRow) {
                grid[j + 1 + i][fromCol] = grid[j + i][fromCol]
                grid[j + i][fromCol] = BACKGROUND
            }
        }

        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

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

        if (emptySize == 0) {
            return FROM_TO_SAME_LINE
        }
        if (emptySize < steps) {
            return FROM_TO_SAME_LINE_CAN_MOVE
        }

        val start = minOf(toRow, pairedRow)
        val end = maxOf(toRow, pairedRow)
        for (i in start + 1 until end) {
            if (grid[i][toCol] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (i in 0 until steps) {
            for (j in firstEmptyIndex + 1..fromCol) {
                row[j - 1 - i] = row[j - i]
                row[j - i] = BACKGROUND
            }
        }

        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

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

        if (emptySize == 0) {
            return FROM_TO_SAME_LINE
        }
        if (emptySize < steps) {
            return FROM_TO_SAME_LINE_CAN_MOVE
        }

        val start = minOf(toRow, pairedRow)
        val end = maxOf(toRow, pairedRow)
        for (i in start + 1 until end) {
            if (grid[i][toCol] != BACKGROUND) return TO_PAIR_SAME_LINE
        }

        for (i in 0 until steps) {
            for (j in firstEmptyIndex - 1 downTo fromCol) {
                row[j + 1 + i] = row[j + i]
                row[j + i] = BACKGROUND
            }
        }

        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[toRow][toCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return ACTION_SUCCESS
    }

    private fun tryRemoveBrick(
        grid: Array<IntArray>,
        fromRow: Int,
        fromCol: Int,
        pairedRow: Int,
        pairedCol: Int,
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

        logRemove(fromRow, fromCol, pairedRow, pairedCol)
        grid[fromRow][fromCol] = BACKGROUND
        grid[pairedRow][pairedCol] = BACKGROUND
        return true
    }

    private fun isGameSuccess(grid: Array<IntArray>): Boolean {
        for (row in grid) {
            for (value in row) {
                if (value != BACKGROUND) return false
            }
        }
        return true
    }

    private fun logRemove(fromRow: Int, fromCol: Int, pairedRow: Int, pairedCol: Int) {
        logger?.invoke("remove brick:$fromRow,$fromCol->$fromRow,$fromCol->$pairedRow,$pairedCol")
    }
}
