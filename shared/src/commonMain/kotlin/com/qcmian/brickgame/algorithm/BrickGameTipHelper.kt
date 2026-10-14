package com.qcmian.brickgame.algorithm

/**
 * 方块消除游戏的提示与执行逻辑。
 *
 * 网格 grid[row][col]：BACKGROUND(0) 表示空位，其他值为方块类型。
 *
 * move（findMoves 返回）：[fromRow, fromCol, toRow, toCol]
 * action（actionStep 入参）：[fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]
 */
object BrickGameTipHelper {

    @JvmField
    var BACKGROUND = 0

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

    fun findMoves(grid: Array<IntArray>): List<IntArray> {
        val moves = ArrayList<IntArray>()
        val rows = grid.size
        val cols = grid[0].size

        for (row in 0 until rows) {
            moveRight(grid, row, moves)
            moveLeft(grid, row, moves)
        }
        for (col in 0 until cols) {
            moveDown(grid, col, moves)
            moveUp(grid, col, moves)
        }
        return uniqueMove(moves)
    }

    /** 去重：A->B 与 B->A 视为同一个 move。 */
    private fun uniqueMove(list: List<IntArray>): List<IntArray> {
        val result = ArrayList<IntArray>()
        for (origin in list) {
            val isDuplicate = result.any { exist ->
                (exist[0] == origin[0] && exist[1] == origin[1] &&
                        exist[2] == origin[2] && exist[3] == origin[3]) ||
                        (exist[0] == origin[2] && exist[1] == origin[3] &&
                                exist[2] == origin[0] && exist[3] == origin[1])
            }
            if (!isDuplicate) {
                result.add(origin)
            }
        }
        return result
    }

    // ---------------------- 纵向（列内移动，行内匹配） ----------------------

    private fun moveUp(grid: Array<IntArray>, colIndex: Int, moves: MutableList<IntArray>) {
        var startIndex = 0
        // 顶部连续的方块不需要移动，直接匹配
        var i = 0
        while (i < grid.size && grid[i][colIndex] != BACKGROUND) {
            matchRow(grid, i, colIndex, i, colIndex, moves)
            i++
        }
        while (true) {
            val region = findNextUpMoveRegionInCol(grid, colIndex, startIndex)
            if (!isValidMoveRegion(region)) return
            val emptySize = region[1] - region[0] + 1
            for (brick in region[2]..region[3]) {
                for (step in 0..emptySize) {
                    matchRow(grid, brick, colIndex, brick - step, colIndex, moves)
                }
            }
            startIndex = region[3] + 1
        }
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
            for (brick in region[3] downTo region[2]) {
                for (step in 0..emptySize) {
                    matchRow(grid, brick, colIndex, brick + step, colIndex, moves)
                }
            }
            endIndex = region[2] - 1
        }
    }

    /**
     * 自上而下查找 "空位段 + 紧随其后的方块段"。
     * 返回 [emptyStart, emptyEnd, brickStart, brickEnd]，找不到的项为 -1。
     */
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

    /**
     * 自下而上查找 "空位段 + 紧随其后的方块段"。
     * 返回 [emptyStart, emptyEnd, brickStart, brickEnd]，找不到的项为 -1。
     */
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

    /**
     * 在 rowIndex 行内，以 colIndex 为起点向左、向右各找最近的非空方块，
     * 若与 (targetRow, targetCol) 同类则记录一个 move。
     */
    private fun matchRow(
        grid: Array<IntArray>, targetRow: Int, targetCol: Int,
        rowIndex: Int, colIndex: Int, moves: MutableList<IntArray>
    ) {
        val target = grid[targetRow][targetCol]

        // 向左
        for (i in colIndex - 1 downTo 0) {
            if (grid[rowIndex][i] != BACKGROUND) {
                if (grid[rowIndex][i] == target) {
                    moves.add(intArrayOf(targetRow, targetCol, rowIndex, i))
                }
                break
            }
        }

        // 向右
        for (i in colIndex + 1 until grid[0].size) {
            if (grid[rowIndex][i] != BACKGROUND) {
                if (grid[rowIndex][i] == target) {
                    moves.add(intArrayOf(targetRow, targetCol, rowIndex, i))
                }
                return
            }
        }
    }

    // ---------------------- 横向（行内移动，列内匹配） ----------------------

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
            for (brick in region[2]..region[3]) {
                for (step in 0..emptySize) {
                    matchColumn(grid, rowIndex, brick, rowIndex, brick - step, moves)
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
            for (brick in region[3] downTo region[2]) {
                for (step in 0..emptySize) {
                    matchColumn(grid, rowIndex, brick, rowIndex, brick + step, moves)
                }
            }
            endIndex = region[2] - 1
        }
    }

    /** 自左而右查找 "空位段 + 紧随其后的方块段"。 */
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

    /** 自右而左查找 "空位段 + 紧随其后的方块段"。 */
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

    /**
     * 在 colIndex 列内，以 rowIndex 为起点向上、向下各找最近的非空方块，
     * 若与 (targetRow, targetCol) 同类则记录一个 move。
     */
    private fun matchColumn(
        grid: Array<IntArray>, targetRow: Int, targetCol: Int,
        rowIndex: Int, colIndex: Int, moves: MutableList<IntArray>
    ) {
        val target = grid[targetRow][targetCol]

        // 向上
        for (i in rowIndex - 1 downTo 0) {
            if (grid[i][colIndex] != BACKGROUND) {
                if (grid[i][colIndex] == target) {
                    moves.add(intArrayOf(targetRow, targetCol, i, colIndex))
                }
                break
            }
        }

        // 向下
        for (i in rowIndex + 1 until grid.size) {
            if (grid[i][colIndex] != BACKGROUND) {
                if (grid[i][colIndex] == target) {
                    moves.add(intArrayOf(targetRow, targetCol, i, colIndex))
                }
                return
            }
        }
    }

    private fun isValidMoveRegion(region: IntArray): Boolean = region.all { it >= 0 }

    // ------------------------------------------------------------------
    // 执行一步操作
    // ------------------------------------------------------------------

    /**
     * @param action [fromRow, fromCol, toRow, toCol, pairedRow, pairedCol]
     * @return GAME_SUCCESS / ACTION_SUCCESS / GAME_FAIL 或各类负数错误码
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

        if (isGameSuccess(grid)) return GAME_SUCCESS
        if (findMoves(grid).isEmpty()) return GAME_FAIL
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

    /** 保持与原实现一致的日志输出（from 坐标出现两次）。 */
    private fun logRemove(fromRow: Int, fromCol: Int, pairedRow: Int, pairedCol: Int) {
        println("remove brick:$fromRow,$fromCol->$fromRow,$fromCol->$pairedRow,$pairedCol")
    }
}
