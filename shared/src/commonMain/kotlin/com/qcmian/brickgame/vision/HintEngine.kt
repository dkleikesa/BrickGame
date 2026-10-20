package com.qcmian.brickgame.vision

import com.qcmian.brickgame.algorithm.BrickGameTipHelper

/**
 * 流水线的最后一环：把"切图 + 识别"出来的**整盘二维数组**（`grid[row][col]`，0 = 空位）交给
 * [BrickGameTipHelper.findMoves] 枚举**全部可消除的砖块对**，再把每对翻成带画面矩形的 [Hint]。
 *
 * `findMoves` 给的只是"同款配对"：两块呈 L 形时还要把其中一块滑到十字交叉点，那个位置
 * 被别的砖占着就**一步消不掉**（得先挪开挡路的砖）。所以每对都带上 [Hint.direct]：
 * 悬浮窗照画不误，但能分清"现在就能点"和"要先腾地方"，不再出现"画了框却点不动"。
 *
 * 同款全靠识别层保证：`TfliteTileClassifier` 一律采信 top-1 原型（运行时不做置信度改判），
 * 没把握的那一帧会被留档给以后重训，但当前这一帧的类号就是它给出来的样子。
 */
object HintEngine {

    /** 全部可消除的对（一行可能同时存在好几对）。 */
    fun hints(scan: BoardScan, state: BoardState): List<Hint> =
        moves(state.asGrid()).map { toHint(scan, it) }

    /** 盘面已清空。 */
    fun isCleared(state: BoardState): Boolean = state.labels.all { it == 0 }

    /**
     * 还剩砖块、但一步都走不成（同款对全被挡住）。
     *
     * 口径要与自动模式一致：判"能不能走"看 [BrickGameTipHelper.resolveAction]，
     * 不看"有没有同款对"—— 后者会把"需要先腾位置"的盘面误判成还没卡住。
     */
    fun isStuck(state: BoardState): Boolean {
        if (isCleared(state)) return false
        val grid = state.asGrid()
        return BrickGameTipHelper.findMoves(grid)
            .none { BrickGameTipHelper.resolveAction(grid, it) != null }
    }

    /** 可消除的格子对（[Pair.first] 是要滑的那块，[Pair.second] 是配对砖）。 */
    fun pairs(grid: Array<IntArray>): List<Pair<CellPoint, CellPoint>> =
        moves(grid).map { it.from to it.pair }

    /** 一对可消除的砖：滑哪块、跟哪块一起消（落点由游戏判定，提示层不需要）。 */
    private data class Move(val from: CellPoint, val pair: CellPoint, val direct: Boolean)

    /**
     * [BrickGameTipHelper.findMoves] 的输出**全量**搬过来，按 [Move.from] 的行优先顺序排列
     * （同一帧里提示序号才稳定）。
     */
    private fun moves(grid: Array<IntArray>): List<Move> {
        if (grid.isEmpty() || grid[0].isEmpty()) return emptyList()
        val cols = grid[0].size
        return BrickGameTipHelper.findMoves(grid)
            .map { raw ->
                Move(
                    CellPoint(raw[0], raw[1]), CellPoint(raw[2], raw[3]),
                    // 落点算得出来 = 现在就能消；算不出来 = 得先挪开挡路的砖
                    direct = BrickGameTipHelper.resolveAction(grid, raw) != null,
                )
            }
            .sortedBy { it.from.row * cols + it.from.col }
    }

    private fun toHint(scan: BoardScan, move: Move): Hint = Hint(
        from = move.from,
        pair = move.pair,
        fromRect = scan.rectOf(move.from),
        pairRect = scan.rectOf(move.pair),
        direct = move.direct,
    )
}
