package com.qcmian.brickgame.vision

import com.qcmian.brickgame.algorithm.BrickGameTipHelper

/**
 * 流水线的最后一环：把"切图 + 识别"得到的盘面接到 `BrickGameTipHelper` 上，给出可操作提示。
 *
 * 这里刻意不接受任何截图/模型类型，只吃 [BoardScan]（几何）+ [BoardState]（类别），
 * 所以识别换实现（像素聚类 → 训练好的模型）时这一层不用动。
 */
object HintEngine {

    /** 求解时最多试算多少个候选动作（见 [nextHint] 的注释）。 */
    private const val MAX_CANDIDATES = 64

    /** 全部可操作提示。候选来自 `findMoves`，落点由 `resolveAction` 补齐。 */
    fun hints(scan: BoardScan, state: BoardState): List<Hint> {
        val grid = state.asGrid()
        val out = ArrayList<Hint>()
        for (move in BrickGameTipHelper.findMoves(grid).take(MAX_CANDIDATES)) {
            val action = BrickGameTipHelper.resolveAction(grid, move) ?: continue
            out.add(toHint(scan, action))
        }
        return out
    }

    /**
     * 下一条可操作提示（游戏内悬浮窗用这个）：命中第一条就返回。
     *
     * 候选数**必须封顶**：`resolveAction` 每个候选都要拷一份盘面、跑一遍 `actionStep`（内部还会再跑
     * `findMoves`），而棋盘被识别歪时 `findMoves` 能吐出成百上千个候选 —— 不封顶就是几十秒的卡顿。
     */
    fun nextHint(scan: BoardScan, state: BoardState): Hint? {
        val grid = state.asGrid()
        for (move in BrickGameTipHelper.findMoves(grid).take(MAX_CANDIDATES)) {
            val action = BrickGameTipHelper.resolveAction(grid, move) ?: continue
            return toHint(scan, action)
        }
        return null
    }

    /** 盘面已清空。 */
    fun isCleared(state: BoardState): Boolean = state.labels.all { it == 0 }

    /** 还剩砖块但已无解 —— 说明识别出错（同类不同号/漏判空位），而不是真死了。 */
    fun isStuck(state: BoardState): Boolean =
        !isCleared(state) && BrickGameTipHelper.findMoves(state.asGrid()).isEmpty()

    private fun toHint(scan: BoardScan, action: IntArray): Hint {
        val from = CellPoint(action[0], action[1])
        val to = CellPoint(action[2], action[3])
        val pair = CellPoint(action[4], action[5])
        return Hint(
            from = from,
            to = to,
            pair = pair,
            fromRect = scan.rectOf(from),
            toRect = scan.rectOf(to),
            pairRect = scan.rectOf(pair),
        )
    }
}
