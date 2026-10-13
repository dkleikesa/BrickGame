package com.qcmian.brickgame.vision

import kotlin.math.roundToInt

/**
 * 一帧的体检报告：标定几何 + 每格平坦度 + 识别类别 + 类别计数 + 提示。
 *
 * 为什么要这份日志：真机上提示不对时，两种原因长得几乎一样 ——
 *  - **网格切歪**（几何错）：平坦度矩阵会显示"每个格子都有结构"，因为格线压在砖块中间；
 *  - **砖块认错**（类别错）：平坦度正常（空位判对），但同类被拆成不同号 —— 计数里出现奇数类是强信号。
 *
 * 所以两者分开列：`score` 一行看一层是几何，`label`/`counts`/`odd` 看识别。
 * 只读 [BoardScan]/[BoardState]，不碰 Bitmap，所以放在 commonMain。
 */
object BoardDiagnostics {

    /** 每条报告都打这个 tag，真机上一句 `adb logcat -s BoardDiag` 就能捞出整帧。 */
    const val TAG = "BoardDiag"

    fun report(scan: BoardScan, state: BoardState, hint: Hint?): List<String> {
        val rows = scan.rows
        val cols = scan.cols
        val out = ArrayList<String>(rows * 2 + 8)

        out += "head ${rows}行x${cols}列 pitch=${scan.pitchX}x${scan.pitchY} " +
            "空位=${scan.empty.count { it }}/${scan.cellCount} 空位阈值=${scan.scoreThreshold.roundToInt()} " +
            "类别数=${state.distinctClasses}"
        out += "xlines " + scan.xlines.joinToString(",") { it.roundToInt().toString() }
        out += "ylines " + scan.ylines.joinToString(",") { it.roundToInt().toString() }

        for (r in 0 until rows) {
            val from = r * cols
            out += "score r${padded(r)} " + (0 until cols).joinToString(" ") {
                scan.score[from + it].roundToInt().toString()
            }
        }
        for (r in 0 until rows) {
            val from = r * cols
            out += "label r${padded(r)} " + (0 until cols).joinToString(" ") {
                state.labels[from + it].toString().padStart(4)
            }
        }

        // 同类计数：偶数才消得掉。出现奇数类 = 识别把同款拆开了（或漏判了空位）。
        val counts = ArrayList<Pair<Int, Int>>()
        val tally = HashMap<Int, Int>()
        for (label in state.labels) {
            if (label != 0) tally[label] = (tally[label] ?: 0) + 1
        }
        counts += tally.entries.sortedByDescending { it.value }.map { it.key to it.value }
        out += "counts " + counts.joinToString(" ") { "${it.first}:${it.second}" }
        out += "odd " + counts.filter { it.second % 2 == 1 }.joinToString(",") { "${it.first}x${it.second}" }

        out += "hint " + (hint?.describe() ?: "无")
        hint?.let {
            out += "hintCells " +
                "from(r${it.from.row},c${it.from.col}) score=${cellScore(scan, state, it.from)} " +
                "to(r${it.to.row},c${it.to.col}) score=${cellScore(scan, state, it.to)} " +
                "pair(r${it.pair.row},c${it.pair.col}) score=${cellScore(scan, state, it.pair)}"
        }
        return out
    }

    private fun cellScore(scan: BoardScan, state: BoardState, point: CellPoint): String {
        if (point.row !in 0 until scan.rows || point.col !in 0 until scan.cols) return "越界"
        val index = scan.indexOf(point.row, point.col)
        return "${scan.score[index].roundToInt()}/label=${state.labels[index]}"
    }

    private fun padded(value: Int): String = value.toString().padStart(2, '0')
}
