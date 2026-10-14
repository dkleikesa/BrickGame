package com.qcmian.brickgame.vision

/**
 * "认不出来"的格子用的类别号起点。
 *
 * 识别没把握的格子（余弦相似度低于模型自带判据）先在帧内找最像的格子纠错；纠不出来才落到这里，
 * 拿一个**每格唯一**的号 —— 于是它永远不会跟别的格子配成一对（宁可不给提示，也不给错提示）。
 */
const val UNKNOWN_LABEL_BASE = 1_000_000

/** 十字坐标（行、列）。 */
data class CellPoint(val row: Int, val col: Int)

/** 格子在原图里的像素矩形，`[left, top, right, bottom)`。 */
data class CellRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val centerX: Int get() = (left + right) / 2
    val centerY: Int get() = (top + bottom) / 2
    val width: Int get() = right - left
    val height: Int get() = bottom - top
}

/**
 * 一帧截图的"切图"结果：网格几何 + 每格判空 + 每格在原图里的矩形。
 *
 * 对应 `tile_recognizer.detect_grid` + `grid_from_lines` 的返回值。
 */
class BoardScan(
    val rows: Int,
    val cols: Int,
    val xlines: DoubleArray,
    val ylines: DoubleArray,
    /** 行优先，长度 rows*cols；>=[scoreThreshold] 视为"有砖块"。 */
    val score: FloatArray,
    val scoreThreshold: Float,
    /** 行优先，长度 rows*cols；格子在原图坐标系里的矩形。 */
    val rects: Array<CellRect>,
    val pitchX: Int,
    val pitchY: Int,
) {
    val cellCount: Int get() = rows * cols

    /** 行优先，长度 rows*cols；true = 已消除的空位。 */
    val empty: BooleanArray = BooleanArray(cellCount) { score[it] < scoreThreshold }

    fun indexOf(row: Int, col: Int): Int = row * cols + col

    fun pointAt(index: Int): CellPoint = CellPoint(index / cols, index % cols)

    fun rectOf(point: CellPoint): CellRect = rects[indexOf(point.row, point.col)]

    override fun toString(): String =
        "BoardScan(${rows}x$cols, pitch=${pitchX}x$pitchY, 空位=${empty.count { it }})"
}

/**
 * 标签矩阵：每格的类别 id（0 = 背景/空位，>0 = 识别出的类别）。
 *
 * `BrickGameTipHelper` 只要求"同款同号、空位为 0"，所以模型输出和像素聚类输出可以直接互换。
 */
class BoardState(val rows: Int, val cols: Int, val labels: IntArray) {
    init {
        require(labels.size == rows * cols) { "标签数量不符：期望 ${rows * cols}，实际 ${labels.size}" }
    }

    /** 转成算法要的 `grid[row][col]`。 */
    fun asGrid(): Array<IntArray> = Array(rows) { r ->
        IntArray(cols) { c -> labels[r * cols + c] }
    }

    val distinctClasses: Int get() = labels.toSet().count { it != 0 }

    companion object {
        /** 从切图结果与每格类别构造；空位强制归零，避免检测器与识别结果打架。 */
        fun of(scan: BoardScan, labels: IntArray): BoardState {
            val cleaned = IntArray(labels.size) { if (scan.empty[it]) 0 else labels[it].coerceAtLeast(0) }
            return BoardState(scan.rows, scan.cols, cleaned)
        }
    }
}

/**
 * 一条可操作提示：把 [from] 处的砖滑到 [to]，与 [pair] 上的同色砖一起消除。
 *
 * [from]/[pair] 来自算法返回的候选对，[to] 是补齐后的落点（见 `BrickGameTipHelper.resolveAction`）。
 */
data class Hint(
    val from: CellPoint,
    val to: CellPoint,
    val pair: CellPoint,
    val fromRect: CellRect,
    val toRect: CellRect,
    val pairRect: CellRect,
) {
    /** 给日志/UI 用的一行描述。 */
    fun describe(): String =
        "(${from.row},${from.col}) 与 (${pair.row},${pair.col}) 中间无阻挡，可消"

    /**
     * 只要"哪两块能消"的短描述（UI 上标对、日志里列对用这个）。
     *
     * 落点 [to] 是算法内部推的，玩家不需要知道 —— 游戏自己会判定落点，所以这里不报。
     */
    fun pairText(): String = "(${from.row},${from.col})+(${pair.row},${pair.col})"
}
