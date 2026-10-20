package com.qcmian.brickgame.vision

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
    /**
     * 非空 = 定位结果"能用但可疑"（网格残差偏大、board 置信度低、砖块太少）。
     *
     * 这类帧照样出提示，但要留档：它往往是"识别错了却看起来正常"的来源，
     * 事后补数据时比低置信度分类帧更值钱。
     */
    val suspect: String? = null,
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

    companion object {
        /** 从切图结果与每格类别构造；空位强制归零，避免检测器与识别结果打架。 */
        fun of(scan: BoardScan, labels: IntArray): BoardState {
            val cleaned = IntArray(labels.size) { if (scan.empty[it]) 0 else labels[it].coerceAtLeast(0) }
            return BoardState(scan.rows, scan.cols, cleaned)
        }
    }
}

/**
 * 一条可操作提示：[from] 与 [pair] 上的同款砖可以消掉（滑哪边、落到哪由游戏自己判定）。
 *
 * 两个格子直接来自 [com.qcmian.brickgame.algorithm.BrickGameTipHelper.findMoves]，
 * 不经过 `resolveAction` 补落点 —— 见 `HintEngine`。
 */
data class Hint(
    val from: CellPoint,
    val pair: CellPoint,
    val fromRect: CellRect,
    val pairRect: CellRect,
    /**
     * true = 这一对现在就能消（落点为空格）。
     *
     * false 表示两块呈 L 形、十字交叉点被别的砖占着：必须先挪开挡路的那块，
     * 属于两步操作。提示层照画，但要让人看得出区别。
     */
    val direct: Boolean = true,
) {
    /** "哪两块能消"的短描述（UI 上标对、日志里列对用这个）。 */
    fun pairText(): String = "(${from.row},${from.col})+(${pair.row},${pair.col})"
}
