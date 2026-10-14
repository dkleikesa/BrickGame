package com.qcmian.brickgame.vision

/**
 * 流水线的最后一环：把"切图 + 识别"得到的盘面解成"哪两块能消"。
 *
 * 规则（对着真机游戏逐项核对出来的）：
 *  1. **只朝右、下、右下三个方向走**（也就是说：配对砖必须在待移动那块砖的右边或下边，
 *     含右下斜角）—— 真机上把左上方那块拖向右下方那一块能消，反着拖（往左/往上）消不了；
 *  2. 两块之间**只经过空格**（贴在一起也算），中间隔着别的砖就消不了；
 *  3. 两块必须同款（同款由识别层保证，见 `TfliteTileClassifier`：低置信格子会先在帧内
 *     找最像的格子纠错，纠不出来才判"未知"、不参与配对）。
 *
 * 这里换掉了 `BrickGameTipHelper`（lib2.jar 移植那套）：那套要求"落点必须是 from×pair 矩形的角"，
 * 是**另一个玩法**的规则，在这款游戏上会漏掉大片可消对、还会给出执行不了的动作（比如往左上滑）。
 * 识别与标定跟它无关，不受影响。
 */
object HintEngine {

    /**
     * 只允许这三个方向：右、下、右下。
     *
     * 真机核对：把左上方那块拖向右下方那一块能消；反过来（往左/往上/左下/右上）拖不动。
     * 所以配对砖必须落在待移动砖的"右边或下边"，扫描时只朝这三个方向看。
     */
    private val DIRS = arrayOf(
        intArrayOf(0, 1),
        intArrayOf(1, 0),
        intArrayOf(1, 1),
    )

    /** 全部可消除的对。 */
    fun hints(scan: BoardScan, state: BoardState): List<Hint> =
        pairs(state.asGrid()).map { toHint(scan, it) }

    /** 与 [hints] 同义；悬浮窗要一次把所有对都标出来。 */
    fun allHints(scan: BoardScan, state: BoardState): List<Hint> = hints(scan, state)

    /** 第一条可消除的对。 */
    fun nextHint(scan: BoardScan, state: BoardState): Hint? =
        pairs(state.asGrid()).firstOrNull()?.let { toHint(scan, it) }

    /** 盘面已清空。 */
    fun isCleared(state: BoardState): Boolean = state.labels.all { it == 0 }

    /** 还剩砖块但已无解 —— 说明识别出错（同类不同号/漏判空位），而不是真死了。 */
    fun isStuck(state: BoardState): Boolean =
        !isCleared(state) && pairs(state.asGrid()).isEmpty()

    /**
     * 可消除的格子对（无序，每对只出现一次，按 from 的行优先顺序排列）。
     *
     * 做法：每块砖朝右/下/右下三个方向各看一眼，一路跳过空格，撞上的第一块砖若同款就是一对。
     * 输出的 `first` 是左上那块（要拖的那块），`second` 是右下那块（配对砖）。
     */
    fun pairs(grid: Array<IntArray>): List<Pair<CellPoint, CellPoint>> {
        val rows = grid.size
        val cols = if (rows == 0) 0 else grid[0].size
        if (rows == 0 || cols == 0) return emptyList()

        val seen = HashSet<Long>()
        val out = ArrayList<Pair<CellPoint, CellPoint>>()
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val type = grid[r][c]
                if (type == 0) continue
                val here = CellPoint(r, c)
                for (d in DIRS) {
                    var nr = r + d[0]
                    var nc = c + d[1]
                    // 一路滑过空格，撞上第一块砖（或出界）为止
                    while (nr in 0 until rows && nc in 0 until cols && grid[nr][nc] == 0) {
                        nr += d[0]
                        nc += d[1]
                    }
                    if (nr !in 0 until rows || nc !in 0 until cols) continue
                    if (grid[nr][nc] == type) addPair(out, seen, cols, here, CellPoint(nr, nc))
                }
            }
        }
        out.sortBy { it.first.row * cols + it.first.col }
        return out
    }

    private fun addPair(
        out: MutableList<Pair<CellPoint, CellPoint>>,
        seen: MutableSet<Long>,
        cols: Int,
        a: CellPoint,
        b: CellPoint,
    ) {
        val ia = a.row * cols + a.col
        val ib = b.row * cols + b.col
        val lo = minOf(ia, ib)
        val hi = maxOf(ia, ib)
        if (seen.add(lo.toLong() shl 32 or hi.toLong())) {
            out += (if (ia <= ib) a to b else b to a)
        }
    }

    /** 规则里没有"落点"这回事（玩家只要把两块同款砖滑到一起/拖到一起），`to` 与 `from` 相同。 */
    private fun toHint(scan: BoardScan, pair: Pair<CellPoint, CellPoint>): Hint = Hint(
        from = pair.first,
        to = pair.first,
        pair = pair.second,
        fromRect = scan.rectOf(pair.first),
        toRect = scan.rectOf(pair.first),
        pairRect = scan.rectOf(pair.second),
    )
}
