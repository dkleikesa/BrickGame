package com.qcmian.brickgame.vision

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * `BrickGame-Tool/tile_recognizer.py` 里"纯数学"部分的 Kotlin 等价实现：把一维投影信号解成网格。
 *
 * 放在 commonMain 而不是 androidMain，是为了这些数值逻辑能在 JVM 上直接单测
 * （像素操作在 androidMain：定位由 YOLO 出网格，裁格走纯 Kotlin 的重采样）。
 *
 * 与 Python 版逐句对应：`detrend` / `fundamental_pitch` / `lattice_phase` / `longest_run` /
 * `refine_lines` / `trim_board_extent`。所有边界行为（取首个极值、`win|1` 强制奇数等）都保留。
 */
object LatticeMath {

    /**
     * 判断"一段砖块是一排真格子还是 UI 噪声"的密度线：真格子每格至少 2 块砖，
     * 而误判进来的状态栏/设置图标那种，一段里平均下来连 1.5 都到不了（真机实测 1.25）。
     */
    private const val MIN_RUN_DENSITY = 2.0

    /** numpy 默认（linear interpolation）的百分位。 */
    fun percentile(values: DoubleArray, q: Double): Double {
        if (values.isEmpty()) return 0.0
        val v = values.sortedArray()
        if (v.size == 1) return v[0]
        val pos = (q / 100.0) * (v.size - 1)
        val lo = pos.toInt()
        val hi = min(lo + 1, v.size - 1)
        val frac = pos - lo
        return v[lo] * (1 - frac) + v[hi] * frac
    }

    /**
     * 减去滑动平均 = 高通滤波。不做这步，投影里的缓慢趋势会让自相关在小滞后上就很高，
     * 周期会被误判成 min_pitch。
     *
     * 等价于 Python 的 `np.pad(sig, pad, "edge")` + 盒子卷积 + 取中段。**窗口越界时用边缘值补足，
     * 宽度恒为 win** —— 早期实现把窗口缩窄到有效区间再求均值，中间区域一致所以真棋盘测不出来，
     * 但低频占主导的画面（app 自身界面、无周期画面）会把自相关整个带偏，实测周期从 248 退化成 12。
     */
    fun detrend(sig: DoubleArray, win: Int): DoubleArray {
        val w = max(5, win or 1)          // |1 保证奇数，与 Python 一致
        val pad = w / 2
        val n = sig.size
        if (n == 0) return DoubleArray(0)
        val prefix = DoubleArray(n + 1)
        for (i in 0 until n) prefix[i + 1] = prefix[i] + sig[i]
        val out = DoubleArray(n)
        for (i in 0 until n) {
            val lo = i - pad
            val hi = i + pad
            var sum = prefix[hi.coerceAtMost(n - 1) + 1] - prefix[lo.coerceAtLeast(0)]
            if (lo < 0) sum += -lo * sig[0]                 // 越下界：复制首元素
            if (hi > n - 1) sum += (hi - (n - 1)) * sig[n - 1]  // 越上界：复制末元素
            out[i] = sig[i] - sum / w
        }
        return out
    }

    /**
     * 高通后做自相关，取最大峰；若是 2 倍谐波则折半到基波。
     *
     * @return 基波周期（像素）与归一化自相关曲线（非负滞后），后者只用于调试可视化。
     */
    fun fundamentalPitch(sig: DoubleArray, minPitch: Int, maxPitch: Int): Pair<Int, DoubleArray> {
        val x = detrend(sig, max(3 * minPitch, (sig.size * 0.12).toInt()))
        val n = x.size
        val ac = DoubleArray(n)
        for (lag in 0 until n) {
            var s = 0.0
            for (i in 0 until n - lag) s += x[i] * x[i + lag]
            ac[lag] = s
        }
        val a0 = ac[0] + 1e-9
        for (i in ac.indices) ac[i] /= a0

        val lo = max(2, minPitch)
        val hi = min(maxPitch, n - 1)
        val end = min(hi, n)
        if (end <= lo) return lo to ac

        var lag = lo
        var best = Double.NEGATIVE_INFINITY
        for (l in lo until end) {
            if (ac[l] > best) {
                best = ac[l]
                lag = l
            }
        }
        while (lag >= 2 * lo && ac[lag / 2] >= 0.6 * ac[lag]) lag /= 2
        return lag to ac
    }

    /**
     * 用"砖块是方的 → x/y 周期必然接近"这个先验，把取歪的大周期除回基波。
     *
     * 自相关取到 2/3 倍谐波时，两个周期会差出 2~3 倍（真机实测：行周期 307 = 3×102，
     * 走完一回合后行投影周期变弱，3 倍峰就赢了）。不修正的话网格会被切成棋盘左上角
     * 一小块（6×6），提示全错。
     *
     * @param tolerance 认为"除回去了"的相对容差
     * @return 修正后的 (px, py)；两边本来就接近、或除不回去时原样返回
     */
    fun reconcilePitches(
        px: Int,
        py: Int,
        tolerance: Double = 0.15,
        maxDivisor: Int = 4,
    ): Pair<Int, Int> {
        val small = min(px, py)
        val big = max(px, py)
        if (small <= 0 || big <= 1.35 * small) return px to py
        for (k in 2..maxDivisor) {
            if (abs(big / k - small) <= tolerance * small) {
                return if (px < py) px to big / k else big / k to py
            }
        }
        return px to py
    }

    /** 在给定周期下求相位：让网格线尽量落在强边上的那个偏移。 */
    fun latticePhase(prof: DoubleArray, pitch: Int): Int {
        val n = prof.size
        val p = detrend(prof, max(3 * pitch, (n * 0.12).toInt()))
        var bestPhase = 0
        var bestSum = Double.NEGATIVE_INFINITY
        for (ph in 0 until pitch) {
            var s = 0.0
            var i = ph
            while (i < n) {
                s += p[i]
                i += pitch
            }
            if (s > bestSum) {
                bestSum = s
                bestPhase = ph
            }
        }
        return bestPhase
    }

    /** 连续为 true 的最长区段 `[start, end)`；全 false 时返回 `(0, 0)`。 */
    fun longestRun(mask: BooleanArray): Pair<Int, Int> {
        var bestStart = 0
        var bestEnd = 0
        var cur = -1
        for (i in mask.indices) {
            val ok = mask[i]
            if (ok && cur < 0) cur = i
            if ((!ok || i == mask.size - 1) && cur >= 0) {
                val end = if (ok) i + 1 else i
                if (end - cur > bestEnd - bestStart) {
                    bestStart = cur
                    bestEnd = end
                }
                cur = -1
            }
        }
        return bestStart to bestEnd
    }

    /**
     * 把每条网格线放到"缝隙中心"，再做亚像素精调与最小二乘拟合。
     *
     * 为什么必须放到缝隙中心：相邻两块砖的描边会形成一对很近的峰（实测间隔 9px，格子间距 164px），
     * 相位锁定可能落在其中任意一个峰上，格子边界就偏了半个缝宽。
     */
    fun refineLines(prof: DoubleArray, lines: DoubleArray, pitch: Int): DoubleArray {
        val n = prof.size
        val half = max(2, pitch / 5)
        val ks = ArrayList<Double>()
        val ys = ArrayList<Double>()

        for (k in lines.indices) {
            val y = lines[k].toInt()
            val lo = max(0, y - half)
            val hi = min(n, y + half + 1)
            if (hi - lo < 3) continue
            val seg = DoubleArray(hi - lo) { prof[lo + it] }
            val segMax = seg.max()
            val peaks = ArrayList<Int>()
            for (i in 1 until seg.size - 1) {
                if (seg[i] >= seg[i - 1] && seg[i] >= seg[i + 1] && seg[i] > 0.35 * segMax) peaks.add(i)
            }

            val pos = when {
                peaks.size >= 2 -> {
                    val top = peaks.sortedByDescending { seg[it] }.take(2).sorted()
                    val a = top[0]
                    val b = top[1]
                    var v = a
                    for (i in a..b) if (seg[i] < seg[v]) v = i
                    var p = (lo + v).toDouble()
                    if (v > 0 && v < seg.size - 1) {            // 抛物线插值到亚像素
                        val d = seg[v - 1] - 2 * seg[v] + seg[v + 1]
                        if (abs(d) > 1e-6) {
                            p += (0.5 * (seg[v - 1] - seg[v + 1]) / d).coerceIn(-0.5, 0.5)
                        }
                    }
                    p
                }

                peaks.isNotEmpty() -> {
                    val i = peaks[0]
                    val thr = 0.5 * seg[i]
                    var a = i
                    while (a - 1 >= 0 && seg[a - 1] >= thr) a--
                    var b = i
                    while (b + 1 < seg.size && seg[b + 1] >= thr) b++
                    var sum = 0.0
                    var weighted = 0.0
                    for (j in a..b) {
                        sum += seg[j]
                        weighted += seg[j] * (j - a)
                    }
                    lo + a + weighted / (sum + 1e-9)
                }

                else -> lines[k]
            }
            ks.add(k.toDouble())
            ys.add(pos)
        }

        if (ks.size < 3) return lines.copyOf()

        var xs = ks.toDoubleArray()
        var vs = ys.toDoubleArray()
        val tol = max(1.5, pitch * 0.08)
        var round = 0
        while (round < 2) {                               // 去掉偏差大的点后重拟合
            round++
            val (slope, intercept) = linearFit(xs, vs) ?: break
            var keep = 0
            for (i in vs.indices) {
                if (abs(vs[i] - (intercept + slope * xs[i])) <= tol) keep++
            }
            if (keep == vs.size || keep < 2) break
            val nx = DoubleArray(keep)
            val nv = DoubleArray(keep)
            var k = 0
            for (i in vs.indices) {
                if (abs(vs[i] - (intercept + slope * xs[i])) <= tol) {
                    nx[k] = xs[i]
                    nv[k] = vs[i]
                    k++
                }
            }
            xs = nx
            vs = nv
        }

        val (slope, intercept) = linearFit(xs, vs) ?: return lines.copyOf()
        return DoubleArray(lines.size) { intercept + slope * it }
    }

    /** 最小二乘直线拟合，返回 (斜率, 截距)；点数不足返回 null。 */
    private fun linearFit(x: DoubleArray, y: DoubleArray): Pair<Double, Double>? {
        val n = x.size
        if (n < 2) return null
        var sx = 0.0
        var sy = 0.0
        var sxy = 0.0
        var sxx = 0.0
        for (i in 0 until n) {
            sx += x[i]
            sy += y[i]
            sxy += x[i] * y[i]
            sxx += x[i] * x[i]
        }
        val denom = n * sxx - sx * sx
        if (abs(denom) < 1e-12) return null
        val slope = (n * sxy - sx * sy) / denom
        return slope to (sy - slope * sx) / n
    }

    /**
     * 在砖块掩码上取"最长连续段"，再去掉两端明显偏稀的行/列。
     *
     * 为什么要裁两端：棋盘上方那条草地条带偶尔会有 1~2 格被误判成砖块，
     * 而真棋盘每行至少有"中位数 30%"的砖块。
     */
    fun trimBoardExtent(counts: IntArray): Pair<Int, Int> {
        // 取"砖块成片"的区段，跨过整列/整行已消完的空洞（见 Python 侧同名函数的注释）：
        // 只取最长连续段会把残局里"中间整列消完"的棋盘切窄（真机第 1 层：10 列棋盘第 7 列
        // 全空 → 切成 14×6 → 被尺寸闸门当离谱网格丢掉整帧），所以相邻的成片段要并回来；
        // 但棋盘上方的 UI 噪声（设置图标/状态栏）密度很低，只并"每格平均 ≥2 块砖"的段。
        val runs = ArrayList<IntArray>()                    // [start, end)
        var i = 0
        while (i < counts.size) {
            if (counts[i] < 1) {
                i++
                continue
            }
            var j = i
            while (j < counts.size && counts[j] >= 1) j++
            runs.add(intArrayOf(i, j))
            i = j
        }
        if (runs.isEmpty()) return 0 to 0
        val main = runs.maxByOrNull { r -> (r[0] until r[1]).sumOf { counts[it] } }!!
        val kept = ArrayList<IntArray>().apply { add(main) }
        for (r in runs) {
            if (r === main) continue
            val near = kept.any { k -> r[0] - k[1] <= 1 || k[0] - r[1] <= 1 }
            val density = (r[0] until r[1]).sumOf { counts[it] }.toDouble() / (r[1] - r[0])
            if (near && density >= MIN_RUN_DENSITY) kept.add(r)
        }
        var a = kept.minOf { it[0] }
        var b = kept.maxOf { it[1] }
        if (b - a < 2) return a to b
        val med = percentile(
            DoubleArray(b - a) { counts[a + it].toDouble() },
            50.0,
        )
        while (a < b - 2 && counts[a] < 0.30 * med) a++
        while (b - a > 2 && counts[b - 1] < 0.30 * med) b--
        return a to b
    }

    /**
     * 从候选格距里挑"能把棋盘外框铺成整格"的那个（见 Python 侧 `_pick_pitch`）。
     *
     * 棋盘是整格铺满底板的：正确格距下 span/pitch 必然接近整数（实测误差 ≤2%），
     * 错的候选会差出小半格（1102/94=11.72 差 0.28、1102/110=10.02 差 0.02）。
     * 这是唯一能把"自相关给 94"和"色块间距给 110"分出对错的判据 —— 两者都不能盲信：
     * 残局里相邻砖块会连成一片（色块只剩个位数），自相关又可能整轴没周期或取到偏周期。
     *
     * @return 选中的格距；候选全都不可用（凑不出 3~30 格）时返回 0。
     */
    fun pickPitch(candidates: List<Double>, span: Double, lo: Int = 3, hi: Int = 30): Double {
        var best = 0.0
        var bestErr = 9.9
        for (p in candidates) {
            if (p <= 0) continue
            val n = (span / p).roundToInt()
            if (n < lo || n > hi) continue
            val err = abs(span / p - n)
            if (err < bestErr) {
                best = p
                bestErr = err
            }
        }
        return best
    }

    /** `np.arange(start, stop, step)` 的等价：半开区间，步长为正。 */
    fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        if (step <= 0 || start >= stop) return DoubleArray(0)
        val n = ((stop - start - 1e-9) / step).toInt() + 1
        return DoubleArray(n) { start + it * step }
    }
}
