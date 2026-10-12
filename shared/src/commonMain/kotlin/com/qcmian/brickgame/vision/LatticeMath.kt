package com.qcmian.brickgame.vision

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * `BrickGame-Tool/tile_recognizer.py` 里"纯数学"部分的 Kotlin 等价实现：把一维投影信号解成网格。
 *
 * 放在 commonMain 而不是 androidMain，是为了这些数值逻辑能在 JVM 上直接单测
 * （OpenCV 只在 androidMain 里做灰度/Sobel/缩放这些像素操作，见 `OpenCvBoardScanner`）。
 *
 * 与 Python 版逐句对应：`detrend` / `fundamental_pitch` / `lattice_phase` / `longest_run` /
 * `refine_lines` / `trim_board_extent`。所有边界行为（取首个极值、`win|1` 强制奇数等）都保留。
 */
object LatticeMath {

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
        val mask = BooleanArray(counts.size) { counts[it] >= 1 }
        val (a0, b0) = longestRun(mask)
        if (b0 - a0 < 2) return a0 to b0
        val med = percentile(
            DoubleArray(b0 - a0) { counts[a0 + it].toDouble() },
            50.0,
        )
        var a = a0
        var b = b0
        while (a < b - 2 && counts[a] < 0.30 * med) a++
        while (b - a > 2 && counts[b - 1] < 0.30 * med) b--
        return a to b
    }

    /** `np.arange(start, stop, step)` 的等价：半开区间，步长为正。 */
    fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        if (step <= 0 || start >= stop) return DoubleArray(0)
        val n = ((stop - start - 1e-9) / step).toInt() + 1
        return DoubleArray(n) { start + it * step }
    }
}
