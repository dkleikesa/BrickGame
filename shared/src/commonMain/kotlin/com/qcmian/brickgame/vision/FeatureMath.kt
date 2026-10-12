package com.qcmian.brickgame.vision

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 一帧所有格子的裁切图，行优先排列。
 *
 * [pixels] 是 `rows*cols*cell*cell*3` 的浮点像素，取值 0..255，**通道顺序固定为 BGR**
 * （与 OpenCV / `tile_recognizer.py` 的 `cv2.imread` 一致，`cellFlatness` 的灰度权重依赖它）。
 */
class TileFrame(
    val rows: Int,
    val cols: Int,
    val cell: Int,
    val pixels: FloatArray,
) {
    val count: Int get() = rows * cols
    val valuesPerPatch: Int get() = cell * cell * 3

    fun patchOffset(index: Int): Int = index * valuesPerPatch

    init {
        require(pixels.size == count * valuesPerPatch) {
            "像素数量不符：期望 ${count * valuesPerPatch}，实际 ${pixels.size}"
        }
    }
}

/** [FeatureMath.iconFeatures] 的结果：去底图标特征与图标像素占比（占比 ≈0 即为空位）。 */
class IconFeatures(val dev: FloatArray, val ratio: FloatArray)

/**
 * `tile_recognizer.py` 里"图标特征 + 帧内聚类"部分的 Kotlin 等价实现。
 *
 * 这一段是纯计算（没有像素采样），所以放在 commonMain 供 JVM 单测；
 * OpenCV 只负责在 androidMain 里把截图裁成 [TileFrame]。
 */
object FeatureMath {

    private const val CELL = 32
    private const val BORDER = 3

    /** 32×32 图标的 3px 外框掩码，用来取"底板色"。 */
    private val BORDER_MASK: BooleanArray = BooleanArray(CELL * CELL).also { m ->
        for (y in 0 until CELL) {
            for (x in 0 until CELL) {
                if (y < BORDER || y >= CELL - BORDER || x < BORDER || x >= CELL - BORDER) {
                    m[y * CELL + x] = true
                }
            }
        }
    }

    /**
     * 返回 (图标特征, 图标占比)。
     *
     * 做法：把"底板像素"**直接挖掉（置 0）**，只保留图标像素本身的绝对颜色。
     * 不能只"减底板色"——底板色是砖块状态（冻结/选中会让底板变蓝），同一图标底板不同时，
     * 减去各自底板色后图标区与背景区的偏差方向相反，空间上的符号翻转抵消不掉，
     * 实测同类距离仍有 0.12~0.21，聚类就开始乱拆。挖掉背景后同类距离回到 ~0。
     */
    fun iconFeatures(frame: TileFrame, tol: Double = 40.0, erode: Int = 1): IconFeatures {
        val n = frame.count
        val dev = FloatArray(frame.pixels.size)
        val ratio = FloatArray(n)
        val cell = frame.cell
        val px = cell * cell
        val borderMask = borderMaskFor(cell)

        val samples = FloatArray(px)

        for (idx in 0 until n) {
            val off = frame.patchOffset(idx)
            val bg = FloatArray(3)
            for (c in 0 until 3) {
                var k = 0
                for (p in 0 until px) {
                    if (borderMask[p]) samples[k++] = frame.pixels[off + p * 3 + c]
                }
                bg[c] = median(samples, k)
            }

            val mask = BooleanArray(px)
            for (p in 0 until px) {
                val db = frame.pixels[off + p * 3] - bg[0]
                val dg = frame.pixels[off + p * 3 + 1] - bg[1]
                val dr = frame.pixels[off + p * 3 + 2] - bg[2]
                mask[p] = sqrt(db * db + dg * dg + dr * dr) > tol
            }

            var m = mask
            repeat(erode) { m = erode3x3(m, cell) }

            var on = 0
            for (p in 0 until px) {
                if (m[p]) {
                    on++
                    val d = off + p * 3
                    dev[d] = frame.pixels[d]
                    dev[d + 1] = frame.pixels[d + 1]
                    dev[d + 2] = frame.pixels[d + 2]
                }
            }
            ratio[idx] = on.toFloat() / px
        }

        return IconFeatures(dev, ratio)
    }

    private fun borderMaskFor(cell: Int): BooleanArray {
        if (cell == CELL) return BORDER_MASK
        return BooleanArray(cell * cell).also { m ->
            val w = max(1, cell * BORDER / CELL)
            for (y in 0 until cell) {
                for (x in 0 until cell) {
                    if (y < w || y >= cell - w || x < w || x >= cell - w) m[y * cell + x] = true
                }
            }
        }
    }

    private fun median(buf: FloatArray, len: Int): Float {
        if (len == 0) return 0f
        val v = buf.copyOf(len)
        v.sort()
        return if (len % 2 == 1) v[len / 2] else 0.5f * (v[len / 2 - 1] + v[len / 2])
    }

    /**
     * 3×3 最小值滤波。OpenCV 形态学的默认边界值是 +inf（腐蚀时不影响结果），
     * 所以这里只对"落在图内"的邻居取最小值，相当于边缘不做任何扩张。
     */
    private fun erode3x3(mask: BooleanArray, cell: Int): BooleanArray {
        val out = BooleanArray(mask.size)
        for (y in 0 until cell) {
            for (x in 0 until cell) {
                var all = true
                for (dy in -1..1) {
                    val yy = y + dy
                    if (yy < 0 || yy >= cell) continue
                    for (dx in -1..1) {
                        val xx = x + dx
                        if (xx < 0 || xx >= cell) continue
                        if (!mask[yy * cell + xx]) {
                            all = false
                            break
                        }
                    }
                    if (!all) break
                }
                out[y * cell + x] = all
            }
        }
        return out
    }

    /**
     * 每格"内部有没有东西"：格子内部（去掉外圈 margin）的 Sobel 边缘能量均值。
     *
     * 为什么不用"图标对比度"判空：已消除的格子只剩棋盘底色 → 内部完全平坦
     * （实测边缘能量 0.0~0.8）；任何真砖块内部都有结构（实测最低 31）。
     * 两者相差 40 倍且无重叠，所以用"平不平"判空稳得多。
     */
    fun cellFlatness(frame: TileFrame, margin: Double = 0.18): FloatArray {
        val cell = frame.cell
        val out = FloatArray(frame.count)
        val gray = DoubleArray(cell * cell)

        val lo = (cell * margin).toInt()
        val hi = (cell * (1 - margin)).toInt()
        val iw = hi - lo
        if (iw <= 0) return out

        for (idx in 0 until frame.count) {
            val off = frame.patchOffset(idx)
            for (p in 0 until cell * cell) {
                val d = off + p * 3
                // OpenCV COLOR_BGR2GRAY 权重；Python 是 `p.astype(np.uint8)`（截断）后再转灰度，
                // 所以这里同样截断而不是四舍五入。
                gray[p] = (
                    0.114 * frame.pixels[d] +
                        0.587 * frame.pixels[d + 1] +
                        0.299 * frame.pixels[d + 2]
                    ).toInt().toDouble()
            }

            var sum = 0.0
            for (y in 0 until iw) {
                for (x in 0 until iw) {
                    val gx = sobelX(gray, cell, lo + x, lo + y)
                    val gy = sobelY(gray, cell, lo + x, lo + y)
                    sum += sqrt(gx * gx + gy * gy)
                }
            }
            out[idx] = (sum / (iw * iw)).toFloat()
        }
        return out
    }

    // Sobel ksize=3：x 方向列核 [-1,0,1] × 行权 [1,2,1]；y 方向行核 [-1,0,1] × 列权 [1,2,1]。
    // 边界按 BORDER_REFLECT_101 取。只取模长，所以整体符号不影响结果。
    private val SOBEL_COL = intArrayOf(-1, 0, 1)
    private val SOBEL_ROW = intArrayOf(-1, 0, 1)
    private val SOBEL_W = intArrayOf(1, 2, 1)

    private fun sobelX(g: DoubleArray, cell: Int, x: Int, y: Int): Double {
        var s = 0.0
        for (j in -1..1) {
            for (i in -1..1) {
                s += SOBEL_COL[i + 1] * SOBEL_W[j + 1] * at(g, cell, x + i, y + j)
            }
        }
        return s
    }

    private fun sobelY(g: DoubleArray, cell: Int, x: Int, y: Int): Double {
        var s = 0.0
        for (j in -1..1) {
            for (i in -1..1) {
                s += SOBEL_W[i + 1] * SOBEL_ROW[j + 1] * at(g, cell, x + i, y + j)
            }
        }
        return s
    }

    /** BORDER_REFLECT_101：`fedcba|abcdefgh|hgfedcb`。 */
    private fun at(g: DoubleArray, cell: Int, x: Int, y: Int): Double {
        return g[reflect101(y, cell) * cell + reflect101(x, cell)]
    }

    private fun reflect101(v: Int, n: Int): Int {
        if (n == 1) return 0
        var r = v
        while (r < 0 || r >= n) {
            if (r < 0) r = -r else r = 2 * n - 2 - r
        }
        return r
    }

    /**
     * 图标特征两两距离，带 ±rad 像素的平移搜索。
     *
     * 这一步是关键：网格线拟合再准也会有亚像素残差，同一个图标在不同行列裁出来会差 1~2 像素，
     * 实测未对齐时同类距离能到 0.58，对齐后 0.002。先用余弦距离粗筛，
     * 只对可能同类的少数格子对做平移搜索，速度也够。
     *
     * @param dev 点阵特征（长度 n*cell*cell*3），只应传入"有砖块"的格子。
     */
    fun alignedDistanceMatrix(
        dev: FloatArray,
        n: Int,
        cell: Int,
        rad: Int = 3,
        prefilter: Double = 0.85,
    ): Array<DoubleArray> {
        val dim = cell * cell * 3
        val norm = DoubleArray(n)
        for (i in 0 until n) {
            var s = 0.0
            val off = i * dim
            for (k in 0 until dim) s += dev[off + k].toDouble() * dev[off + k]
            norm[i] = sqrt(s) + 1e-8
        }

        val d = Array(n) { DoubleArray(n) }
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                var dot = 0.0
                val oi = i * dim
                val oj = j * dim
                for (k in 0 until dim) dot += dev[oi + k].toDouble() * dev[oj + k]
                val dist = (1.0 - dot / (norm[i] * norm[j])).coerceIn(0.0, 2.0)
                d[i][j] = dist
                d[j][i] = dist
            }
        }

        for (i in 0 until n) {
            for (j in i + 1 until n) {
                if (d[i][j] > prefilter) continue
                val v = shiftAlignedDistance(dev, i, j, cell, rad)
                d[i][j] = v
                d[j][i] = v
            }
        }
        return d
    }

    /**
     * `cv2.matchTemplate(pad(i), j, TM_SQDIFF_NORMED).min()`：±rad 内的归一化平方差最小值。
     *
     * 注意 OpenCV 把结果**截断在 1.0**（实测 `Σ(T-I)²/sqrt(ΣT²ΣI²)` = 4.0 时它给 1.0），
     * 这里必须跟着截断，否则完全不同的两块砖会算出 >1 的距离，把下游阈值全带偏。
     */
    private fun shiftAlignedDistance(dev: FloatArray, i: Int, j: Int, cell: Int, rad: Int): Double {
        val dim = cell * cell * 3
        val oi = i * dim
        val oj = j * dim

        var sumT2 = 0.0
        for (k in 0 until dim) {
            val v = dev[oj + k].toDouble()
            sumT2 += v * v
        }

        var best = Double.MAX_VALUE
        for (dy in -rad..rad) {
            for (dx in -rad..rad) {
                var num = 0.0
                var sumI2 = 0.0
                for (y in 0 until cell) {
                    val sy = (y + dy).coerceIn(0, cell - 1)
                    for (x in 0 until cell) {
                        val sx = (x + dx).coerceIn(0, cell - 1)
                        val a = oi + (sy * cell + sx) * 3
                        val b = oj + (y * cell + x) * 3
                        for (c in 0 until 3) {
                            val av = dev[a + c].toDouble()
                            val bv = dev[b + c].toDouble()
                            sumI2 += av * av
                            val diff = av - bv
                            num += diff * diff
                        }
                    }
                }
                val denom = sqrt(sumT2 * sumI2) + 1e-9
                val r = num / denom
                if (r < best) best = r
            }
        }
        return min(1.0, best)
    }

    /**
     * 距离分布里最大的那个"空隙"的中点。
     *
     * 同类距离实测都挤在 0~0.3，异类在 0.8 以上，中间是空旷地带。
     * 这个值用来给合并设上界——没有上界的话，一旦出现某个奇数大小的连通分量，
     * 聚类会一路合并到几乎全并成一个簇（实测从 12 个簇塌成 1 个）。
     */
    fun gapThreshold(dist: Array<DoubleArray>, lo: Double = 0.02, hi: Double = 0.70): Double {
        val values = ArrayList<Double>()
        for (i in dist.indices) for (j in i + 1 until dist.size) values.add(dist[i][j])
        if (values.size < 2) return hi
        values.sort()

        var bestGap = 0.0
        var bestIdx = 0
        for (i in 0 until values.size - 1) {
            if (values[i] >= lo && values[i] <= hi) {
                val g = values[i + 1] - values[i]
                if (g > bestGap) {
                    bestGap = g
                    bestIdx = i
                }
            }
        }
        return 0.5 * (values[bestIdx] + values[bestIdx + 1])
    }

    /** [clusterByParity] 的结果。 */
    class Clustering(val labels: IntArray, val threshold: Double, val evenSatisfied: Boolean)

    /**
     * Kruskal 式按距离升序加边，维护"奇数大小簇的个数"，首次降到 0 即为阈值。
     *
     * 直接用游戏规则当目标：一步消两块 → 每类数量必须偶数。这比在距离直方图里找空隙稳得多。
     */
    fun clusterByParity(dist: Array<DoubleArray>, maxDist: Double = 0.9): Clustering {
        val n = dist.size
        if (n == 0) return Clustering(IntArray(0), 0.0, true)

        val parent = IntArray(n) { it }
        val size = IntArray(n) { 1 }
        var odd = n

        fun find(a0: Int): Int {
            var a = a0
            while (parent[a] != a) {
                parent[a] = parent[parent[a]]
                a = parent[a]
            }
            return a
        }

        val edges = ArrayList<IntArray>()
        val order = ArrayList<Long>()
        for (i in 0 until n) for (j in i + 1 until n) order.add(i.toLong() shl 32 or j.toLong())
        order.sortBy { dist[(it shr 32).toInt()][(it and 0xFFFFFFFFL).toInt()] }

        var bestOdd = Int.MAX_VALUE
        var bestNegCls = 0
        var bestEdges = 0
        var bestThr = 0.0
        var thr = 0.0

        for (key in order) {
            val i = (key shr 32).toInt()
            val j = (key and 0xFFFFFFFFL).toInt()
            val d = dist[i][j]
            if (d <= 1e-6 || d > maxDist) continue
            val ri = find(i)
            val rj = find(j)
            if (ri == rj) continue

            if (size[ri] % 2 == 1) odd--
            if (size[rj] % 2 == 1) odd--
            parent[ri] = rj
            size[rj] += size[ri]
            if (size[rj] % 2 == 1) odd++
            edges.add(intArrayOf(ri, rj))
            thr = d

            var nCls = 0
            for (k in 0 until n) if (find(k) == k) nCls++
            if (bestOdd == Int.MAX_VALUE || odd < bestOdd || (odd == bestOdd && -nCls < bestNegCls)) {
                bestOdd = odd
                bestNegCls = -nCls
                bestEdges = edges.size
                bestThr = d
            }
            if (odd == 0) break
        }

        if (bestOdd == Int.MAX_VALUE) {
            // 没有任何一对进入候选距离：退化成"一格一类"
            val labels = IntArray(n) { it + 1 }
            return Clustering(labels, 0.0, false)
        }

        // 偶数约束可能永远满足不了（比如有格子被误判成空位，总数成了奇数）。
        // 这时候必须回退到"奇数类个数最少"的那个中间状态，否则会一路合并成一个大簇。
        val fresh = IntArray(n) { it }
        val fp = fresh
        fun find2(a0: Int): Int {
            var a = a0
            while (fp[a] != a) {
                fp[a] = fp[fp[a]]
                a = fp[a]
            }
            return a
        }
        for (k in 0 until bestEdges) {
            fp[edges[k][0]] = edges[k][1]
        }

        val rootToLabel = HashMap<Int, Int>()
        val labels = IntArray(n)
        for (i in 0 until n) {
            val r = find2(i)
            val l = rootToLabel.getOrPut(r) { rootToLabel.size + 1 }
            labels[i] = l
        }
        return Clustering(labels, bestThr, bestOdd == 0)
    }

    /**
     * 把"明显还是同类"的簇对并回去。
     *
     * 为什么需要这一步：偶数约束的合并是"贪心凑偶数"，凑够偶数时立刻停手。
     * 一个真实 4 块的一类，如果其中一块状态不同（底色变了导致距离略大），
     * 贪心会先配掉更近的两块，剩下两块凑成另一对 → 被拆成 2+2，求解器就丢掉了跨子簇的解。
     */
    fun mergeSimilar(labels: IntArray, dist: Array<DoubleArray>, hard: Double): IntArray {
        val out = labels.copyOf()
        while (true) {
            val groups = LinkedHashMap<Int, MutableList<Int>>()
            for (i in out.indices) groups.getOrPut(out[i]) { ArrayList() }.add(i)
            if (groups.size < 2) return out

            val keys = groups.keys.sorted()
            var bestMean = Double.MAX_VALUE
            var bestA = -1
            var bestB = -1
            for (ai in keys.indices) {
                for (bi in ai + 1 until keys.size) {
                    val a = keys[ai]
                    val b = keys[bi]
                    val ga = groups[a]!!
                    val gb = groups[b]!!
                    if (ga.size % 2 != gb.size % 2) continue   // 合并后会变奇数，不能并
                    var sum = 0.0
                    for (x in ga) for (y in gb) sum += dist[x][y]
                    val mean = sum / (ga.size * gb.size)
                    if (mean < hard && mean < bestMean) {
                        bestMean = mean
                        bestA = a
                        bestB = b
                    }
                }
            }
            if (bestA < 0) return out
            for (i in out.indices) if (out[i] == bestA || out[i] == bestB) out[i] = bestA
        }
    }
}
