package com.qcmian.brickgame.vision

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 截图 → 棋盘网格 + 每格裁切图。是 `BrickGame-Tool/tile_recognizer.py` 里
 * `detect_grid` / `grid_from_lines` / `cut_cells` / `cell_ring_score` 的 Kotlin 移植。
 *
 * 只做"切图"，不认砖块类型（识别见 `TileClassifier`）；纯数学部分委托给 `LatticeMath`，
 * 这样那部分能在 JVM 上单测（见 `LatticeMathTest`）。
 *
 * 两个信号分工明确（Python 版踩过坑，这里照搬）：
 *  - **棋盘范围**用环带四边最小值：能顶住进度条/按钮这类横贯画面的 UI；
 *  - **空位判定**用图标内部边缘能量：空格子只剩棋盘底色 → 完全平坦（实测 0.0~0.8），
 *    任何真砖块内部都有结构（实测最低 31），两者相差 40 倍且无重叠。
 */
object OpenCvBoardScanner {

    private const val TAG = "OpenCvBoardScanner"

    /** 每格归一化后的边长，与 `tile_recognizer.CELL` 一致（聚类阈值是照它调的）。 */
    const val CELL = 32

    /** 裁切时向内收缩的比例（去掉格子边框/圆角）。 */
    const val INSET = 0.14

    /** 内部边缘能量低于它 = 平坦 = 已消除的空位。 */
    const val FLAT_THR = 5.0f

    /** 格子最小像素间距。 */
    const val MIN_PITCH = 12

    /**
     * 网格尺寸的合理范围。真实棋盘在 `build_dataset/frames.csv` 台账里是 **14 行 × 10 列**，
     * 当前关卡实测 **16 行 × 10 列** —— 列数稳定在 10，行数随关卡变，上下限按这个留余量。
     *
     * 为什么必须卡住上限：自相关会被画面内容带偏，实测拿到过 **199×10（1990 格）** 的假网格，
     * 而下游是 O(n²)（图标两两距离）甚至更差，那一帧跑了 **24.8 秒**，整个提示循环看起来就是"卡死"。
     * 尺寸离谱就直接当"没定位到棋盘"丢掉这一帧 —— 宁可不出提示，也不能卡住。
     *
     * 下限同样重要：周期取到谐波时会给出 **6×6** 这种"看着像棋盘"的网格，照样能算出一堆提示，
     * 但格子全是错的（真机实测：行周期 307 = 3×102，棋盘被切成左上角 6×6）。
     */
    const val MIN_ROWS = 8
    const val MAX_ROWS = 20
    const val MIN_COLS = 8
    const val MAX_COLS = 16

    private const val RING_LO = 0.04
    private const val RING_HI = 0.20
    private const val RING_KEEP_RATIO = 0.25

    private var nativeReady = false

    /** 幂等加载 OpenCV 的 .so；还没加载过时返回 false。 */
    fun ensureNativeLoaded(): Boolean {
        if (!nativeReady) nativeReady = OpenCVLoader.initLocal()
        return nativeReady
    }

    /**
     * 标定 + 切格。
     *
     * @return 定位失败（没找到成排的格子）时返回 null，调用方应当跳过这一帧。
     */
    fun scan(bitmap: Bitmap, minPitch: Int = MIN_PITCH, inset: Double = INSET): BoardScan? {
        ensureNativeLoaded()
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        try {
            when (rgba.channels()) {
                4 -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
                3 -> rgba.copyTo(bgr)
                else -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_GRAY2BGR)
            }
            return scanBgr(bgr, minPitch, inset)
        } finally {
            rgba.release()
            bgr.release()
        }
    }

    private fun scanBgr(img: Mat, minPitch: Int, inset: Double): BoardScan? {
        val h = img.rows()
        val w = img.cols()
        if (h < 32 || w < 32) return null

        val gray = Mat()
        val gx = Mat()
        val gy = Mat()
        val mag = Mat()
        val blurred = Mat()
        try {
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3)
            Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3)
            Core.magnitude(gx, gy, mag)
            Imgproc.GaussianBlur(mag, blurred, Size(3.0, 3.0), 0.0)

            val rowProf = rowSums(blurred)
            val colProf = colSums(blurred)

            val (pitchY, _) = LatticeMath.fundamentalPitch(rowProf, minPitch, (h * 0.25).toInt())
            val (pitchX, _) = LatticeMath.fundamentalPitch(colProf, minPitch, (w * 0.25).toInt())

            var xlines = LatticeMath.refineLines(
                colProf,
                LatticeMath.arange(LatticeMath.latticePhase(colProf, pitchX).toDouble(), w.toDouble(), pitchX.toDouble()),
                pitchX,
            )
            var ylines = LatticeMath.refineLines(
                rowProf,
                LatticeMath.arange(LatticeMath.latticePhase(rowProf, pitchY).toDouble(), h.toDouble(), pitchY.toDouble()),
                pitchY,
            )
            var px = pitchX
            var py = pitchY
            var fromBlobs = false
            var diag = ""

            // 砖块是正方形的 → x/y 周期必然接近。差得远说明自相关取歪了，两种都实测过：
            //  - 取到 2/3 倍谐波：行周期 307 = 3×102，棋盘被切成左上角 6×6，提示全错；
            //  - 被"空棋盘 + 天空/马路/UI 边缘"带偏：残局截图给 x23/y94，真值 110/110，砖块被切成半块。
            // 先按方形砖块先验把大周期除回基波；除不动再走棋盘底色 + 残余砖块间距反推；
            // 两条路都不认就丢掉这一帧 —— 宁可不出提示，也不能拿错网格给提示。
            val (fixedPx, fixedPy) = LatticeMath.reconcilePitches(px, py)
            // 修正必须给出"列数合理"的格距才算数：残局那种 x23/y94（真值 110/110）除一下
            // 也能凑成 23/23，但那样一屏会有 40+ 列，明显不是棋盘 —— 这种情况留给反推兜底。
            val harmonicPlausible =
                (fixedPx != px || fixedPy != py) && w / fixedPx in MIN_COLS..MAX_COLS
            if (harmonicPlausible) {
                px = fixedPx
                py = fixedPy
                // 用修正后的周期重新求相位 + 精调网格线
                xlines = LatticeMath.refineLines(
                    colProf,
                    LatticeMath.arange(
                        LatticeMath.latticePhase(colProf, px).toDouble(),
                        w.toDouble(),
                        px.toDouble(),
                    ),
                    px,
                )
                ylines = LatticeMath.refineLines(
                    rowProf,
                    LatticeMath.arange(
                        LatticeMath.latticePhase(rowProf, py).toDouble(),
                        h.toDouble(),
                        py.toDouble(),
                    ),
                    py,
                )
                diag = "谐波修正（$pitchX/$pitchY → $px/$py）"
                Log.i(TAG, "x/y 周期不一致，按方形砖块先验做谐波修正：$pitchX/$pitchY → $px/$py")
            } else if (max(px, py) > 1.35 * max(1, min(px, py))) {
                val lat = BlobLattice.estimate(img)
                if (lat != null) {
                    xlines = lat.xlines
                    ylines = lat.ylines
                    px = lat.pitchX
                    py = lat.pitchY
                    fromBlobs = true
                } else {
                    Log.w(TAG, "x/y 周期差太远（$pitchX/$pitchY）且棋盘反推失败 → 丢弃这一帧")
                    return null
                }
            }

            if (!fromBlobs) {
                // 自相关路径：网格线会溢出棋盘，用"砖块成排"的密度裁掉两端多余的行/列。
                val ring = cellRingScore(gx, gy, xlines, ylines)
                val ringRows = ylines.size - 1
                val ringCols = xlines.size - 1
                if (ringRows < 1 || ringCols < 1) return null

                val keepThreshold = RING_KEEP_RATIO * LatticeMath.percentile(ring, 95.0)
                val rowCounts = IntArray(ringRows)
                val colCounts = IntArray(ringCols)
                for (j in 0 until ringRows) {
                    for (i in 0 until ringCols) {
                        if (ring[j * ringCols + i] >= keepThreshold) {
                            rowCounts[j]++
                            colCounts[i]++
                        }
                    }
                }
                val (r0, r1) = LatticeMath.trimBoardExtent(rowCounts)
                val (c0, c1) = LatticeMath.trimBoardExtent(colCounts)
                diag += "环带阈值=${"%.2f".format(keepThreshold)} 行计数=${rowCounts.joinToString("/")} " +
                    "列计数=${colCounts.joinToString("/")} trim=[$r0,$r1]x[$c0,$c1]"
                if (r1 - r0 < 2 || c1 - c0 < 2) {
                    Log.w(TAG, "棋盘范围裁空了（trim=[$r0,$r1]x[$c0,$c1]）pitch=($px,$py) $diag → 丢弃这一帧")
                    return null
                }

                xlines = xlines.copyOfRange(c0, c1 + 1)
                ylines = ylines.copyOfRange(r0, r1 + 1)
            }
            // 反推路径不裁：棋盘外框已是实测值，而 trim 用的是"最长连续段"，
            // 残局里中间若干行全空，会把 10x14 的棋盘截成左上角那一小簇。

            val rows = ylines.size - 1
            val cols = xlines.size - 1
            // 标定诊断：这块画面在真机上出过 6x6，而参考实现是 16x10，靠这行日志对齐中间量
            Log.i(
                TAG,
                "标定：图像 ${w}x$h pitch=($pitchX,$pitchY) " +
                    (if (fromBlobs) "反推" else "自相关") +
                    " 精调线=${xlines.size - 1}列x${ylines.size - 1}行 → 裁后=${cols}列x${rows}行 $diag",
            )
            if (rows < MIN_ROWS || rows > MAX_ROWS || cols < MIN_COLS || cols > MAX_COLS) {
                Log.w(
                    TAG,
                    "网格尺寸不合理：${rows}x$cols（合理 ${MIN_ROWS}~$MAX_ROWS 行 / ${MIN_COLS}~$MAX_COLS 列）" +
                        "pitch=($px,$py) 自相关原始=($pitchX,$pitchY) $diag → 丢弃这一帧",
                )
                return null
            }

            val (rects, frame) = cutCells(img, xlines, ylines, inset)
            val score = FeatureMath.cellFlatness(frame)

            return BoardScan(
                rows = ylines.size - 1,
                cols = xlines.size - 1,
                xlines = xlines,
                ylines = ylines,
                score = score,
                scoreThreshold = FLAT_THR,
                rects = rects,
                pitchX = px,
                pitchY = py,
            )
        } finally {
            gray.release()
            gx.release()
            gy.release()
            mag.release()
            blurred.release()
        }
    }

    // ------------------------------------------------------------------ 投影

    /** `g.sum(axis=1)`：每行一个和，长度 = 行数。 */
    private fun rowSums(m: Mat): DoubleArray {
        val buf = FloatArray(m.rows() * m.cols())
        m.get(0, 0, buf)
        val out = DoubleArray(m.rows())
        for (y in 0 until m.rows()) {
            var s = 0.0
            val base = y * m.cols()
            for (x in 0 until m.cols()) s += buf[base + x]
            out[y] = s
        }
        return out
    }

    /** `g.sum(axis=0)`：每列一个和，长度 = 列数。 */
    private fun colSums(m: Mat): DoubleArray {
        val buf = FloatArray(m.rows() * m.cols())
        m.get(0, 0, buf)
        val out = DoubleArray(m.cols())
        for (y in 0 until m.rows()) {
            val base = y * m.cols()
            for (x in 0 until m.cols()) out[x] += buf[base + x]
        }
        return out
    }

    // ------------------------------------------------------------------ 环带打分

    /**
     * 格子四条边内侧环带的方向梯度，取**四边最小值** —— 专门用来判"棋盘范围"。
     *
     * 取四边最小值：真砖块四边都有圆角描边；棋盘横梁、进度条、UI、草地只有单边。
     * 按方向取：上下边看 |gy|，左右边看 |gx|，否则一条横贯画面的进度条在左右环带里
     * 也会因为"横向边"拿到高分，实测会把整行 UI 误判成一行砖块。
     */
    private fun cellRingScore(gx: Mat, gy: Mat, xlines: DoubleArray, ylines: DoubleArray): DoubleArray {
        val agx = Mat()
        val agy = Mat()
        try {
            Core.absdiff(gx, org.opencv.core.Scalar.all(0.0), agx)
            Core.absdiff(gy, org.opencv.core.Scalar.all(0.0), agy)

            val ny = ylines.size - 1
            val nx = xlines.size - 1
            val out = DoubleArray(ny * nx)
            for (j in 0 until ny) {
                // 与 Python 的 `int(ys[j])` 一致用截断（不是四舍五入）：环带边界的取值直接决定
                // "谁算棋盘内"，四舍五入会在半像素上偏移一格边框。
                val y0 = ylines[j].toInt()
                val y1 = ylines[j + 1].toInt()
                val hh = y1 - y0
                for (i in 0 until nx) {
                    val x0 = xlines[i].toInt()
                    val x1 = xlines[i + 1].toInt()
                    val ww = x1 - x0
                    var best = Double.MAX_VALUE
                    var any = false
                    val means = arrayOf(
                        meanRegion(agy, y0 + (hh * RING_LO).toInt(), y0 + (hh * RING_HI).toInt(), x0, x1),
                        meanRegion(agy, y1 - (hh * RING_HI).toInt(), y1 - (hh * RING_LO).toInt(), x0, x1),
                        meanRegion(agx, y0, y1, x0 + (ww * RING_LO).toInt(), x0 + (ww * RING_HI).toInt()),
                        meanRegion(agx, y0, y1, x1 - (ww * RING_HI).toInt(), x1 - (ww * RING_LO).toInt()),
                    )
                    for (v in means) {
                        if (v != null) {
                            any = true
                            if (v < best) best = v
                        }
                    }
                    out[j * nx + i] = if (any) best else 0.0
                }
            }
            return out
        } finally {
            agx.release()
            agy.release()
        }
    }

    /** 子区域均值（自动裁剪到图内）；区域为空返回 null。 */
    private fun meanRegion(m: Mat, y0: Int, y1: Int, x0: Int, x1: Int): Double? {
        val yy0 = min(max(y0, 0), m.rows())
        val yy1 = min(max(y1, 0), m.rows())
        val xx0 = min(max(x0, 0), m.cols())
        val xx1 = min(max(x1, 0), m.cols())
        if (yy1 <= yy0 || xx1 <= xx0) return null
        return Core.mean(m.submat(yy0, yy1, xx0, xx1)).`val`[0]
    }

    // ------------------------------------------------------------------ 裁格

    /**
     * 以每格中心对称裁一个统一尺寸的方块 —— 尺寸一致才好逐像素比对。
     *
     * @return 原图坐标系下的格子矩形 + 归一化到 [CELL]×[CELL] 的裁切图（BGR 浮点）。
     */
    private fun cutCells(
        img: Mat,
        xlines: DoubleArray,
        ylines: DoubleArray,
        inset: Double,
    ): Pair<Array<CellRect>, TileFrame> {
        val nx = xlines.size - 1
        val ny = ylines.size - 1

        val pitches = DoubleArray(nx + ny)
        for (i in 0 until nx) pitches[i] = xlines[i + 1] - xlines[i]
        for (j in 0 until ny) pitches[nx + j] = ylines[j + 1] - ylines[j]
        var side = (median(pitches) * (1 - 2 * inset)).roundToInt()
        side = max(8, side)
        // 裁格用的 side 与 Python 的 `int(round(...))` 一致走四舍五入；上面环带那处走截断，别弄反

        val pixels = FloatArray(nx * ny * CELL * CELL * 3)
        val rects = Array(nx * ny) { CellRect(0, 0, 0, 0) }
        // resize 的输出是 CV_8UC3（跟着源图类型走），必须按字节读再转浮点：
        // 直接 `Mat.get(0, 0, FloatArray)` 会抛 `UnsupportedOperationException: Mat data type is not compatible: 16`。
        // Python 侧是 `patch.astype(np.float32)`，语义一致（0~255 的浮点副本）。
        val buf = ByteArray(CELL * CELL * 3)
        val dst = Mat()

        try {
            var k = 0
            for (j in 0 until ny) {
                val cy = (ylines[j] + ylines[j + 1]) / 2.0
                for (i in 0 until nx) {
                    val cx = (xlines[i] + xlines[i + 1]) / 2.0
                    var a = (cx - side / 2.0).roundToInt()
                    var c = (cy - side / 2.0).roundToInt()
                    a = max(0, min(a, img.cols() - 2))
                    c = max(0, min(c, img.rows() - 2))
                    val b = min(img.cols(), a + side)
                    val d = min(img.rows(), c + side)
                    rects[k] = CellRect(a, c, b, d)

                    val cell = img.submat(c, d, a, b)
                    Imgproc.resize(cell, dst, Size(CELL.toDouble(), CELL.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                    dst.get(0, 0, buf)
                    val base = k * CELL * CELL * 3
                    for (p in buf.indices) pixels[base + p] = (buf[p].toInt() and 0xFF).toFloat()
                    k++
                }
            }
        } finally {
            dst.release()
        }

        return rects to TileFrame(ny, nx, CELL, pixels)
    }

    private fun median(values: DoubleArray): Double {
        val v = values.sortedArray()
        return if (v.size % 2 == 1) v[v.size / 2] else 0.5 * (v[v.size / 2 - 1] + v[v.size / 2])
    }
}
