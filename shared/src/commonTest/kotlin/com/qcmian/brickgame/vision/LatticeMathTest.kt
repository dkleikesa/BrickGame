package com.qcmian.brickgame.vision

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 把 Kotlin 实现钉在 `BrickGame-Tool/tile_recognizer.py` 上。
 *
 * 只覆盖"纯数学"部分：投影 → 周期/相位/网格线，以及图标特征 → 距离 → 聚类。
 * 截图里 OpenCV 那一段（灰度/Sobel/缩放）由 androidMain 负责，靠真机调试图核对。
 */
class LatticeMathTest {

    private fun parse(s: String) = s.split(',').map { it.toDouble() }.toDoubleArray()

    private fun synthPatch(dy: Int, dx: Int, checker: Boolean, cell: Int = VisionGolden.CELL): FloatArray {
        val p = FloatArray(cell * cell * 3) { 200f }
        for (y in 0 until cell) {
            for (x in 0 until cell) {
                val i = (y * cell + x) * 3
                if (checker) {
                    if ((y / 4 + x / 4) % 2 == 0) {
                        p[i] = 10f
                        p[i + 1] = 200f
                        p[i + 2] = 200f
                    }
                } else {
                    val sy = y - dy
                    val sx = x - dx
                    if (sy in 10..21 && sx in 10..21) {
                        p[i] = 20f
                        p[i + 1] = 30f
                        p[i + 2] = 40f
                    }
                }
            }
        }
        return p
    }

    private fun frameOf(cell: Int, vararg patches: FloatArray) =
        TileFrame(1, patches.size, cell, patches.reduce { a, b -> a + b })

    @Test
    fun pitchPhaseAndLinesMatchPython() {
        val rowProf = parse(VisionGolden.ROW_PROF)
        val colProf = parse(VisionGolden.COL_PROF)

        val minPitch = VisionGolden.MIN_PITCH
        val (pitchY, _) = LatticeMath.fundamentalPitch(rowProf, minPitch, (rowProf.size * 0.25).toInt())
        val (pitchX, _) = LatticeMath.fundamentalPitch(colProf, minPitch, (colProf.size * 0.25).toInt())
        assertEquals(VisionGolden.PITCH_Y, pitchY)
        assertEquals(VisionGolden.PITCH_X, pitchX)

        assertEquals(VisionGolden.PHASE_Y, LatticeMath.latticePhase(rowProf, pitchY))
        assertEquals(VisionGolden.PHASE_X, LatticeMath.latticePhase(colProf, pitchX))

        val ylines = LatticeMath.refineLines(
            rowProf,
            LatticeMath.arange(VisionGolden.PHASE_Y.toDouble(), rowProf.size.toDouble(), pitchY.toDouble()),
            pitchY,
        )
        val xlines = LatticeMath.refineLines(
            colProf,
            LatticeMath.arange(VisionGolden.PHASE_X.toDouble(), colProf.size.toDouble(), pitchX.toDouble()),
            pitchX,
        )

        val expectY = parse(VisionGolden.YLINES)
        val expectX = parse(VisionGolden.XLINES)
        assertEquals(expectY.size, ylines.size, "ylines 数量")
        assertEquals(expectX.size, xlines.size, "xlines 数量")
        for (i in expectY.indices) {
            assertTrue(abs(expectY[i] - ylines[i]) < 1e-3, "ylines[$i] ${ylines[i]} vs ${expectY[i]}")
        }
        for (i in expectX.indices) {
            assertTrue(abs(expectX[i] - xlines[i]) < 1e-3, "xlines[$i] ${xlines[i]} vs ${expectX[i]}")
        }
    }

    @Test
    fun iconFeaturesMatchPython() {
        val frame = frameOf(
            VisionGolden.CELL,
            synthPatch(0, 0, false),
            synthPatch(3, 0, false),
            synthPatch(0, 0, true),
        )
        val features = FeatureMath.iconFeatures(frame)

        for (i in VisionGolden.ICON_RATIO.indices) {
            assertTrue(
                abs(VisionGolden.ICON_RATIO[i] - features.ratio[i]) < 1e-6,
                "ratio[$i] ${features.ratio[i]} vs ${VisionGolden.ICON_RATIO[i]}",
            )
        }

        var sum = 0.0
        var nnz = 0
        for (v in features.dev) {
            sum += v
            if (v != 0f) nnz++
        }
        assertTrue(abs(VisionGolden.ICON_DEV_SUM[0] + VisionGolden.ICON_DEV_SUM[1] + VisionGolden.ICON_DEV_SUM[2] - sum) < 1.0)
        assertEquals(VisionGolden.ICON_DEV_NNZ.sum(), nnz)
    }

    @Test
    fun alignedDistanceMatrixMatchesPython() {
        val frame = frameOf(
            VisionGolden.CELL,
            synthPatch(0, 0, false),
            synthPatch(3, 0, false),
            synthPatch(0, 7, false),
            synthPatch(0, 0, true),
        )
        val dev = FeatureMath.iconFeatures(frame).dev
        val d = FeatureMath.alignedDistanceMatrix(dev, 4, VisionGolden.CELL, rad = 3, prefilter = 1.0)

        for (i in VisionGolden.ALIGNED.indices) {
            for (j in VisionGolden.ALIGNED.indices) {
                assertTrue(
                    abs(VisionGolden.ALIGNED[i][j] - d[i][j]) < 1e-5,
                    "D[$i][$j] ${d[i][j]} vs ${VisionGolden.ALIGNED[i][j]}",
                )
            }
        }
    }

    @Test
    fun clusteringMatchesPython() {
        val dist = VisionGolden.CLUSTER_DIST
        val gap = FeatureMath.gapThreshold(dist)
        assertTrue(abs(VisionGolden.CLUSTER_GAP - gap) < 1e-6, "gap $gap vs ${VisionGolden.CLUSTER_GAP}")

        val clustering = FeatureMath.clusterByParity(dist, gap)
        assertTrue(abs(VisionGolden.CLUSTER_THR - clustering.threshold) < 1e-6)
        assertEquals(VisionGolden.CLUSTER_OK, clustering.evenSatisfied)
        for (i in VisionGolden.CLUSTER_LABELS.indices) {
            assertEquals(VisionGolden.CLUSTER_LABELS[i], clustering.labels[i], "labels[$i]")
        }

        val merged = FeatureMath.mergeSimilar(clustering.labels, dist, gap)
        for (i in VisionGolden.MERGED_LABELS.indices) {
            assertEquals(VisionGolden.MERGED_LABELS[i], merged[i], "merged[$i]")
        }
    }
}
