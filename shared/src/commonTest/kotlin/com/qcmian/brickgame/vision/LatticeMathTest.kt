package com.qcmian.brickgame.vision

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 把 Kotlin 实现钉在 `BrickGame-Tool/tile_recognizer.py` 上。
 *
 * 只覆盖"纯数学"的投影部分：周期 → 相位 → 网格线。
 * 截图里 OpenCV 那一段（灰度/Sobel/缩放）由 androidMain 负责，靠真机调试图核对。
 *
 * 图标特征 / 距离矩阵 / 聚类的对拍随着像素聚类识别一起删掉了（识别现在全走模型），
 * 那几组 golden 常量和 `gen_vision_golden.py` 里的生成逻辑还在，要用时随时能捞回来。
 */
class LatticeMathTest {

    private fun parse(s: String) = s.split(',').map { it.toDouble() }.toDoubleArray()

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
}
