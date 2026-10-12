package com.qcmian.brickgame.vision

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 真机画面（微信小程序消除游戏，**16 行 × 10 列**）的标定对拍。
 *
 * app 端在这块画面上给出的是 6×6/6×10，而 Python 参考实现给 16×10 —— 说明 Kotlin 标定跑偏了。
 * 这个测试把 Python 的真实投影数组喂给 [LatticeMath]，把分叉点从"OpenCV 算投影"和"纯数学求网格"
 * 两段里分离出来：这里过了，就说明问题在投影那一段。
 */
class RealBoardMathTest {

    private fun parse(text: String): DoubleArray =
        text.split(',').map { it.trim().toDouble() }.toDoubleArray()

    @Test
    fun pitchMatchesPythonOnRealBoard() {
        val rowProf = parse(RealBoardGolden.ROW_PROF)
        val colProf = parse(RealBoardGolden.COL_PROF)

        val (pitchY, _) = LatticeMath.fundamentalPitch(rowProf, RealBoardGolden.MIN_PITCH, rowProf.size / 4)
        val (pitchX, _) = LatticeMath.fundamentalPitch(colProf, RealBoardGolden.MIN_PITCH, colProf.size / 4)

        assertEquals(RealBoardGolden.PITCH_Y, pitchY, "行方向周期")
        assertEquals(RealBoardGolden.PITCH_X, pitchX, "列方向周期")
    }

    @Test
    fun phaseMatchesPythonOnRealBoard() {
        val rowProf = parse(RealBoardGolden.ROW_PROF)
        val colProf = parse(RealBoardGolden.COL_PROF)

        assertEquals(RealBoardGolden.PHASE_Y, LatticeMath.latticePhase(rowProf, RealBoardGolden.PITCH_Y))
        assertEquals(RealBoardGolden.PHASE_X, LatticeMath.latticePhase(colProf, RealBoardGolden.PITCH_X))
    }

    @Test
    fun refinedLinesMatchPythonOnRealBoard() {
        val rowProf = parse(RealBoardGolden.ROW_PROF)
        val colProf = parse(RealBoardGolden.COL_PROF)

        val ylines = LatticeMath.refineLines(
            rowProf,
            LatticeMath.arange(
                RealBoardGolden.PHASE_Y.toDouble(),
                rowProf.size.toDouble(),
                RealBoardGolden.PITCH_Y.toDouble(),
            ),
            RealBoardGolden.PITCH_Y,
        )
        val xlines = LatticeMath.refineLines(
            colProf,
            LatticeMath.arange(
                RealBoardGolden.PHASE_X.toDouble(),
                colProf.size.toDouble(),
                RealBoardGolden.PITCH_X.toDouble(),
            ),
            RealBoardGolden.PITCH_X,
        )

        val wantY = parse(RealBoardGolden.YLINES)
        val wantX = parse(RealBoardGolden.XLINES)
        assertEquals(wantY.size, ylines.size, "行网格线数量")
        assertEquals(wantX.size, xlines.size, "列网格线数量")
        for (i in wantY.indices) {
            assertTrue(abs(wantY[i] - ylines[i]) < 0.01, "第 $i 条行网格线：期望 ${wantY[i]}，实际 ${ylines[i]}")
        }
        for (i in wantX.indices) {
            assertTrue(abs(wantX[i] - xlines[i]) < 0.01, "第 $i 条列网格线：期望 ${wantX[i]}，实际 ${xlines[i]}")
        }
    }
}
