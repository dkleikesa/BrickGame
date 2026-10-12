package com.qcmian.brickgame.vision

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * 用**真机 app 界面截图**的投影数组对拍周期估计。
 *
 * 这块画面在真机上退化成了 `pitch=(12,12)`（12 = 最小下限，等于没找到周期），
 * 而 Python 参考给的是 `(248,12)`。真棋盘那张图两边一致，所以这个用例专门盯"低频占主导、
 * 没有明显周期"的输入 —— 如果这里也过了，就说明分叉在 app 侧算投影那一段（OpenCV），而不是纯数学。
 */
class AppUiMathTest {

    private fun parse(text: String): DoubleArray =
        text.split(',').map { it.trim().toDouble() }.toDoubleArray()

    @Test
    fun pitchMatchesPythonOnAppUiScreenshot() {
        val rowProf = parse(AppUiGolden.ROW_PROF)
        val colProf = parse(AppUiGolden.COL_PROF)

        val (pitchY, _) = LatticeMath.fundamentalPitch(rowProf, AppUiGolden.MIN_PITCH, rowProf.size / 4)
        val (pitchX, _) = LatticeMath.fundamentalPitch(colProf, AppUiGolden.MIN_PITCH, colProf.size / 4)

        println(
            "同一张图：Kotlin pitch=($pitchX,$pitchY)，Python pitch=" +
                "(${AppUiGolden.PITCH_X},${AppUiGolden.PITCH_Y})",
        )
        assertEquals(AppUiGolden.PITCH_X, pitchX, "列方向周期（Python=${AppUiGolden.PITCH_X}）")
        assertEquals(AppUiGolden.PITCH_Y, pitchY, "行方向周期（Python=${AppUiGolden.PITCH_Y}）")
    }
}
