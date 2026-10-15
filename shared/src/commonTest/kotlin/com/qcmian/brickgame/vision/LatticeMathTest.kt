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

    /**
     * 裁边要跨过"整列/整行已消完"的空洞。
     *
     * 数据取真机第 1 层的实测列计数：10 列棋盘，第 7 列（下标 6）被玩家消空，
     * 右边三列（7/8/9）还有 6/2/6 块砖。老实现只取最长连续段 → 0..6（6 列），
     * 再被 MIN_COLS 当"离谱网格"丢掉整帧 → 界面报"未识别到棋盘"。
     */
    @Test
    fun trimKeepsBoardAcrossFullyClearedColumn() {
        val cols = intArrayOf(12, 11, 13, 7, 7, 7, 0, 6, 2, 6)
        assertEquals(0 to 10, LatticeMath.trimBoardExtent(cols))
    }

    /** 棋盘上方的 UI 噪声（设置图标/状态栏：每行 1~2 块"砖"）不能被并进棋盘。 */
    @Test
    fun trimDropsUiNoiseAboveBoard() {
        val rows = intArrayOf(1, 1, 2, 1, 0, 7, 4, 6, 7, 5, 4, 2, 6, 4, 5, 6, 1, 3, 6)
        assertEquals(5 to 19, LatticeMath.trimBoardExtent(rows))
    }

    /** 正常棋盘（无空洞、两端不稀）不许被动到。 */
    @Test
    fun trimLeavesDenseBoardAlone() {
        val cols = intArrayOf(9, 10, 10, 9, 10, 9, 10, 9, 10, 9)
        assertEquals(0 to 10, LatticeMath.trimBoardExtent(cols))
    }

    /**
     * 候选格距里挑"能把棋盘外框铺成整格"的那个 —— 残局里自相关和色块间距都不可信，只能靠这个分对错。
     * 数字都取真机实测的底板外框（宽 1017 / 高 1440，另有残局帧 1102x1559）。
     */
    @Test
    fun pickPitchPrefersIntegralFit() {
        // 残局帧：错的 94 把 1102px 铺成 11.72 格（差 0.28），对的 110 铺成 10.02 格（差 0.02）
        assertEquals(110.0, LatticeMath.pickPitch(listOf(94.0, 110.0), 1102.0))
        // 第 1 层横向：自相关给的 101 铺 1017px 出 10.07 格，色块间距给的 111 出 9.16 格 → 取 101
        assertEquals(101.0, LatticeMath.pickPitch(listOf(101.0, 111.0), 1017.0))
        // 同一帧纵向：101 铺 1440px 出 14.26 格，色块间距给的 114 出 12.63 格 → 取 101（14 行）
        assertEquals(101.0, LatticeMath.pickPitch(listOf(101.0, 114.0), 1440.0))
        // 候选全都凑不出 3~30 格：返回 0，调用方据此丢弃这一帧
        assertEquals(0.0, LatticeMath.pickPitch(listOf(3.0), 1000.0))
    }
}
