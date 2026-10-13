package com.qcmian.brickgame.vision

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * 谐波修正的回归测试。
 *
 * 真机实测：走完一回合后行投影周期变弱，自相关取到 **3 倍谐波**（307 = 3×102），
 * 棋盘被切成左上角 6×6，提示全错。`reconcilePitches` 用"砖块是方的"这个先验把它除回去。
 */
class PitchReconcileTest {

    @Test
    fun thirdHarmonicIsDividedBack() {
        // 真机失败帧：列周期 102（对），行周期 307（= 3×102）
        assertEquals(102 to 102, LatticeMath.reconcilePitches(102, 307))
        // 对称：行对列歪
        assertEquals(102 to 102, LatticeMath.reconcilePitches(307, 102))
    }

    @Test
    fun secondHarmonicIsDividedBack() {
        assertEquals(102 to 102, LatticeMath.reconcilePitches(102, 205))
    }

    @Test
    fun consistentPitchesAreLeftAlone() {
        assertEquals(102 to 102, LatticeMath.reconcilePitches(102, 102))
        // 差 10% 以内算一致，不动
        assertEquals(110 to 118, LatticeMath.reconcilePitches(110, 118))
    }

    @Test
    fun irreconcilablePitchesAreReturnedAsIs() {
        // 110 和 23：除 2/3/4 都回不到 110 → 原样返回，交给调用方走 BlobLattice 反推
        assertEquals(110 to 23, LatticeMath.reconcilePitches(110, 23))
    }

    @Test
    fun divisibleButImplausibleCorrectionIsLeftToCaller() {
        // 残局的 x23/y94：94/4 = 23 确实"除得回去"，但那样一屏会有 40+ 列，不是棋盘。
        // 函数只做数学；合理性由调用方再用列数范围拦一道（见 OpenCvBoardScanner）。
        assertEquals(23 to 23, LatticeMath.reconcilePitches(23, 94))
    }
}
