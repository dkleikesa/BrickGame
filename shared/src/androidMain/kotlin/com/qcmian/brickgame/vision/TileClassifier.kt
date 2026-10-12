package com.qcmian.brickgame.vision

import android.graphics.Bitmap

/**
 * 识别层：切好的格子 → 每格的类别 id（0 = 背景/空位）。
 *
 * 接口放在 androidMain，是因为两种实现（像素聚类、TFLite 模型）都要碰 [Bitmap] 或 Android 位图裁切。
 * 上层（[HintEngine]）只认 [BoardState]，所以换识别实现不影响算法与 UI。
 */
interface TileClassifier {

    /** 给 UI 显示用的名字（"像素聚类（未接模型）"/"TFLite 模型"）。 */
    val name: String

    /**
     * @param frame 原始截图，只读；需要 64×64 之类别的裁切时由实现自己裁。
     * @param scan 切图结果（几何 + 32×32 裁切图 + 判空）。
     * @return 行优先、长度 `rows*cols` 的类别 id；0 表示背景/空位。
     */
    fun classify(frame: Bitmap, scan: BoardScan): IntArray
}
