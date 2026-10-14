package com.qcmian.brickgame.vision

import android.graphics.Bitmap

/**
 * 识别层：切好的格子 → 每格的类别 id（0 = 背景/空位）。
 *
 * 接口放在 androidMain，是因为实现要碰 [Bitmap] 与 Android 位图裁切。
 * 上层（[HintEngine]）只认 [BoardState]，所以换识别实现不影响算法与 UI。
 */
interface TileClassifier {

    /** 给 UI 显示用的名字（如 "ONNX 身份向量（2369 类，256 维）"）。 */
    val name: String

    /**
     * @param frame 原始截图，只读；需要 64×64 之类别的裁切时由实现自己裁。
     * @param scan 切图结果（几何 + 32×32 裁切图 + 判空）。
     * @return 行优先、长度 `rows*cols` 的类别 id；0 表示背景/空位。
     */
    fun classify(frame: Bitmap, scan: BoardScan): IntArray

    /**
     * 上一帧"模型没把握"的格子（行优先序号 → 相似度百分制）；没有就返回空。
     *
     * 识别**不按它改判**（低分也照样给 top-1），这个列表只用来决定"这一帧要不要留档给以后重训"。
     */
    fun lowConfidenceCells(): List<Pair<Int, Int>> = emptyList()
}
