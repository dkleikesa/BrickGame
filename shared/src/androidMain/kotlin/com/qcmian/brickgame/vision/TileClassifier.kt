package com.qcmian.brickgame.vision

import android.graphics.Bitmap

/**
 * 识别层：切好的格子 → 每格的类别 id（0 = 背景/空位）。
 *
 * 接口放在 androidMain，是因为实现要碰 [Bitmap] 与 Android 位图裁切。
 * 上层（[HintEngine]）只认 [BoardState]，所以换识别实现不影响算法与 UI。
 */
interface TileClassifier {

    /** 给日志用的完整名字（如 "TFLite Small（2369 类，256 维，GPU）"）。 */
    val name: String

    /**
     * 给控制条用的**短名**（如 "Small"）：一行里要跟帧号与耗时并排显示，长了放不下。
     *
     * 界面显示它而不是显示"用户选了什么"，就是为了能一眼看出换模型到底生效没有。
     */
    val label: String

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

    /**
     * 上一次 [classify] 的内部耗时分解（阶段名 → 毫秒），没有就返回空。
     *
     * 为什么要在接口上开这个：整帧 7~10s 里到底是谁占的，光看总数分不出来，
     * 而每个实现的热点完全不同（这里是"转图 / 裁格 / 推理 / 检索"）。
     * 真机实测前只靠读代码猜，多半会去优化错的那一段。
     */
    fun phaseTimings(): List<Pair<String, Long>> = emptyList()
}
