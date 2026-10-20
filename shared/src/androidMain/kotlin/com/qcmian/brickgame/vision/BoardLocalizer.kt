package com.qcmian.brickgame.vision

import android.graphics.Bitmap

/**
 * 定位器：一张截图 → 网格几何 + 每格判空。OpenCV 与 YOLO 两条实现都满足它。
 *
 * 抽成接口是为了让 [HintPipeline] 不依赖具体的定位实现 —— YOLO 的 TFLite 解释器要在
 * app 模块里构造（assets 在那边），shared 模块不能反向依赖。
 */
fun interface BoardLocalizer {
    fun locate(frame: Bitmap): BoardScan?
}
