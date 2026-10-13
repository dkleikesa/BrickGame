package com.qcmian.brickgame

import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/** 一帧分析结果在界面上的快照。 */
data class HintUiState(
    val running: Boolean = false,
    val frames: Int = 0,
    val scanSummary: String = "",
    val hintText: String = "",
    val cleared: Boolean = false,
    val stuck: Boolean = false,
    val elapsedMs: Long = 0,
    val message: String? = null,
)

/**
 * 无障碍服务 ↔ Compose 界面 的共享状态。
 *
 * 单进程、单入口，直接用 Compose 的快照状态，省掉 Flow/ViewModel 一层：
 * 服务在后台线程算完就写这里，界面读到变化自动重组。
 */
object HintUiBus {

    var state by mutableStateOf(HintUiState())

    /** 最近一帧的调试图（截图 + 网格 + 类别 + 提示箭头）。界面不可见时不生成，省 10MB/帧 的分配。 */
    var preview by mutableStateOf<Bitmap?>(null)

    var previewEnabled: Boolean = false

    /** 调试图层，默认关：屏幕上只留"能消的那几格"，别糊一团颜色。 */
    var showGrid by mutableStateOf(false)

    var showLabels by mutableStateOf(false)

    /**
     * 两种窗口类型都没挂上（无障碍类型被系统拒 + 没有"显示在其他应用上层"授权）时为 true，
     * 界面据此弹出授权引导。见 overlay/OverlayController 的降级链。
     */
    var needsOverlayPermission by mutableStateOf(false)

    /** 实际生效的悬浮窗窗口类型（2032=无障碍专属 / 2038=应用上层），null=还没挂上。 */
    var overlayWindowType by mutableStateOf<Int?>(null)

    /**
     * 点"开始提示"失败时的说明（例如服务在系统里开着但进程里还没连上）。
     * 有它就别把用户直接扔到设置页——那是两回事。
     */
    var serviceMessage by mutableStateOf<String?>(null)
}
