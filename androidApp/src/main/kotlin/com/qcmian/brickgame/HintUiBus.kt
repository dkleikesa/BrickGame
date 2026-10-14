package com.qcmian.brickgame

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * 运行状态快照。
 *
 * 一帧的提示本体、帧信息、报错都只显示在**悬浮条**上（见 `overlay/ControlBarView`），
 * 界面这边只留"跑没跑、第几帧"。
 */
data class HintUiState(
    val running: Boolean = false,
    val frames: Int = 0,
)

/**
 * 无障碍服务 ↔ Compose 界面 的共享状态。
 *
 * 单进程、单入口，直接用 Compose 的快照状态，省掉 Flow/ViewModel 一层：
 * 服务在后台线程算完就写这里，界面读到变化自动重组。
 */
object HintUiBus {

    var state by mutableStateOf(HintUiState())

    /** 调试图层，默认关：屏幕上只留"能消的那几格"，别糊一团颜色。 */
    var showGrid by mutableStateOf(false)

    var showLabels by mutableStateOf(false)

    /** 实际生效的悬浮窗窗口类型（2032=无障碍专属 / 2038=应用上层），null=还没挂上。 */
    var overlayWindowType by mutableStateOf<Int?>(null)
}
