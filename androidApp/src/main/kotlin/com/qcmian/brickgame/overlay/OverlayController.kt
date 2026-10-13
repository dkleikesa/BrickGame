package com.qcmian.brickgame.overlay

import android.content.Context
import android.graphics.PixelFormat
import android.os.Build
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager

/**
 * 两个悬浮窗的载体：
 *  - **提示层**（[HintOverlayView]）：整屏、`FLAG_NOT_TOUCHABLE`，触摸全穿透给游戏；
 *  - **控制条**（[ControlBarView]）：贴屏幕最底部、可触摸，提供手动刷新与关闭，并显示提示信息。
 *
 * ## 窗口类型（踩过坑，别乱改）
 * 首选 `TYPE_ACCESSIBILITY_OVERLAY`：无障碍服务专属类型，**不需要** `SYSTEM_ALERT_WINDOW`，
 * 也不弹"在其他应用上层显示"的授权页。设计上就是冲这个来的。
 *
 * 但实测它并不总被放行：真机上 `addView` 会抛
 * `BadTokenException: Unable to add window -- token null is not valid`。
 * 所以这里改成**降级链**——无障碍类型挂不上、且用户已授权"显示在其他应用上层"时，
 * 自动改用 `TYPE_APPLICATION_OVERLAY`。每一路的成败都打日志（tag `OverlayController`），
 * 真机上跑一次看 logcat 就知道这台设备吃哪一路。
 *
 * 还有一条：这里**必须用 Service 自己的 Context**，不能用 `applicationContext` ——
 * 后者挂无障碍悬浮窗根本拿不到合法窗口 token（同样的 BadTokenException）。
 */
class OverlayController(context: Context) {

    private val context = context
    private val windowManager = context.getSystemService(WindowManager::class.java)

    private val hintView = HintOverlayView(context)
    private val barView = ControlBarView(context)

    private var hintAttached = false
    private var barAttached = false

    /** 实际生效的窗口类型；null = 没挂上。 */
    var activeType: Int? = null
        private set

    /** 两路都没挂上（一般是缺"显示在其他应用上层"授权）时回调，界面据此引导。 */
    var onOverlayUnavailable: (() -> Unit)? = null

    var onRefresh: (() -> Unit)?
        get() = barView.onRefresh
        set(value) {
            barView.onRefresh = value
        }

    var onClose: (() -> Unit)?
        get() = barView.onClose
        set(value) {
            barView.onClose = value
        }

    /** 提示层 + 控制条一起挂。@return 是否挂上。 */
    fun show(): Boolean {
        if (hintAttached && barAttached) return true
        for (type in candidates()) {
            if (attachHint(type) && attachBar(type)) {
                activeType = type
                Log.i(TAG, "两个悬浮窗就位，窗口类型 $type")
                return true
            }
            detachHint()
        }
        onOverlayUnavailable?.invoke()
        return false
    }

    /** 只挂整屏提示层（"验证一次"用，不弹控制条）。 */
    fun showHint(): Boolean {
        if (hintAttached) return true
        for (type in candidates()) {
            if (attachHint(type)) {
                activeType = type
                return true
            }
        }
        onOverlayUnavailable?.invoke()
        return false
    }

    /** 摘掉两个窗口。 */
    fun hide() {
        detachHint()
        detachBar()
        hintView.frame = null
        activeType = null
    }

    fun updateHint(frame: HintOverlayView.Frame?) {
        hintView.frame = frame
    }

    /**
     * 截图期间把提示层藏起来（只改可见性，不摘窗口）。
     *
     * 上一帧画在屏幕上的提示框会进本次截图：框边的强边缘会带偏标定，
     * 50% 的红色也会改变砖块颜色、影响识别。控制条**不藏** —— 它在棋盘外面，
     * 而且用户要靠它点「刷新」。
     */
    fun setHintLayerVisible(visible: Boolean) {
        hintView.visibility = if (visible) View.VISIBLE else View.GONE
    }

    fun updateBar(state: ControlBarView.State?) {
        barView.state = state
    }

    private fun attachHint(type: Int): Boolean {
        if (hintAttached) return true
        hintAttached = add(hintView, params(type, fullScreen = true), type)
        return hintAttached
    }

    private fun attachBar(type: Int): Boolean {
        if (barAttached) return true
        // 控制条在提示层之后 attach，保证它在上面接得到触摸
        barAttached = add(barView, params(type, fullScreen = false), type)
        return barAttached
    }

    private fun detachHint() {
        if (!hintAttached) return
        remove(hintView)
        hintAttached = false
    }

    private fun detachBar() {
        if (!barAttached) return
        remove(barView)
        barAttached = false
    }

    /** 降级链：先免权限的无障碍类型；不行再看有没有"上层显示"授权走 APPLICATION_OVERLAY。 */
    private fun candidates(): List<Int> = buildList {
        add(WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY)
        if (Settings.canDrawOverlays(context)) {
            add(WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY)
        }
    }

    private fun add(view: View, params: WindowManager.LayoutParams, type: Int): Boolean {
        val wm = windowManager ?: return false
        return try {
            wm.addView(view, params)
            Log.i(TAG, "悬浮窗已挂上：${view.javaClass.simpleName} type=$type")
            true
        } catch (e: Throwable) {
            Log.w(TAG, "悬浮窗挂不上：${view.javaClass.simpleName} type=$type", e)
            false
        }
    }

    private fun remove(view: View) {
        try {
            windowManager?.removeViewImmediate(view)
        } catch (e: Throwable) {
            Log.w(TAG, "悬浮窗移除失败：${view.javaClass.simpleName}", e)
        }
    }

    private fun params(type: Int, fullScreen: Boolean): WindowManager.LayoutParams {
        if (!fullScreen) return barParams(type)

        // 提示层：整屏 + 不吃触摸（触摸全穿透给游戏），所以要铺到最边上
        val flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
        return WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            type,
            flags,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
        }
    }

    /**
     * 控制条的窗口：可触摸，所以**必须避开系统手势区**。
     *
     * 两个关键点，少一个都会把"上滑回桌面/多任务"吃掉（真机踩过）：
     *  1. 不加 `FLAG_LAYOUT_NO_LIMITS`，并且视图高度只算内容；
     *  2. 用 [navBarHeight] 把窗口整体上移，正好停在导航栏/手势条上沿。
     */
    private fun barParams(type: Int): WindowManager.LayoutParams =
        WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.START
            y = navBarHeight          // Gravity.BOTTOM 下 y 是"离底边多远"
        }

    private val navBarHeight: Int by lazy {
        val id = context.resources.getIdentifier("navigation_bar_height", "dimen", "android")
        if (id > 0) context.resources.getDimensionPixelSize(id) else 0
    }

    private companion object {
        const val TAG = "OverlayController"
    }
}
