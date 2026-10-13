package com.qcmian.brickgame.capture

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import com.qcmian.brickgame.HintUiBus
import com.qcmian.brickgame.ml.TfliteTileClassifier
import com.qcmian.brickgame.overlay.OverlayController
import com.qcmian.brickgame.vision.TileClassifier
import java.util.concurrent.Executor

/**
 * 截屏 + 悬浮窗的载体。
 *
 * 为什么走无障碍而不是 MediaProjection：
 *  - `takeScreenshot` 静默截屏，没有"开始录屏"的系统弹窗；
 *  - 悬浮窗可用 `TYPE_ACCESSIBILITY_OVERLAY`，连 `SYSTEM_ALERT_WINDOW` 都不用申请。
 * 代价是要求 API 30+，且用户要在系统设置里手动开启本服务。
 */
class BrickGameAccessibilityService : AccessibilityService(), ScreenshotProvider {

    private var overlay: OverlayController? = null
    private var session: HintSession? = null

    private val mainExecutorRef = Executor { runnable -> Handler(Looper.getMainLooper()).post(runnable) }

    override fun onServiceConnected() {
        super.onServiceConnected()
        // 系统可能重复回调，先把上一次的会话收干净
        session?.release()
        overlay?.hide()
        val controller = OverlayController(this)
        controller.onOverlayUnavailable = {
            HintUiBus.needsOverlayPermission = true
            Log.w(TAG, "悬浮窗挂不上：无障碍类型被拒且没有『显示在其他应用上层』授权")
        }
        overlay = controller
        session = HintSession(this, this, controller) { buildClassifier() }
        active = this
        HintUiBus.state = HintUiBus.state.copy(message = null)
        HintUiBus.needsOverlayPermission = false
        Log.i(TAG, "无障碍服务已连接")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    override fun onUnbind(intent: Intent?): Boolean {
        session?.stop()
        overlay?.hide()
        active = null
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        session?.release()
        overlay?.hide()
        active = null
        super.onDestroy()
    }

    override fun capture(onResult: (Bitmap?, String?) -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            onResult(null, "无障碍截屏需要 Android 11（API 30）以上")
            return
        }
        try {
            takeScreenshot(
                Display.DEFAULT_DISPLAY,
                mainExecutorRef,
                object : TakeScreenshotCallback {
                    override fun onSuccess(screenshot: ScreenshotResult) {
                        val buffer = screenshot.hardwareBuffer
                        val bitmap = try {
                            // wrapHardwareBuffer 拿到的是 HARDWARE 位图，OpenCV/OpenGL 之外读不了像素，
                            // 必须先拷成 ARGB_8888；buffer 关掉之后那张包装位图就失效了。
                            Bitmap.wrapHardwareBuffer(buffer, screenshot.colorSpace)
                                ?.copy(Bitmap.Config.ARGB_8888, false)
                        } catch (t: Throwable) {
                            Log.w(TAG, "截图解码失败", t)
                            null
                        } finally {
                            buffer.close()
                        }
                        onResult(bitmap, if (bitmap == null) "截图解码失败" else null)
                    }

                    override fun onFailure(errorCode: Int) {
                        onResult(null, describeError(errorCode))
                    }
                },
            )
        } catch (t: Throwable) {
            onResult(null, "发起截屏异常：${t.message}")
        }
    }

    private fun describeError(code: Int): String = when (code) {
        ERROR_TAKE_SCREENSHOT_NO_ACCESSIBILITY_ACCESS -> "无障碍服务没有截屏能力（重开一次服务）"
        ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT -> "截屏请求太密"
        ERROR_TAKE_SCREENSHOT_INVALID_DISPLAY -> "无效的屏幕"
        else -> "截屏失败（code=$code）"
    }

    /**
     * 加载 assets 里的身份模型。
     *
     * 只在 worker 线程第一次用到时才构造（见 [HintSession]），这里不挡主线程。
     */
    private fun buildClassifier(): TileClassifier = TfliteTileClassifier.load(this)

    companion object {
        private const val TAG = "BrickGameA11y"

        @Volatile
        private var active: BrickGameAccessibilityService? = null

        /** 无障碍服务已经在系统里开启并绑定。 */
        fun isConnected(): Boolean = active != null

        /** @return 服务没连上时返回 false，界面据此提示去设置里开启。 */
        fun start(): Boolean {
            val service = active ?: return false
            service.session?.start()
            return true
        }

        fun stop() {
            active?.session?.stop()
        }

        /** 只跑一次（"验证一次"按钮）。 */
        fun once(): Boolean {
            val service = active ?: return false
            service.session?.once()
            return true
        }
    }
}
