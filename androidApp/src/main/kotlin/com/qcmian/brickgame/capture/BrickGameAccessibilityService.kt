package com.qcmian.brickgame.capture

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.app.KeyguardManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Path
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import com.qcmian.brickgame.ml.ModelVariant
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
 *
 * 悬浮窗层级在锁屏之上，所以这里还盯着亮灭屏：[onScreenOff] 收窗口、解锁再挂回来。
 */
class BrickGameAccessibilityService : AccessibilityService(), ScreenshotProvider, GestureExecutor {

    private var overlay: OverlayController? = null
    private var session: HintSession? = null

    private val mainExecutorRef = Executor { runnable -> Handler(Looper.getMainLooper()).post(runnable) }

    /** `dispatchGesture` 只收 Handler（`takeScreenshot` 收的是 Executor），单独留一个主线程 Handler。 */
    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * 锁屏期间把悬浮窗收掉。
     *
     * 无障碍悬浮窗（以及降级用的应用上层悬浮窗）层级**在锁屏之上**：锁屏后提示框照样画在锁屏画面上，
     * 控制条也还在、还能点到 —— 权限给高了。所以锁屏就摘窗口，解锁再原样挂回（会话不动）。
     */
    private val lockReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                Intent.ACTION_SCREEN_OFF -> overlay?.suspendForLock()
                Intent.ACTION_USER_PRESENT -> overlay?.resumeAfterUnlock()
                // 没设锁屏的机器等不到 USER_PRESENT：亮屏时锁屏已经不在，直接挂回
                Intent.ACTION_SCREEN_ON ->
                    if (keyguard?.isKeyguardLocked == false) overlay?.resumeAfterUnlock()
            }
        }
    }

    private var lockReceiverRegistered = false

    private val keyguard: KeyguardManager? by lazy { getSystemService(KeyguardManager::class.java) }

    override fun onServiceConnected() {
        super.onServiceConnected()
        // 系统可能重复回调，先把上一次的会话收干净
        session?.release()
        overlay?.hide()
        val controller = OverlayController(this)
        controller.onOverlayUnavailable = {
            Log.w(TAG, "悬浮窗挂不上：无障碍类型被拒且没有『显示在其他应用上层』授权")
        }
        overlay = controller
        session = HintSession(this, this, this, controller) { variant -> buildClassifier(variant) }
        registerLockReceiver()
        active = this
        Log.i(TAG, "无障碍服务已连接")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    override fun onUnbind(intent: Intent?): Boolean {
        session?.stop()
        overlay?.hide()
        unregisterLockReceiver()
        active = null
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        session?.release()
        overlay?.hide()
        unregisterLockReceiver()
        active = null
        super.onDestroy()
    }

    /** 亮灭屏/解锁广播。都是系统保护广播，注册失败也不影响别的功能。 */
    private fun registerLockReceiver() {
        if (lockReceiverRegistered) return
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_USER_PRESENT)
        }
        lockReceiverRegistered = try {
            // targetSdk 34+ 要求显式声明是否接收外部广播；这几个是系统广播，按"不导出"注册
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(lockReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                registerReceiver(lockReceiver, filter)
            }
            true
        } catch (t: Throwable) {
            Log.w(TAG, "锁屏广播注册失败：锁屏期间悬浮窗可能仍会显示", t)
            false
        }
    }

    private fun unregisterLockReceiver() {
        if (!lockReceiverRegistered) return
        lockReceiverRegistered = false
        runCatching { unregisterReceiver(lockReceiver) }
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
     * 直消：在**同一条手势**里先后点两块砖。
     *
     * 为什么必须点两次：一块砖可能同时和左右（或上下）两块同款砖同线且中间无砖，
     * 只点它一下游戏不知道要消哪一对。第一下选中、第二下指定，才没有歧义。
     * 唯一配对时也照样点两下 —— 第二下落在已经是空位的那格上，没有副作用。
     */
    override fun tapPair(
        x1: Float, y1: Float, x2: Float, y2: Float,
        tapMs: Long, gapMs: Long, onDone: (Boolean) -> Unit,
    ) {
        val first = Path().apply { moveTo(x1, y1) }
        val second = Path().apply { moveTo(x2, y2) }
        val hold = tapMs.coerceAtLeast(1L)
        val gesture = GestureDescription.Builder()
            // 一条手势里的两段互不重叠的 stroke：第二段的起点落在第一段结束 + 间隔之后。
            // 分成两次 dispatch 也行，但那样回调要串两层、看门狗也要两份，没必要。
            .addStroke(GestureDescription.StrokeDescription(first, 0L, hold))
            .addStroke(GestureDescription.StrokeDescription(second, hold + gapMs.coerceAtLeast(1L), hold))
            .build()
        dispatch(gesture, "点击($x1,$y1)+($x2,$y2)", onDone)
    }

    /**
     * 自动运行的手势出口：`dispatchGesture` 合成一次滑动。
     *
     * 用于"两块没有对齐"的情况：把一块砖滑到落点，靠游戏自己判定对齐后消除。
     */
    override fun swipe(
        x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long, onDone: (Boolean) -> Unit,
    ) {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val gesture = GestureDescription.Builder()
            // willContinue=false：一次独立的手势，后面还要接着发下一次，别被合成一条长滑
            .addStroke(GestureDescription.StrokeDescription(path, 0L, durationMs.coerceAtLeast(1L)))
            .build()
        dispatch(gesture, "滑动($x1,$y1 → $x2,$y2)", onDone)
    }

    /**
     * 一次独立手势的公共出口。
     *
     * 几个必须踩对的点：
     *  - 需要无障碍配置里声明 `android:canPerformGestures="true"`（见 res/xml/brickgame_accessibility.xml），
     *    没声明的话 [dispatchGesture] 直接返回 false；
     *  - 坐标必须是**屏幕绝对坐标**，调用方给的就是截图坐标（全屏截图，1:1 对齐）；
     *  - 回调走 [mainHandler]：`dispatchGesture` 的回调线程不定，而回调会去改悬浮窗，
     *    统一挪回主线程。回调**可能永远不来**（游戏抢焦点、系统手势冲突），
     *    所以自动运行的循环另挂了一道看门狗，别只指望这个回调（见 `HintSession.playAction`）。
     */
    private fun dispatch(gesture: GestureDescription, what: String, onDone: (Boolean) -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            onDone(false)
            return
        }
        val delivered = try {
            dispatchGesture(
                gesture,
                object : GestureResultCallback() {
                    override fun onCompleted(description: GestureDescription?) = onDone(true)
                    override fun onCancelled(description: GestureDescription?) = onDone(false)
                },
                mainHandler,
            )
        } catch (t: Throwable) {
            Log.w(TAG, "手势派发异常", t)
            false
        }
        if (!delivered) {
            Log.w(TAG, "手势派发被拒：$what")
            onDone(false)
        }
    }

    /**
     * 加载 assets 里的身份模型（[variant] 决定哪一套）。
     *
     * 只在 worker 线程真正用到时才构造（见 [HintSession.acquireClassifier]），这里不挡主线程。
     */
    private fun buildClassifier(variant: ModelVariant): TileClassifier =
        TfliteTileClassifier.load(this, variant)

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

        /**
         * 开始自动通关。
         *
         * @return 服务没连上时返回 false（界面据此提示去设置里开启）。
         */
        fun startAuto(): Boolean {
            val service = active ?: return false
            service.session?.startAuto()
            return true
        }

        /**
         * 停止自动通关，回到手动提示。
         *
         * 只停"自动"这一层：会话与悬浮窗留着，用户还能继续手动点「刷新」。
         */
        fun stopAuto() {
            active?.session?.stopAuto()
        }

        /**
         * 切换识别模型（主页上的选择器走这里）。
         *
         * 换模型要重建解释器，只有服务手里的会话做得到；服务没连上就什么也不做 ——
         * 存档已经写好了（见 `ModelChoice`），下次服务起来读到的就是新模型。
         */
        fun selectModel(variant: ModelVariant) {
            active?.session?.selectModel(variant)
        }
    }
}
