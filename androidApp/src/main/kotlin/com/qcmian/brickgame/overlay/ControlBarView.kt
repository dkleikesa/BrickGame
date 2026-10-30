package com.qcmian.brickgame.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.text.TextPaint
import android.text.TextUtils
import android.view.MotionEvent
import android.view.View

/**
 * 屏幕最底部的控制条（独立悬浮窗，**可触摸**）。
 *
 * 为什么单独一个窗口：主提示层要整屏覆盖且 `FLAG_NOT_TOUCHABLE`（触摸全穿透给游戏），
 * 而这里必须能点按钮，两者的窗口标志互斥，只能拆成两个窗口。
 *
 * 自己画而不是用 Button：无障碍服务的 Context 不是 Activity，拿不到 Activity 主题，
 * 用平台控件容易踩主题解析的坑；纯 onDraw + onTouchEvent 反倒更可控。
 *
 * 布局分两行：
 *  - 上行：提示文字 + 状态小字（占满整宽，不再被按钮挤）；
 *  - 下行：快速模式开关 → 自动通关开关 → 信息开关（网格 + 类别号一起）→ 刷新。
 *
 * 为什么拆成两行：加上「自动」以后这一排控件在窄屏（360dp 甚至 320dp）上会把提示文字
 * 挤到只剩几十 dp。控件单独一行之后，文字与按钮互不抢占。
 */
class ControlBarView(context: Context) : View(context) {

    data class State(
        /** 提示本体，一行。 */
        val hintText: String,
        /** 状态小字（帧号、单帧耗时、识别实现）。 */
        val statusText: String,
        /** 出错时的提示；非空时状态行显示成红色。 */
        val message: String?,
    )

    var state: State? = null
        set(value) {
            field = value
            invalidate()
        }

    var onRefresh: (() -> Unit)? = null

    /** 信息开关：网格与类别号一起显示/隐藏。点一下立刻回调，由上层即时重画（不用等下一次刷新）。 */
    var onToggleInfo: ((Boolean) -> Unit)? = null

    /** 快速模式开关：自动通关时连续出手，每若干块砖才识别一次校准（详见 HintSession）。 */
    var onToggleFast: ((Boolean) -> Unit)? = null

    /** 快速模式开关的显示状态。 */
    var fastOn: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    /** 无解继续：true = 自动模式遇无解不退出，继续贪心消。 */
    var onToggleContinue: ((Boolean) -> Unit)? = null

    /** 自动通关开关：点一下把期望的新状态回调出去（true = 开始自动）。 */
    var onToggleAuto: ((Boolean) -> Unit)? = null

    /** 信息开关的显示状态（写进去就重绘）。 */
    var infoOn: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    /**
     * 本帧各段耗时，**独立显示在状态行最右侧**。
     *
     * 单独一个字段而不是拼进 statusText：自动模式会不停改写状态行（求解中／第 k 步），
     * 拼在里面一转眼就被覆盖，用户根本看不到。
     */
    var timingText: String = ""
        set(value) {
            field = value
            invalidate()
        }

    /** 无解继续开关的显示状态（真值在 AutoContinue 里）。 */
    var continueOn: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    /**
     * "自动"的显示状态。
     *
     * **由会话回写**，不是"点了就翻"：自动可能自己停下来（通关 / 无解 / 手势没生效），
     * 只看按钮的本地状态会跟真实运行对不上。
     */
    var autoOn: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    private val density = resources.displayMetrics.density
    private fun dp(v: Float) = v * density

    // 一行布局：5（上留白）+ 11（文字行）+ 3 + 17（控件行）+ 6（下留白）≈ 42。
    // 两行太占竖屏空间，会压到游戏底部的道具栏；控件行也比常规按钮矮 1/3。
    private val heightPx = dp(42f).toInt()
    private val padding = dp(8f)
    private val topPadding = dp(5f)
    private val rowGap = dp(3f)
    private val buttonWidth = dp(48f)
    private val buttonHeight = dp(17f)
    private val buttonGap = dp(4f)

    /** 控件行的上沿（与 onSizeChanged / onDraw 共用同一套算式，改一处就得改两处）。 */
    private val controlsTop: Float
        get() = topPadding + textPaint.textSize + rowGap

    // 开关：一条小轨道 + 滑块 + 右边标签（自己画，见 drawToggle）
    private val trackWidth = dp(30f)
    private val trackHeight = dp(14f)
    private val trackGap = dp(4f)

    private val background = Paint().apply { color = 0xE60E1216.toInt() }

    private val divider = Paint().apply { color = 0x33FFFFFF }

    // 只有一行文字：盘面信息和状态用 " · " 拼起来，超出就截断（剩下的看日志/App）
    // TextPaint 而不是 Paint：TextUtils.ellipsize 要的是 TextPaint
    private val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = dp(11f)
    }

    private val buttonFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF2A6FDB.toInt() }
    private val buttonFillPressed = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF1B4E9B.toInt() }
    private val buttonFillClose = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF3A424B.toInt() }
    private val buttonFillClosePressed = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF23282E.toInt() }
    private val buttonText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = dp(13f)
        textAlign = Paint.Align.CENTER
    }

    /** 开关标签：轨道右边的小字，颜色跟着开/关变。 */
    private val toggleText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF9AA4AE.toInt()
        textSize = dp(12f)
        textAlign = Paint.Align.LEFT
    }

    /** 耗时文本：右对齐、比状态行亮一点，和左侧的普通状态分开看。 */
    private val timingPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF6FD1FF.toInt()
        textSize = dp(11f)
        textAlign = Paint.Align.RIGHT
    }

    /** 开关轨道与滑块的填充（颜色每次绘制前按状态设）。 */
    private val trackFill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val thumbFill = Paint(Paint.ANTI_ALIAS_FLAG)

    /** 开关被按住时的底衬，点着有反馈。 */
    private val togglePressed = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x1FFFFFFF }

    private val refreshRect = RectF()
    private val autoRect = RectF()
    private val infoRect = RectF()
    private val fastRect = RectF()
    private val continueRect = RectF()
    private val rect = RectF()

    /** 开关的宽度：轨道 + 标题文字按实际宽度算，窄屏上别把标签挤掉。 */
    private var autoToggleWidth = 0f
    private var infoToggleWidth = 0f
    private var fastToggleWidth = 0f
    private var continueToggleWidth = 0f

    /** 当前按下的控件：[REFRESH]/[INFO]/[FAST]/[AUTO]/[CONTINUE]；-1 = 没有。 */
    private var pressed = -1

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        // 高度就是内容高度：**不要**把导航栏高度加进来。铺到屏幕最下沿会盖住系统手势区，
        // 上滑回桌面/多任务会被这个窗口吃掉（实测踩过）。让窗口整体上移交给 OverlayController 做。
        setMeasuredDimension(MeasureSpec.getSize(widthMeasureSpec), heightPx)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        val top = controlsTop
        val bottom = top + buttonHeight
        autoToggleWidth = trackWidth + trackGap + toggleText.measureText(AUTO_LABEL)
        infoToggleWidth = trackWidth + trackGap + toggleText.measureText(INFO_LABEL)
        fastToggleWidth = trackWidth + trackGap + toggleText.measureText(FAST_LABEL)
        continueToggleWidth = trackWidth + trackGap + toggleText.measureText(CONTINUE_LABEL)

        // 从左往右：信息 → 续玩 → 快速 → 自动 → 刷新
        // 关闭与"定位算法"都移回 App 了，这里只留操作/调试开关
        //
        // **按槽位均分**而不是"右对齐 + 均分间隔"：各个控件宽度不一样（按钮 48dp、开关是个
        // 轨道+两字标签），只均分间隔的话视觉重心还是偏右。切成等宽槽位后每个控件在自己的槽里
        // 居中，屏幕上看着就是均匀的一排；槽位比控件窄时（窄屏/大字号）就压到槽宽，宁可紧贴
        // 也不叠到隔壁槽。
        val slot = (w - padding * 2) / TOGGLE_COUNT.toFloat()
        fun put(target: RectF, itemWidth: Float, index: Int) {
            val width = itemWidth.coerceAtMost(slot)
            val center = padding + slot * (index + 0.5f)
            target.set(center - width / 2f, top, center + width / 2f, bottom)
        }
        put(infoRect, infoToggleWidth, 0)
        put(continueRect, continueToggleWidth, 1)
        put(fastRect, fastToggleWidth, 2)
        put(autoRect, autoToggleWidth, 3)
        put(refreshRect, buttonWidth, 4)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), background)
        canvas.drawRect(0f, 0f, width.toFloat(), dp(1f), divider)

        // 一行文字：出错时整行红色并只显示原因；否则"盘面信息 · 状态"拼起来
        val st = state
        val line = if (st?.message != null) {
            textPaint.color = 0xFFFF6B6B.toInt()
            st.message.orEmpty()
        } else {
            textPaint.color = 0xFFFFFFFF.toInt()
            listOfNotNull(
                st?.hintText?.takeIf { it.isNotBlank() },
                st?.statusText?.takeIf { it.isNotBlank() },
            ).joinToString(" · ")
        }
        val timingW = if (timingText.isEmpty()) 0f else timingPaint.measureText(timingText)
        val lineW = (width - padding * 2 - (if (timingW > 0f) timingW + dp(10f) else 0f))
            .coerceAtLeast(dp(60f))
        val baseline = topPadding + textPaint.textSize
        canvas.drawText(
            TextUtils.ellipsize(line, textPaint, lineW, TextUtils.TruncateAt.END).toString(),
            padding,
            baseline,
            textPaint,
        )
        if (timingW > 0f) {
            canvas.drawText(timingText, width - padding, baseline, timingPaint)
        }

        drawToggle(canvas, autoRect, AUTO_LABEL, autoOn, pressed == AUTO)
        drawToggle(canvas, fastRect, FAST_LABEL, fastOn, pressed == FAST)
        drawToggle(canvas, infoRect, INFO_LABEL, infoOn, pressed == INFO)
        drawToggle(canvas, continueRect, CONTINUE_LABEL, continueOn, pressed == CONTINUE)
        drawButton(canvas, refreshRect, "刷新", pressed == REFRESH, false)
    }

    private fun drawButton(canvas: Canvas, bounds: RectF, label: String, isPressed: Boolean, isClose: Boolean) {
        rect.set(bounds)
        val fill = when {
            isClose && isPressed -> buttonFillClosePressed
            isClose -> buttonFillClose
            isPressed -> buttonFillPressed
            else -> buttonFill
        }
        canvas.drawRoundRect(rect, dp(8f), dp(8f), fill)
        val baseline = rect.centerY() - (buttonText.ascent() + buttonText.descent()) / 2f
        canvas.drawText(label, rect.centerX(), baseline, buttonText)
    }

    /**
     * 开关（**不是按钮**，是按状态显形的 toggle）：左边一条轨道 + 滑块，右边标签。
     * 亮蓝轨道 + 滑块靠右 = 开，深灰轨道 + 滑块靠左 = 关；按住时加一层浅底衬。
     * 命中区是整块（含标签），点哪都算。
     */
    private fun drawToggle(canvas: Canvas, bounds: RectF, label: String, on: Boolean, isPressed: Boolean) {
        if (isPressed) {
            rect.set(bounds)
            canvas.drawRoundRect(rect, dp(6f), dp(6f), togglePressed)
        }

        trackFill.color = when {
            on && isPressed -> buttonFillPressed.color
            on -> buttonFill.color
            isPressed -> buttonFillClosePressed.color
            else -> buttonFillClose.color
        }
        val trackTop = bounds.centerY() - trackHeight / 2f
        rect.set(bounds.left, trackTop, bounds.left + trackWidth, trackTop + trackHeight)
        canvas.drawRoundRect(rect, trackHeight / 2f, trackHeight / 2f, trackFill)

        val radius = trackHeight / 2f - dp(2f)
        val centerX = if (on) {
            bounds.left + trackWidth - radius - dp(2f)
        } else {
            bounds.left + radius + dp(2f)
        }
        thumbFill.color = if (on) 0xFFFFFFFF.toInt() else 0xFF9AA4AE.toInt()
        canvas.drawCircle(centerX, bounds.centerY(), radius, thumbFill)

        toggleText.color = if (on) 0xFFFFFFFF.toInt() else 0xFF9AA4AE.toInt()
        val baseline = bounds.centerY() - (toggleText.ascent() + toggleText.descent()) / 2f
        canvas.drawText(label, bounds.left + trackWidth + trackGap, baseline, toggleText)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val hit = hitTest(event.x, event.y)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                pressed = hit
                invalidate()
                return hit >= 0
            }

            MotionEvent.ACTION_MOVE -> {
                if (hit != pressed) {
                    pressed = hit
                    invalidate()
                }
                return true
            }

            MotionEvent.ACTION_UP -> {
                val was = pressed
                pressed = -1
                invalidate()
                if (hit >= 0 && hit == was) {
                    when (hit) {
                        REFRESH -> onRefresh?.invoke()
                        // 自动**不在这里翻本地状态**：它会自己停下（通关/无解/手势没生效），
                        // 本地翻完就没机会被纠正了。只报"期望的新状态"，由会话回写 autoOn。
                        AUTO -> onToggleAuto?.invoke(!autoOn)
                        FAST -> {
                            fastOn = !fastOn
                            onToggleFast?.invoke(fastOn)
                        }
                        // 纯显示开关，立刻翻 + 回调，上层拿信号即时重画提示层
                        INFO -> {
                            infoOn = !infoOn
                            onToggleInfo?.invoke(infoOn)
                        }
                        CONTINUE -> {
                            continueOn = !continueOn
                            onToggleContinue?.invoke(continueOn)
                        }
                    }
                    performClick()
                }
                return true
            }

            MotionEvent.ACTION_CANCEL -> {
                pressed = -1
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean = super.performClick()

    private fun hitTest(x: Float, y: Float): Int = when {
        refreshRect.contains(x, y) -> REFRESH
        autoRect.contains(x, y) -> AUTO
        fastRect.contains(x, y) -> FAST
        infoRect.contains(x, y) -> INFO
        continueRect.contains(x, y) -> CONTINUE
        else -> NONE
    }

    private companion object {
        /** 控件行的个数（续玩 / 信息 / 自动 / 快速 / 刷新），槽位按它均分。 */
        const val TOGGLE_COUNT = 5

        const val NONE = -1
        const val REFRESH = 0
        const val AUTO = 4
        const val FAST = 5
        const val INFO = 2
        const val CONTINUE = 6

        const val AUTO_LABEL = "自动"
        const val FAST_LABEL = "快速"
        const val INFO_LABEL = "信息"
        const val CONTINUE_LABEL = "续玩"
    }
}
