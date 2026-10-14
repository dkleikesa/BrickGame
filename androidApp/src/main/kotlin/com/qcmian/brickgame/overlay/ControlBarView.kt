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
 * 内容从左到右：提示文字（两行）→ 两个调试开关（网格 / 类别号，按状态显形的 toggle）→ 刷新 → 关闭。
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
    var onClose: (() -> Unit)? = null

    /** 调试开关：网格 / 类别号。点一下立刻回调，由上层即时重画（不用等下一次刷新）。 */
    var onToggleGrid: ((Boolean) -> Unit)? = null
    var onToggleLabels: ((Boolean) -> Unit)? = null

    /** 两个调试开关的显示状态（写进去就重绘）。 */
    var showGrid: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    var showLabels: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    private val density = resources.displayMetrics.density
    private fun dp(v: Float) = v * density

    // 高度取原来的一半（72dp → 38dp）：栏里的两行字与按钮都按新高度重排过，
    // 8 + 13 + 2 + 10 + 8 ≈ 41 的账算下来 38dp 装得下，别只改高度不改字号。
    private val heightPx = dp(38f).toInt()
    private val padding = dp(8f)
    private val topPadding = dp(6f)
    private val lineGap = dp(2f)
    private val buttonWidth = dp(50f)
    private val buttonHeight = dp(26f)
    private val buttonGap = dp(5f)

    // 开关：一条小轨道 + 滑块 + 右边标签（自己画，见 drawToggle）
    private val trackWidth = dp(26f)
    private val trackHeight = dp(16f)
    private val trackGap = dp(4f)

    private val background = Paint().apply { color = 0xE60E1216.toInt() }

    private val divider = Paint().apply { color = 0x33FFFFFF }

    // TextPaint 而不是 Paint：TextUtils.ellipsize 要的是 TextPaint
    private val hintPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = dp(13f)
        isFakeBoldText = true
    }

    private val statusPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF9AA4AE.toInt()
        textSize = dp(10f)
    }

    private val buttonFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF2A6FDB.toInt() }
    private val buttonFillPressed = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF1B4E9B.toInt() }
    private val buttonFillClose = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF3A424B.toInt() }
    private val buttonFillClosePressed = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF23282E.toInt() }
    private val buttonText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = dp(12f)
        textAlign = Paint.Align.CENTER
    }

    /** 开关标签：轨道右边的小字，颜色跟着开/关变。 */
    private val toggleText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF9AA4AE.toInt()
        textSize = dp(11f)
        textAlign = Paint.Align.LEFT
    }

    /** 开关轨道与滑块的填充（颜色每次绘制前按状态设）。 */
    private val trackFill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val thumbFill = Paint(Paint.ANTI_ALIAS_FLAG)

    /** 开关被按住时的底衬，点着有反馈。 */
    private val togglePressed = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x1FFFFFFF }

    private val refreshRect = RectF()
    private val closeRect = RectF()
    private val gridRect = RectF()
    private val labelsRect = RectF()
    private val rect = RectF()

    /** 两个开关的宽度：轨道 + 标题文字按实际宽度算，窄屏上别把标签挤掉。 */
    private var gridToggleWidth = 0f
    private var labelsToggleWidth = 0f

    /** 当前按下的控件：[REFRESH]/[CLOSE]/[GRID]/[LABELS]；-1 = 没有。 */
    private var pressed = -1

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        // 高度就是内容高度：**不要**把导航栏高度加进来。铺到屏幕最下沿会盖住系统手势区，
        // 上滑回桌面/多任务会被这个窗口吃掉（实测踩过）。让窗口整体上移交给 OverlayController 做。
        setMeasuredDimension(MeasureSpec.getSize(widthMeasureSpec), heightPx)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        val top = topPadding
        val bottom = top + buttonHeight
        gridToggleWidth = trackWidth + trackGap + toggleText.measureText(GRID_LABEL)
        labelsToggleWidth = trackWidth + trackGap + toggleText.measureText(LABELS_LABEL)
        closeRect.set(width - padding - buttonWidth, top, width - padding, bottom)
        refreshRect.set(closeRect.left - buttonGap - buttonWidth, top, closeRect.left - buttonGap, bottom)
        labelsRect.set(refreshRect.left - buttonGap - labelsToggleWidth, top, refreshRect.left - buttonGap, bottom)
        gridRect.set(labelsRect.left - buttonGap - gridToggleWidth, top, labelsRect.left - buttonGap, bottom)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), background)
        canvas.drawRect(0f, 0f, width.toFloat(), dp(1f), divider)

        val textRight = gridRect.left - buttonGap
        val textWidth = (textRight - padding).coerceAtLeast(dp(40f))

        val hint = state?.hintText.orEmpty()
        canvas.drawText(
            TextUtils.ellipsize(hint, hintPaint, textWidth, TextUtils.TruncateAt.END).toString(),
            padding,
            topPadding + hintPaint.textSize,
            hintPaint,
        )

        // 状态行也按提示行的宽度截断：按钮是画在后面的，写满整行会被按钮盖住半截
        val status = state?.message ?: state?.statusText.orEmpty()
        statusPaint.color = if (state?.message != null) 0xFFFF6B6B.toInt() else 0xFF9AA4AE.toInt()
        canvas.drawText(
            TextUtils.ellipsize(status, statusPaint, textWidth, TextUtils.TruncateAt.END).toString(),
            padding,
            topPadding + hintPaint.textSize + lineGap + statusPaint.textSize,
            statusPaint,
        )

        drawToggle(canvas, gridRect, GRID_LABEL, showGrid, pressed == GRID)
        drawToggle(canvas, labelsRect, LABELS_LABEL, showLabels, pressed == LABELS)
        drawButton(canvas, refreshRect, "刷新", pressed == REFRESH, false)
        drawButton(canvas, closeRect, "关闭", pressed == CLOSE, true)
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
                        CLOSE -> onClose?.invoke()
                        // 立刻翻状态并回调：上层拿这个信号即时重画提示层，不等下一次刷新
                        GRID -> {
                            showGrid = !showGrid
                            onToggleGrid?.invoke(showGrid)
                        }
                        LABELS -> {
                            showLabels = !showLabels
                            onToggleLabels?.invoke(showLabels)
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
        closeRect.contains(x, y) -> CLOSE
        gridRect.contains(x, y) -> GRID
        labelsRect.contains(x, y) -> LABELS
        else -> NONE
    }

    private companion object {
        const val NONE = -1
        const val REFRESH = 0
        const val CLOSE = 1
        const val GRID = 2
        const val LABELS = 3

        const val GRID_LABEL = "网格"
        const val LABELS_LABEL = "类别号"
    }
}
