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
    private val buttonGap = dp(6f)

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

    private val refreshRect = RectF()
    private val closeRect = RectF()
    private val rect = RectF()

    /** 当前按下的按钮：0=刷新，1=关闭，-1=没有。 */
    private var pressed = -1

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        // 高度就是内容高度：**不要**把导航栏高度加进来。铺到屏幕最下沿会盖住系统手势区，
        // 上滑回桌面/多任务会被这个窗口吃掉（实测踩过）。让窗口整体上移交给 OverlayController 做。
        setMeasuredDimension(MeasureSpec.getSize(widthMeasureSpec), heightPx)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        val top = topPadding
        closeRect.set(width - padding - buttonWidth, top, width - padding, top + buttonHeight)
        refreshRect.set(closeRect.left - buttonGap - buttonWidth, top, closeRect.left - buttonGap, top + buttonHeight)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), background)
        canvas.drawRect(0f, 0f, width.toFloat(), dp(1f), divider)

        val textRight = refreshRect.left - buttonGap
        val textWidth = (textRight - padding).coerceAtLeast(dp(40f))

        val hint = state?.hintText.orEmpty()
        canvas.drawText(
            TextUtils.ellipsize(hint, hintPaint, textWidth, TextUtils.TruncateAt.END).toString(),
            padding,
            topPadding + hintPaint.textSize,
            hintPaint,
        )

        val status = state?.message ?: state?.statusText.orEmpty()
        statusPaint.color = if (state?.message != null) 0xFFFF6B6B.toInt() else 0xFF9AA4AE.toInt()
        canvas.drawText(
            TextUtils.ellipsize(status, statusPaint, width - 2 * padding, TextUtils.TruncateAt.END).toString(),
            padding,
            topPadding + hintPaint.textSize + lineGap + statusPaint.textSize,
            statusPaint,
        )

        drawButton(canvas, refreshRect, "刷新", pressed == 0, false)
        drawButton(canvas, closeRect, "关闭", pressed == 1, true)
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
                    if (hit == 0) onRefresh?.invoke() else onClose?.invoke()
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
        refreshRect.contains(x, y) -> 0
        closeRect.contains(x, y) -> 1
        else -> -1
    }
}
