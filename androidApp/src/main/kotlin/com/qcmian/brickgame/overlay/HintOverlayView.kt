package com.qcmian.brickgame.overlay

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View
import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.BoardState
import com.qcmian.brickgame.vision.CellRect
import com.qcmian.brickgame.vision.Hint
import kotlin.math.hypot
import kotlin.math.min

/**
 * 悬浮提示层：把"能消的那几格"画在游戏画面上。
 *
 * 设计原则：**屏幕上没有提示的地方一律不涂色**。整屏给每格上色（按识别类别）看着就是一团乱，
 * 而且会盖住游戏本身的元素。所以默认只画三件事：
 *  - 绿框 = 要滑动的那块（[Hint.from]）
 *  - 蓝框 = 落点（[Hint.to]，与起点重合时不画）
 *  - 橙框 = 配对砖（[Hint.pair]）
 *  - 一条从起点指向落点的箭头
 *
 * 网格和类别号是**调试图层**，默认关（界面上「显示网格」「显示类别号」开关）；
 * 只有那个开关打开时才会给所有砖块上色写字，用来核对标定/识别对不对。
 *
 * 坐标就是截图坐标（[BoardScan.rects] 用的也是它），窗口铺满全屏时 1:1 对齐，不需要缩放。
 */
class HintOverlayView(context: Context) : View(context) {

    data class Frame(
        /** 调试图预览时带上截图；实时悬浮窗为 null（游戏画面本来就在下面）。 */
        val background: Bitmap?,
        val scan: BoardScan,
        val state: BoardState,
        val hint: Hint?,
        /** 调试图层：画网格线。默认关。 */
        val showGrid: Boolean,
        /** 调试图层：给所有砖块上类别色 + 写类别号。默认关。 */
        val showLabels: Boolean,
    )

    var frame: Frame? = null
        set(value) {
            field = value
            invalidate()
        }

    private val rect = RectF()

    // ---- 调试图层（默认不开）----
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f
        color = 0x66FFFFFF
    }
    private val classFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val classStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }
    private val labelText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        color = Color.WHITE
    }
    private val labelHalo = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = 0xCC000000.toInt()
    }

    // ---- 提示本体 ----
    private val fromFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; color = 0x4429E06B }
    private val fromStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = 0xFF29E06B.toInt(); strokeWidth = 3f; strokeJoin = Paint.Join.ROUND
    }
    private val toFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; color = 0x4435D0FF }
    private val toStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = 0xFF35D0FF.toInt(); strokeWidth = 3f; strokeJoin = Paint.Join.ROUND
    }
    private val pairFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL; color = 0x33FFB020 }
    private val pairStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = 0xFFFFB020.toInt(); strokeWidth = 2.5f; strokeJoin = Paint.Join.ROUND
    }

    private val arrowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = 0xFF29E06B.toInt()
        strokeCap = Paint.Cap.ROUND
    }
    private val arrowHead = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xFF29E06B.toInt()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame ?: return
        f.background?.let { canvas.drawBitmap(it, 0f, 0f, null) }
        if (f.scan.rows == 0 || f.scan.cols == 0) return

        // 只有调试图层打开时才给"所有格子"上色；默认屏幕上除了提示那几格没有别的涂色
        if (f.showLabels) drawClasses(canvas, f)
        if (f.showGrid) drawGrid(canvas, f)
        f.hint?.let { drawHint(canvas, it) }
    }

    private fun drawGrid(canvas: Canvas, f: Frame) {
        val top = f.scan.ylines.first().toFloat()
        val bottom = f.scan.ylines.last().toFloat()
        val left = f.scan.xlines.first().toFloat()
        val right = f.scan.xlines.last().toFloat()
        for (x in f.scan.xlines) canvas.drawLine(x.toFloat(), top, x.toFloat(), bottom, gridPaint)
        for (y in f.scan.ylines) canvas.drawLine(left, y.toFloat(), right, y.toFloat(), gridPaint)
    }

    /** 调试用：所有砖块按类别上色 + 写类别号。 */
    private fun drawClasses(canvas: Canvas, f: Frame) {
        val first = f.scan.rects.getOrNull(0) ?: return
        labelText.textSize = min(first.width, first.height) * 0.34f
        labelHalo.textSize = labelText.textSize

        for (index in 0 until f.scan.cellCount) {
            if (f.scan.empty[index]) continue
            val r = f.scan.rects[index]
            rect.set(r.left.toFloat(), r.top.toFloat(), r.right.toFloat(), r.bottom.toFloat())
            val color = classColor(f.state.labels[index])
            classFill.color = (color and 0x00FFFFFF) or 0x33000000
            classStroke.color = (color and 0x00FFFFFF) or 0x99000000.toInt()
            canvas.drawRect(rect, classFill)
            canvas.drawRect(rect, classStroke)

            val x = r.centerX.toFloat()
            val y = r.centerY - (labelText.ascent() + labelText.descent()) / 2f
            canvas.drawText(f.state.labels[index].toString(), x, y, labelHalo)
            canvas.drawText(f.state.labels[index].toString(), x, y, labelText)
        }
    }

    private fun drawHint(canvas: Canvas, hint: Hint) {
        drawMarker(canvas, hint.pairRect, pairFill, pairStroke)
        drawMarker(canvas, hint.fromRect, fromFill, fromStroke)
        if (hint.to != hint.from) drawMarker(canvas, hint.toRect, toFill, toStroke)
        drawArrow(canvas, hint.fromRect, hint.toRect)
    }

    private fun drawMarker(canvas: Canvas, r: CellRect, fill: Paint, stroke: Paint) {
        val inset = 3f
        rect.set(r.left + inset, r.top + inset, r.right - inset, r.bottom - inset)
        canvas.drawRoundRect(rect, 10f, 10f, fill)
        canvas.drawRoundRect(rect, 10f, 10f, stroke)
    }

    private fun drawArrow(canvas: Canvas, from: CellRect, to: CellRect) {
        val x0 = from.centerX.toFloat()
        val y0 = from.centerY.toFloat()
        val x1 = to.centerX.toFloat()
        val y1 = to.centerY.toFloat()
        if (x0 == x1 && y0 == y1) return

        val len = hypot(x1 - x0, y1 - y0)
        val ux = (x1 - x0) / len
        val uy = (y1 - y0) / len
        val startX = x0 + ux * from.width * 0.5f
        val startY = y0 + uy * from.height * 0.5f
        val endX = x0 + ux * (len - to.width * 0.5f)
        val endY = y0 + uy * (len - to.height * 0.5f)
        canvas.drawLine(startX, startY, endX, endY, arrowPaint)

        val head = min(maxOf(from.width, to.width) * 0.28f, 24f)
        val px = -uy
        val py = ux
        val path = Path().apply {
            moveTo(endX, endY)
            lineTo(endX - ux * head + px * head * 0.5f, endY - uy * head + py * head * 0.5f)
            lineTo(endX - ux * head - px * head * 0.5f, endY - uy * head - py * head * 0.5f)
            close()
        }
        canvas.drawPath(path, arrowHead)
    }

    /** 类别调色板：只为了肉眼区分，颜色本身没有语义。 */
    private fun classColor(label: Int): Int = PALETTE[(label - 1).mod(PALETTE.size)]

    private companion object {
        val PALETTE = intArrayOf(
            0xFFE6194B.toInt(), 0xFF3CB44B.toInt(), 0xFFFFE119.toInt(), 0xFF4363D8.toInt(),
            0xFFF58231.toInt(), 0xFF911EB4.toInt(), 0xFF46F0F0.toInt(), 0xFFF032E6.toInt(),
            0xFFBCF60C.toInt(), 0xFFFABEBE.toInt(), 0xFF008080.toInt(), 0xFFE6BEFF.toInt(),
        )
    }
}
