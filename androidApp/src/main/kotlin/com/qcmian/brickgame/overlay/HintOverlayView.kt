package com.qcmian.brickgame.overlay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import com.qcmian.brickgame.vision.BoardScan
import com.qcmian.brickgame.vision.BoardState
import com.qcmian.brickgame.vision.CellRect
import com.qcmian.brickgame.vision.Hint
import kotlin.math.min

/**
 * 悬浮提示层：把"能消的那几对"画在游戏画面上。
 *
 * 设计原则：**屏幕上没有提示的地方一律不涂色**。整屏给每格上色（按识别类别）看着就是一团乱，
 * 而且会盖住游戏本身的元素。所以默认只画提示那几格：
 *  - 全部统一**红色 50% 透明度**的框，尺寸内缩到跟砖块一样大；
 *  - 同一对的两块砖上写**同一个序号**（颜色都一样，靠序号认对）；
 *  - **不画箭头**：玩家只要知道哪两块可以消，滑哪边由游戏自己判定。
 *
 * 网格和类别号是**调试图层**，默认关（悬浮条上的「网格」「类别号」开关，点了立刻重画）；
 * 只有那个开关打开时才会给所有砖块上色写字，用来核对标定/识别对不对。
 *
 * 坐标就是截图坐标（[BoardScan.rects] 用的也是它），窗口铺满全屏时 1:1 对齐，不需要缩放。
 */
class HintOverlayView(context: Context) : View(context) {

    data class Frame(
        val scan: BoardScan,
        val state: BoardState,
        /** 当前盘面上全部可消除的对，一对画两个同色框。 */
        val hints: List<Hint>,
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

    // ---- 提示本体：统一红色，填充 50%、描边实心 ----
    private val hintFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        // 必须先用 0x00FFFFFF 把 HINT_COLOR 自带的 0xFF alpha 抹掉，再或上 0x80 ——
        // 直接 `HINT_COLOR or 0x80000000` 的 alpha 仍是 0xFF（或运算只能加位不能减），
        // 画出来是纯红不是半透明（真机上量过像素：#FFFF0000）。
        color = (HINT_COLOR and 0x00FFFFFF) or HINT_FILL_ALPHA
    }
    private val hintStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        strokeJoin = Paint.Join.ROUND
        color = HINT_COLOR
    }

    /** 对序号（两块钱上写同一个数字），对多了以后靠它而不是靠颜色认对。 */
    private val pairText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        style = Paint.Style.FILL
        color = Color.WHITE
        isFakeBoldText = true
    }
    private val pairHalo = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = 0xCC000000.toInt()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame ?: return
        if (f.scan.rows == 0 || f.scan.cols == 0) return

        // 只有调试图层打开时才给"所有格子"上色；默认屏幕上除了提示那几格没有别的涂色
        if (f.showLabels) drawClasses(canvas, f)
        if (f.showGrid) drawGrid(canvas, f)
        drawHints(canvas, f)
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

    /**
     * 画出全部可消除的对：全部统一红色框，两块砖上写**类别号**。
     *
     * 以前写的是"第几对"（1、2、3…），和砖块本身没关系，用户没法拿它跟盘面对照；
     * 类别号才是识别模型给的编号，能直接说"编号 1 的斧子"。
     *
     * 只标 [Hint.fromRect] 与 [Hint.pairRect] 这两块砖（就是会被消掉的两块）；
     * 不画落点、也不画箭头（滑哪边由游戏自己判定）。
     */
    private fun drawHints(canvas: Canvas, f: Frame) {
        val hints = f.hints
        if (hints.isEmpty()) return
        val first = hints.first().fromRect
        pairText.textSize = min(first.width, first.height) * 0.5f
        pairHalo.textSize = pairText.textSize
        for (hint in hints) {
            drawMarker(canvas, hint.fromRect)
            drawMarker(canvas, hint.pairRect)
            // 两块是同款，类别号必然一样；写同一个数字正好说明"它们是一对"
            val cls = f.state.labels[f.scan.indexOf(hint.from.row, hint.from.col)]
            val label = cls.toString()
            drawPairIndex(canvas, hint.fromRect, label)
            drawPairIndex(canvas, hint.pairRect, label)
        }
    }

    /**
     * 提示框缩到跟砖块一样大：格子比砖块大一圈（格子含砖块之间的缝），
     * 所以按格子尺寸按比例内缩，而不是写死几个像素 —— 不同分辨率/缩放下都跟砖块对齐。
     */
    private fun drawMarker(canvas: Canvas, r: CellRect) {
        val size = min(r.width, r.height).toFloat()
        val inset = size * BRICK_INSET
        rect.set(r.left + inset, r.top + inset, r.right - inset, r.bottom - inset)
        val radius = size * 0.18f
        canvas.drawRoundRect(rect, radius, radius, hintFill)
        canvas.drawRoundRect(rect, radius, radius, hintStroke)
    }

    /** 两块砖上写同一个类别号，位置固定在格子上沿，不挡中间的图案。 */
    private fun drawPairIndex(canvas: Canvas, r: CellRect, label: String) {
        val x = r.centerX.toFloat()
        val y = r.top + (r.height * 0.26f) - (pairText.ascent() + pairText.descent()) / 2f
        canvas.drawText(label, x, y, pairHalo)
        canvas.drawText(label, x, y, pairText)
    }


    /** 类别调色板：只为了肉眼区分，颜色本身没有语义。 */
    private fun classColor(label: Int): Int = PALETTE[(label - 1).mod(PALETTE.size)]

    private companion object {
        /** 砖块比格子小一圈（格子含砖块间的缝），提示框按这个比例内缩，缩到跟砖块一样大。 */
        const val BRICK_INSET = 0.06f

        /** 提示色（红）与填充透明度 50%（0x80）。 */
        const val HINT_COLOR = 0xFFFF0000.toInt()
        const val HINT_FILL_ALPHA = 0x80000000.toInt()


        val PALETTE = intArrayOf(
            0xFFE6194B.toInt(), 0xFF3CB44B.toInt(), 0xFFFFE119.toInt(), 0xFF4363D8.toInt(),
            0xFFF58231.toInt(), 0xFF911EB4.toInt(), 0xFF46F0F0.toInt(), 0xFFF032E6.toInt(),
            0xFFBCF60C.toInt(), 0xFFFABEBE.toInt(), 0xFF008080.toInt(), 0xFFE6BEFF.toInt(),
        )
    }
}
