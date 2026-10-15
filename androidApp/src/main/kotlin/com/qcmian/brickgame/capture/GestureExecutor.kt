package com.qcmian.brickgame.capture

/**
 * 派发手势 —— 自动运行唯一的"出手"通道。
 *
 * 抽成接口而不是直接调无障碍 API：一是自动运行的循环需要能被离线驱动，
 * 二是手势的成败必须**有回调**（`onDone`），否则一旦系统把这次手势吃掉，
 * 循环会拿着一个没变的盘面反复求解、原地打转。
 *
 * 两种手势对应游戏的两种操作：
 *  - [tapPair]：两块同款砖**已经同线且中间没有别的砖**时的直消 —— 点第一块是选中，
 *    点第二块才确定"消哪一对"；
 *  - [swipe]：其余情况把一块砖滑到落点，靠"两块对齐"来消。
 *
 * 注意这里不是 `fun interface`：两个抽象方法，SAM 转换用不了。
 */
interface GestureExecutor {

    /**
     * 依次点两块砖（同一条手势里的两次点击）。
     *
     * **为什么要两次而不是一次**：一块砖可能同时和**左右（或上下）两块同款砖**同线且中间无砖，
     * 这时只点它一下，游戏不知道要消哪一对；必须再点一下指定那一块。
     * 只点在唯一配对的情况下也能用 —— 第二下落在已经是空位的那格上，没有任何副作用。
     *
     * @param gapMs 两次点击之间的间隔：太短会被游戏当成一次双击/连点，给足"选中 → 消掉"的时间。
     * @param onDone 是否被系统成功派发并执行完（`false` = 被取消或派发失败）。
     *   它只代表"手势出去了"，不代表游戏一定消了牌 —— 那是下一帧复检的事。
     */
    fun tapPair(
        x1: Float, y1: Float, x2: Float, y2: Float,
        tapMs: Long, gapMs: Long, onDone: (Boolean) -> Unit,
    )

    /**
     * 从 (x1,y1) 滑到 (x2,y2)。
     *
     * @param durationMs 按住时长。太短会被当成 fling/滚动丢掉，太长则拖慢每一步。
     */
    fun swipe(x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long, onDone: (Boolean) -> Unit)
}
