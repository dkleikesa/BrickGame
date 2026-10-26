package com.qcmian.brickgame.algorithm

/**
 * 定长 Long 数组，只暴露"读"和"CAS"两种操作。
 *
 * 求解器的死表要跨线程共享：多线程写同一个槽时，只有一个能成功（CAS），
 * 不会互相覆盖成半个值。这里抽成 expect/actual，是为了 common 层不出现 java.*。
 */
internal expect class AtomicLongTable(size: Int) {
    fun get(index: Int): Long
    fun compareAndSet(index: Int, expected: Long, update: Long): Boolean
    fun set(index: Int, value: Long)
}
