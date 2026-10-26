package com.qcmian.brickgame.algorithm

import java.util.concurrent.atomic.AtomicLongArray

internal actual class AtomicLongTable actual constructor(size: Int) {
    private val array = AtomicLongArray(size)

    actual fun get(index: Int): Long = array.get(index)

    actual fun compareAndSet(index: Int, expected: Long, update: Long): Boolean =
        array.compareAndSet(index, expected, update)

    actual fun set(index: Int, value: Long) {
        array.set(index, value)
    }
}
