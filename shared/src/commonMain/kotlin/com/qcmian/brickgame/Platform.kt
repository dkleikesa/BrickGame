package com.qcmian.brickgame

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform