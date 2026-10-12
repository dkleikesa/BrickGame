package com.example.brickgame

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform