package com.qcmian.brickgame

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview
import com.qcmian.brickgame.capture.BrickGameAccessibilityService

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent {
            HintApp()
        }
        maybeAutostart(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        maybeAutostart(intent)
    }

    /**
     * 带上 `--ez autostart true` 启动就直接开提示 —— 调试与自动化用：
     * `adb shell am start -n com.qcmian.brickgame/.MainActivity --ez autostart true`
     * 也可以给快捷方式 / Tasker 用；无障碍服务还没连上时等半秒再试一次。
     */
    private fun maybeAutostart(intent: Intent?) {
        if (intent?.getBooleanExtra(EXTRA_AUTOSTART, false) != true) return
        window.decorView.postDelayed({ startHintingWithRetry(0) }, 300)
    }

    private fun startHintingWithRetry(attempt: Int) {
        if (BrickGameAccessibilityService.start()) return
        // 服务没连上：界面会提示去设置里开启，这里只负责重试几次
        if (attempt < 5) window.decorView.postDelayed({ startHintingWithRetry(attempt + 1) }, 500)
    }

    override fun onStart() {
        super.onStart()
        // 界面可见时才渲染调试图：一帧 ARGB 位图不小，不值得在后台一直生成
        HintUiBus.previewEnabled = true
    }

    override fun onStop() {
        HintUiBus.previewEnabled = false
        super.onStop()
    }

    companion object {
        const val EXTRA_AUTOSTART = "autostart"
    }
}

@Preview
@Composable
fun HintAppPreview() {
    HintApp()
}
