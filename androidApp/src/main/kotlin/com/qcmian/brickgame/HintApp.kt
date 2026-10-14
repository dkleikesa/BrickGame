package com.qcmian.brickgame

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.qcmian.brickgame.capture.BrickGameAccessibilityService
import com.qcmian.brickgame.capture.LowConfStore
import com.qcmian.brickgame.ml.TfliteTileClassifier
import com.qcmian.brickgame.vision.OpenCvBoardScanner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 一条链路的操作台：授权 → 启停，剩下的（提示、帧信息、调试开关）都在屏幕底部的悬浮条上。
 *
 * 真正的截屏/识别/绘制都在无障碍服务里跑，这里只是它的面板 ——
 * 所以界面退到后台、甚至被杀掉，悬浮窗提示都不受影响。
 */
@Composable
fun HintApp() {
    MaterialTheme() {
        val context = LocalContext.current
        val scope = rememberCoroutineScope()
        val ui = HintUiBus.state
        var connected by remember { mutableStateOf(BrickGameAccessibilityService.isConnected()) }
        var enabledInSystem by remember { mutableStateOf(isAccessibilityServiceEnabled(context)) }
        var busy by remember { mutableStateOf(false) }
        var sharing by remember { mutableStateOf(false) }
        var deleting by remember { mutableStateOf(false) }
        var confirmDelete by remember { mutableStateOf(false) }
        // 放大看的哪张留档（null = 没开大图）
        var zoom by remember { mutableStateOf<File?>(null) }
        var openCvReady by remember { mutableStateOf<Boolean?>(null) }
        // 识别只跑 GPU delegate；本机不支持时会退到 CPU，界面照实说明（见 TfliteTileClassifier）
        val gpuReady = remember { TfliteTileClassifier.gpuSupported() }

        LaunchedEffect(Unit) {
            while (true) {
                connected = BrickGameAccessibilityService.isConnected()
                enabledInSystem = isAccessibilityServiceEnabled(context)
                delay(800)
            }
        }
        LaunchedEffect(Unit) {
            openCvReady = withContext(Dispatchers.Default) { OpenCvBoardScanner.ensureNativeLoaded() }
        }
        // 已经有多少低置信度留档：列一次目录（之后服务每存/删一次自己刷新，见 LowConfStore）
        LaunchedEffect(Unit) {
            withContext(Dispatchers.IO) { LowConfStore.refresh(context) }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .systemBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("BrickGame 提示", style = MaterialTheme.typography.headlineSmall)

            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    StatusLine(
                        "无障碍服务",
                        when {
                            connected -> "已连接"
                            // 系统里开着 ≠ 进程里连上了：App 被系统重启后要重新绑定，会有一两秒空窗
                            enabledInSystem -> "系统里已开启，尚未连上"
                            else -> "未开启"
                        },
                    )
                    StatusLine("OpenCV", when (openCvReady) { null -> "加载中…"; true -> "已加载"; false -> "加载失败" })
                    StatusLine("识别实现", if (gpuReady) "TFLite（GPU）" else "TFLite（CPU 回退）")
                    StatusLine("运行状态", if (ui.running) "提示中（第 ${ui.frames} 帧）" else "已停止")
                    StatusLine(
                        "悬浮窗",
                        when (HintUiBus.overlayWindowType) {
                            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY -> "无障碍内置（免授权）"
                            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY -> "应用上层（需授权）"
                            else -> "未挂上"
                        },
                    )
                    StatusLine(
                        "低置信度留档",
                        if (HintUiBus.lowConfCount == 0) "暂无" else "${HintUiBus.lowConfCount} 张",
                    )
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    modifier = Modifier.weight(1f),
                    onClick = { context.startActivity(accessibilitySettingsIntent(context)) },
                ) { Text(if (connected) "无障碍设置" else "开启无障碍服务") }

                Button(
                    modifier = Modifier.weight(1f),
                    enabled = !busy,
                    onClick = {
                        if (ui.running) {
                            BrickGameAccessibilityService.stop()
                            return@Button
                        }
                        busy = true
                        scope.launch {
                            val started = startHintingWithRetry()
                            busy = false
                            // 只有"系统里就没开过"才跳设置页；只是进程还没连上就把人扔过去是误导
                            if (!started && !isAccessibilityServiceEnabled(context)) {
                                context.startActivity(accessibilitySettingsIntent(context))
                            }
                        }
                    },
                ) { Text(if (ui.running) "停止提示" else if (busy) "连接中…" else "开始提示") }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    modifier = Modifier.weight(1f),
                    enabled = !sharing && !busy && HintUiBus.lowConfCount > 0,
                    onClick = {
                        sharing = true
                        scope.launch {
                            val zip = withContext(Dispatchers.IO) { LowConfStore.zipAll(context) }
                            sharing = false
                            if (zip == null) {
                                Log.w(TAG, "打包失败或没有可分享的留档")
                            } else {
                                shareZip(context, zip, HintUiBus.lowConfCount)
                            }
                        }
                    },
                ) {
                    Text(
                        when {
                            sharing -> "打包中…"
                            HintUiBus.lowConfCount == 0 -> "分享（暂无）"
                            else -> "分享（${HintUiBus.lowConfCount} 张 ZIP）"
                        },
                    )
                }

                OutlinedButton(
                    modifier = Modifier.weight(1f),
                    enabled = !deleting && HintUiBus.lowConfCount > 0,
                    onClick = { confirmDelete = true },
                ) { Text(if (deleting) "删除中…" else "删除留档") }
            }

            // 删了就找不回来（真机上的留档是补标注的唯一副本），先确认一次
            if (confirmDelete) {
                AlertDialog(
                    onDismissRequest = { confirmDelete = false },
                    title = { Text("删除全部留档？") },
                    text = {
                        Text(
                            "${HintUiBus.lowConfCount} 张低置信度截图连同说明文件会被删掉，不能恢复。" +
                                "已经分享出去的 ZIP 不受影响。",
                        )
                    },
                    confirmButton = {
                        TextButton(
                            onClick = {
                                confirmDelete = false
                                deleting = true
                                scope.launch {
                                    withContext(Dispatchers.IO) { LowConfStore.deleteAll(context) }
                                    deleting = false
                                }
                            },
                        ) { Text("删除") }
                    },
                    dismissButton = {
                        TextButton(onClick = { confirmDelete = false }) { Text("取消") }
                    },
                )
            }

            if (HintUiBus.lowConfShots.isNotEmpty()) {
                LowConfGallery(HintUiBus.lowConfShots) { zoom = it }
            }

            Box(Modifier.height(24.dp))
        }

        zoom?.let { file -> LowConfZoomDialog(file) { zoom = null } }
    }
}

private const val TAG = "HintApp"

private const val START_TRIES = 10
private const val START_RETRY_MS = 300L

/**
 * 把打包好的 ZIP 交给系统分享。
 *
 * 必须经 FileProvider 换成 `content://`：Android 7+ 直接给 `file://` 会抛 FileUriExposedException。
 * 另外补一份 `clipData` —— 有些应用只从里面取 URI，光给 EXTRA_STREAM 会拿不到读权限。
 */
private fun shareZip(context: Context, zip: File, frames: Int) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", zip)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "application/zip"
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_SUBJECT, "BrickGame 低置信度截图（$frames 张）")
        clipData = ClipData.newUri(context.contentResolver, "lowconf", uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "分享低置信度截图"))
}

/**
 * 点"开始提示"后的等待。
 *
 * App 进程被系统重启后，无障碍服务要由系统重新绑定，会有一两秒空窗。
 * 这期间直接判失败、把用户弹去设置页是误导（设置页里明明已经开着）。
 */
private suspend fun startHintingWithRetry(): Boolean {
    for (attempt in 0 until START_TRIES) {
        if (BrickGameAccessibilityService.start()) return true
        delay(START_RETRY_MS)
    }
    return false
}

private fun accessibilitySettingsIntent(context: Context): Intent =
    Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

/**
 * 系统侧是否启用了我们的无障碍服务。
 *
 * 注意这跟"进程里连上了"是两件事：读的是 `Settings.Secure` 的启用列表，
 * 所以 App 刚被重启、服务还没绑定时它照样返回 true —— 界面靠它区分这两种状态，
 * 避免"明明开着却被弹去设置页"。
 */
private fun isAccessibilityServiceEnabled(context: Context): Boolean {
    val raw = Settings.Secure.getString(
        context.contentResolver,
        Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
    ) ?: return false
    val target = "${context.packageName}/${BrickGameAccessibilityService::class.java.name}"
    return raw.split(':').any { entry ->
        entry.trim()
            .replace("${context.packageName}/.", "${context.packageName}/")
            .equals(target, ignoreCase = true)
    }
}

@Composable
private fun StatusLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Medium,
        )
    }
}
