package com.qcmian.brickgame

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.view.WindowManager
import androidx.compose.foundation.Image
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.qcmian.brickgame.capture.BrickGameAccessibilityService
import com.qcmian.brickgame.vision.OpenCvBoardScanner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 一条链路的操作台：授权 → 启停 → 看提示 → 看调试图（核对标定）。
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
        var openCvReady by remember { mutableStateOf<Boolean?>(null) }

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
                    StatusLine("运行状态", if (ui.running) "提示中（第 ${ui.frames} 帧）" else "已停止")
                    StatusLine(
                        "悬浮窗",
                        when (HintUiBus.overlayWindowType) {
                            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY -> "无障碍内置（免授权）"
                            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY -> "应用上层（需授权）"
                            else -> "未挂上"
                        },
                    )
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    modifier = Modifier.weight(1f),
                    onClick = { context.startActivity(accessibilitySettingsIntent(context)) },
                ) { Text(if (connected) "无障碍设置" else "去开启服务") }

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
                            if (started) {
                                HintUiBus.serviceMessage = null
                            } else if (isAccessibilityServiceEnabled(context)) {
                                // 系统里开着、进程里没连上：**不要**跳设置页，那是误导
                                HintUiBus.serviceMessage =
                                    "无障碍服务在系统里是开着的，但当前进程还没连上（App 刚被系统重启过）。" +
                                        "稍等几秒再点一次；一直不行就在设置里把它关掉再打开一次。"
                            } else {
                                context.startActivity(accessibilitySettingsIntent(context))
                            }
                        }
                    },
                ) { Text(if (ui.running) "停止提示" else if (busy) "连接中…" else "开始提示") }
            }

            OutlinedButton(
                modifier = Modifier.fillMaxWidth(),
                enabled = !busy,
                onClick = {
                    busy = true
                    scope.launch {
                        val ok = startHintingWithRetry()
                        busy = false
                        if (!ok) {
                            HintUiBus.serviceMessage =
                                if (isAccessibilityServiceEnabled(context)) {
                                    "服务还没连上，稍后再试。"
                                } else {
                                    "还没开启无障碍服务。"
                                }
                        }
                    }
                },
            ) { Text("验证一次（跑一帧看调试图）") }

            HintUiBus.serviceMessage?.let { message ->
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
                    Text(
                        message,
                        modifier = Modifier.padding(14.dp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                }
            }

            if (HintUiBus.needsOverlayPermission) {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            "悬浮窗被系统拒了",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                        )
                        Text(
                            "本机不吃无障碍专属的悬浮窗类型。允许「显示在其他应用上层」后会降级到普通应用悬浮窗，效果一样。",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                        )
                        Button(
                            onClick = {
                                context.startActivity(
                                    Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION)
                                        .setData(Uri.parse("package:${context.packageName}"))
                                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                                )
                            },
                        ) { Text("去授权") }
                    }
                }
            }

            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("提示", style = MaterialTheme.typography.labelLarge)
                    Text(
                        ui.hintText.ifBlank { "还没有结果" },
                        style = MaterialTheme.typography.titleMedium,
                    )
                    if (ui.scanSummary.isNotBlank()) {
                        Text(
                            ui.scanSummary,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = HintUiBus.showGrid, onCheckedChange = { HintUiBus.showGrid = it })
                Text("显示网格", Modifier.padding(start = 8.dp), style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.weight(1f))
                Switch(checked = HintUiBus.showLabels, onCheckedChange = { HintUiBus.showLabels = it })
                Text("显示类别号", Modifier.padding(start = 8.dp), style = MaterialTheme.typography.bodyMedium)
            }
            Text(
                "这两个是调试图层，默认关；打开后会标出所有格子（用来核对标定/识别准不准）。" +
                    "平时屏幕上只有能消的那几格：绿框=要滑的、蓝框=落点、橙框=配对砖。",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            ui.message?.let {
                Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium)
            }

            HintUiBus.preview?.let { bitmap ->
                Text("调试图（绿框=要滑动的那块，蓝框=落点，橙框=配对砖）", style = MaterialTheme.typography.labelMedium)
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "调试图",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp)),
                )
            }

            Box(Modifier.height(24.dp))
        }
    }
}

private const val START_TRIES = 10
private const val START_RETRY_MS = 300L

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
