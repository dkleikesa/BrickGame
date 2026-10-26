package com.qcmian.brickgame

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.clickable
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.qcmian.brickgame.capture.BrickGameAccessibilityService
import com.qcmian.brickgame.capture.AutoContinue
import com.qcmian.brickgame.capture.LowConfStore
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
        // 调试开关从悬浮条挪到 App：控制条只留日常会用到的四个，这样每个能画得更宽
        var gridOn by remember { mutableStateOf(HintUiBus.showGrid) }
        var labelsOn by remember { mutableStateOf(HintUiBus.showLabels) }
        var confirmDelete by remember { mutableStateOf(false) }
        var showSettings by remember { mutableStateOf(false) }
        // 放大看的哪张留档（null = 没开大图）
        var zoom by remember { mutableStateOf<File?>(null) }

        LaunchedEffect(Unit) {
            while (true) {
                connected = BrickGameAccessibilityService.isConnected()
                enabledInSystem = isAccessibilityServiceEnabled(context)
                delay(800)
            }
        }
        // 已经有多少低置信度留档：列一次目录（之后服务每存/删一次自己刷新，见 LowConfStore）
        LaunchedEffect(Unit) {
            withContext(Dispatchers.IO) { LowConfStore.refresh(context) }
        }

        Box(Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .systemBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("BrickGame 提示", style = MaterialTheme.typography.headlineSmall)
                Spacer(Modifier.weight(1f))
                Button(onClick = { showSettings = true }) { Text("设置") }
            }

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
                    StatusLine("运行状态", if (ui.running) "提示中（第 ${ui.frames} 帧）" else "已停止")
                    StatusLine(
                        "自动通关",
                        when {
                            ui.auto -> "运行中 · 已消 ${ui.autoSteps} 对"
                            ui.autoSteps > 0 -> "已停止 · 共消 ${ui.autoSteps} 对"
                            else -> "未启用"
                        },
                    )
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
                    StatusLine(
                        "定位失败留档",
                        if (HintUiBus.scanFailCount == 0) "暂无" else "${HintUiBus.scanFailCount} 张",
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
                ) { Text(if (ui.running) "隐藏悬浮窗" else if (busy) "连接中…" else "显示悬浮窗") }
            }

            // 自动通关：进游戏前点它，之后由无障碍服务在游戏画面上自己滑。
            // 停下来只有三条路 —— 通关 / 无解（要洗牌）/ 手势没生效，都会写进下面的报错行。
            Button(
                modifier = Modifier.fillMaxWidth(),
                enabled = !busy,
                onClick = {
                    if (ui.auto) {
                        BrickGameAccessibilityService.stopAuto()
                        return@Button
                    }
                    busy = true
                    scope.launch {
                        val started = startAutoWithRetry()
                        busy = false
                        // 同"开始提示"：只有系统里就没开过才跳设置页
                        if (!started && !isAccessibilityServiceEnabled(context)) {
                            context.startActivity(accessibilitySettingsIntent(context))
                        }
                    }
                },
            ) { Text(if (ui.auto) "停止自动" else if (busy) "连接中…" else "自动通关") }

            ui.autoMessage?.let { message ->
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                    Text(
                        message,
                        modifier = Modifier.padding(14.dp),
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }

            // 截图和日志一起分享：即使什么都没留，也能把"当前日志"发出来排查
            val shareFrames = HintUiBus.captureCount
            val shareLogs = HintUiBus.logCount
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    modifier = Modifier.weight(1f),
                    enabled = !sharing && !busy,
                    onClick = {
                        sharing = true
                        scope.launch {
                            val zip = withContext(Dispatchers.IO) { LowConfStore.zipAll(context) }
                            sharing = false
                            if (zip == null) {
                                Log.w(TAG, "打包失败或没有可分享的留档")
                            } else {
                                shareZip(context, zip, shareFrames, shareLogs)
                            }
                        }
                    },
                ) {
                    Text(
                        when {
                            sharing -> "打包中…"
                            shareFrames == 0 && shareLogs == 0 -> "分享当前日志"
                            shareFrames == 0 -> "分享日志（$shareLogs 份）"
                            shareLogs == 0 -> "分享（$shareFrames 张 ZIP）"
                            else -> "分享（$shareFrames 张 + 日志）"
                        },
                    )
                }

                OutlinedButton(
                    modifier = Modifier.weight(1f),
                    enabled = !deleting && (HintUiBus.captureCount > 0 || HintUiBus.logCount > 0),
                    onClick = { confirmDelete = true },
                ) { Text(if (deleting) "删除中…" else "删除留档") }
            }

            // 只发日志：留档原图可能上百 MB，排查一次报错没必要整包发
            OutlinedButton(
                modifier = Modifier.fillMaxWidth(),
                enabled = !sharing,
                onClick = {
                    sharing = true
                    scope.launch {
                        val zip = withContext(Dispatchers.IO) { LowConfStore.zipLogs(context) }
                        sharing = false
                        if (zip == null) {
                            Log.w(TAG, "打包日志失败")
                        } else {
                            shareZip(context, zip, 0, HintUiBus.logCount, "分享日志")
                        }
                    }
                },
            ) {
                Text(if (sharing) "打包日志中…" else "分享日志（仅 logcat，排查用）")
            }

            // 删了就找不回来（真机上的留档是补标注的唯一副本），先确认一次
            if (confirmDelete) {
                AlertDialog(
                    onDismissRequest = { confirmDelete = false },
                    title = { Text("删除全部留档与日志？") },
                    text = {
                        Text(
                            "${HintUiBus.captureCount} 张留档截图（低置信 + 定位失败）和 " +
                                "${HintUiBus.logCount} 份诊断日志会被删掉，不能恢复。" +
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

        // 设置页做成覆盖层：主页一直渲染在下面，动画只作用在这一层。
        // 用 AnimatedContent 替换内容时容器尺寸会跳变，退出动画会被裁掉。
        AnimatedVisibility(
            visible = showSettings,
            enter = slideInHorizontally(tween(240)) { it },
            exit = slideOutHorizontally(tween(200)) { it },
        ) {
            SettingsPage(
                gridOn = gridOn,
                labelsOn = labelsOn,
                onGrid = {
                    gridOn = it
                    HintUiBus.showGrid = it
                },
                onLabels = {
                    labelsOn = it
                    HintUiBus.showLabels = it
                },
                onBack = { showSettings = false },
            )
        }
        }
        zoom?.let { file -> LowConfZoomDialog(file) { zoom = null } }
    }
}


/**
 * 设置页，照 iOS「设置」的样子：浅灰分组底 + 白色圆角卡片 + 行内左标题右控件。
 *
 * 用覆盖层渲染（见调用处的 AnimatedVisibility），所以这里自己铺满、自己滚动。
 */
@Composable
private fun SettingsPage(
    gridOn: Boolean,
    labelsOn: Boolean,
    onGrid: (Boolean) -> Unit,
    onLabels: (Boolean) -> Unit,
    onBack: () -> Unit,
) {
    val groupBg = Color(0xFFF2F2F7)
    val cardBg = Color.White
    val secondary = Color(0xFF8E8E93)
    val separator = Color(0xFFC6C6C8).copy(alpha = 0.55f)

    Column(
        Modifier
            .fillMaxSize()
            .background(groupBg)
            .systemBarsPadding()
            .verticalScroll(rememberScrollState()),
    ) {
        // 顶部导航条：返回在左、标题居中
        Row(
            Modifier
                .fillMaxWidth()
                .background(cardBg)
                .padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = onBack) { Text("返回") }
            Spacer(Modifier.weight(1f))
            Text("设置", fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.weight(1f))
            Spacer(Modifier.width(72.dp))
        }

        Spacer(Modifier.height(8.dp))

        GroupTitle("识别模型")
        SettingsCard(cardBg) {
            Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp)) {
                Text("MobileNetV2（V2）", fontSize = 16.sp)
                Spacer(Modifier.height(4.dp))
                Text(
                    "精度与端上性能综合最好，已是唯一内置模型",
                    fontSize = 13.sp,
                    color = secondary,
                )
            }
        }

        GroupTitle("调试显示")
        SettingsCard(cardBg) {
            SettingsRow("显示网格", gridOn, onGrid)
            HorizontalDivider(
                color = separator,
                thickness = 0.5.dp,
                modifier = Modifier.padding(start = 16.dp),
            )
            SettingsRow("显示类别号", labelsOn, onLabels)
        }
        Text(
            "改完下一帧生效：截屏分析本身就是一轮一轮跑的",
            fontSize = 13.sp,
            color = secondary,
            modifier = Modifier.padding(start = 32.dp, top = 6.dp),
        )

        Spacer(Modifier.height(28.dp))
    }
}

@Composable
private fun GroupTitle(text: String) {
    Text(
        text,
        fontSize = 13.sp,
        color = Color(0xFF8E8E93),
        modifier = Modifier.padding(start = 32.dp, top = 18.dp, bottom = 6.dp),
    )
}

@Composable
private fun SettingsCard(bg: Color, content: @Composable ColumnScope.() -> Unit) {
    Column(
        Modifier
            .padding(horizontal = 16.dp)
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(bg),
        content = content,
    )
}

@Composable
private fun SettingsRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clickable { onChange(!checked) }
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, fontSize = 16.sp)
        Spacer(Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = null)
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
private fun shareZip(
    context: Context,
    zip: File,
    frames: Int,
    logs: Int,
    title: String = "分享留档与日志",
) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", zip)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "application/zip"
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_SUBJECT, "BrickGame 留档（${frames} 张截图 / ${logs} 份落盘日志 + 实时 logcat）")
        clipData = ClipData.newUri(context.contentResolver, "lowconf", uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    Log.i(TAG, "分享 ZIP：${zip.name}（${frames} 帧 / ${logs} 份落盘日志，${zip.length() / 1024}KB）")
    context.startActivity(Intent.createChooser(send, title))
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

/**
 * 点"自动通关"后的等待。
 *
 * 与"开始提示"同一套重试逻辑（服务可能正被系统重新绑定，见 [startHintingWithRetry]）；
 * 区别是这里走 `startAuto`，它会顺带把会话也开起来。
 */
private suspend fun startAutoWithRetry(): Boolean {
    for (attempt in 0 until START_TRIES) {
        if (BrickGameAccessibilityService.startAuto()) return true
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
