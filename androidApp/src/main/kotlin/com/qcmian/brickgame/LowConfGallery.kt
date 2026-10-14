package com.qcmian.brickgame

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 低置信度留档的"相册"：一行可横滑的缩略小图，点一张放大看原图。
 *
 * 缩略图**按需解码**（LazyRow 只组合可见项，一项一张，见 [decodeScaled]）：留档是整屏截图，
 * 原图一张就好几十 MB，一次全解会直接 OOM。
 */
@Composable
fun LowConfGallery(files: List<File>, onOpen: (File) -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("低置信度留档（${files.size} 张，点开看大图）", style = MaterialTheme.typography.labelLarge)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(files, key = { it.name }) { file -> Thumb(file, onOpen) }
            }
        }
    }
}

@Composable
private fun Thumb(file: File, onOpen: (File) -> Unit) {
    val bitmap by produceState<Bitmap?>(null, file) {
        value = withContext(Dispatchers.IO) { decodeScaled(file, THUMB_PX) }
    }
    Box(
        modifier = Modifier
            .width(64.dp)
            .height(120.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(Color.Black)
            .clickable { onOpen(file) },
    ) {
        bitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = file.name,
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }
}

/**
 * 放大看原图：黑底整屏 + 顶部一行说明（文件名 + 留档 txt 里那两行头），点任意处关闭。
 */
@Composable
fun LowConfZoomDialog(file: File, onDismiss: () -> Unit) {
    val bitmap by produceState<Bitmap?>(null, file) {
        value = withContext(Dispatchers.IO) { decodeScaled(file, ZOOM_PX) }
    }
    val note by produceState("", file) {
        value = withContext(Dispatchers.IO) { noteOf(file) }
    }
    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .clickable(onClick = onDismiss)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Text(file.name, color = Color.White, style = MaterialTheme.typography.labelMedium)
            if (note.isNotBlank()) {
                Text(note, color = Color(0xFF9AA4AE), style = MaterialTheme.typography.bodySmall)
            }
            bitmap?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = file.name,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                )
            }
            Text("点任意处关闭", color = Color(0xFF9AA4AE), style = MaterialTheme.typography.bodySmall)
        }
    }
}

/** 同名 txt 里的说明行（`#` 开头那两行：盘面尺寸/耗时、没把握的格子），放大时显示。 */
private fun noteOf(file: File): String {
    val txt = File(file.parentFile, file.name.removeSuffix(".png") + ".txt")
    return runCatching {
        txt.readLines().filter { it.startsWith("#") }.take(2).joinToString("\n")
    }.getOrDefault("")
}

/**
 * 按目标高度降采样解码（目标高度是像素）。
 *
 * 先只读尺寸算出 [BitmapFactory.Options.inSampleSize] 再解：整屏截图原样 decode 一张就几十 MB，
 * 缩略图完全用不到那么细。
 */
private fun decodeScaled(file: File, targetHeightPx: Int): Bitmap? {
    if (!file.isFile) return null
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.path, bounds)
    if (bounds.outHeight <= 0) return null
    var sample = 1
    while (bounds.outHeight / (sample * 2) >= targetHeightPx) sample *= 2
    return runCatching {
        BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
    }.getOrNull()
}

/** 缩略图按 120dp 高、xhdpi 往上算，取够用的 480px。 */
private const val THUMB_PX = 480

/** 放大图给足屏幕高度（真机整屏 2400px 上下），采样后仍能看清格子。 */
private const val ZOOM_PX = 2400
