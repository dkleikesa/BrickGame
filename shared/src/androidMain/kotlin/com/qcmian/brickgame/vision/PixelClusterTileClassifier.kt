package com.qcmian.brickgame.vision

import android.graphics.Bitmap

/**
 * 不依赖训练模型就能跑的识别：把**帧内**同款砖块聚成一类。
 *
 * 对应 `build_dataset.frame_cells` 里那段：
 * 去底图标特征 → ±3px 平移对齐的两两距离 → 找距离分布的最大空隙 → 偶约束聚类 → 合并回被拆开的类。
 *
 * 为什么"只要帧内一致"就够算法用：`BrickGameTipHelper` 只要求"同款同号、空位为 0"，
 * 不需要跨帧稳定的类 id。所以模型没训练完也能先跑通整条链路。
 *
 * 已知局限（Python 版同样存在，不是移植引入的）：
 *  - 颜色/轮廓相近的不同物品会互相抢（菠萝的最近邻是胡萝卜、柠檬最近邻是黄色圆盘）；
 *  - 同一款砖的"冻结/选中"底色不同时可能被拆成两类；
 *  - 判空的格子在 [BoardState.of] 里被强制归零，所以聚类只在非空位之间做。
 */
class PixelClusterTileClassifier(
    private val rad: Int = 3,
    private val prefilter: Double = 0.85,
) : TileClassifier {

    override val name: String = "像素聚类（未接模型）"

    override fun classify(frame: Bitmap, scan: BoardScan): IntArray {
        val out = IntArray(scan.cellCount)
        val keep = (0 until scan.cellCount).filter { !scan.empty[it] }
        // 少于 4 块砖时聚类没有意义（Python 同款阈值）
        if (keep.size < MIN_CLUSTER_CELLS) return out

        val cell = scan.patches.cell
        val stride = scan.patches.valuesPerPatch
        val pixels = FloatArray(keep.size * stride)
        for ((k, index) in keep.withIndex()) {
            scan.patches.pixels.copyInto(pixels, k * stride, index * stride, (index + 1) * stride)
        }
        val sub = TileFrame(1, keep.size, cell, pixels)

        val features = FeatureMath.iconFeatures(sub)
        val dist = FeatureMath.alignedDistanceMatrix(features.dev, keep.size, cell, rad, prefilter)
        val gap = FeatureMath.gapThreshold(dist)
        val clustering = FeatureMath.clusterByParity(dist, gap)
        val merged = FeatureMath.mergeSimilar(clustering.labels, dist, gap)

        // 标签重排成 1..k，便于观色与日志（算法本身不关心具体数值）
        val remap = HashMap<Int, Int>()
        for (k in merged.indices) {
            val label = remap.getOrPut(merged[k]) { remap.size + 1 }
            out[keep[k]] = label
        }
        return out
    }

    private companion object {
        const val MIN_CLUSTER_CELLS = 4
    }
}

/**
 * 另一种识别实现（接了训练好的身份模型）在 androidApp 侧，见
 * `com.qcmian.brickgame.ml.OnnxTileClassifier` —— 那里要读 assets 与 ONNX Runtime，
 * 依赖方向不允许反过来放进 shared。
 */
