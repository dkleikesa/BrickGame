#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享底层库：棋盘标定 + 裁格 + 图标特征 + 帧内聚类。**不做训练，也不写数据集。**

三个环节（全部自动，不依赖机型/分辨率/皮肤）：
  1. 定位棋盘网格：梯度图行/列投影 → 自相关求格子周期 pitch → 相位锁定求网格线
                   → 取最长连续的一段 = 棋盘 → 得到行数/列数/每格矩形（`detect_grid`）
  2. 特征与空位：每格裁切 → 缩放 → 去底板 → 图标特征；接近纯底色的格子判为空位（`icon_features`）
  3. 帧内聚类：特征两两算距离（带 ±3px 平移对齐，`aligned_distance_matrix`）→
               偶数约束下的等价类（`cluster_by_parity` / `merge_similar`）

谁在用：
  build-dataset/build_dataset.py   标定 + 裁格
  build-dataset/auto_classify.py   图标特征（像素特征兜底路线）
  build-dataset/visualize_groups.py
  train_classifier.py              标定 / 像素基线对比
（早期那套"标定缓存 + 稀疏棋盘兜底 + 奇偶纠错 + 调参 CLI"已随数据集管线落定而删除。）

单独跑一张图只为**看网格定位对不对**（调试图，不产数据集）：
  python tile_recognizer.py samples/screenshot_01.jpg --out out
"""
from __future__ import annotations

import argparse
import itertools
import os

import cv2
import numpy as np

CELL = 32          # 每格归一化后的边长
INSET = 0.14       # 裁切时向内收缩的比例（去掉格子边框/圆角）


# ---------------------------------------------------------------- 图像与投影
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def edge_profile(img: np.ndarray) -> np.ndarray:
    """梯度幅值图：格子边界会形成成排的强边，棋盘底色/天空则是平滑的。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g


def detrend(sig: np.ndarray, win: int) -> np.ndarray:
    """减去滑动平均 = 高通滤波。不做这步，投影里的缓慢趋势会让自相关在
    小滞后上就很高，周期会被误判成 min_pitch。"""
    win = max(5, int(win) | 1)
    pad = win // 2
    box = np.ones(win) / win
    padded = np.pad(sig.astype(np.float64), pad, mode="edge")
    trend = np.convolve(padded, box, mode="same")[pad:pad + len(sig)]
    return sig.astype(np.float64) - trend


def fundamental_pitch(sig: np.ndarray, min_pitch: int, max_pitch: int) -> tuple[int, np.ndarray]:
    """高通后做自相关，取最大峰；若是 2 倍谐波则折半到基波。"""
    x = detrend(sig, max(3 * min_pitch, int(len(sig) * 0.12)))
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo = max(2, min_pitch)
    hi = max(lo + 1, min(n - 2, max_pitch))
    seg = ac[lo:hi]
    if seg.size == 0:
        return lo, ac
    lag = lo + int(np.argmax(seg))
    while lag >= 2 * lo and ac[lag // 2] >= 0.6 * ac[lag]:
        lag //= 2
    return lag, ac


def lattice_phase(prof: np.ndarray, pitch: int) -> int:
    """在给定周期下求相位：让网格线尽量落在强边上的那个偏移。"""
    n = len(prof)
    p = detrend(prof, max(3 * pitch, int(n * 0.12)))
    return int(np.argmax([float(p[ph::pitch].sum()) for ph in range(pitch)]))


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    best, cur = (0, 0), None
    for i, ok in enumerate(mask.tolist()):
        if ok and cur is None:
            cur = i
        if (not ok or i == len(mask) - 1) and cur is not None:
            end = i + 1 if ok else i
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    return best


def refine_lines(prof: np.ndarray, lines: np.ndarray, pitch: int) -> np.ndarray:
    """把每条网格线放到"缝隙中心"，再做亚像素精调与最小二乘拟合。

    为什么必须放到缝隙中心：相邻两块砖的描边会形成一对很近的峰（实测间隔 9px，
    而格子间距 164px）。相位锁定可能落在其中任意一个峰上，格子边界就偏了半个缝宽，
    裁切窗口跟着错位。所以这里显式找"两条描边之间的谷底"。
    """
    n = len(prof)
    half = max(2, pitch // 5)
    ks, ys = [], []
    for k, y in enumerate(lines):
        lo, hi = max(0, int(y) - half), min(n, int(y) + half + 1)
        seg = prof[lo:hi].astype(np.float64)
        if seg.size < 3:
            continue
        peaks = [i for i in range(1, len(seg) - 1)
                 if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.35 * seg.max()]
        if len(peaks) >= 2:
            a, b = sorted(sorted(peaks, key=lambda i: -seg[i])[:2])
            v = a + int(np.argmin(seg[a:b + 1]))
            pos = float(lo + v)
            if 0 < v < len(seg) - 1:                     # 抛物线插值到亚像素
                d = seg[v - 1] - 2 * seg[v] + seg[v + 1]
                if abs(d) > 1e-6:
                    pos += float(np.clip(0.5 * (seg[v - 1] - seg[v + 1]) / d, -0.5, 0.5))
        elif peaks:
            i = peaks[0]
            thr = 0.5 * seg[i]
            a = i
            while a - 1 >= 0 and seg[a - 1] >= thr:
                a -= 1
            b = i
            while b + 1 < seg.size and seg[b + 1] >= thr:
                b += 1
            w = seg[a:b + 1]
            pos = lo + a + float((w * np.arange(len(w))).sum() / (w.sum() + 1e-9))
        else:
            pos = float(y)
        ks.append(float(k))
        ys.append(pos)

    if len(ks) < 3:
        return lines.astype(np.float64)

    ks_a, ys_a = np.array(ks), np.array(ys)
    for _ in range(2):                       # 去掉偏差大的点后重拟合
        slope, intercept = np.polyfit(ks_a, ys_a, 1)
        resid = np.abs(ys_a - (intercept + slope * ks_a))
        keep = resid <= max(1.5, pitch * 0.08)
        if keep.all():
            break
        ks_a, ys_a = ks_a[keep], ys_a[keep]
    slope, intercept = np.polyfit(ks_a, ys_a, 1)
    return intercept + slope * np.arange(len(lines))


def cell_ring_score(img: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    lo: float = 0.04, hi: float = 0.20) -> np.ndarray:
    """格子四条边内侧环带的方向梯度，取**四边最小值** —— 专门用来判"棋盘范围"。

    两个要点：
    1. 取四边最小值：真砖块四边都有圆角描边；棋盘横梁、进度条、UI、草地只有单边。
    2. **按方向取**：上下边看 |gy|，左右边看 |gx|。否则一条横贯画面的进度条
       在左右环带里也会因为"横向边"拿到高分，实测会把整行 UI 误判成一行砖块。
    这个量只用于定范围（谁是棋盘内、谁是棋盘外），不用于区分砖块类别。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    ny, nx = len(ys) - 1, len(xs) - 1
    R = np.zeros((ny, nx), dtype=np.float64)
    for j in range(ny):
        y0, y1 = int(ys[j]), int(ys[j + 1])
        h = y1 - y0
        for i in range(nx):
            x0, x1 = int(xs[i]), int(xs[i + 1])
            w = x1 - x0
            vals = [gy[y0 + int(h * lo):y0 + int(h * hi), x0:x1],
                    gy[y1 - int(h * hi):y1 - int(h * lo), x0:x1],
                    gx[y0:y1, x0 + int(w * lo):x0 + int(w * hi)],
                    gx[y0:y1, x1 - int(w * hi):x1 - int(w * lo)]]
            means = [v.mean() for v in vals if v.size]
            R[j, i] = min(means) if means else 0.0
    return R


def _border_mask(size: int = CELL, w: int = 3) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[:w, :] = m[-w:, :] = m[:, :w] = m[:, -w:] = True
    return m


_BORDER = _border_mask()


def icon_features(patches: np.ndarray, tol: float = 40.0,
                  erode: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """返回 (图标特征, 图标占比)。

    做法：把"底板像素"**直接挖掉（置 0）**，只保留图标像素本身的绝对颜色。

    为什么不能只"减底板色"：底板色是砖块状态（冻结/选中会让底板变蓝等），
    同一图标若底板不同，减去各自的底板色后，图标区和背景区的偏差方向相反，
    这个空间上的符号翻转抵消不掉 —— 实测同类距离仍有 0.12~0.21，
    和真不同款的距离一个量级，聚类就开始乱拆（#11/#14 就是这么被拆开的）。
    挖掉背景之后，同一图标的特征逐像素一致，距离回到 ~0，与底板状态无关。

    第二个返回值 = 图标像素占比：空格子几乎全是底板 → 接近 0，是天然的"空位"判据。
    """
    bg = np.median(patches[:, _BORDER, :], axis=1)
    diff = patches - bg[:, None, None, :]
    mask = (np.linalg.norm(diff, axis=3) > tol).astype(np.uint8)
    if erode:
        # 图标轮廓那一圈像素被底板色抗锯齿染色（蓝底与黄底混出来不一样），
        # 腐蚀掉 1 圈再比对，同类距离才真正回到 0（实测 0.21 → 0.04）。
        mask = np.stack([cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode) for m in mask])
    mask = mask[..., None].astype(bool)
    ratio = mask.mean(axis=(1, 2, 3))
    return patches * mask, ratio


def trim_board_extent(tile: np.ndarray, axis: int) -> tuple[int, int]:
    """在砖块掩码上取"最长连续段"，再去掉两端明显偏稀的行/列。

    为什么要裁两端：棋盘上方那条草地条带偶尔会有 1~2 格被误判成砖块，
    而真棋盘每行至少有"中位数 30%"的砖块。
    """
    cnt = tile.sum(axis=1 - axis)
    a, b = longest_run(cnt >= 1)
    if b - a < 2:
        return a, b
    med = float(np.median(cnt[a:b]))
    while a < b - 2 and cnt[a] < 0.30 * med:
        a += 1
    while b - a > 2 and cnt[b - 1] < 0.30 * med:
        b -= 1
    return a, b


def detect_grid(img: np.ndarray, min_pitch: int, inset: float = INSET,
                verbose: bool = True) -> dict:
    """求周期/相位 → 精调网格线 → 定棋盘范围 → 裁格、算图标特征、标空位。

    两个信号分工明确（实测各自单独用都会出错）：
      - **棋盘范围**用环带四边最小值：它能顶住进度条/按钮这类横贯画面的 UI；
      - **空位判定**用图标对比度：空格子相对底板几乎没有偏差，
        而其它信号会把"选中高亮"的砖块误判成空格。
    """
    g = edge_profile(img)
    row_prof = g.sum(axis=1)
    col_prof = g.sum(axis=0)

    h, w = g.shape
    pitch_y, ac_y = fundamental_pitch(row_prof, min_pitch, int(h * 0.25))
    pitch_x, ac_x = fundamental_pitch(col_prof, min_pitch, int(w * 0.25))

    ylines = refine_lines(row_prof, np.arange(lattice_phase(row_prof, pitch_y), h, pitch_y), pitch_y)
    xlines = refine_lines(col_prof, np.arange(lattice_phase(col_prof, pitch_x), w, pitch_x), pitch_x)

    ring = cell_ring_score(img, xlines, ylines)
    tile = ring >= 0.25 * float(np.percentile(ring, 95))
    r0, r1 = trim_board_extent(tile, axis=0)
    c0, c1 = trim_board_extent(tile, axis=1)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise SystemExit("网格定位失败：没找到成排的格子，检查截图或调小 --min-pitch")

    xlines, ylines = xlines[c0:c1 + 1], ylines[r0:r1 + 1]
    out = grid_from_lines(img, xlines, ylines, inset)
    out.update({"pitch_x": int(pitch_x), "pitch_y": int(pitch_y),
                "row_prof": row_prof, "col_prof": col_prof, "ac_x": ac_x, "ac_y": ac_y})
    if verbose:
        print(f"候选网格: x 周期{pitch_x} ({len(xlines) - 1}列) / y 周期{pitch_y} ({len(ylines) - 1}行)")
        print(f"  每行砖块数: " + " ".join(str(int(v)) for v in (out["score"] >= out["score_thr"]).sum(axis=1)))
        print(f"  每列砖块数: " + " ".join(str(int(v)) for v in (out["score"] >= out["score_thr"]).sum(axis=0)))
    return out


def grid_from_lines(img: np.ndarray, xlines, ylines, inset: float = INSET) -> dict:
    """给定网格线，裁格 + 算图标特征 + 标空位。标定回退路径也用这个。"""
    rects, patches = cut_cells(img, xlines, ylines, inset)
    dev, ratio = icon_features(patches)
    rows, cols = len(ylines) - 1, len(xlines) - 1
    score = ratio.reshape(rows, cols)
    thr = max(0.25 * float(np.percentile(score, 90)), 0.04)
    return {
        "xlines": np.asarray(xlines, dtype=np.float64), "ylines": np.asarray(ylines, dtype=np.float64),
        "score": score, "score_thr": thr,
        "patches": patches, "dev": dev, "rects": rects,
        "cols": cols, "rows": rows,
    }


# ---------------------------------------------------------------- 切格与特征
def cut_cells(img: np.ndarray, xlines, ylines, inset: float):
    """以每格中心对称裁一个统一尺寸的方块 —— 尺寸一致才好逐像素比对。"""
    nx, ny = len(xlines) - 1, len(ylines) - 1
    side = int(round(float(np.median(np.concatenate([np.diff(xlines), np.diff(ylines)])))
                     * (1 - 2 * inset)))
    side = max(8, side)
    rects, patches = [], []
    for j in range(ny):
        cy = (ylines[j] + ylines[j + 1]) / 2.0
        for i in range(nx):
            cx = (xlines[i] + xlines[i + 1]) / 2.0
            a = int(round(cx - side / 2.0))
            c = int(round(cy - side / 2.0))
            b, d = a + side, c + side
            a, c = max(0, a), max(0, c)
            b, d = min(img.shape[1], b), min(img.shape[0], d)
            cell = img[c:d, a:b]
            rects.append((a, c, b, d))
            patch = cv2.resize(cell, (CELL, CELL), interpolation=cv2.INTER_AREA)
            patches.append(patch.astype(np.float32))
    return np.array(rects, dtype=np.int32), np.stack(patches)


def aligned_distance_matrix(dev: np.ndarray, rad: int = 3, prefilter: float = 0.85) -> np.ndarray:
    """图标特征两两距离，带 ±rad 像素的平移搜索。

    这一步是关键：网格线拟合再准也会有亚像素残差，同一个图标在不同行列裁出来
    会差 1~2 个像素，实测未对齐时同类距离能到 0.58，对齐后 0.002。
    先用余弦距离粗筛，只对可能同类的少数格子对做平移搜索，速度也够。
    """
    n = len(dev)
    f = dev.reshape(n, -1)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    D = np.clip(1.0 - f @ f.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)

    pads = [cv2.copyMakeBorder(c.astype(np.float32), rad, rad, rad, rad, cv2.BORDER_REPLICATE)
            for c in dev]
    tmpls = [c.astype(np.float32) for c in dev]
    iu = np.triu_indices(n, 1)
    for i, j in zip(iu[0].tolist(), iu[1].tolist()):
        if D[i, j] > prefilter:
            continue
        d = float(cv2.matchTemplate(pads[i], tmpls[j], cv2.TM_SQDIFF_NORMED).min())
        D[i, j] = D[j, i] = d
    return D


def gap_threshold(dist: np.ndarray, lo: float = 0.02, hi: float = 0.70) -> float:
    """距离分布里最大的那个"空隙"的中点。

    同类距离实测都挤在 0~0.3，异类在 0.8 以上，中间是空旷地带。
    这个值用来给合并设上界——没有上界的话，一旦出现某个奇数大小的连通分量，
    聚类会一路合并到几乎全并成一个簇（实测从 12 个簇塌成 1 个）。
    """
    v = np.sort(dist[np.triu_indices_from(dist, 1)])
    if v.size < 2:
        return hi
    gaps = np.where((v[:-1] >= lo) & (v[:-1] <= hi), np.diff(v), 0.0)
    k = int(np.argmax(gaps))
    return float((v[k] + v[k + 1]) / 2)


def merge_similar(labels: np.ndarray, dist: np.ndarray, hard: float) -> np.ndarray:
    """把"明显还是同类"的簇对并回去。

    为什么需要这一步：偶数约束的合并是"贪心凑偶数"，它会在凑够偶数时立刻停手。
    一个真实 4 块的一类，如果其中一块的状态不同（底色变了导致距离略大），
    贪心会先配掉更近的两块，剩下两块凑成另一对 → 被拆成 2+2。
    2+2 的每一步虽然仍合法，但求解器只能在子簇内部配对，**丢掉跨子簇的解**。
    这里按"距离仍在自然空隙以下 + 合并后仍是偶数"两条，把这类拆开的重并回去。
    """
    labels = labels.copy()
    while True:
        groups: dict = {}
        for i, l in enumerate(labels.tolist()):
            groups.setdefault(l, []).append(i)
        if len(groups) < 2:
            return labels
        best = None
        keys = sorted(groups)
        for a, b in itertools.combinations(keys, 2):
            if (len(groups[a]) % 2) != (len(groups[b]) % 2):
                continue                      # 合并后会变奇数，不能并
            d = float(dist[np.ix_(groups[a], groups[b])].mean())
            if d < hard and (best is None or d < best[0]):
                best = (d, a, b)
        if best is None:
            return labels
        _, a, b = best
        labels[np.isin(labels, [a, b])] = a


def cluster_by_parity(dist: np.ndarray, max_dist: float = 0.9):
    """Kruskal 式按距离升序加边，维护"奇数大小簇的个数"，首次降到 0 即为阈值。

    直接用游戏规则当目标：一步消两块 → 每类数量必须偶数。
    这比在距离直方图里找空隙稳得多。
    """
    n = len(dist)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0.0, True

    parent = list(range(n))
    size = [1] * n
    odd = n                       # 全是单元素簇 → 全是奇数

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu = np.triu_indices(n, 1)
    order = np.argsort(dist[iu])
    edges, best = [], None
    thr = 0.0
    for k in order:
        d = float(dist[iu[0][k], iu[1][k]])
        if d <= 1e-6 or d > max_dist:
            continue
        ri, rj = find(int(iu[0][k])), find(int(iu[1][k]))
        if ri == rj:
            continue
        if size[ri] % 2 == 1:
            odd -= 1
        if size[rj] % 2 == 1:
            odd -= 1
        parent[ri] = rj
        size[rj] += size[ri]
        if size[rj] % 2 == 1:
            odd += 1
        edges.append((ri, rj))
        thr = d
        n_cls = len({find(i) for i in range(n)})
        if best is None or (odd, -n_cls) < best[:2]:
            best = (odd, -n_cls, len(edges), d)
        if odd == 0:
            break

    # 偶数约束可能永远满足不了（比如有格子被误判成空位，总数成了奇数）。
    # 这时候必须回退到"奇数类个数最少"的那个中间状态，否则会一路合并成一个大簇。
    odd_best, neg_cls, n_edges, thr_best = best
    parent = list(range(n))
    size = [1] * n
    for ri, rj in edges[:n_edges]:
        parent[ri] = rj

    roots, labels = {}, np.zeros(n, dtype=np.int32)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots) + 1
        labels[i] = roots[r]
    return labels, thr_best, odd_best == 0


# ---------------------------------------------------------------- 可视化
def draw_debug(img: np.ndarray, grid: dict, labels: np.ndarray, empty: np.ndarray) -> np.ndarray:
    vis = img.copy()
    xlines, ylines = grid["xlines"], grid["ylines"]
    for x in xlines:
        cv2.line(vis, (int(x), int(ylines[0])), (int(x), int(ylines[-1])), (0, 0, 255), 1)
    for y in ylines:
        cv2.line(vis, (int(xlines[0]), int(y)), (int(xlines[-1]), int(y)), (0, 0, 255), 1)

    rng = np.random.default_rng(7)
    palette = {l: tuple(int(c) for c in rng.integers(60, 255, 3)) for l in set(labels.tolist())}
    k = 0
    for j in range(grid["rows"]):
        for i in range(grid["cols"]):
            x0, y0 = int(xlines[i]), int(ylines[j])
            x1, y1 = int(xlines[i + 1]), int(ylines[j + 1])
            if empty[k]:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (128, 128, 128), -1)
                cv2.putText(vis, "0", (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                color = palette.get(int(labels[k]), (0, 255, 0))
                overlay = vis.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
                vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
                cv2.putText(vis, str(int(labels[k])), (x0 + 3, y0 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
                cv2.putText(vis, str(int(labels[k])), (x0 + 3, y0 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            k += 1
    return vis


# ---------------------------------------------------------------- 单独调试
def main() -> None:
    """只跑标定并输出调试图 —— 用来确认新截图的网格定位对不对（不产数据集）。"""
    ap = argparse.ArgumentParser(description="单张截图 → 棋盘标定 + 调试图")
    ap.add_argument("image", help="游戏截图路径")
    ap.add_argument("--out", default="out", help="调试图输出目录")
    ap.add_argument("--min-pitch", type=int, default=12, help="格子最小像素间距")
    ap.add_argument("--inset", type=float, default=INSET, help="裁切内缩比例")
    args = ap.parse_args()

    img = load_image(args.image)
    print(f"图片: {img.shape[1]}x{img.shape[0]}")
    g = detect_grid(img, args.min_pitch, args.inset)
    holes = (g["score"] < g["score_thr"]).ravel()
    print(f"网格: {g['cols']} 列 x {g['rows']} 行 = {holes.size} 格，"
          f"其中空位 {int(holes.sum())} 格；周期 x={g['pitch_x']} y={g['pitch_y']}px")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "debug_lattice.png")
    cv2.imwrite(path, draw_debug(img, g, np.zeros(holes.size, np.int32), holes))
    print(f"→ {path}")


if __name__ == "__main__":
    main()
