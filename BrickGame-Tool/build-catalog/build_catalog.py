#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从游戏图鉴页(brick-all)提取物品素材，建立带标签的条目库并去重。

为什么不用人工标注素材：图鉴页本身就是"每格一个物品、统一底板、统一样式"的结构化
清单 —— 行列位置就是稳定标识，裁出来即可当参考图用。

四个要点（都是踩过的坑）：

1. 网格用"全部页的**像素中位数图**"定位：物品每页不同，中位数把图标抹平，只剩固定
   面板 + 均匀底板。但图标区会被抹成浑浊色，底板掩码因此**裂成两条带**（图标左右各
   一条），必须成对合成一个格子跨度。
2. **底板要用"外圈像素拟合二次曲面再外推"**，不能用"环中位数"或"跨页中位数模板"：
   底板是渐变、且有多个色系（已解锁=亮绿、未解锁=浅米），环中位数当背景会让渐变区整片
   进入掩码 —— 实测 18 个完全不同的物品被并成一类（距离 0.04）；跨页中位数在中心区等于
   "37 个图标的中间色"，根本不是底板。拟合时按残差剔除两轮离群点（图标会顶到外圈）。
3. 掩码阈值用**自适应**的：未解锁物品是降饱和渲染，与底板差异很小；固定阈值会把它
   整格判成背景，几十个不同物品的特征全变零向量 → 又并成一类。
4. **默认不去重**：图鉴每格就是一个类别。实测这 37 页里没有跨页重复（md5 全不同，结构距离
   低于 0.05 的只有 1 组：同一把刷子的"新/旧"两版），而"并错"的代价远大于"没并"——
   图鉴里本身就存在形状相同、仅耐久/颜色不同的砖块（实测 0.10~0.20 区间有 197 对），
   一旦并错，求解器会去消一个消不掉的组合。需要合并时显式给 --dedup-thr（见下）。

产出：
  catalog/tiles/p001_b0_r0_c0.png   96x96 原样裁切（含底板）—— 训练素材，保留底板是为了让模型
                                    自己学"忽略底板"（配合底色重染增广），比去底掩码更稳
  catalog/index.csv                 每格元信息（页序/时间戳/分组/行列/图标占比/饱和度/faded/class_id）
  catalog/classes.csv               类别：class_id -> 成员条目（默认每格一类，共 2368 类）
  catalog/gray_pairs.csv            相似但未合并的条目对（仅 --dedup-thr > 0 时生成，人工抽查用）
  catalog/icons/...（仅 --with-icons）96x96 BGRA 去底图标，实测效果差已弃用，留开关备查
用法：
  python build_catalog.py --src brick-all --out catalog                 # 每格一类
  python build_catalog.py --src brick-all --out catalog --dedup-thr 0.05 # 顺带合并近似重复
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

CROP = 96           # 输出裁切尺寸
INSET = 3           # 内缩像素，去掉砖块描边
FEAT = 32           # 去重用的特征尺寸
DEDUP_THR = 0.0     # 去重阈值（结构特征余弦距离），0 = 不合并（每格一个类别）
FADED_CONTRAST = 90.0   # 图标与底板对比度低于此值 → 疑似未解锁（降饱和渲染）


def load_pages(src: str) -> list[tuple[str, np.ndarray]]:
    names = sorted(f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg")))
    pages = []
    for n in names:
        img = cv2.imread(os.path.join(src, n), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"读不到 {n}")
        pages.append((n, img))
    if not pages:
        raise SystemExit(f"{src} 里没有图片")
    if len({p[1].shape[:2] for p in pages}) != 1:
        print(f"警告: 分辨率不一致 {({p[1].shape[:2] for p in pages})}，请先统一")
    return pages


def tile_mask(img: np.ndarray) -> np.ndarray:
    """浅绿底板掩码。中位数图上图标已被抹平，所以只剩底板。"""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    return ((g > r + 6) & (g > b + 6) & (g > 170)).astype(np.uint8)


def segments(proj: np.ndarray, frac: float = 0.45) -> list[tuple[int, int]]:
    """一维投影上的连续高值段 = 砖块所在的列带/行带。"""
    thr = frac * float(proj.max()) if proj.max() > 0 else 0.0
    on = proj >= thr
    out, start = [], None
    for i, v in enumerate(on.tolist()):
        if v and start is None:
            start = i
        if (not v or i == len(on) - 1) and start is not None:
            end = i + 1 if v else i
            if end - start > 8:
                out.append((start, end))
            start = None
    return out


def pair_bands(bands: list[tuple[int, int]], name: str) -> list[tuple[int, int]]:
    """把"格子两侧的底板带"合成一个格子跨度，并做均匀性校验。"""
    if len(bands) % 2 == 0 and len(bands) >= 4:
        spans = [(bands[2 * k][0], bands[2 * k + 1][1]) for k in range(len(bands) // 2)]
    else:
        spans = bands
    widths = np.array([b - a for a, b in spans], dtype=np.float64)
    gaps = np.diff([a for a, _ in spans]) if len(spans) > 1 else np.array([1.0])
    w_cv = float(widths.std() / (widths.mean() + 1e-9))
    g_cv = float(gaps.std() / (gaps.mean() + 1e-9)) if len(gaps) > 1 else 0.0
    print(f"  {name}: {len(bands)} 条底板带 → {len(spans)} 个格子，"
          f"宽度 {widths.mean():.1f}±{widths.std():.1f}(cv {w_cv:.2f})，"
          f"间距 {gaps.mean():.1f}(cv {g_cv:.2f})")
    if w_cv > 0.15 or g_cv > 0.30:
        print(f"  警告: {name} 方向网格不均匀，可能有格子被遮挡，请人工核对")
    return spans


def grid_from_median(pages: list[tuple[str, np.ndarray]]) -> tuple[list, list]:
    med = np.median(np.stack([p[1] for p in pages]), axis=0).astype(np.uint8)
    m = tile_mask(med)
    col_seg = pair_bands(segments(m.sum(axis=0).astype(np.float64)), "列")
    row_seg = pair_bands(segments(m.sum(axis=1).astype(np.float64)), "行")
    return col_seg, row_seg


def cut_all(pages: list[tuple[str, np.ndarray]], row_seg, col_seg, inset: int) -> np.ndarray:
    """(n_pages, n_rows, n_cols, CROP, CROP, 3) 的原样裁切。"""
    out = np.zeros((len(pages), len(row_seg), len(col_seg), CROP, CROP, 3), np.uint8)
    for pi, (_, img) in enumerate(pages):
        for ri, (y0, y1) in enumerate(row_seg):
            for ci, (x0, x1) in enumerate(col_seg):
                cell = img[y0 + inset:y1 - inset, x0 + inset:x1 - inset]
                if cell.size == 0:
                    continue
                interp = cv2.INTER_AREA if cell.shape[0] > CROP else cv2.INTER_CUBIC
                out[pi, ri, ci] = cv2.resize(cell, (CROP, CROP), interpolation=interp)
    return out


_S = CROP
_yy, _xx = np.mgrid[0:_S, 0:_S]
_XX = np.stack([np.ones(_S * _S), _xx.ravel(), _yy.ravel(), (_xx ** 2).ravel(),
                (_xx * _yy).ravel(), (_yy ** 2).ravel()], 1).astype(np.float64)
_RING_W = 8
_RING = np.zeros((_S, _S), bool)
_RING[:_RING_W, :] = _RING[-_RING_W:, :] = _RING[:, :_RING_W] = _RING[:, -_RING_W:] = True


def bg_surface(crop: np.ndarray, rounds: int = 2) -> np.ndarray:
    """用外圈像素拟合二次曲面来还原底板（含渐变），再外推到中心。

    为什么不能用"边框环中位数"或"跨页模板"当底板：
    - 底板是渐变且有多个色系（已解锁是亮绿、未解锁是浅米色），环中位数当背景会让
      渐变区整片进入掩码 —— 实测 18 个完全不同的物品被去重成一类（距离 0.04）。
    - 跨页中位数在中心区等于"37 个图标的中间色"，不是底板。
    拟合时按残差做两轮剔除：图标偶尔会顶到外圈，不剔除会把曲面拉歪。
    """
    bg = np.zeros((_S, _S, 3), np.float32)
    ring_flat = _RING.ravel()
    for c in range(3):
        coef, m, vals = None, ring_flat.copy(), crop[:, :, c].ravel()
        for _ in range(rounds):
            coef, *_ = np.linalg.lstsq(_XX[m], vals[m], rcond=None)
            r = np.abs(vals - _XX @ coef)
            med = float(np.median(r[m]))
            mad = float(np.median(np.abs(r[m] - med))) + 1e-3
            keep = ring_flat & (r < med + 4 * mad)
            if keep.sum() < 100 or bool((keep == m).all()):
                break
            m = keep
        bg[:, :, c] = (_XX @ coef).reshape(_S, _S)
    return bg


def icon_mask(crop: np.ndarray, bg: np.ndarray, tol_lo: float = 10.0, frac: float = 0.30
              ) -> tuple[np.ndarray, float]:
    """图标掩码 = 与拟合底板的差异超过自适应阈值。

    自适应是必须的：未解锁物品整格降饱和（与底板对比很弱），固定阈值会把它判成
    纯背景，特征退化成零向量 → 几十个不同物品又被去重成一类。
    返回 (掩码, 图标对比度)。
    """
    mag = np.linalg.norm(crop - bg, axis=2)
    contrast = float(np.percentile(mag, 99.5))
    m = (mag > max(tol_lo, frac * contrast)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)
    return m, contrast


def icon_feature(crop: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """图标特征（L2 归一化）：**残差去掉直流色**之后的结构/色差模式。

    为什么不能直接用"掩码后的原色"：同一色系的物品（橙色系月饼/玉米/饼干）
    掩码后的向量方向几乎相同，余弦距离只有 0.03~0.06 —— 实测 21 个完全不同的
    橙色物品被并成一类。去掉直流分量（减去掩码内均值）后，剩下的是形状与色差
    模式，同一色系的不同形状才拉得开。
    """
    r = cv2.resize((crop - bg), (FEAT, FEAT), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(mask.astype(np.uint8), (FEAT, FEAT), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not m.any():
        return np.zeros(FEAT * FEAT * 3, np.float32)
    r = r - r[m].mean(axis=0)
    f = (r * m[..., None]).reshape(-1)
    n = float(np.linalg.norm(f))
    return f / n if n > 1e-6 else f


def saturation(crop: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1][mask].mean()) if mask.sum() > 8 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="图鉴页 → 物品素材库 + 去重类别")
    ap.add_argument("--src", default=os.path.join(_HERE, "brick-all"))
    ap.add_argument("--out", default=os.path.join(_HERE, "catalog"))
    ap.add_argument("--inset", type=int, default=INSET)
    ap.add_argument("--dedup-thr", type=float, default=DEDUP_THR,
                    help="合并阈值(结构余弦距离)，0 = 每格一个类别")
    ap.add_argument("--expect-cols", type=int, default=8, help="图鉴每行的格数")
    ap.add_argument("--with-icons", action="store_true",
                    help="额外导出 96x96 去底 BGRA 图标（默认不导出：训练直接用 tiles 整格）")
    args = ap.parse_args()

    pages = load_pages(args.src)
    print(f"图鉴页 {len(pages)} 张，尺寸 {pages[0][1].shape[1]}x{pages[0][1].shape[0]}")
    col_seg, row_seg = grid_from_median(pages)
    print(f"网格: {len(col_seg)} 列 x {len(row_seg)} 行 = {len(col_seg) * len(row_seg)} 格/页")
    if len(col_seg) != args.expect_cols:
        raise SystemExit(f"列数 {len(col_seg)} != {args.expect_cols}，请检查图鉴页是否有遮挡/滚动")

    crops = cut_all(pages, row_seg, col_seg, args.inset)

    icon_dir = os.path.join(args.out, "icons")
    tile_dir = os.path.join(args.out, "tiles")
    os.makedirs(tile_dir, exist_ok=True)
    if args.with_icons:
        os.makedirs(icon_dir, exist_ok=True)

    rows, feats = [], []
    for pi, (name, _) in enumerate(pages):
        stem = os.path.splitext(name)[0]
        for ri in range(len(row_seg)):
            for ci in range(len(col_seg)):
                crop = crops[pi, ri, ci]
                bg = bg_surface(crop)
                m, contrast = icon_mask(crop, bg)
                stem_p = f"p{pi + 1:03d}_b{ri // 2}_r{ri % 2}_c{ci}"
                if args.with_icons:
                    bgra = np.dstack([crop, (m * 255).astype(np.uint8)])
                    cv2.imwrite(os.path.join(icon_dir, stem_p + ".png"), bgra)
                cv2.imwrite(os.path.join(tile_dir, stem_p + ".png"), crop)
                feats.append(icon_feature(crop, bg, m))
                rows.append({
                    "entry": stem_p, "page": pi + 1, "page_file": stem,
                    "block": ri // 2, "sub_row": ri % 2, "row": ri, "col": ci,
                    "ratio": round(float(m.mean()), 4), "contrast": round(contrast, 1),
                    "sat": round(saturation(crop, m), 1),
                    "faded": int(contrast < FADED_CONTRAST),
                    # 掩码只用于统计/去重特征；去底图标不落盘时留空，避免下游读到不存在的文件
                    "icon": os.path.join("icons", stem_p + ".png") if args.with_icons else "",
                    "tile": os.path.join("tiles", stem_p + ".png"),
                })
        print(f"  {name}: 裁出 {len(row_seg) * len(col_seg)} 格", flush=True)

    cons = np.array([r["contrast"] for r in rows])
    print(f"\n图标对比度: 中位 {np.median(cons):.1f}，分位 "
          f"{[round(float(np.percentile(cons, q)), 1) for q in (1, 5, 10, 25, 50, 75, 95)]}")
    print(f"  低于 {FADED_CONTRAST} 的 {int((cons < FADED_CONTRAST).sum())} 格（疑似未解锁/降饱和渲染）")
    print(f"图标占比: 中位 {np.median([r['ratio'] for r in rows]):.3f}")

    # ---------- 分类 ----------
    # 默认 thr=0 → 不合并，每格自成一类（图鉴页每格就是一款砖块）。
    F = np.stack(feats)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    D = np.clip(1.0 - F @ F.T, 0.0, 2.0)
    np.fill_diagonal(D, 2.0)
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(D < args.dedup_thr)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if j > i:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    labels = np.zeros(len(rows), dtype=np.int32)
    for cid, (_, members) in enumerate(sorted(groups.items()), start=1):
        for i in members:
            labels[i] = cid

    dup = [m for m in sorted(groups.values(), key=len, reverse=True) if len(m) > 1]
    mode = "不去重" if args.dedup_thr <= 0 else f"结构余弦阈值 {args.dedup_thr}"
    print(f"分类({mode}): {len(rows)} 个条目 → {len(groups)} 个类别"
          f"（合并 {len(rows) - len(groups)} 个，涉及 {len(dup)} 组）")
    for m in dup[:8]:
        print(f"  {len(m)} 个成员: {[rows[i]['entry'] for i in m[:6]]}"
              f"{' ...' if len(m) > 6 else ''}")
    # 灰区提示：结构距离在 [thr, 3*thr) 的条目对（只在开启合并时才有意义），供人工抽查。
    # 宁可留成两类，也不要误并（误并会让求解器去消一个消不掉的组合）。
    gray = ([(float(D[i, j]), i, j) for i, j in
             zip(*np.where((D >= args.dedup_thr) & (D < 3 * args.dedup_thr))) if j > i]
            if args.dedup_thr > 0 else [])
    if args.dedup_thr > 0:
        print(f"灰区(相似但未合并) {len(gray)} 对，最相近 5 对：")
        for d, i, j in sorted(gray)[:5]:
            print(f"  {d:.3f} {rows[i]['entry']} ~ {rows[j]['entry']}")
        with open(os.path.join(args.out, "gray_pairs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["struct_dist", "entry_a", "entry_b"])
            for d, i, j in sorted(gray):
                w.writerow([round(d, 4), rows[i]["entry"], rows[j]["entry"]])

    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["class_id"])
        w.writeheader()
        for r, l in zip(rows, labels.tolist()):
            w.writerow({**r, "class_id": l})
    with open(os.path.join(args.out, "classes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "n", "entries"])
        for cid, (_, members) in enumerate(sorted(groups.items()), start=1):
            w.writerow([cid, len(members), "|".join(rows[i]["entry"] for i in members)])
    icon_msg = f"{args.out}/icons/（{len(rows)} 张 BGRA）/ " if args.with_icons else ""
    print(f"\n输出: {args.out}/tiles/（{len(rows)} 张） / {icon_msg}index.csv"
          f" / classes.csv（{len(groups)} 类）")


if __name__ == "__main__":
    main()
