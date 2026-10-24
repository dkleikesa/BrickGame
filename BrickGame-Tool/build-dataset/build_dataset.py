#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一条命令跑完"裁剪新帧 → 模型识别 → 人工标注界面"。**默认增量**：

 ① 裁图：只处理 samples/ 里"台账中没有"的新帧 → 棋盘标定 → 每格裁切（保留底色）
 ② 识别：预训练 CNN（默认 mobilenet_v3_large，见 auto_classify.DEFAULT_BACKBONE）提特征
         → 最近邻匹配参考库 → 每格得到 class_id + top-N 候选；已消除的格子=背景类(class_id 0)
 ③ 标注：生成 annotate.html（按"视觉一样"分组，**只列 frames.csv 里还没标完的帧**）；
         人工改完 → 每帧点"本帧标完" → 导出 JSON → apply_corrections.py 写回 index.csv + frames.csv。

**增量是默认行为**：旧帧一旦入库就永不重切（图片、标签、标注都不动），
只处理新帧，并且只给新帧打标签 —— 这样"标注只针对增量"才是成立的。
标注这条线同样闭环：写回的帧 status=done，下次生成界面就不再铺出来（只剩新帧），
所以标注也只需处理增量。要复查某帧就把它从 frames.csv 的 status 改回 pending（或加 --all）。
帧的身份按**截图内容 md5**认（不认文件名）：同一张图重复导入会自动跳过，改名也不影响。
要整体重切才加 --rebuild（配合 --keep-labels 保住已有标注）。

产出：
  dataset/images/frame/<截图名>/r03c05.png  截图每格 64x64，含底色（状态信息就在底色里）
  dataset/index.csv    每格 5 列：path（图片）、frame（哪张截图）、class_id（标签）、
                       cls_dist（识别距离，仅供"只看低置信"筛选）、cls_src（auto/manual/unknown/empty）
                       行列已编码在文件名里（r03c05 = 第3行第5列）
  dataset/frames.csv   帧台账：frame/source_file/source_md5/batch/rows/cols/cells/bricks/empty/
                       status(pending|done)/added_at/labeled_at —— 增量与标注进度的唯一依据
  dataset/pairs/<帧>.csv  帧内配对（仅 --pairs 导出：训练按帧内 label 采样，用不到它）
  cache/emb_<骨干>.npz    预训练特征缓存（识别与分组都基于它）
  cache/auto_labels.csv   每格 top-N 候选 class_id（标注界面点选用）
  annotate.html           人工标注界面（--html 可改名；默认只显示待标注帧）

**每帧的格子一律全部落盘**（rows×cols 一个不少），"不丢图"是硬要求：
已消除、只剩棋盘底色的格子自成一类 —— class_id = 0（背景）、cls_src = empty；
其余格子由 auto_classify 填 1..2368 里的某类。
（以前是把空位丢掉，结果"判空"一旦判错，砖块就从数据集里凭空没了，事后查不出来；
 实测误丢过 37 格浅色砖块。现在判空判错最多是标签错，格子仍在、能人工改。）
帧目录名 = 截图原文件名（中文名当命令行参数时记得加引号），所以帧名本身就能对回 samples/ 里的原图。

为什么要"保留底色"：底色 = 砖块状态（冻结/选中）的唯一信息源。
状态要作为独立任务学（多任务里的分类头），身份头才能顺带学会"忽略底色"。
参考库里的图鉴 tiles 同样保留底板：训练靠底色重染造状态多样性，不用去底掩码
（实测去底掩码的边缘残留会伤训练，且"只减底板色"会让同类距离涨到 0.12~0.21）。

用法：
  python build_dataset.py                      # 增量：只处理新帧（日常就用这个）
  python build_dataset.py --rebuild --keep-labels   # 整体重切（改判据/改尺寸后用）
  python build_dataset.py --no-classify        # 只裁图，不识别、不出界面
  python build_dataset.py --batch batch03      # 给本批新帧打批次号（默认当天日期）
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
import shutil
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
sys.path.insert(0, os.path.dirname(_HERE))   # 项目根：tile_recognizer / build_catalog
sys.path.insert(0, _HERE)                    # 本目录：同流水线的兄弟模块
import tile_recognizer as T  # noqa: E402
import yolo_grid  # noqa: E402
import auto_classify  # noqa: E402
import embed_features  # noqa: E402
import visualize_groups  # noqa: E402

TILE = 64          # 数据集里每格存成多大
OUT_INSET = 0.02   # 裁切时几乎不内缩：要把整块砖（含边框和底色）都保留


def cut_tile(img: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    a, c = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    a, c = max(0, a), max(0, c)
    b, d = min(img.shape[1], a + side), min(img.shape[0], c + side)
    patch = img[c:d, a:b]
    if patch.size == 0:
        patch = img[:TILE, :TILE]
    return cv2.resize(patch, (TILE, TILE), interpolation=cv2.INTER_AREA if patch.shape[0] > TILE
                      else cv2.INTER_CUBIC)


def frame_cells(img: np.ndarray) -> dict | None:
    """一帧 → 每格的裁切图 + 伪标签 + 元信息。

    定位**只走 YOLO**。残局/空盘时 OpenCV 的自相关投影没有周期可数，会把 14 行算成 13 行、
    格线整体错一格；这条兜底路径已经从代码里删掉，判不出就跳过，别退回投影法。
    """
    g = None
    got = yolo_grid.grid_of(img)
    if got is not None:
        xl, yl, _, _ = got
        # 网格定好了，后续裁格/判空/特征提取复用同一套（与识别训练对齐）
        g = T.grid_from_lines(img, xl, yl, T.INSET)
    if g is None:
        # YOLO 都判不出（砖块 < 2，多在半空/空盘）：这一帧直接跳过。
        return None
    holes = (g["score"] < g["score_thr"]).ravel()
    idx = np.where(~holes)[0]
    labels = np.zeros(len(holes), dtype=np.int32)
    dist = None
    if idx.size >= 4:
        D = T.aligned_distance_matrix(g["dev"][idx])
        gap = T.gap_threshold(D)
        sub, _, _ = T.cluster_by_parity(D, gap)
        sub = T.merge_similar(sub, D, gap)
        uniq = sorted(set(sub.tolist()))
        remap = {u: i + 1 for i, u in enumerate(uniq)}
        labels[idx] = np.array([remap[int(v)] for v in sub], dtype=np.int32)
        dist = np.zeros((idx.size, idx.size), dtype=np.float32)
        dist[:] = D
    else:
        gap = 0.0

    xs, ys = g["xlines"], g["ylines"]
    pitch = float(np.median(np.concatenate([np.diff(xs), np.diff(ys)])))
    side = max(8, int(round(pitch * (1 - 2 * OUT_INSET))))
    ring = T._border_mask(TILE, max(2, TILE // 16))

    cells = []
    for j in range(g["rows"]):
        cy = (ys[j] + ys[j + 1]) / 2.0
        for i in range(g["cols"]):
            cx = (xs[i] + xs[i + 1]) / 2.0
            tile = cut_tile(img, cx, cy, side)
            bg = np.median(tile[ring], axis=0)          # 底板/边框色 = 状态代理
            cells.append({
                "row": j, "col": i,
                "label": int(labels[j * g["cols"] + i]),
                "hole": bool(holes[j * g["cols"] + i]),
                "ratio": float(g["score"][j, i]),
                "bg_bgr": bg.round(1).tolist(),
                "tile": tile,
            })
    return {"cells": cells, "rows": g["rows"], "cols": g["cols"],
            "dist": dist, "idx": idx, "gap": gap, "pitch": pitch}


# ---------------------------------------------------------------- 帧台账
# 为什么要台账：帧是一轮一轮加进来的，而你只想标注**新增的帧**。
# 台账按"截图内容 md5"认帧（不认文件名）—— 微信导出的图重名/改名都不怕，
# 同一张截图重复导入会自动跳过；旧帧一旦入库就**永不重切**（见 main 里的增量逻辑）。
LEDGER = "frames.csv"
LEDGER_COLS = ["frame", "source_file", "source_md5", "batch", "rows", "cols",
               "cells", "bricks", "empty", "status", "added_at", "labeled_at"]


def file_md5(path: str) -> str:
    """文件内容 md5：帧的身份，用它判断"这张截图是不是已经入库过"。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_ledger(out: str) -> list[dict]:
    p = os.path.join(out, LEDGER)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def read_index(out: str) -> list[dict]:
    p = os.path.join(out, "index.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def write_ledger(out: str, rows: list[dict]) -> None:
    with open(os.path.join(out, LEDGER), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        w.writeheader()
        w.writerows(rows)


def drop_high_conf(dataset: str, frames: list[str], keep_quantile: float,
                   abs_thr: float = 0.0, quiet: bool = False) -> int:
    """新增帧里只留"模型最不确定"的那部分格子。

    不用固定阈值：auto 组的 cls_dist 中位才 0.118，拿 app 换算出来的 0.30 当线
    会丢掉 98% 的格子（实测 1805 格里 1778 个 < 0.30）。改按本批自身分布取分位：
    keep_quantile=0.3 → 保留距离最大的 30%（即最难的样本）。

    整帧被过滤光时写进 rejected.txt，否则台账行没了、md5 也不在台账里，
    下次又会被当新帧重切，无限循环还一直往 deleted.csv 追加。
    """
    if not frames:
        return 0
    idx_path = os.path.join(dataset, "index.csv")
    rows = [r for r in csv.DictReader(open(idx_path, encoding="utf-8")) if r and r.get("path")]
    want = set(frames)
    cand = []
    for r in rows:
        if r["frame"] in want and r.get("cls_src") == "auto" and (r.get("cls_dist") or "").strip():
            try:
                cand.append((r, float(r["cls_dist"])))
            except ValueError:
                pass
    if not cand:
        return 0

    vals = sorted(d for _, d in cand)
    k = int(len(vals) * (1.0 - keep_quantile))
    k = min(max(k, 0), len(vals) - 1)
    cut = vals[k]
    if abs_thr > 0:
        cut = max(cut, abs_thr)

    drop = [r for r, d in cand if d < cut]
    if not drop:
        return 0

    trash = os.path.join(dataset, "deleted")
    ts = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(dataset, "deleted.csv"), "a", newline="") as f:
        w = csv.writer(f)
        for r in drop:
            w.writerow([r["path"], r["frame"], r["class_id"], r["cls_dist"], r["cls_src"],
                        ts, "highconf-skip"])
            src = os.path.join(dataset, r["path"])
            if os.path.exists(src):
                dst = os.path.join(trash, r["path"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

    drop_paths = {r["path"] for r in drop}
    keep = [r for r in rows if r["path"] not in drop_paths]
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)

    by = {}
    for r in keep:
        by[r["frame"]] = by.get(r["frame"], 0) + 1
    led_path = os.path.join(dataset, "frames.csv")
    led = [r for r in csv.DictReader(open(led_path, encoding="utf-8")) if r and r.get("frame")]
    out, gone = [], []
    for r in led:
        if r["frame"] in want:
            n = by.get(r["frame"], 0)
            if n == 0:
                gone.append(r["frame"])
                continue
            empty = sum(1 for x in keep if x["frame"] == r["frame"] and x.get("cls_src") == "empty")
            r["cells"], r["bricks"], r["empty"] = str(n), str(n - empty), str(empty)
        out.append(r)
    with open(led_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(led[0].keys()))
        w.writeheader()
        w.writerows(out)

    if gone:
        rej = os.path.join(dataset, "rejected.txt")
        old = {ln.strip() for ln in open(rej, encoding="utf-8")} if os.path.exists(rej) else set()
        with open(rej, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(old | set(gone))) + "\n")
        if not quiet:
            print(f"  {len(gone)} 帧全被过滤光，已写入 rejected.txt（不会再重切）")

    if not quiet:
        print(f"  高置信过滤: 丢弃 {len(drop)}/{len(cand)} 格"
              f"（保留距离最大的 {int(keep_quantile * 100)}%，cutoff {cut:.3f}）")
    return len(drop)


def main() -> None:
    ap = argparse.ArgumentParser(description="游戏截图 → 训练数据集（含底色）+ 底板库")
    ap.add_argument("--samples", default=os.path.join(_HERE, "samples"), help="截图目录")
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"),
                    help="图鉴素材库（取 tiles 当真实身份标签样本）；留空则不纳入")
    ap.add_argument("--out", default=os.path.join(_HERE, "dataset"), help="输出目录")
    ap.add_argument("--min-pitch", type=int, default=12)
    ap.add_argument("--rebuild", action="store_true",
                    help="整体重切（清空 images/frame 重来）。默认是**增量**：只处理 samples/ 里的新帧，"
                         "已入库的帧一律不碰（不动图片、不动标签）")
    ap.add_argument("--batch", default="",
                    help="本次新增帧的批次号，默认取当天日期（写进 frames.csv，便于按批追溯）")
    ap.add_argument("--keep-empty", action="store_true",
                    help="已废弃：现在每帧的格子一律全部落盘（空位标 class_id=0），不再丢图")
    ap.add_argument("--keep-labels", action="store_true",
                    help="重跑时保留旧 index.csv 里的 class_id 标注（人工改过的不会被清掉）")
    # ---- 后两步：模型识别 + 标注界面（都可单独关掉）----
    ap.add_argument("--backbone", default=auto_classify.DEFAULT_BACKBONE,
                    choices=["resnet18", "resnet50", "mobilenet_v3_small",
                             "mobilenet_v3_large", "dinov2_vits14"],
                    help="预训练骨干（识别与分组的特征来源）。改 auto_classify.DEFAULT_BACKBONE 即整体切换")
    ap.add_argument("--cache", default="", help="中间产物目录（默认 数据集同级 cache/）")
    ap.add_argument("--keep-quantile", type=float, default=0.30,
                    help="新增帧里只保留距离最大的这个比例（0.3=留下最难的 30%%，1=全留）")
    ap.add_argument("--highconf-thr", type=float, default=0.0,
                    help="额外的绝对下限（0=只用分位）。别再用 0.30：auto 组中位才 0.118")
    ap.add_argument("--keep-highconf", action="store_true", help="不做高置信过滤（保留全部新增格）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印会切哪些新帧，不写任何文件")
    ap.add_argument("--prune-empty", action="store_true",
                    help="把「台账里有、但一格都没切出来」的帧从台账剔除（下次会重切）")
    ap.add_argument("--skip-embed", action="store_true",
                    help="复用已有的 cache/emb_*.npz（图片没变时省 10 秒）")
    ap.add_argument("--no-classify", action="store_true", help="只裁图，不做模型识别")
    ap.add_argument("--no-html", action="store_true", help="不生成标注界面")
    ap.add_argument("--html", default=os.path.join(_HERE, "annotate.html"), help="标注界面输出路径")
    ap.add_argument("--auto-top", type=int, default=12, help="每格保存几个候选 class_id")
    ap.add_argument("--same-thr", type=float, default=0.10, help="分组：同款 embedding 距离阈值")
    ap.add_argument("--color-thr", type=float, default=0.35, help="分组：同款色相距离阈值")
    ap.add_argument("--pairs", action="store_true", help="额外导出帧内配对 pairs/*.csv（默认不导出）")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    frame_dir = os.path.join(img_dir, "frame")
    os.makedirs(frame_dir, exist_ok=True)
    stats = []
    batch = args.batch or datetime.date.today().strftime("%Y%m%d")

    # ---------- 0) 台账：按截图内容 md5 找出"新帧"，旧帧一律不碰 ----------
    prev_index, ledger = read_index(args.out), load_ledger(args.out)
    keep_status: dict = {}          # --rebuild --keep-labels 时带过来的"已标完"帧
    if not args.rebuild and prev_index and not ledger:
        # 一次性迁移：老数据集还没有台账。按帧名去 samples/ 找原图算 md5 补出台账，
        # 否则首次跑增量会把已入库的帧全当"新帧"重切一遍（还会污染 index.csv）。
        by_frame: dict[str, list[dict]] = {}
        for r in prev_index:
            by_frame.setdefault(r["frame"], []).append(r)
        for fr, rs in by_frame.items():
            src = next((fr + e for e in (".jpg", ".png", ".jpeg")
                        if os.path.exists(os.path.join(args.samples, fr + e))), "")
            if not src:
                print(f"  警告：台账补建时找不到样本文件 {fr}.*，该帧会被当作新帧重切")
                continue
            n_bg = sum(1 for r in rs if r.get("cls_src") == "empty")
            rows = max(int(os.path.basename(r["path"])[1:3]) for r in rs) + 1
            cols = max(int(os.path.basename(r["path"])[4:6]) for r in rs) + 1
            ledger.append({
                "frame": fr, "source_file": src, "batch": "bootstrap",
                "source_md5": file_md5(os.path.join(args.samples, src)),
                "rows": rows, "cols": cols, "cells": len(rs),
                "bricks": len(rs) - n_bg, "empty": n_bg,
                "status": "pending", "added_at": "", "labeled_at": "",
            })
        write_ledger(args.out, ledger)
        print(f"  台账缺失：已按现有 {len(ledger)} 帧补建 {LEDGER}（batch=bootstrap）")
    if args.rebuild:
        print("  --rebuild：清空 images/frame 并整体重切（旧帧也重切）")
        for d in os.listdir(frame_dir):
            shutil.rmtree(os.path.join(frame_dir, d), ignore_errors=True)
        if args.keep_labels:
            # 重切但保住标注：把旧台账的"标完"状态一并带过来，否则已标完的帧会重新冒出来
            keep_status = {r["frame"]: (r.get("status", ""), r.get("labeled_at", ""))
                           for r in ledger if r.get("status") == "done"}
        prev_index, ledger = [], []

    names = sorted(f for f in os.listdir(args.samples) if f.lower().endswith((".jpg", ".png")))
    md5_of = {n: file_md5(os.path.join(args.samples, n)) for n in names}
    by_md5 = {r["source_md5"]: r for r in ledger if r.get("source_md5")}
    # 黑名单：明确不再入库的截图（删过的重复图/弃用批次）。只跳过、不删原图 ——
    # 没有它的话，每次重跑都会把 samples/ 里没登记的图当新帧切回来。
    rej_path = os.path.join(args.out, "rejected.txt")
    rejected = set()
    if os.path.exists(rej_path):
        rejected = {ln.strip() for ln in open(rej_path, encoding="utf-8") if ln.strip()}
    new_names = [n for n in names
                 if md5_of[n] not in by_md5 and os.path.splitext(n)[0] not in rejected]
    skipped_rej = [n for n in names
                   if md5_of[n] not in by_md5 and os.path.splitext(n)[0] in rejected]
    if skipped_rej:
        print(f"  黑名单跳过 {len(skipped_rej)} 张（rejected.txt）")
    dup = [(n, by_md5[md5_of[n]]["frame"]) for n in names if md5_of[n] in by_md5]
    if dup:
        print(f"  跳过 {len(dup)} 张重复截图（内容与已有帧相同）："
              + "、".join(f"{n}=={f}" for n, f in dup[:3]) + ("…" if len(dup) > 3 else ""))
    if args.dry_run:
        print(f"  [dry-run] samples 里 {len(names)} 张，其中会被当新帧切的有 {len(new_names)} 张："
              f"{'、'.join(n[:20] for n in new_names[:5])}{'…' if len(new_names) > 5 else ''}")
        print("  [dry-run] 什么都没写。确认无误去掉 --dry-run 再跑。")
        return
    if not new_names:
        print(f"  没有新帧：samples/ 里 {len(names)} 张截图都已入库（要整体重切加 --rebuild）")
    else:
        print(f"  新帧 {len(new_names)} 张：" + "、".join(n[:16] for n in new_names[:4])
              + ("…" if len(new_names) > 4 else ""))

    index_rows, ledger_rows = [], []

    # ---------- 1) 新帧：自动标定 + 裁格（已入库的帧不重切，避免动到旧标注）----------
    # 帧名 = 截图原文件名（去掉扩展名）：目录/索引/命令行都能直接对上原图
    used = {r["frame"] for r in ledger} | {r["source_file"] for r in ledger}
    for name in new_names:
        img = T.load_image(os.path.join(args.samples, name))
        r = frame_cells(img)
        if r is None:
            print(f"  跳过 {name}（标定失败）")
            continue
        frame = os.path.splitext(name)[0]
        if frame in used:                      # 同名但内容不同：加后缀，绝不覆盖已入库的帧
            k = 2
            while f"{frame}_{k}" in used:
                k += 1
            frame = f"{frame}_{k}"
        used.add(frame)
        used.add(name)
        os.makedirs(os.path.join(frame_dir, frame), exist_ok=True)
        cells, idx, D, gap = r["cells"], r["idx"], r["dist"], r["gap"]
        rows_n = r["rows"]
        n_empty = 0
        for c in cells:
            # 15 行布局的最后一行被 App 悬浮窗盖住，切出来是黑格（真值是空位）。
            # 收进训练会污染 class 0 原型，也让"每帧最多 140 格"这条对不上。
            if rows_n >= 15 and c["row"] >= 14:
                continue
            if c["hole"]:
                n_empty += 1
            # 每帧的格子**全部落盘**（rows×cols 一个不少），保证"不丢图"：
            # 已消除的格子（只剩棋盘底色）标成 class_id=0、cls_src=empty —— 它自成一类，
            # 不再被丢掉。否则一旦"判空"判错（实测误丢过 37 格白栅栏/白花菜），
            # 砖块就从数据集里凭空消失，而且事后无法察觉。
            rel = os.path.join("images", "frame", frame, f"r{c['row']:02d}c{c['col']:02d}.png")
            cv2.imwrite(os.path.join(args.out, rel), c["tile"])
            index_rows.append({
                # 只要"图片 + class_id"这条线要的东西：path 指图、frame 用于分组/留出/回原图、
                # class_id 是标签（0=空位；auto_classify 填 auto；人工在 annotate.html 改的记 manual）、
                # cls_dist 只用于"只看低置信"筛选、cls_src 用来保护人工标注不被覆盖
                "path": rel, "frame": frame,
                "class_id": 0 if c["hole"] else "", "cls_dist": "",
                "cls_src": "empty" if c["hole"] else "",
            })
        # 同一帧内的配对：同类=正、异类且距离大=负（灰区不采信）。
        # 默认不导出：训练按帧内 label 采样，这份清单没人读，只用于人工核对。
        n_pairs = n_pos = 0
        if args.pairs and D is not None and idx.size:
            pairs = []
            for a in range(idx.size):
                for b in range(a + 1, idx.size):
                    ca, cb = cells[idx[a]], cells[idx[b]]
                    same = int(ca["label"] == cb["label"])
                    d = float(D[a, b])
                    conf = 1 if (same or d > max(gap, 0.15)) else 0
                    pairs.append((frame, ca["row"], ca["col"], frame, cb["row"], cb["col"],
                                  same, round(d, 4), conf))
            n_pairs, n_pos = len(pairs), sum(1 for p in pairs if p[6])
            os.makedirs(os.path.join(args.out, "pairs"), exist_ok=True)
            with open(os.path.join(args.out, "pairs", f"{frame}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_a", "ra", "ca", "frame_b", "rb", "cb",
                            "same", "distance", "confident"])
                w.writerows(pairs)
        # cells 记"实际落盘的格子数"：15 行布局丢掉被遮挡的最后一行，不等于 rows×cols
        n_cells = sum(1 for c in r["cells"]
                      if not (r["rows"] >= 15 and c["row"] >= 14))
        st, lab = keep_status.get(frame, ("pending", ""))
        ledger_rows.append({
            "frame": frame, "source_file": name, "source_md5": md5_of[name], "batch": batch,
            "rows": r["rows"], "cols": r["cols"], "cells": n_cells,
            "bricks": int(idx.size), "empty": n_empty,
            "status": st or "pending",
            "added_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "labeled_at": lab,
        })
        stats.append(f"{frame} {r['rows']:2d}x{r['cols']:<2d} 砖块{int(idx.size):3d} "
                     f"空位{n_empty:3d} 组{len(set(cells[i]['label'] for i in idx)):3d}"
                     + (f" 配对{n_pairs}" if args.pairs else ""))
        print("  " + stats[-1])

    if not index_rows and not prev_index:
        raise SystemExit("没产出任何数据：检查 --samples 里是否有截图")

    # 重切/重跑不丢标注：按 path 把上一版 index.csv 的 class_id/cls_dist/cls_src 回填到新行
    if args.keep_labels and prev_index:
        old = {r["path"]: r for r in prev_index}
        kept = 0
        for r in index_rows:
            o = old.get(r["path"])
            if o and o.get("cls_src"):
                r["class_id"], r["cls_dist"], r["cls_src"] = (
                    o.get("class_id", ""), o.get("cls_dist", ""), o.get("cls_src", ""))
                kept += 1
        print(f"  保留旧标注 {kept} 行（--keep-labels）")

    total_rows = prev_index + index_rows                 # 旧帧的行原样保留，只追加新帧
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(total_rows[0].keys()))
        w.writeheader()
        w.writerows(total_rows)

    # 台账里有、却一格都没切出来的帧：空目录 + status=done，永远不会再出现在标注页，
    # 也永远进不了训练。剔掉台账行，下次运行因 md5 不在台账里会被当新帧重切。
    have = {}
    for r in total_rows:
        have[r["frame"]] = have.get(r["frame"], 0) + 1
    all_ledger = ledger + ledger_rows
    bad = [r["frame"] for r in all_ledger if have.get(r["frame"], 0) == 0]
    if bad and args.prune_empty:
        # 默认完全静默：这些帧是已经处理完的历史批次（刻意不收高置信样本），
        # 反复提醒只会让人白花时间复查。要清理时显式 --prune-empty。
        print(f"\n  --prune-empty：从台账剔除 {len(bad)} 帧（一格都没有）")
        all_ledger = [r for r in all_ledger if have.get(r["frame"], 0) > 0]
    write_ledger(args.out, all_ledger)
    n_empty = sum(1 for r in total_rows if r.get("cls_src") == "empty")
    n_pending = sum(1 for r in all_ledger if r.get("status") != "done")
    new_frames = [r["frame"] for r in ledger_rows]
    print(f"\n① 裁图: 本次新增 {len(ledger_rows)} 帧 / {len(index_rows)} 格"
          f"（{TILE}x{TILE}）；库内共 {len(all_ledger)} 帧 / {len(total_rows)} 格"
          f"（砖块 {len(total_rows) - n_empty} + 背景 {n_empty}）")
    print(f"  索引 {args.out}/index.csv；台账 {args.out}/{LEDGER}：待标注 {n_pending} 帧"
          f"{'、本次新增 ' + str(len(new_frames)) + ' 帧' if new_frames else ''}")

    if args.no_classify:
        print("（--no-classify：跳过模型识别与标注界面）")
        return

    # ---------- 2) 模型识别：预训练特征最近邻 → class_id（只补新帧，不动旧帧的标签）----------
    cache = args.cache or auto_classify.cache_dir_of(args.out)
    emb_path = auto_classify.pick_emb(cache)
    if new_frames and args.skip_embed:
        print("  注意：有新帧却指定了 --skip-embed，缓存里没有新格子，识别会缺数据")
    need_embed = args.rebuild or bool(new_frames) or not emb_path
    if need_embed:
        print(f"② 识别: 提取特征（{args.backbone}）")
        emb_path = embed_features.extract(args.out, args.catalog, args.backbone)
    else:
        print(f"② 识别: 复用已有特征 {os.path.basename(emb_path)}（本次没有新帧）")
    auto_classify.classify(args.out, args.catalog, top=args.auto_top, emb=emb_path, cache=cache,
                           frames=None if args.rebuild else new_frames)

    # ---------- 2.5) 高置信样本不入库：现有数据够了，只补充"模型拿不准"的格子 ----------
    # 硬约束：只传本次新增帧（new_frames）。--rebuild 传空列表，等于不过滤 ——
    # 存量数据里高置信的样本是训练集的主体，删了模型就没法学了。
    # 别把这里改成"全库扫描"。
    if not args.keep_highconf:
        drop_high_conf(args.out, [] if args.rebuild else new_frames,
                       args.keep_quantile, args.highconf_thr)

    # ---------- 3) 标注界面 ----------
    if args.no_html:
        print("（--no-html：未生成标注界面）")
        return
    # only_pending：只铺 frames.csv 里 status != done 的帧（增量标注的入口）。
    # --rebuild 时台账刚被整体重置为 pending，所以这里传 True 也是全量。
    visualize_groups.build_html(args.out, args.catalog, args.samples, args.html,
                                top=args.auto_top, emb=emb_path, cache=cache,
                                same_thr=args.same_thr, color_thr=args.color_thr,
                                only_pending=True)
    print(f"\n③ 标注: 打开 {args.html}（只列未标完的帧）。人工改完 → 每帧点"
          f"'本帧标完' → 导出 JSON → apply_corrections.py --inplace 写回台账")


if __name__ == "__main__":
    main()
