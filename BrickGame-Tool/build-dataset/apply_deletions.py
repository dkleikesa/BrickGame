#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""删除数据集里指定的截图格：从 dataset/index.csv 摘掉这些行（review.html / 本地服务都调这里的函数）。

默认**只摘行 + 把图片挪进 dataset/deleted/**（按原相对路径存放），不是真删 —— 标错了能恢复：
  dataset/deleted.csv      被摘掉的行（path/frame/class_id/cls_dist/cls_src + 摘除时间）
  dataset/deleted/<原路径>  对应的图片文件
  --keep-files  图片原地不动，只摘 index.csv 的行
  --purge       真删图片文件（不可逆，只有确认不要了才用）

为什么摘行而不是就地加一个"废弃"标记：训练（train_classifier.py 的 Real）就是按 index.csv
逐行收样本，摘掉即生效，不用改训练代码；被摘的行全量落进 deleted.csv，统计和回滚都不丢。

撤销：restore_rows() 把行插回 index.csv 并把图片从 deleted/ 挪回原位（review_server.py 的
"撤销上次删除"用的就是它）。--purge 过的批次恢复不了图片，只能恢复行。

**删完要刷新标注界面/特征缓存**（embed 缓存按 index.csv 对齐，多余条目无害，要干净就重跑）：
  .venv/bin/python embed_features.py --dataset dataset --catalog ../build-catalog/catalog
注意：build_dataset.py --rebuild 会重切整库，被摘的行会跟着回来（deleted.csv 还在，可再跑一次）。

用法：
  python apply_deletions.py --deletions deletions.json              # 预览（默认）
  python apply_deletions.py --deletions deletions.json --inplace    # 摘行 + 挪图到 deleted/
  python apply_deletions.py --deletions deletions.json --inplace --purge
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = "frames.csv"
TRASH = "deleted"
TRASH_CSV = "deleted.csv"
INDEX = "index.csv"


def load_index(dataset: str) -> tuple[list[str], list[dict]]:
    with open(os.path.join(dataset, INDEX)) as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def write_index(dataset: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(os.path.join(dataset, INDEX), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def recount_ledger(dataset: str, kept: list[dict], touched: set[str]) -> int:
    """按剩下的行重算被删帧的 cells/bricks/empty，别让台账里留着老的格子数。"""
    p = os.path.join(dataset, LEDGER)
    if not touched or not os.path.exists(p):
        return 0
    with open(p) as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames or [])
        ledger = list(rd)
    cnt: dict[str, list[int]] = {}
    for r in kept:
        c = cnt.setdefault(r["frame"], [0, 0])          # [格子数, 背景数]
        c[0] += 1
        if r.get("cls_src") == "empty":
            c[1] += 1
    hit = 0
    for r in ledger:
        fr = r.get("frame")
        if fr not in touched:
            continue
        cells, empty = cnt.get(fr, [0, 0])
        r["cells"], r["empty"], r["bricks"] = cells, empty, cells - empty
        hit += 1
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(ledger)
    return hit


def _move_files(dataset: str, rows: list[dict], purge: bool) -> int:
    """图片挪进 deleted/（保留原目录结构）或真删；文件本来就不在的跳过。"""
    n = 0
    for r in rows:
        src = os.path.join(dataset, r["path"])
        if not os.path.exists(src):
            continue
        if purge:
            os.remove(src)
        else:
            dst = os.path.join(dataset, TRASH, r["path"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        n += 1
    return n


def _restore_files(dataset: str, rows: list[dict]) -> int:
    n = 0
    for r in rows:
        src = os.path.join(dataset, TRASH, r["path"])
        dst = os.path.join(dataset, r["path"])
        if not os.path.exists(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        n += 1
    if n:
        _prune_empty_dirs(os.path.join(dataset, TRASH))
    return n


def _prune_empty_dirs(root: str) -> None:
    """搬空后的空目录一并清掉，别留一堆空壳（自底向上删）。"""
    for cur, _dirs, _files in os.walk(root, topdown=False):
        try:
            if not os.listdir(cur):         # 现查现删，os.walk 给的列表是扫描时的快照，会漏掉刚空的父目录
                os.rmdir(cur)
        except OSError:
            pass


def _drop_empty_frame_dirs(dataset: str, frames: list[str]) -> None:
    for fr in frames:                       # 整帧被摘空 → 目录也清掉，别留空壳
        d = os.path.join(dataset, "images", "frame", fr)
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass


def _trash_cols(fieldnames: list[str]) -> list[str]:
    return [c for c in fieldnames if c not in ("deleted_at", "reason")] + ["deleted_at", "reason"]


def apply_paths(dataset: str, paths: list[str], purge: bool = False,
                keep_files: bool = False, reason: str = "dedup") -> dict:
    """摘掉这些 path（相对 dataset 的路径）。返回统计 dict；没有任何命中时不动文件。"""
    fieldnames, rows = load_index(dataset)
    want = list(dict.fromkeys(paths))                    # 去重，保持顺序
    wantset, have = set(want), {r["path"] for r in rows}
    hits = [r for r in rows if r["path"] in wantset]
    missing = [p for p in want if p not in have]
    res = {"matched": len(hits), "missing": missing, "rows": hits, "purged": bool(purge),
           "keep_files": bool(keep_files), "total_before": len(rows),
           "total_after": len(rows) - len(hits), "files": 0, "frames": []}
    if not hits:
        return res
    kept = [r for r in rows if r["path"] not in wantset]
    write_index(dataset, fieldnames, kept)

    trash_csv = os.path.join(dataset, TRASH_CSV)
    new = not os.path.exists(trash_csv)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(trash_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_trash_cols(fieldnames))
        if new:
            w.writeheader()
        for r in hits:
            w.writerow({**r, "deleted_at": stamp, "reason": reason})

    if not keep_files:
        res["files"] = _move_files(dataset, hits, purge)
    frames = sorted({r["frame"] for r in hits})
    res["frames"] = frames
    res["ledger"] = recount_ledger(dataset, kept, set(frames))
    if not keep_files:
        _drop_empty_frame_dirs(dataset, frames)
    return res


def restore_rows(dataset: str, rows: list[dict], move_back: bool = True) -> dict:
    """撤销一批删除：行插回 index.csv（追加，顺序不影响训练/标注），图片从 deleted/ 挪回原位。"""
    fieldnames, cur = load_index(dataset)
    have = {r["path"] for r in cur}
    back = [{k: r.get(k, "") for k in fieldnames} for r in rows if r["path"] not in have]
    if not back:
        return {"restored": 0, "files": 0}
    write_index(dataset, fieldnames, cur + back)
    files = _restore_files(dataset, back) if move_back else 0
    keep = {r["path"] for r in back}
    p = os.path.join(dataset, TRASH_CSV)
    if os.path.exists(p):                                # 从 deleted.csv 里划掉这批（别留重复档案）
        with open(p) as f:
            rd = csv.DictReader(f)
            cols = list(rd.fieldnames or [])
            left = [r for r in rd if r["path"] not in keep]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(left)
    frames = {r["frame"] for r in back}
    res = {"restored": len(back), "files": files, "frames": sorted(frames),
           "ledger": recount_ledger(dataset, cur + back, frames)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="deletions.json → 从 dataset 摘掉这些图")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--deletions", default="deletions.json")
    ap.add_argument("--inplace", action="store_true", help="写回文件（默认只预览）")
    ap.add_argument("--purge", action="store_true", help="真删图片文件（默认挪进 dataset/deleted/）")
    ap.add_argument("--keep-files", action="store_true", help="图片原地不动，只摘 index.csv 的行")
    args = ap.parse_args()

    with open(args.deletions) as f:
        job = json.load(f)
    fieldnames, rows = load_index(args.dataset)
    by_path = {r["path"]: r for r in rows}
    want = {}
    for d in job.get("deletes", []):
        want[d["path"]] = d
    hits = [by_path[p] for p in want if p in by_path]
    missing = [p for p in want if p not in by_path]
    gone = [r["path"] for r in hits if not os.path.exists(os.path.join(args.dataset, r["path"]))]

    by_cid: dict[str, int] = {}
    for r in hits:
        k = r.get("class_id") or "未识别"
        by_cid[k] = by_cid.get(k, 0) + 1
    print(f"清单: {args.deletions}（{len(want)} 条，{job.get('created', '')[:19]}）")
    print(f"命中 {len(hits)} 条 / 数据库 {len(rows)} 条 → 摘掉后剩 {len(rows) - len(hits)} 条")
    top = sorted(by_cid.items(), key=lambda kv: -kv[1])[:12]
    print("  按类: " + "、".join(f"{c}:{n}" for c, n in top) + ("…" if len(by_cid) > len(top) else ""))
    if missing:
        print(f"  无效 {len(missing)} 条（index.csv 里没有，可能已摘过）：" + "、".join(missing[:3]))
    if gone:
        print(f"  注意 {len(gone)} 条图片文件已不存在（只摘行）：" + "、".join(gone[:3]))
    if not hits:
        print("\n没有可摘的行，未改任何文件。")
        return
    how = "真删图片文件" if args.purge else ("图片原地不动" if args.keep_files else f"图片挪到 {TRASH}/")
    print(f"落地方式: 摘 index.csv 的行 + {how}；被摘的行记入 {TRASH_CSV}")
    if not args.inplace:
        print("\n预览模式：没有写文件。加 --inplace 落地。")
        return

    res = apply_paths(args.dataset, [p for p in want if p in by_path], purge=args.purge,
                      keep_files=args.keep_files)
    print(f"\n已摘掉 {res['matched']} 行 → {os.path.join(args.dataset, INDEX)}")
    print(f"已留档 {res['matched']} 行 → {os.path.join(args.dataset, TRASH_CSV)}")
    if not args.keep_files:
        print(f"{'已删除' if args.purge else '已挪走'} {res['files']} 个图片文件"
              f"{'' if args.purge else f' → {os.path.join(args.dataset, TRASH)}/'}")
    if res.get("ledger"):
        print(f"已按剩余行重算 {res['ledger']} 帧的 cells/bricks/empty → "
              f"{os.path.join(args.dataset, LEDGER)}")


if __name__ == "__main__":
    main()
