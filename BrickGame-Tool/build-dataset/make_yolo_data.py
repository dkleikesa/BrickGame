#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装配 YOLO 定位训练集（0=board，1=tile）：**只搬框，不生成框**。

标注唯一来源，按优先级：

  1. `yolo/labels/{train,val}/<base>.txt` —— 已经装配过的框，box_server 保存后就是人工确认过的
  2. `yolo/prelabel/<帧名>.txt`         —— prelabel_yolo.py 用当前模型出的预标注
  两者都没有 → 跳过并打印原因。

**这里不再从逐格标签反推框，也不再调用 OpenCV 定网格。**
逐格标签只覆盖棋盘的一部分（实测 3047/5050 = 60%），反推框又必须重新算像素网格；
残局/空盘帧的自相关会取到离谱的网格（实测 191×48），一帧就是 140 个错框。
这条弯路已经删了，完整流程见 `.codebuddy/skills/brickgame-yolo-locator/SKILL.md`：
`prelabel_yolo.py` → `box_server.py`（人工删/补）→ 本脚本 → `train_yolo.py`。

产物：
  yolo/images/{train,val}/<base>.*   帧图（按 split 复制，长边缩到 MAX_SIDE=1024）
  yolo/labels/{train,val}/<base>.txt 框（被 box_server 就地修改，本脚本不覆盖已确认的）
  yolo/meta.json                     帧清单 + split + reviewed 标记
  yolo/data.yaml                     ultralytics 数据集描述

存图尺寸与模型输入尺寸都在 yolo_config.py 里定：MAX_SIDE = IMG_H = 1024，
模型输入 1024x480（高x宽）。存图高度与模型输入高度相等，内容 1:1 不放大；
宽 480 覆盖 461 宽的内容，只补少量灰边。改尺寸后**必须**重跑本脚本，把旧尺寸的图
从 samples/ 原图重做（框是归一化的，重做不会错位）。

用法：
  .venv-train/bin/python build-dataset/make_yolo_data.py
  ... --refresh-prelabel    # 用新预标注覆盖"还没人工确认"的帧
  ... --rebalance           # 重新抽 val（会让指标失去可比性，除非换代）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))       # BrickGame-Tool/，为了 import yolo_config
from yolo_config import IMG_H, IMG_W, MAX_SIDE   # noqa: E402

OUT = os.path.join(HERE, "yolo")
REVIEWED = os.path.join(OUT, "reviewed.json")
PRELABEL = os.path.join(OUT, "prelabel")


def target_size(w: int, h: int) -> tuple[int, int]:
    """按 MAX_SIDE 等比缩放后的目标尺寸；本来就不超的不放大。"""
    longest = max(w, h)
    if longest <= MAX_SIDE:
        return w, h
    return (round(w * MAX_SIDE / longest), round(h * MAX_SIDE / longest))


def copy_image(src: str, dst: str) -> tuple[bool, tuple[int, int]]:
    """复制帧图并按 MAX_SIDE 等比缩放。

    @return (是否重写了文件, 存图尺寸)。

    判据是**目标尺寸**，不是"长边没超 MAX_SIDE"。640 → 1024 升级时旧图长边
    640 也满足 ≤1024，旧逻辑会直接跳过，训练于是还在用 640 的图 ——
    放大了但小砖的像素信息并没有回来，白跑一轮。目标尺寸一致才跳过，
    顺带保证反复装配不会把 PNG/JPG 反复重编码。
    """
    with Image.open(src) as im:
        w, h = im.size
        tw, th = target_size(w, h)
        if os.path.exists(dst):
            with Image.open(dst) as done:
                if done.size == (tw, th):
                    return False, (tw, th)      # 已按当前 MAX_SIDE 装配过
        if (tw, th) != (w, h):
            im = im.resize((tw, th), Image.LANCZOS)
        if dst.lower().endswith((".jpg", ".jpeg")):
            im.convert("RGB").save(dst, "JPEG", quality=90, subsampling=0)
        else:
            im.save(dst, "PNG", optimize=True)
    return True, (tw, th)


def base_of(stem: str) -> str:
    """帧名 → 稳定的 ASCII 文件名。

    用内容无关的定长哈希而不是 f000 序号：以后新加截图不会把后面的编号整体后移，
    否则 labels/<base>.txt 会跟帧对不上（人工改过的框就串到别的帧去了）。
    """
    return "f" + hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]


def reviewed_set() -> set[str]:
    """box_server 点过保存的帧（人工确认过）。"""
    if not os.path.exists(REVIEWED):
        return set()
    try:
        return set(json.load(open(REVIEWED, encoding="utf-8")).get("frames", []))
    except (json.JSONDecodeError, OSError):
        return set()


def label_dir(split: str) -> str:
    return os.path.join(OUT, "labels", split)


def find_label(stem: str) -> tuple[str | None, str | None]:
    """已在训练集里的框文件（可能在 train 也可能在 val，split 改过也能找到）。"""
    name = base_of(stem) + ".txt"
    for split in ("train", "val"):
        path = os.path.join(label_dir(split), name)
        if os.path.exists(path):
            return path, split
    return None, None


def read_labels(path: str) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cls = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
        except ValueError:
            continue
        if cls in (0, 1):
            out.append((cls, cx, cy, bw, bh))
    return out


def ledger_shapes() -> dict[str, tuple[int, int]]:
    """行列数（只用来统计、分层抽 val、按格比漏标）—— 不参与框的计算。

    frames.csv 只覆盖早期那批（114 帧），后来收的 lowconf 帧没有台账。
    所以先把**上一次装配**的 meta.json 里的 rows/cols 兜进来，否则每次重跑
    都会把这 86 帧洗成 0，audit_labels 的"按格比漏标"直接退化回按框数比。
    台账优先：人工修正过 frames.csv 的帧以台账为准。
    """
    shapes: dict[str, tuple[int, int]] = {}
    meta_path = os.path.join(OUT, "meta.json")
    if os.path.exists(meta_path):
        try:
            for m in json.load(open(meta_path, encoding="utf-8")):
                if m.get("rows") and m.get("cols"):
                    shapes[m["src"]] = (int(m["rows"]), int(m["cols"]))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass            # meta 坏了就当没有，下面从台账重建
    path = os.path.join(HERE, "dataset", "frames.csv")
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            shapes.update({r["frame"]: (int(r["rows"]), int(r["cols"]))
                           for r in csv.DictReader(f)})
    return shapes


def audit(meta: list[dict]) -> None:
    """标注质量报表：覆盖率、异常框、无台账帧。

    漏检在训练里等价于"这里没有砖"，看不出来就会一直训错。没有 rows/cols 的帧
    （frames.csv 台账没覆盖）用 tile 面积和 / board 面积估算，不让它们从报表里消失。
    """
    import statistics as st
    by_cell, weird = [], []
    dup_frames, dup_total = [], 0
    for m in meta:
        p = os.path.join(OUT, "labels", m["split"], m["file"] + ".txt")
        board = tiles = 0.0
        board_box = None
        cell_hit: dict[tuple, int] = {}
        for line in open(p, encoding="utf-8"):
            q = line.split()
            if len(q) < 5:
                continue
            c, cx, cy, w, h = int(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])
            if c == 0:
                if w * h > board:
                    board, board_box = w * h, (cx, cy, w, h)
            elif c == 1:
                tiles += w * h
                if w > 0.5 or h > 0.5:
                    weird.append((m["src"], round(w, 2), round(h, 2)))
                # 同格多框：旧 NMS 0.7 生成的预标注会带进来，一格两块砖会把格距算大
                if m["rows"] and m["cols"] and board_box:
                    bx, by, bw, bh = board_box
                    bc = int((cx - (bx - bw / 2)) / (bw / m["cols"]))
                    br = int((cy - (by - bh / 2)) / (bh / m["rows"]))
                    cell_hit[(br, bc)] = cell_hit.get((br, bc), 0) + 1
        n_dup = sum(v - 1 for v in cell_hit.values() if v > 1)
        if n_dup:
            dup_frames.append((m["src"], n_dup))
            dup_total += n_dup
        if m["rows"] and m["cols"]:
            by_cell.append((m, m["tiles"] / (m["rows"] * m["cols"])))
    # 覆盖率只作参考：残局帧天然低（2 块砖 = 1.4%），拿它判漏标会大面积误报。
    # 真要查漏标跑 audit_labels.py（按格子集合比）。
    if by_cell:
        vals = [c for _, c in by_cell]
        low = sum(1 for _, c in by_cell if c < 0.5)
        print(f"\n标注覆盖：{len(vals)} 帧有台账，中位 {st.median(vals):.0%}，<50% 的 {low} 帧"
              f"（残局天然低，漏标请用 audit_labels.py 判）")
    if weird:
        print(f"  ⚠ {len(weird)} 个 tile 框边长 >50%（多半是把半块棋盘框成了砖）：{weird[:3]}")
    if dup_total:
        print(f"  ⚠ {len(dup_frames)} 帧存在同格多框（共 {dup_total} 个）：{dup_frames[:3]}"
              f"\n    一格最多一块砖，重复框会把格距算大，建议删掉多出来的")


def archive_stale(keep_images: set[str], keep_labels: set[str], reviewed: set[str]) -> tuple[list[str], int, int]:
    """把本次没装配到的图/框挪进 _trash，不再直接删。

    旧版无条件 os.remove：samples 少一帧，脚本就会把这一帧人工确认过的 label 一起删掉
    （实测一次丢了 47 个）。现在已确认的原地保留并告警，其余进 _trash 可捞回。
    """
    trash = os.path.join(OUT, "_trash")
    orphaned, n_img, n_lab = [], 0, 0
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(OUT, sub)
        for name in os.listdir(d):
            full = os.path.join(d, name)
            if not os.path.isfile(full):
                continue
            if sub.startswith("images"):
                if f"{sub[7:]}/{name}" in keep_images:
                    continue
                n_img += 1
            else:
                if not name.endswith(".txt") or name in keep_labels:
                    continue
                base = os.path.splitext(name)[0]
                if base in reviewed:
                    orphaned.append(base)      # 人工确认过的框，宁可留着也不动
                    continue
                n_lab += 1
            dest = os.path.join(trash, sub, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(full, dest)
    return orphaned, n_img, n_lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--val", type=int, default=6)
    ap.add_argument("--rebalance", action="store_true", help="重新抽 val（指标会失去可比性）")
    ap.add_argument("--refresh-prelabel", action="store_true",
                    help="用 prelabel 覆盖还没人工确认的帧（确认过的一律不动）")
    args = ap.parse_args()

    shapes = ledger_shapes()
    reviewed = reviewed_set()
    frames, skipped, no_prelabel = [], [], []

    for name in sorted(os.listdir(args.samples)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        lab_path, _ = find_label(stem)
        is_reviewed = base_of(stem) in reviewed
        use_existing = lab_path is not None and not (args.refresh_prelabel and not is_reviewed)
        if use_existing:
            labels, src = read_labels(lab_path), "labels"
        else:
            pre = os.path.join(PRELABEL, stem + ".txt")
            if not os.path.exists(pre):
                no_prelabel.append(stem)
                skipped.append((stem, "无预标注"))
                continue
            labels, src = read_labels(pre), "prelabel"

        n_board = sum(1 for l in labels if l[0] == 0)
        n_tile = sum(1 for l in labels if l[0] == 1)
        if not labels:
            skipped.append((stem, "标注为空"))
            continue
        if n_board == 0:
            skipped.append((stem, "缺 board 框"))
            continue
        # 空盘帧是有效样本（定位器要能认"有板没砖"），但必须是人工确认过的 ——
        # 未确认的零 tile 更可能是模型整体漏检，放进去等于教它"整板都没砖"。
        if n_tile == 0 and not (is_reviewed and n_board > 0):
            skipped.append((stem, "没有 tile 框"))
            continue

        rows, cols = shapes.get(stem, (0, 0))
        frames.append({"stem": stem, "file": name, "rows": rows, "cols": cols,
                       "tiles": n_tile, "labels": labels, "src": src,
                       "overwrite": args.refresh_prelabel and not is_reviewed})

    if not frames:
        raise SystemExit("没有可用的帧：先跑 prelabel_yolo.py，再用 box_server.py 人工过一遍")

    # 划分必须**持久化**：每次重新抽 val 会让指标不可比，也等于偷偷把评测集喂进训练。
    # 规则：split.json 里已有的帧沿用；新帧一律进 train；只有首次运行才按布局分层抽 val。
    frames.sort(key=lambda f: f["stem"])
    split_path = os.path.join(OUT, "split.json")
    split: dict[str, str] = {}
    if os.path.exists(split_path):
        split = json.load(open(split_path, encoding="utf-8"))
    if args.rebalance or not split:
        split = {}
        seen: dict[str, list[int]] = {}
        for i, f in enumerate(frames):
            seen.setdefault(f"{f['rows']}x{f['cols']}", []).append(i)
        val_idx: set[int] = set()
        for key in sorted(seen):
            ids = seen[key]
            if len(ids) >= 3:
                val_idx.add(ids[len(ids) // 2])
        while len(val_idx) < min(args.val, len(frames) - 1):
            for i in range(len(frames) - 1, -1, -1):
                if i not in val_idx:
                    val_idx.add(i)
                    break
            else:
                break
        for i, f in enumerate(frames):
            split[f["stem"]] = "val" if i in val_idx else "train"
        json.dump(split, open(split_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for f in frames:
        split.setdefault(f["stem"], "train")
    # 每次都落盘：之前只在首次/--rebalance 时写，新帧只存在内存里，
    # 结果 split.json 永远停在首批 44 条，val 永远是那 6 帧。
    json.dump(split, open(split_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    keep = {base_of(f["stem"]) + ".txt" for f in frames}
    keep_images: set[str] = set()

    meta = []
    rebuilt, lowres = 0, []
    for f in frames:
        stem, split_of = f["stem"], split[f["stem"]]
        base = base_of(stem)
        img_name = base + os.path.splitext(f["file"])[1]
        wrote, (sw, sh) = copy_image(os.path.join(args.samples, f["file"]),
                                     os.path.join(OUT, "images", split_of, img_name))
        rebuilt += int(wrote)
        if max(sw, sh) < MAX_SIDE:
            lowres.append(f"{img_name}({sw}x{sh})")
        keep_images.add(f"{split_of}/{img_name}")

        target = os.path.join(label_dir(split_of), base + ".txt")
        old_path, _ = find_label(stem)
        if f["src"] == "labels" and old_path:
            if os.path.abspath(old_path) != os.path.abspath(target):
                os.replace(old_path, target)                       # split 变了，跟着搬
                orig = old_path + ".orig"
                if os.path.exists(orig):
                    os.replace(orig, target + ".orig")
        else:
            # 来自预标注：默认只在目标不存在时写，绝不覆盖人工改过的框；
            # --refresh-prelabel 只对"没确认过"的帧覆盖。
            if f["overwrite"] or not os.path.exists(target):
                with open(target, "w", encoding="utf-8") as fh:
                    for cls, cx, cy, bw, bh in f["labels"]:
                        fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # labels_from 要反映框的真实来源：未确认帧即便上次已落盘成 labels/*.txt，
        # 框还是模型给的，不能记成人工标注（否则第二次装配后全部洗白成 labels）。
        is_rev = base_of(stem) in reviewed
        from_src = f["src"] if (f["src"] != "labels" or is_rev) else "prelabel"
        meta.append({"file": base, "src": stem, "split": split_of, "rows": f["rows"],
                     "cols": f["cols"], "tiles": f["tiles"],
                     "labels_from": from_src, "reviewed": is_rev})

    orphaned, n_trash_img, n_trash_lab = archive_stale(keep_images, keep, reviewed)
    audit(meta)

    with open(os.path.join(OUT, "data.yaml"), "w") as fh:
        fh.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n")
        fh.write("names:\n  0: board\n  1: tile\n")
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    n_train = sum(1 for m in meta if m["split"] == "train")
    n_tile = sum(m["tiles"] for m in meta)
    n_unreviewed = sum(1 for m in meta if not m["reviewed"])
    print(f"帧 {len(meta)}（train {n_train} / val {len(meta) - n_train}），"
          f"tile 框 {n_tile}，board 框 {len(meta)}")
    print(f"存图长边 {MAX_SIDE}（模型输入 {IMG_H}x{IMG_W}，只补左右灰边），"
          f"本次重做 {rebuilt} 张")
    if lowres:
        print(f"  ⚠ {len(lowres)} 张原图本身不到 {MAX_SIDE}px，只能原样存（细节回不来）："
              f"{', '.join(lowres[:5])}{' …' if len(lowres) > 5 else ''}")
    by_layout: dict[str, list[int]] = {}
    for m in meta:
        if m["rows"]:
            by_layout.setdefault(f"{m['rows']}x{m['cols']}", []).append(m["tiles"])
    for k in sorted(by_layout):
        v = by_layout[k]
        print(f"  {k}: {len(v)} 帧，tile 框 {sum(v)}（每帧 {min(v)}~{max(v)}）")
    if skipped:
        print(f"跳过 {len(skipped)} 帧：" + ", ".join(f"{a}({b})" for a, b in skipped))
    if no_prelabel:
        print(f"\n⚠ 有 {len(no_prelabel)} 帧连预标注都没有，先跑：\n"
              f"  .venv-train/bin/python build-dataset/prelabel_yolo.py")
    if n_unreviewed:
        print(f"\n⚠ {n_unreviewed}/{len(meta)} 帧还没人工确认（模型预标注直接进了训练集）。\n"
              f"  训练前先过一遍：.venv-train/bin/python build-dataset/box_server.py\n"
              f"  确认完重新跑本脚本，再 train_yolo.py。")
    if n_trash_img or n_trash_lab:
        print(f"\n归档到 yolo/_trash：{n_trash_img} 张图、{n_trash_lab} 个未确认框"
              f"（samples 里已没有对应原图）")
    if orphaned:
        print(f"\n⚠ {len(orphaned)} 个人工确认过的框对应原图不在 samples，已原地保留："
              f"{', '.join(orphaned[:5])}{' ...' if len(orphaned) > 5 else ''}\n"
              f"  这些帧不会进本次训练集，把原图放回 samples 再重跑本脚本即可恢复。")
    print(f"输出 → {OUT}")


if __name__ == "__main__":
    main()
