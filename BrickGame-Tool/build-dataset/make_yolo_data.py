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
  yolo/images/{train,val}/<base>.*   帧图（按 split 复制）
  yolo/labels/{train,val}/<base>.txt 框（被 box_server 就地修改，本脚本不覆盖已确认的）
  yolo/meta.json                     帧清单 + split + reviewed 标记
  yolo/data.yaml                     ultralytics 数据集描述

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
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "yolo")
REVIEWED = os.path.join(OUT, "reviewed.json")
PRELABEL = os.path.join(OUT, "prelabel")


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
    """台账里的行列数，只用来统计和分层抽 val —— 不参与框的计算。"""
    path = os.path.join(HERE, "dataset", "frames.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["frame"]: (int(r["rows"]), int(r["cols"])) for r in csv.DictReader(f)}


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
        if n_tile == 0:
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

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    for sub in ("images/train", "images/val"):
        for old in os.listdir(os.path.join(OUT, sub)):
            os.remove(os.path.join(OUT, sub, old))

    keep = {base_of(f["stem"]) + ".txt" for f in frames}
    for s in ("train", "val"):
        for old in os.listdir(label_dir(s)):
            if old.endswith(".txt") and old not in keep:
                os.remove(os.path.join(label_dir(s), old))

    meta = []
    for f in frames:
        stem, split_of = f["stem"], split[f["stem"]]
        base = base_of(stem)
        shutil.copyfile(os.path.join(args.samples, f["file"]),
                        os.path.join(OUT, "images", split_of, base + os.path.splitext(f["file"])[1]))

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

        meta.append({"file": base, "src": stem, "split": split_of, "rows": f["rows"],
                     "cols": f["cols"], "tiles": f["tiles"],
                     "labels_from": f["src"], "reviewed": base_of(stem) in reviewed})

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
    print(f"输出 → {OUT}")


if __name__ == "__main__":
    main()
