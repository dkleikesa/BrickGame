#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 annotate.html 导出的 corrections.json 写回 dataset/index.csv 和 dataset/frames.csv。

corrections 的三种条目：
  {"path": "...", "class_id": 1428}   → class_id 改成人工判定值，cls_src=manual
  {"path": "...", "unknown": true}    → class_id 清空、cls_src=unknown（训练时当"未识别"忽略）
  "frames": ["a", "b", ...]           → 这帧人工看完了：frames.csv 里 status=done + labeled_at

**这一步就是增量的闭环**：写回后，下次 build_dataset.py / visualize_groups.py 生成标注界面时，
status=done 的帧不再铺出来 —— 界面里只剩还没标完的帧（新帧 + 上一批没点"本帧标完"的）。
"标完"的判定 = 导出时的 frames 显式声明 ∪ 本次改动涉及的帧（改了东西说明这帧看过）。

为什么不直接删行：留着才能事后统计"自动识别错了多少张"，而且重跑数据时人工标注还能按 path 回填。
为什么"标完"要写进台账而不是只看有没有改动：人工可能整帧复查后一个都没改 —— 不记账它就会一直冒出来。

默认只预览，加 --inplace 才写文件。（index.csv 可随时由 build_dataset.py 重建，
人工标注的原始记录请留好导出的 corrections.json。）

`apply()` 也被 `annotate_server.py` 复用 —— 标注页起了本地服务后，点「本帧标完」直接走这里
落库，不再需要导出 JSON。两条路径必须是同一份实现，否则"点按钮"与"跑脚本"的结果会漂。
用法：
  python apply_corrections.py --corrections corrections.json            # 预览
  python apply_corrections.py --corrections corrections.json --inplace  # 落地
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
LEDGER = "frames.csv"


def apply_to_ledger(ledger_path: str, done_frames: set[str]) -> list[str]:
    """把 done_frames 在 frames.csv 里标成 status=done + labeled_at，返回本次真正改动的帧名。

    没动的帧不重写（保持原有 labeled_at）—— 重复导出同一批不会把时间戳刷成新的。
    """
    if not os.path.exists(ledger_path) or not done_frames:
        return []
    with open(ledger_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("status", "labeled_at"):
        if col not in fieldnames:
            fieldnames.append(col)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hit = []
    for r in rows:
        if r.get("frame") in done_frames and r.get("status") != "done":
            r["status"], r["labeled_at"] = "done", stamp
            hit.append(r["frame"])
    if not hit:
        return []
    with open(ledger_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return hit


def apply(dataset: str, corr: dict, inplace: bool = False) -> dict:
    """corrections 字典 → dataset/index.csv + frames.csv。

    @param corr {"corrections": [{path, class_id} | {path, unknown}], "frames": [帧名]}
    @param inplace False 时只算不写（预览 / 自检用）。
    @return 统计：改动条数、标完的帧、无效条目（annotate_server 直接把这份回给页面）。
    """
    index_path = os.path.join(dataset, "index.csv")
    with open(index_path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("class_id", "cls_dist", "cls_src"):
        if col not in fieldnames:
            fieldnames.append(col)
            for r in rows:
                r.setdefault(col, "")
    by_path = {r["path"]: r for r in rows}

    n_set = n_unk = n_same = n_missing = 0
    missing: list[str] = []
    touched: set[str] = set()                     # 本次改动涉及的帧（"改了=看过"）
    for c in corr.get("corrections", []):
        r = by_path.get(c.get("path", ""))
        if r is None:
            n_missing += 1
            missing.append(str(c.get("path", "")))
            continue
        if c.get("unknown"):
            if r.get("cls_src") == "unknown":
                n_same += 1
            else:
                r["class_id"], r["cls_dist"], r["cls_src"] = "", "", "unknown"
                n_unk += 1
                touched.add(r.get("frame", ""))
        else:
            cid = int(c["class_id"])
            if r.get("cls_src") == "manual" and str(r.get("class_id")) == str(cid):
                n_same += 1
            else:
                r["class_id"], r["cls_src"] = cid, "manual"
                n_set += 1
                touched.add(r.get("frame", ""))

    # 标完的帧 = 导出时显式声明的 frames ∪ 改了东西的帧
    done_frames = {f for f in (corr.get("frames") or []) if f} | {f for f in touched if f}
    out = {"n_set": n_set, "n_unk": n_unk, "n_same": n_same, "n_missing": n_missing,
           "missing": missing, "frames": sorted(done_frames), "done": [],
           "has_ledger": os.path.exists(os.path.join(dataset, LEDGER)), "wrote": False}
    if not inplace:
        return out

    if n_set or n_unk:
        with open(index_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        out["wrote"] = True
    hit = apply_to_ledger(os.path.join(dataset, LEDGER), done_frames)
    out["done"] = hit
    out["wrote"] = out["wrote"] or bool(hit)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="标注 JSON → dataset/index.csv + frames.csv")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--corrections", default="corrections.json")
    ap.add_argument("--inplace", action="store_true", help="写回文件（默认只预览）")
    args = ap.parse_args()

    with open(args.corrections) as f:
        corr = json.load(f)
    res = apply(args.dataset, corr, inplace=args.inplace)
    for p in res["missing"]:
        print(f"  跳过（index.csv 里没有这条）：{p}")
    print(f"标注文件: {args.corrections}（{len(corr.get('corrections', []))} 条改动 / "
          f"{len(corr.get('frames') or [])} 帧声明标完）")
    print(f"将应用: 指派 {res['n_set']} 条 / 标未知 {res['n_unk']} 条 / "
          f"无变化 {res['n_same']} 条 / 无效 {res['n_missing']} 条")
    if res["done"]:
        print(f"将标完: {len(res['done'])} 帧（frames.csv 的 status=done，之后不再出现在标注界面）")
    elif res["frames"]:
        print("（没有 frames.csv 台账，跳过标完；老数据集重跑 build_dataset.py 会补建台账）")
    if not args.inplace:
        print("\n预览模式：没有写文件。加 --inplace 落地。")
        return
    if res["wrote"]:
        print(f"\n已写回 {args.dataset}/index.csv"
              + (f" + {LEDGER}（{len(res['done'])} 帧标完）" if res["done"] else ""))
    else:
        print("\n没有任何改动，未写文件。")


if __name__ == "__main__":
    main()
