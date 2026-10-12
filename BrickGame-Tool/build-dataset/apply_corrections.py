#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 annotate.html 导出的 corrections.json 写回 dataset/index.csv。

corrections 的两种条目：
  {"path": "...", "class_id": 1428}   → class_id 改成人工判定值，cls_src=manual
  {"path": "...", "unknown": true}    → class_id 清空、cls_src=unknown（训练时当"未识别"忽略）

为什么不直接删行：留着才能事后统计"自动识别错了多少张"，而且 --keep-labels 重跑数据时
人工标注还能按 path 回填。

默认只预览，加 --inplace 才写文件。（index.csv 可随时由 build_dataset.py 重建，
人工标注的原始记录请留好导出的 corrections.json。）
用法：
  python apply_corrections.py --corrections corrections.json            # 预览
  python apply_corrections.py --corrections corrections.json --inplace  # 落地
"""
from __future__ import annotations

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）


def main() -> None:
    ap = argparse.ArgumentParser(description="标注 JSON → dataset/index.csv")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--corrections", default="corrections.json")
    ap.add_argument("--inplace", action="store_true", help="写回 index.csv（默认只预览）")
    args = ap.parse_args()

    with open(args.corrections) as f:
        corr = json.load(f)
    index_path = os.path.join(args.dataset, "index.csv")
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
    for c in corr.get("corrections", []):
        r = by_path.get(c["path"])
        if r is None:
            n_missing += 1
            print(f"  跳过（index.csv 里没有这条）：{c['path']}")
            continue
        if c.get("unknown"):
            if r.get("cls_src") == "unknown":
                n_same += 1
            else:
                r["class_id"], r["cls_dist"], r["cls_src"] = "", "", "unknown"
                n_unk += 1
        else:
            cid = int(c["class_id"])
            if r.get("cls_src") == "manual" and str(r.get("class_id")) == str(cid):
                n_same += 1
            else:
                r["class_id"], r["cls_src"] = cid, "manual"
                n_set += 1

    print(f"标注文件: {args.corrections}（{len(corr.get('corrections', []))} 条）")
    print(f"将应用: 指派 {n_set} 条 / 标未知 {n_unk} 条 / 无变化 {n_same} 条 / 无效 {n_missing} 条")
    if not args.inplace:
        print("\n预览模式：没有写文件。加 --inplace 落地。")
        return
    if n_set == 0 and n_unk == 0:
        print("\n没有任何改动，不写文件。")
        return
    with open(index_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写回 {index_path}")


if __name__ == "__main__":
    main()
