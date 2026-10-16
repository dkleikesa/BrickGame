#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三臂的 metrics.json 合成一张对照表（现模型 vs legacy 闭集 softmax），并给出结论提示。

读的是各臂自己写下的 metrics.json，字段名不同（E 用 train_classifier 的 "real"，
L1/L2 用 train_legacy_softmax 的 "softmax"），这里只做"同一把尺子"的归一化：
两边都报 top1 / top5 / top1_macro / seen / unseen；L 侧额外有闭集口径。
三臂的 val 划分必须同 seed 同参数，否则表里会打上 ⚠ 不一致的警告。

用法:
  python compare_legacy.py --out out/compare_legacy
"""
from __future__ import annotations

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_variant(path: str, key: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    return {"config": d.get("config", {}), "metrics": d.get(key, {})}


def row(name: str, m: dict) -> dict:
    if not m:
        return {"臂": name, "top1": "", "top5": "", "top1_macro": "",
                "seen": "", "unseen": "", "闭集top1": "", "闭集top5": "", "n_query": ""}
    closed = "top1_closed" in m
    return {
        "臂": name,
        "top1": round(m.get("top1", 0), 4),
        "top5": round(m.get("top5", 0), 4),
        "top1_macro": round(m.get("top1_macro", 0), 4),
        "seen": round(m.get("top1_seen_cls", 0), 4),
        "unseen": round(m.get("top1_unseen_cls", 0), 4),
        "闭集top1": round(m.get("top1_closed", 0), 4) if closed else "—",
        "闭集top5": round(m.get("top5_closed", 0), 4) if closed else "—",
        "n_query": m.get("n_query", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="现模型 vs legacy 闭集 softmax 的对照表")
    ap.add_argument("--e", default=os.path.join(_HERE, "out/clf_real/metrics.json"),
                    help="现模型（train_classifier.py）的 metrics.json")
    ap.add_argument("--l1", default=os.path.join(_HERE, "out/legacy_softmax/L1/metrics.json"))
    ap.add_argument("--l2", default=os.path.join(_HERE, "out/legacy_softmax/L2/metrics.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "out/compare_legacy"))
    args = ap.parse_args()

    e = load_variant(args.e, "real")
    l1 = load_variant(args.l1, "softmax")
    l2 = load_variant(args.l2, "softmax")

    # 划分一致性检查：seed / val_cls_ratio 必须相同，否则对比无意义
    warn = []
    cfgs = [(n, v["config"]) for n, v in (("E", e), ("L1", l1), ("L2", l2)) if v]
    for k in ("seed", "val_cls_ratio"):
        vals = {str(c.get(k)) for _, c in cfgs if k in c}
        if len(vals) > 1:
            warn.append(f"{k} 不一致：{vals}")

    rows = [row("E 现模型（SupCon+原型检索）", e["metrics"] if e else {}),
            row("L1 legacy（冻结骨干+2369路softmax）", l1["metrics"] if l1 else {}),
            row("L2 softmax（微调骨干+增广）", l2["metrics"] if l2 else {})]

    os.makedirs(args.out, exist_ok=True)
    cols = list(rows[0].keys())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    n_class = (e or l1 or l2)["metrics"].get("n_class", 2369) if (e or l1 or l2) else 2369
    md = [
        "# 现模型 vs legacy 闭集 softmax（同一批留出格，同一 seed）",
        "",
        f"- 随机基线（{n_class} 类）：{1 / n_class:.4f}",
        f"- 警告：{'；'.join(warn) if warn else '无（三臂划分一致）'}",
        "",
        *lines,
        "",
        "> `闭集` = 候选类限制在验证集里真实出现过的类；`—` 表示该臂没有这一列（现模型的检索天然是开集）。",
    ]
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n输出: {args.out}/summary.md / summary.csv")


if __name__ == "__main__":
    main()
