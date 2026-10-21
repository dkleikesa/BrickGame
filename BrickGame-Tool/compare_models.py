#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两个身份模型的同图对比：同一批截图 → 逐格预测 → 可视化 + 指标。

为什么不用 train_classifier.py 的评测：那边的指标是"留出帧的真实格"，尺子是数据集自己的
标注。这里要比的是**上机后的实际表现**（低置信度留档的那几张），而且必须和 Android 端
逐像素对齐，否则比的是两个不同的预处理。

预处理严格照抄 Android（`CellCropper.fill`）：
  格心为中心裁 round(格距中位数 × (1 - 2×0.02)) 的正方形
  → 缩到 64（缩小用 INTER_AREA、放大用 INTER_CUBIC）→ 缩到模型输入边长（INTER_LINEAR）
  → BGR→RGB、/255（ImageNet mean/std 已在权重里，Python 侧由 to_tensor 施加，等价）
  相似度 = 余弦相似度 × 100 取整（与 `TfliteTileClassifier` 的 similarity 同式）。
所以 `--ref` 传入 App 落盘的 `lowconf_*.txt` 时，模型 A（线上那个）应当能逐格对上 —— 对不上
就说明这条 Python 管线没复刻对，后面的对比也就不用看了。

产出（`--out` 目录）：
  <stem>_compare.png   三个面板：模型A 标注 / 模型B 标注 / 差异高亮
  <stem>_diff.txt      逐格表格 + 不一致清单（含物品名，便于人眼判谁对）
  summary.csv          每张图的汇总（一致率、低置信格数、平均相似度、与 ref 的对齐率）
用法：
  .venv/bin/python compare_models.py \
      --a out/clf_v2/best.pt --b <旧模型目录>/best.pt \
      --images ~/Downloads/brickgame-lowconf --out out/compare
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import tile_recognizer as T  # noqa: E402
import train_classifier as TC  # noqa: E402

OUT_INSET = 0.02      # 与 CellCropper.OUT_INSET 一致
CROP = 64             # 与 CellCropper.CROP 一致
LOW_SIM = 85          # 与 TfliteTileClassifier.DUMP_SIMILARITY 一致
TIE = 0.05            # 像素相关性差多少才算"谁更像"，以下算平票


class Model:
    """一个已训练模型 + 它的原型库，接口与 Android 端一致（预测 = 原型最近邻）。"""

    def __init__(self, ckpt: str, proto: str, dev: str, tag: str):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.tag = tag
        self.backbone = ck["backbone"]
        self.img = int(ck["img"])
        self.dim = int(ck["embed_dim"])
        self.model = TC.make_model(self.backbone, self.dim, int(ck.get("width", 32)))
        self.model.load_state_dict(ck["model"])
        self.model.to(dev).eval()
        self.dev = dev
        z = np.load(proto)
        self.proto = z["proto"].astype(np.float32)
        self.class_ids = [int(c) for c in z["class_ids"]]
        self.entries = [str(e) for e in z["entries"]]
        self.n_param = sum(p.numel() for p in self.model.parameters())
        self.epoch = ck.get("epoch")

    @torch.no_grad()
    def predict(self, patches_bgr: np.ndarray, bs: int = 128, topk: int = 3
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N,H,W,3) uint8 BGR，已按 Android 预处理成模型输入尺寸。

        @return (类下标(N,), 相似度百分制(N,), top-k 类下标(N,k))
        返回的是**原型行下标**，不是 class_id —— 两者靠 class_ids 映射（与 Android 相同）。
        """
        best, sim, tops = [], [], []
        for i in range(0, len(patches_bgr), bs):
            x = TC.to_tensor(patches_bgr[i:i + bs]).to(self.dev)
            e, _ = self.model(x)
            e = e.float().cpu().numpy()
            score = e @ self.proto.T                       # 两边都 L2 归一化 → 点积即余弦
            order = np.argsort(-score, axis=1)[:, :topk]
            best.append(order[:, 0])
            tops.append(order)
            sim.append(np.round(score[np.arange(len(score)), order[:, 0]] * 100).astype(np.int32))
        return (np.concatenate(best), np.concatenate(sim), np.concatenate(tops))

    def name_of(self, proto_idx: int) -> str:
        cid = self.class_ids[proto_idx]
        return f"{cid}:{self.entries[proto_idx]}" if proto_idx < len(self.entries) else str(cid)


def detect(img: np.ndarray) -> dict:
    if os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-dataset") not in sys.path:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-dataset"))
    import yolo_grid
    return yolo_grid.locate(img, T.INSET)


def android_crops(img: np.ndarray, g: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """按 CellCropper 的裁切规格取每格（返回 64x64 的中间图；模型输入边长后面再缩）。"""
    xs, ys = g["xlines"], g["ylines"]
    pitch = float(np.median(np.concatenate([np.diff(xs), np.diff(ys)])))
    side = max(8, int(round(pitch * (1 - 2 * OUT_INSET))))
    h, w = img.shape[:2]
    crops = []
    for j in range(g["rows"]):
        cy = int(round((ys[j] + ys[j + 1]) / 2))
        for i in range(g["cols"]):
            cx = int(round((xs[i] + xs[i + 1]) / 2))
            a = min(max(0, int(round(cx - side / 2))), w - 2)
            c = min(max(0, int(round(cy - side / 2))), h - 2)
            b, d = min(w, a + side), min(h, c + side)
            cell = img[c:d, a:b]
            interp = cv2.INTER_AREA if cell.shape[0] > CROP else cv2.INTER_CUBIC
            crops.append(cv2.resize(cell, (CROP, CROP), interpolation=interp))
    return crops, (g["score"] < g["score_thr"]).ravel()


def to_input(crops: list[np.ndarray], img_size: int) -> np.ndarray:
    return np.stack([cv2.resize(c, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                     for c in crops])


def load_catalog(catalog: str) -> dict[int, str]:
    """class_id → 图鉴整格图路径（像素基线的参考库）。"""
    out = {}
    for r in csv.DictReader(open(os.path.join(catalog, "index.csv"))):
        out.setdefault(int(r["class_id"]), os.path.join(catalog, r["tile"]))
    return out


def pixel_match(patch32: np.ndarray, tile_path: str, rad: int = 3) -> float:
    """去底图标特征 + ±3px 平移搜索的相关性（与两个模型都无关的第三种意见）。

    模型 A/B 给出两个不同答案时，谁的图鉴图在像素上更像这一格，谁就更可能对。
    像素特征不是可靠的身份识别器（`auto_classify.py` 的文件头说了为什么：色系/轮廓相近的
    物品会互相抢），所以只当**平票时的仲裁**，不做独立判定。

    用 TM_CCOEFF_NORMED 而不是 SQDIFF_NORMED：去底后的特征是"大片 0 + 稀疏图标像素"，
    SQDIFF 的归一化在这种稀疏图上会饱和到 1.0（实测同类与异类都返回 1.0，完全没有分辨力），
    零均值归一化相关才对得上边。
    @return 相关性 [-1,1]，越大越像；-9 表示这一格去底后什么都没剩（纯底板/全被遮挡），无效。
    """
    tile = cv2.imread(tile_path, cv2.IMREAD_COLOR) if tile_path else None
    if tile is None:
        return -9.0
    tile = cv2.resize(tile, (T.CELL, T.CELL), interpolation=cv2.INTER_AREA).astype(np.float32)
    q = T.icon_features(patch32[None].astype(np.float32))[0][0]
    r = T.icon_features(tile[None])[0][0]
    if q.max() == 0 or r.max() == 0:
        return -9.0
    pad = cv2.copyMakeBorder(q, rad, rad, rad, rad, cv2.BORDER_REPLICATE)
    return float(cv2.matchTemplate(pad, r, cv2.TM_CCOEFF_NORMED).max())


def parse_ref(txt_path: str) -> dict | None:
    """解析 App 落盘的 lowconf_*.txt：矩阵用 class_id 表示。"""
    if not os.path.exists(txt_path):
        return None
    labels, head = [], []
    for line in open(txt_path):
        line = line.rstrip("\n")
        if line.startswith("#"):
            head.append(line)
            continue
        toks = line.split()
        if toks and all(t.lstrip("-").isdigit() for t in toks):
            labels.append([int(t) for t in toks])
    return {"labels": np.array(labels, np.int32) if labels else None, "head": head}


def put_label(vis: np.ndarray, x0: int, y0: int, x1: int, y1: int, text: str, color, scale=0.42):
    """格子左上角写字，先描黑边再写白字（浅色底板上也看得清）。"""
    org = (x0 + 3, y0 + 14)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel(img: np.ndarray, g: dict, labels: np.ndarray, sims: np.ndarray,
               title: str, low: np.ndarray) -> np.ndarray:
    vis = img.copy()
    xs, ys = g["xlines"], g["ylines"]
    for x in xs:
        cv2.line(vis, (int(x), int(ys[0])), (int(x), int(ys[-1])), (0, 0, 255), 1)
    for y in ys:
        cv2.line(vis, (int(xs[0]), int(y)), (int(xs[-1]), int(y)), (0, 0, 255), 1)
    k = 0
    for j in range(g["rows"]):
        for i in range(g["cols"]):
            x0, y0, x1, y1 = int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1])
            if low[k]:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (150, 150, 150), -1)
                put_label(vis, x0, y0, x1, y1, "空", (255, 255, 255), 0.6)
            else:
                put_label(vis, x0, y0, x1, y1, str(labels[k]), (255, 255, 255), 0.62)
                put_label(vis, x0, y0 + 20, x1, y1, str(sims[k]), (0, 255, 255), 0.5)
            k += 1
    return add_title(vis, title)


def draw_diff(img: np.ndarray, g: dict, la: np.ndarray, sa: np.ndarray,
              lb: np.ndarray, sb: np.ndarray, low: np.ndarray, verdict: list[str]
              ) -> np.ndarray:
    """只画差异：一致=绿框，不一致=红框并写出两边的答案，任一边低置信=橙框。

    不一致格的框色再按像素基线的仲裁上色：蓝=L 更像、紫=S 更像、红=平票。
    """
    vis = img.copy()
    xs, ys = g["xlines"], g["ylines"]
    k = 0
    for j in range(g["rows"]):
        for i in range(g["cols"]):
            x0, y0, x1, y1 = int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1])
            if low[k]:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (150, 150, 150), -1)
            else:
                same = la[k] == lb[k]
                poor = sa[k] < LOW_SIM or sb[k] < LOW_SIM
                color = (0, 200, 0) if same else (0, 0, 255)
                if poor and same:
                    color = (0, 165, 255)
                elif not same:
                    color = {"A": (255, 128, 0), "B": (255, 0, 200), "-": (0, 0, 255)}[verdict[k]]
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3 if not same else 2)
                if not same:
                    put_label(vis, x0, y0, x1, y1, f"L {la[k]}", (0, 0, 255), 0.5)
                    put_label(vis, x0, y0 + 18, x1, y1, f"S {lb[k]}", (255, 0, 0), 0.5)
                    put_label(vis, x0, y0 + 36, x1, y1, f"{sa[k]}/{sb[k]}"
                              + (f"={verdict[k]}" if verdict[k] != "-" else ""), (0, 255, 255), 0.42)
            k += 1
    return add_title(vis, "差异（绿=一致 橙=一致但低置信 灰=空位；不一致框色 蓝=像素更像L 紫=更像S 红=平票）")


def add_title(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((30, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def run_image(path: str, ma: "Model", mb: "Model", out_dir: str, ref_dir: str,
              catalog: dict[int, str]) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    g = detect(img)
    crops, low = android_crops(img, g)
    ia, sa, ta = ma.predict(to_input(crops, ma.img))
    ib, sb, tb = mb.predict(to_input(crops, mb.img))
    la = np.array([ma.class_ids[i] for i in ia], np.int32)
    lb = np.array([mb.class_ids[i] for i in ib], np.int32)
    na = [ma.name_of(int(i)) for i in ia]
    nb = [mb.name_of(int(i)) for i in ib]

    ref = parse_ref(os.path.join(ref_dir, stem + ".txt")) if ref_dir else None
    ref_lab = None
    if ref and ref["labels"] is not None and ref["labels"].size == la.size:
        ref_lab = ref["labels"].ravel()

    # ---- 不一致格的第三种意见（像素基线，与两个模型都无关）----
    verdict = ["-"] * len(la)
    pix = {}
    for k in np.where(~low & (la != lb))[0].tolist():
        ra = pixel_match(g["patches"][k], catalog.get(int(la[k]), ""))
        rb = pixel_match(g["patches"][k], catalog.get(int(lb[k]), ""))
        pix[k] = (ra, rb)
        invalid = ra < -1 or rb < -1
        verdict[k] = "-" if (invalid or abs(ra - rb) < TIE) else ("A" if ra > rb else "B")

    keep = ~low
    agree = float((la[keep] == lb[keep]).mean()) if keep.any() else 0.0
    row_ref = {}
    if ref_lab is not None:
        row_ref = {"ref_with_a": float((la[keep] == ref_lab[keep]).mean()),
                   "ref_with_b": float((lb[keep] == ref_lab[keep]).mean())}

    # ---- 可视化 ----
    cv2.imwrite(os.path.join(out_dir, f"{stem}_compare.png"),
                np.vstack([draw_panel(img, g, la, sa, f"A {ma.backbone} 逐格标注（类id + 相似度）", low),
                           draw_panel(img, g, lb, sb, f"B {mb.backbone} 逐格标注（类id + 相似度）", low),
                           draw_diff(img, g, la, sa, lb, sb, low, verdict)]))

    # ---- 逐格表格 ----
    dis = np.where(~low & (la != lb))[0]
    n_a = sum(1 for k in dis.tolist() if verdict[k] == "A")
    n_b = sum(1 for k in dis.tolist() if verdict[k] == "B")
    with open(os.path.join(out_dir, f"{stem}_diff.txt"), "w") as f:
        f.write(f"{stem}  网格 {g['rows']}x{g['cols']} 空位 {int(low.sum())}/{low.size}"
                f"  A={ma.backbone}  B={mb.backbone}\n")
        if row_ref:
            f.write(f"A 与线上留档一致率 {row_ref['ref_with_a']:.3f} / B 与线上留档一致率 {row_ref['ref_with_b']:.3f}\n")
        f.write(f"非空格一致率 {agree:.3f}；不一致 {dis.size} 格"
                f"（像素基线仲裁：更像 A 的 {n_a} 格 / 更像 B 的 {n_b} 格 / 平票 {dis.size - n_a - n_b} 格）\n")
        f.write("\n# 不一致格（row,col  A类id:物品 A相似 / B类id:物品 B相似  像素距离  谁更像）\n")
        for k in dis.tolist():
            r, c = divmod(int(k), g["cols"])
            ra, rb = pix[k]
            f.write(f"({r:2d},{c:2d})  A {na[k]} {sa[k]}  /  B {nb[k]} {sb[k]}"
                    f"   像素相关 {ra:.3f} vs {rb:.3f} → {verdict[k]}\n")
            f.write(f"          A候选: {' '.join(ma.name_of(int(j)) for j in ta[k])}\n")
            f.write(f"          B候选: {' '.join(mb.name_of(int(j)) for j in tb[k])}\n")
        f.write("\n# 逐格矩阵（A / B 两层，与 lowconf txt 同格式，行优先）\n")
        for r in range(g["rows"]):
            f.write(" ".join(str(int(la[r * g["cols"] + c])) for c in range(g["cols"])) + "\n")
        f.write("--\n")
        for r in range(g["rows"]):
            f.write(" ".join(str(int(lb[r * g["cols"] + c])) for c in range(g["cols"])) + "\n")

    return {"image": os.path.basename(path), "cells": int(g["cols"] * g["rows"]),
            "empty": int(low.sum()), "grid": f"{g['rows']}x{g['cols']}",
            "agree": agree, "n_disagree": int(dis.size),
            "pix_a": n_a, "pix_b": n_b, "pix_tie": int(dis.size) - n_a - n_b,
            "a_low_conf": int((keep & (sa < LOW_SIM)).sum()),
            "b_low_conf": int((keep & (sb < LOW_SIM)).sum()),
            "a_sim_mean": float(sa[keep].mean()), "b_sim_mean": float(sb[keep].mean()),
            **row_ref}


def main() -> None:
    ap = argparse.ArgumentParser(description="两个模型的同图逐格对比")
    ap.add_argument("--a", default=os.path.join(_HERE, "out/clf_v2/best.pt"), help="模型 A（线上 V2）")
    ap.add_argument("--b", required=True, help="模型 B（对照，必填）")
    ap.add_argument("--proto-a", default="", help="默认取 --a 同目录 prototypes.npz")
    ap.add_argument("--proto-b", default="", help="默认取 --b 同目录 prototypes.npz")
    ap.add_argument("--images", default=os.path.expanduser("~/Downloads/brickgame-lowconf"),
                    help="截图目录（递归取 png/jpg）")
    ap.add_argument("--ref", default="",
                    help="App 落盘 lowconf_*.txt 所在目录（默认与 --images 同目录；对不上就不写这一项）")
    ap.add_argument("--out", default=os.path.join(_HERE, "out/compare"))
    ap.add_argument("--catalog", default=os.path.join(_HERE, "build-catalog", "catalog"),
                    help="图鉴目录：像素基线仲裁用它的整格图")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out, exist_ok=True)

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.images)
                   for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise SystemExit(f"{args.images} 里没有图片")
    ma = Model(args.a, args.proto_a or os.path.join(os.path.dirname(args.a), "prototypes.npz"), dev, "A")
    mb = Model(args.b, args.proto_b or os.path.join(os.path.dirname(args.b), "prototypes.npz"), dev, "B")
    for m in (ma, mb):
        print(f"模型 {m.tag}: {m.backbone} 输入 {m.img} 维度 {m.dim} 原型 {len(m.class_ids)} 类 "
              f"参数 {m.n_param / 1e6:.2f}M（epoch {m.epoch}）")

    catalog = load_catalog(args.catalog)
    rows = [run_image(p, ma, mb, args.out, args.ref or args.images, catalog) for p in files]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = ("agree", "pix_a", "pix_b", "pix_tie", "a_low_conf", "b_low_conf",
            "a_sim_mean", "b_sim_mean")
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"A": {"ckpt": args.a, "backbone": ma.backbone, "params": ma.n_param},
                   "B": {"ckpt": args.b, "backbone": mb.backbone, "params": mb.n_param},
                   "images": rows, "agg": agg}, f, indent=2, ensure_ascii=False)

    print(f"\n{'图片':<26}{'非空格':>7}{'一致率':>9}{'不一致':>7}{'像素似A':>8}{'像素似B':>8}{'平票':>6}"
          f"{'A低置信':>8}{'B低置信':>8}{'A均相似':>9}{'B均相似':>9}")
    for r in rows:
        print(f"{r['image']:<26}{r['cells'] - r['empty']:>7}{r['agree']:>9.3f}{r['n_disagree']:>7}"
              f"{r['pix_a']:>8}{r['pix_b']:>7}{r['pix_tie']:>6}"
              f"{r['a_low_conf']:>8}{r['b_low_conf']:>8}{r['a_sim_mean']:>9.1f}{r['b_sim_mean']:>9.1f}")
    print(f"\n合计: 非空格一致率 {agg['agree']:.3f}；不一致格里像素基线判 A {agg['pix_a']:.1f} / "
          f"B {agg['pix_b']:.1f} / 平票 {agg['pix_tie']:.1f}（格/张）")
    print(f"     低置信格 A {agg['a_low_conf']:.1f} vs B {agg['b_low_conf']:.1f} 格/张；"
          f"平均相似度 A {agg['a_sim_mean']:.1f} vs B {agg['b_sim_mean']:.1f}")
    print(f"输出: {args.out}/summary.csv / summary.json / <图片>_compare.png / <图片>_diff.txt")


if __name__ == "__main__":
    main()
