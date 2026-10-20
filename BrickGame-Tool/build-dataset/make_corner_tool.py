#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成棋盘四角标注页：corner_tool.html（单文件，双击用浏览器打开即可，不联网）。

为什么要人工标四角：仓库里没有任何独立于算法的定位真值 —— frames.csv 的 rows/cols
是 app 自己的输出，拿它评估就是自证。四角 + 行列数才是最省事的真值（一帧点 4 下）。

页面里每张图都**已经叠了算法网格（绿线）**，所以多数帧只需点「网格正确」；
只有网格画错的帧才点「标四角」，按 左上 → 右上 → 右下 → 左下 的顺序点。
导出的 CSV 直接喂给 eval_truth.py。

用法：
  uv run --no-project --with numpy --with opencv-python-headless python make_corner_tool.py
  uv run ... python make_corner_tool.py --samples samples-fail --out corner_tool_fail.html
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from tile_recognizer import INSET, detect_grid  # noqa: E402

TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>棋盘网格标注</title>
<style>
 body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f5f5f7;color:#111}
 header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #ddd;
        padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 header b{font-size:15px}
 button{font:inherit;padding:5px 12px;border:1px solid #c9c9cf;border-radius:7px;background:#fff;cursor:pointer}
 button:hover{background:#f0f0f4}
 button.primary{background:#0a84ff;border-color:#0a84ff;color:#fff}
 button.primary:hover{background:#0070e0}
 button.on{background:#34c759;border-color:#34c759;color:#fff}
 .hint{color:#666}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;padding:14px}
 .card{background:#fff;border:1px solid #e2e2e6;border-radius:10px;padding:8px;overflow:hidden}
 .hd{display:flex;justify-content:space-between;font-size:12px;color:#555;margin-bottom:6px}
 .tag{background:#eee;border-radius:5px;padding:1px 6px}
 canvas{display:block;width:100%;height:auto;background:#000;border-radius:6px;cursor:crosshair}
 .row{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
 .msg{font-size:12px;color:#888;margin-left:auto}
 .done{outline:2px solid #34c759}
 .bad{outline:2px solid #ff3b30}
</style></head><body>
<header>
  <b>棋盘网格标注</b>
  <span class="hint">绿线 = 算法网格。对就点「网格正确」；错就点「标四角」，按 左上→右上→右下→左下 点 4 下。</span>
  <span id="prog" class="hint"></span>
  <button class="primary" id="export">导出 CSV</button>
  <button id="clear">清空全部</button>
</header>
<div id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = 'brick-corner-tool-v1';
const state = JSON.parse(localStorage.getItem(KEY) || '{}');
const cache = {};

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); progress(); }
function progress(){
  const n = DATA.filter(d => (state[d.frame]||{}).verdict).length;
  document.getElementById('prog').textContent = '已标 ' + n + ' / ' + DATA.length;
}
function ensure(f){ if(!state[f]) state[f] = {verdict:null, pts:[]}; return state[f]; }

function draw(d, cv){
  const ctx = cv.getContext('2d');
  const img = cache[d.frame];
  if(!img){ return; }
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  const st = ensure(d.frame);
  const s = d.scale;
  if(st.pts.length){
    ctx.strokeStyle = '#ff3b30'; ctx.lineWidth = 2;
    ctx.beginPath();
    st.pts.forEach((p,i)=>{ const x=p[0]*s, y=p[1]*s; i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    if(st.pts.length===4) ctx.closePath();
    ctx.stroke();
    st.pts.forEach((p,i)=>{
      const x=p[0]*s, y=p[1]*s;
      ctx.fillStyle='#ff3b30'; ctx.beginPath(); ctx.arc(x,y,5,0,7); ctx.fill();
      ctx.fillStyle='#fff'; ctx.font='bold 11px sans-serif'; ctx.fillText(String(i+1), x+7, y-6);
    });
  }
}

function refreshCard(d){
  const card = document.getElementById('c-' + d.frame);
  const cv = card.querySelector('canvas');
  const st = ensure(d.frame);
  card.classList.toggle('done', st.verdict === 'ok');
  card.classList.toggle('bad', st.verdict === 'quad');
  card.querySelector('.msg').textContent =
    st.verdict === 'ok' ? '已确认正确' :
    st.verdict === 'quad' ? (st.pts.length === 4 ? '四角已标' : '已点 ' + st.pts.length + ' / 4') : '';
  draw(d, cv);
}

function build(){
  const host = document.getElementById('grid');
  DATA.forEach(d=>{
    const card = document.createElement('div');
    card.className = 'card'; card.id = 'c-' + d.frame;
    const w = Math.round(d.w * d.scale), h = Math.round(d.h * d.scale);
    card.innerHTML =
      '<div class="hd"><span>' + d.frame + '</span><span class="tag">' +
      (d.rows ? d.rows + '×' + d.cols + ' pitch ' + d.pitch : '定位失败') +
      '</span></div>' +
      '<canvas width="' + w + '" height="' + h + '"></canvas>' +
      '<div class="row">' +
      '<button data-act="ok">网格正确</button>' +
      '<button data-act="quad">标四角</button>' +
      '<button data-act="reset">清除</button>' +
      '<span class="msg"></span></div>';
    host.appendChild(card);

    const img = new Image();
    img.onload = ()=>{ cache[d.frame] = img; draw(d, card.querySelector('canvas')); };
    img.src = 'data:image/jpeg;base64,' + d.img;

    const cv = card.querySelector('canvas');
    cv.addEventListener('click', e=>{
      const st = ensure(d.frame);
      if(st.verdict !== 'quad' || st.pts.length >= 4) return;
      const rect = cv.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (cv.width / rect.width) / d.scale;
      const y = (e.clientY - rect.top) * (cv.height / rect.height) / d.scale;
      st.pts.push([Math.round(x), Math.round(y)]);
      save(); refreshCard(d);
    });
  });
  document.addEventListener('click', e=>{
    const btn = e.target.closest('button[data-act]');
    if(!btn) return;
    const frame = btn.closest('.card').id.slice(2);
    const d = DATA.find(x => x.frame === frame);
    const st = ensure(frame);
    if(btn.dataset.act === 'ok'){ st.verdict = 'ok'; st.pts = []; }
    if(btn.dataset.act === 'quad'){ st.verdict = 'quad'; st.pts = []; }
    if(btn.dataset.act === 'reset'){ st.verdict = null; st.pts = []; }
    save(); refreshCard(d);
  });
  progress();
}

document.getElementById('export').onclick = ()=>{
  const rows = ['frame,verdict,x0,y0,x1,y1,x2,y2,x3,y3'];
  DATA.forEach(d=>{
    const st = state[d.frame] || {};
    let pts = null;
    if(st.verdict === 'ok') pts = d.quad;
    if(st.verdict === 'quad' && st.pts.length === 4) pts = st.pts;
    rows.push([d.frame, st.verdict || '', ...(pts && pts.length === 8 ? pts : Array(8).fill(''))].join(','));
  });
  const blob = new Blob([rows.join('\n')], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'ground_truth.csv';
  a.click();
};

document.getElementById('clear').onclick = ()=>{
  if(!confirm('清空全部标注？')) return;
  Object.keys(state).forEach(k => delete state[k]);
  save(); DATA.forEach(refreshCard);
};

build();
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(HERE, "samples"))
    ap.add_argument("--out", default=os.path.join(HERE, "corner_tool.html"))
    ap.add_argument("--width", type=int, default=460, help="页面里每张图的显示宽度")
    args = ap.parse_args()

    names = sorted(
        f for f in os.listdir(args.samples)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    )
    data = []
    for name in names:
        frame = os.path.splitext(name)[0]
        img = cv2.imdecode(np.fromfile(os.path.join(args.samples, name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] 读不到 {name}")
            continue
        h, w = img.shape[:2]
        rows = cols = 0
        quad: list[float] = []
        vis = img.copy()
        try:
            g = detect_grid(img, 12, INSET, verbose=False)
            rows, cols = int(g["score"].shape[0]), int(g["score"].shape[1])
            xl, yl = np.asarray(g["xlines"]), np.asarray(g["ylines"])
            y0, y1 = int(yl[0]), int(yl[-1])
            for x in xl:
                cv2.line(vis, (int(x), y0), (int(x), y1), (0, 255, 0), 2)
            x0, x1 = int(xl[0]), int(xl[-1])
            for y in yl:
                cv2.line(vis, (x0, int(y)), (x1, int(y)), (0, 255, 0), 2)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 3)
            quad = [float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1)]
        except (SystemExit, Exception):
            pass

        scale = args.width / w
        small = cv2.resize(vis, (args.width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            print(f"[skip] 编码失败 {name}")
            continue
        data.append({
            "frame": frame, "w": w, "h": h, "scale": scale,
            "rows": rows, "cols": cols,
            "pitch": int(g["pitch_x"]) if rows else 0,
            "quad": quad,
            "img": base64.b64encode(buf.tobytes()).decode(),
        })

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"生成 {args.out}（{len(data)} 帧，{size_mb:.1f} MB）")
    print("双击打开 → 逐帧确认 → 导出 CSV → 存成 build-dataset/ground_truth.csv")


if __name__ == "__main__":
    main()
