#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 YOLO 框标注页（box_tool.html），只放"模型预标注"的那批帧。

为什么要人工过一遍：预标注是当前模型自己出的框，漏检的砖在训练里等于"这里没有砖"，
错误会被固化。逐格点 140 个框不现实，所以这里只做两件事：删掉误检、补上漏检。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/make_box_tool.py
  打开生成的 box_tool.html → 改完点「导出」→ 得到 box_fixes.json
  再跑 apply_box_fixes.py 写回 labels
"""
from __future__ import annotations

import base64
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO = os.path.join(HERE, "yolo")
OUT = os.path.join(HERE, "box_tool.html")
WIDTH = 460
TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>砖块框标注</title>
<style>
 body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f5f5f7;color:#111}
 header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #ddd;
        padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 header b{font-size:15px}
 button{font:inherit;padding:5px 12px;border:1px solid #c9c9cf;border-radius:7px;background:#fff;cursor:pointer}
 button:hover{background:#f0f0f4}
 button.primary{background:#0a84ff;border-color:#0a84ff;color:#fff}
 .hint{color:#666}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;padding:14px}
 .card{background:#fff;border:1px solid #e2e2e6;border-radius:10px;padding:8px}
 .hd{display:flex;justify-content:space-between;font-size:12px;color:#555;margin-bottom:6px}
 canvas{display:block;width:100%;height:auto;background:#000;border-radius:6px;cursor:crosshair}
 .row{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
 .msg{font-size:12px;color:#888;margin-left:auto}
</style></head><body>
<header>
  <b>砖块框标注</b>
  <span class="hint">蓝色 = 砖块（点一下 = 删掉它）；白色 = 棋盘，一般不用动；拖空白 = 补漏检</span>
  <span id="prog" class="hint"></span>
  <button class="primary" id="export">保存到训练集</button>
  <button id="resetall">清空修改</button>
</header>
<div id="grid"></div>
<script>
let DATA = [];
async function boot(){
  try {
    const r = await fetch('/data');
    DATA = await r.json();
  } catch (e) {
    document.getElementById('grid').innerHTML =
      '<p style="padding:20px;color:#c00">请通过本地服务打开：先运行 box_server.py，再访问 http://127.0.0.1:8771/</p>';
    return;
  }
  build();
}
const state = JSON.parse(localStorage.getItem('brick-box-tool') || '{}');
const cache = {};

function st(f){ if(!state[f]) state[f] = {deleted:[], added:[]}; return state[f]; }
function save(){ localStorage.setItem('brick-box-tool', JSON.stringify(state)); progress(); }
function progress(){
  const n = DATA.filter(d => { const s=state[d.file]; return s && (s.deleted.length || s.added.length); }).length;
  document.getElementById('prog').textContent = '已修改 ' + n + ' / ' + DATA.length;
}
function draw(d, cv, draft){
  const ctx = cv.getContext('2d');
  const img = cache[d.file]; if(!img) return;
  const s = d.scale, W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H); ctx.drawImage(img,0,0,W,H);
  const cur = st(d.file);
  // 原始框已经画在图片里了，这里只覆盖"被删掉的"（半透明遮住）和"新补的"（黄色）
  d.boxes.forEach((b,i)=>{
    if (!cur.deleted.includes(i)) return;
    const [cls,cx,cy,bw,bh] = b;
    const x=(cx-bw/2)*W, y=(cy-bh/2)*H, w=bw*W, h=bh*H;
    ctx.fillStyle = 'rgba(240,240,240,.75)';
    ctx.fillRect(x,y,w,h);
    ctx.setLineDash([4,4]); ctx.lineWidth = 1.5; ctx.strokeStyle = '#999';
    ctx.strokeRect(x,y,w,h); ctx.setLineDash([]);
  });
  ctx.setLineDash([]);
  cur.added.forEach(a=>{
    ctx.lineWidth = 2; ctx.strokeStyle = '#ffcc00';
    ctx.strokeRect(a[0]*W, a[1]*H, (a[2]-a[0])*W, (a[3]-a[1])*H);
  });
  // 砖很少时给一句醒目提示：这类帧本来就是"要人判断是空盘还是漏检"的
  const nTile = d.boxes.filter(b=>b[0]===1).length - 0;
  const liveTile = nTile - cur.deleted.filter(i=>d.boxes[i][0]===1).length + cur.added.length;
  if (liveTile < 3){
    ctx.fillStyle='rgba(0,0,0,.55)';
    ctx.fillRect(0, 0, W, 46);
    ctx.fillStyle='#ffcc00'; ctx.font='bold 20px sans-serif'; ctx.textAlign='center';
    ctx.fillText('只有 '+liveTile+' 块砖 —— 是空盘，还是模型漏检？', W/2, 30);
    ctx.textAlign='left';
  }
  if (draft){
    ctx.lineWidth = 2; ctx.strokeStyle = '#ffcc00'; ctx.setLineDash([6,4]);
    ctx.strokeRect(draft[0], draft[1], draft[2]-draft[0], draft[3]-draft[1]);
  }
}
function hit(d, px, py, W, H){
  const cur = st(d.file);
  for (let i=d.boxes.length-1; i>=0; i--){
    const [cls,cx,cy,bw,bh] = d.boxes[i];
    const x=(cx-bw/2)*W, y=(cy-bh/2)*H, w=bw*W, h=bh*H;
    if (px>=x && px<=x+w && py>=y && py<=y+h) return i;
  }
  return -1;
}
function hitAdded(d, px, py, W, H){
  const cur = st(d.file);
  for (let i=cur.added.length-1; i>=0; i--){
    const a = cur.added[i];
    if (px>=a[0]*W && px<=a[2]*W && py>=a[1]*H && py<=a[3]*H) return i;
  }
  return -1;
}
function build(){
  const host = document.getElementById('grid');
  DATA.forEach(d=>{
    const card = document.createElement('div');
    card.className='card'; card.id='c-'+d.file;
    const w = Math.round(d.w*d.scale), h = Math.round(d.h*d.scale);
    card.innerHTML =
      '<div class="hd"><span>'+d.src+'</span><span>砖 '+d.boxes.filter(b=>b[0]===1).length+' 块</span></div>'+
      '<canvas width="'+w+'" height="'+h+'"></canvas>'+
      '<div class="row"><span class="msg"></span></div>';
    host.appendChild(card);
    const cv = card.querySelector('canvas');
    const img = new Image();
    img.onload = ()=>{ cache[d.file]=img; draw(d,cv,null); };
    img.src = 'data:image/jpeg;base64,'+d.img;

    let down=null, draft=null, moved=false;
    cv.addEventListener('mousedown', e=>{
      const r = cv.getBoundingClientRect();
      down = [(e.clientX-r.left)*(cv.width/r.width), (e.clientY-r.top)*(cv.height/r.height)];
      moved=false; draft=null;
    });
    cv.addEventListener('mousemove', e=>{
      if(!down) return;
      const r = cv.getBoundingClientRect();
      const p = [(e.clientX-r.left)*(cv.width/r.width), (e.clientY-r.top)*(cv.height/r.height)];
      if (Math.abs(p[0]-down[0])+Math.abs(p[1]-down[1]) > 6) moved = true;
      if (moved){ draft=[Math.min(p[0],down[0]),Math.min(p[1],down[1]),Math.max(p[0],down[0]),Math.max(p[1],down[1])]; draw(d,cv,draft); }
    });
    cv.addEventListener('mouseup', e=>{
      if(!down) return;
      const r = cv.getBoundingClientRect();
      const p = [(e.clientX-r.left)*(cv.width/r.width), (e.clientY-r.top)*(cv.height/r.height)];
      const cur = st(d.file);
      if (!moved){
        // 先看是不是点在自己补的黄框上（在图层最上面），是就撤掉它
        const ai = hitAdded(d, down[0], down[1], cv.width, cv.height);
        if (ai >= 0) {
          cur.added.splice(ai, 1);
          save(); draw(d,cv,null); msg(d);
          down=null; return;
        }
        const i = hit(d, down[0], down[1], cv.width, cv.height);
        if (i>=0){
          // toggle：点一次标删、再点一次取消
          const k = cur.deleted.indexOf(i);
          if (k>=0) cur.deleted.splice(k,1); else cur.deleted.push(i);
          save(); draw(d,cv,null); msg(d);
        }
      } else {
        const x1=Math.min(p[0],down[0])/cv.width, y1=Math.min(p[1],down[1])/cv.height;
        const x2=Math.max(p[0],down[0])/cv.width, y2=Math.max(p[1],down[1])/cv.height;
        if ((x2-x1)>0.01 && (y2-y1)>0.01){ cur.added.push([x1,y1,x2,y2]); save(); draw(d,cv,null); msg(d); }
      }
      down=null; draft=null;
    });
    card.querySelector('.msg').textContent = '';
  });
  document.getElementById('export').onclick = async ()=>{
    const out = {};
    DATA.forEach(d=>{ const s=state[d.file]; if (s && (s.deleted.length||s.added.length)) out[d.file]=s; });
    const btn = document.getElementById('export');
    btn.textContent = '保存中…'; btn.disabled = true;
    try {
      const r = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
                                      body: JSON.stringify(out)});
      const j = await r.json();
      btn.textContent = j.ok ? ('已保存（删 '+j.deleted+' 补 '+j.added+'）') : '保存失败';
    } catch (e) {
      btn.textContent = '保存失败：需要从 localhost 打开';
    }
    setTimeout(()=>{ btn.textContent='保存到训练集'; btn.disabled=false; }, 2200);
  };
  document.getElementById('resetall').onclick = ()=>{
    if(!confirm('清空全部修改？')) return;
    Object.keys(state).forEach(k=>delete state[k]); save();
    DATA.forEach(d=>{ const cv=document.querySelector('#c-'+d.file+' canvas'); draw(d,cv,null); });
  };
  progress();
}
function msg(d){
  const s=st(d.file); const el=document.querySelector('#c-'+d.file+' .msg');
  if(el) el.textContent = (s.deleted.length?('删 '+s.deleted.length+'  '):'') + (s.added.length?('补 '+s.added.length):'');
}
boot();
</script></body></html>"""


def build_data() -> list:
    """把"预标注的那批帧"打包成页面要的 JSON（服务每次请求现算，页面本身不变）。"""
    meta = json.load(open(os.path.join(YOLO, "meta.json"), encoding="utf-8"))
    frames = [m for m in meta if m.get("labels_from") == "prelabel"]
    data = []
    for m in frames:
        split, base = m["split"], m["file"]
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(YOLO, "images", split, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        boxes = []
        for line in open(os.path.join(YOLO, "labels", split, base + ".txt")):
            p = line.split()
            if len(p) >= 5:
                boxes.append([int(p[0])] + [round(float(v), 6) for v in p[1:5]])
        scale = WIDTH / w
        small = cv2.resize(img, (WIDTH, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        # 框直接烤进图里：canvas 那层如果被浏览器安全策略挡住，至少还能看见框
        for b in boxes:
            cls, cx, cy, bw, bh = b
            x1 = int((cx - bw / 2) * WIDTH); y1 = int((cy - bh / 2) * small.shape[0])
            x2 = int((cx + bw / 2) * WIDTH); y2 = int((cy + bh / 2) * small.shape[0])
            color = (255, 255, 255) if cls == 0 else (255, 160, 0)
            cv2.rectangle(small, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        # 文字也烤进图：canvas 那层若被浏览器挡掉，提示就丢了
        n_tile = sum(1 for b in boxes if b[0] == 1)
        if n_tile < 3:
            pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.rectangle([0, 0, pil.width, 44], fill=(0, 0, 0))
            try:
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, 8), f"只有 {n_tile} 块砖 —— 空盘，还是模型漏检？",
                      font=font, fill=(255, 204, 0))
            small = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        data.append({
            "file": base, "src": m["src"], "w": w, "h": h, "scale": scale,
            "boxes": boxes, "img": base64.b64encode(buf.tobytes()).decode(),
        })
    return data


def main() -> None:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE)
    print(f"生成静态页 {OUT}（数据由 box_server.py 动态提供）")
    print("启动服务：.venv-train/bin/python build-dataset/box_server.py")


if __name__ == "__main__":
    main()
