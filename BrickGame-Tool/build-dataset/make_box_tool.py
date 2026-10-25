#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 YOLO 框标注页（box_tool.html），只放"还没人工确认"的帧。

为什么必须人工过一遍：预标注是当前模型自己出的框，漏检的砖在训练里等于"这里没有砖"，
错误会被固化。逐格点 140 个框不现实，所以这里只做两件事：删掉误检、补上漏检。
保存后该帧写进 yolo/reviewed.json，页面不再展示它。

用法：
  cd BrickGame-Tool && .venv-train/bin/python build-dataset/make_box_tool.py
  打开生成的 box_tool.html → 改完点「导出」→ 得到 box_fixes.json
  再跑 apply_box_fixes.py 写回 labels
"""
from __future__ import annotations

import base64
import time
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO = os.path.join(HERE, "yolo")
SAMPLES = os.path.join(HERE, "samples")
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
 .card.recheck{border:2px solid #c00;box-shadow:0 0 0 3px rgba(204,0,0,.08)}
 .cov{font-weight:600} .cov.low{color:#c00}
 .hd{display:flex;justify-content:space-between;font-size:12px;color:#555;margin-bottom:6px}
 canvas{display:block;width:100%;height:auto;background:#000;border-radius:6px;cursor:crosshair}
 .row{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
 .msg{font-size:12px;color:#888;margin-left:auto}
</style></head><body>
<header>
  <b>砖块框标注</b>
  <span class="hint">蓝色 = 砖块（点一下 = 删掉它）；拖空白 = 补漏检；棋盘偏了点卡片上的「重画棋盘」再拖一个框</span>
  <span id="prog" class="hint"></span>
  <a id="allbtn" class="hint" style="cursor:pointer;text-decoration:underline"></a>
  <button class="primary" id="export">保存到训练集</button>
  <button id="resetall">清空修改</button>
</header>
<div id="grid"></div>
<script>
let DATA = [], REVIEWED = 0, TOTAL = 0, AUDIT = null, SHOWALL = false, NEED = 0;
async function boot(){
  try {
    const q = new URLSearchParams(location.search); q.set('batch', '40');
    const r = await fetch('/data?' + q.toString());
    const j = await r.json();
    DATA = j.frames; REVIEWED = j.reviewed; TOTAL = j.total;
    AUDIT = j.audit; SHOWALL = j.show_all; NEED = j.need;
  } catch (e) {
    document.getElementById('grid').innerHTML =
      '<p style="padding:20px;color:#c00">请通过本地服务打开：先运行 box_server.py，再访问 http://127.0.0.1:8771/</p>';
    return;
  }
  const ab = document.getElementById('allbtn');
  ab.textContent = SHOWALL ? '只看待办' : '查看全部 ' + TOTAL + ' 帧';
  ab.onclick = () => { location.href = SHOWALL ? '/' : '/?all=1'; };
  if (AUDIT && AUDIT.stale) {
    const el = document.getElementById('prog');
    el.style.color = '#b42318';
    el.textContent = '⚠ 复检未生效：' + AUDIT.why + ' ｜ 列表为空不代表标签正确';
  }
  build();
  updateMore();
}

// 一次只拉 40 帧：255 帧的 base64 图加起来 48MB，一口气返回浏览器直接卡死
function updateMore(){
  const box = document.getElementById('more');
  if (!NEED || DATA.length >= NEED) { box.style.display = 'none'; return; }
  box.style.display = 'block';
  document.getElementById('morebtn').textContent = '加载更多（' + DATA.length + '/' + NEED + '）';
  document.getElementById('morebtn').onclick = async () => {
    const btn = document.getElementById('morebtn');
    btn.disabled = true; btn.textContent = '加载中…';
    const q = new URLSearchParams(location.search); q.set('batch', '40'); q.set('offset', DATA.length);
    try {
      const r = await fetch('/data?' + q.toString());
      const j = await r.json();
      NEED = j.need;
      const from = DATA.length;
      DATA = DATA.concat(j.frames);
      build(from); updateMore(); progress();
    } catch (e) {
      btn.textContent = '加载失败，点这里重试'; btn.disabled = false;
    }
  };
}
// 只存在内存里：浏览器不存任何标注状态，所有改动一律当场写回服务端。
// 之前用 localStorage 当缓存，页面看着"记住了"，服务端其实没收到，刷新即丢。
const state = {};
const cache = {};
const queues = {};         // 同一帧的请求串行，避免快速连点时乱序覆盖

function st(f){ if(!state[f]) state[f] = {deleted:[], added:[], board:null, ok:false}; return state[f]; }
const armed = new Set();   // 哪些卡片处于"重画棋盘"模式（纯界面状态）

/** 把某帧的当前状态立刻写回服务端；失败就在卡片上报错，不留含糊。 */
function push(d){
  const send = ()=>fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
                                   body: JSON.stringify({[d.file]: st(d.file)})})
    .then(r=>r.json()).then(j=>{
      if (!j.ok) throw new Error(j.error || 'save failed');
      REVIEWED = j.reviewed;
      progress(); msg(d);
      const el = document.querySelector('#c-'+d.file+' .msg');
      if (j.missing && j.missing.length) {
        el.textContent = '未写入：服务端没有这帧的框文件';
        return false;
      }
      const s = st(d.file);
      const brief = (s.deleted.length ? '删 '+s.deleted.length+' ' : '')
                  + (s.added.length ? '补 '+s.added.length+' ' : '')
                  + (s.board ? '重画棋盘 ' : '');
      el.textContent = brief + (s.ok ? '已确认，服务端已记录' : '已写入');
      return true;
    }).catch(e=>{
      const el = document.querySelector('#c-'+d.file+' .msg');
      if (el) el.textContent = '未写入：' + (e.message || '需要通过 127.0.0.1:8771 打开');
      return false;
    });
  queues[d.file] = (queues[d.file] || Promise.resolve()).then(send, send);
  return queues[d.file];
}
function save(){ progress(); }
function progress(){
  const n = DATA.filter(d => { const s=state[d.file]; return s && (s.deleted.length || s.added.length || s.board || s.ok); }).length;
  document.getElementById('prog').textContent =
    '已确认 ' + REVIEWED + '/' + TOTAL + '（本次改动 ' + n + '/' + DATA.length + '）';
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
  if (cur.ok){
    ctx.fillStyle = 'rgba(0,150,60,.85)';
    ctx.fillRect(W - 88, 8, 80, 24);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 14px sans-serif';
    ctx.fillText('已确认', W - 76, 25);
  }
  if (cur.board){
    const b = cur.board;
    ctx.lineWidth = 3; ctx.strokeStyle = '#00e5ff';
    ctx.strokeRect(b[0]*W, b[1]*H, (b[2]-b[0])*W, (b[3]-b[1])*H);
    ctx.fillStyle = '#00e5ff'; ctx.font = 'bold 13px sans-serif';
    ctx.fillText('新棋盘', b[0]*W + 4, b[1]*H + 15);
  }
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
    ctx.lineWidth = 2;
    ctx.strokeStyle = armed.has(d.file) ? '#00e5ff' : '#ffcc00';
    ctx.setLineDash([6,4]);
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
function build(from = 0){
  const host = document.getElementById('grid');
  if (!DATA.length){
    host.innerHTML = '<p style="padding:28px;font-size:16px;color:#0a7">'
      + '没有待标帧：全部已人工确认，模型复检也没发现漏标。<br>'
      + '<span style="color:#666;font-size:14px">接着跑 make_yolo_data.py → train_yolo.py</span></p>';
    return;
  }
  DATA.slice(from).forEach(d=>{
    const card = document.createElement('div');
    card.className = 'card' + (d.recheck ? ' recheck' : ''); card.id='c-'+d.file;
    const w = Math.round(d.w*d.scale), h = Math.round(d.h*d.scale);
    const covPct = d.cov == null ? null : Math.round(d.cov * 100);
    const covTag = (d.recheck ? '⚠ 复查 · ' : '')
      + (covPct == null ? '覆盖率未知'
                        : '<span class="cov' + (covPct < 50 ? ' low' : '') + '">覆盖 ' + covPct + '%</span>');
    card.innerHTML =
      '<div class="hd"><span>'+d.src+'</span><span>砖 '+d.boxes.filter(b=>b[0]===1).length+' 块 · '+covTag+'</span></div>'+
      '<canvas width="'+w+'" height="'+h+'"></canvas>'+
      '<div class="row"><button class="bd">重画棋盘</button>'+
      '<button class="ok">确认无误</button><span class="msg"></span></div>';
    host.appendChild(card);
    const cv = card.querySelector('canvas');
    const img = new Image();
    img.onload = ()=>{ cache[d.file]=img; draw(d,cv,null); };
    img.src = 'data:image/jpeg;base64,'+d.img;

    const bbtn = card.querySelector('.bd');
    const obtn = card.querySelector('.ok');
    const paintOk = ()=>{
      const on = st(d.file).ok;
      obtn.textContent = on ? '已确认 ✓' : '确认无误';
      obtn.classList.toggle('primary', on);
    };
    paintOk();
    obtn.onclick = ()=>{
      const s = st(d.file);
      const next = !s.ok;
      s.ok = next; paintOk(); save(); draw(d,cv,null); msg(d);
      push(d).then(ok=>{
        if (ok) return;
        s.ok = !next; paintOk(); save(); draw(d,cv,null); msg(d);   // 没写进去就别显示成已确认
      });
    };
    bbtn.onclick = ()=>{
      if (armed.has(d.file)) armed.delete(d.file); else armed.add(d.file);
      bbtn.textContent = armed.has(d.file) ? '拖一个棋盘框…' : '重画棋盘';
      bbtn.classList.toggle('primary', armed.has(d.file));
    };

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
          save(); draw(d,cv,null); msg(d); push(d);
          down=null; return;
        }
        const i = hit(d, down[0], down[1], cv.width, cv.height);
        if (i>=0){
          // toggle：点一次标删、再点一次取消
          const k = cur.deleted.indexOf(i);
          if (k>=0) cur.deleted.splice(k,1); else cur.deleted.push(i);
          save(); draw(d,cv,null); msg(d); push(d);
        }
      } else {
        const x1=Math.min(p[0],down[0])/cv.width, y1=Math.min(p[1],down[1])/cv.height;
        const x2=Math.max(p[0],down[0])/cv.width, y2=Math.max(p[1],down[1])/cv.height;
        if ((x2-x1)>0.01 && (y2-y1)>0.01){
          if (armed.has(d.file)){
            cur.board = [x1,y1,x2,y2];
            armed.delete(d.file);
            bbtn.textContent = '重画棋盘'; bbtn.classList.remove('primary');
          } else {
            cur.added.push([x1,y1,x2,y2]);
          }
          save(); draw(d,cv,null); msg(d); push(d);
        }
      }
      down=null; draft=null;
    });
    card.querySelector('.msg').textContent = '';
  });
  document.getElementById('export').onclick = async ()=>{
    const out = {};
    DATA.forEach(d=>{ const s=state[d.file]; if (s && (s.deleted.length||s.added.length||s.board||s.ok)) out[d.file]=s; });
    const btn = document.getElementById('export');
    btn.textContent = '保存中…'; btn.disabled = true;
    try {
      const r = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
                                      body: JSON.stringify(out)});
      const j = await r.json();
      REVIEWED = j.reviewed || REVIEWED;
      btn.textContent = j.ok ? ('已保存（删 '+j.deleted+' 补 '+j.added+'）') : '保存失败';
      if (j.missing && j.missing.length) btn.textContent += '｜⚠ ' + j.missing.length + ' 帧没找到框文件';
      progress();
    } catch (e) {
      btn.textContent = '保存失败：需要从 localhost 打开';
    }
    setTimeout(()=>{ btn.textContent='保存到训练集'; btn.disabled=false; }, 2200);
  };
  document.getElementById('resetall').onclick = ()=>{
    // 改动已经逐条写到服务端，本地清不掉；只能重新装配或用 .orig 还原
    if(!confirm('所有改动已实时写入服务端，这里无法撤销。刷新页面重新加载？')) return;
    location.reload();
  };
  progress();
}
function msg(d){
  const s=st(d.file); const el=document.querySelector('#c-'+d.file+' .msg');
  if(el) el.textContent = (s.deleted.length?('删 '+s.deleted.length+'  '):'')
    + (s.added.length?('补 '+s.added.length+'  '):'')
    + (s.board?'棋盘已重画  ':'') + (s.ok?'已确认':'');
}
boot();
</script></body></html>"""


def reviewed_frames() -> set:
    """box_server 保存过的帧（人工确认完毕）—— 这些不用再铺到页面上。"""
    p = os.path.join(YOLO, "reviewed.json")
    if not os.path.exists(p):
        return set()
    try:
        return set(json.load(open(p, encoding="utf-8")).get("frames", []))
    except (json.JSONDecodeError, OSError):
        return set()


def coverage(boxes: list, m: dict) -> float | None:
    """一帧的标注覆盖率：优先按格数，台账缺 rows/cols 时按面积估。

    覆盖率低 = 漏检多，而漏检在训练里等于"这里没有砖"，比没标更糟。
    """
    if m.get("rows") and m.get("cols"):
        return m["tiles"] / (m["rows"] * m["cols"])
    board = max([b[3] * b[4] for b in boxes if b[0] == 0] or [0])
    tiles = sum(b[3] * b[4] for b in boxes if b[0] == 1)
    return (tiles / board) if board else None


def audit_frames() -> dict:
    """audit_labels.py 写下的逐帧复检结果（按 base 索引）。"""
    p = os.path.join(YOLO, "audit.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8")).get("frames", {})
    except (json.JSONDecodeError, OSError):
        return {}


def audit_gap() -> int:
    """复检用的阈值，页面必须与它一致，否则两边筛出来的帧不一样。"""
    p = os.path.join(YOLO, "audit.json")
    try:
        return int(json.load(open(p, encoding="utf-8")).get("gap", 10))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 10


def _cells(boxes: list, board: list, rows: int, cols: int) -> set:
    """框 → 占用格子集合。boxes 每项是 [cls, cx, cy, w, h]（归一化）。"""
    _, bx, by, bw, bh = board
    x0, y0 = bx - bw / 2, by - bh / 2
    got = set()
    for _, cx, cy, _, _ in boxes:
        c = int((cx - x0) / (bw / cols)); r = int((cy - y0) / (bh / rows))
        if 0 <= r < rows and 0 <= c < cols:
            got.add((r, c))
    return got


def solved(rec: dict, boxes: list, m: dict) -> bool:
    """复查帧是否已经修好 —— 按当初报的原因重算。

    不能用"框数比之前多"来判断：整行缺 10 块、补 2 块也会变多，
    那一行还缺 8 块。整行/整列要查那行现在是否有框，散点才退回数量。
    """
    import re
    reason = rec.get("reason") or ""
    if not reason:
        return False
    tiles = [b for b in boxes if b[0] == 1]
    boards = [b for b in boxes if b[0] == 0]
    if not (m["rows"] and m["cols"] and boards):
        return len(tiles) >= rec.get("model", 0) - 2
    lab = _cells(tiles, boards[0], m["rows"], m["cols"])
    if reason.startswith("整行缺失"):
        want = [int(x) for x in re.findall(r"\d+", reason)]
        return bool(want) and all(any(rr == r for rr, _ in lab) for r in want)
    if reason.startswith("整列缺失"):
        want = [int(x) for x in re.findall(r"\d+", reason)]
        return bool(want) and all(any(cc == c for _, cc in lab) for c in want)
    return len(tiles) >= rec.get("model", 0) - 2


def is_suspect(v: dict, gap: int) -> bool:
    """这一帧当时是否被判为疑似漏标。

    判据以结构性缺失为主（整行/整列），散点用格子差兜底 —— 只看框数会被
    模型对同一格的重复检出带偏。
    """
    if v.get("reason"):
        return True
    extra = v.get("extra_cells")
    if extra is not None:
        return extra >= gap
    if v["model"] - v["label"] < gap:      # 无台账：退回数量差
        return False
    cells = v.get("cells") or 0
    return not (cells and v["label"] >= cells * 0.95)


def audit_state() -> dict:
    """复检结果是否还可用。

    文件缺失/损坏时退回空集合，页面会变成"全部确认完毕"—— 那是最容易骗人的状态，
    所以这里把状态显式带出去，前端必须提示。
    """
    p = os.path.join(YOLO, "audit.json")
    if not os.path.exists(p):
        return {"stale": True, "why": "还没跑过 audit_labels.py"}
    try:
        a = json.load(open(p, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"stale": True, "why": "audit.json 损坏"}
    best = os.path.join(os.path.dirname(HERE), "runs", "brick", "weights", "best.pt")
    # 文件名永远是 best.pt，重训只是覆盖内容 —— 只能比 mtime
    if os.path.exists(best) and os.path.getmtime(best) > os.path.getmtime(p):
        return {"stale": True, "why": "模型比复检结果新，需要重跑 audit_labels.py"}
    age = time.time() - os.path.getmtime(p)
    if age > 7 * 86400:
        return {"stale": True, "why": f"复检已 {int(age // 86400)} 天未更新"}
    return {"stale": False, "why": "", "at": a.get("at")}


def build_data(show_all: bool = False, offset: int = 0, limit: int | None = None) -> tuple[list, int]:
    """待标帧 + 复查帧打包成页面要的 JSON（服务每次请求现算，页面本身不变）。

    已确认但被 `audit_labels.py` 判为漏标的帧也会拉回来（recheck=True），排在前面 ——
    点过确认不等于标全了，那批漏检会在下一轮变成负样本。
    """
    meta = json.load(open(os.path.join(YOLO, "meta.json"), encoding="utf-8"))
    done = reviewed_frames()
    recs = audit_frames()
    gap = audit_gap()
    suspects = {b for b, v in recs.items() if is_suspect(v, gap)}
    # 先按 label 算覆盖率，决定哪些帧要上页面（图片编码很贵，只处理入选的）
    need: list[tuple] = []
    for m in meta:
        base = m["file"]
        lab = os.path.join(YOLO, "labels", m["split"], base + ".txt")
        if not os.path.exists(lab):
            continue
        boxes = []
        for line in open(lab, encoding="utf-8"):
            q = line.split()
            if len(q) >= 5:
                boxes.append([int(q[0])] + [round(float(v), 6) for v in q[1:5]])
        cov = coverage(boxes, m)
        is_done = base in done
        # 已确认的帧只在"模型能检出更多砖"时拉回复查；show_all 时全部列出供回看
        if is_done and base not in suspects and not show_all:
            continue
        if is_done and base in suspects and not show_all:
            # 补过框就当场撤出复查列表；show_all 时不撤，否则"查看全部"会少帧
            if solved(recs.get(base) or {}, boxes, m):
                continue
        need.append((m, boxes, cov, is_done))

    # 复查帧优先，其余按覆盖率从低到高：最容易出错的最先看到。
    # 排序要在切片之前，否则分页顺序会乱。
    need.sort(key=lambda t: (not t[3], t[2] if t[2] is not None else 1.0))
    total = len(need)
    page = need[offset: (offset + limit) if limit else None]

    data = []
    for m, boxes, cov, is_done in page:
        split, base = m["split"], m["file"]
        # 优先取 samples 原图：yolo/images 已按 MAX_SIDE（=IMG_H）缩过，拿它画
        # 460 宽的预览等于再缩一次，tile 边缘会糊，人工补框看不准。
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(SAMPLES, m["src"] + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            for ext in (".jpg", ".png", ".jpeg"):
                p = os.path.join(YOLO, "images", split, base + ext)
                if os.path.exists(p):
                    img_path = p
                    break
        if not img_path:
            continue
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
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
            "cov": cov, "recheck": is_done,
        })

    return data, total


def main() -> None:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE)
    print(f"生成静态页 {OUT}（数据由 box_server.py 动态提供）")
    print("启动服务：.venv-train/bin/python build-dataset/box_server.py")


if __name__ == "__main__":
    main()
