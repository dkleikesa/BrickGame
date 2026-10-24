#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 class_id 铺开整个数据集，人工挑出重复/对训练没帮助的图，标了删掉。

一行 = 一个 class_id：
  左栏：图鉴参考图 + class_id + entry + 张数（含已标删数），下面三个批量按钮
  右栏：这一类在数据集里的**全部**截图格，按 (帧, 行, 列) 排序 —— 同一帧里的同款图天然挨在一起，
        重复的一眼就能连着一片标掉
点一张图 = 标记待删（再点取消）。悬停放大 2.2 倍看细节。标完点"导出待删清单 JSON"，
再跑 apply_deletions.py --inplace 从 dataset/index.csv 摘掉（训练就不再用它们）。

为什么页面不直接删：file:// 打开的页面碰不到文件系统。沿用 annotate.html 的老办法 ——
页面只产清单，落地交给 python 脚本（可预览、有留档、能回滚）。

标删只存在内存：提交前刷新会丢标记（index.csv 没动，数据不丢）。
用法：
  .venv/bin/python review_dataset.py            # → review.html
  .venv/bin/python review_dataset.py --open     # 生成并用默认浏览器打开
"""
from __future__ import annotations

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>数据集去重 / 删除</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,'PingFang SC',sans-serif;background:#f5f6f8;color:#1f2328;font-size:12px}
header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #e3e5e8;padding:7px 13px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
header b{font-size:13px}
button,select,input{font:inherit;padding:3px 8px;border:1px solid #d0d5da;background:#fff;border-radius:5px}
button{cursor:pointer}button:hover{background:#f0f2f5}
button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
button.danger{color:#b42318;border-color:#f0c9c5}
.spacer{flex:1}
.stat{color:#5b6169;font-variant-numeric:tabular-nums}
#wrap{padding:11px;--ts:56px}
.head{display:flex;gap:9px;align-items:baseline;margin:2px 0 9px;flex-wrap:wrap}
.head b{font-size:13px}
.head span{color:#8b8f96}
/* 一个 class_id 一块：左栏固定宽，右栏铺图 */
.cls{display:flex;gap:11px;align-items:flex-start;background:#fff;border:1px solid #e9ecf0;border-radius:8px;padding:9px;margin-bottom:9px}
.cls.has{border-color:#f3c6c2;background:#fffafa}
.left{width:128px;flex:none;text-align:center}
.left .ref{width:74px;height:74px;image-rendering:pixelated;border-radius:6px;background:#eef0f2;display:block;margin:0 auto}
.left .noref{width:74px;height:74px;margin:0 auto;border-radius:6px;background:#eef0f2;color:#8b8f96;line-height:74px}
.left .cid{font-weight:600;font-size:14px;font-variant-numeric:tabular-nums;margin-top:4px}
.left .entry{color:#8b8f96;font-size:10px;word-break:break-all;line-height:1.3}
.left .cnt{color:#5b6169;font-size:11px;margin-top:3px}
.left .cnt b{color:#b42318}
.left .src{color:#8b8f96;font-size:10px;line-height:1.4;margin-top:2px}
.left .src .man{color:#1a7f37;font-weight:600}
#warn{background:#fff8e6;border:1px solid #f2d38a;color:#7a4b00;border-radius:6px;padding:7px 10px;margin:0 0 9px;line-height:1.6}
#warn code{background:#fff;border:1px solid #e6d3a8;border-radius:4px;padding:0 4px}
#sync{background:#eef4ff;border:1px solid #c7d9ff;color:#1f4fa8;border-radius:6px;padding:7px 10px;margin:0 0 9px}
#sync b{text-decoration:underline;cursor:pointer}
.left .acts{display:flex;flex-direction:column;gap:4px;margin-top:6px}
.left .acts button{font-size:11px;padding:2px 6px}
.tiles{flex:1;display:flex;flex-wrap:wrap;gap:6px;min-width:0}
/* 悬停放大是这里唯一的"看清细节"手段：不占版面、不打断连点标记的节奏 */
.tile{width:var(--ts);cursor:pointer;text-align:center}
.tile img{width:var(--ts);height:var(--ts);image-rendering:pixelated;border-radius:5px;background:#eef0f2;display:block;border:2px solid transparent;transition:transform .08s}
.tile:hover img{border-color:#1f6feb}
.body-boxing .tile:hover img{border-color:transparent}
.tile .cap{font-size:9px;color:#8b8f96;line-height:1.3;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.tile .cap.k-manual{color:#1a7f37;font-weight:600}
.tile.del img{opacity:.25;border-color:#d1242f}
.tile.chg img{border-color:#1f6feb;box-shadow:0 0 0 2px rgba(31,111,235,.35)}
.tile.del .cap{color:#d1242f;text-decoration:line-through}
.tile.del{position:relative}
.tile.del::after{content:'✕';position:absolute;top:1px;right:3px;color:#d1242f;font-weight:700;font-size:15px;text-shadow:0 0 3px #fff}
.empty{color:#8b8f96;padding:20px;text-align:center}
/* 右侧抽屉：在去重页里直接改 class_id（服务模式才写得了 index.csv） */
#mask{position:fixed;inset:0;background:rgba(20,22,26,.35);z-index:40;display:none}
#mask.show{display:block}
#panel{position:fixed;top:0;right:0;height:100vh;width:420px;max-width:94vw;background:#fff;z-index:50;
  overflow:auto;padding:13px;box-shadow:-3px 0 14px rgba(0,0,0,.14);transform:translateX(102%);transition:transform .16s ease-out}
#panel.show{transform:none}
#panel .close{position:absolute;top:9px;right:11px}
#panel .sec{color:#8b8f96;font-size:10px;margin:11px 0 5px}
#panel .acts{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap}
#panel .tag{font-size:10px;color:#8b8f96;line-height:1.5}
#panel .cur{color:#1a7f37;font-weight:600}
.grid{display:flex;flex-wrap:wrap;gap:5px}
.cand{display:inline-block;text-align:center;cursor:pointer;padding:2px;border-radius:6px;border:2px solid transparent}
.cand:hover{border-color:#1f6feb;background:#f4f7ff}
.cand img{width:64px;height:64px;image-rendering:pixelated;border-radius:4px;background:#eee;display:block}
.cand .ct{font-size:9px;color:#8b8f96}
.tile .edit{display:none;position:absolute;top:1px;left:2px;padding:0 4px;font-size:11px;line-height:1.4;border-radius:4px;border:1px solid #d0d5da;background:#fff;z-index:31}
body.serve .tile:hover .edit{display:block}
</style>
</head>
<body>
<header>
  <b>数据集去重 / 删除</b>
  <input id="q" placeholder="搜 class_id / entry / 帧名" style="width:200px">
  <select id="sz" title="图片大小">
    <option value="44">小</option><option value="56" selected>中</option><option value="72">大</option>
  </select>
  <button id="onlydel">只看已标删</button>
  <button id="hidedel">隐藏已标删</button>
  <span class="stat" id="stat"></span>
  <span class="spacer"></span>
  <button id="cp">复制已标删路径</button>
  <button id="exp">导出待删清单 JSON</button>
  <button id="undo" class="danger" hidden>撤销上次删除</button>
  <button id="del" class="danger" hidden>删除选中</button>
  <button id="clr" class="danger">清空标记</button>
  <button id="reload">刷新页面</button>
</header>
<div id="wrap">
  <div class="head"><b id="ds"></b><span id="hint">点图 = 标记删除（再点取消）· 按住拖拽框选 = 标记框住的图（要整张都在框里）</span></div>
  <div id="sync" hidden></div>
  <div id="warn" hidden></div>
  <div id="list"></div>
</div>
<div id="mask"></div>
<div id="panel"><button class="close" id="pclose">关闭</button><div id="pbody"></div></div>
<script>
const DATA = __DATA__;
// 标删状态只放内存：提交前刷新会丢标记（数据不会丢，index.csv 没动）。
// 不做任何本地持久化：刷新后标记清零，一切以 index.csv 为准。
let S = {del: {}};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const REF = DATA.refs;                                   // class_id → entry（JSON 的对象键已转成字符串）
const refSrc = cid => (cid === null || REF[cid] === undefined) ? '' : DATA.tiles + REF[cid] + '.png';
const save = () => { stats(); };

// 每类的成员路径表：点一下只重算本行计数，不整体重绘（上千张图，innerHTML 重绘会闪）
const PATHS = {}; DATA.classes.forEach(c => PATHS[c.key] = c.members.map(m => m.path));
function cntDel(key) { let n = 0; PATHS[key].forEach(p => { if (S.del[p]) n++; }); return n; }

// 复制文本：Clipboard API 优先；file:// 下老浏览器没有它，退回 textarea + execCommand
function copyText(text, btn) {
  const ok = () => {
    if (!btn) return;
    clearTimeout(btn._t);
    if (btn._label === undefined) btn._label = btn.textContent;
    btn.textContent = '已复制 ✓'; btn.classList.add('on');
    btn._t = setTimeout(() => { btn.textContent = btn._label; btn.classList.remove('on'); }, 900);
  };
  const fallback = () => {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta); ta.select();
    let done = false;
    try { done = document.execCommand('copy'); } catch(e) { done = false; }
    ta.remove();
    if (done) ok(); else prompt('自动复制失败，请手动复制：', text);
  };
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(ok, fallback);
  else fallback();
}

function tileHtml(m) {
  const del = S.del[m.path] ? ' del' : '';
  const t = `${esc(m.path)}\n${esc(m.frame)} 第${m.row}行${m.col}列 · ${m.kind}${m.d ? ' · 距离 ' + m.d : ''}`;
  return `<div class="tile${del}" data-path="${esc(m.path)}" data-frame="${esc(m.frame)}" data-cid="${m.cid === null ? '' : m.cid}" title="${t}">
    <button class="edit" title="改这一格的 class_id（写回 index.csv）">✎</button>
    <img src="${m.src}" loading="lazy">
    <div class="cap${m.kind === 'manual' ? ' k-manual' : ''}">#${m.fi} r${m.row}c${m.col}</div>
  </div>`;
}

const SRCN = {auto: '自动', app: '应用', manual: '人工', empty: '背景', unknown: '未识别'};
// 这一类的标签从哪来：一眼看出"我在标注页改的那条到底写回 index.csv 没有"——
// 改完写回的行 cls_src 会变成 manual，这里"人工"N 就会涨。
function srcLine(c) {
  return Object.entries(c.src || {}).sort((a, b) => b[1] - a[1])
    .map(([k, n]) => k === 'manual'
      ? `<span class="man">人工 ${n}</span>` : `${SRCN[k] || k} ${n}`).join(' · ');
}

// 标注页现在逐格即时写回 index.csv，不存在"标了但没写回"的状态，这里恒为 0。
function pendingEdits() { return 0; }
function showWarn() {
  document.getElementById('warn').hidden = true;
}

function rowsHtml() {
  const q = document.getElementById('q').value.trim().toLowerCase();
  const onlyDel = document.getElementById('onlydel').classList.contains('on');
  const hideDel = document.getElementById('hidedel').classList.contains('on');
  let out = '';
  for (const c of DATA.classes) {
    const hit = !q || String(c.cid).includes(q) || (c.entry || '').toLowerCase().includes(q)
      || c.members.some(m => m.frame.toLowerCase().includes(q) || m.path.toLowerCase().includes(q));
    if (!hit) continue;
    const nd = cntDel(c.key);
    if (onlyDel && !nd) continue;
    const mem = hideDel ? c.members.filter(m => !S.del[m.path]) : c.members;
    if (!mem.length) continue;
    out += `<div class="cls${nd ? ' has' : ''}" data-key="${esc(c.key)}">
      <div class="left">
        ${refSrc(c.cid) ? `<img class="ref" src="${refSrc(c.cid)}" loading="lazy">` : '<div class="noref">无图鉴</div>'}
        <div class="cid">${c.cid === null ? '未识别' : c.cid}</div>
        <div class="entry">${esc(c.entry || '')}</div>
        <div class="cnt">共 ${c.n} 张 · 已标删 <b class="nd">${nd}</b></div>
        <div class="src">${srcLine(c)}</div>
        <div class="acts">
          <button class="b1" title="标删这一类里除第一张外的全部">只留首张</button>
          <button class="b2">全标删</button>
          <button class="b3">撤销本类</button>
        </div>
      </div>
      <div class="tiles">${mem.map(tileHtml).join('')}</div>
    </div>`;
  }
  return out;
}

function stats() {
  const n = Object.keys(S.del).length;
  let cls = 0;
  for (const c of DATA.classes) if (cntDel(c.key)) cls++;
  document.getElementById('stat').textContent =
    `共 ${DATA.classes.length} 类 / ${DATA.total} 张 · 已标删 ${n} 张（涉及 ${cls} 类）`
    + (n ? ' · 未提交，刷新会丢' : '');
}
function render() {
  document.getElementById('list').innerHTML = rowsHtml() || '<div class="empty">没有匹配的类</div>';
  stats();
  showWarn();
}

// ---- 改标注：点图上的 ✎ 打开抽屉，指派新的 class_id 并写回 index.csv ----
let SEL = null;
const M = {};                                    // path → 成员（面板要它的 frame/行列/当前类别）
DATA.classes.forEach(c => c.members.forEach(m => M[m.path] = m));
const REFLIST = Object.entries(REF).map(([cid, entry]) => [cid, entry])
  .sort((a, b) => +a[0] - +b[0]);
const KINDNAME = {auto: '自动识别', app: '应用标注', manual: '人工', bgx: '', empty: '空位/背景', unknown: '导入未定'};
const openPanel = () => { document.getElementById('mask').classList.add('show');
                          document.getElementById('panel').classList.add('show'); };
const closePanel = () => { document.getElementById('mask').classList.remove('show');
                           document.getElementById('panel').classList.remove('show'); SEL = null; };

// 写回 index.csv（服务模式）。frames 留空：去重页只管类别，不碰 frames.csv 的"标完"状态。
let CHG = 0;
function writeCorr(corr, msg) {
  if (!DATA.serve) {
    alert('当前页面是 file:// 打开的，浏览器写不了 index.csv。\n'
      + '请用 http://127.0.0.1:8765/build-dataset/review.html 打开再改。');
    return;
  }
  fetch('/api/corrections', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({corrections: corr, frames: []})})
    .then(r => r.json().then(j => ({ok: r.ok, j})))
    .then(({ok, j}) => {
      if (!ok || !j.ok) throw new Error(j.error || '写回失败');
      // 改类别会把这格挪到别的类下，但**不整页刷新**：标一下就继续往下标，
      // 想看它归位时点右上角「刷新页面」即可（原来每次都 reload，标注完全没法连续做）。
      corr.forEach(c => {
        document.querySelectorAll('.tile').forEach(el => {
          if (el.dataset.path === c.path) el.classList.add('chg');
        });
      });
      CHG += corr.length;
      const sc = document.getElementById('stat');
      if (sc) sc.textContent = '已改 ' + CHG + ' 格（刷新后归位）';
    })
    .catch(e => alert('写回失败：' + e.message));
}

function renderPanel() {
  const p = document.getElementById('pbody');
  if (!SEL) { p.innerHTML = ''; return; }
  const m = SEL;
  const cur = m.cid === null ? '未识别' : m.cid;
  p.innerHTML = `
    <div><img src="${m.src}" style="width:100%;max-width:300px;height:auto;border-radius:8px;background:#eee;display:block">
      <div style="margin-top:5px"><a href="${m.src}" target="_blank" style="font-size:11px">在新标签打开原图（100%）</a></div>
      <div class="tag" style="margin-top:4px">${esc(m.path)}<br>帧 ${esc(m.frame)} · 第 ${m.row} 行 ${m.col} 列<br>
        当前类别 <span class="cur">${cur}</span>${REF[m.cid] ? '（' + esc(REF[m.cid]) + '）' : ''}
        · 来源 ${KINDNAME[m.kind] || m.kind}${m.d ? ' · 距离 ' + m.d : ''}</div></div>
    <div class="acts">
      <button id="pdel" class="danger">${S.del[m.path] ? '取消标记删除' : '标记删除'}</button>
      <button id="pbg" title="已消除的空位，正式类别就是 class 0">标为空位 0</button>
      <button id="punk" title="认不出是什么物品（训练时忽略），和空位不是一回事">标为未知</button>
      <button id="pcp">复制路径</button>
    </div>
    <div class="sec">点下面的图鉴条目 = 把这一格改成那个 class_id 并写回 index.csv
      （共 ${REFLIST.length} 个，可按编号或 entry 搜）</div>
    <input id="pq" placeholder="如 1428 / p010" style="width:100%">
    <div class="grid" id="pall" style="margin-top:8px"></div>`;
  const one = cid => writeCorr([{path: m.path, class_id: cid}], `把 ${m.path.split('/').pop()} 改成 ${cid}`);
  const box = p.querySelector('#pall');
  const paint = list => {
    box.innerHTML = list.map(([cid, entry]) => `<div class="cand" data-cid="${cid}" title="${esc(entry)}">
        <img src="${DATA.tiles}${entry}.png" loading="lazy"><div class="ct">${cid}</div></div>`).join('')
      || '<div class="tag">没有匹配</div>';
    box.querySelectorAll('.cand').forEach(c => c.onclick = () => one(+c.dataset.cid));
  };
  paint(REFLIST);
  const q = p.querySelector('#pq');
  q.oninput = () => {
    const s = q.value.trim().toLowerCase();
    paint(s ? REFLIST.filter(([cid, entry]) => String(cid).includes(s)
      || String(entry || '').toLowerCase().includes(s)) : REFLIST);
  };
  p.querySelector('#pdel').onclick = () => { toggleDel(m); renderPanel(); };
  p.querySelector('#pbg').onclick = () => writeCorr([{path: m.path, class_id: 0}], '已标为空位 0');
  p.querySelector('#punk').onclick = () => writeCorr([{path: m.path, unknown: true}], '已标为未知');
  p.querySelector('#pcp').onclick = () => copyText(DATA.root + '/' + m.path, p.querySelector('#pcp'));
}

function toggleDel(m) {
  if (S.del[m.path]) delete S.del[m.path];
  else S.del[m.path] = {frame: m.frame, cid: m.cid};
  save();
  const t = document.querySelector(`.tile[data-path="${m.path}"]`);
  if (!t) return;
  t.classList.toggle('del');
  const row = t.closest('.cls'), n = cntDel(row.dataset.key);
  row.querySelector('.nd').textContent = n;
  row.classList.toggle('has', n > 0);
  stats();
}

// ---- 框选批量标记：按住拖拽画一个框，框内所有缩略图一次标删 ----
let boxStart = null, boxLayer = null, suppressClick = false;
function ensureBoxLayer() {
  if (boxLayer) return boxLayer;
  boxLayer = document.createElement('div');
  boxLayer.style.cssText =
    'position:fixed;border:2px solid #ff3b30;background:rgba(255,59,48,.18);' +
    'pointer-events:none;z-index:9998;display:none;border-radius:4px';
  document.body.appendChild(boxLayer);
  return boxLayer;
}
document.addEventListener('mousedown', ev => {
  if (ev.button !== 0) return;
  if (ev.target.closest('button, #panel, .hd')) return;
  boxStart = {x: ev.clientX, y: ev.clientY, moved: false};
  document.body.classList.add('body-boxing');
}, true);
document.addEventListener('mousemove', ev => {
  if (!boxStart) return;
  const dx = ev.clientX - boxStart.x, dy = ev.clientY - boxStart.y;
  if (!boxStart.moved && (Math.abs(dx) > 6 || Math.abs(dy) > 6)) boxStart.moved = true;
  if (!boxStart.moved) return;
  ev.preventDefault();
  const el = ensureBoxLayer();
  el.style.display = 'block';
  el.style.left = Math.min(boxStart.x, ev.clientX) + 'px';
  el.style.top = Math.min(boxStart.y, ev.clientY) + 'px';
  el.style.width = Math.abs(dx) + 'px';
  el.style.height = Math.abs(dy) + 'px';
}, true);
document.addEventListener('mouseup', ev => {
  if (!boxStart) return;
  const st = boxStart; boxStart = null;
  document.body.classList.remove('body-boxing');
  if (boxLayer) boxLayer.style.display = 'none';
  if (!st.moved) return;          // 单击：交给原来的 click 处理
  suppressClick = true;           // 拖拽结束别再触发 click（否则会把拖到的图取消标记）
  const l = Math.min(st.x, ev.clientX), r = Math.max(st.x, ev.clientX);
  const t = Math.min(st.y, ev.clientY), b = Math.max(st.y, ev.clientY);
  let n = 0;
  document.querySelectorAll('.tile').forEach(el => {
    const rc = el.getBoundingClientRect();
    if (rc.width < 1) return;
    // 四角都在框内才算选中：只看中心点会把一整排同类的图都圈进去
    if (rc.left < l || rc.right > r || rc.top < t || rc.bottom > b) return;
    const ds = el.dataset;
    if (!S.del[ds.path]) {
      S.del[ds.path] = {frame: ds.frame, cid: ds.cid === '' ? null : Number(ds.cid)};
      n++;
    }
  });
  render();
  const sc = document.getElementById('stat');
  if (sc) sc.textContent = '框选标记 ' + n + ' 张';
}, true);
document.addEventListener('click', ev => {
  if (!suppressClick) return;
  suppressClick = false;
  ev.stopPropagation();
  ev.preventDefault();
}, true);

document.getElementById('reload').onclick = () => location.reload();

document.getElementById('list').addEventListener('click', ev => {
  // ✎：打开改标注抽屉（不是标删，别顺手把图标掉了）
  const ed = ev.target.closest('.edit');
  if (ed) {
    ev.stopPropagation();
    SEL = M[ed.closest('.tile').dataset.path] || null;
    renderPanel(); openPanel();
    return;
  }
  const btn = ev.target.closest('.left button');
  if (btn) {
    const row = btn.closest('.cls');
    const c = DATA.classes.find(x => x.key === row.dataset.key);
    const mark = t => S.del[t.path] = {frame: t.frame, cid: t.cid};
    if (btn.classList.contains('b1')) {
      if (!confirm(`「${c.cid === null ? '未识别' : c.cid}」标删除第一张外的 ${c.members.length - 1} 张？`)) return;
      c.members.forEach((m, i) => { if (i) mark(m); else delete S.del[m.path]; });
    } else if (btn.classList.contains('b2')) {
      if (!confirm(`「${c.cid === null ? '未识别' : c.cid}」整类 ${c.members.length} 张全标删？`)) return;
      c.members.forEach(mark);
    } else {
      c.members.forEach(m => delete S.del[m.path]);
    }
    save(); render(); return;
  }
  const t = ev.target.closest('.tile');
  if (!t) return;
  toggleDel(M[t.dataset.path] || {path: t.dataset.path, frame: t.dataset.frame,
                                  cid: t.dataset.cid === '' ? null : +t.dataset.cid});
});
document.getElementById('pclose').onclick = closePanel;
document.getElementById('mask').onclick = closePanel;
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePanel(); });

document.getElementById('q').oninput = render;
document.getElementById('sz').onchange = e => { document.getElementById('wrap').style.setProperty('--ts', e.target.value + 'px'); };
['onlydel', 'hidedel'].forEach(id => document.getElementById(id).onclick = e => { e.target.classList.toggle('on'); render(); });
document.getElementById('cp').onclick = ev => {
  const ps = Object.keys(S.del).map(p => DATA.root + '/' + p);
  if (!ps.length) return alert('还没标删任何图');
  copyText(ps.join('\n'), ev.currentTarget);
};
document.getElementById('exp').onclick = () => {
  const ds = Object.entries(S.del).map(([path, e]) => ({path, frame: e.frame, cid: e.cid}));
  if (!ds.length) return alert('还没标删任何图');
  if (!confirm(`导出 ${ds.length} 张待删？\n\n接着跑：\n  python apply_deletions.py --deletions deletions.json          预览\n  python apply_deletions.py --deletions deletions.json --inplace 落地`)) return;
  const out = {dataset: DATA.dataset, created: new Date().toISOString(), total: ds.length, deletes: ds};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'}));
  a.download = 'deletions.json'; a.click();
};
document.getElementById('clr').onclick = () => {
  if (!confirm('清空所有标记？（已导出的 JSON 不受影响）')) return;
  S = {del: {}}; save(); render();
};
document.getElementById('ds').textContent = DATA.dataset + ' · ' + DATA.frames.length + ' 帧';

// ---- 本地服务模式（review_server.py）：点"删除选中"直接落地，不用导 JSON ----
async function api(path, body) {
  const r = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'},
                                      body: JSON.stringify(body)} : {});
  let j = {};
  try { j = await r.json(); } catch (e) {}
  if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}
function serveUI() {
  document.getElementById('del').hidden = false;
  document.body.classList.add('serve');
  document.getElementById('hint').innerHTML =
    '点图 = 标记删除（再点取消）· 悬停图片左上角 <b>✎</b> = 改这一格的标注（写回 index.csv）'
    + ' · 标完点右上角「删除选中」直接落地（可撤销）'
    + ' · 分类按 index.csv 实时生成（标注写回后刷新即变）'
    + ' · 标注界面 <a href="annotate.html" target="_blank">annotate.html</a>';
  api('/api/state').then(j => {
    if (!j.undo_n) return;
    const b = document.getElementById('undo');
    b.hidden = false;
    b.textContent = `撤销上次删除（${j.undo_n} 张）`;
  }).catch(() => {});
}
document.getElementById('del').onclick = async () => {
  const ds = Object.entries(S.del).map(([path, e]) => ({path, frame: e.frame, cid: e.cid}));
  if (!ds.length) return alert('还没标删任何图');
  if (!confirm(`确认删除选中的 ${ds.length} 张？\n\n`
    + (DATA.purge ? '图片会被直接删除（不可恢复）。'
                  : '图片会挪到 dataset/deleted/，index.csv 摘掉这些行，可用"撤销上次删除"恢复。'))) return;
  try {
    const j = await api('/api/delete', {paths: ds.map(d => d.path)});
    ds.forEach(d => delete S.del[d.path]);
    save();
    alert(`已删除 ${j.matched} 张${j.missing && j.missing.length ? `（${j.missing.length} 张已不存在，跳过）` : ''}`
      + `\n剩余 ${j.total_after} 张 / 共 ${j.total_before} 张。页面马上刷新。`);
    location.reload();
  } catch (e) { alert('删除失败：' + e.message); }
};
document.getElementById('undo').onclick = async () => {
  try {
    const j = await api('/api/undo', {});
    alert(`已恢复 ${j.restored} 张${j.files ? `（图片挪回原位 ${j.files} 个）` : ''}。页面马上刷新。`);
    location.reload();
  } catch (e) { alert('撤销失败：' + e.message); }
};

// ---- 数据同步：标注页（另一个标签）写回后，本页不会自己变 ----
// 服务模式轮询 /api/state 里的 rev（index.csv 指纹）：变了就在页面上提示刷新；
// 本标签不在前台时直接刷（用户看不见闪烁，回来就是最新）。
let REV = '';
async function pollState() {
  try {
    const j = await api('/api/state');
    const r = j.rev || '';
    if (!r) return;
    if (!REV) { REV = r; return; }
    if (r === REV) return;
    REV = r;
    if (document.hidden) { location.reload(); return; }
    showSync();
  } catch (e) {}
}
function showSync() {
  const el = document.getElementById('sync');
  el.hidden = false;
  el.innerHTML = '数据已更新（标注页写回 / 别处改了 dataset/index.csv）· <b>点这里刷新</b>'
    + '<span style="color:#6b7280"> · 本页分类与数量按 index.csv 实时生成，刷新即一致</span>';
  el.querySelector('b').onclick = () => location.reload();
}
if (DATA.serve) {
  pollState();
  setInterval(pollState, 4000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) pollState(); });
}

render();
if (DATA.serve) serveUI();
</script>
</body>
</html>
"""


def kind_counts(ms: list[dict]) -> dict[str, int]:
    """这一类里每个格子各自的标签来源（auto/app/manual/empty/unknown）—— 页面上写"人工 4"，
    用来判断"我在标注页改的那条到底写回 index.csv 没有"（写回了就会算进 manual）。"""
    out: dict[str, int] = {}
    for m in ms:
        out[m["kind"]] = out.get(m["kind"], 0) + 1
    return out


def build_review(dataset: str = "", catalog: str = "", html: str = "", quiet: bool = False,
                 base: str = "", serve: bool = False, purge: bool = False) -> str:
    """按 class_id 铺开数据集，生成去重界面，返回 html 路径。

    base：图片路径相对谁（默认 = html 所在目录，file:// 打开用）。
    serve=True 时由 review_server.py 提供 /api/delete，页面上的"删除选中"按钮才出现；
    purge=True 表示点击就是真删文件（默认挪进 dataset/deleted/ 可撤销）。
    """
    dataset = dataset or os.path.join(_HERE, "dataset")
    catalog = catalog or os.path.join(_ROOT, "build-catalog", "catalog")
    html = html or os.path.join(_HERE, "review.html")
    base = base or os.path.dirname(os.path.abspath(html))
    rel = lambda p: os.path.relpath(os.path.abspath(p), base).replace(os.sep, "/")  # noqa: E731

    index_path = os.path.join(dataset, "index.csv")
    if not os.path.exists(index_path):
        raise SystemExit(f"缺少 {index_path}")
    rows = list(csv.DictReader(open(index_path)))

    # class_id → entry：左栏的参考图就是图鉴里那一条（没有就只显示编号）
    refs: dict[int, str] = {}
    cat_index = os.path.join(catalog, "index.csv")
    if os.path.exists(cat_index):
        for r in csv.DictReader(open(cat_index)):
            refs[int(r["class_id"])] = r["entry"]

    frames = sorted({r["frame"] for r in rows})
    fi = {f: i + 1 for i, f in enumerate(frames)}      # 帧序号：图下面只写 #3 r03c05，帧名太长

    by_cid: dict[int, list] = {}
    unknown: list = []
    for r in rows:
        fn = os.path.basename(r["path"])
        cid = (r.get("class_id") or "").strip()
        m = {
            "path": r["path"], "src": rel(os.path.join(dataset, r["path"])),
            "frame": r["frame"], "fi": fi[r["frame"]],
            "row": int(fn[1:3]), "col": int(fn[4:6]),
            "kind": r.get("cls_src") or "auto",
            "d": float(r["cls_dist"]) if r.get("cls_dist") else 0.0,
            "cid": None,
        }
        # 未识别（cls_src=unknown，或 class_id 空）：训练本来就会丢掉，单独列一块给人工挑
        if not cid or r.get("cls_src") == "unknown":
            unknown.append(m)
        else:
            m["cid"] = int(cid)
            by_cid.setdefault(int(cid), []).append(m)

    classes = []
    for cid in sorted(by_cid):
        ms = sorted(by_cid[cid], key=lambda m: (m["fi"], m["row"], m["col"]))
        classes.append({"key": str(cid), "cid": cid, "entry": refs.get(cid, ""), "n": len(ms),
                        "src": kind_counts(ms), "members": ms})
    if unknown:
        unknown.sort(key=lambda m: (m["fi"], m["row"], m["col"]))
        classes.append({"key": "u", "cid": None, "entry": "", "n": len(unknown),
                        "src": kind_counts(unknown), "members": unknown})

    data = {
        "dataset": dataset, "root": os.path.abspath(dataset),
        "tiles": rel(os.path.join(catalog, "tiles")) + "/",
        "refs": refs, "frames": frames, "classes": classes, "total": len(rows),
        "serve": bool(serve), "purge": bool(purge),
    }
    with open(html, "w") as f:
        f.write(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    if not quiet:
        n_unk = len(unknown)
        print(f"  去重界面: {len(classes)} 类 / {len(rows)} 张"
              f"（{'未识别 ' + str(n_unk) + ' 张，' if n_unk else ''}{len(frames)} 帧）→ {html}")
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="按 class_id 铺开数据集，人工挑删重复/低价值图")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--html", default=os.path.join(_HERE, "review.html"))
    ap.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    ap.add_argument("--quiet", action="store_true", help="不打印统计")
    args = ap.parse_args()
    p = build_review(args.dataset, args.catalog, args.html, quiet=args.quiet)
    print(f"  打开 {p}：点图标记删除 → 导出 deletions.json → "
          f"python apply_deletions.py --deletions deletions.json --inplace")
    if args.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(p))


if __name__ == "__main__":
    main()
