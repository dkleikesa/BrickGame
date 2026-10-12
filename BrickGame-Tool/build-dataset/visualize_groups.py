#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截图格 → class_id 的标注界面。

一行 = 同一帧里"相同的几张图"（帧内聚类组，一步消两块所以通常 2~4 张）：
  第 1 列 待标注的图   ：这一组的所有截图格并排，每张下面写它自己当前的 class_id
  第 2 列 自动识别的类别：组内多数 class_id + 对应参考图 + 一致性（3/4 一致 = 有一张识别得不一样）
  第 3 列 近似图      ：组代表（组内特征均值）的 12 个最近邻候选，**点一下整组指派**
点第 1 列的单张图 → 右侧面板：这一张的候选 / 搜索全量 class_id / 保持 / 标未知（单张粒度）。

标注存浏览器 localStorage；"导出标注 JSON"下载后由 apply_corrections.py 写回 index.csv。

**增量**：默认只铺出 frames.csv 里 status != done 的帧（没有台账则退回全量）。
所以每批新帧标完、apply_corrections 写了台账后，下次界面里就只剩新帧 —— 不用在旧格子里翻。
用法：
  .venv/bin/python visualize_groups.py             # 生成 annotate.html（只待标注帧）
  .venv/bin/python visualize_groups.py --all       # 全量铺出（复查旧帧时用）
  .venv/bin/python visualize_groups.py --thr 0.30  # 低置信阈值（页面上还能改）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np


def hue_hist(path: str, bins: int = 24) -> np.ndarray:
    """图标的色相直方图（只统计高饱和像素 = 图标本体，避开底板状态色）。

    为什么单靠 embedding 不够：resnet18 是 ImageNet 特征，偏形状/纹理，对"同样是圆润物体、
    颜色不同"的两款（橙色麻袋 vs 粉色水壶）拉得不够开。实测这两者 embedding 距离 0.15~0.21，
    而色相距离 0.83 —— 加上颜色判据后这类混入直接消失。
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.ones(bins) / bins
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = hsv[:, :, 1] > 70
    if m.sum() < 20:
        return np.ones(bins) / bins
    h = np.bincount(hsv[:, :, 0][m].astype(np.int32) * bins // 180, minlength=bins)
    return h / h.sum()


def hue_matrix(hists: list[np.ndarray]) -> np.ndarray:
    """色相直方图的 L1/2 距离（0 = 一样，1 = 完全不同的颜色构成）。"""
    H = np.stack(hists)
    n = len(H)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        D[i] = np.abs(H - H[i]).sum(1) / 2.0
    return D

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # 项目根（build-catalog 在那边）
sys.path.insert(0, os.path.dirname(_HERE))   # 项目根：tile_recognizer / build_catalog
sys.path.insert(0, _HERE)                    # 本目录：同流水线的兄弟模块
import tile_recognizer as T  # noqa: E402
from auto_classify import cache_dir_of, pick_emb  # noqa: E402


def complete_linkage(D, thr: float) -> list[int]:
    """按"组内两两都足够像"分组（complete-linkage），不用游戏规则的偶数约束。

    为什么不能直接用 build_dataset 的帧内聚类分组：那套聚类为了满足"一步消两块 → 每类数量
    必须是偶数"，在 cluster_by_parity / merge_similar 里会一路合并到奇数簇清零。实测因此
    把毫不相干的两种图并进同一组（诊断里出现过组内两两距离 1.000 的配对）。
    这里只认距离：合并后**簇内最大两两距离**仍 ≤ thr 才合并 —— 保证同一行里图肯定是一类。
    """
    n = len(D)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu = np.triu_indices(n, 1)
    for k in np.argsort(D[iu]):
        a, b = int(iu[0][k]), int(iu[1][k])
        if D[a, b] > thr:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if D[np.ix_(members[ra], members[rb])].max() > thr:   # 合并后会超阈值 → 不并
            continue
        parent[ra] = rb
        members[rb] = members[rb] + members[ra]
        del members[ra]

    labels, roots = [0] * n, {}
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots) + 1
        labels[i] = roots[r]
    return labels

HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>截图格 class_id 标注</title>
<style>
*{box-sizing:border-box}
:root{--hdr:44px}                                  /* 顶栏实测高度，表头靠它定位，避免遮住第一行 */
/* 整体按 0.9 缩放：字体、图片、间距统一缩一档 */
body{margin:0;font-family:-apple-system,'PingFang SC',sans-serif;background:#f5f6f8;color:#1f2328;font-size:12px}
header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #e3e5e8;padding:7px 13px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
header b{font-size:13px}
button,select,input{font:inherit;padding:3px 8px;border:1px solid #d0d5da;background:#fff;border-radius:5px}
button{cursor:pointer}button:hover{background:#f0f2f5}
button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
button.danger{color:#b42318;border-color:#f0c9c5}
.spacer{flex:1}
.stat{color:#5b6169;font-variant-numeric:tabular-nums}
#wrap{padding:11px}
/* 这里**不能**加 overflow（哪怕只是 overflow-x:auto）：那会让 #list 变成滚动容器，
   sticky 的偏移量就以它的内容盒为基准而不是视口 —— 表头会被整体下移一个 --hdr 的距离，
   正好压在帧标题行上面（实测 th 从 53 被挪到 95，把 frow 的下半行盖住）。
   表格本身是 width:100% + table-layout:fixed，不会横向溢出；长内容由 .memwrap 自己横滚。 */
#list{width:100%}
#mask{position:fixed;inset:0;background:rgba(20,22,26,.35);z-index:40;display:none}
#mask.show{display:block}
#panel{position:fixed;top:0;right:0;height:100vh;width:407px;max-width:94vw;background:#fff;z-index:50;
  overflow:auto;padding:13px;box-shadow:-3px 0 14px rgba(0,0,0,.14);transform:translateX(102%);transition:transform .16s ease-out}
#panel.show{transform:none}
#panel .close{position:absolute;top:9px;right:11px}
.fhead{font-weight:600;font-size:13px;margin:14px 0 5px}
.fhead:first-child{margin-top:0}
.fhead span{font-weight:400;color:#8b8f96;font-size:11px}
/* 严格按比例铺满：内容再宽也不会把表格撑开 */
table{border-collapse:separate;border-spacing:0;width:100%;table-layout:fixed}
th,td{padding:5px 7px;border-bottom:1px solid #eef0f2;vertical-align:middle;text-align:left;background:#fff}
th{color:#8b8f96;font-size:10px;font-weight:400;position:sticky;top:var(--hdr);z-index:5;border-bottom:1px solid #e3e5e8;white-space:nowrap;box-shadow:0 1px 0 #e3e5e8}
tr.frow td{background:#eef1f4;font-weight:600;font-size:12px;padding:7px;border-bottom:1px solid #e3e5e8}
tr.frow span{font-weight:400;color:#8b8f96;font-size:10px}
/* "本帧标完"：靠右，点了变绿；不过滤掉本帧（可当场撤销） */
.fdone{float:right;font-size:11px;padding:2px 9px}
tr.frow.done td{background:#f0f9f2}
td.c1,th.c1{width:40%;white-space:nowrap}
td.c2,th.c2{width:10%}
td.c3,th.c3{width:50%}
.candwrap{display:flex;flex-wrap:wrap;gap:4px}
.memwrap{max-width:504px;overflow-x:auto;white-space:nowrap;padding-bottom:4px}
tr:hover td{background:#f7f9ff}
tr.sel td{background:#eef4ff}
tr.split td{background:#fffaf0}
.mem{display:inline-block;text-align:center;margin-right:5px;cursor:pointer}
.mem img{width:58px;height:58px;image-rendering:pixelated;border-radius:5px;background:#eee;display:block;border:2px solid transparent}
.mem:hover img{border-color:#1f6feb}
.mem.man img{border-color:#2da44e}
.mem.unk img{opacity:.35;border-color:#c9ced4}
.cid{font-weight:600;font-variant-numeric:tabular-nums}
.cid.aut{color:#8250df}.cid.man{color:#1a7f37}.cid.non{color:#b54708}.cid.emp{color:#6b7280}
.tag{font-size:9px;color:#8b8f96;line-height:1.35;min-height:12px}
.cellimg{width:58px;height:58px;image-rendering:pixelated;border-radius:5px;background:#eee;display:block;border:1px solid #e4e7ea}
.cand{display:inline-block;text-align:center;margin-right:4px;cursor:pointer;padding:2px;border-radius:6px;border:2px solid transparent}
.cand:hover{border-color:#1f6feb;background:#f4f7ff}
.cand.cur{border-color:#2da44e}
.cand img{width:50px;height:50px;image-rendering:pixelated;border-radius:4px;background:#eee;display:block}
.grid{display:flex;flex-wrap:wrap;gap:5px}
.tile{text-align:center;cursor:pointer;padding:2px;border-radius:6px;border:2px solid transparent}
.tile:hover{border-color:#1f6feb;background:#f4f7ff}
.tile.cur{border-color:#2da44e}
.tile img{width:52px;height:52px;image-rendering:pixelated;border-radius:5px;background:#eee;display:block}
.sec{color:#8b8f96;font-size:10px;margin:11px 0 5px}
.acts{display:flex;gap:7px;margin-top:13px;flex-wrap:wrap}
</style>
</head>
<body>
<header>
  <b>截图格 class_id 标注</b>
  <span id="mdl" style="background:#eef4ff;border:1px solid #c7d9ff;color:#1f6feb;border-radius:5px;padding:2px 7px;font-weight:600"></span>
  <select id="fs"></select>
  <button id="low">只看低置信</button>
  <button id="man">只看已改</button>
  <input id="thr" type="number" step="0.05" value="__THR__" style="width:62px" title="低置信阈值(距离)">
  <span class="stat" id="stat"></span>
  <span class="spacer"></span>
  <button id="exp">导出标注 JSON</button>
  <button id="clr" class="danger">清空标注</button>
</header>
<div id="wrap">
  <div id="list"></div>
</div>
<div id="mask"></div>
<div id="panel"><button class="close" id="pclose">关闭</button><div id="pbody"></div></div>
<script>
const DATA = __DATA__;
const KEY = 'bgcls:' + DATA.dataset;
const LS = (() => { try { localStorage.setItem('_t','1'); localStorage.removeItem('_t'); return localStorage; } catch(e) { return null; } })();
// S.done = 本轮"本帧标完"的帧；S.edits = 逐格改动。两者都会写进导出的 JSON。
let S = (LS && JSON.parse(LS.getItem(KEY) || 'null')) || {edits: {}};
if (!S.edits) S.edits = {};
if (!S.done) S.done = {};
let THR = parseFloat(document.getElementById('thr').value) || 0.3;
let SEL = null, SCOPE = 'one';   // 抽屉里的改动范围：one = 当前这张，g = 整组
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ENTRY = {}; DATA.refs.forEach(([cid, entry]) => ENTRY[String(cid)] = entry);
const refSrc = cid => { const e = ENTRY[String(cid)]; return e ? DATA.tiles + e + '.png' : ''; };
const absPath = p => DATA.root ? DATA.root + '/' + p : p;   // index.csv 里的相对路径 → 绝对路径
const save = () => { if (LS) LS.setItem(KEY, JSON.stringify(S)); render(); renderPanel(); stats(); };

// 复制文本：优先 Clipboard API；file:// 下老浏览器没有它，退回 textarea + execCommand。
// 成功后在按钮上闪一下"已复制"，不然点了没有任何反馈。
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
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    let done = false;
    try { done = document.execCommand('copy'); } catch (e) { done = false; }
    ta.remove();
    if (done) ok(); else prompt('自动复制失败，请手动复制：', text);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(ok, fallback);
  } else fallback();
}

function mview(m) {                                  // 单张图最终显示什么
  const e = S.edits[m.path];
  if (e && e.unknown) return {cid: '未知', kind: 'non', cls: 'unk', src: ''};
  if (e) return {cid: String(e.class_id), kind: 'man', cls: 'man', src: refSrc(e.class_id)};
  if (m.emp || m.cid === 0) return {cid: '背景 0', kind: 'non', cls: 'emp', src: refSrc(0), d: m.d};
  if (!m.cid) return {cid: '未识别', kind: 'non', cls: '', src: ''};
  return {cid: String(m.cid), kind: 'aut', cls: '', src: refSrc(m.cid), d: m.d};
}
function gview(g) {                                  // 这一组：多数 class_id + 一致性
  // 背景在 mview 里显示成文字"背景 0"，这里先还原成类别 0 再统计 ——
  // 否则整组背景会因为"背景 0"不匹配 /^\d+$/ 而落到下面的"未识别"分支：
  // 第 2 列恒显示"未识别"、参考图空白（成员明明写着"背景 0"）。
  const cidOf = v => v.cid === '背景 0' ? '0' : v.cid;
  const cids = g.members.map(m => cidOf(mview(m))).filter(v => /^\d+$/.test(v));
  if (!cids.length) return {cid: '未识别', kind: 'non', same: 0, n: 0, src: ''};
  const cnt = {}; cids.forEach(c => cnt[c] = (cnt[c] || 0) + 1);
  const best = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a] || a - b)[0];
  const isBg = best === '0' || g.members.every(m => m.emp);
  // 背景组用灰色(non)显示，别混进"自动识别(aut)"的紫色，一眼看出这不是要认的物品
  const kind = isBg ? 'non' : (S.edits[g.members[0].path] ? 'man' : 'aut');
  return {cid: isBg ? '背景 0' : best, kind,
          same: cnt[best], n: g.members.length, src: refSrc(best),
          split: cids.length === g.members.length && cnt[best] !== g.members.length};
}

function memberHtml(m) {
  const v = mview(m);
  return `<div class="mem ${v.cls}" data-path="${m.path}" title="点这里只改这一张">
    <img src="${m.src}" loading="lazy">
    <div class="tag"><span class="cid ${v.kind}">${v.cid}</span></div>
  </div>`;
}
function candHtml(cid, d, cur) {
  return `<div class="cand${cur === String(cid) ? ' cur' : ''}" data-cid="${cid}" title="点一下把整组指派成 ${cid}">
    <img src="${refSrc(cid)}" loading="lazy"><div class="tag">${cid}</div></div>`;
}

const BYKEY = {};
const P2G = {};
DATA.frames.forEach(f => f.groups.forEach(g => {
  BYKEY[g.key] = g;
  g.members.forEach(m => P2G[m.path] = g);
}));

function rowsHtml() {
  const sel = document.getElementById('fs').value;
  const onlyLow = document.getElementById('low').classList.contains('on');
  const onlyMan = document.getElementById('man').classList.contains('on');
  // 整页只有一个 table：表头只粘一次，帧标题作为表内一行 —— 之前每帧一个 table，
  // 各自的 sticky 表头滚动时会互相叠加，把第一行盖住
  let body = '';
  for (const f of DATA.frames) {
    if (sel && sel !== f.frame) continue;
    const groups = f.groups.filter(g => {
      const anyEdit = g.members.some(m => S.edits[m.path]);
      if (onlyMan && !anyEdit) return false;
      // 背景格不算低置信（没有距离可比），否则"只看低置信"会把整页背景全捞出来
      if (onlyLow) return anyEdit || g.members.some(m =>
        !(m.emp || m.cid === 0) && (!m.cid || (m.d || 0) > THR));
      return true;
    });
    if (!groups.length) continue;
    const done = !!S.done[f.frame];
    body += `<tr class="frow${done ? ' done' : ''}"><td colspan="3">`
      + `<button class="fdone${done ? ' on' : ''}" data-frame="${esc(f.frame)}">`
      + `${done ? '已标完 ✓' : '本帧标完'}</button>`
      + `${esc(f.frame)}<span> 棋盘 ${f.rows}x${f.cols}，${f.cells} 格 / ${f.groups.length} 组，显示 ${groups.length} 组</span></td></tr>`
      + groups.map(g => {
        const v = gview(g);
        return `<tr data-key="${esc(g.key)}" class="${v.split ? 'split' : ''}${SEL && SEL.key === g.key ? ' sel' : ''}">
          <td class="c1"><div class="memwrap">${g.members.map(memberHtml).join('')}</div></td>
          <td class="c2"><div class="mem" style="margin:0" data-path="${g.members[0].path}">
              <img class="cellimg" src="${v.src}" loading="lazy">
              <div class="tag"><span class="cid ${v.kind}">${v.cid}</span><br>${v.same} 张同号</div>
            </div></td>
          <td class="c3"><div class="candwrap">${g.cands.map(([cid, d]) => candHtml(cid, d, v.cid)).join('')}</div></td>
        </tr>`;
      }).join('');
  }
  if (!body) return DATA.frames.length
    ? '<div class="sec">没有符合筛选条件的组。</div>'
    : '<div class="sec">没有待标注的帧（都标完了）。加一批新截图重跑 build_dataset.py 即可。</div>';
  return `<table><tr>
      <th class="c1">① 待标注的图</th>
      <th class="c2">② 自动类别</th>
      <th class="c3">③ 近似图（点一下＝整组指派）</th>
    </tr>${body}</table>`;
}

function render() { document.getElementById('list').innerHTML = rowsHtml(); }

function tileHtml(cid, d, cur, extra) {
  return `<div class="tile${cur === String(cid) ? ' cur' : ''}" data-cid="${cid}">
    <img src="${refSrc(cid)}" loading="lazy"><div class="tag">${cid}${extra ? '<br>' + extra : ''}</div></div>`;
}

const HELP = '<div class="sec">怎么看 / 怎么标</div>'
  + '<div class="tag" style="line-height:1.8;font-size:11.5px">'
  + '① <b>待标注的图</b>：游戏截图裁出来的格子，这一帧里相同的几张排在一起<br>'
  + '② <b>自动类别</b>：我识别出的 class_id + 图鉴参考图，x/y 一致 = 这组里有几张被识别成同一个<br>'
  + '③ <b>近似图</b>：与这组最像的 12 个图鉴物品，<b>点一下 = 整组指派</b><br><br>'
  + '点第 1 列的单张图 → 抽屉里选"改动范围"（这一张 / 整组），再点图鉴条目即可指派<br>'
  + '顶栏：<b>只看低置信</b>（d &gt; 阈值）、<b>只看已改</b>、阈值可调<br>'
  + '一帧看完点它标题行右侧的<b>本帧标完</b>；标完点右上角<b>导出标注 JSON</b>，'
  + '再跑 apply_corrections.py 写回（写回的帧下次不再出现，界面只剩新帧）'
  + '</div>';

const openPanel = () => {
  document.getElementById('mask').classList.add('show');
  document.getElementById('panel').classList.add('show');
};
const closePanel = () => {
  document.getElementById('mask').classList.remove('show');
  document.getElementById('panel').classList.remove('show');
};

function renderPanel() {
  const p = document.getElementById('pbody');
  if (!SEL) { p.innerHTML = HELP; return; }
  const m = SEL, v = mview(m), e = S.edits[m.path];
  const g = P2G[m.path] || {members: [m]};
  const targets = () => SCOPE === 'g' ? g.members : [m];
  p.innerHTML = `
    <div><img src="${m.src}" style="width:158px;height:158px;image-rendering:pixelated;border-radius:8px;background:#eee">
      <div class="tag" style="margin-top:4px">${esc(m.path)} · 帧 ${esc(m.frame)} · 第 ${m.row} 行 ${m.col} 列
        ${m.raw ? ` · <a href="${m.raw}" target="_blank">查看原始截图</a>` : ''}</div></div>
    <div class="acts" style="margin-top:10px">
      <span class="sec" style="margin:0">改动范围</span>
      <button id="sc1" class="${SCOPE !== 'g' ? 'on' : ''}">这一张</button>
      <button id="scg" class="${SCOPE === 'g' ? 'on' : ''}">整组（${g.members.length} 张）</button>
    </div>
    <div class="acts" style="margin-top:6px">
      <button id="keep">保持判定</button>
      <button id="unk">标为未知</button>
      <button id="cp" title="复制这张图的绝对路径：${esc(absPath(m.path))}">复制图片路径</button>
    </div>
    <div class="sec">点下面的图鉴条目 → 指派给${SCOPE === 'g' ? '整组' : '这一张'}（共 ${DATA.refs.length} 个，可按编号或 entry 搜）</div>
    <input id="q" placeholder="如 1428 / p010" style="width:100%">
    <div class="grid" id="all" style="margin-top:8px"></div>`;
  const one = cid => {
    targets().forEach(x => S.edits[x.path] = {class_id: cid, unknown: false});
    save(); closePanel();
  };
  p.querySelector('#sc1').onclick = () => { SCOPE = 'one'; renderPanel(); };
  p.querySelector('#scg').onclick = () => { SCOPE = 'g'; renderPanel(); };
  p.querySelector('#keep').onclick = () => {
    targets().forEach(x => delete S.edits[x.path]); save(); closePanel();
  };
  p.querySelector('#unk').onclick = () => {
    targets().forEach(x => S.edits[x.path] = {unknown: true}); save(); closePanel();
  };
  // 复制路径：不关抽屉、不改标注，只把这张图的绝对路径写进剪贴板
  p.querySelector('#cp').onclick = ev => copyText(absPath(m.path), ev.currentTarget);
  const q = p.querySelector('#q'), box = p.querySelector('#all');
  // 一次性全量渲染（2368 个），图片走 loading=lazy，滚动到哪加载到哪
  const paint = list => {
    box.innerHTML = list.map(([cid, entry]) => tileHtml(cid, undefined, v.cid, entry)).join('')
      || '<div class="tag">没有匹配</div>';
    box.querySelectorAll('.tile').forEach(t => t.onclick = () => one(+t.dataset.cid));
  };
  paint(DATA.refs);
  q.oninput = () => {
    const s = q.value.trim().toLowerCase();
    paint(s ? DATA.refs.filter(([cid, entry]) =>
      String(cid).includes(s) || entry.toLowerCase().includes(s)) : DATA.refs);
  };
}

function stats() {
  let man = 0, unk = 0, low = 0, total = 0;
  for (const f of DATA.frames) for (const g of f.groups) for (const m of g.members) {
    total++;
    const e = S.edits[m.path];
    if (e && e.unknown) unk++; else if (e) man++;
    // 背景格（class_id=0 / cls_src=empty）不算"低置信"：它们本来就没有距离，
    // 算进去会让"低置信"虚高一份背景（202 格），真正要看的疑难格子被淹掉
    else if (!(m.emp || m.cid === 0) && (!m.cid || (m.d || 0) > THR)) low++;
  }
  const doneN = DATA.frames.filter(f => S.done[f.frame]).length;
  document.getElementById('stat').textContent =
    `已改 ${man} / 未知 ${unk} / 低置信 ${low} / 共 ${total} 张 · 本帧标完 ${doneN}/${DATA.frames.length} 帧`;
}

document.getElementById('list').addEventListener('click', ev => {
  const fd = ev.target.closest('.fdone');
  if (fd) {                                     // 帧标题行：标记本帧标完（再点撤销）
    const fr = fd.dataset.frame;
    if (S.done[fr]) delete S.done[fr]; else S.done[fr] = 1;
    save(); return;
  }
  const cand = ev.target.closest('.cand');
  if (cand) {                                   // 第 3 列：整组指派
    const tr = cand.closest('tr'), g = BYKEY[tr.dataset.key];
    g.members.forEach(m => S.edits[m.path] = {class_id: +cand.dataset.cid, unknown: false});
    save(); return;
  }
  const mem = ev.target.closest('.mem');
  if (!mem) return;
  const path = mem.dataset.path;
  for (const f of DATA.frames) for (const g of f.groups) for (const m of g.members) {
    if (m.path === path) { SEL = m; render(); renderPanel(); openPanel(); return; }
  }
});
document.getElementById('fs').onchange = render;
document.getElementById('pclose').onclick = closePanel;
document.getElementById('mask').onclick = closePanel;
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePanel(); });
['low', 'man'].forEach(id => document.getElementById(id).onclick = e => { e.target.classList.toggle('on'); render(); });
document.getElementById('thr').onchange = e => { THR = parseFloat(e.target.value) || 0.3; render(); stats(); };
document.getElementById('clr').onclick = () => {
  if (!confirm('清空所有标注？（已导出的 JSON 不受影响）')) return;
  S = {edits: {}, done: {}}; save();
};
document.getElementById('exp').onclick = () => {
  const out = {dataset: DATA.dataset, created: new Date().toISOString(),
    // frames：本帧标完的帧 —— apply_corrections 据此把 frames.csv 的 status 写 done，
    // 下次生成界面时这些帧就不再出现（增量的闭环）。
    frames: Object.keys(S.done).filter(f => S.done[f]),
    corrections: Object.entries(S.edits).map(([path, e]) => e.unknown
      ? {path, unknown: true} : {path, class_id: e.class_id})};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'}));
  a.download = 'corrections.json'; a.click();
};
document.getElementById('fs').innerHTML = '<option value="">全部帧</option>' +
  DATA.frames.map(f => `<option value="${f.frame}">${esc(f.frame)}</option>`).join('');
// 页头标明这套标签是哪个模型打出来的（多模型对比时靠它区分）
const mdl = document.getElementById('mdl');
mdl.textContent = '识别模型：' + DATA.model;
mdl.title = '特征缓存：cache/' + DATA.emb;
// 表头的粘性偏移 = 顶栏实测高度。
// 必须在 render()/stats() 之后量：顶栏里的下拉框选项、统计文字都会改变它的高度；
// 量早了 --hdr 就会偏小，表头被顶栏压住（这也是之前反复出问题的原因之一）。
function setHdr() {
  document.documentElement.style.setProperty(
    '--hdr', document.querySelector('header').offsetHeight + 'px');
}
render(); stats(); setHdr();
window.addEventListener('resize', setHdr);
window.addEventListener('load', setHdr);
if (window.ResizeObserver) new ResizeObserver(setHdr).observe(document.querySelector('header'));
</script>
</body>
</html>
"""


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def pending_frames_of(dataset: str) -> set[str] | None:
    """台账里还没标完的帧（status != done）。没有台账时返回 None = "无从判断，全部当待标注"。

    为什么要这一步：标注界面是**增量**的入口 —— 只该铺出还没标完的帧。
    否则每加一批新帧，旧帧都会随全量重铺一遍，人工就得在已经标过的格子里重新找新格子。
    """
    p = os.path.join(dataset, "frames.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {r["frame"] for r in csv.DictReader(f) if r.get("status") != "done"}


def sample_file_of(samples: str, frame: str) -> str:
    """帧名 → 原始截图文件名（帧名就是原名去掉扩展名，所以直接按扩展名找）。"""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        if os.path.exists(os.path.join(samples, frame + ext)):
            return frame + ext
    return ""


def build_html(dataset: str = "dataset", catalog: str = "catalog", samples: str = "samples",
               html: str = "annotate.html", thr: float = 0.30, top: int = 12, emb: str = "",
               same_thr: float = 0.10, color_thr: float = 0.35, quiet: bool = False,
               cache: str = "", only_pending: bool = True) -> str:
    """生成标注界面，返回 html 路径。

    only_pending=True（默认）时只铺出 frames.csv 里 status != done 的帧 —— 这就是"标注支持增量"：
    每批新帧标完并写回台账后，下一次界面里只剩新帧。没有台账（老数据集）则退回全量。
    """
    cache = cache or cache_dir_of(dataset)
    rows = load_rows(os.path.join(dataset, "index.csv"))
    if only_pending:
        pend = pending_frames_of(dataset)
        if pend is not None:
            n_all = len(rows)
            rows = [r for r in rows if r["frame"] in pend]
            if not quiet:
                print(f"  只显示待标注帧：{n_all} → {len(rows)} 格"
                      f"（{len(pend)} 帧未标完；要全量加 --all）")
    frame_rows = rows
    refs = load_rows(os.path.join(catalog, "index.csv"))
    base = os.path.dirname(os.path.abspath(html))
    rel = lambda p: os.path.relpath(os.path.abspath(p), base).replace(os.sep, "/")  # noqa: E731

    cands: dict[str, list] = {}
    auto_path = os.path.join(cache, "auto_labels.csv")
    if os.path.exists(auto_path):
        with open(auto_path) as f:
            rd = csv.reader(f)
            n_top = max(1, (len(next(rd, [])) - 1) // 2)        # 首列是 path
            for r in rd:
                if r and r[0]:
                    cands[r[0]] = [[int(r[1 + i * 2]), float(r[2 + i * 2])]
                                   for i in range(n_top) if r[1 + i * 2]]

    # 分组：逐帧按"预训练特征的余弦距离 + 色相距离"做 complete-linkage —— 组内任意两张都要达标。
    # 不再依赖识别号（识别可能错），也不再带偶数约束（那会把不同物品硬并到一起）。
    emb_path = emb or pick_emb(cache)
    if not emb_path or not os.path.exists(emb_path):
        raise SystemExit(f"缺少特征缓存，先跑：.venv/bin/python embed_features.py "
                         f"--dataset {dataset} --catalog {catalog}")
    z = np.load(emb_path)
    emb_of = dict(zip(z["cell_path"].tolist(), z["cell"]))
    emb_ref, ref_class_of = z["ref"], z["ref_class"].tolist()

    by_frame: dict[str, list[dict]] = {}
    for r in frame_rows:
        by_frame.setdefault(r["frame"], []).append(r)
    groups: dict[str, list] = {}
    group_dev, group_keys = [], []
    for fr in sorted(by_frame):
        ms = sorted(by_frame[fr], key=lambda r: r["path"])
        dev = np.stack([emb_of[r["path"]] for r in ms])     # 已 L2 归一化
        D = np.clip(1.0 - dev @ dev.T, 0.0, 2.0)
        H = hue_matrix([hue_hist(os.path.join(dataset, r["path"])) for r in ms])
        # 两个判据都要过：各自除以自己的阈值后取 max，complete_linkage 阈值固定为 1.0
        Dn = np.maximum(D / same_thr, H / color_thr)
        lab = complete_linkage(Dn, 1.0)
        sf = sample_file_of(samples, fr)
        raw = rel(os.path.join(samples, sf)) if sf else ""
        for gi in sorted(set(lab)):
            idx = [i for i, v in enumerate(lab) if v == gi]
            dmax = float(D[np.ix_(idx, idx)].max()) if len(idx) > 1 else 0.0
            key = f"{fr}|{gi}"
            groups.setdefault(fr, []).append({
                "key": key, "gid": gi, "n": len(idx), "dmax": round(dmax, 3),
                "members": [{
                    "path": ms[i]["path"], "src": rel(os.path.join(dataset, ms[i]["path"])),
                    "frame": fr, "row": int(os.path.basename(ms[i]["path"])[1:3]),
                    "col": int(os.path.basename(ms[i]["path"])[4:6]), "raw": raw,
                    "cid": int(ms[i]["class_id"]) if ms[i].get("class_id") else 0,
                    "emp": 1 if ms[i].get("cls_src") == "empty" else 0,
                    "d": float(ms[i]["cls_dist"]) if ms[i].get("cls_dist") else 0.0,
                } for i in idx],
            })
            group_keys.append(key)
            group_dev.append(dev[idx].mean(0))
    # 组代表 = 成员特征的均值（再归一化），和全部图鉴图算余弦距离取 top-N
    if not group_dev:                                      # 没有待标注帧（增量常态）：空页面
        groups = {}
    else:
        rep = np.stack(group_dev)
        rep = rep / (np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8)
        Dc = np.clip(1.0 - rep @ emb_ref.T, 0.0, 2.0)
        order = np.argsort(Dc, axis=1)[:, :top]
        gi_of = {k: i for i, k in enumerate(group_keys)}
        for fr in groups:
            for g in groups[fr]:
                i = gi_of[g["key"]]
                g["cands"] = [[int(ref_class_of[j]), round(float(Dc[i, j]), 3)] for j in order[i]]
    print(f"按 embedding+色相 分组: {len(group_keys)} 组 / {len(frame_rows)} 张，"
          f"阈值 embedding≤{same_thr} 且 色相≤{color_thr}")
    # 棋盘行列直接由格子里最大行列推出（文件名 r03c05 = 第3行第5列），不再需要 frames.csv
    frames = []
    for fr in sorted(groups):
        cells = [m for g in groups[fr] for m in g["members"]]
        frames.append({
            "frame": fr,
            "rows": max(c["row"] for c in cells) + 1,
            "cols": max(c["col"] for c in cells) + 1,
            "cells": sum(g["n"] for g in groups[fr]), "groups": groups[fr]})
    data = {
        "dataset": dataset,
        "root": os.path.abspath(dataset),          # 抽屉里"复制图片路径"给的是绝对路径
        # 页头显示"这套标签是用哪个模型打出来的"——多模型对比时不用猜
        "model": str(z["backbone"]) if "backbone" in z else os.path.basename(emb_path),
        "emb": os.path.basename(emb_path),
        "tiles": rel(os.path.join(catalog, "tiles")) + "/",
        "refs": [[int(r["class_id"]), r["entry"]] for r in refs],
        "frames": frames, "cands": cands,
    }
    with open(html, "w") as f:
        f.write(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
                    .replace("__THR__", str(thr)))
    n_auto = sum(1 for r in frame_rows if r.get("cls_src") == "auto")
    n_bg = sum(1 for r in frame_rows if r.get("cls_src") == "empty")
    if not quiet:
        print(f"  标注界面: 砖块 {n_auto} + 背景 {n_bg} = {len(frame_rows)} 张 → {html}")
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="截图格 class_id 标注界面")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--catalog", default=os.path.join(_ROOT, "build-catalog", "catalog"))
    ap.add_argument("--samples", default=os.path.join(_HERE, "samples"), help="原始截图目录（'查看原始截图'用）")
    ap.add_argument("--html", default=os.path.join(_HERE, "annotate.html"))
    ap.add_argument("--thr", type=float, default=0.30, help="低置信阈值（距离），页面上可改")
    ap.add_argument("--top", type=int, default=12, help="每组显示几个近似图")
    ap.add_argument("--emb", default="", help="预训练特征缓存（默认自动找 cache/emb_*.npz）")
    ap.add_argument("--cache", default="", help="中间产物目录（默认 dataset 同级的 cache/）")
    ap.add_argument("--same-thr", type=float, default=0.10,
                    help="同款阈值（embedding 余弦距离）：实测同款 0.005~0.04、异款 ≥0.15，"
                         "0.10 落在两簇中间的空隙里")
    ap.add_argument("--color-thr", type=float, default=0.35,
                    help="色相直方图距离阈值：同款 0.01~0.07、异款 0.8+，用于挡住同形状不同色")
    ap.add_argument("--all", action="store_true",
                    help="铺出全部帧（默认只铺 frames.csv 里 status != done 的待标注帧）")
    args = ap.parse_args()
    build_html(args.dataset, args.catalog, args.samples, args.html, args.thr, args.top,
               args.emb, args.same_thr, args.color_thr, cache=args.cache,
               only_pending=not args.all)


if __name__ == "__main__":
    main()
