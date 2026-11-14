"""Zero-dependency offline 3D web dashboard for live Rokoko / synthetic tactile scanning.

Uses only Python's standard `http.server` + an embedded, 100% offline HTML5 Canvas 3D engine
(zero CDN links -> works reliably even on an isolated Rokoko / Unitree G1 Wi-Fi router without
internet access).
"""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any
import webbrowser

from .pipeline import ScanTrackSession

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Lambda Junction — Tactile 3D Scan, Convex Hull & Unitree G1 Tracker</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #171a21;
    --card: #1e232d;
    --border: #2c3342;
    --text: #eef1f6;
    --muted: #9aa4b8;
    --accent: #ff6b2c;
    --lime: #b8ff3b;
    --cyan: #38d9f5;
    --ok: #22c55e;
    --warn: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: grid;
    grid-template-columns: 1fr 420px;
    overflow: hidden;
  }
  #viewport-wrap { position: relative; width: 100%; height: 100%; background: radial-gradient(circle at 50% 40%, #1b202b, #0c0e12); }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .hud-top {
    position: absolute; top: 14px; left: 14px; right: 14px;
    display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; pointer-events: none;
  }
  .hud-group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .pill {
    pointer-events: auto;
    background: rgba(23, 26, 33, 0.88);
    border: 1px solid var(--border);
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    backdrop-filter: blur(6px);
  }
  .pill.gesture-active { border-color: var(--lime); color: var(--lime); }
  .pill.phase-scanning { border-color: var(--cyan); color: var(--cyan); }
  .pill.phase-carrying { border-color: var(--warn); color: var(--warn); }
  .pill.phase-placed { border-color: var(--ok); color: var(--ok); }
  .pill.confirmed-90 {
    border-color: var(--lime);
    background: rgba(22, 54, 34, 0.92);
    color: var(--lime);
    box-shadow: 0 0 18px rgba(184, 255, 59, 0.28);
  }
  .hud-bottom {
    position: absolute; bottom: 14px; left: 14px; max-width: calc(100% - 310px);
    background: rgba(23, 26, 33, 0.86);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--muted);
    pointer-events: none;
  }
  .map-panel {
    position: absolute; bottom: 14px; right: 14px; width: 276px;
    background: rgba(23, 26, 33, 0.91);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 10px 11px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.42);
    display: flex; flex-direction: column; gap: 7px;
  }
  .map-panel-head {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
  }
  .map-coords { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; color: var(--cyan); font-size: 11px; text-transform: none; }
  .map-btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .map-btn-row button {
    padding: 6px 8px; font-size: 11.5px; border-radius: 6px; white-space: nowrap;
  }
  .slider-row {
    display: grid; grid-template-columns: 54px 1fr 48px; align-items: center; gap: 6px; font-size: 11px; color: #cbd5e1;
  }
  .slider-row input[type=range] {
    width: 100%; accent-color: var(--cyan); cursor: pointer; height: 5px;
  }
  .slider-val {
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 11px; text-align: right; color: var(--lime);
  }
  #sidebar {
    background: var(--panel);
    border-left: 1px solid var(--border);
    padding: 14px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 12px;
  }
  .section.section-confirmed {
    border-color: rgba(184, 255, 59, 0.65);
    box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.22);
  }
  .section h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .btn-stack { display: flex; flex-direction: column; gap: 6px; margin-bottom: 9px; }
  button, select {
    background: #262c3a;
    color: var(--text);
    border: 1px solid #394256;
    border-radius: 7px;
    padding: 7px 9px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: 0.12s ease;
  }
  button:hover, select:hover { border-color: var(--accent); background: #2e3546; }
  button:active { background: #353e52; transform: translateY(1px); }
  button.active-toggle { border-color: var(--lime); color: var(--lime); background: rgba(184, 255, 59, 0.10); }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; text-align: center; }
  .metric-box { background: #151922; border: 1px solid var(--border); border-radius: 7px; padding: 6px 4px; }
  .metric-val { font-size: 15px; font-weight: 700; color: var(--lime); font-variant-numeric: tabular-nums; }
  .metric-lbl { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .bar-row { margin-bottom: 7px; }
  .bar-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
  .bar-track { position: relative; height: 9px; background: #12151c; border-radius: 99px; overflow: hidden; }
  .bar-track::after {
    content: ''; position: absolute; left: 90%; top: 0; bottom: 0; width: 1.5px;
    background: rgba(255, 255, 255, 0.35);
  }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--cyan)); transition: width 0.15s; }
  .bar-fill.confirmed { background: linear-gradient(90deg, #10b981, var(--lime)); box-shadow: 0 0 10px rgba(184, 255, 59, 0.5); }
  .confirmed-banner {
    display: none;
    margin-bottom: 8px;
    padding: 7px 9px;
    border-radius: 7px;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(184, 255, 59, 0.55);
    color: var(--lime);
    font-size: 11.5px;
    font-weight: 700;
  }
  .advice { font-size: 12px; color: #d8dee9; margin-top: 6px; line-height: 1.35; }
  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    background: #12151c;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px;
    white-space: pre-wrap;
    color: #cbd5e1;
    max-height: 165px;
    overflow-y: auto;
  }
  .row-inline { display: flex; gap: 6px; margin-bottom: 7px; }
  .row-inline > * { flex: 1; }
</style>
</head>
<body>
  <div id="viewport-wrap">
    <canvas id="cv"></canvas>
    <div class="hud-top">
      <div class="hud-group">
        <div id="conn-pill" class="pill">Connecting...</div>
        <div id="gesture-pill" class="pill">Gesture: IDLE</div>
      </div>
      <div class="hud-group">
        <div id="confirm-pill" class="pill" style="display:none;">✓ CONFIRMED &gt;90%</div>
        <div id="phase-pill" class="pill phase-scanning">PHASE: SCANNING</div>
      </div>
    </div>

    <div class="hud-bottom">
      <b>Move X-Z Map:</b> Right-Drag / Shift+Drag / <b>[WASD / Arrows]</b> &nbsp;|&nbsp; <b>Orbit:</b> Left-Drag &nbsp;|&nbsp; <b>Height:</b> Alt+Drag / <b>[Q/E]</b> &nbsp;|&nbsp; <b>[F]</b> Center &nbsp;|&nbsp; <b>[O/H/V]</b> Object / Hull / Points<br/>
      <b>Scan Keys:</b> <b>[T]</b> Trace &nbsp;<b>[G]</b> 5-Finger Snap &nbsp;<b>[C]</b> Caliper &nbsp;<b>[Space]</b> Lock/Place &nbsp;<b>[B]</b> Calibrate Table &nbsp;<b>[R]</b> Reset
    </div>

    <div class="map-panel" id="map-panel">
      <div class="map-panel-head">
        <span>3D Space / X-Z Plane</span>
        <span class="map-coords" id="map-coords-lbl">X:+8 Z:+34 cm</span>
      </div>
      <div class="map-btn-row">
        <button onclick="centerOnActiveScene()" title="Center 3D space on scanned object or live hands [F]">Center [F]</button>
        <button id="btn-follow" onclick="toggleFollow()" title="Auto-follow live hands / object across Coil Pro offset">Follow: OFF</button>
      </div>
      <div class="slider-row" title="Move 3D map left/right along X axis">
        <span>X (L/R)</span>
        <input type="range" id="sl-x" min="-200" max="200" step="1" value="8" oninput="onSliderPan()"/>
        <span class="slider-val" id="val-x">+8 cm</span>
      </div>
      <div class="slider-row" title="Move 3D map forward/back along horizontal Z/Depth plane">
        <span>Z (Depth)</span>
        <input type="range" id="sl-z" min="-200" max="200" step="1" value="34" oninput="onSliderPan()"/>
        <span class="slider-val" id="val-z">+34 cm</span>
      </div>
      <div class="slider-row" title="Adjust vertical view height">
        <span>Height</span>
        <input type="range" id="sl-h" min="-50" max="150" step="1" value="6" oninput="onSliderPan()"/>
        <span class="slider-val" id="val-h">+6 cm</span>
      </div>
    </div>
  </div>

  <aside id="sidebar">
    <div class="section">
      <h3>Hardware / Demo Controls</h3>
      <div class="row-inline" id="synth-controls" style="display:none;">
        <select id="obj-select" onchange="setSynthObj(this.value)">
          <option value="redbull">Red Bull 250 ml Can</option>
          <option value="profeel">Valio PROfeel Carton</option>
          <option value="wood_sphere">Wooden Sphere (r=50mm)</option>
        </select>
        <button id="auto-btn" onclick="toggleAuto()">Auto Loop: ON</button>
      </div>
      <div class="row-inline" id="rokoko-controls" style="display:none;">
        <select id="mode-select" onchange="setMode(this.value)">
          <option value="planted">Smartgloves Only (Planted Wrist)</option>
          <option value="coil">Smartgloves + Coil Pro (6-DoF)</option>
        </select>
      </div>
      <div class="btn-grid">
        <button id="btn-trace" onclick="act('trace_toggle')">[T] Trace Index (👌)</button>
        <button onclick="act('snapshot')">[G] 5-Finger Snap (✊)</button>
        <button onclick="act('caliper')">[C] Caliper Pinch (🤏)</button>
        <button onclick="act('calibrate_table')">[B] Calibrate Table</button>
        <button id="btn-phase" onclick="act('phase')" style="grid-column: span 2;">[Space] Lock Grasp / Place Object (✌️)</button>
        <button onclick="act('reset')" style="grid-column: span 2;">[R] Reset / New Object</button>
      </div>
    </div>

    <div class="section">
      <h3>Live 3D Cloud &amp; Convex Hull</h3>
      <div class="metrics">
        <div class="metric-box"><div class="metric-val" id="m-pts">0</div><div class="metric-lbl">Voxels</div></div>
        <div class="metric-box"><div class="metric-val" id="m-h">0</div><div class="metric-lbl">H (mm)</div></div>
        <div class="metric-box"><div class="metric-val" id="m-lw">0&times;0</div><div class="metric-lbl">L&times;W (mm)</div></div>
        <div class="metric-box"><div class="metric-val" id="m-vol">0</div><div class="metric-lbl">Hull cm&sup3;</div></div>
      </div>
      <div class="advice" id="prim-line" style="color:var(--muted); margin-top:7px;">Primitives: waiting for points...</div>
    </div>

    <div class="section" id="catalog-section">
      <h3>Known Catalog Match &rarr; Physics (90% Gate)</h3>
      <div class="btn-stack">
        <button id="btn-obj-vis" class="active-toggle" onclick="toggleObjectVisMode()" title="Cycle predefined 3D object between Visible and See-Through [O]">Object: Visible</button>
        <button id="btn-hull-vis" onclick="toggleHullVisMode()" title="Cycle 3D Convex Hull between Visible and See-Through [H]">Hull: See-Through</button>
        <button id="btn-pts-vis" class="active-toggle" onclick="togglePointsVis()" title="Cycle tactile points between Visible and Hidden [V]">Points: Visible</button>
      </div>
      <div id="confirmed-banner" class="confirmed-banner">✓ &gt;90% CONFIRMED — 3D STRUCTURE LOCKED</div>
      <div id="catalog-bars"></div>
      <div class="advice" id="match-advice">Waiting for tactile contacts...</div>
    </div>

    <div class="section">
      <h3>Event Log</h3>
      <div id="event-log" class="mono"></div>
    </div>
  </aside>

<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let state = null;
let autoLoop = true;

// Movable 3D Camera & X-Z Map State
// Note: Workspace uses right-handed Z-up metres [X_right, Y_depth (Rokoko Z), Z_height (Rokoko Y)]
let camAz = -0.65, camEl = 0.48, camDist = 0.78;
let target = [0.08, 0.34, 0.06];
let activeDrag = null;        // null | 'orbit' | 'pan_xz' | 'pan_3d'
let lastX = 0, lastY = 0;
let followScene = false;
let hasInitialRokokoFocus = false;
let objectVisMode = 'solid';     // 'solid' ('Visible') | 'see_through' ('See-Through')
let hullVisMode = 'see_through'; // 'solid' ('Visible') | 'see_through' ('See-Through')
let showPoints = true;
let lastFrameTime = performance.now();
const keysDown = new Set();

cv.addEventListener('contextmenu', e => e.preventDefault());

cv.addEventListener('mousedown', e => {
  lastX = e.clientX;
  lastY = e.clientY;
  if (e.button === 2 || (e.button === 0 && e.shiftKey)) {
    activeDrag = 'pan_xz';
  } else if (e.button === 1 || (e.button === 0 && (e.altKey || e.ctrlKey))) {
    activeDrag = 'pan_3d';
  } else if (e.button === 0) {
    activeDrag = 'orbit';
  }
  if (activeDrag === 'pan_xz' || activeDrag === 'pan_3d') {
    setFollow(false);
  }
});

window.addEventListener('mouseup', () => { activeDrag = null; });

window.addEventListener('mousemove', e => {
  if (!activeDrag) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;

  if (activeDrag === 'orbit') {
    camAz -= dx * 0.007;
    camEl = Math.max(0.05, Math.min(1.48, camEl + dy * 0.007));
  } else if (activeDrag === 'pan_xz') {
    const basis = getCameraBasis();
    const panScale = camDist / basis.fov;
    const depthScale = panScale / Math.max(Math.sin(camEl), 0.30);
    target[0] += -dx * basis.rx * panScale + dy * basis.fhx * depthScale;
    target[1] += -dx * basis.ry * panScale + dy * basis.fhy * depthScale;
    syncSlidersFromTarget();
  } else if (activeDrag === 'pan_3d') {
    const basis = getCameraBasis();
    const panScale = camDist / basis.fov;
    target[0] += (-dx * basis.rx + dy * basis.ux) * panScale;
    target[1] += (-dx * basis.ry + dy * basis.uy) * panScale;
    target[2] += (-dx * basis.rz + dy * basis.uz) * panScale;
    syncSlidersFromTarget();
  }
});

cv.addEventListener('wheel', e => {
  e.preventDefault();
  camDist = Math.max(0.18, Math.min(3.50, camDist * (1 + e.deltaY * 0.001)));
}, { passive: false });

window.addEventListener('blur', () => keysDown.clear());

window.addEventListener('keyup', e => {
  keysDown.delete(e.key.toLowerCase());
});

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (['w','a','s','d','q','e','arrowup','arrowdown','arrowleft','arrowright','pageup','pagedown'].includes(k)) {
    e.preventDefault();
    keysDown.add(k);
    setFollow(false);
    return;
  }
  if (k === 'f') { centerOnActiveScene(); return; }
  if (k === '0' || k === 'home') { resetCameraView(); return; }
  if (k === 'o') { toggleObjectVisMode(); return; }
  if (k === 'h') { toggleHullVisMode(); return; }
  if (k === 'v') { togglePointsVis(); return; }
  if (k === 't') act('trace_toggle');
  else if (k === 'g') act('snapshot');
  else if (k === 'c') act('caliper');
  else if (k === 'b') act('calibrate_table');
  else if (k === 'r') act('reset');
  else if (k === ' ' || k === 'p') { e.preventDefault(); act('phase'); }
});

function setFollow(enabled) {
  followScene = Boolean(enabled);
  const btn = document.getElementById('btn-follow');
  btn.textContent = 'Follow: ' + (followScene ? 'ON' : 'OFF');
  btn.className = followScene ? 'active-toggle' : '';
}

function toggleFollow() {
  setFollow(!followScene);
  if (followScene) centerOnActiveScene();
}

function setObjectVisMode(mode) {
  objectVisMode = mode === 'see_through' ? 'see_through' : 'solid';
  const isSolid = objectVisMode === 'solid';
  const btn = document.getElementById('btn-obj-vis');
  btn.textContent = isSolid ? 'Object: Visible' : 'Object: See-Through';
  btn.className = isSolid ? 'active-toggle' : '';
}

function toggleObjectVisMode() {
  setObjectVisMode(objectVisMode === 'solid' ? 'see_through' : 'solid');
}

function setHullVisMode(mode) {
  hullVisMode = mode === 'see_through' ? 'see_through' : 'solid';
  const isSolid = hullVisMode === 'solid';
  const btn = document.getElementById('btn-hull-vis');
  btn.textContent = isSolid ? 'Hull: Visible' : 'Hull: See-Through';
  btn.className = isSolid ? 'active-toggle' : '';
}

function toggleHullVisMode() {
  setHullVisMode(hullVisMode === 'solid' ? 'see_through' : 'solid');
}

function setPointsVis(visible) {
  showPoints = Boolean(visible);
  const btn = document.getElementById('btn-pts-vis');
  btn.textContent = showPoints ? 'Points: Visible' : 'Points: Hidden';
  btn.className = showPoints ? 'active-toggle' : '';
}

function togglePointsVis() {
  setPointsVis(!showPoints);
}

function resetCameraView() {
  setFollow(false);
  camAz = -0.65;
  camEl = 0.48;
  camDist = 0.78;
  const zT = state?.calibration?.z_table || 0.0;
  target = [0.08, 0.34, zT + 0.06];
  syncSlidersFromTarget();
}

function getActiveSceneFocus() {
  const zT = state?.calibration?.z_table || 0.0;
  if (!state) return [0.08, 0.34, zT + 0.06];

  // 1. Prefer tracked / confirmed object pose when available
  if (state.tracked_object && state.tracked_object.pose) {
    const p = state.tracked_object.pose;
    return [p.x_m, p.y_m, p.z_center_m];
  }
  // 2. Else centroid of accumulated 3D tactile cloud
  if (state.cloud && state.cloud.points && state.cloud.points.length >= 3) {
    const pts = state.cloud.points;
    let sx = 0, sy = 0, sz = 0;
    for (const q of pts) { sx += q[0]; sy += q[1]; sz += q[2]; }
    const n = pts.length;
    return [sx / n, sy / n, sz / n];
  }
  // 3. Else live gloved hands (crucial when Coil Pro offset puts hands far to the side!)
  if (state.hands) {
    const r = state.hands.right;
    const l = state.hands.left;
    if (r && r.pads && r.pads.index) {
      const p = r.pads.index;
      return [p[0], p[1], Math.max(p[2], zT + 0.05)];
    }
    if (r && r.wrist) return [r.wrist[0], r.wrist[1], Math.max(r.wrist[2], zT + 0.05)];
    if (l && l.wrist) return [l.wrist[0], l.wrist[1], Math.max(l.wrist[2], zT + 0.05)];
  }
  return [0.08, 0.34, zT + 0.06];
}

function centerOnActiveScene() {
  const f = getActiveSceneFocus();
  target = [f[0], f[1], f[2]];
  syncSlidersFromTarget();
}

function onSliderPan() {
  setFollow(false);
  target[0] = parseFloat(document.getElementById('sl-x').value) / 100.0;
  target[1] = parseFloat(document.getElementById('sl-z').value) / 100.0;
  target[2] = parseFloat(document.getElementById('sl-h').value) / 100.0;
  updateSliderLabelsOnly();
}

function fmtCm(mVal) {
  const cm = Math.round(mVal * 100);
  return (cm >= 0 ? '+' : '') + cm + ' cm';
}

function updateSliderLabelsOnly() {
  const xCm = Math.round(target[0] * 100);
  const zCm = Math.round(target[1] * 100);
  document.getElementById('val-x').textContent = fmtCm(target[0]);
  document.getElementById('val-z').textContent = fmtCm(target[1]);
  document.getElementById('val-h').textContent = fmtCm(target[2]);
  document.getElementById('map-coords-lbl').textContent = `X:${xCm>=0?'+':''}${xCm} Z:${zCm>=0?'+':''}${zCm} cm`;
}

function syncSlidersFromTarget() {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  document.getElementById('sl-x').value = clamp(Math.round(target[0] * 100), -200, 200);
  document.getElementById('sl-z').value = clamp(Math.round(target[1] * 100), -200, 200);
  document.getElementById('sl-h').value = clamp(Math.round(target[2] * 100), -50, 150);
  updateSliderLabelsOnly();
}

async function act(action, payload={}) {
  const res = await fetch('/api/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action, payload})
  });
  state = await res.json();
  if (action === 'calibrate_table') {
    centerOnActiveScene();
  }
  updateSidebar();
}
function setSynthObj(v) { act('set_synthetic_object', {object_key: v}); }
function setMode(v) { act('set_mode', {mode: v}); }
function toggleAuto() {
  autoLoop = !autoLoop;
  document.getElementById('auto-btn').textContent = 'Auto Loop: ' + (autoLoop ? 'ON' : 'MANUAL');
  act('set_auto_cycle', {enabled: autoLoop});
}

function getCameraBasis() {
  const w = cv.width || 800, h = cv.height || 600;
  const cx = Math.cos(camAz), sx = Math.sin(camAz);
  const ce = Math.cos(camEl), se = Math.sin(camEl);
  const eye = [
    target[0] + camDist * ce * sx,
    target[1] - camDist * ce * cx,
    target[2] + camDist * se
  ];
  let fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  const fn = Math.hypot(fx, fy, fz) || 1;
  fx /= fn; fy /= fn; fz /= fn;
  let rx = fy, ry = -fx, rz = 0;
  const rn = Math.hypot(rx, ry) || 1;
  rx /= rn; ry /= rn;
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
  const fov = Math.max(Math.min(w, h) * 1.05, 220);
  return { w, h, eye, fx, fy, fz, rx, ry, rz, ux, uy, uz, fhx: -sx, fhy: cx, fov };
}

function project3DWithBasis(p, b) {
  const dx = p[0] - b.eye[0], dy = p[1] - b.eye[1], dz = p[2] - b.eye[2];
  const zCam = dx * b.fx + dy * b.fy + dz * b.fz;
  if (zCam < 0.035) return null;
  const invZ = b.fov / zCam;
  const xScreen = b.w * 0.5 + (dx * b.rx + dy * b.ry + dz * b.rz) * invZ;
  const yScreen = b.h * 0.52 - (dx * b.ux + dy * b.uy + dz * b.uz) * invZ;
  return [xScreen, yScreen, zCam];
}

function drawLine3D(a, bPt, stroke, width, dash, basis) {
  const pa = project3DWithBasis(a, basis), pb = project3DWithBasis(bPt, basis);
  if (!pa || !pb) return;
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width || 1.5;
  if (dash && dash.length) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
  ctx.restore();
}

function drawRoundedPill2D(x, y, w, h, r, fill, stroke, lineW) {
  const rr = Math.min(r, w * 0.5, h * 0.5);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineW || 1.5; ctx.stroke(); }
  ctx.restore();
}

function xformLocal(lx, ly, lz, R, c, laidOnSide) {
  // If a cylinder/box was explicitly placed on its side, rotate local Z onto horizontal Y
  const ux = lx;
  const uy = laidOnSide ? -lz : ly;
  const uz = laidOnSide ? ly : lz;
  return [
    c[0] + R[0][0] * ux + R[0][1] * uy + R[0][2] * uz,
    c[1] + R[1][0] * ux + R[1][1] * uy + R[1][2] * uz,
    c[2] + R[2][0] * ux + R[2][1] * uy + R[2][2] * uz,
  ];
}

function resolveTrackedCandidate(st) {
  if (st.tracked_object && st.tracked_object.pose) {
    return st.tracked_object;
  }
  // Fallback if user scans a custom/unlisted object whose primitive fit alone exceeds 90%
  const pBest = st.cloud?.primitive_best;
  const pWeight = pBest ? (st.cloud?.primitive_weights?.[pBest] || 0) : 0;
  const fit = pBest ? st.cloud?.primitive_fits?.[pBest] : null;
  if (pBest && fit && pWeight > 0.90 && (st.cloud?.n_voxels || 0) >= 14) {
    const zT = st.calibration?.z_table || 0.0;
    let ex = 0.06, ey = 0.06, ez = 0.06;
    if (pBest === 'sphere') {
      ex = ey = ez = 2.0 * (fit.dims_m.radius || 0.04);
    } else if (pBest === 'cylinder') {
      ex = ey = 2.0 * (fit.dims_m.radius || 0.03);
      ez = fit.dims_m.height || 0.10;
    } else {
      ex = fit.dims_m.depth || 0.06;
      ey = fit.dims_m.width || 0.06;
      ez = fit.dims_m.height || 0.10;
    }
    const yaw = fit.yaw_rad || 0;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    return {
      key: 'custom_' + pBest,
      label: 'Fitted ' + pBest.toUpperCase(),
      shape: pBest,
      material: 'unknown',
      container: false,
      sloped_top_m: 0.0,
      mass_g: 0,
      max_grip_n: 0,
      extents_m: [ex, ey, ez],
      confidence: pWeight,
      confirmed_90: true,
      pose: {
        phase: st.phase || 'scanning',
        x_m: fit.center_xy[0],
        y_m: fit.center_xy[1],
        z_center_m: zT + 0.5 * ez,
        z_bottom_m: zT,
        yaw_rad: yaw,
        yaw_deg: yaw * 180 / Math.PI,
        upright: true,
        rotation: [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]],
      }
    };
  }
  return null;
}

function drawPreviewWireframe(tracked, basis) {
  const p = tracked.pose;
  const [ex, ey, ez] = tracked.extents_m;
  const yaw = p.yaw_rad || 0;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const R = p.rotation || [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]];
  const center = [p.x_m, p.y_m, p.z_center_m];
  const stroke = 'rgba(56, 217, 245, 0.44)';

  if (tracked.shape === 'cylinder') {
    const rad = ex * 0.5, hz = ez * 0.5, N = 24;
    for (const zOff of [-hz, hz]) {
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1) / N) * Math.PI * 2;
        drawLine3D(
          xformLocal(rad * Math.cos(a0), rad * Math.sin(a0), zOff, R, center, false),
          xformLocal(rad * Math.cos(a1), rad * Math.sin(a1), zOff, R, center, false),
          stroke, 1.6, [4, 4], basis
        );
      }
    }
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      drawLine3D(
        xformLocal(rad * Math.cos(a), rad * Math.sin(a), -hz, R, center, false),
        xformLocal(rad * Math.cos(a), rad * Math.sin(a), hz, R, center, false),
        stroke, 1.3, [4, 4], basis
      );
    }
  } else if (tracked.shape === 'sphere') {
    const rad = ex * 0.5, N = 24;
    for (let ring = 0; ring < 3; ring++) {
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1) / N) * Math.PI * 2;
        const p0 = ring === 0 ? [rad*Math.cos(a0), rad*Math.sin(a0), 0]
                 : (ring === 1 ? [rad*Math.cos(a0), 0, rad*Math.sin(a0)] : [0, rad*Math.cos(a0), rad*Math.sin(a0)]);
        const p1 = ring === 0 ? [rad*Math.cos(a1), rad*Math.sin(a1), 0]
                 : (ring === 1 ? [rad*Math.cos(a1), 0, rad*Math.sin(a1)] : [0, rad*Math.cos(a1), rad*Math.sin(a1)]);
        drawLine3D(xformLocal(p0[0], p0[1], p0[2], R, center, false), xformLocal(p1[0], p1[1], p1[2], R, center, false), stroke, 1.6, [4, 4], basis);
      }
    }
  } else {
    const hx = ex * 0.5, hy = ey * 0.5, hz = ez * 0.5;
    const slope = tracked.sloped_top_m || 0;
    const corners = [
      [-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
      [-hx,-hy,hz],[hx,-hy,hz - slope],[hx,hy,hz - slope],[-hx,hy,hz]
    ].map(([lx, ly, lz]) => xformLocal(lx, ly, lz, R, center, false));
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for (const [i, j] of edges) drawLine3D(corners[i], corners[j], stroke, 1.6, [4, 4], basis);
  }
}

function buildConfirmedStructure(tracked) {
  const [ex, ey, ez] = tracked.extents_m;
  const p = tracked.pose;
  const yaw = p.yaw_rad || 0;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const R = p.rotation || [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]];
  const center = [p.x_m, p.y_m, p.z_center_m];
  const laidOnSide = (p.phase === 'placed' && p.upright === false && tracked.shape !== 'sphere');

  const tris = [];   // { v0, v1, v2, rgb } outward-oriented world triangles
  const ribs = [];   // { a, b, nrm, bold } structural CAD lines with surface normal for X-ray front/back styling

  const W = (lx, ly, lz) => xformLocal(lx, ly, lz, R, center, laidOnSide);
  const Nrm = (nx, ny, nz) => {
    const c0 = [0, 0, 0];
    return xformLocal(nx, ny, nz, R, c0, laidOnSide);
  };
  const addQuad = (p00, p10, p11, p01, rgb) => {
    tris.push({ v0: p00, v1: p10, v2: p11, rgb });
    tris.push({ v0: p00, v1: p11, v2: p01, rgb });
  };

  if (tracked.shape === 'cylinder') {
    const rad = ex * 0.5, hz = ez * 0.5;
    const isCan = tracked.key === 'redbull' || tracked.container;
    const N = 28;
    // Profile rings [zFrac (-0.5..+0.5), rScale]
    const profile = isCan
      ? [[-0.50, 0.88], [-0.44, 1.00], [0.0, 1.00], [0.39, 1.00], [0.48, 0.86], [0.50, 0.88]]
      : [[-0.50, 1.00], [0.0, 1.00], [0.50, 1.00]];

    for (let b = 0; b < profile.length - 1; b++) {
      const [zf0, rs0] = profile[b];
      const [zf1, rs1] = profile[b + 1];
      const z0 = zf0 * ez, z1 = zf1 * ez;
      const r0 = rs0 * rad, r1 = rs1 * rad;
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2;
        const a1 = ((i + 1) / N) * Math.PI * 2;
        const am = 0.5 * (a0 + a1);
        let rgb;
        if (isCan) {
          if (b === 0 || b >= 3) {
            // Brushed aluminum chime, tapered shoulder & rolled top lip
            rgb = [214, 226, 242];
          } else {
            // Iconic quartered Red Bull blue & silver-ice high-contrast body panels
            const quadIdx = Math.floor(((am / (Math.PI * 0.5)) % 4 + 4) % 4);
            const isBlue = ((quadIdx + (b === 2 ? 1 : 0)) % 2) === 0;
            rgb = isBlue ? [38, 112, 246] : [226, 236, 248];
          }
        } else {
          rgb = (i % 2 === 0) ? [34, 198, 176] : [52, 214, 194];
        }
        const p00 = W(r0 * Math.cos(a0), r0 * Math.sin(a0), z0);
        const p10 = W(r0 * Math.cos(a1), r0 * Math.sin(a1), z0);
        const p11 = W(r1 * Math.cos(a1), r1 * Math.sin(a1), z1);
        const p01 = W(r1 * Math.cos(a0), r1 * Math.sin(a0), z1);
        addQuad(p00, p10, p11, p01, rgb);
      }
    }

    // Top recessed metallic lid & bottom cap
    const rTop = (isCan ? 0.86 : 1.0) * rad;
    const zTop = (isCan ? 0.485 : 0.50) * ez;
    const rBot = (isCan ? 0.88 : 1.0) * rad;
    const topC = W(0, 0, zTop);
    const botC = W(0, 0, -hz);
    for (let i = 0; i < N; i++) {
      const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1) / N) * Math.PI * 2;
      tris.push({
        v0: topC,
        v1: W(rTop * Math.cos(a0), rTop * Math.sin(a0), zTop),
        v2: W(rTop * Math.cos(a1), rTop * Math.sin(a1), zTop),
        rgb: isCan ? [205, 218, 235] : [72, 230, 205],
      });
      tris.push({
        v0: botC,
        v1: W(rBot * Math.cos(a1), rBot * Math.sin(a1), -hz),
        v2: W(rBot * Math.cos(a0), rBot * Math.sin(a0), -hz),
        rgb: [140, 160, 185],
      });
    }

    // Pull-tab & score-opening detail on top lid for beverage can
    if (isCan) {
      const tabN = 12;
      const openR = 0.28 * rTop;
      const openCx = 0.34 * rTop;
      for (let i = 0; i < tabN; i++) {
        const a0 = (i / tabN) * Math.PI * 2, a1 = ((i + 1) / tabN) * Math.PI * 2;
        ribs.push({
          a: W(openCx + openR * Math.cos(a0), 0.78 * openR * Math.sin(a0), hz + 0.0008),
          b: W(openCx + openR * Math.cos(a1), 0.78 * openR * Math.sin(a1), hz + 0.0008),
          nrm: Nrm(0, 0, 1),
          bold: true,
        });
        ribs.push({
          a: W(-0.18 * rTop + 0.22 * rTop * Math.cos(a0), 0.18 * rTop * Math.sin(a0), hz + 0.0012),
          b: W(-0.18 * rTop + 0.22 * rTop * Math.cos(a1), 0.18 * rTop * Math.sin(a1), hz + 0.0012),
          nrm: Nrm(0, 0, 1),
          bold: false,
        });
      }
    }

    // Structural rings & 8 longitudinal meridian ribs
    const ringLevels = isCan ? [[-0.50, 0.88, true], [0.0, 1.00, false], [0.39, 1.00, false], [0.50, 0.88, true]] : [[-0.50, 1.0, true], [0.0, 1.0, false], [0.50, 1.0, true]];
    for (const [zf, rs, isBold] of ringLevels) {
      const zVal = zf * ez, rVal = rs * rad;
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1) / N) * Math.PI * 2;
        const am = 0.5 * (a0 + a1);
        ribs.push({
          a: W(rVal * Math.cos(a0), rVal * Math.sin(a0), zVal),
          b: W(rVal * Math.cos(a1), rVal * Math.sin(a1), zVal),
          nrm: Nrm(Math.cos(am), Math.sin(am), zf > 0.45 ? 0.5 : 0),
          bold: isBold,
        });
      }
    }
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * Math.PI * 2;
      const ca = Math.cos(a), sa = Math.sin(a);
      for (let b = 0; b < profile.length - 1; b++) {
        ribs.push({
          a: W(profile[b][1] * rad * ca, profile[b][1] * rad * sa, profile[b][0] * ez),
          b: W(profile[b+1][1] * rad * ca, profile[b+1][1] * rad * sa, profile[b+1][0] * ez),
          nrm: Nrm(ca, sa, 0),
          bold: (k % 2 === 0),
        });
      }
    }
  } else if (tracked.shape === 'sphere') {
    const rad = ex * 0.5;
    const isWood = tracked.key === 'wood_sphere' || tracked.material === 'wood';
    const nLat = 12, nLon = 24;
    for (let i = 0; i < nLat; i++) {
      const v0 = -0.5 * Math.PI + (i / nLat) * Math.PI;
      const v1 = -0.5 * Math.PI + ((i + 1) / nLat) * Math.PI;
      const cv0 = Math.cos(v0), sv0 = Math.sin(v0);
      const cv1 = Math.cos(v1), sv1 = Math.sin(v1);
      // Warm lathe-turned beech/birch wood grain bands or vibrant emerald-cyan
      const bandShade = (i % 2 === 0) ? 1.0 : 0.91;
      const baseRgb = isWood
        ? [Math.round(238 * bandShade), Math.round(176 * bandShade), Math.round(86 * bandShade)]
        : [Math.round(45 * bandShade), Math.round(212 * bandShade), Math.round(191 * bandShade)];

      for (let j = 0; j < nLon; j++) {
        const u0 = (j / nLon) * Math.PI * 2;
        const u1 = ((j + 1) / nLon) * Math.PI * 2;
        const p00 = W(rad * cv0 * Math.cos(u0), rad * cv0 * Math.sin(u0), rad * sv0);
        const p10 = W(rad * cv0 * Math.cos(u1), rad * cv0 * Math.sin(u1), rad * sv0);
        const p11 = W(rad * cv1 * Math.cos(u1), rad * cv1 * Math.sin(u1), rad * sv1);
        const p01 = W(rad * cv1 * Math.cos(u0), rad * cv1 * Math.sin(u0), rad * sv1);
        if (i === 0) {
          tris.push({ v0: p00, v1: p11, v2: p01, rgb: baseRgb });
        } else if (i === nLat - 1) {
          tris.push({ v0: p00, v1: p10, v2: p11, rgb: baseRgb });
        } else {
          addQuad(p00, p10, p11, p01, baseRgb);
        }
      }
    }

    // 3 orthogonal great-circle structural rings so 3D sphere position & rotation pop clearly
    const segs = 32;
    for (let ring = 0; ring < 3; ring++) {
      for (let i = 0; i < segs; i++) {
        const a0 = (i / segs) * Math.PI * 2, a1 = ((i + 1) / segs) * Math.PI * 2;
        const am = 0.5 * (a0 + a1);
        const rBump = rad * 1.006;
        let l0, l1, ln;
        if (ring === 0) {
          l0 = [rBump * Math.cos(a0), rBump * Math.sin(a0), 0];
          l1 = [rBump * Math.cos(a1), rBump * Math.sin(a1), 0];
          ln = [Math.cos(am), Math.sin(am), 0];
        } else if (ring === 1) {
          l0 = [rBump * Math.cos(a0), 0, rBump * Math.sin(a0)];
          l1 = [rBump * Math.cos(a1), 0, rBump * Math.sin(a1)];
          ln = [Math.cos(am), 0, Math.sin(am)];
        } else {
          l0 = [0, rBump * Math.cos(a0), rBump * Math.sin(a0)];
          l1 = [0, rBump * Math.cos(a1), rBump * Math.sin(a1)];
          ln = [0, Math.cos(am), Math.sin(am)];
        }
        ribs.push({ a: W(l0[0], l0[1], l0[2]), b: W(l1[0], l1[1], l1[2]), nrm: Nrm(ln[0], ln[1], ln[2]), bold: ring === 0 });
      }
    }
  } else {
    // Box / Valio PROfeel Tetra Brik Aseptic Edge Carton (with sloped top + screw cap!)
    const hx = ex * 0.5, hy = ey * 0.5, hz = ez * 0.5;
    const isCarton = tracked.key === 'profeel' || (tracked.sloped_top_m || 0) > 0;
    const slope = tracked.sloped_top_m || 0.0;
    const zTopBack = hz;
    const zTopFront = hz - slope;

    // 4 vertical side walls subdivided into 3 height bands for sharp depth sorting & carton graphics
    const sideWalls = [
      // +X front wall (lowers to zTopFront), -X back wall (at zTopBack), +Y right wall, -Y left wall
      { c0: [hx, -hy], c1: [hx, hy], zt0: zTopFront, zt1: zTopFront, n: [1, 0, 0] },
      { c0: [-hx, hy], c1: [-hx, -hy], zt0: zTopBack, zt1: zTopBack, n: [-1, 0, 0] },
      { c0: [hx, hy], c1: [-hx, hy], zt0: zTopFront, zt1: zTopBack, n: [0, 1, 0] },
      { c0: [-hx, -hy], c1: [hx, -hy], zt0: zTopBack, zt1: zTopFront, n: [0, -1, 0] },
    ];

    const bandFracs = [0.0, 0.34, 0.68, 1.0];
    for (const wDef of sideWalls) {
      for (let b = 0; b < 3; b++) {
        const f0 = bandFracs[b], f1 = bandFracs[b + 1];
        const za0 = -hz + f0 * (wDef.zt0 + hz);
        const za1 = -hz + f1 * (wDef.zt0 + hz);
        const zb0 = -hz + f0 * (wDef.zt1 + hz);
        const zb1 = -hz + f1 * (wDef.zt1 + hz);
        let rgb;
        if (isCarton) {
          rgb = (b === 1) ? [20, 184, 166] : (b === 2 ? [234, 242, 250] : [210, 224, 238]);
        } else {
          rgb = (b === 1) ? [45, 212, 191] : [56, 189, 248];
        }
        addQuad(
          W(wDef.c0[0], wDef.c0[1], za0),
          W(wDef.c1[0], wDef.c1[1], zb0),
          W(wDef.c1[0], wDef.c1[1], zb1),
          W(wDef.c0[0], wDef.c0[1], za1),
          rgb
        );
      }
      // Mid-carton horizontal seam rib
      const zm0 = -hz + 0.68 * (wDef.zt0 + hz), zm1 = -hz + 0.68 * (wDef.zt1 + hz);
      ribs.push({
        a: W(wDef.c0[0], wDef.c0[1], zm0),
        b: W(wDef.c1[0], wDef.c1[1], zm1),
        nrm: Nrm(wDef.n[0], wDef.n[1], wDef.n[2]),
        bold: false,
      });
    }

    // Sloped top panel & flat bottom panel
    addQuad(W(-hx, -hy, zTopBack), W(hx, -hy, zTopFront), W(hx, hy, zTopFront), W(-hx, hy, zTopBack), isCarton ? [242, 246, 252] : [94, 234, 212]);
    addQuad(W(-hx, hy, -hz), W(hx, hy, -hz), W(hx, -hy, -hz), W(-hx, -hy, -hz), [130, 150, 175]);

    // 3D Raised Screw-Cap on sloped Tetra Edge top for Valio PROfeel carton
    if (isCarton) {
      const capR = Math.min(hx, hy) * 0.56;
      const capH = 0.007;
      const zMidTop = 0.5 * (zTopBack + zTopFront);
      const capN = 14;
      const capTopCenter = W(0.002, 0, zMidTop + capH);
      for (let i = 0; i < capN; i++) {
        const a0 = (i / capN) * Math.PI * 2, a1 = ((i + 1) / capN) * Math.PI * 2;
        const x0 = 0.002 + capR * Math.cos(a0), y0 = capR * Math.sin(a0);
        const x1 = 0.002 + capR * Math.cos(a1), y1 = capR * Math.sin(a1);
        const zb0 = zMidTop - (x0 / Math.max(ex, 0.02)) * slope;
        const zb1 = zMidTop - (x1 / Math.max(ex, 0.02)) * slope;
        addQuad(W(x0, y0, zb0), W(x1, y1, zb1), W(x1, y1, zMidTop + capH), W(x0, y0, zMidTop + capH), [255, 118, 56]);
        tris.push({ v0: capTopCenter, v1: W(x0, y0, zMidTop + capH), v2: W(x1, y1, zMidTop + capH), rgb: [255, 142, 82] });
        ribs.push({ a: W(x0, y0, zMidTop + capH), b: W(x1, y1, zMidTop + capH), nrm: Nrm(0, 0, 1), bold: true });
      }
    }

    // 12 outer box structural edges with outward normals
    const c8 = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, zTopBack], [hx, -hy, zTopFront], [hx, hy, zTopFront], [-hx, hy, zTopBack]
    ];
    const edgeDefs = [
      [0,1,[0,-1,-0.5]], [1,2,[1,0,-0.5]], [2,3,[0,1,-0.5]], [3,0,[-1,0,-0.5]],
      [4,5,[0,-1,0.8]],  [5,6,[1,0,0.8]],  [6,7,[0,1,0.8]],  [7,4,[-1,0,0.8]],
      [0,4,[-1,-1,0]],   [1,5,[1,-1,0]],   [2,6,[1,1,0]],    [3,7,[-1,1,0]],
    ];
    for (const [i, j, en] of edgeDefs) {
      ribs.push({
        a: W(c8[i][0], c8[i][1], c8[i][2]),
        b: W(c8[j][0], c8[j][1], c8[j][2]),
        nrm: Nrm(en[0], en[1], en[2]),
        bold: true,
      });
    }
  }

  return { tris, ribs };
}

function drawTriList(tris) {
  for (const t of tris) {
    ctx.beginPath();
    ctx.moveTo(t.p0[0], t.p0[1]);
    ctx.lineTo(t.p1[0], t.p1[1]);
    ctx.lineTo(t.p2[0], t.p2[1]);
    ctx.closePath();
    ctx.fillStyle = t.fill;
    ctx.fill();
    if (t.strokeEdge) {
      ctx.strokeStyle = t.edge;
      ctx.lineWidth = t.lineW || 0.65;
      ctx.stroke();
    }
  }
}

function buildHullDrawTris(hull, hullIsSeeThrough, objSolidActive, objIsSeeThrough, basis) {
  if (!hull || !hull.vertices || !hull.faces) return { front: [], back: [], all: [] };
  const V = hull.vertices;
  const L1 = [0.40, -0.56, 0.72];
  const L2 = [-0.62, 0.35, 0.42];
  const baseRgb = [36, 198, 232];
  const rimRgb = [184, 255, 59];
  const front = [];
  const back = [];

  for (const [i0, i1, i2] of hull.faces) {
    const v0 = V[i0], v1 = V[i1], v2 = V[i2];
    const p0 = project3DWithBasis(v0, basis), p1 = project3DWithBasis(v1, basis), p2 = project3DWithBasis(v2, basis);
    if (!p0 || !p1 || !p2) continue;

    const ux = v1[0] - v0[0], uy = v1[1] - v0[1], uz = v1[2] - v0[2];
    const vx = v2[0] - v0[0], vy = v2[1] - v0[1], vz = v2[2] - v0[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const nn = Math.hypot(nx, ny, nz);
    if (nn < 1e-10) continue;
    nx /= nn; ny /= nn; nz /= nn;

    const cx = (v0[0] + v1[0] + v2[0]) / 3;
    const cy = (v0[1] + v1[1] + v2[1]) / 3;
    const cz = (v0[2] + v1[2] + v2[2]) / 3;
    let vxEye = basis.eye[0] - cx, vyEye = basis.eye[1] - cy, vzEye = basis.eye[2] - cz;
    const vn = Math.hypot(vxEye, vyEye, vzEye) || 1;
    vxEye /= vn; vyEye /= vn; vzEye /= vn;

    const ndv = nx * vxEye + ny * vyEye + nz * vzEye;
    const isFront = ndv > -0.02;
    if (!hullIsSeeThrough && !isFront) continue; // Backface cull when Hull: Visible (solid)

    const depth = (p0[2] + p1[2] + p2[2]) / 3;
    if (!hullIsSeeThrough) {
      const diff1 = Math.max(0, nx * L1[0] + ny * L1[1] + nz * L1[2]);
      const diff2 = Math.max(0, nx * L2[0] + ny * L2[1] + nz * L2[2]);
      const diff = 0.38 + 0.50 * diff1 + 0.20 * diff2;
      let hx = L1[0] + vxEye, hy = L1[1] + vyEye, hz = L1[2] + vzEye;
      const hn = Math.hypot(hx, hy, hz) || 1;
      hx /= hn; hy /= hn; hz /= hn;
      const spec = Math.pow(Math.max(0, nx * hx + ny * hy + nz * hz), 18) * 0.38;
      const rim = Math.pow(1.0 - Math.max(0, ndv), 2.3) * 0.30;
      const r = Math.min(255, Math.round(baseRgb[0] * diff + 255 * spec + rimRgb[0] * rim));
      const g = Math.min(255, Math.round(baseRgb[1] * diff + 255 * spec + rimRgb[1] * rim));
      const b = Math.min(255, Math.round(baseRgb[2] * diff + 255 * spec + rimRgb[2] * rim));
      front.push({
        p0, p1, p2, depth,
        fill: `rgba(${r},${g},${b},0.88)`,
        edge: 'rgba(184, 255, 59, 0.90)',
        lineW: 1.2,
        strokeEdge: true,
      });
    } else {
      const shade = Math.max(0.25, Math.min(1.0, 0.55 + 0.45 * (0.4 * nx - 0.3 * ny + 0.85 * nz)));
      const g = Math.round(165 * shade + 55);
      const overSolidObj = objSolidActive && !objIsSeeThrough;
      if (isFront) {
        front.push({
          p0, p1, p2, depth,
          fill: `rgba(56, ${g}, 245, ${overSolidObj ? 0.11 : 0.25})`,
          edge: overSolidObj ? 'rgba(184, 255, 59, 0.46)' : 'rgba(184, 255, 59, 0.65)',
          lineW: overSolidObj ? 0.95 : 1.05,
          strokeEdge: true,
        });
      } else {
        back.push({
          p0, p1, p2, depth,
          fill: `rgba(56, ${g}, 245, ${overSolidObj ? 0.05 : 0.08})`,
          edge: overSolidObj ? 'rgba(56, 217, 245, 0.14)' : 'rgba(56, 217, 245, 0.22)',
          lineW: 0.75,
          strokeEdge: true,
        });
      }
    }
  }

  front.sort((a, b) => b.depth - a.depth);
  back.sort((a, b) => b.depth - a.depth);
  const all = [...back, ...front];
  all.sort((a, b) => b.depth - a.depth);
  return { front, back, all };
}

function renderConfirmedStructure(tracked, isConfirmed90, isSeeThrough, basis, zT, mergedSolidTris, overlayHullTris) {
  const p = tracked.pose;
  const [ex, ey, ez] = tracked.extents_m;
  const phase = p.phase || 'scanning';
  const accentHex = phase === 'placed' ? '#22c55e' : (phase === 'carrying' ? '#f59e0b' : '#b8ff3b');
  const accentRgb = phase === 'placed' ? [34, 197, 94] : (phase === 'carrying' ? [245, 158, 11] : [184, 255, 59]);
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() * 0.005);

  // 1. Ground Footprint Halo, Orientation Arrow & Carry Plumb-Line on z = zT
  const footR = Math.max(ex, ey) * 0.58 + 0.008;
  const yaw = p.yaw_rad || 0;
  const segN = 28;
  for (let i = 0; i < segN; i++) {
    const a0 = (i / segN) * Math.PI * 2, a1 = ((i + 1) / segN) * Math.PI * 2;
    drawLine3D(
      [p.x_m + footR * Math.cos(a0), p.y_m + footR * Math.sin(a0), zT + 0.0005],
      [p.x_m + footR * Math.cos(a1), p.y_m + footR * Math.sin(a1), zT + 0.0005],
      accentHex, 2.2, [], basis
    );
    if (isConfirmed90) {
      const outerR = footR + 0.010 + 0.004 * pulse;
      drawLine3D(
        [p.x_m + outerR * Math.cos(a0), p.y_m + outerR * Math.sin(a0), zT + 0.0004],
        [p.x_m + outerR * Math.cos(a1), p.y_m + outerR * Math.sin(a1), zT + 0.0004],
        `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},${(0.32 + 0.25 * pulse).toFixed(2)})`, 1.4, [], basis
      );
    }
  }
  // Yaw direction tick on table
  const dirLen = footR + 0.024;
  drawLine3D(
    [p.x_m + footR * 0.7 * Math.cos(yaw), p.y_m + footR * 0.7 * Math.sin(yaw), zT + 0.0008],
    [p.x_m + dirLen * Math.cos(yaw), p.y_m + dirLen * Math.sin(yaw), zT + 0.0008],
    accentHex, 2.5, [], basis
  );
  // Vertical laser drop-line while carrying in mid-air
  if (phase === 'carrying' || (p.z_bottom_m - zT) > 0.012) {
    drawLine3D([p.x_m, p.y_m, p.z_center_m - ez * 0.48], [p.x_m, p.y_m, zT], '#f59e0b', 1.8, [5, 4], basis);
    drawLine3D([p.x_m - 0.018, p.y_m, zT + 0.001], [p.x_m + 0.018, p.y_m, zT + 0.001], '#f59e0b', 2.0, [], basis);
    drawLine3D([p.x_m, p.y_m - 0.018, zT + 0.001], [p.x_m, p.y_m + 0.018, zT + 0.001], '#f59e0b', 2.0, [], basis);
  }

  // 2. Studio-Lit Backface-Culled 3D Mesh (Solid or See-Through Glass Shell)
  const mesh = buildConfirmedStructure(tracked);
  const drawTris = [];
  // Normalized studio key + fill lights
  const L1 = [0.40, -0.56, 0.72];
  const L2 = [-0.62, 0.35, 0.42];
  const fillAlpha = isSeeThrough ? 0.10 : (isConfirmed90 ? 0.91 : 0.78);

  for (const t of mesh.tris) {
    const ux = t.v1[0] - t.v0[0], uy = t.v1[1] - t.v0[1], uz = t.v1[2] - t.v0[2];
    const vx = t.v2[0] - t.v0[0], vy = t.v2[1] - t.v0[1], vz = t.v2[2] - t.v0[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const nn = Math.hypot(nx, ny, nz);
    if (nn < 1e-10) continue;
    nx /= nn; ny /= nn; nz /= nn;

    const cx = (t.v0[0] + t.v1[0] + t.v2[0]) / 3;
    const cy = (t.v0[1] + t.v1[1] + t.v2[1]) / 3;
    const cz = (t.v0[2] + t.v1[2] + t.v2[2]) / 3;
    let vxEye = basis.eye[0] - cx, vyEye = basis.eye[1] - cy, vzEye = basis.eye[2] - cz;
    const vn = Math.hypot(vxEye, vyEye, vzEye) || 1;
    vxEye /= vn; vyEye /= vn; vzEye /= vn;

    const ndv = nx * vxEye + ny * vyEye + nz * vzEye;
    if (ndv <= -0.02) continue; // Backface cull for clean artifact-free rendering

    const p0 = project3DWithBasis(t.v0, basis);
    const p1 = project3DWithBasis(t.v1, basis);
    const p2 = project3DWithBasis(t.v2, basis);
    if (!p0 || !p1 || !p2) continue;

    const diff1 = Math.max(0, nx * L1[0] + ny * L1[1] + nz * L1[2]);
    const diff2 = Math.max(0, nx * L2[0] + ny * L2[1] + nz * L2[2]);
    const diff = 0.36 + 0.52 * diff1 + 0.20 * diff2;

    let hx = L1[0] + vxEye, hy = L1[1] + vyEye, hz = L1[2] + vzEye;
    const hn = Math.hypot(hx, hy, hz) || 1;
    hx /= hn; hy /= hn; hz /= hn;
    const spec = Math.pow(Math.max(0, nx * hx + ny * hy + nz * hz), 20) * 0.42;
    const rim = Math.pow(1.0 - Math.max(0, ndv), 2.4) * 0.28;

    const r = Math.min(255, Math.round(t.rgb[0] * diff + 255 * spec + accentRgb[0] * rim));
    const g = Math.min(255, Math.round(t.rgb[1] * diff + 255 * spec + accentRgb[1] * rim));
    const b = Math.min(255, Math.round(t.rgb[2] * diff + 255 * spec + accentRgb[2] * rim));
    drawTris.push({
      p0, p1, p2,
      depth: (p0[2] + p1[2] + p2[2]) / 3,
      fill: `rgba(${r},${g},${b},${fillAlpha})`,
      edge: `rgba(${r},${g},${b},${Math.min(1, fillAlpha + 0.06)})`,
      lineW: 0.65,
      strokeEdge: !isSeeThrough,
    });
  }

  if (mergedSolidTris && mergedSolidTris.length) {
    drawTris.push(...mergedSolidTris);
  }
  drawTris.sort((a, b) => b.depth - a.depth);
  drawTriList(drawTris);

  // 3. Crisp Structural CAD Seams & Ribs (bright on front faces, subtle X-ray dashed on back)
  for (const rb of mesh.ribs) {
    const mx = 0.5 * (rb.a[0] + rb.b[0]), my = 0.5 * (rb.a[1] + rb.b[1]), mz = 0.5 * (rb.a[2] + rb.b[2]);
    const front = ((basis.eye[0] - mx) * rb.nrm[0] + (basis.eye[1] - my) * rb.nrm[1] + (basis.eye[2] - mz) * rb.nrm[2]) >= -0.01;
    if (front) {
      const frontAlpha = isSeeThrough ? (rb.bold ? 0.72 : 0.48) : 0.72;
      const col = (!isSeeThrough && rb.bold)
        ? accentHex
        : `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},${frontAlpha})`;
      drawLine3D(rb.a, rb.b, col, rb.bold ? (isSeeThrough ? 1.8 : 2.5) : 1.25, [], basis);
    } else {
      drawLine3D(rb.a, rb.b, `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},${isSeeThrough ? 0.24 : 0.18})`, 1.0, [3, 4], basis);
    }
  }

  if (overlayHullTris && overlayHullTris.length) {
    drawTriList(overlayHullTris);
  }

  // 4. High-Visibility Floating 3D Confirmation Badge above the Structure
  const topAnchor = [p.x_m, p.y_m, p.z_center_m + ez * 0.5 + 0.012];
  const labelAnchor = [p.x_m, p.y_m, p.z_center_m + ez * 0.5 + 0.048];
  const spTop = project3DWithBasis(topAnchor, basis);
  const spLbl = project3DWithBasis(labelAnchor, basis);
  if (spTop && spLbl) {
    ctx.save();
    ctx.strokeStyle = accentHex;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(spTop[0], spTop[1]);
    ctx.lineTo(spLbl[0], spLbl[1]);
    ctx.stroke();

    const pct = Math.round((tracked.confidence || 0) * 100);
    let titleLine;
    if (phase === 'placed') {
      titleLine = `✓ G1 PICK TARGET · ${tracked.label}`;
    } else if (phase === 'carrying') {
      titleLine = `CARRYING · ${tracked.label} (${pct}%)`;
    } else {
      titleLine = `✓ CONFIRMED ${pct}% · ${tracked.label}`;
    }

    const dimStr = tracked.shape === 'sphere'
      ? `Ø${Math.round(ex * 1000)} mm`
      : (tracked.shape === 'cylinder'
          ? `Ø${Math.round(ex * 1000)} × ${Math.round(ez * 1000)} mm`
          : `${Math.round(ex * 1000)} × ${Math.round(ey * 1000)} × ${Math.round(ez * 1000)} mm`);
    const subLine = phase === 'placed'
      ? `${dimStr} · Pos (${(p.x_m * 100).toFixed(1)}, ${(p.y_m * 100).toFixed(1)}) cm`
      : `${dimStr}${tracked.mass_g ? ' · ' + Math.round(tracked.mass_g) + 'g · max ' + tracked.max_grip_n + 'N' : ''}`;

    ctx.font = '700 12px system-ui, sans-serif';
    const w1 = ctx.measureText(titleLine).width;
    ctx.font = '500 11px ui-monospace, Consolas, monospace';
    const w2 = ctx.measureText(subLine).width;
    const boxW = Math.max(w1, w2) + 22;
    const boxH = 38;
    const bx = spLbl[0] - boxW * 0.5;
    const by = spLbl[1] - boxH - 2;

    drawRoundedPill2D(bx, by, boxW, boxH, 8, 'rgba(13, 16, 22, 0.92)', accentHex, 1.8);
    ctx.fillStyle = accentHex;
    ctx.font = '700 12px system-ui, sans-serif';
    ctx.fillText(titleLine, bx + 11, by + 16);
    ctx.fillStyle = '#d8dee9';
    ctx.font = '500 11px ui-monospace, Consolas, monospace';
    ctx.fillText(subLine, bx + 11, by + 31);
    ctx.restore();
  }
}

function drawPointSubset(points, basis, zT, objDepth, drawFrontHalf, dimBehind) {
  for (const pt of points) {
    const sp = project3DWithBasis(pt, basis);
    if (!sp) continue;
    const isFront = objDepth === null || sp[2] <= objDepth + 0.008;
    if (drawFrontHalf !== isFront) continue;

    const hFrac = Math.min(1, Math.max(0, (pt[2] - zT) / 0.15));
    const r = Math.round(255 * (1 - 0.55 * hFrac));
    const g = Math.round(125 + 130 * hFrac);
    const alpha = (!isFront && dimBehind) ? 0.28 : 0.95;
    const rad = Math.max(2.3, Math.min(5.2, 2.7 / sp[2]));

    ctx.beginPath();
    ctx.arc(sp[0], sp[1], rad + (isFront ? 0.9 : 0), 0, Math.PI * 2);
    if (isFront) {
      ctx.fillStyle = 'rgba(10, 12, 16, 0.75)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(sp[0], sp[1], rad, 0, Math.PI * 2);
    }
    ctx.fillStyle = `rgba(${r},${g},65,${alpha})`;
    ctx.fill();
  }
}

function stepKeyboardAndFollow(dtSec) {
  if (keysDown.size > 0) {
    const basis = getCameraBasis();
    const speed = 0.72 * camDist * dtSec;
    let fwd = 0, right = 0, up = 0;
    if (keysDown.has('w') || keysDown.has('arrowup')) fwd += 1;
    if (keysDown.has('s') || keysDown.has('arrowdown')) fwd -= 1;
    if (keysDown.has('d') || keysDown.has('arrowright')) right += 1;
    if (keysDown.has('a') || keysDown.has('arrowleft')) right -= 1;
    if (keysDown.has('q') || keysDown.has('pageup')) up += 1;
    if (keysDown.has('e') || keysDown.has('pagedown')) up -= 1;
    if (fwd !== 0 || right !== 0 || up !== 0) {
      target[0] += (right * basis.rx + fwd * basis.fhx) * speed;
      target[1] += (right * basis.ry + fwd * basis.fhy) * speed;
      target[2] += up * speed * 0.75;
      syncSlidersFromTarget();
    }
  } else if (followScene && state) {
    const f = getActiveSceneFocus();
    const alpha = 1.0 - Math.exp(-7.0 * dtSec);
    target[0] += (f[0] - target[0]) * alpha;
    target[1] += (f[1] - target[1]) * alpha;
    target[2] += (f[2] - target[2]) * alpha;
    syncSlidersFromTarget();
  }
}

function render(nowMs) {
  const dtSec = Math.min(0.1, Math.max(0.001, (nowMs - lastFrameTime) * 0.001));
  lastFrameTime = nowMs;

  const rect = cv.parentElement.getBoundingClientRect();
  if (cv.width !== rect.width || cv.height !== rect.height) {
    cv.width = rect.width; cv.height = rect.height;
  }
  ctx.clearRect(0, 0, cv.width, cv.height);

  stepKeyboardAndFollow(dtSec);
  const basis = getCameraBasis();
  const zT = state?.calibration?.z_table || 0.0;

  // 1. Dynamic World-Anchored Table Grid (follows target across any Coil Pro X-Z offset!)
  const gx0 = Math.round(target[0] / 0.10) * 0.10;
  const gy0 = Math.round(target[1] / 0.10) * 0.10;
  const halfCells = 9;
  const span = halfCells * 0.10;
  for (let i = -halfCells; i <= halfCells; i++) {
    const x = gx0 + i * 0.10;
    const isZero = Math.abs(x) < 1e-4;
    const isMajor = Math.round(Math.abs(x) * 100) % 50 === 0;
    const stroke = isZero ? 'rgba(239,68,68,0.28)' : (isMajor ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.065)');
    drawLine3D([x, gy0 - span, zT], [x, gy0 + span, zT], stroke, isMajor ? 1.2 : 1.0, [], basis);
  }
  for (let j = -halfCells; j <= halfCells; j++) {
    const y = gy0 + j * 0.10;
    const isZero = Math.abs(y) < 1e-4;
    const isMajor = Math.round(Math.abs(y) * 100) % 50 === 0;
    const stroke = isZero ? 'rgba(34,197,94,0.28)' : (isMajor ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.065)');
    drawLine3D([gx0 - span, y, zT], [gx0 + span, y, zT], stroke, isMajor ? 1.2 : 1.0, [], basis);
  }

  // World / Coil Pro Origin Triad at (0, 0, zT) + subtle offset line if workspace is shifted
  drawLine3D([0, 0, zT], [0.08, 0, zT], '#ef4444', 2.5, [], basis);
  drawLine3D([0, 0, zT], [0, 0.08, zT], '#22c55e', 2.5, [], basis);
  drawLine3D([0, 0, zT], [0, 0, zT + 0.08], '#3b82f6', 2.5, [], basis);
  const spOrig = project3DWithBasis([0, 0, zT], basis);
  if (spOrig) {
    ctx.fillStyle = 'rgba(154, 164, 184, 0.72)';
    ctx.font = '600 10px ui-monospace, Consolas, monospace';
    ctx.fillText('ORIGIN (0,0)', spOrig[0] + 6, spOrig[1] + 12);
  }
  // Camera orbit/pan pivot reticle on table
  drawLine3D([target[0] - 0.016, target[1], zT + 0.0005], [target[0] + 0.016, target[1], zT + 0.0005], 'rgba(56,217,245,0.42)', 1.4, [], basis);
  drawLine3D([target[0], target[1] - 0.016, zT + 0.0005], [target[0], target[1] + 0.016, zT + 0.0005], 'rgba(56,217,245,0.42)', 1.4, [], basis);

  if (!state) { requestAnimationFrame(render); return; }

  const tracked = resolveTrackedCandidate(state);
  const conf = tracked ? (tracked.confidence || 0) : 0;
  const isConfirmed90 = Boolean(tracked && (tracked.confirmed_90 || conf > 0.90));
  const showSolidStructure = Boolean(tracked && (isConfirmed90 || state.phase === 'carrying' || state.phase === 'placed'));
  const objIsSeeThrough = objectVisMode === 'see_through';
  const hullIsSeeThrough = hullVisMode === 'see_through';

  // Show Coil-to-Action offset tether on table when action happens >35 cm away from origin
  const focusPt = getActiveSceneFocus();
  if (Math.hypot(focusPt[0], focusPt[1]) > 0.35) {
    drawLine3D([0, 0, zT + 0.0003], [focusPt[0], focusPt[1], zT + 0.0003], 'rgba(154,164,184,0.22)', 1.1, [5, 5], basis);
  }

  // Build 3D Convex Hull triangles according to Hull: Visible (solid) vs Hull: See-Through
  const hullTris = buildHullDrawTris(state.hull, hullIsSeeThrough, showSolidStructure, objIsSeeThrough, basis);
  const hasHull = hullTris.all.length > 0;

  // Compute solid center camera depth so far/near tactile points layer cleanly around any solid mesh
  let objCamDepth = null;
  const anySolidMesh = (showSolidStructure && !objIsSeeThrough) || (hasHull && !hullIsSeeThrough);
  if (anySolidMesh) {
    const refCenter = (tracked && tracked.pose)
      ? [tracked.pose.x_m, tracked.pose.y_m, tracked.pose.z_center_m]
      : focusPt;
    const spObj = project3DWithBasis(refCenter, basis);
    if (spObj) objCamDepth = spObj[2];
  }

  // 2a. Far-side tactile points (behind solid object / solid hull)
  const hasPts = showPoints && state.cloud && state.cloud.points && state.cloud.points.length > 0;
  if (hasPts && objCamDepth !== null) {
    drawPointSubset(state.cloud.points, basis, zT, objCamDepth, false, true);
  }

  // 2b. Render 3D Convex Hull + Predefined 3D Object across all 4 (Visible / See-Through) combinations
  if (!showSolidStructure) {
    drawTriList(hullTris.all);
    if (tracked) {
      drawPreviewWireframe(tracked, basis);
    }
  } else if (!objIsSeeThrough && !hullIsSeeThrough) {
    // Object: Visible + Hull: Visible -> unified 3D depth-sorted solid triangles
    renderConfirmedStructure(tracked, isConfirmed90, false, basis, zT, hullTris.front, null);
  } else if (objIsSeeThrough && !hullIsSeeThrough) {
    // Object: See-Through + Hull: Visible -> solid 3D Convex Hull inside see-through Object shell
    drawTriList(hullTris.front);
    renderConfirmedStructure(tracked, isConfirmed90, true, basis, zT, null, null);
  } else if (!objIsSeeThrough && hullIsSeeThrough) {
    // Object: Visible + Hull: See-Through -> back hull behind solid Object, front see-through hull veil over it
    drawTriList(hullTris.back);
    renderConfirmedStructure(tracked, isConfirmed90, false, basis, zT, null, hullTris.front);
  } else {
    // Object: See-Through + Hull: See-Through -> both translucent
    drawTriList(hullTris.all);
    renderConfirmedStructure(tracked, isConfirmed90, true, basis, zT, null, null);
  }

  // 2c. Near-side tactile points (hugging the front surface of the 3D render!)
  if (hasPts) {
    drawPointSubset(state.cloud.points, basis, zT, objCamDepth, true, false);
  }

  // 4. Left & Right Rokoko Gloved Hands
  if (state.hands) {
    for (const [side, col] of [['left', '#ff8b3d'], ['right', '#38d9f5']]) {
      const h = state.hands[side];
      if (!h) continue;
      for (const [a, bPt] of h.bones) drawLine3D(a, bPt, col, 2.6, [], basis);
      for (const [fname, padPt] of Object.entries(h.pads)) {
        const sp = project3DWithBasis(padPt, basis);
        if (!sp) continue;
        const isTracing = side === 'right' && fname === 'index' && state.gesture?.stream_index;
        ctx.fillStyle = isTracing ? '#b8ff3b' : (side === 'right' ? '#eef1f6' : '#ffd1b0');
        ctx.beginPath();
        ctx.arc(sp[0], sp[1], isTracing ? 6.5 : 3.8, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  requestAnimationFrame(render);
}

function updateSidebar() {
  if (!state) return;
  const isSynth = state.source === 'synthetic';
  document.getElementById('synth-controls').style.display = isSynth ? 'flex' : 'none';
  document.getElementById('rokoko-controls').style.display = isSynth ? 'none' : 'flex';

  // Auto-focus camera onto live Rokoko hands on initial packet arrival so Coil Pro offset never hides hands!
  if (!isSynth && !hasInitialRokokoFocus && state.hands && (state.hands.right || state.hands.left)) {
    hasInitialRokokoFocus = true;
    centerOnActiveScene();
  }

  const conn = document.getElementById('conn-pill');
  if (isSynth) {
    conn.textContent = `SYNTHETIC DEMO (${state.fps} Hz)`;
  } else {
    const rs = state.rokoko_status || {};
    conn.textContent = `ROKOKO UDP:${rs.port} | pkts:${rs.packets_received || 0} | R:${rs.has_right?'✓':'✗'} L:${rs.has_left?'✓':'✗'} | ${state.mode.toUpperCase()}`;
  }

  const gPill = document.getElementById('gesture-pill');
  gPill.textContent = state.gesture?.badge || 'IDLE';
  gPill.className = 'pill' + (state.gesture?.sign !== 'idle' ? ' gesture-active' : '');

  const tracked = resolveTrackedCandidate(state);
  const confPct = tracked ? Math.round((tracked.confidence || 0) * 100) : 0;
  const isConfirmed90 = Boolean(tracked && (tracked.confirmed_90 || (tracked.confidence || 0) > 0.90));

  const cPill = document.getElementById('confirm-pill');
  if (isConfirmed90 && tracked) {
    cPill.style.display = 'block';
    cPill.className = 'pill confirmed-90';
    cPill.textContent = `✓ CONFIRMED ${confPct}%: ${tracked.label}`;
  } else {
    cPill.style.display = 'none';
  }

  const pPill = document.getElementById('phase-pill');
  pPill.textContent = 'PHASE: ' + state.phase.toUpperCase();
  pPill.className = 'pill phase-' + state.phase;

  document.getElementById('btn-trace').className = state.gesture?.stream_index ? 'active-toggle' : '';
  const phaseBtn = document.getElementById('btn-phase');
  if (state.phase === 'scanning') phaseBtn.textContent = '[Space] Lock Grasp to Pick Up (✌️)';
  else if (state.phase === 'carrying') phaseBtn.textContent = '[Space] Object Placed -> Run G1 (✌️)';
  else phaseBtn.textContent = '[Space] Pick Up & Move Again (✌️)';

  document.getElementById('m-pts').textContent = state.cloud.n_voxels;
  document.getElementById('m-h').textContent = state.cloud.height_mm.toFixed(0);
  document.getElementById('m-lw').textContent = `${state.cloud.length_mm.toFixed(0)}×${state.cloud.width_mm.toFixed(0)}`;
  document.getElementById('m-vol').textContent = state.hull ? state.hull.volume_cm3.toFixed(0) : '0';

  const pw = state.cloud.primitive_weights || {};
  document.getElementById('prim-line').textContent =
    `Primitive BIC: cyl ${((pw.cylinder||0)*100).toFixed(0)}% · box ${((pw.box||0)*100).toFixed(0)}% · sph ${((pw.sphere||0)*100).toFixed(0)}%` +
    (state.cloud.calipers_mm.length ? ` | calipers: ${state.cloud.calipers_mm.join(', ')} mm` : '');

  const catSec = document.getElementById('catalog-section');
  const confBanner = document.getElementById('confirmed-banner');
  catSec.className = isConfirmed90 ? 'section section-confirmed' : 'section';
  if (isConfirmed90 && tracked) {
    confBanner.style.display = 'block';
    confBanner.textContent = `✓ ${confPct}% CONFIRMED — ${tracked.label.toUpperCase()} 3D STRUCTURE LOCKED`;
  } else {
    confBanner.style.display = 'none';
  }

  const barsEl = document.getElementById('catalog-bars');
  const probs = state.catalog.probabilities || {};
  const items = state.catalog.items || {};
  barsEl.innerHTML = Object.keys(items).map(k => {
    const pct = ((probs[k] || 0) * 100);
    const hit90 = pct > 90 && state.cloud.n_voxels >= 10;
    const info = items[k];
    return `<div class="bar-row">
      <div class="bar-head"><span><b style="color:${hit90 ? 'var(--lime)' : 'var(--text)'}">${hit90 ? '✓ ' : ''}${info.label}</b> <small style="color:#9aa4b8">(${info.mass_g}g, max ${info.max_grip_n}N)</small></span><span style="font-weight:${hit90?'700':'400'};color:${hit90?'var(--lime)':'inherit'}">${pct.toFixed(0)}%</span></div>
      <div class="bar-track"><div class="bar-fill${hit90 ? ' confirmed' : ''}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');
  document.getElementById('match-advice').textContent = state.catalog.advice || '';

  document.getElementById('event-log').textContent = (state.events || []).slice().reverse().join('\n');
}

async function pollLoop() {
  try {
    const res = await fetch('/api/state');
    state = await res.json();
    updateSidebar();
  } catch (e) {}
  setTimeout(pollLoop, 45);
}

syncSlidersFromTarget();
requestAnimationFrame(render);
pollLoop();
</script>
</body>
</html>
"""


def run_live_dashboard(
    session: ScanTrackSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    tick_hz: float = 50.0,
) -> None:
    """Start the background session ticker + HTTP dashboard server and block until Ctrl-C."""
    stop_event = threading.Event()
    latest_snapshot: dict[str, Any] = session.tick()

    def ticker() -> None:
        nonlocal latest_snapshot
        period = 1.0 / max(tick_hz, 5.0)
        while not stop_event.is_set():
            t0 = time.time()
            latest_snapshot = session.tick()
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    bg = threading.Thread(target=ticker, daemon=True)
    bg.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # silence noisy per-poll HTTP logs
            return

        def _send_json(self, doc: dict[str, Any]) -> None:
            body = json.dumps(doc).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/state"):
                self._send_json(latest_snapshot)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal latest_snapshot
            if self.path == "/api/action":
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                req = json.loads(raw.decode("utf-8"))
                latest_snapshot = session.handle_action(
                    str(req.get("action", "")),
                    req.get("payload") or {},
                )
                self._send_json(latest_snapshot)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    url = f"http://{host}:{port}"
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Live 3D Tactile Scan & Unitree G1 Dashboard running at: {url}")
    print(f"Session artifacts saving to: {session.session_dir}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    finally:
        stop_event.set()
        server.server_close()
        session.close()
