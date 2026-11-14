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
    display: flex; justify-content: space-between; align-items: center; gap: 10px; pointer-events: none;
  }
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
  .hud-bottom {
    position: absolute; bottom: 14px; left: 14px;
    background: rgba(23, 26, 33, 0.84);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--muted);
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
  .section h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
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
  button.primary { background: var(--accent); color: #111; border-color: var(--accent); grid-column: span 2; }
  button.primary:hover { filter: brightness(1.08); }
  button.active-toggle { border-color: var(--lime); color: var(--lime); }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; text-align: center; }
  .metric-box { background: #151922; border: 1px solid var(--border); border-radius: 7px; padding: 6px 4px; }
  .metric-val { font-size: 15px; font-weight: 700; color: var(--lime); font-variant-numeric: tabular-nums; }
  .metric-lbl { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .bar-row { margin-bottom: 7px; }
  .bar-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px; }
  .bar-track { height: 8px; background: #12151c; border-radius: 99px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--lime)); transition: width 0.15s; }
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
  .badge-ok { color: var(--ok); font-weight: 700; }
  .badge-warn { color: var(--warn); font-weight: 700; }
  .row-inline { display: flex; gap: 6px; margin-bottom: 7px; }
  .row-inline > * { flex: 1; }
</style>
</head>
<body>
  <div id="viewport-wrap">
    <canvas id="cv"></canvas>
    <div class="hud-top">
      <div id="conn-pill" class="pill">Connecting...</div>
      <div id="gesture-pill" class="pill">Gesture: IDLE</div>
      <div id="phase-pill" class="pill phase-scanning">PHASE: SCANNING</div>
    </div>
    <div class="hud-bottom">
      Drag: Orbit 3D &nbsp;|&nbsp; Scroll: Zoom &nbsp;|&nbsp; Keys: <b>[T]</b> Trace &nbsp;<b>[G]</b> 5-Finger Snap &nbsp;<b>[C]</b> Caliper &nbsp;<b>[Space]</b> Lock/Place &nbsp;<b>[B]</b> Calibrate Table &nbsp;<b>[R]</b> Reset
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
        <button class="primary" id="btn-phase" onclick="act('phase')">[Space] Lock Grasp / Place Object (✌️)</button>
        <button onclick="act('reset')" style="grid-column: span 2;">[R] Reset Session</button>
      </div>
    </div>

    <div class="section">
      <h3>Live 3D Cloud &amp; Convex Hull</h3>
      <div class="metrics">
        <div class="metric-box"><div class="metric-val" id="m-pts">0</div><div class="metric-lbl">Voxels</div></div>
        <div class="metric-box"><div class="metric-val" id="m-h">0</div><div class="metric-lbl">H (mm)</div></div>
        <div class="metric-box"><div class="metric-val" id="m-lw">0×0</div><div class="metric-lbl">L×W (mm)</div></div>
        <div class="metric-box"><div class="metric-val" id="m-vol">0</div><div class="metric-lbl">Hull cm³</div></div>
      </div>
      <div class="advice" id="prim-line" style="color:var(--muted); margin-top:7px;">Primitives: waiting for points...</div>
    </div>

    <div class="section">
      <h3>Known Catalog Match &rarr; Physics</h3>
      <div id="catalog-bars"></div>
      <div class="advice" id="match-advice">Waiting for tactile contacts...</div>
    </div>

    <div class="section">
      <h3>MuJoCo Twin &amp; Unitree G1 Pick Plan</h3>
      <div id="g1-summary" class="mono">Scan object, grab &amp; relocate on flat surface, then signal PLACED (✌️ or Space) to verify in MuJoCo and generate Unitree G1 command.</div>
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

// Orbit camera around workspace center (0.08, 0.34, 0.07)
let camAz = -0.65, camEl = 0.48, camDist = 0.78;
let target = [0.08, 0.34, 0.06];
let dragging = false, lastX = 0, lastY = 0;

cv.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  camAz -= (e.clientX - lastX) * 0.007;
  camEl = Math.max(0.06, Math.min(1.45, camEl + (e.clientY - lastY) * 0.007));
  lastX = e.clientX; lastY = e.clientY;
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  camDist = Math.max(0.30, Math.min(1.80, camDist * (1 + e.deltaY * 0.001)));
}, { passive: false });

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 't') act('trace_toggle');
  else if (k === 'g') act('snapshot');
  else if (k === 'c') act('caliper');
  else if (k === 'b') act('calibrate_table');
  else if (k === 'r') act('reset');
  else if (k === ' ' || k === 'p') { e.preventDefault(); act('phase'); }
});

async function act(action, payload={}) {
  const res = await fetch('/api/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action, payload})
  });
  state = await res.json();
  updateSidebar();
}
function setSynthObj(v) { act('set_synthetic_object', {object_key: v}); }
function setMode(v) { act('set_mode', {mode: v}); }
function toggleAuto() {
  autoLoop = !autoLoop;
  document.getElementById('auto-btn').textContent = 'Auto Loop: ' + (autoLoop ? 'ON' : 'MANUAL');
  act('set_auto_cycle', {enabled: autoLoop});
}

function project3D(p) {
  const w = cv.width, h = cv.height;
  const cx = Math.cos(camAz), sx = Math.sin(camAz);
  const ce = Math.cos(camEl), se = Math.sin(camEl);
  const eye = [
    target[0] + camDist * ce * sx,
    target[1] - camDist * ce * cx,
    target[2] + camDist * se
  ];
  // Forward = normalize(target - eye)
  let fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  const fn = Math.hypot(fx, fy, fz) || 1;
  fx /= fn; fy /= fn; fz /= fn;
  // Right = normalize(forward x [0,0,1])
  let rx = fy, ry = -fx, rz = 0;
  const rn = Math.hypot(rx, ry) || 1;
  rx /= rn; ry /= rn;
  // Up = right x forward
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;

  const dx = p[0] - eye[0], dy = p[1] - eye[1], dz = p[2] - eye[2];
  const zCam = dx * fx + dy * fy + dz * fz;
  if (zCam < 0.04) return null;
  const fov = Math.min(w, h) * 1.05;
  const xScreen = w * 0.5 + (dx * rx + dy * ry + dz * rz) * (fov / zCam);
  const yScreen = h * 0.52 - (dx * ux + dy * uy + dz * uz) * (fov / zCam);
  return [xScreen, yScreen, zCam];
}

function drawLine3D(a, b, stroke, width=1.5, dash=[]) {
  const pa = project3D(a), pb = project3D(b);
  if (!pa || !pb) return;
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  if (dash.length) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
  ctx.restore();
}

function drawTrackedWireframe(tracked) {
  if (!tracked || !tracked.pose) return;
  const p = tracked.pose;
  const [ex, ey, ez] = tracked.extents_m;
  const yaw = p.yaw_rad || 0;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const cx = p.x_m, cy = p.y_m, cz = p.z_center_m;
  const color = p.phase === 'placed' ? '#22c55e' : (p.phase === 'carrying' ? '#f59e0b' : 'rgba(56,217,245,0.45)');

  if (tracked.shape === 'cylinder' || tracked.shape === 'sphere') {
    const R = ex * 0.5, H = ez;
    const N = 24;
    for (const zOff of [-H * 0.5, H * 0.5]) {
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1) / N) * Math.PI * 2;
        const rScale = tracked.shape === 'sphere' ? 0.85 : 1.0;
        drawLine3D(
          [cx + R * rScale * Math.cos(a0), cy + R * rScale * Math.sin(a0), cz + zOff * (tracked.shape === 'sphere' ? 0.55 : 1)],
          [cx + R * rScale * Math.cos(a1), cy + R * rScale * Math.sin(a1), cz + zOff * (tracked.shape === 'sphere' ? 0.55 : 1)],
          color, 2.0
        );
      }
    }
    if (tracked.shape === 'sphere') {
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1) / N) * Math.PI * 2;
        drawLine3D([cx + R * Math.cos(a0), cy + R * Math.sin(a0), cz], [cx + R * Math.cos(a1), cy + R * Math.sin(a1), cz], color, 2.2);
      }
    }
  } else {
    const hx = ex * 0.5, hy = ey * 0.5, hz = ez * 0.5;
    const corners = [
      [-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
      [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]
    ].map(([lx, ly, lz]) => [cx + c * lx - s * ly, cy + s * lx + c * ly, cz + lz]);
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    for (const [i, j] of edges) drawLine3D(corners[i], corners[j], color, 2.0);
  }
}

function render() {
  const rect = cv.parentElement.getBoundingClientRect();
  if (cv.width !== rect.width || cv.height !== rect.height) {
    cv.width = rect.width; cv.height = rect.height;
  }
  ctx.clearRect(0, 0, cv.width, cv.height);

  const zT = state?.calibration?.z_table || 0.0;

  // 1. Flat Workspace Table Grid (10 cm cells)
  for (let i = -4; i <= 5; i++) {
    const x = i * 0.10;
    drawLine3D([x, 0.0, zT], [x, 0.70, zT], i === 0 ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.07)', 1);
  }
  for (let j = 0; j <= 7; j++) {
    const y = j * 0.10;
    drawLine3D([-0.40, y, zT], [0.50, y, zT], 'rgba(255,255,255,0.07)', 1);
  }
  // Coordinate axes at origin
  drawLine3D([0,0,zT], [0.08,0,zT], '#ef4444', 2.5);
  drawLine3D([0,0,zT], [0,0.08,zT], '#22c55e', 2.5);
  drawLine3D([0,0,zT], [0,0,zT+0.08], '#3b82f6', 2.5);

  if (!state) { requestAnimationFrame(render); return; }

  // 2. Shaded Translucent 3D Convex Hull (painter's depth-sorted faces)
  if (state.hull && state.hull.vertices && state.hull.faces) {
    const V = state.hull.vertices;
    const tris = [];
    for (const [i0, i1, i2] of state.hull.faces) {
      const p0 = project3D(V[i0]), p1 = project3D(V[i1]), p2 = project3D(V[i2]);
      if (!p0 || !p1 || !p2) continue;
      // Simple directional shading from top-right light
      const ux = V[i1][0]-V[i0][0], uy = V[i1][1]-V[i0][1], uz = V[i1][2]-V[i0][2];
      const vx = V[i2][0]-V[i0][0], vy = V[i2][1]-V[i0][1], vz = V[i2][2]-V[i0][2];
      let nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx;
      const nn = Math.hypot(nx, ny, nz) || 1;
      nx /= nn; ny /= nn; nz /= nn;
      const shade = Math.max(0.25, Math.min(1.0, 0.55 + 0.45 * (0.4*nx - 0.3*ny + 0.85*nz)));
      tris.push({ p0, p1, p2, depth: (p0[2] + p1[2] + p2[2]) / 3, shade });
    }
    tris.sort((a, b) => b.depth - a.depth);
    for (const t of tris) {
      ctx.beginPath();
      ctx.moveTo(t.p0[0], t.p0[1]);
      ctx.lineTo(t.p1[0], t.p1[1]);
      ctx.lineTo(t.p2[0], t.p2[1]);
      ctx.closePath();
      const g = Math.round(165 * t.shade + 55);
      ctx.fillStyle = `rgba(56, ${g}, 245, 0.26)`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(184, 255, 59, 0.42)';
      ctx.lineWidth = 0.9;
      ctx.stroke();
    }
  }

  // 3. Tracked / Relocated Object Twin Wireframe
  if (state.tracked_object) {
    drawTrackedWireframe(state.tracked_object);
    if (state.phase === 'placed') {
      const p = state.tracked_object.pose;
      const topPt = project3D([p.x_m, p.y_m, p.z_center_m + state.tracked_object.extents_m[2]*0.6 + 0.03]);
      if (topPt) {
        ctx.fillStyle = '#22c55e';
        ctx.font = 'bold 12px system-ui';
        ctx.fillText('G1 PICK TARGET (' + (p.x_m*100).toFixed(1) + ', ' + (p.y_m*100).toFixed(1) + ' cm)', topPt[0] - 65, topPt[1]);
      }
    }
  }

  // 4. Tactile 3D Point Cloud
  if (state.cloud && state.cloud.points) {
    for (const pt of state.cloud.points) {
      const sp = project3D(pt);
      if (!sp) continue;
      const hFrac = Math.min(1, Math.max(0, (pt[2] - zT) / 0.15));
      const r = Math.round(255 * (1 - 0.55 * hFrac));
      const g = Math.round(120 + 135 * hFrac);
      ctx.fillStyle = `rgb(${r},${g},65)`;
      const rad = Math.max(2.2, Math.min(5.0, 2.6 / sp[2]));
      ctx.beginPath();
      ctx.arc(sp[0], sp[1], rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 5. Left & Right Rokoko Gloved Hands
  if (state.hands) {
    for (const [side, col] of [['left', '#ff8b3d'], ['right', '#38d9f5']]) {
      const h = state.hands[side];
      if (!h) continue;
      for (const [a, b] of h.bones) drawLine3D(a, b, col, 2.6);
      for (const [fname, padPt] of Object.entries(h.pads)) {
        const sp = project3D(padPt);
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

  const pPill = document.getElementById('phase-pill');
  pPill.textContent = 'PHASE: ' + state.phase.toUpperCase();
  pPill.className = 'pill phase-' + state.phase;

  document.getElementById('btn-trace').className = state.gesture?.stream_index ? 'active-toggle' : '';
  const phaseBtn = document.getElementById('btn-phase');
  if (state.phase === 'scanning') phaseBtn.textContent = '[Space] Lock Grasp to Pick Up (✌️)';
  else if (state.phase === 'carrying') phaseBtn.textContent = '[Space] Object Placed -> Run G1 (✌️)';
  else phaseBtn.textContent = 'Placed! Press [R] to Scan Next Object';

  document.getElementById('m-pts').textContent = state.cloud.n_voxels;
  document.getElementById('m-h').textContent = state.cloud.height_mm.toFixed(0);
  document.getElementById('m-lw').textContent = `${state.cloud.length_mm.toFixed(0)}×${state.cloud.width_mm.toFixed(0)}`;
  document.getElementById('m-vol').textContent = state.hull ? state.hull.volume_cm3.toFixed(0) : '0';

  const pw = state.cloud.primitive_weights || {};
  document.getElementById('prim-line').textContent =
    `Primitive BIC: cyl ${((pw.cylinder||0)*100).toFixed(0)}% · box ${((pw.box||0)*100).toFixed(0)}% · sph ${((pw.sphere||0)*100).toFixed(0)}%` +
    (state.cloud.calipers_mm.length ? ` | calipers: ${state.cloud.calipers_mm.join(', ')} mm` : '');

  const barsEl = document.getElementById('catalog-bars');
  const probs = state.catalog.probabilities || {};
  const items = state.catalog.items || {};
  barsEl.innerHTML = Object.keys(items).map(k => {
    const pct = ((probs[k] || 0) * 100);
    const info = items[k];
    return `<div class="bar-row">
      <div class="bar-head"><span><b>${info.label}</b> <small style="color:#9aa4b8">(${info.mass_g}g, max ${info.max_grip_n}N)</small></span><span>${pct.toFixed(0)}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');
  document.getElementById('match-advice').textContent = state.catalog.advice || '';

  if (state.g1_plan) {
    const g1 = state.g1_plan;
    const mj = state.mujoco_result || {};
    document.getElementById('g1-summary').textContent =
      `MuJoCo Twin Gate : ${mj.outcome?.toUpperCase()} (squeeze ${mj.peak_squeeze_n} N / max ${g1.object.max_safe_grip_n} N)\n` +
      `Unitree G1 Arm   : ${g1.arm.toUpperCase()} (safe_to_execute=${g1.safe_to_execute})\n` +
      `G1 Base Target   : [${g1.object_pose_g1_base.position_m.join(', ')}] m, yaw=${g1.object_pose_g1_base.yaw_deg}°\n` +
      `Dex3-1 q_grasp   : [${g1.dex3_1_hand.q_grasp_rad.join(', ')}]\n` +
      `Dex3-1 tau_ff Nm : [${g1.dex3_1_hand.tau_ff_nm.join(', ')}]\n` +
      `Saved JSON       : ${state.session_dir}/unitree_g1_pick.json`;
  } else if (state.phase === 'carrying' && state.tracked_object) {
    const p = state.tracked_object.pose;
    document.getElementById('g1-summary').textContent =
      `CARRYING IN HAND: (${(p.x_m*100).toFixed(1)}, ${(p.y_m*100).toFixed(1)}, ${(p.z_center_m*100).toFixed(1)}) cm | yaw ${p.yaw_deg}°\n` +
      `Set object anywhere on flat surface & trigger PLACED (✌️ or Space).`;
  }

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
