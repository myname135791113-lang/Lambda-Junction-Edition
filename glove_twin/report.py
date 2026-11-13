"""Self-contained HTML report for the visual test: inline SVG charts with hover, plus GIFs."""
from __future__ import annotations

import html
import json
import math

import numpy as np

W, H = 560, 190
M = {"l": 52, "r": 14, "t": 12, "b": 34}


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    step = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= step * mult:
            step *= mult
            break
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step) + 1)]


def _fmt(v: float) -> str:
    if abs(v) >= 100 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}"


def _downsample(x: np.ndarray, ys: list[np.ndarray], max_pts: int = 700):
    if len(x) <= max_pts:
        return x, ys
    # min/max decimation keeps peaks of fast signals visible
    idx = np.linspace(0, len(x), max_pts // 2 + 1).astype(int)
    keep = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b <= a:
            continue
        seg = ys[0][a:b]
        keep += sorted({a + int(np.argmin(seg)), a + int(np.argmax(seg))})
    keep = np.array(keep)
    return x[keep], [y[keep] for y in ys]


def line_chart(title: str, x: np.ndarray, series: list[dict], *, x_label: str, y_unit: str,
               x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None,
               refs: list[dict] | None = None, bands: list[dict] | None = None,
               dots: dict | None = None, note: str = "") -> str:
    """series: [{"name", "y", "slot" (1..4), "dash": bool}]. One shared x (needed for the tooltip).

    refs:  horizontal reference lines [{"y", "label"}]
    bands: shaded x ranges [{"x0", "x1", "label"}]
    dots:  sampled points drawn as markers {"name", "x", "y", "slot"}
    """
    x = np.asarray(x, float)
    if x_range:
        sel = (x >= x_range[0]) & (x <= x_range[1])
        x = x[sel]
        series = [{**s, "y": np.asarray(s["y"], float)[sel]} for s in series]
    else:
        x_range = (float(x.min()), float(x.max()))
    xs, ys = _downsample(x, [np.asarray(s["y"], float) for s in series])
    all_y = np.concatenate(ys + ([np.asarray(dots["y"])] if dots else []))
    if y_range is None:
        lo, hi = float(all_y.min()), float(all_y.max())
        for r in refs or []:
            hi = max(hi, r["y"]); lo = min(lo, r["y"])
        pad = (hi - lo) * 0.08 or 1.0
        y_range = (lo - pad if lo < 0 else min(0.0, lo), hi + pad)
    pw, ph = W - M["l"] - M["r"], H - M["t"] - M["b"]
    sx = lambda v: M["l"] + (v - x_range[0]) / (x_range[1] - x_range[0]) * pw
    sy = lambda v: M["t"] + ph - (v - y_range[0]) / (y_range[1] - y_range[0]) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    for b in bands or []:
        x0, x1 = sx(max(b["x0"], x_range[0])), sx(min(b["x1"], x_range[1]))
        parts.append(f'<rect class="band" x="{x0:.1f}" y="{M["t"]}" width="{max(x1 - x0, 1.5):.1f}" height="{ph}"/>')
        parts.append(f'<text class="band-label" x="{x1 + 4:.1f}" y="{M["t"] + 12}">{html.escape(b["label"])}</text>')
    for t in _ticks(*y_range):
        if y_range[0] <= t <= y_range[1]:
            parts.append(f'<line class="grid" x1="{M["l"]}" x2="{W - M["r"]}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
            parts.append(f'<text class="tick" x="{M["l"] - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
    for t in _ticks(*x_range, n=6):
        if x_range[0] <= t <= x_range[1]:
            parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - M["b"] + 16}" text-anchor="middle">{_fmt(t)}</text>')
    parts.append(f'<line class="axis" x1="{M["l"]}" x2="{W - M["r"]}" y1="{sy(max(y_range[0], 0) if y_range[0] < 0 < y_range[1] else y_range[0]):.1f}" '
                 f'y2="{sy(max(y_range[0], 0) if y_range[0] < 0 < y_range[1] else y_range[0]):.1f}"/>')
    parts.append(f'<text class="axis-label" x="{M["l"] + pw / 2}" y="{H - 4}" text-anchor="middle">{html.escape(x_label)}</text>')
    parts.append(f'<text class="axis-label" x="12" y="{M["t"] + ph / 2}" text-anchor="middle" transform="rotate(-90 12 {M["t"] + ph / 2})">{html.escape(y_unit)}</text>')
    for r in refs or []:
        parts.append(f'<line class="ref" x1="{M["l"]}" x2="{W - M["r"]}" y1="{sy(r["y"]):.1f}" y2="{sy(r["y"]):.1f}"/>')
        parts.append(f'<text class="ref-label" x="{W - M["r"] - 4}" y="{sy(r["y"]) - 5:.1f}" text-anchor="end">{html.escape(r["label"])}</text>')
    for s, y in zip(series, ys):
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, y))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        parts.append(f'<polyline class="line" points="{pts}" style="stroke:var(--s{s["slot"]})"{dash}/>')
    if dots:
        for a, b in zip(dots["x"], dots["y"]):
            if x_range[0] <= a <= x_range[1]:
                parts.append(f'<circle class="dot" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4.5" style="fill:var(--s{dots["slot"]})">'
                             f'<title>{html.escape(dots["name"])}: {b:.1f} at {a:.1f} ms</title></circle>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{M["t"]}" y2="{M["t"] + ph}" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{M["l"]}" y="{M["t"]}" width="{pw}" height="{ph}"/>')
    parts.append("</svg>")

    payload = {"x": [round(float(v), 4) for v in xs], "xLabel": x_label, "unit": y_unit,
               "series": [{"name": s["name"], "y": [round(float(v), 4) for v in y]} for s, y in zip(series, ys)],
               "plot": {"l": M["l"], "w": pw, "W": W, "x0": x_range[0], "x1": x_range[1]}}
    legend = ""
    entries = [(s["name"], s["slot"], s.get("dash")) for s in series] + ([(dots["name"], dots["slot"], None)] if dots else [])
    if len(entries) >= 2:
        legend = '<div class="legend">' + "".join(
            f'<span><i class="key{" dashed" if d else ""}" style="--c:var(--s{slot})"></i>{html.escape(n)}</span>'
            for n, slot, d in entries) + "</div>"
    note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
    return (f'<figure class="chart" data-chart=\'{html.escape(json.dumps(payload), quote=True)}\'>'
            f'<figcaption>{html.escape(title)}</figcaption>{legend}{"".join(parts)}'
            f'<div class="tip" hidden></div>{note_html}</figure>')


def bars(probs: dict[str, float], top: int = 4) -> str:
    rows = []
    for name, p in list(probs.items())[:top]:
        rows.append(f'<div class="bar-row"><span class="bar-name">{html.escape(name.replace("_", " "))}</span>'
                    f'<span class="bar-track"><span class="bar" style="width:{max(p * 100, 0.8):.1f}%"></span></span>'
                    f'<span class="bar-val">{p:.0%}</span></div>')
    return '<div class="bars">' + "".join(rows) + "</div>"


def ortho_views(points: np.ndarray, model: str, theta: np.ndarray, kinds: list[str]) -> str:
    """Three orthographic views (front x-z, side y-z, top x-y) of the contacts in the fitted
    object frame, with the fitted primitive's outline. Pure SVG."""
    size, pad = 190, 16
    if model == "sphere":
        ext = np.full(3, theta[0])
    elif model == "cylinder":
        ext = np.array([theta[0], theta[0], theta[1] / 2])
    else:
        ext = np.asarray(theta) / 2
    lim = max(float(np.max(ext)), float(np.max(np.abs(points)))) * 1.15
    sc = (size / 2 - pad) / lim
    views = [("front (x-z)", 0, 2), ("side (y-z)", 1, 2), ("top (x-y)", 0, 1)]
    kind_slot = {"enclosure": 1, "top": 2, "caliper": 3}
    out = ['<div class="ortho">']
    for title, a, b in views:
        cx = cy = size / 2
        X = lambda v: cx + v * sc
        Y = lambda v: cy - v * sc
        el = [f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="{html.escape(title)}">',
              f'<line class="grid" x1="{pad}" x2="{size - pad}" y1="{cy}" y2="{cy}"/>',
              f'<line class="grid" x1="{cx}" x2="{cx}" y1="{pad}" y2="{size - pad}"/>']
        if model == "sphere" or (model == "cylinder" and (a, b) == (0, 1)):
            el.append(f'<circle class="outline" cx="{cx}" cy="{cy}" r="{ext[0] * sc:.1f}"/>')
        else:
            w, h = ext[a] * sc, ext[b] * sc
            el.append(f'<rect class="outline" x="{cx - w:.1f}" y="{cy - h:.1f}" width="{2 * w:.1f}" height="{2 * h:.1f}"/>')
        order = np.argsort(points[:, 3 - a - b])                # draw far points first
        for i in order:
            p, k = points[i], kinds[i]
            el.append(f'<circle class="pt" cx="{X(p[a]):.1f}" cy="{Y(p[b]):.1f}" r="3.6" '
                      f'style="fill:var(--s{kind_slot.get(k, 1)})"><title>{k}: '
                      f'{p[0] * 1000:.0f}, {p[1] * 1000:.0f}, {p[2] * 1000:.0f} mm</title></circle>')
        el.append(f'<text class="tick" x="{pad}" y="{pad - 3}">{html.escape(title)}</text></svg>')
        out.append("".join(el))
    out.append("</div>")
    legend = ('<div class="legend"><span><i class="dot-key" style="--c:var(--s1)"></i>enclosure</span>'
              '<span><i class="dot-key" style="--c:var(--s2)"></i>top grasp</span>'
              '<span><i class="dot-key" style="--c:var(--s3)"></i>caliper pinch</span>'
              '<span><i class="key dashed" style="--c:var(--ink2)"></i>fitted shape</span></div>')
    return legend + "".join(out)


STATUS = {"lifted": ("good", "&#10003;", "Lifted safely"), "dropped": ("serious", "&#8595;", "Dropped"),
          "crushed": ("critical", "&#10005;", "Crushed")}


def badge(outcome: str) -> str:
    cls, icon, text = STATUS[outcome]
    return f'<span class="badge {cls}"><b aria-hidden="true">{icon}</b>{text}</span>'


CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--good:#0ca30c;--serious:#ec835a;--critical:#d03b3b;--good-ink:#006300;
--band:rgba(42,120,214,.10);--border:#e1e0d9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good-ink:#0ca30c;--band:rgba(57,135,229,.16);--border:#2c2c2a}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good-ink:#0ca30c;--band:rgba(57,135,229,.16);--border:#2c2c2a}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2.part{font-size:22px;margin:48px 0 8px;padding-top:18px;border-top:2px solid var(--border)}h2{font-size:20px;margin:44px 0 6px}h3{font-size:17px;margin:0}
.lede{color:var(--ink2);max-width:780px;margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .v{font-size:34px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.1}
.tile .k{color:var(--ink2);font-size:13px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;margin:16px 0}
.card-head{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;justify-content:space-between;margin-bottom:12px}
.sub{color:var(--ink2);font-size:13px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
img.anim{width:100%;height:auto;border-radius:8px;border:1px solid var(--border);display:block;background:#ddd}
.chart{margin:0 0 10px;position:relative}.chart figcaption{font-size:13px;font-weight:600;margin-bottom:2px}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.axis-label{fill:var(--muted);font-size:11px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}
.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}.ref-label{fill:var(--ink2);font-size:11px}
.band{fill:var(--band)}.band-label{fill:var(--ink2);font-size:11px}
.crosshair{stroke:var(--ink2);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;
padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap;font-variant-numeric:tabular-nums;z-index:2}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin:2px 0 4px}
.key{display:inline-block;width:14px;height:0;border-top:2px solid var(--c);vertical-align:middle;margin-right:6px}
.key.dashed{border-top-style:dashed}
.note{color:var(--muted);font-size:12px;margin:2px 0 0}
.bars{display:grid;gap:6px;margin:6px 0 10px}.bar-row{display:grid;grid-template-columns:110px 1fr 44px;gap:8px;align-items:center;font-size:13px}
.bar-track{background:var(--grid);border-radius:4px;height:10px;overflow:hidden}.bar{display:block;height:100%;background:var(--s1);border-radius:4px}
.bar-val{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}th{color:var(--ink2);font-weight:600}
td.num,th.num{text-align:right}
.badge{display:inline-flex;gap:6px;align-items:center;font-weight:600;font-size:13px;padding:2px 10px;border-radius:999px;border:1.5px solid}
.badge b{font-size:13px}.badge.good{color:var(--good-ink);border-color:var(--good)}
.badge.serious{color:var(--ink);border-color:var(--serious)}.badge.critical{color:var(--critical);border-color:var(--critical)}
.ab{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0}.ab div{font-size:13px;color:var(--ink2)}
.pill{display:inline-block;font-size:12px;color:var(--ink2);border:1px solid var(--border);border-radius:999px;padding:1px 9px;margin:0 4px 4px 0}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
.theme{position:fixed;top:12px;right:12px;font:inherit;font-size:12px;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:6px;padding:4px 10px;cursor:pointer}
footer{color:var(--muted);font-size:13px;margin-top:40px}
.ortho{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:4px 0 8px}
.ortho svg{width:100%;height:auto;background:var(--page);border:1px solid var(--border);border-radius:8px}
.outline{fill:none;stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4}
.pt{stroke:var(--surface);stroke-width:1.5}
.dot-key{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--c);margin-right:6px;vertical-align:middle}
.note-box{border-left:3px solid var(--s1);padding:6px 12px;margin:10px 0;color:var(--ink2);font-size:13px;background:var(--surface)}
code{font-size:12px;background:var(--page);padding:1px 5px;border-radius:4px;border:1px solid var(--border)}
"""

JS = """
document.querySelectorAll('figure.chart').forEach(fig=>{
  const d=JSON.parse(fig.dataset.chart), svg=fig.querySelector('svg'), hit=fig.querySelector('.hit'),
        cross=fig.querySelector('.crosshair'), tip=fig.querySelector('.tip');
  if(!hit||!d.x.length) return;
  const fmt=v=>Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):v.toFixed(3);
  function show(ev){
    const r=svg.getBoundingClientRect(), k=d.plot.W/r.width, px=(ev.clientX-r.left)*k;
    const xv=d.plot.x0+(px-d.plot.l)/d.plot.w*(d.plot.x1-d.plot.x0);
    let i=0,best=1e18; d.x.forEach((v,j)=>{const e=Math.abs(v-xv); if(e<best){best=e;i=j}});
    const sxp=d.plot.l+(d.x[i]-d.plot.x0)/(d.plot.x1-d.plot.x0)*d.plot.w;
    cross.setAttribute('x1',sxp);cross.setAttribute('x2',sxp);cross.setAttribute('visibility','visible');
    tip.innerHTML='<b>'+d.xLabel+': '+fmt(d.x[i])+'</b>'+d.series.map(s=>'<br>'+s.name+': '+fmt(s.y[i])+' '+d.unit).join('');
    tip.hidden=false; const fx=(sxp/k), w=tip.offsetWidth;
    tip.style.left=Math.min(Math.max(fx+12,0),r.width-w)+'px'; tip.style.top='26px';
  }
  hit.addEventListener('mousemove',show);
  hit.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');tip.hidden=true});
});
const tb=document.querySelector('.theme');
tb&&tb.addEventListener('click',()=>{const r=document.documentElement, dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme=dark?'light':'dark';tb.textContent=dark?'Dark mode':'Light mode';});
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'
            f'<button class="theme" type="button">Toggle theme</button><main>{body}</main>'
            f'<script>{JS}</script></body></html>')
