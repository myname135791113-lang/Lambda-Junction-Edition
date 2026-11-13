"""Self-contained HTML report for the visual test: inline SVG charts with hover, plus GIFs."""
from __future__ import annotations

import html
import json
import math
import shutil
from pathlib import Path

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


_ICON = {  # 16px line icons, drawn in currentColor (no unicode glyphs)
    "good": '<path d="M3.5 8.5l3 3 6-7"/>',
    "serious": '<path d="M8 3v9M4 8.5l4 4 4-4"/>',
    "critical": '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
}
STATUS = {"lifted": ("good", "Lifted safely"), "dropped": ("serious", "Dropped"), "crushed": ("critical", "Crushed")}


def badge(outcome: str) -> str:
    cls, text = STATUS[outcome]
    icon = (f'<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON[cls]}</svg>')
    return f'<span class="badge {cls}">{icon}{text}</span>'


FONT_DIR = Path(__file__).parent / "assets" / "fonts"


def write_assets(out: Path) -> None:
    """Copy the self-hosted fonts next to the report (the page falls back to system fonts without them)."""
    dest = out / "assets" / "fonts"
    dest.mkdir(parents=True, exist_ok=True)
    for f in FONT_DIR.iterdir():
        if f.suffix in (".woff2", ".txt"):
            shutil.copyfile(f, dest / f.name)


CSS = """
@font-face{font-family:"Schibsted Grotesk";src:url("assets/fonts/schibsted-grotesk-latin-wght.woff2") format("woff2");font-weight:400 900;font-display:swap}
@font-face{font-family:"IBM Plex Mono";src:url("assets/fonts/ibm-plex-mono-latin-400.woff2") format("woff2");font-weight:400;font-display:swap}
@font-face{font-family:"IBM Plex Mono";src:url("assets/fonts/ibm-plex-mono-latin-500.woff2") format("woff2");font-weight:500;font-display:swap}
:root{color-scheme:light;
--page:#fffaf3;--surface:#ffffff;--sunk:#fbf3e8;--ink:#1d150e;--ink2:#56493d;--muted:#75685c;
--grid:#f1e7da;--axis:#d9c9b6;--border:#eadcca;--hair:#e4d4c0;
--tangerine:#f47a16;--tangerine-ink:#b04f06;--tangerine-wash:#fff0e2;--lime:#cdf23a;--lime-wash:#f2fbcf;
--s1:#ee6f0e;--s2:#2b241d;--s3:#c4ea2a;--s4:#c79a5c;
--good:#13804a;--good-ink:#0d6b3c;--good-wash:#e7f5ec;--serious:#a8650a;--serious-ink:#7f4a03;--serious-wash:#fbf0dc;
--critical:#c62f2f;--critical-ink:#a82424;--critical-wash:#fdeaea;
--band:rgba(244,122,22,.10);--mark:var(--lime);--shadow:0 1px 2px rgba(74,40,8,.06),0 4px 12px -6px rgba(74,40,8,.14);
--sans:"Schibsted Grotesk",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,"Cascadia Mono",Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#15110d;--surface:#1e1913;--sunk:#191410;--ink:#f7efe5;--ink2:#cdbfae;--muted:#a3937f;
--grid:#2b241d;--axis:#3d342a;--border:#2f2820;--hair:#3a3027;
--tangerine:#ff8a2b;--tangerine-ink:#ffa257;--tangerine-wash:#2c1d10;--lime:#d4f542;--lime-wash:#252b0e;
--s1:#ff8a2b;--s2:#efe5d8;--s3:#d4f542;--s4:#d6ab70;
--good:#3fbf78;--good-ink:#6fd69b;--good-wash:#12271b;--serious:#e0a040;--serious-ink:#f0bd6c;--serious-wash:#2b2010;
--critical:#ef6060;--critical-ink:#ff8a8a;--critical-wash:#311616;
--band:rgba(255,138,43,.14);--mark:rgba(212,245,66,.26);--shadow:0 1px 2px rgba(0,0,0,.35),0 4px 12px -6px rgba(0,0,0,.5)}}
:root[data-theme="dark"]{color-scheme:dark;
--page:#15110d;--surface:#1e1913;--sunk:#191410;--ink:#f7efe5;--ink2:#cdbfae;--muted:#a3937f;
--grid:#2b241d;--axis:#3d342a;--border:#2f2820;--hair:#3a3027;
--tangerine:#ff8a2b;--tangerine-ink:#ffa257;--tangerine-wash:#2c1d10;--lime:#d4f542;--lime-wash:#252b0e;
--s1:#ff8a2b;--s2:#efe5d8;--s3:#d4f542;--s4:#d6ab70;
--good:#3fbf78;--good-ink:#6fd69b;--good-wash:#12271b;--serious:#e0a040;--serious-ink:#f0bd6c;--serious-wash:#2b2010;
--critical:#ef6060;--critical-ink:#ff8a8a;--critical-wash:#311616;
--band:rgba(255,138,43,.14);--mark:rgba(212,245,66,.26);--shadow:0 1px 2px rgba(0,0,0,.35),0 4px 12px -6px rgba(0,0,0,.5)}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scrollbar-color:var(--axis) var(--page);accent-color:var(--tangerine);caret-color:var(--tangerine)}
::selection{background:var(--lime);color:#1d150e}
body{margin:0;background:var(--page);color:var(--ink);font:400 15.5px/1.6 var(--sans);
-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
main{max-width:1180px;margin:0 auto;padding:clamp(56px,9vw,104px) 20px 72px}
h1,h2,h3{font-weight:620;letter-spacing:-.018em;text-wrap:balance;color:var(--ink)}
h1{font-size:clamp(34px,5.4vw,60px);line-height:1.02;letter-spacing:-.034em;font-weight:640;margin:0 0 24px}
h1 span{display:block;color:var(--tangerine-ink);font-weight:520}
h2{font-size:clamp(22px,2.4vw,27px);line-height:1.15;margin:56px 0 10px}
h2.part{position:relative;margin:clamp(64px,8vw,96px) 0 14px;padding-top:26px;border-top:1px solid var(--ink)}
h2.part::before{content:"";position:absolute;left:0;top:-3px;width:56px;height:5px;background:var(--tangerine);border-radius:0 0 2px 2px}
h3{font-size:17px;line-height:1.3;margin:0}
.lede{color:var(--ink2);max-width:68ch;margin:0 0 22px;font-size:16.5px;text-wrap:pretty}
.lede b{color:var(--ink);font-weight:620;background:linear-gradient(transparent 56%,var(--mark) 56%,var(--mark) 90%,transparent 90%);
padding:0 1px;-webkit-box-decoration-break:clone;box-decoration-break:clone}
a{color:var(--tangerine-ink);text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;
text-decoration-color:color-mix(in srgb,var(--tangerine-ink) 45%,transparent);transition:text-decoration-color .2s,background-color .2s}
a:hover{text-decoration-color:currentColor;background:var(--lime-wash)}
:focus-visible{outline:2px solid var(--tangerine);outline-offset:3px;border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin:26px 0 30px;
border-top:1px solid var(--ink);border-bottom:1px solid var(--hair)}
.tile{padding:18px 20px 20px 0;position:relative}
.tile+.tile{padding-left:20px;border-left:1px solid var(--hair)}
.tile .v{font-size:clamp(34px,3.6vw,44px);font-weight:600;letter-spacing:-.03em;font-variant-numeric:lining-nums;white-space:nowrap;line-height:1;margin-bottom:10px}
.tile:first-child .v{display:inline-block;background:linear-gradient(transparent 70%,var(--mark) 70%);padding-right:2px}
.tile .k{color:var(--ink2);font-size:13.5px;line-height:1.4;max-width:24ch}
@media (max-width:640px){.tiles{grid-template-columns:1fr 1fr}.tile:nth-child(odd){padding-left:0;border-left:0}
.tile:nth-child(n+3){border-top:1px solid var(--hair)}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:clamp(16px,2.4vw,26px);margin:20px 0;box-shadow:var(--shadow)}
.card-head{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;justify-content:space-between;
margin:-2px 0 18px;padding-bottom:14px;border-bottom:1px solid var(--grid)}
.card-head h3{font-size:19px;letter-spacing:-.02em}
.card-head>span{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px}
.sub{color:var(--ink2);font-size:13px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:clamp(16px,2.4vw,28px)}
@media (max-width:900px){.grid2{grid-template-columns:minmax(0,1fr)}}
img.anim{width:100%;height:auto;border-radius:12px;border:1px solid var(--border);display:block;background:var(--sunk)}
.chart{margin:0 0 14px;position:relative}.chart figcaption{font-size:13.5px;font-weight:600;margin-bottom:4px}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font:11px var(--mono);font-variant-numeric:tabular-nums}.axis-label{fill:var(--muted);font-size:11px}
.line{fill:none;stroke-width:2.25;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface);stroke-width:2}
.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}.ref-label{fill:var(--ink2);font-size:11px}
.ref-label,.band-label{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.band{fill:var(--band)}.band-label{fill:var(--tangerine-ink);font-size:11px;font-weight:600}
.crosshair{stroke:var(--tangerine);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--page);border-radius:8px;
padding:7px 10px;font:12px/1.5 var(--mono);box-shadow:0 6px 18px -6px rgba(0,0,0,.35);white-space:nowrap;font-variant-numeric:tabular-nums;z-index:2}
.tip b{font-weight:500;color:var(--lime)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--ink2);margin:2px 0 6px}
.key{display:inline-block;width:18px;height:0;border-top:2.5px solid var(--c);vertical-align:middle;margin-right:7px}
.key.dashed{border-top-style:dashed}
.note{color:var(--muted);font-size:12.5px;margin:4px 0 0;max-width:62ch}
.bars{display:grid;gap:8px;margin:10px 0 14px}.bar-row{display:grid;grid-template-columns:118px 1fr 46px;gap:10px;align-items:center;font-size:13px}
.bar-name{color:var(--ink2)}
.bar-track{background:var(--grid);border-radius:999px;height:8px;overflow:hidden}.bar{display:block;height:100%;background:var(--s1);border-radius:999px}
.bar-row:first-child .bar-name{color:var(--ink);font-weight:600}
.bar-val{text-align:right;font:12.5px var(--mono);font-variant-numeric:tabular-nums;color:var(--ink2)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:600;font-size:12px;letter-spacing:.01em;border-bottom-color:var(--hair);vertical-align:bottom}
th:first-child,td:first-child{padding-left:0}th:last-child,td:last-child{padding-right:0}
td.num,th.num{text-align:right}td.num{font-family:var(--mono);font-size:12.5px}
tbody tr:last-child td{border-bottom:0}
tbody tr{transition:background-color .15s}tbody tr:hover{background:var(--sunk)}
@media (max-width:720px){table{display:block;overflow-x:auto;white-space:nowrap;-webkit-overflow-scrolling:touch}}
.badge{display:inline-flex;gap:5px;align-items:center;font-weight:600;font-size:12.5px;line-height:1;padding:5px 10px 5px 8px;border-radius:999px;white-space:nowrap}
.badge svg{flex:none}
.badge.good{color:var(--good-ink);background:var(--good-wash)}
.badge.serious{color:var(--serious-ink);background:var(--serious-wash)}
.badge.critical{color:var(--critical-ink);background:var(--critical-wash)}
.ab{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}
.ab div{font-size:13px;color:var(--ink2);background:var(--sunk);border-radius:10px;padding:10px 12px}
.ab b{color:var(--ink)}
@media (max-width:640px){.ab{grid-template-columns:1fr}}
.pill{display:inline-block;font:12px var(--mono);color:var(--ink2);border:1px solid var(--border);border-radius:999px;padding:2px 10px;margin:8px 4px 0 0}
details{margin-top:12px}summary{cursor:pointer;color:var(--ink2);font-size:13px;font-weight:600;width:max-content}
summary:hover{color:var(--tangerine-ink)}
.theme{position:fixed;top:14px;right:14px;z-index:5;display:inline-flex;align-items:center;gap:8px;font:600 12.5px var(--sans);
background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:999px;padding:8px 14px 8px 11px;cursor:pointer;
box-shadow:var(--shadow);transition:transform .15s,border-color .15s}
.theme::before{content:"";width:10px;height:10px;border-radius:50%;background:linear-gradient(90deg,var(--tangerine) 50%,var(--lime) 50%)}
.theme:hover{border-color:var(--tangerine)}.theme:active{transform:scale(.97)}
footer{color:var(--muted);font-size:13px;margin-top:64px;padding-top:18px;border-top:1px solid var(--hair);max-width:none}
.ortho{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:6px 0 12px}
.ortho svg{width:100%;height:auto;background:var(--sunk);border:1px solid var(--border);border-radius:10px}
.ortho .grid{stroke:var(--hair)}
.outline{fill:var(--tangerine-wash);fill-opacity:.55;stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4}
.pt{stroke:var(--ink);stroke-width:1;stroke-opacity:.55}
.dot-key{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--c);margin-right:7px;vertical-align:middle;box-shadow:0 0 0 1px rgba(29,21,14,.35)}
.note-box{background:var(--tangerine-wash);border:1px solid color-mix(in srgb,var(--tangerine) 28%,transparent);border-radius:12px;
padding:12px 16px;margin:14px 0;color:var(--ink2);font-size:13.5px;max-width:none}
.note-box b{color:var(--ink)}
code{font:12px var(--mono);background:var(--sunk);padding:2px 6px;border-radius:5px;border:1px solid var(--border)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media print{.theme{display:none}.card{box-shadow:none;break-inside:avoid}}
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
if(tb){const root=document.documentElement,
  isDark=()=>root.dataset.theme?root.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches,
  label=()=>{tb.textContent=isDark()?'Light mode':'Dark mode';tb.setAttribute('aria-pressed',String(isDark()))};
  label();
  tb.addEventListener('click',()=>{root.dataset.theme=isDark()?'light':'dark';label()});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(!root.dataset.theme)label()});}
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="theme-color" content="#fffaf3" media="(prefers-color-scheme: light)">'
            f'<meta name="theme-color" content="#15110d" media="(prefers-color-scheme: dark)">'
            f'<link rel="preload" href="assets/fonts/schibsted-grotesk-latin-wght.woff2" as="font" type="font/woff2" crossorigin>'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'
            f'<button class="theme" type="button">Toggle theme</button><main>{body}</main>'
            f'<script>{JS}</script></body></html>')
