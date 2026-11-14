"""index.html for a showcase folder: one card per run, video + outcome."""
from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """body{font-family:Segoe UI,Arial,sans-serif;background:#12151a;color:#e8eaed;margin:0;padding:24px}
h1{font-weight:600;margin:0 0 4px}p.sub{color:#9aa0a6;margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}
.card{background:#1b2027;border-radius:10px;padding:12px;border:1px solid #2a313a}
video{width:100%;border-radius:6px;background:#000}.t{display:flex;justify-content:space-between;align-items:center;margin:2px 0 8px}
.n{font-weight:600}.b{padding:2px 10px;border-radius:12px;font-size:13px;font-weight:600}
.SUCCESS{background:#1f6b3a}.SLIPPED{background:#8a6a12}.DROPPED,.BROKEN{background:#8a2a22}
.CAN_PICK{background:#1f6b3a}.TOO_FAR,.TOO_BIG,.TOO_HEAVY,.NO_SAFE_FORCE,.MISSING_INPUT{background:#4a4f57}
.m{color:#9aa0a6;font-size:13px;line-height:1.5}code{color:#c8d3e0}"""


def build(folder: Path, title: str, subtitle: str = "") -> Path:
    folder = Path(folder)
    items = []
    for rj in sorted(folder.glob("*/result.json")):
        d = json.loads(rj.read_text())
        items.append((rj.parent, d))
    cards = []
    for sub, d in items:
        name = d.get("name") or d.get("object") or sub.name
        sim = d.get("simulation") or (d if "outcome" in d else None)
        feas = d.get("feasibility") or {"verdict": d.get("verdict", ""), "detail": d.get("detail", "")}
        status = sim["outcome"] if sim else feas["verdict"]
        vid = f'<video controls loop muted preload="metadata" src="{sub.name}/video.mp4"></video>' if (sub / "video.mp4").exists() else ""
        lines = [f"<b>feasibility</b>: {html.escape(feas['verdict'])} - {html.escape(str(feas.get('detail', '')))}"]
        if sim:
            lines.append(f"peak contact <b>{sim['peak_contact_force_N']}</b> / limit {sim['max_force_N']} N &nbsp; "
                         f"slip <b>{sim['slip_mm']}</b> mm &nbsp; lifted <b>{sim['final_lift_height_mm']}</b> mm &nbsp; "
                         f"target grip {sim['target_grip_force_N']} N")
            g = d.get("grasp_used") or sim.get("grasp_used")
            if g:
                lines.append(f"grasp: <b>{g['kind']}</b>, approach {g['approach_yaw_deg']:+d} deg, width {g['object_width_mm']} mm")
            for n in sim.get("notes", []):
                lines.append(f"<i>{html.escape(n)}</i>")
        if d.get("source"):
            lines.append(f"<code>{html.escape(d['source'])}</code>")
        if d.get("object_extents_mm"):
            e = d["object_extents_mm"]
            lines.append(f"object {e[0]:.0f} x {e[1]:.0f} x {e[2]:.0f} mm")
        cards.append(f'<div class="card"><div class="t"><span class="n">{html.escape(name)}</span>'
                     f'<span class="b {status}">{status}</span></div>{vid}<div class="m">{"<br>".join(lines)}</div></div>')
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><p class='sub'>{html.escape(subtitle)}</p><div class='grid'>{''.join(cards)}</div></body></html>")
    out = folder / "index.html"
    out.write_text(page, encoding="utf-8")
    return out
