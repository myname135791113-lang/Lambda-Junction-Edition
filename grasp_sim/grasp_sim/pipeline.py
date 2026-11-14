"""Part 1 (instant feasibility check, no physics) and part 2 (simulated lift) glued together."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .card import Card, load_card_mesh
from .grasp import MAX_APERTURE, MAX_PAYLOAD_KG, kinds_for, plan_grasps
from .scene import build_scene
from .sim import Runner, SimResult

G = 9.81
SAFETY_FACTOR = 2.0        # on the friction demand
FORCE_HEADROOM = 0.85      # target and minimum must stay below this fraction of max_force_N
MIN_GRIP_N = 2.0           # lowest grip force the hand can regulate


def min_grip_force(mass_kg: float, mu: float, sf: float = SAFETY_FACTOR) -> float:
    """Grip force per opposing side (thumb side = finger side) so that friction on both carries the weight:
    2 * mu * F >= sf * m * g."""
    return sf * mass_kg * G / (2.0 * mu)


def check(card: Card) -> dict:
    """Return {"verdict": CAN_PICK | reason, "detail": str, ...}; with a grasp and target force when it can."""
    if card.missing:
        return {"verdict": "MISSING_INPUT", "detail": "missing: " + ", ".join(card.missing)}
    try:
        mesh = load_card_mesh(card)
    except Exception as e:                                              # unreadable mesh counts as missing input
        return {"verdict": "MISSING_INPUT", "detail": f"mesh could not be read: {e}"}
    pos = card.pos_m.copy()
    if card.mass_kg > MAX_PAYLOAD_KG:
        # report TOO_BIG/TOO_FAR first if they apply (spec order), so keep going but remember
        heavy = True
    else:
        heavy = False

    si = build_scene(mesh, [pos[0], pos[1], pos[2]], [1, 0, 0, 0], card.mass_kg, card.friction, card.hollow)
    grasps, best_miss = plan_grasps(si, mesh, pos[:2], max_n=200)
    if not grasps:
        return {"verdict": "TOO_FAR", "detail": f"no arm IK solution for any grasp (closest miss {best_miss:.0f} mm)"}
    rank = {k: i for i, k in enumerate(kinds_for(mesh))}
    fits = sorted((g for g in grasps if g.width_m <= MAX_APERTURE),
                  key=lambda g: (rank[g.kind], round(g.width_m / 0.01), abs(g.psi_deg)))
    if not fits:
        w = min(g.width_m for g in grasps) * 1000
        return {"verdict": "TOO_BIG", "detail": f"narrowest graspable width {w:.0f} mm > hand opening {MAX_APERTURE*1000:.0f} mm"}
    if heavy:
        return {"verdict": "TOO_HEAVY", "detail": f"{card.mass_kg*1000:.0f} g > {MAX_PAYLOAD_KG*1000:.0f} g (Dex3 rating)"}
    fmin = min_grip_force(card.mass_kg, card.friction)
    if fmin > FORCE_HEADROOM * card.max_force_N:
        return {"verdict": "NO_SAFE_FORCE",
                "detail": f"needs >= {fmin:.1f} N per side (SF {SAFETY_FACTOR}), damage limit {card.max_force_N:.1f} N"}
    # a real hand cannot regulate below ~2 N, and light objects need a firm hold against hand motion anyway
    target = float(min(max(1.25 * fmin, MIN_GRIP_N), FORCE_HEADROOM * card.max_force_N))
    g = fits[0]
    return {"verdict": "CAN_PICK", "detail": "ok", "grasp": g.summary(), "target_force_N": round(target, 2),
            "min_force_N": round(fmin, 2), "_grasp": g, "_si": si, "_mesh": mesh, "_fits": fits}


def run_lift(card: Card, verdict: dict, render: bool = False, size=(480, 640), grasp=None) -> SimResult:
    """Part 2. `verdict` must be a CAN_PICK result from check()."""
    g, mesh = grasp or verdict["_grasp"], verdict["_mesh"]
    pos = card.pos_m
    si = build_scene(mesh, [pos[0], pos[1], pos[2]], [1, 0, 0, 0], card.mass_kg, card.friction, card.hollow)
    r = Runner(si, g, verdict["target_force_N"], card.max_force_N, render=render, size=size)
    r.obj_xy0 = (pos[0], pos[1])
    res = r.run()
    res.grasp = g.summary()
    return res


_RANK = {"SUCCESS": 0, "SLIPPED": 1, "DROPPED": 2, "BROKEN": 3}


def diverse(fits, n):
    """Up to n candidate grasps that differ in family or in approach yaw by at least 20 degrees."""
    chosen = []
    for g in fits:
        if all(g.kind != c.kind or abs(g.psi_deg - c.psi_deg) >= 20 for c in chosen):
            chosen.append(g)
        if len(chosen) >= n:
            break
    return chosen


def run_best(card: Card, verdict: dict, attempts: int = 3, render: bool = False, size=(480, 640), log=None):
    """Try up to `attempts` different grasps headless (fast), stop at the first SUCCESS, then re-run the chosen one
    with video. Returns (SimResult, [(grasp summary, outcome), ...])."""
    cands = diverse(verdict["_fits"], max(1, attempts))
    tried, best, best_g = [], None, None
    for g in cands:
        r = run_lift(card, verdict, render=False, grasp=g)
        tried.append((g.summary(), r.outcome))
        if log:
            log(f"      attempt {len(tried)}/{len(cands)}: {g.kind} grasp, approach {g.psi_deg:+d} deg -> {r.outcome}")
        if best is None or _RANK[r.outcome] < _RANK[best.outcome]:
            best, best_g = r, g
        if r.outcome == "SUCCESS":
            break
    if render:
        best = run_lift(card, verdict, render=True, size=size, grasp=best_g)
    best.attempts = tried
    best.notes = ([f"{len(tried)} grasp attempt(s); used {best_g.kind} approach {best_g.psi_deg:+d} deg"]
                  if len(tried) > 1 else []) + best.notes
    return best, tried


def public(v: dict) -> dict:
    return {k: x for k, x in v.items() if not k.startswith("_")}
