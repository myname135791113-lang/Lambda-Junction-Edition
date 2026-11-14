#!/usr/bin/env python3
"""
touchmap_grasp.py — grasp snapshots -> point cloud -> box / cylinder / sphere, with a live 3D view.

Each "grasp" = 5 fingertip points (thumb, index, middle, ring, pinky) captured when you press a button.
Points accumulate in ONE world frame (the object must stay still). After every grasp, all three
templates are fitted (position + orientation + size are free), scored with a robust Cauchy error
(with adaptive MAD noise floor so real hardware jitter cannot cause overconfident wrong labels) + a
BIC complexity penalty, and turned into probabilities.

Sources:
  --source synthetic   fake grasps on a known object (no hardware needed)
  --source rokoko      Rokoko Studio "Custom Streaming" (UDP, JSON v4) — verify joint names with --dump

Trigger a grasp: press Enter in the terminal (or --auto N for every N s, or --serial PORT for an Arduino
button that prints "B"). Commands: Enter = grasp, u = undo last, r = reset, q = quit.

Outputs (sessions/<time>_<source>/):  grasps.jsonl, result.json, cloud.ply (all atomic), view.rrd (optional)
Benchmark:  python touchmap/touchmap_grasp.py --bench 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import sys
import threading
import time

import numpy as np

try:
    from scipy.optimize import least_squares as _scipy_least_squares
    from scipy.spatial.transform import Rotation as _ScipyRot
except ImportError:
    _scipy_least_squares = None
    _ScipyRot = None

SIGMA = 3.0                      # mm: nominal minimum error floor (sensor + pad offset)
MODELS = ("box", "cylinder", "sphere")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COLORS = {"box": (230, 120, 60), "cylinder": (70, 140, 230), "sphere": (90, 190, 110)}
LZ4_MAGIC = b"\x04\x22\x4d\x18"

# Strict token sets (camelCase / snake_case split) — no bare "1".."5" substring false matches!
FINGER_TOKENS: dict[str, set[str]] = {
    "thumb": {"thumb"},
    "index": {"index"},
    "middle": {"middle", "mid"},
    "ring": {"ring"},
    "pinky": {"little", "pinky"},
}


# ----------------------------------------------------------------------------- small geometry helpers
def unit(v, fallback=None):
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n < 1e-12:
        if fallback is not None:
            return np.asarray(fallback, dtype=float)
        out = np.zeros_like(arr, dtype=float)
        out[0] = 1.0
        return out
    return arr / n


def axis_from_angles(th, ph):
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def angles_from_axis(a):
    a = unit(a, fallback=[0.0, 0.0, 1.0])
    return math.acos(float(np.clip(a[2], -1, 1))), math.atan2(a[1], a[0])


def perp_basis(a):
    """Two unit vectors u, v perpendicular to a (and to each other)."""
    a = unit(a, fallback=[0.0, 0.0, 1.0])
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(a, t), fallback=[1.0, 0.0, 0.0])
    return u, np.cross(a, u)


def fit_circle_2d(xy):
    """Algebraic (Kasa) circle fit: fast, closed-form, good as a starting guess."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
    c = sol[:2]
    val = float(sol[2] + c @ c)
    return c, math.sqrt(val if math.isfinite(val) and val > 1.0 else 1.0)


def fit_sphere_alg(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c = sol[:3]
    val = float(sol[3] + c @ c)
    return c, math.sqrt(val if math.isfinite(val) and val > 1.0 else 1.0)


def pca_frame(P):
    """Columns = principal axes (largest spread first), forced to be a proper rotation."""
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    E = Vt.T
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1
    return E


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- SO(3) + Cauchy optimizer (SciPy + pure-NumPy fallback)
def _rotvec_to_mat(w: np.ndarray) -> np.ndarray:
    if _ScipyRot is not None:
        return _ScipyRot.from_rotvec(w).as_matrix()
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    if _ScipyRot is not None:
        return _ScipyRot.from_matrix(R).as_rotvec()
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-6:
        M = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(M), 0.0, None))
        i = int(np.argmax(k))
        v = M[i] / max(float(k[i]), 1e-12)
        return math.pi * v / np.linalg.norm(v)
    return (th / (2.0 * math.sin(th))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _euler_z_deg(ang_deg: float) -> np.ndarray:
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _euler_zyx_deg(angles_deg) -> np.ndarray:
    if _ScipyRot is not None:
        return _ScipyRot.from_euler("zyx", angles_deg, degrees=True).as_matrix()
    az, ay, ax = (math.radians(float(v)) for v in angles_deg)
    cz, sz = math.cos(az), math.sin(az)
    cy, sy = math.cos(ay), math.sin(ay)
    cx, sx = math.cos(ax), math.sin(ax)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def _cauchy_least_squares(f, x0_free: np.ndarray, lo_free: np.ndarray, hi_free: np.ndarray, sigma: float):
    """Minimize `0.5 * sigma^2 * sum(log(1 + (r/sigma)^2))` via SciPy or pure-NumPy IRLS LM."""
    if _scipy_least_squares is not None:
        sol = _scipy_least_squares(f, x0_free, bounds=(lo_free, hi_free), loss="cauchy", f_scale=sigma, max_nfev=300)
        return float(sol.cost), sol.x

    x = np.clip(x0_free.astype(float), lo_free + 1e-6, hi_free - 1e-6)
    k = len(x)
    lam = 1e-2
    eps = 1e-4

    def cost_of(r_vec: np.ndarray) -> float:
        z = r_vec / sigma
        return 0.5 * (sigma ** 2) * float(np.sum(np.log1p(z * z)))

    r = f(x)
    cur_cost = cost_of(r)
    for _ in range(28):
        z = r / sigma
        w = 1.0 / (1.0 + z * z)
        J = np.empty((len(r), k), dtype=float)
        for j in range(k):
            xp = x.copy()
            xp[j] = min(xp[j] + eps, float(hi_free[j]))
            step = xp[j] - x[j]
            if abs(step) < 1e-9:
                xp[j] = max(x[j] - eps, float(lo_free[j]))
                step = xp[j] - x[j]
            J[:, j] = (f(xp) - r) / step
        JW = J * w[:, None]
        try:
            dx = -np.linalg.solve(JW.T @ J + lam * np.eye(k), JW.T @ r)
        except np.linalg.LinAlgError:
            break
        xn = np.clip(x + dx, lo_free, hi_free)
        rn = f(xn)
        cn = cost_of(rn)
        if cn < cur_cost:
            rel = cur_cost - cn
            x, r, cur_cost = xn, rn, cn
            lam = max(lam * 0.5, 1e-6)
            if rel < 1e-4:
                break
        else:
            lam *= 4.0
            if lam > 1e4:
                break
    return cur_cost, x


# ----------------------------------------------------------------------------- templates
class Sphere:
    name = "sphere"

    def free_mask(self, upright):
        return np.ones(4, bool)

    def residuals(self, x, P):
        return np.linalg.norm(P - x[:3], axis=1) - x[3]

    def inits(self, P, upright):
        c, r = fit_sphere_alg(P)
        c2 = P.mean(0)
        return [np.r_[c, r], np.r_[c2, np.linalg.norm(P - c2, axis=1).mean() + 10]]

    def bounds(self):
        return np.r_[[-1e4] * 3, 5], np.r_[[1e4] * 3, 400]

    def describe(self, x):
        return {"center_mm": x[:3].round(1).tolist(), "radius_mm": round(float(x[3]), 1)}

    def mesh(self, x, n=24):
        th, ph = np.meshgrid(np.linspace(0, np.pi, n // 2 + 1), np.linspace(0, 2 * np.pi, n + 1), indexing="ij")
        V = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
        return x[:3] + x[3] * V, _grid_faces(n // 2 + 1, n + 1)


class Cylinder:
    name = "cylinder"

    def free_mask(self, upright):
        m = np.ones(7, bool)
        if upright:
            m[3:5] = False
        return m

    def residuals(self, x, P):
        a = axis_from_angles(x[3], x[4])
        d = P - x[:3]
        z = d @ a
        rad = np.linalg.norm(d - np.outer(z, a), axis=1)
        dx, dz = rad - x[5], np.abs(z) - x[6] / 2
        return np.minimum(np.maximum(dx, dz), 0) + np.hypot(np.maximum(dx, 0), np.maximum(dz, 0))

    def inits(self, P, upright):
        axes = [np.array([0, 0, 1.0])] if upright else list(pca_frame(P).T) + [np.array([0, 0, 1.0])]
        out = []
        for a in axes:
            u, v = perp_basis(a)
            c2, r = fit_circle_2d(np.c_[P @ u, P @ v])
            z = P @ a
            c = c2[0] * u + c2[1] * v + (z.min() + z.max()) / 2 * a
            th, ph = (0.0, 0.0) if upright else angles_from_axis(a)
            out.append(np.r_[c, th, ph, r, max(np.ptp(z) + 20, 20)])
        return out

    def bounds(self):
        return np.r_[[-1e4] * 3, -10, -10, 5, 5], np.r_[[1e4] * 3, 10, 10, 300, 800]

    def describe(self, x):
        return {
            "center_mm": x[:3].round(1).tolist(),
            "axis": axis_from_angles(x[3], x[4]).round(3).tolist(),
            "radius_mm": round(float(x[5]), 1),
            "height_mm": round(float(x[6]), 1),
        }

    def mesh(self, x, n=32):
        a = axis_from_angles(x[3], x[4])
        u, v = perp_basis(a)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = x[5] * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
        top, bot = x[:3] + ring + a * x[6] / 2, x[:3] + ring - a * x[6] / 2
        V = np.vstack([top, bot, x[:3] + a * x[6] / 2, x[:3] - a * x[6] / 2])
        F = []
        for i in range(n):
            j = (i + 1) % n
            F += [[i, j, n + i], [j, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        return V, np.array(F)


class Box:
    name = "box"

    def free_mask(self, upright):
        m = np.ones(9, bool)
        if upright:
            m[3:5] = False
        return m

    def residuals(self, x, P):
        R = _rotvec_to_mat(x[3:6])
        q = np.abs((P - x[:3]) @ R) - x[6:9] / 2
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)

    def inits(self, P, upright):
        frames = []
        if upright:
            frames = [_euler_z_deg(a) for a in range(0, 90, 15)]
        else:
            E = pca_frame(P)
            for k in range(3):
                Ek = np.roll(E, k, axis=1)
                for a in (0, 30, 60):
                    frames.append(Ek @ _euler_z_deg(a))
        out = []
        for R in frames:
            L = (P - P.mean(0)) @ R
            mid = (L.min(0) + L.max(0)) / 2
            rv = _mat_to_rotvec(R)
            out.append(np.r_[P.mean(0) + R @ mid, rv, np.maximum(np.ptp(L, 0), 10)])
        return out

    def bounds(self):
        return np.r_[[-1e4] * 3, [-10] * 3, [5] * 3], np.r_[[1e4] * 3, [10] * 3, [600] * 3]

    def describe(self, x):
        R = _rotvec_to_mat(x[3:6])
        return {
            "center_mm": x[:3].round(1).tolist(),
            "dims_mm": sorted(x[6:9].round(1).tolist(), reverse=True),
            "axes": R.T.round(3).tolist(),
        }

    def mesh(self, x):
        R = _rotvec_to_mat(x[3:6])
        c = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])
        V = x[:3] + (c * x[6:9]) @ R.T
        F = np.array([
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
            [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ])
        return V, F


def _grid_faces(rows, cols):
    F = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b = i * cols + j, i * cols + j + 1
            F += [[a, a + cols, b], [b, a + cols, b + cols]]
    return np.array(F)


TEMPLATES = {"box": Box(), "cylinder": Cylinder(), "sphere": Sphere()}


# ----------------------------------------------------------------------------- fitting + model choice
def robust_deviance(r, sigma: float = SIGMA):
    """-2 log-likelihood with a Cauchy (heavy-tailed) error."""
    z = r / max(float(sigma), 1e-6)
    return float(np.sum(np.log1p(z * z)))


def fit_template(t, P, upright):
    mask = t.free_mask(upright)
    lo, hi = t.bounds()
    best = None
    inits = t.inits(P, upright)
    for x0 in inits:
        if not np.all(np.isfinite(x0)):
            continue
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

        def f(xf, x0=x0):
            x = x0.copy()
            x[mask] = xf
            return t.residuals(x, P)

        try:
            cost_val, x_free = _cauchy_least_squares(f, x0[mask], lo[mask], hi[mask], SIGMA)
            if np.all(np.isfinite(x_free)) and (best is None or cost_val < best[0]):
                x = x0.copy()
                x[mask] = x_free
                best = (cost_val, x)
        except (ValueError, np.linalg.LinAlgError):
            continue

    if best is None:
        x = np.clip(np.nan_to_num(inits[0], nan=0.0, posinf=100.0, neginf=-100.0), lo + 1e-6, hi - 1e-6)
    else:
        x = best[1]

    r = t.residuals(x, P)
    k = int(mask.sum())
    dev = robust_deviance(r, SIGMA)
    return {
        "x": x,
        "r": r,
        "k": k,
        "rms_mm": float(np.sqrt(np.mean(r ** 2))),
        "median_abs_mm": float(np.median(np.abs(r))),
        "bic": dev + k * math.log(len(P)),
    }


def classify(P, upright=False):
    """Sanitize non-finite rows, fit templates, calibrate BIC scale via MAD floor, and score."""
    arr = np.asarray(P, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return {
            "n_points": 0, "label": None, "best": None, "probs": {}, "fits": {},
            "sigma_eff_mm": SIGMA, "advice": "grasp the object",
        }
    clean = arr[np.all(np.isfinite(arr), axis=1)]
    if len(clean) == 0:
        return {
            "n_points": 0, "label": None, "best": None, "probs": {}, "fits": {},
            "sigma_eff_mm": SIGMA, "advice": "no valid finite points yet (grasp the object)",
        }

    fits = {}
    for name, t in TEMPLATES.items():
        k = int(t.free_mask(upright).sum())
        if len(clean) >= k + 1:
            fits[name] = fit_template(t, clean, upright)

    if len(fits) < len(TEMPLATES):
        return {
            "n_points": len(clean), "label": None, "best": None, "probs": {},
            "fits": {m: {k: v for k, v in d.items() if k != "r"} for m, d in fits.items()},
            "sigma_eff_mm": SIGMA,
            "advice": "need more points before comparing (one more grasp)",
        }

    # Adaptive noise scale: floors at SIGMA (3 mm) on clean data, expands via MAD if real hardware is noisier
    best_mad = 1.4826 * min(f["median_abs_mm"] for f in fits.values())
    sigma_eff = max(SIGMA, best_mad)
    for f in fits.values():
        f["bic"] = robust_deviance(f["r"], sigma_eff) + f["k"] * math.log(len(clean))
        del f["r"]

    bic = np.array([fits[m]["bic"] for m in fits])
    w = np.exp(-(bic - bic.min()) / 2)
    probs = dict(zip(fits, (w / w.sum()).tolist()))
    ranked = sorted(probs, key=probs.get, reverse=True)
    label = ranked[0] if probs[ranked[0]] >= 0.9 else None
    advice = "confident" if label else _advice(ranked[:2])
    return {
        "n_points": len(clean),
        "label": label,
        "best": ranked[0],
        "probs": probs,
        "fits": fits,
        "sigma_eff_mm": round(float(sigma_eff), 2),
        "advice": advice,
    }


def _advice(top2):
    pair = set(top2)
    if pair == {"sphere", "cylinder"}:
        return "sphere vs cylinder: grasp much higher or lower than before (a sphere narrows, a cylinder doesn't)"
    if pair == {"box", "cylinder"}:
        return "box vs cylinder: grasp from a direction rotated ~45 deg (across a corner/edge)"
    if pair == {"box", "sphere"}:
        return "box vs sphere: spread the fingers wider across one face"
    return "grasp from a new direction"


# ----------------------------------------------------------------------------- synthetic source
class SyntheticObject:
    """A fixed object somewhere on a table, plus a generator of realistic-ish 5-finger grasps."""

    def __init__(self, shape=None, rng=None, tilted=False):
        self.rng = rng or np.random.default_rng()
        r = self.rng
        self.shape = shape or str(r.choice(MODELS))
        if self.shape == "box":
            self.dims = r.uniform([40, 40, 40], [150, 120, 200])
            H = self.dims[2]
        elif self.shape == "cylinder":
            self.dims = np.array([r.uniform(25, 60), r.uniform(60, 200)])
            H = self.dims[1]
        else:
            self.dims = np.array([r.uniform(30, 70)])
            H = 2 * self.dims[0]
        self.H = H
        self.R = _euler_zyx_deg(
            [r.uniform(0, 360), *(r.uniform(-90, 90, 2) if tilted else (0, 0))]
        )
        self.center = np.array([r.uniform(300, 500), r.uniform(-150, 150), H / 2])

    def _surface(self, ang, z):
        c, s = math.cos(ang), math.sin(ang)
        if self.shape == "box":
            a, b, _ = self.dims
            t = min(a / 2 / max(abs(c), 1e-9), b / 2 / max(abs(s), 1e-9))
            return np.array([t * c, t * s, z])
        if self.shape == "cylinder":
            return np.array([self.dims[0] * c, self.dims[0] * s, z])
        R = self.dims[0]
        rho = math.sqrt(max(R * R - z * z, 0))
        return np.array([rho * c, rho * s, z])

    def _radius_at(self, z):
        if self.shape == "sphere":
            return math.sqrt(max(self.dims[0] ** 2 - z * z, 1.0))
        return self.dims[0] if self.shape == "cylinder" else min(self.dims[:2]) / 2

    def grasp(self, noise=2.0, miss_prob=0.1, phalanges=False):
        r = self.rng
        phi = r.uniform(0, 2 * np.pi)
        lim = 0.6 * self.dims[0] if self.shape == "sphere" else 0.42 * self.H
        top = r.random() < 0.25
        if top:
            zc = (0.7 * self.dims[0]) if self.shape == "sphere" else self.H / 2 - 8
            angs = [phi] + [phi + np.pi + d for d in np.radians([-50, -17, 17, 50]) + r.normal(0, 0.1, 4)]
            zs = [zc] * 5
        else:
            zc = r.uniform(-lim, lim)
            angs = [phi] + list(phi + np.pi + r.normal(0, 0.35, 4))
            zs = [zc] + [zc + (1.5 - k) * 20 + r.normal(0, 3) for k in range(4)]
        pts, miss = [], [miss_prob] * 5
        if phalanges and not top:
            for k in range(1, 5):
                arc = 28.0 / max(self._radius_at(zs[k]), 20.0)
                angs.append(angs[k] - arc)
                zs.append(zs[k])
                miss.append(0.3)
        for ang, z, mp in zip(angs, zs, miss):
            zmax = (self.dims[0] - 2) if self.shape == "sphere" else self.H / 2
            p = self._surface(ang, float(np.clip(z, -zmax, zmax)))
            p = p + r.normal(0, noise, 3)
            if r.random() < mp:
                p[:2] += unit(p[:2], fallback=[1.0, 0.0]) * r.uniform(5, 20)
            pts.append(self.R @ p + self.center)
        return np.array(pts)

    def truth(self):
        return {
            "shape": self.shape,
            "dims_mm": self.dims.round(1).tolist(),
            "center_mm": self.center.round(1).tolist(),
        }


class SyntheticSource:
    def __init__(self, shape=None, tilted=False, seed=None):
        self.obj = SyntheticObject(shape, np.random.default_rng(seed), tilted)
        print("synthetic object (hidden truth):", self.obj.truth())

    def grasp(self):
        return self.obj.grasp()


# ----------------------------------------------------------------------------- Rokoko source
def _tokenize_joint_name(name: str) -> set[str]:
    """Split camelCase and snake_case/dot/slash joint names into lowercase word tokens."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return {tok.lower() for tok in re.findall(r"[A-Za-z]+|\d+", spaced)}


def extract_rokoko_tips_mm(frame: dict, hand: str = "right") -> np.ndarray:
    """Extract `(5, 3)` right-handed Z-up fingertip coordinates in mm from a Rokoko JSON frame.

    Uses strict token matching (never bare single digits '1'..'5') and checks for finite values
    and duplicate/ambiguous matches.
    """
    hand_tok = hand.lower()
    joints = list(_walk_joints(frame))
    any_side_present = any(
        ("left" in _tokenize_joint_name(n) or "right" in _tokenize_joint_name(n))
        for n, _ in joints
    )

    tips: dict[str, list[float]] = {}
    matched_names: dict[str, str] = {}
    for name, pos in joints:
        toks = _tokenize_joint_name(name)
        if "tip" not in toks:
            continue
        if any_side_present and hand_tok not in toks:
            continue
        for finger, f_toks in FINGER_TOKENS.items():
            if toks & f_toks:
                if finger in tips and matched_names[finger] != name:
                    raise RuntimeError(
                        f"ambiguous duplicate {finger} tip joints: {matched_names[finger]!r} vs {name!r}"
                    )
                tips[finger] = pos
                matched_names[finger] = name

    missing = [f for f in FINGERS if f not in tips]
    if missing:
        raise RuntimeError(f"fingertips not found: {missing} (run with --dump to see joint names)")

    out = []
    for f in FINGERS:
        x, y, z = tips[f]
        out.append(1000.0 * np.array([x, z, y], dtype=float))  # LH Y-up m -> RH Z-up mm
    arr = np.array(out, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("non-finite (NaN/Inf) fingertip coordinates in Rokoko frame")
    return arr


class RokokoSource:
    """Resilient UDP listener for Rokoko Studio Custom Streaming (JSON v4, optional LZ4)."""

    def __init__(self, port: int, hand: str = "right", dump: bool = False, max_age_s: float = 1.0):
        self.hand = hand.lower()
        self.dump = dump
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self.latest: dict | None = None
        self.last_recv_mono: float = 0.0
        self.last_error: str | None = None
        self._running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.25)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"listening for Rokoko UDP on port {port} ...")

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self.sock.recvfrom(1 << 20)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                if data[:4] == LZ4_MAGIC:
                    try:
                        import lz4.frame  # type: ignore
                    except ImportError:
                        with self._lock:
                            self.last_error = (
                                "received LZ4-compressed Rokoko packet, but `lz4` is not installed "
                                "(run `pip install lz4` or disable compression in Rokoko Studio)"
                            )
                        continue
                    data = lz4.frame.decompress(data)

                frame = json.loads(data.decode("utf-8"))
                if not isinstance(frame, dict):
                    raise ValueError("expected JSON object at packet root")
                if self.dump:
                    self.dump = False
                    print(json.dumps(frame)[:4000])
                    print("\njoint names:", sorted({n for n, _ in _walk_joints(frame)}))
                with self._lock:
                    self.latest = frame
                    self.last_recv_mono = time.monotonic()
                    self.last_error = None
            except Exception as exc:
                with self._lock:
                    self.last_error = f"bad Rokoko packet: {exc}"

    def grasp(self) -> np.ndarray:
        with self._lock:
            frame = self.latest
            recv_mono = self.last_recv_mono
            err = self.last_error

        if frame is None:
            detail = f" ({err})" if err else ""
            raise RuntimeError(f"no Rokoko data received yet{detail}")

        age = time.monotonic() - recv_mono
        if self.max_age_s > 0 and age > self.max_age_s:
            detail = f"; last error: {err}" if err else ""
            raise RuntimeError(f"Rokoko frame is stale ({age:.1f}s old > {self.max_age_s:.1f}s){detail}")

        return extract_rokoko_tips_mm(frame, self.hand)

    def close(self) -> None:
        self._running = False
        try:
            self.sock.close()
        except OSError:
            pass


def _walk_joints(obj):
    """Yield `(name, [x, y, z])` for every dict entry with a 3D `position`."""
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("id")
        pos = obj.get("position")
        if isinstance(name, str) and pos is not None:
            coords = _parse_pos3(pos)
            if coords is not None:
                yield name, coords
        for k, v in obj.items():
            if k == "position":
                continue
            if isinstance(v, dict) and "position" in v and "name" not in v and "id" not in v:
                coords = _parse_pos3(v.get("position"))
                if coords is not None:
                    yield str(k), coords
            else:
                yield from _walk_joints(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_joints(v)


def _parse_pos3(pos) -> list[float] | None:
    try:
        if isinstance(pos, dict):
            vals = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            vals = [float(pos[0]), float(pos[1]), float(pos[2])]
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return vals


# ----------------------------------------------------------------------------- files
class Session:
    def __init__(self, root, source):
        root_p = Path(root)
        root_p.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}_{source}"
        candidate = root_p / stamp
        idx = 1
        while candidate.exists():
            candidate = root_p / f"{stamp}_{idx:02d}"
            idx += 1
        candidate.mkdir(parents=True, exist_ok=False)
        self.dir = candidate
        print("saving to", self.dir)

    def write(self, grasps, result, truth=None):
        jsonl_lines = [
            json.dumps({
                "i": i,
                "t": g["t"],
                "fingers": FINGERS[: len(g["p"])],
                "points_mm": np.asarray(g["p"], dtype=float).round(2).tolist(),
            })
            for i, g in enumerate(grasps)
        ]
        _atomic_write_text(self.dir / "grasps.jsonl", ("\n".join(jsonl_lines) + "\n") if jsonl_lines else "")

        doc = {
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_grasps": len(grasps),
            "n_points": result["n_points"],
            "label": result["label"],
            "best_guess": result.get("best"),
            "sigma_eff_mm": result.get("sigma_eff_mm", SIGMA),
            "advice": result["advice"],
            "probabilities": {k: round(v, 4) for k, v in result["probs"].items()},
            "models": {
                m: {
                    "params": TEMPLATES[m].describe(f["x"]),
                    "rms_mm": round(f["rms_mm"], 2),
                    "free_params": f["k"],
                    "bic": round(f["bic"], 2),
                }
                for m, f in result["fits"].items()
            },
        }
        if truth:
            doc["ground_truth"] = truth
        _atomic_write_text(self.dir / "result.json", json.dumps(doc, indent=2))

        P = np.vstack([g["p"] for g in grasps]) if grasps else np.zeros((0, 3))
        P = P[np.all(np.isfinite(P), axis=1)] if len(P) else P
        body = "\n".join(f"{row[0]:.2f} {row[1]:.2f} {row[2]:.2f}" for row in P)
        ply_text = (
            f"ply\nformat ascii 1.0\nelement vertex {len(P)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            + (body + "\n" if body else "")
        )
        _atomic_write_text(self.dir / "cloud.ply", ply_text)


# ----------------------------------------------------------------------------- live view (Rerun)
class LiveView:
    def __init__(self, rrd=None, spawn=True):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        rr.init("touchmap_grasp", spawn=spawn and rrd is None)
        if rrd:
            rr.save(rrd)
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="probability", name="P(shape) per grasp"),
                rrb.TextDocumentView(origin="summary", name="Result"),
            ),
            column_shares=[3, 2],
        )))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for m in MODELS:
            rr.log(f"probability/{m}", rr.SeriesLines(colors=[COLORS[m]], names=m), static=True)

    def update(self, grasps, result):
        rr = self.rr
        rr.set_time("grasp", sequence=len(grasps))
        if not grasps:
            for e in ("world/cloud", "world/latest_grasp", "world/fit"):
                rr.log(e, rr.Clear(recursive=True))
            return
        P = np.vstack([g["p"] for g in grasps])
        ids = np.concatenate([np.full(len(g["p"]), i, dtype=int) for i, g in enumerate(grasps)])
        hue = ids / max(len(grasps) - 1, 1)
        cols = np.c_[80 + 175 * hue, 80 + 60 * (1 - hue), 255 - 175 * hue].astype(np.uint8)
        radii = np.where(ids == len(grasps) - 1, 4.0, 2.5)
        rr.log("world/cloud", rr.Points3D(P, colors=cols, radii=radii))
        g = grasps[-1]["p"]
        if len(g) >= 2:
            rr.log("world/latest_grasp", rr.LineStrips3D([[g[0], p] for p in g[1:]], colors=[255, 200, 0], radii=0.8))
        best = result.get("best")
        for m in MODELS:
            if m not in result["fits"]:
                rr.log(f"world/fit/{m}", rr.Clear(recursive=False))
                continue
            V, F = TEMPLATES[m].mesh(result["fits"][m]["x"])
            alpha = 150 if m == best else 25
            rr.log(f"world/fit/{m}", rr.Mesh3D(vertex_positions=V, triangle_indices=F, albedo_factor=(*COLORS[m], alpha)))
        for m in MODELS:
            rr.log(f"probability/{m}", rr.Scalars(result["probs"].get(m, 0.0)))
        lines = [
            f"## {result['label'].upper() if result['label'] else 'undecided'}",
            f"grasps: {len(grasps)}   points: {len(P)}",
            "",
        ]
        for m in sorted(result["probs"], key=result["probs"].get, reverse=True):
            f = result["fits"][m]
            desc = {k: v for k, v in TEMPLATES[m].describe(f["x"]).items() if k in ("radius_mm", "height_mm", "dims_mm")}
            lines.append(f"- **{m}** P={result['probs'][m]:.2f}  rms={f['rms_mm']:.1f} mm  {json.dumps(desc)}")
        lines += ["", f"_next:_ {result['advice']}"]
        rr.log("summary", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


# ----------------------------------------------------------------------------- serial button helper
def start_serial_button_thread(port: str, events: queue.Queue, baud: int = 115200) -> threading.Thread:
    """Open Arduino button serial port up-front (fail fast if invalid) and watch for 'B' triggers."""
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise SystemExit("`--serial` requires pyserial (`pip install pyserial`)") from exc

    try:
        ser = serial.Serial(port, baud, timeout=1.0)
    except Exception as exc:
        raise SystemExit(f"[serial] could not open port {port!r}: {exc}") from exc

    def _button_loop() -> None:
        with ser:
            while True:
                try:
                    raw = ser.readline()
                except Exception as exc:
                    print(f"\n! [serial] read error on {port}: {exc}", file=sys.stderr)
                    break
                if raw.strip() == b"B":
                    events.put("g")

    t = threading.Thread(target=_button_loop, daemon=True)
    t.start()
    return t


# ----------------------------------------------------------------------------- main loop
def run(args):
    src = (
        SyntheticSource(args.shape, args.tilted, args.seed)
        if args.source == "synthetic"
        else RokokoSource(args.port, args.hand, args.dump, max_age_s=args.max_age)
    )
    truth = src.obj.truth() if args.source == "synthetic" else None
    session = Session(args.out, args.source)
    view = (
        None
        if args.no_view
        else LiveView(rrd=args.rrd or (str(session.dir / "view.rrd") if args.save_rrd else None))
    )
    events: queue.Queue = queue.Queue()

    def keyboard():
        for line in sys.stdin:
            events.put(line.strip().lower() or "g")

    threading.Thread(target=keyboard, daemon=True).start()
    if args.auto:
        def ticker():
            while True:
                time.sleep(args.auto)
                events.put("g")

        threading.Thread(target=ticker, daemon=True).start()
    if args.serial:
        start_serial_button_thread(args.serial, events)

    grasps = []
    print("Enter = grasp, u = undo, r = reset, q = quit")
    try:
        while True:
            ev = events.get()
            if ev == "q":
                break
            if ev == "u" and grasps:
                grasps.pop()
            elif ev == "r":
                grasps.clear()
            elif ev == "g":
                try:
                    pts = src.grasp()
                    if not np.all(np.isfinite(pts)):
                        raise ValueError("grasp contains non-finite coordinates")
                    grasps.append({"t": time.time(), "p": pts})
                except (RuntimeError, ValueError) as e:
                    print("!", e)
                    continue
            P = np.vstack([g["p"] for g in grasps]) if grasps else np.zeros((0, 3))
            t0 = time.time()
            result = (
                classify(P, args.upright)
                if len(P)
                else {"n_points": 0, "label": None, "best": None, "probs": {}, "fits": {}, "advice": "grasp the object"}
            )
            session.write(grasps, result, truth)
            if view:
                view.update(grasps, result)
            probs = "  ".join(f"{m}={p:.2f}" for m, p in result["probs"].items())
            print(
                f"[{len(grasps)} grasps, {len(P)} pts, fit {1000 * (time.time() - t0):.0f} ms] "
                f"label={result['label']}  {probs}  -> {result['advice']}"
            )
            if args.auto and args.max_grasps and len(grasps) >= args.max_grasps:
                break
    finally:
        if hasattr(src, "close"):
            src.close()


def bench(trials, max_grasps, upright, tilted, seed=0, phalanges=False):
    """How many grasps until the shape is recognised? Safe even when max_grasps == 1."""
    if max_grasps < 1 or trials < 1:
        raise ValueError("trials and max_grasps must be >= 1")
    rng = np.random.default_rng(seed)
    correct = np.zeros((3, max_grasps))
    confident_ok = np.zeros(max_grasps)
    confident_wrong = np.zeros(max_grasps)
    dim_err, confusion = [], np.zeros((3, 3), int)
    undecided_final = 0

    for si, shape in enumerate(MODELS):
        for _ in range(trials):
            obj = SyntheticObject(shape, rng, tilted)
            pts = []
            res = {"probs": {}, "best": None, "label": None, "fits": {}}
            for k in range(max_grasps):
                pts.append(obj.grasp(phalanges=phalanges))
                res = classify(np.vstack(pts), upright)
                if not res["probs"]:
                    continue
                correct[si, k] += res["best"] == shape
                confident_ok[k] += res["label"] == shape
                confident_wrong[k] += res["label"] is not None and res["label"] != shape

            if res.get("best") in MODELS:
                confusion[si, MODELS.index(res["best"])] += 1
            else:
                undecided_final += 1

            if shape in res.get("fits", {}):
                f = res["fits"][shape]["x"]
                est = {"sphere": lambda: [f[3]], "cylinder": lambda: [f[5]], "box": lambda: []}[shape]()
                if est:
                    dim_err.append(abs(est[0] - obj.dims[0]))

    n = trials
    print(f"\nupright prior: {upright}   tilted: {tilted}   phalanx contacts: {phalanges}   trials per shape: {n}")
    print("grasps | " + " | ".join(f"{m:>8}" for m in MODELS) + " | all  | confident&right | confident&WRONG")
    for k in range(max_grasps):
        row = " | ".join(f"{correct[i, k] / n:8.0%}" for i in range(3))
        print(
            f"{k + 1:6d} | {row} | {correct[:, k].sum() / (3 * n):4.0%} | "
            f"{confident_ok[k] / (3 * n):15.0%} | {confident_wrong[k] / (3 * n):6.0%}"
        )
    print(f"confusion after {max_grasps} grasps (rows = truth, cols = guess {MODELS}):\n{confusion}")
    if undecided_final:
        print(f"({undecided_final}/{3 * n} runs had too few points after {max_grasps} grasp(s) to fit all 3 models)")
    if dim_err:
        print(f"radius error (sphere+cylinder, after {max_grasps} grasps): median {np.median(dim_err):.1f} mm")
    return {"correct": correct, "confusion": confusion, "undecided_final": undecided_final}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "rokoko"], default="synthetic")
    ap.add_argument("--shape", choices=MODELS, help="synthetic: force a shape (default random)")
    ap.add_argument("--tilted", action="store_true", help="synthetic: random object orientation")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--upright", action="store_true", help="prior: object stands upright on a table (faster, fewer grasps)")
    ap.add_argument("--port", type=int, default=14043, help="rokoko: UDP port set in Studio's custom streaming")
    ap.add_argument("--hand", default="right")
    ap.add_argument("--max-age", type=float, default=1.0, help="rokoko: max allowed frame age in seconds")
    ap.add_argument("--dump", action="store_true", help="rokoko: print the first packet + joint names")
    ap.add_argument("--serial", help="Arduino port whose button prints 'B'")
    ap.add_argument("--auto", type=float, help="auto-grasp every N seconds (synthetic demo)")
    ap.add_argument("--max-grasps", type=int, default=0)
    ap.add_argument("--out", default="sessions")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--rrd", help="write the view to this .rrd file instead of opening the viewer")
    ap.add_argument("--save-rrd", action="store_true", help="also keep view.rrd in the session folder")
    ap.add_argument("--bench", type=int, metavar="TRIALS", help="run the synthetic benchmark and exit")
    ap.add_argument("--bench-grasps", type=int, default=6)
    ap.add_argument("--phalanges", action="store_true", help="synthetic: also use middle-phalanx contacts")
    a = ap.parse_args()
    if a.bench:
        bench(a.bench, a.bench_grasps, a.upright, a.tilted, phalanges=a.phalanges)
    else:
        run(a)
