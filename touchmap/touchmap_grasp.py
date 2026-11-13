#!/usr/bin/env python3
"""
touchmap_grasp.py — grasp snapshots -> point cloud -> box / cylinder / sphere, with a live 3D view.

Each "grasp" = 5 fingertip points (thumb, index, middle, ring, pinky) captured when you press a button.
Points accumulate in ONE world frame (the object must stay still). After every grasp, all three
templates are fitted (position + orientation + size are free), scored with a robust error + a penalty
for extra parameters (BIC), and turned into probabilities.

Sources:
  --source synthetic   fake grasps on a known object (no hardware needed)
  --source rokoko      Rokoko Studio "Custom Streaming" (UDP, JSON v4) — verify joint names with --dump

Trigger a grasp: press Enter in the terminal (or --auto N for every N s, or --serial PORT for an Arduino
button that prints "B"). Commands: Enter = grasp, u = undo last, r = reset, q = quit.

Outputs (sessions/<time>_<source>/):  grasps.jsonl, result.json (atomic), cloud.ply, view.rrd (optional)
Benchmark:  python touchmap_grasp.py --bench 30
"""
import argparse, json, math, os, queue, socket, sys, threading, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

SIGMA = 3.0                      # mm: expected error of one contact point (sensor + pad offset)
MODELS = ("box", "cylinder", "sphere")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COLORS = {"box": (230, 120, 60), "cylinder": (70, 140, 230), "sphere": (90, 190, 110)}


# ----------------------------------------------------------------------------- small geometry helpers
def unit(v):
    return v / np.linalg.norm(v)

def axis_from_angles(th, ph):
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])

def angles_from_axis(a):
    a = unit(a)
    return math.acos(float(np.clip(a[2], -1, 1))), math.atan2(a[1], a[0])

def perp_basis(a):
    """Two unit vectors u, v perpendicular to a (and to each other)."""
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(a, t))
    return u, np.cross(a, u)

def fit_circle_2d(xy):
    """Algebraic (Kasa) circle fit: fast, closed-form, good as a starting guess."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)
    c = sol[:2]
    return c, math.sqrt(max(sol[2] + c @ c, 1.0))

def fit_sphere_alg(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    c = sol[:3]
    return c, math.sqrt(max(sol[3] + c @ c, 1.0))

def pca_frame(P):
    """Columns = principal axes (largest spread first), forced to be a proper rotation."""
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    E = Vt.T
    if np.linalg.det(E) < 0:
        E[:, 2] *= -1
    return E


# ----------------------------------------------------------------------------- templates
# Each template has: full parameter vector x, a residual = signed distance of every point to its
# surface, starting guesses, bounds, a mesh for display, and a human-readable description.

class Sphere:
    name = "sphere"
    def free_mask(self, upright):            # center(3) + r
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
    name = "cylinder"                        # x = center(3), axis angles th, ph, radius, height
    def free_mask(self, upright):
        m = np.ones(7, bool)
        if upright:
            m[3:5] = False                       # axis fixed to world z
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
        return {"center_mm": x[:3].round(1).tolist(), "axis": axis_from_angles(x[3], x[4]).round(3).tolist(),
                "radius_mm": round(float(x[5]), 1), "height_mm": round(float(x[6]), 1)}
    def mesh(self, x, n=32):
        a = axis_from_angles(x[3], x[4]); u, v = perp_basis(a)
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
    name = "box"                             # x = center(3), rotation vector(3), dims(3)
    def free_mask(self, upright):
        m = np.ones(9, bool)
        if upright:
            m[3:5] = False                       # only yaw (rotation about world z) is free
        return m
    def residuals(self, x, P):
        R = Rot.from_rotvec(x[3:6]).as_matrix()
        q = np.abs((P - x[:3]) @ R) - x[6:9] / 2
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    def inits(self, P, upright):
        frames = []
        if upright:
            frames = [Rot.from_euler("z", a, degrees=True).as_matrix() for a in range(0, 90, 15)]
        else:
            E = pca_frame(P)
            for k in range(3):                    # each principal axis as "up", a few spins around it
                Ek = np.roll(E, k, axis=1)
                for a in (0, 30, 60):
                    frames.append(Ek @ Rot.from_euler("z", a, degrees=True).as_matrix())
        out = []
        for R in frames:
            L = (P - P.mean(0)) @ R
            mid = (L.min(0) + L.max(0)) / 2
            rv = Rot.from_matrix(R).as_rotvec()
            out.append(np.r_[P.mean(0) + R @ mid, rv, np.maximum(np.ptp(L, 0), 10)])
        return out
    def bounds(self):
        return np.r_[[-1e4] * 3, [-10] * 3, [5] * 3], np.r_[[1e4] * 3, [10] * 3, [600] * 3]
    def describe(self, x):
        R = Rot.from_rotvec(x[3:6]).as_matrix()
        return {"center_mm": x[:3].round(1).tolist(), "dims_mm": sorted(x[6:9].round(1).tolist(), reverse=True),
                "axes": R.T.round(3).tolist()}
    def mesh(self, x):
        R = Rot.from_rotvec(x[3:6]).as_matrix()
        c = np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)])
        V = x[:3] + (c * x[6:9]) @ R.T
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
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
def robust_deviance(r):
    """-2 log-likelihood with a Cauchy (heavy-tailed) error: quadratic for small errors, only logarithmic
    for outliers (e.g. a finger that didn't actually touch), so a few bad points can't decide the shape."""
    z = r / SIGMA
    return float(np.sum(np.log1p(z * z)))

def fit_template(t, P, upright):
    mask = t.free_mask(upright)
    lo, hi = t.bounds()
    best = None
    for x0 in t.inits(P, upright):
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        def f(xf, x0=x0):
            x = x0.copy(); x[mask] = xf
            return t.residuals(x, P)
        sol = least_squares(f, x0[mask], bounds=(lo[mask], hi[mask]), loss="cauchy", f_scale=SIGMA, max_nfev=300)
        if best is None or sol.cost < best[0]:
            x = x0.copy(); x[mask] = sol.x
            best = (sol.cost, x)
    x = best[1]
    r = t.residuals(x, P)
    k = int(mask.sum())
    dev = robust_deviance(r)
    return {"x": x, "k": k, "rms_mm": float(np.sqrt(np.mean(r ** 2))),
            "median_abs_mm": float(np.median(np.abs(r))), "bic": dev + k * math.log(len(P))}

def classify(P, upright=False):
    """Fit every template that has enough points, return probabilities and a suggestion."""
    fits = {}
    for name, t in TEMPLATES.items():
        k = int(t.free_mask(upright).sum())
        if len(P) >= k + 1:
            fits[name] = fit_template(t, P, upright)
    if len(fits) < len(TEMPLATES):          # comparing a fitted model against an unfitted one is meaningless
        return {"n_points": len(P), "label": None, "best": None, "probs": {}, "fits": fits,
                "advice": "need more points before comparing (one more grasp)"}
    bic = np.array([fits[m]["bic"] for m in fits])
    w = np.exp(-(bic - bic.min()) / 2)
    probs = dict(zip(fits, (w / w.sum()).tolist()))
    ranked = sorted(probs, key=probs.get, reverse=True)
    label = ranked[0] if probs[ranked[0]] >= 0.9 else None
    advice = "confident" if label else _advice(ranked[:2])
    return {"n_points": len(P), "label": label, "best": ranked[0], "probs": probs, "fits": fits, "advice": advice}

def _advice(top2):
    pair = set(top2)
    if pair == {"sphere", "cylinder"}:
        return "sphere vs cylinder: grasp much higher or lower than before (a sphere narrows, a cylinder doesn't)"
    if pair == {"box", "cylinder"}:
        return "box vs cylinder: grasp from a direction rotated ~45° (across a corner/edge)"
    if pair == {"box", "sphere"}:
        return "box vs sphere: spread the fingers wider across one face"
    return "grasp from a new direction"


# ----------------------------------------------------------------------------- synthetic source
class SyntheticObject:
    """A fixed object somewhere on a table, plus a generator of realistic-ish 5-finger grasps."""
    def __init__(self, shape=None, rng=None, tilted=False):
        self.rng = rng or np.random.default_rng()
        r = self.rng
        self.shape = shape or r.choice(MODELS)
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
        self.R = Rot.from_euler("zyx", [r.uniform(0, 360), *(r.uniform(-90, 90, 2) if tilted else (0, 0))],
                                degrees=True).as_matrix()
        self.center = np.array([r.uniform(300, 500), r.uniform(-150, 150), H / 2])   # sits on the table

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
        """5 fingertip points (thumb first). phalanges=True also returns the middle-phalanx contact of the
        4 fingers, which sits further around the object (finger wraps), missing more often (30%)."""
        r = self.rng
        phi = r.uniform(0, 2 * np.pi)
        lim = 0.6 * self.dims[0] if self.shape == "sphere" else 0.42 * self.H
        top = r.random() < 0.25
        if top:   # grasp near the top rim, fingers spread around it
            zc = (0.7 * self.dims[0]) if self.shape == "sphere" else self.H / 2 - 8
            angs = [phi] + [phi + np.pi + d for d in np.radians([-50, -17, 17, 50]) + r.normal(0, 0.1, 4)]
            zs = [zc] * 5
        else:     # wrap grasp: thumb on one side, fingers stacked on the other side
            zc = r.uniform(-lim, lim)
            angs = [phi] + list(phi + np.pi + r.normal(0, 0.35, 4))
            zs = [zc] + [zc + (1.5 - k) * 20 + r.normal(0, 3) for k in range(4)]
        pts, miss = [], [miss_prob] * 5
        if phalanges and not top:
            for k in range(1, 5):
                arc = 28.0 / max(self._radius_at(zs[k]), 20.0)      # ~28 mm further back along the surface
                angs.append(angs[k] - arc); zs.append(zs[k]); miss.append(0.3)
        for ang, z, mp in zip(angs, zs, miss):
            zmax = (self.dims[0] - 2) if self.shape == "sphere" else self.H / 2
            p = self._surface(ang, float(np.clip(z, -zmax, zmax)))
            p = p + r.normal(0, noise, 3)
            if r.random() < mp:                   # finger hovering, not touching
                p[:2] += unit(p[:2]) * r.uniform(5, 20)
            pts.append(self.R @ p + self.center)
        return np.array(pts)

    def truth(self):
        return {"shape": self.shape, "dims_mm": self.dims.round(1).tolist(), "center_mm": self.center.round(1).tolist()}


class SyntheticSource:
    def __init__(self, shape=None, tilted=False, seed=None):
        self.obj = SyntheticObject(shape, np.random.default_rng(seed), tilted)
        print("synthetic object (hidden truth):", self.obj.truth())
    def grasp(self):
        return self.obj.grasp()


# ----------------------------------------------------------------------------- Rokoko source
class RokokoSource:
    """Listens to Rokoko Studio Custom Streaming (UDP, JSON v4, optionally LZ4).
    v4: joint transforms are world space, metres, LEFT-handed Y-up -> we convert to right-handed Z-up mm.
    Joint names are matched loosely (contains hand side + 'tip'); run with --dump once and adjust
    FINGER_KEYS if your packets name joints differently."""
    FINGER_KEYS = {"thumb": ("thumb", "1"), "index": ("index", "2"), "middle": ("middle", "3"),
                   "ring": ("ring", "4"), "pinky": ("little", "pinky", "5")}

    def __init__(self, port, hand="right", dump=False):
        self.hand, self.latest, self.dump = hand.lower(), None, dump
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"listening for Rokoko UDP on port {port} ...")

    def _loop(self):
        while True:
            data, _ = self.sock.recvfrom(1 << 20)
            if data[:4] == b"\x04\x22\x4d\x18":
                import lz4.frame
                data = lz4.frame.decompress(data)
            frame = json.loads(data)
            if self.dump:
                print(json.dumps(frame)[:4000]); self.dump = False
                print("\njoint names:", sorted({n for n, _ in _walk_joints(frame)}))
            self.latest = frame

    def grasp(self):
        if self.latest is None:
            raise RuntimeError("no Rokoko data received yet")
        tips = {}
        for name, pos in _walk_joints(self.latest):
            n = name.lower()
            if "tip" not in n or self.hand not in n:
                continue
            for finger, keys in self.FINGER_KEYS.items():
                if any(k in n for k in keys):
                    tips.setdefault(finger, pos)
        missing = [f for f in FINGERS if f not in tips]
        if missing:
            raise RuntimeError(f"fingertips not found: {missing} (run with --dump to see joint names)")
        out = []
        for f in FINGERS:
            x, y, z = tips[f]
            out.append(1000.0 * np.array([x, z, y]))       # left-handed Y-up -> right-handed Z-up, m -> mm
        return np.array(out)


def _walk_joints(obj):
    """Yield (name, [x,y,z]) for every dict that looks like a joint with a name and a position."""
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("id")
        pos = obj.get("position")
        if isinstance(name, str) and pos is not None:
            if isinstance(pos, dict):
                pos = [pos.get("x"), pos.get("y"), pos.get("z")]
            yield name, [float(v) for v in pos[:3]]
        for k, v in obj.items():
            if isinstance(v, dict) and "position" in v and "name" not in v:
                p = v["position"]
                p = [p.get("x"), p.get("y"), p.get("z")] if isinstance(p, dict) else p
                yield k, [float(t) for t in p[:3]]
            else:
                yield from _walk_joints(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_joints(v)


# ----------------------------------------------------------------------------- files
class Session:
    def __init__(self, root, source):
        self.dir = Path(root) / time.strftime(f"%Y%m%d_%H%M%S_{source}")
        self.dir.mkdir(parents=True, exist_ok=True)
        print("saving to", self.dir)

    def write(self, grasps, result, truth=None):
        with open(self.dir / "grasps.jsonl", "w") as f:            # small: rewrite (keeps undo simple)
            for i, g in enumerate(grasps):
                f.write(json.dumps({"i": i, "t": g["t"], "fingers": FINGERS, "points_mm": g["p"].round(2).tolist()}) + "\n")
        doc = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_grasps": len(grasps), "n_points": result["n_points"],
               "label": result["label"], "best_guess": result.get("best"), "advice": result["advice"],
               "probabilities": {k: round(v, 4) for k, v in result["probs"].items()},
               "models": {m: {"params": TEMPLATES[m].describe(f["x"]), "rms_mm": round(f["rms_mm"], 2),
                              "free_params": f["k"], "bic": round(f["bic"], 2)} for m, f in result["fits"].items()}}
        if truth:
            doc["ground_truth"] = truth
        tmp = self.dir / "result.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2))
        os.replace(tmp, self.dir / "result.json")                   # atomic: readers never see half a file
        P = np.vstack([g["p"] for g in grasps]) if grasps else np.zeros((0, 3))
        with open(self.dir / "cloud.ply", "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
                    f"property float z\nend_header\n")
            np.savetxt(f, P, fmt="%.2f")


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
            rrb.Vertical(rrb.TimeSeriesView(origin="probability", name="P(shape) per grasp"),
                         rrb.TextDocumentView(origin="summary", name="Result")),
            column_shares=[3, 2])))
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
        ids = np.repeat(np.arange(len(grasps)), 5)
        hue = ids / max(len(grasps) - 1, 1)
        cols = np.c_[80 + 175 * hue, 80 + 60 * (1 - hue), 255 - 175 * hue].astype(np.uint8)
        radii = np.where(ids == len(grasps) - 1, 4.0, 2.5)
        rr.log("world/cloud", rr.Points3D(P, colors=cols, radii=radii))
        g = grasps[-1]["p"]
        rr.log("world/latest_grasp", rr.LineStrips3D([[g[0], p] for p in g[1:]], colors=[255, 200, 0], radii=0.8))
        best = result.get("best")
        for m in MODELS:
            if m not in result["fits"]:
                rr.log(f"world/fit/{m}", rr.Clear(recursive=False)); continue
            V, F = TEMPLATES[m].mesh(result["fits"][m]["x"])
            alpha = 150 if m == best else 25
            rr.log(f"world/fit/{m}", rr.Mesh3D(vertex_positions=V, triangle_indices=F,
                                               albedo_factor=(*COLORS[m], alpha)))
        for m in MODELS:
            rr.log(f"probability/{m}", rr.Scalars(result["probs"].get(m, 0.0)))
        lines = [f"## {result['label'].upper() if result['label'] else 'undecided'}",
                 f"grasps: {len(grasps)}   points: {len(P)}", ""]
        for m in sorted(result["probs"], key=result["probs"].get, reverse=True):
            f = result["fits"][m]
            lines.append(f"- **{m}** P={result['probs'][m]:.2f}  rms={f['rms_mm']:.1f} mm  "
                         f"{json.dumps({k: v for k, v in TEMPLATES[m].describe(f['x']).items() if k in ('radius_mm', 'height_mm', 'dims_mm')})}")
        lines += ["", f"_next:_ {result['advice']}"]
        rr.log("summary", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN))


# ----------------------------------------------------------------------------- main loop
def run(args):
    src = SyntheticSource(args.shape, args.tilted, args.seed) if args.source == "synthetic" else \
        RokokoSource(args.port, args.hand, args.dump)
    truth = src.obj.truth() if args.source == "synthetic" else None
    session = Session(args.out, args.source)
    view = None if args.no_view else LiveView(rrd=args.rrd or (str(session.dir / "view.rrd") if args.save_rrd else None))
    events = queue.Queue()

    def keyboard():
        for line in sys.stdin:
            events.put(line.strip().lower() or "g")
    threading.Thread(target=keyboard, daemon=True).start()
    if args.auto:
        def ticker():
            while True:
                time.sleep(args.auto); events.put("g")
        threading.Thread(target=ticker, daemon=True).start()
    if args.serial:
        import serial  # pyserial
        def button():
            with serial.Serial(args.serial, 115200, timeout=1) as s:
                while True:
                    if s.readline().strip() == b"B":
                        events.put("g")
        threading.Thread(target=button, daemon=True).start()

    grasps = []
    print("Enter = grasp, u = undo, r = reset, q = quit")
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
                grasps.append({"t": time.time(), "p": src.grasp()})
            except RuntimeError as e:
                print("!", e); continue
        P = np.vstack([g["p"] for g in grasps]) if grasps else np.zeros((0, 3))
        t0 = time.time()
        result = classify(P, args.upright) if len(P) else {"n_points": 0, "label": None, "probs": {}, "fits": {}, "advice": "grasp the object"}
        session.write(grasps, result, truth)
        if view:
            view.update(grasps, result)
        probs = "  ".join(f"{m}={p:.2f}" for m, p in result["probs"].items())
        print(f"[{len(grasps)} grasps, {len(P)} pts, fit {1000 * (time.time() - t0):.0f} ms] "
              f"label={result['label']}  {probs}  -> {result['advice']}")
        if args.auto and args.max_grasps and len(grasps) >= args.max_grasps:
            break


def bench(trials, max_grasps, upright, tilted, seed=0, phalanges=False):
    """How many grasps until the shape is recognised? Accuracy of argmax(P) after k grasps."""
    rng = np.random.default_rng(seed)
    correct = np.zeros((3, max_grasps)); confident_ok = np.zeros(max_grasps); confident_wrong = np.zeros(max_grasps)
    dim_err, confusion = [], np.zeros((3, 3), int)
    for si, shape in enumerate(MODELS):
        for _ in range(trials):
            obj = SyntheticObject(shape, rng, tilted)
            pts = []
            for k in range(max_grasps):
                pts.append(obj.grasp(phalanges=phalanges))
                res = classify(np.vstack(pts), upright)
                if not res["probs"]:
                    continue
                correct[si, k] += res["best"] == shape
                confident_ok[k] += res["label"] == shape
                confident_wrong[k] += res["label"] is not None and res["label"] != shape
            confusion[si, MODELS.index(res["best"])] += 1
            f = res["fits"][shape]["x"]
            est = {"sphere": lambda: [f[3]], "cylinder": lambda: [f[5]], "box": lambda: []}[shape]()
            if est:
                dim_err.append(abs(est[0] - obj.dims[0]))
    n = trials
    print(f"\nupright prior: {upright}   tilted: {tilted}   phalanx contacts: {phalanges}   trials per shape: {n}")
    print("grasps | " + " | ".join(f"{m:>8}" for m in MODELS) + " | all  | confident&right | confident&WRONG")
    for k in range(max_grasps):
        row = " | ".join(f"{correct[i, k] / n:8.0%}" for i in range(3))
        print(f"{k + 1:6d} | {row} | {correct[:, k].sum() / (3 * n):4.0%} | {confident_ok[k] / (3 * n):15.0%} | {confident_wrong[k] / (3 * n):6.0%}")
    print(f"confusion after {max_grasps} grasps (rows = truth, cols = guess {MODELS}):\n{confusion}")
    if dim_err:
        print(f"radius error (sphere+cylinder, after {max_grasps} grasps): median {np.median(dim_err):.1f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "rokoko"], default="synthetic")
    ap.add_argument("--shape", choices=MODELS, help="synthetic: force a shape (default random)")
    ap.add_argument("--tilted", action="store_true", help="synthetic: random object orientation")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--upright", action="store_true", help="prior: object stands upright on a table (faster, fewer grasps)")
    ap.add_argument("--port", type=int, default=14043, help="rokoko: UDP port set in Studio's custom streaming")
    ap.add_argument("--hand", default="right")
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
