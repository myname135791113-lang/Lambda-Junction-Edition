"""Fit a box, cylinder or sphere to touch contacts from several grasps, with gloves only.

Without a Rokoko Coil Pro the glove reports finger positions relative to the back of the hand, but
not where the hand is in the room. So contacts from different grasps live in different frames.

The trick: an object's dimensions don't depend on where the hand is. One joint fit estimates the
shared shape parameters theta plus an unknown pose T_g for every grasp g:

    x = R(w_g) p + t_g            (hand-frame contact -> object frame)
    residuals: signed distance of x to the primitive (sigma_p), and the angle between the measured
               contact normal and the surface normal (sigma_n ~ 20 deg; a bone's palmar axis is a
               weak normal estimate)

Solved by Levenberg-Marquardt with Huber weights (numpy only). The three models compete by BIC
(Schwarz 1978); differences below 6 count as ambiguous (Kass & Raftery 1995). A thumb-to-finger
caliper pinch is a grasp with two contacts: the distance between them is pose-independent, which is
the gloves-only way to measure height and length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIGMA_P = 0.002                    # m: Rokoko finger accuracy 1.6 mm typ + contact-model error
SIGMA_N = np.sin(np.radians(35))   # normal error: bone-axis normals were off 15-55 deg in simulation
SYSTEMATIC = 0.002                 # m: bias floor that averaging can't remove (bone length, tissue)
HUBER_K = 2.5
N_SHAPE = {"sphere": 1, "cylinder": 2, "box": 3}
PARAM_NAMES = {"sphere": ["radius"], "cylinder": ["radius", "height"], "box": ["x", "y", "z"]}


@dataclass
class Grasp:
    """Contacts from one grasp, in that grasp's hand (hub) frame."""
    points: np.ndarray             # (n, 3) m
    normals: np.ndarray            # (n, 3) estimated OUTWARD surface normals (unit)
    kind: str = "enclosure"        # "enclosure" | "caliper" | "top"
    segments: list = field(default_factory=list)


@dataclass
class ModelFit:
    model: str
    theta: np.ndarray              # sphere [r]; cylinder [r, h]; box [x, y, z] (object frame)
    poses: list                    # per grasp (w, t)
    chi2: float
    bic: float
    n_res: int
    sigma_theta: np.ndarray
    points_obj: np.ndarray         # all contacts mapped into the object frame
    normals_obj: np.ndarray
    grasp_index: np.ndarray        # which grasp each contact came from


@dataclass
class ShapeResult:
    best: str
    fits: dict                     # model -> ModelFit
    weights: dict                  # model -> BIC weight
    ambiguous: bool
    reasons: list
    dims: dict                     # reported dimensions (m), with observed-extent correction
    sigma: dict                    # 1-sigma uncertainty incl. systematic floor (m)
    coverage: dict                 # bin -> number of contacts
    touch_here: str


# ---------------------------------------------------------------- geometry helpers

def rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rot_to_vec(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:  # 180 degrees: take the axis from the symmetric part
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        k = M[i] / k[i]
        return np.pi * k / np.linalg.norm(k)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def sdf(model: str, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    if model == "sphere":
        return np.linalg.norm(x, axis=1) - theta[0]
    if model == "cylinder":
        r, h = theta
        q = np.stack([np.hypot(x[:, 0], x[:, 1]) - r, np.abs(x[:, 2]) - h / 2], axis=1)
    else:
        q = np.abs(x) - theta / 2
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    return outside + inside


def sdf_normal(model: str, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    e = 1e-5
    g = np.stack([(sdf(model, theta, x + e * d) - sdf(model, theta, x - e * d)) / (2 * e)
                  for d in np.eye(3)], axis=1)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def support_depth(model: str, theta: np.ndarray, n_obj: np.ndarray) -> np.ndarray:
    """Rough distance from the centre to the surface along each object-frame normal."""
    if model == "sphere":
        return np.full(len(n_obj), theta[0])
    if model == "cylinder":
        return np.where(np.abs(n_obj[:, 2]) > 0.7, theta[1] / 2, theta[0])
    return theta[np.argmax(np.abs(n_obj), axis=1)] / 2


# ---------------------------------------------------------------- residuals and LM

def _grasp_residuals(model, theta, w, t, g):
    R = rodrigues(w)
    x = g.points @ R.T + t
    n = g.normals @ R.T
    return np.concatenate([sdf(model, theta, x) / SIGMA_P,
                           ((n - sdf_normal(model, theta, x)) / SIGMA_N).ravel()])


def _residuals(model, theta, poses, grasps):
    return np.concatenate([_grasp_residuals(model, theta, w, t, g) for (w, t), g in zip(poses, grasps)])


def _huber(r):
    a = np.abs(r)
    return r * np.sqrt(np.where(a <= HUBER_K, 1.0, HUBER_K / np.maximum(a, 1e-12)))


def _joint_lm(model, theta, poses, grasps, *, fit_theta=True, iters=60):
    """Levenberg-Marquardt over theta (optional) and all grasp poses.

    Each grasp's residuals depend only on theta and that grasp's own pose, so the Jacobian is
    built block by block: n_shape full evaluations + 6 single-grasp evaluations per grasp.
    """
    theta = np.asarray(theta, float).copy()
    poses = [(np.asarray(w, float).copy(), np.asarray(t, float).copy()) for w, t in poses]
    ns = N_SHAPE[model] if fit_theta else 0
    sizes = [4 * len(g.points) for g in grasps]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    n_par = ns + 6 * len(grasps)

    def evaluate(th, ps):
        return [_huber(_grasp_residuals(model, th, w, t, g)) for (w, t), g in zip(ps, grasps)]

    blocks = evaluate(theta, poses)
    r = np.concatenate(blocks)
    cost = r @ r
    lam = 1e-3
    J = np.zeros((len(r), n_par))
    for _ in range(iters):
        J[:] = 0
        for j in range(ns):
            th = theta.copy()
            th[j] += 1e-6
            J[:, j] = (np.concatenate(evaluate(th, poses)) - r) / 1e-6
        for gi, ((w, t), g) in enumerate(zip(poses, grasps)):
            rows = slice(offs[gi], offs[gi + 1])
            for j in range(6):
                w2, t2 = w.copy(), t.copy()
                h = 1e-5 if j < 3 else 1e-6
                (w2 if j < 3 else t2)[j % 3] += h
                J[rows, ns + 6 * gi + j] = (_huber(_grasp_residuals(model, theta, w2, t2, g)) - blocks[gi]) / h
        A = J.T @ J
        grad = J.T @ r
        improved = False
        for _ in range(8):
            delta = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -grad)
            th_new = np.maximum(theta + delta[:ns], 0.004) if ns else theta
            ps_new = [(w + delta[ns + 6 * i:ns + 6 * i + 3], t + delta[ns + 6 * i + 3:ns + 6 * i + 6])
                      for i, (w, t) in enumerate(poses)]
            b_new = evaluate(th_new, ps_new)
            r_new = np.concatenate(b_new)
            c_new = r_new @ r_new
            if c_new < cost:
                rel = (cost - c_new) / max(cost, 1e-12)
                theta, poses, blocks, r, cost = th_new, ps_new, b_new, r_new, c_new
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 4
        if not improved or rel < 1e-7:
            break
    return theta, poses, J


# ---------------------------------------------------------------- initialisation

def caliper_distances(grasps: list[Grasp]) -> list[float]:
    """Pose-independent widths from two-contact pinches.

    Pinches recorded by the exploration routine (kind "caliper") are trusted; other two-contact
    grasps count only if their normals are roughly opposed. The straight distance between the two
    contacts is used: a pinch goes straight across, and noisy normals would shorten a projection.
    """
    out = []
    for g in grasps:
        if len(g.points) != 2:
            continue
        if g.kind == "caliper" or g.normals[0] @ g.normals[1] < -0.3:
            out.append(float(np.linalg.norm(g.points[0] - g.points[1])))
    return out


def _kasa_sphere(p: np.ndarray):
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 1e-8)))


def _circle_radius(p2: np.ndarray) -> float:
    A = np.c_[2 * p2, np.ones(len(p2))]
    b = (p2 ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.sqrt(max(sol[2] + sol[:2] @ sol[:2], 1e-8)))


def _frame_from_axis(a: np.ndarray) -> np.ndarray:
    """Rotation whose third row is a (maps a -> z)."""
    a = a / np.linalg.norm(a)
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, tmp)
    u /= np.linalg.norm(u)
    return np.vstack([u, np.cross(a, u), a])


def _signed_perms() -> list[np.ndarray]:
    out = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
            P = np.zeros((3, 3))
            P[[0, 1, 2], perm] = signs
            if np.linalg.det(P) > 0:
                out.append(P)
    return out


SIGNED_PERMS = _signed_perms()  # the 24 rotations of a cube


def _box_base_frame(n: np.ndarray) -> np.ndarray:
    cosang = n @ n.T
    votes = (cosang > np.cos(np.radians(25))).sum(axis=1)
    u1 = n[int(np.argmax(votes))]
    ortho = np.abs(n @ u1) < 0.35
    if ortho.any():
        cand = n[ortho]
        v2 = (cand @ cand.T > np.cos(np.radians(25))).sum(axis=1)
        u2 = cand[int(np.argmax(v2))]
        u2 = u2 - (u2 @ u1) * u1
        u2 /= np.linalg.norm(u2)
    else:
        u2 = _frame_from_axis(u1)[0]
    return np.vstack([u1, u2, np.cross(u1, u2)])


def _rotation_candidates(model: str, g: Grasp) -> list[np.ndarray]:
    if model == "sphere":
        return [np.eye(3)]
    if model == "cylinder":
        evals, evecs = np.linalg.eigh(g.normals.T @ g.normals)
        cands = []
        for i in range(3):
            F = _frame_from_axis(evecs[:, i])
            cands += [F, np.diag([1, -1, -1]) @ F]
        return cands
    B = _box_base_frame(g.normals)
    return [P @ B for P in SIGNED_PERMS]


def _init_translation(model, theta, R, g):
    n_obj = g.normals @ R.T
    d = support_depth(model, theta, n_obj)
    return np.mean(d[:, None] * n_obj - g.points @ R.T, axis=0)


def _init_theta(model: str, grasps: list[Grasp]) -> np.ndarray:
    cal = sorted(caliper_distances(grasps))
    encl = [g for g in grasps if len(g.points) >= 4]
    if model == "sphere":
        rs = [d / 2 for d in cal] + [_kasa_sphere(g.points)[1] for g in encl]
        return np.array([np.median(rs) if rs else 0.04])
    if model == "cylinder":
        rs = []
        for g in encl:
            evals, evecs = np.linalg.eigh(g.normals.T @ g.normals)
            F = _frame_from_axis(evecs[:, 0])
            rs.append(_circle_radius((g.points @ F.T)[:, :2]))
        r = float(np.median(rs)) if rs else (cal[0] / 2 if cal else 0.03)
        long = [d for d in cal if d > 2.3 * r]
        h = max(long) if long else 4 * r
        return np.array([r, h])
    dims = list(cal[-3:])
    for g in encl:
        if len(dims) >= 3:
            break
        B = _box_base_frame(g.normals)
        ext = np.ptp(g.points @ B.T, axis=0)
        dims += sorted(ext)[len(dims) - 3:] if len(dims) < 3 else []
    while len(dims) < 3:
        dims.append(dims[-1] if dims else 0.05)
    return np.sort(np.array(dims[:3]))


# ---------------------------------------------------------------- fitting

def _theta_starts(model: str, grasps: list[Grasp]) -> list[np.ndarray]:
    """Several starting shapes; the joint fit keeps whichever ends lowest."""
    base = _init_theta(model, grasps)
    starts = [base]
    spans = [np.max(np.linalg.norm(g.points[:, None] - g.points[None], axis=2)) for g in grasps if len(g.points) >= 3]
    longest = max(caliper_distances(grasps) + spans + [0.0])
    if model == "box" and longest > base[2] * 1.15:
        starts.append(np.sort([base[0], base[1], longest]))
    if model == "cylinder" and longest > base[1] * 1.15:
        starts.append(np.array([base[0], longest]))
    return starts


def _fit_from(model, grasps, theta, rounds):
    poses = [None] * len(grasps)
    J = None
    for _ in range(rounds):
        for i, g in enumerate(grasps):  # best pose per grasp for the current shape
            best = None
            for R in _rotation_candidates(model, g):
                pose0 = (rot_to_vec(R), _init_translation(model, theta, R, g))
                _, (pose,), _ = _joint_lm(model, theta, [pose0], [g], fit_theta=False, iters=12)
                r = _grasp_residuals(model, theta, pose[0], pose[1], g)
                c = float(np.sum(np.minimum(r ** 2, HUBER_K ** 2)))
                if best is None or c < best[0]:
                    best = (c, pose)
            poses[i] = best[1]
        theta, poses, J = _joint_lm(model, theta, poses, grasps, iters=60)
    r = _residuals(model, theta, poses, grasps)
    return float(np.sum(np.minimum(r ** 2, 9.0))), theta, poses, J


def fit_model(model: str, grasps: list[Grasp], rounds: int = 2, theta0=None) -> ModelFit:
    starts = _theta_starts(model, grasps) if theta0 is None else [np.asarray(theta0, float)]
    _, theta, poses, J = min((_fit_from(model, grasps, th, rounds) for th in starts), key=lambda x: x[0])

    r = _residuals(model, theta, poses, grasps)
    n = len(r)
    chi2 = float(np.sum(np.minimum(r ** 2, 9.0)))
    k = N_SHAPE[model] + 6 * len(grasps)
    bic = chi2 + k * np.log(n)
    try:
        # Unobservable directions (e.g. a pinch's roll about its own axis) are near-singular;
        # drop them instead of letting them leak into the shape uncertainty.
        cov = np.linalg.pinv(J.T @ J, rcond=1e-6, hermitian=True) * max(1.0, chi2 / max(n - k, 1))
        sig = np.sqrt(np.clip(np.diag(cov)[:N_SHAPE[model]], 0, None))
    except (np.linalg.LinAlgError, AttributeError, TypeError):
        sig = np.full(N_SHAPE[model], np.nan)
    pts, nrm, idx = [], [], []
    for i, ((w, t), g) in enumerate(zip(poses, grasps)):
        R = rodrigues(w)
        pts.append(g.points @ R.T + t)
        nrm.append(g.normals @ R.T)
        idx.append(np.full(len(g.points), i))
    return ModelFit(model, theta, poses, chi2, bic, n, sig, np.vstack(pts), np.vstack(nrm), np.concatenate(idx))


def _coverage(model: str, fit: ModelFit) -> dict:
    n = fit.normals_obj
    if model == "box":
        names = ["+x", "-x", "+y", "-y", "+z", "-z"]
        axis = np.argmax(np.abs(n), axis=1)
        sign = np.sign(n[np.arange(len(n)), axis])
        keys = [names[2 * a + (0 if s > 0 else 1)] for a, s in zip(axis, sign)]
        return {k: keys.count(k) for k in names}
    if model == "cylinder":
        cov = {"top cap": int(np.sum(n[:, 2] > 0.8)), "bottom cap": int(np.sum(n[:, 2] < -0.8))}
        side = np.abs(n[:, 2]) <= 0.8
        az = np.degrees(np.arctan2(n[side, 1], n[side, 0])) % 360
        for b in range(4):
            cov[f"side {b * 90}-{b * 90 + 90} deg"] = int(np.sum((az >= b * 90) & (az < b * 90 + 90)))
        return cov
    octs = [f"{'+' if sx else '-'}x{'+' if sy else '-'}y{'+' if sz else '-'}z"
            for sx in (1, 0) for sy in (1, 0) for sz in (1, 0)]
    keys = [f"{'+' if a[0] >= 0 else '-'}x{'+' if a[1] >= 0 else '-'}y{'+' if a[2] >= 0 else '-'}z" for a in n]
    return {k: keys.count(k) for k in octs}


def fit_shape(grasps: list[Grasp], models=("box", "cylinder", "sphere")) -> ShapeResult:
    if sum(len(g.points) for g in grasps) < 8:
        raise ValueError("Need at least 8 contacts in total")
    fits = {m: fit_model(m, grasps) for m in models}
    bics = {m: f.bic for m, f in fits.items()}
    best = min(bics, key=bics.get)
    lo = bics[best]
    raw = {m: np.exp(-(b - lo) / 2) for m, b in bics.items()}
    total = sum(raw.values())
    weights = {m: float(v / total) for m, v in sorted(raw.items(), key=lambda kv: -kv[1])}

    reasons = []
    second = sorted(bics.values())[1] if len(bics) > 1 else np.inf
    if second - lo < 6:
        reasons.append(f"BIC margin only {second - lo:.1f} (< 6)")
    fit = fits[best]
    coverage = _coverage(best, fit)
    theta = fit.theta.copy()
    sigma = np.sqrt(fit.sigma_theta ** 2 + SYSTEMATIC ** 2)
    if best == "cylinder":
        caps = coverage["top cap"] > 0 and coverage["bottom cap"] > 0
        pinched = any(abs(d - fit.theta[1]) < 0.15 * fit.theta[1] for d in caliper_distances(grasps))
        if coverage["top cap"] + coverage["bottom cap"] == 0:
            reasons.append("no end-cap touches: could be a short cylinder or a sphere")
        elif not (caps or pinched):
            reasons.append("height not measured: pinch from one end cap to the other")
        if not (caps or pinched):
            sigma[1] = np.inf

    cal = caliper_distances(grasps)
    if best == "box":
        # Contacts never lie outside the object, so on the long axis their extent is a lower
        # bound (sloped "Edge" tops). Capped, so one misplaced grasp cannot inflate it.
        k = int(np.argmax(theta))
        ext = np.ptp(fit.points_obj[:, k])
        theta[k] = min(max(theta[k], ext), theta[k] + 0.015)
        names = ["x", "y", "z"]
        dom_ax = np.argmax(np.abs(fit.normals_obj), axis=1)
        dom_sgn = np.sign(fit.normals_obj[np.arange(len(fit.normals_obj)), dom_ax])
        for ax in range(3):  # each extent needs opposite faces touched in the same grasp, or a pinch across it
            both = any(
                np.any((fit.grasp_index == gid) & (dom_ax == ax) & (dom_sgn > 0))
                and np.any((fit.grasp_index == gid) & (dom_ax == ax) & (dom_sgn < 0))
                for gid in range(len(grasps))
            )
            pinched = any(abs(d - theta[ax]) < 0.15 * theta[ax] for d in cal)
            if not (both or pinched):
                reasons.append(f"extent along {names[ax]} not measured: pinch across it")
                sigma[ax] = np.inf                       # a lower bound only, not a measurement
        order = np.argsort(theta)
        theta, sigma = theta[order], sigma[order]
        dims = {"width": theta[0], "depth": theta[1], "height": theta[2]}
        sig = {"width": sigma[0], "depth": sigma[1], "height": sigma[2]}
    elif best == "cylinder":
        dims = {"radius": theta[0], "height": max(theta[1], np.ptp(fit.points_obj[:, 2]))}
        sig = {"radius": sigma[0], "height": sigma[1]}
    else:
        dims = {"radius": theta[0]}
        sig = {"radius": sigma[0]}
    least = min(coverage, key=coverage.get)
    return ShapeResult(best=best, fits=fits, weights=weights, ambiguous=bool(reasons), reasons=reasons,
                       dims={k: float(v) for k, v in dims.items()},
                       sigma={k: float(v) for k, v in sig.items()},
                       coverage=coverage, touch_here=least)


# ---------------------------------------------------------------- synthetic sessions (tests, demos)

def _sample_surface(shape: str, dims: np.ndarray, rng, n: int, region: str, center_dir=None):
    """Points + outward normals on a primitive (object frame), from a plausible touch region."""
    pts, nrm = [], []
    while len(pts) < n:
        if shape == "sphere":
            r = dims[0]
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            if center_dir is not None and d @ center_dir < np.cos(np.radians(70)):
                continue
            pts.append(r * d)
            nrm.append(d)
        elif shape == "cylinder":
            r, h = dims
            if region == "cap":
                ang, rad = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0, 0.8))
                s = 1 if rng.random() < 0.5 or center_dir is None else np.sign(center_dir[2])
                pts.append([rad * np.cos(ang), rad * np.sin(ang), s * h / 2])
                nrm.append([0, 0, s])
            else:
                base = 0.0 if center_dir is None else np.arctan2(center_dir[1], center_dir[0])
                ang = base + rng.uniform(-2.1, 2.1)
                z = rng.uniform(-h / 2 * 0.8, h / 2 * 0.8)
                pts.append([r * np.cos(ang), r * np.sin(ang), z])
                nrm.append([np.cos(ang), np.sin(ang), 0])
        else:
            half = dims / 2
            long_ax = int(np.argmax(dims))
            faces = [i for i in range(3) if i != long_ax] if region == "side" else [long_ax]
            ax = faces[rng.integers(len(faces))] if region != "any" else rng.integers(3)
            s = rng.choice([-1, 1])
            p = rng.uniform(-half * 0.85, half * 0.85)
            p[ax] = s * half[ax]
            nv = np.zeros(3)
            nv[ax] = s
            pts.append(p)
            nrm.append(nv)
    return np.array(pts), np.array(nrm)


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _to_hand(points, normals, rng, noise_p, noise_n_deg, bias):
    R = _random_rotation(rng)
    t = rng.uniform(-0.1, 0.1, 3)
    p = points @ R.T + t + rng.normal(0, noise_p, points.shape) + bias
    n = normals @ R.T + rng.normal(0, np.sin(np.radians(noise_n_deg)), normals.shape)
    return p, n / np.linalg.norm(n, axis=1, keepdims=True)


def synthetic_session(shape: str, dims: np.ndarray, rng, *, noise_p=0.0016, noise_n_deg=20.0,
                      bias_m=0.001, n_encl=8, end_caps=True, calipers=True) -> list[Grasp]:
    """Mimics the exploration routine: 3 enclosure grasps (+ end-cap touches) + 3 caliper pinches.

    Box dims are object-frame extents [x, y, z] with z the long axis.
    """
    dims = np.asarray(dims, float)
    bias = rng.normal(0, bias_m, 3)
    grasps = []
    for k in range(3):
        if shape == "sphere":
            c = rng.normal(size=3)
            c /= np.linalg.norm(c)
            p, n = _sample_surface(shape, dims, rng, n_encl, "any", c)
        elif shape == "cylinder":
            c = np.array([np.cos(k * 2.1), np.sin(k * 2.1), 0.0])
            p, n = _sample_surface(shape, dims, rng, n_encl, "side", c)
            if end_caps and k == 2:
                pc, nc = _sample_surface(shape, dims, rng, 3, "cap", np.array([0, 0, 1.0]))
                p, n = np.vstack([p, pc]), np.vstack([n, nc])
        else:
            p, n = _sample_surface(shape, dims, rng, n_encl, "side")
            if end_caps and k == 2:
                pc, nc = _sample_surface(shape, dims, rng, 3, "top")
                p, n = np.vstack([p, pc]), np.vstack([n, nc])
        hp, hn = _to_hand(p, n, rng, noise_p, noise_n_deg, bias)
        grasps.append(Grasp(hp, hn, "top" if (end_caps and k == 2) else "enclosure"))
    if calipers:
        if shape == "sphere":
            axes = [rng.normal(size=3) for _ in range(3)]
            axes = [a / np.linalg.norm(a) for a in axes]
            pairs = [(dims[0] * a, -dims[0] * a, a) for a in axes]
        elif shape == "cylinder":
            r, h = dims
            pairs = [(np.array([r, 0, 0]), np.array([-r, 0, 0]), np.array([1.0, 0, 0])),
                     (np.array([0, r, 0]), np.array([0, -r, 0]), np.array([0, 1.0, 0])),
                     (np.array([0, 0, h / 2]), np.array([0, 0, -h / 2]), np.array([0, 0, 1.0]))]
            if not end_caps:
                pairs = pairs[:2]
        else:
            pairs = []
            for ax in range(3):
                e = np.zeros(3)
                e[ax] = 1.0
                off = rng.uniform(-dims * 0.3, dims * 0.3)
                off[ax] = 0
                pairs.append((off + e * dims[ax] / 2, off - e * dims[ax] / 2, e))
        for a, b, e in pairs:
            hp, hn = _to_hand(np.array([a, b]), np.array([e, -e]), rng, noise_p, noise_n_deg, bias)
            grasps.append(Grasp(hp, hn, "caliper"))
    return grasps
