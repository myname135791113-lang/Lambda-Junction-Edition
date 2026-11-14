"""Live 3D point cloud accumulator, table-aware Convex Hull, 2D OBB, and Cauchy primitive fitter.

Designed to run at >= 30 Hz in pure NumPy (automatically uses `scipy.spatial.ConvexHull` when
installed, with a built-in vectorized 3D incremental hull fallback so zero extra C-packages are
needed on a fresh hackathon machine).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

SIGMA_FIT_M = 0.0035  # 3.5 mm Cauchy scale (Rokoko ~1.6 mm + skin compliance)


# --------------------------------------------------------------------------- 2D Hull & Rotating Calipers OBB

def convex_hull_2d(xy: np.ndarray) -> np.ndarray:
    """Monotone-chain 2D convex hull returning counter-clockwise vertices `(H, 2)`."""
    pts = np.unique(np.round(xy, 6), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return np.vstack(lower[:-1] + upper[:-1])


def minimum_area_obb_2d(xy: np.ndarray) -> dict[str, Any]:
    """Minimum-area 2D oriented bounding rectangle in XY (`length >= width`)."""
    if len(xy) == 0:
        return {"center_xy": np.zeros(2), "length_m": 0.0, "width_m": 0.0, "yaw_rad": 0.0}
    if len(xy) == 1:
        return {"center_xy": xy[0].copy(), "length_m": 0.0, "width_m": 0.0, "yaw_rad": 0.0}

    hull = convex_hull_2d(xy)
    if len(hull) == 2:
        d = hull[1] - hull[0]
        L = float(np.linalg.norm(d))
        yaw = float(math.atan2(d[1], d[0]))
        return {"center_xy": 0.5 * (hull[0] + hull[1]), "length_m": L, "width_m": 0.0, "yaw_rad": yaw}

    best_area = float("inf")
    best = None
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    for ang in angles:
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, s], [-s, c]])  # rotate world -> edge-aligned frame
        proj = hull @ R.T
        mn, mx = proj.min(axis=0), proj.max(axis=0)
        ext = mx - mn
        area = float(ext[0] * ext[1])
        if area < best_area:
            best_area = area
            mid_local = 0.5 * (mn + mx)
            center_xy = R.T @ mid_local
            if ext[0] >= ext[1]:
                length_m, width_m, yaw = float(ext[0]), float(ext[1]), float(ang)
            else:
                length_m, width_m, yaw = float(ext[1]), float(ext[0]), float(ang + 0.5 * math.pi)
            # Wrap yaw to [-pi/2, pi/2] for canonical rectangle orientation
            yaw = (yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi
            best = {"center_xy": center_xy, "length_m": length_m, "width_m": width_m, "yaw_rad": yaw}

    assert best is not None
    return best


# --------------------------------------------------------------------------- 3D Convex Hull (SciPy + Pure-NumPy)

@dataclass
class Hull3D:
    vertices: np.ndarray     # (V, 3) m
    faces: np.ndarray        # (F, 3) int indices, outward-oriented
    volume_m3: float
    area_m2: float


def compute_convex_hull_3d(points_m: np.ndarray) -> Hull3D | None:
    """Compute outward-oriented 3D Convex Hull mesh, volume, and surface area."""
    P = np.asarray(points_m, dtype=float)
    if len(P) < 4:
        return None

    # Check 3D thickness (avoid calling hull on strictly coplanar/collinear point sets)
    centered = P - P.mean(axis=0)
    svals = np.linalg.svd(centered, compute_uv=False)
    if len(svals) < 3 or svals[2] < 1.5e-3:
        return None

    try:
        from scipy.spatial import ConvexHull  # type: ignore
        hull = ConvexHull(P)
        faces = np.asarray(hull.simplices, dtype=int)
        faces = _orient_faces_outward(P, faces)
        return Hull3D(vertices=P, faces=faces, volume_m3=float(hull.volume), area_m2=float(hull.area))
    except ImportError:
        return _incremental_hull_3d_numpy(P)
    except Exception:
        return _incremental_hull_3d_numpy(P)


def _orient_faces_outward(V: np.ndarray, faces: np.ndarray, interior_pt: np.ndarray | None = None) -> np.ndarray:
    c = V[np.unique(faces)].mean(axis=0) if interior_pt is None else interior_pt
    a, b, d = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
    normals = np.cross(b - a, d - a)
    flip = np.einsum("ij,ij->i", normals, a - c) < 0.0
    out = faces.copy()
    out[flip, 1], out[flip, 2] = out[flip, 2], out[flip, 1]
    return out


def _incremental_hull_3d_numpy(points_m: np.ndarray) -> Hull3D | None:
    """Self-contained pure-NumPy incremental 3D Convex Hull for <= ~500 voxelized points."""
    rng = np.random.default_rng(42)
    V = points_m + rng.normal(0.0, 1e-6, points_m.shape)  # general-position tie-breaker above float rounding
    n = len(V)

    i0 = int(np.argmin(V[:, 0]))
    i1 = int(np.argmax(np.linalg.norm(V - V[i0], axis=1)))
    line_dir = V[i1] - V[i0]
    line_dir /= max(float(np.linalg.norm(line_dir)), 1e-12)
    line_dist = np.linalg.norm(np.cross(V - V[i0], line_dir), axis=1)
    i2 = int(np.argmax(line_dist))
    if line_dist[i2] < 1e-4:
        return None

    plane_n = np.cross(V[i1] - V[i0], V[i2] - V[i0])
    plane_n /= max(float(np.linalg.norm(plane_n)), 1e-12)
    plane_dist = np.abs((V - V[i0]) @ plane_n)
    i3 = int(np.argmax(plane_dist))
    if plane_dist[i3] < 1e-4:
        return None

    init_ids = [i0, i1, i2, i3]
    interior = V[init_ids].mean(axis=0)
    faces = _orient_faces_outward(
        V,
        np.array([[i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3]], dtype=int),
        interior,
    )

    used = set(init_ids)
    for idx in range(n):
        if idx in used:
            continue
        p = V[idx]
        a, b, c = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
        normals = np.cross(b - a, c - a)
        norms = np.maximum(np.linalg.norm(normals, axis=1), 1e-15)
        signed_d = np.einsum("ij,ij->i", normals, p - a) / norms
        f_start = int(np.argmax(signed_d))
        if signed_d[f_start] <= 1e-6:
            continue

        # Flood-fill connected visible patch from furthest-visible face to preserve 2-manifold convexity
        edge_to_face: dict[tuple[int, int], int] = {}
        for fi, (f0, f1, f2) in enumerate(faces):
            edge_to_face[(int(f0), int(f1))] = fi
            edge_to_face[(int(f1), int(f2))] = fi
            edge_to_face[(int(f2), int(f0))] = fi

        vis_set = {f_start}
        stack = [f_start]
        while stack:
            fi = stack.pop()
            f0, f1, f2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            for u, v in ((f0, f1), (f1, f2), (f2, f0)):
                nb = edge_to_face.get((v, u))
                if nb is not None and nb not in vis_set and signed_d[nb] > 1e-12:
                    vis_set.add(nb)
                    stack.append(nb)

        # Horizon edges: directed edges of visible faces whose reverse edge is NOT in visible faces
        vis_edges = set()
        for fi in vis_set:
            f0, f1, f2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            vis_edges.add((f0, f1))
            vis_edges.add((f1, f2))
            vis_edges.add((f2, f0))

        horizon = [(u, v) for (u, v) in vis_edges if (v, u) not in vis_edges]
        if len(horizon) < 3:
            continue

        # Ensure the horizon forms a simple closed loop (every boundary vertex has out-degree 1, in-degree 1)
        out_deg: dict[int, int] = {}
        in_deg: dict[int, int] = {}
        for u, v in horizon:
            out_deg[u] = out_deg.get(u, 0) + 1
            in_deg[v] = in_deg.get(v, 0) + 1
        if any(d != 1 for d in out_deg.values()) or set(out_deg.keys()) != set(in_deg.keys()):
            continue

        keep_mask = np.ones(len(faces), dtype=bool)
        for fi in vis_set:
            keep_mask[fi] = False
        keep_faces = faces[keep_mask]
        new_arr = _orient_faces_outward(V, np.array([[u, v, idx] for (u, v) in horizon], dtype=int), interior)
        faces = np.vstack([keep_faces, new_arr]) if len(keep_faces) else new_arr

    # Volume & surface area from outward triangles on original unjittered coordinates
    P = points_m
    faces = _orient_faces_outward(P, faces, interior)
    a, b, c = P[faces[:, 0]] - interior, P[faces[:, 1]] - interior, P[faces[:, 2]] - interior
    cross_bc = np.cross(b - a, c - a)
    area = 0.5 * float(np.sum(np.linalg.norm(cross_bc, axis=1)))
    vol = abs(float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c)))) / 6.0)
    return Hull3D(vertices=P, faces=faces, volume_m3=vol, area_m2=area)


# --------------------------------------------------------------------------- Table-Anchored Cauchy Primitive Fit

def _sdf_table_primitive(
    shape: str, params: np.ndarray, P: np.ndarray, z_table: float
) -> np.ndarray:
    """Signed surface distance for an object resting on `z = z_table`."""
    if shape == "sphere":
        # params = [cx, cy, R], sphere rests on z_table -> cz = z_table + R
        cx, cy, R = params
        c = np.array([cx, cy, z_table + R])
        return np.linalg.norm(P - c, axis=1) - R

    if shape == "cylinder":
        # params = [cx, cy, R, H], upright cylinder on z_table -> cz = z_table + H/2
        cx, cy, R, H = params
        dr = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - R
        dz = np.abs(P[:, 2] - (z_table + 0.5 * H)) - 0.5 * H
        return np.minimum(np.maximum(dr, dz), 0.0) + np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))

    if shape == "cylinder_side":
        # params = [cx, cy, yaw, R, L], horizontal cylinder resting on z_table -> cz = z_table + R
        cx, cy, yaw, R, L = params
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        dx, dy = P[:, 0] - cx, P[:, 1] - cy
        u = np.abs(cyaw * dx + syaw * dy) - 0.5 * L
        v = -syaw * dx + cyaw * dy
        dr = np.hypot(v, P[:, 2] - (z_table + R)) - R
        return np.minimum(np.maximum(dr, u), 0.0) + np.hypot(np.maximum(dr, 0.0), np.maximum(u, 0.0))

    # box: params = [cx, cy, yaw, L, W, H], resting flat on z_table -> cz = z_table + H/2
    cx, cy, yaw, L, W, H = params
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy
    u = cyaw * dx + syaw * dy
    v = -syaw * dx + cyaw * dy
    w = P[:, 2] - (z_table + 0.5 * H)
    q = np.column_stack([np.abs(u) - 0.5 * L, np.abs(v) - 0.5 * W, np.abs(w) - 0.5 * H])
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def _fit_single_primitive_irls(
    shape: str,
    x0_list: list[np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    P: np.ndarray,
    z_table: float,
    max_iter: int = 18,
) -> dict[str, Any]:
    """Damped Gauss-Newton with Cauchy IRLS weights (pure NumPy, < 2 ms per template)."""
    best_dev = float("inf")
    best_x = x0_list[0].copy()
    eps = 1e-5
    k = len(lo)

    for x_init in x0_list:
        x = np.clip(x_init.astype(float), lo + 1e-5, hi - 1e-5)
        lam = 1e-3
        for _ in range(max_iter):
            r = _sdf_table_primitive(shape, x, P, z_table)
            z = r / SIGMA_FIT_M
            dev = float(np.sum(np.log1p(z * z)))
            weights = 1.0 / (1.0 + z * z)  # Cauchy IRLS weights

            J = np.empty((len(P), k), dtype=float)
            for j in range(k):
                step = 1e-4 if (j == 2 and shape in ("box", "cylinder_side")) else eps
                xp = x.copy()
                xp[j] += step
                J[:, j] = (_sdf_table_primitive(shape, xp, P, z_table) - r) / step

            JW = J * weights[:, None]
            H_mat = JW.T @ J + lam * np.eye(k)
            g_vec = JW.T @ r
            try:
                dx = -np.linalg.solve(H_mat, g_vec)
            except np.linalg.LinAlgError:
                break

            x_new = np.clip(x + dx, lo, hi)
            r_new = _sdf_table_primitive(shape, x_new, P, z_table)
            z_new = r_new / SIGMA_FIT_M
            dev_new = float(np.sum(np.log1p(z_new * z_new)))
            if dev_new < dev:
                x = x_new
                lam = max(lam * 0.5, 1e-6)
                if abs(dev - dev_new) < 1e-4:
                    break
            else:
                lam *= 4.0
                if lam > 1e3:
                    break

        r_fin = _sdf_table_primitive(shape, x, P, z_table)
        z_fin = r_fin / SIGMA_FIT_M
        dev_fin = float(np.sum(np.log1p(z_fin * z_fin)))
        if dev_fin < best_dev:
            best_dev = dev_fin
            best_x = x

    r_best = _sdf_table_primitive(shape, best_x, P, z_table)
    rms_m = float(np.sqrt(np.mean(r_best ** 2)))
    med_abs_m = float(np.median(np.abs(r_best)))
    bic = best_dev + k * math.log(max(len(P), k + 1))
    return {
        "shape": shape, "params": best_x, "residuals": r_best, "k": k,
        "rms_m": rms_m, "median_abs_m": med_abs_m, "deviance": best_dev, "bic": bic,
    }


def fit_table_primitives(P: np.ndarray, z_table: float, obb: dict[str, Any]) -> dict[str, Any]:
    """Fit table-resting `sphere`, `cylinder`, and `box` models and score via BIC."""
    if len(P) < 7:
        return {"best": None, "weights": {"box": 0.0, "cylinder": 0.0, "sphere": 0.0}, "fits": {}}

    cx, cy = float(obb["center_xy"][0]), float(obb["center_xy"][1])
    H_obs = max(float(P[:, 2].max() - z_table), 0.02)
    L_obs = max(float(obb["length_m"]), 0.02)
    W_obs = max(float(obb["width_m"]), 0.02)
    yaw0 = float(obb["yaw_rad"])
    R_est = max(0.25 * (L_obs + W_obs), 0.015)

    fits: dict[str, dict[str, Any]] = {}

    # 1. Sphere: [cx, cy, R]
    fits["sphere"] = _fit_single_primitive_irls(
        "sphere",
        [np.array([cx, cy, 0.5 * H_obs]), np.array([cx, cy, R_est])],
        np.array([-2.0, -2.0, 0.012]),
        np.array([2.0, 2.0, 0.25]),
        P,
        z_table,
    )
    sp = fits["sphere"]["params"]
    fits["sphere"]["dims_m"] = {"radius": float(sp[2])}
    fits["sphere"]["center_xy"] = [float(sp[0]), float(sp[1])]
    fits["sphere"]["yaw_rad"] = 0.0

    # 2. Cylinder: Upright [cx, cy, R, H] OR Sideways [cx, cy, yaw, R, L] resting on z_table
    cyl_up = _fit_single_primitive_irls(
        "cylinder",
        [np.array([cx, cy, R_est, H_obs]), np.array([cx, cy, 0.5 * L_obs, H_obs])],
        np.array([-2.0, -2.0, 0.010, 0.015]),
        np.array([2.0, 2.0, 0.25, 0.45]),
        P,
        z_table,
    )
    R_side_est = max(0.25 * (W_obs + H_obs), 0.012)
    cyl_side = _fit_single_primitive_irls(
        "cylinder_side",
        [
            np.array([cx, cy, yaw0, R_side_est, L_obs]),
            np.array([cx, cy, yaw0 + 0.5 * math.pi, 0.5 * H_obs, max(L_obs, W_obs)]),
        ],
        np.array([-2.0, -2.0, -math.pi, 0.010, 0.015]),
        np.array([2.0, 2.0, math.pi, 0.25, 0.45]),
        P,
        z_table,
    )

    # 3. Upright Box: [cx, cy, yaw, L, W, H]
    fits["box"] = _fit_single_primitive_irls(
        "box",
        [
            np.array([cx, cy, yaw0, L_obs, W_obs, H_obs]),
            np.array([cx, cy, yaw0 + math.pi / 4.0, R_est * 1.6, R_est * 1.6, H_obs]),
        ],
        np.array([-2.0, -2.0, -math.pi, 0.015, 0.015, 0.015]),
        np.array([2.0, 2.0, math.pi, 0.45, 0.45, 0.45]),
        P,
        z_table,
    )
    bp = fits["box"]["params"]
    bx_l, bx_w = sorted([float(bp[3]), float(bp[4])], reverse=True)
    fits["box"]["dims_m"] = {"width": bx_w, "depth": bx_l, "height": float(bp[5])}
    fits["box"]["center_xy"] = [float(bp[0]), float(bp[1])]
    fits["box"]["yaw_rad"] = float(bp[2])

    # Adaptive MAD noise floor prevents overconfident wrong labels when real-sensor jitter > 3.5 mm
    med_candidates = [fits["box"]["median_abs_m"], fits["sphere"]["median_abs_m"], cyl_up["median_abs_m"], cyl_side["median_abs_m"]]
    sigma_eff = max(SIGMA_FIT_M, 1.4826 * min(med_candidates))
    n_pts = max(len(P), 7)

    for c_fit in (cyl_up, cyl_side):
        z_sc = c_fit["residuals"] / sigma_eff
        c_fit["bic"] = float(np.sum(np.log1p(z_sc * z_sc))) + c_fit["k"] * math.log(n_pts)

    cp_s = cyl_side["params"]
    use_sideways_cyl = bool(
        cyl_side["bic"] < cyl_up["bic"] and float(cp_s[4]) > 1.25 * (2.0 * float(cp_s[3]))
    )
    if use_sideways_cyl:
        fits["cylinder"] = cyl_side
        fits["cylinder"]["shape"] = "cylinder"
        fits["cylinder"]["dims_m"] = {"radius": float(cp_s[3]), "height": float(cp_s[4])}
        fits["cylinder"]["center_xy"] = [float(cp_s[0]), float(cp_s[1])]
        fits["cylinder"]["yaw_rad"] = float(cp_s[2])
        fits["cylinder"]["sideways"] = True
    else:
        fits["cylinder"] = cyl_up
        cp = cyl_up["params"]
        fits["cylinder"]["dims_m"] = {"radius": float(cp[2]), "height": float(cp[3])}
        fits["cylinder"]["center_xy"] = [float(cp[0]), float(cp[1])]
        fits["cylinder"]["yaw_rad"] = 0.0
        fits["cylinder"]["sideways"] = False

    for m in ("box", "cylinder", "sphere"):
        if "residuals" in fits[m]:
            z_scaled = fits[m]["residuals"] / sigma_eff
            fits[m]["bic"] = float(np.sum(np.log1p(z_scaled * z_scaled))) + fits[m]["k"] * math.log(n_pts)
            del fits[m]["residuals"]

    bics = np.array([fits[m]["bic"] for m in ("box", "cylinder", "sphere")])
    w = np.exp(-0.5 * (bics - bics.min()))
    w /= w.sum()
    weights = {m: round(float(val), 4) for m, val in zip(("box", "cylinder", "sphere"), w)}
    best = max(weights, key=weights.get)
    return {"best": best, "weights": weights, "sigma_eff_m": round(sigma_eff, 5), "fits": fits}


def fit_reference_pose_xy_yaw(
    points_m: np.ndarray,
    shape: str,
    dims: dict[str, float],
    z_table: float,
    init_xy: np.ndarray,
    init_yaw: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Robustly align a known catalog template's `(cx, cy, yaw)` to `points_m` on `z_table`.

    Uses Cauchy IRLS on the exact surface SDF so stray non-touching fingers in a 5-finger
    snapshot cannot pull the rendered 3D structure off the true touched object surface.
    """
    P = np.asarray(points_m, dtype=float)
    base_xy = np.asarray(init_xy, dtype=float)[:2].copy()
    base_yaw = float(init_yaw)
    if len(P) < 6:
        return base_xy, base_yaw

    med_xy = np.median(P[:, :2], axis=0)
    sigma = 0.0045  # 4.5 mm Cauchy scale for skin + sensor tolerance

    def eval_sdf(st: np.ndarray) -> np.ndarray:
        if shape == "sphere":
            return _sdf_table_primitive("sphere", np.array([st[0], st[1], float(dims["radius"])]), P, z_table)
        if shape == "cylinder":
            return _sdf_table_primitive(
                "cylinder",
                np.array([st[0], st[1], float(dims["radius"]), float(dims["height"])]),
                P,
                z_table,
            )
        if shape == "cylinder_side":
            return _sdf_table_primitive(
                "cylinder_side",
                np.array([st[0], st[1], st[2], float(dims["radius"]), float(dims["height"])]),
                P,
                z_table,
            )
        return _sdf_table_primitive(
            "box",
            np.array([st[0], st[1], st[2], float(dims["x"]), float(dims["y"]), float(dims["z"])]),
            P,
            z_table,
        )

    has_yaw = shape in ("box", "cylinder_side")
    k = 3 if has_yaw else 2
    lo = np.array([med_xy[0] - 0.12, med_xy[1] - 0.12, -math.pi])[:k]
    hi = np.array([med_xy[0] + 0.12, med_xy[1] + 0.12, math.pi])[:k]

    seeds: list[np.ndarray] = []
    if has_yaw:
        seeds.append(np.array([med_xy[0], med_xy[1], base_yaw]))
        seeds.append(np.array([base_xy[0], base_xy[1], base_yaw]))
        seeds.append(np.array([med_xy[0], med_xy[1], base_yaw + 0.25 * math.pi]))
    else:
        seeds.append(med_xy.copy())
        seeds.append(base_xy.copy())

    init_state = np.array([base_xy[0], base_xy[1], base_yaw])[:k]
    r_init = eval_sdf(init_state)
    best_dev = float(np.sum(np.log1p((r_init / sigma) ** 2)))
    best_st = init_state.copy()

    eps = 1e-5
    for s0 in seeds:
        st = np.clip(s0.astype(float), lo + 1e-5, hi - 1e-5)
        lam = 1e-2
        for _ in range(14):
            r = eval_sdf(st)
            z = r / sigma
            dev = float(np.sum(np.log1p(z * z)))
            w_irls = 1.0 / (1.0 + z * z)

            J = np.empty((len(P), k), dtype=float)
            for j in range(k):
                step = 1e-4 if (has_yaw and j == 2) else eps
                stp = st.copy()
                stp[j] += step
                J[:, j] = (eval_sdf(stp) - r) / step

            JW = J * w_irls[:, None]
            H_mat = JW.T @ J + lam * np.eye(k)
            g_vec = JW.T @ r
            try:
                dx = -np.linalg.solve(H_mat, g_vec)
            except np.linalg.LinAlgError:
                break

            st_new = np.clip(st + dx, lo, hi)
            if has_yaw:
                st_new[2] = math.atan2(math.sin(st_new[2]), math.cos(st_new[2]))
            r_new = eval_sdf(st_new)
            dev_new = float(np.sum(np.log1p((r_new / sigma) ** 2)))
            if dev_new < dev:
                st = st_new
                lam = max(lam * 0.5, 1e-4)
                if abs(dev - dev_new) < 1e-4:
                    break
            else:
                lam *= 4.0
                if lam > 1e2:
                    break

        r_fin = eval_sdf(st)
        dev_fin = float(np.sum(np.log1p((r_fin / sigma) ** 2)))
        if dev_fin < best_dev:
            best_dev = dev_fin
            best_st = st.copy()

    yaw_out = math.atan2(math.sin(best_st[2]), math.cos(best_st[2])) if has_yaw else base_yaw
    return best_st[:2].copy(), float(yaw_out)


# --------------------------------------------------------------------------- Cloud Accumulator

@dataclass
class ScanSummary:
    n_raw: int
    n_voxels: int
    height_m: float
    length_m: float
    width_m: float
    yaw_rad: float
    center_xy: np.ndarray
    hull_volume_m3: float
    hull_area_m2: float
    calipers_m: list[float]
    primitive_fit: dict[str, Any]
    hull: Hull3D | None = None


class TactileCloudBuilder:
    """Voxel-downsampled 3D touch cloud + live table-anchored Convex Hull & primitive fit."""

    def __init__(self, *, voxel_m: float = 0.004, z_table: float = 0.0):
        self.voxel_m = voxel_m
        self.z_table = z_table
        self.locked: bool = False
        self._voxels: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
        self._n_raw: int = 0
        self.calipers_m: list[float] = []
        self._dirty: bool = True
        self._cached_summary: ScanSummary | None = None

    def reset(self, z_table: float | None = None) -> None:
        if z_table is not None:
            self.z_table = z_table
        self.locked = False
        self._voxels.clear()
        self._n_raw = 0
        self.calipers_m.clear()
        self._dirty = True
        self._cached_summary = None

    def add_points(self, points_m: np.ndarray) -> int:
        """Filter table penetration, snap near-table seam contacts, and update voxel centroids."""
        if self.locked:
            return 0
        pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
        added = 0
        for p in pts:
            if not np.all(np.isfinite(p)):
                continue
            # Reject deep below-table sensor glitches (> 6 mm below table)
            if p[2] < self.z_table - 0.006:
                continue
            p_clean = p.copy()
            if p_clean[2] < self.z_table + 0.003:
                p_clean[2] = self.z_table

            # Outlier guard once cloud has >= 15 voxels: ignore wild stray points > 30 cm from centroid
            if len(self._voxels) >= 15:
                cur_mean = np.mean([v[0] for v in self._voxels.values()], axis=0)
                if float(np.linalg.norm(p_clean - cur_mean)) > 0.30:
                    continue

            key = tuple(int(math.floor(c / self.voxel_m)) for c in p_clean)
            if key in self._voxels:
                prev_mean, cnt = self._voxels[key]
                self._voxels[key] = ((prev_mean * cnt + p_clean) / (cnt + 1), cnt + 1)
            else:
                self._voxels[key] = (p_clean, 1)
                added += 1
            self._n_raw += 1

        if added > 0 or self._cached_summary is None:
            self._dirty = True
        return added

    def add_caliper_width(self, width_m: float) -> None:
        if self.locked:
            return
        if 0.010 <= width_m <= 0.250:
            self.calipers_m.append(round(float(width_m), 4))
            self._dirty = True

    def transform_planar(
        self,
        src_xy: np.ndarray,
        src_yaw: float,
        dst_xy: np.ndarray,
        dst_yaw: float,
    ) -> None:
        """Rigidly relocate accumulated cloud on the table from (src_xy, src_yaw) to (dst_xy, dst_yaw)."""
        if not self._voxels:
            return
        d_yaw = float(dst_yaw - src_yaw)
        c, s = math.cos(d_yaw), math.sin(d_yaw)
        R2 = np.array([[c, -s], [s, c]], dtype=float)
        src = np.asarray(src_xy, dtype=float)[:2]
        dst = np.asarray(dst_xy, dtype=float)[:2]

        new_voxels: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
        for pt, cnt in self._voxels.values():
            p_new = pt.copy()
            p_new[:2] = dst + R2 @ (pt[:2] - src)
            key = tuple(int(math.floor(coord / self.voxel_m)) for coord in p_new)
            if key in new_voxels:
                prev_pt, prev_cnt = new_voxels[key]
                tot = prev_cnt + cnt
                new_voxels[key] = ((prev_pt * prev_cnt + p_new * cnt) / tot, tot)
            else:
                new_voxels[key] = (p_new, cnt)

        self._voxels = new_voxels
        self._dirty = True
        self._cached_summary = None

    def transform_se3(
        self,
        T_src: np.ndarray,
        T_dst: np.ndarray,
        *,
        snap_min_z_to_table: bool = False,
    ) -> None:
        """Rigidly relocate accumulated 3D cloud in full SE(3) from `T_src` to `T_dst`."""
        if not self._voxels:
            return
        R_src, t_src = T_src[:3, :3], T_src[:3, 3]
        R_dst, t_dst = T_dst[:3, :3], T_dst[:3, 3]
        R_rel = R_dst @ R_src.T

        transformed: list[tuple[np.ndarray, int]] = []
        for pt, cnt in self._voxels.values():
            p_new = R_rel @ (pt - t_src) + t_dst
            transformed.append((p_new, cnt))

        if snap_min_z_to_table and transformed:
            min_z = min(float(p[2]) for p, _ in transformed)
            dz = self.z_table - min_z
            for p, _ in transformed:
                p[2] += dz

        new_voxels: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
        for p_new, cnt in transformed:
            if p_new[2] < self.z_table:
                p_new = p_new.copy()
                p_new[2] = self.z_table
            key = tuple(int(math.floor(coord / self.voxel_m)) for coord in p_new)
            if key in new_voxels:
                prev_pt, prev_cnt = new_voxels[key]
                tot = prev_cnt + cnt
                new_voxels[key] = ((prev_pt * prev_cnt + p_new * cnt) / tot, tot)
            else:
                new_voxels[key] = (p_new, cnt)

        self._voxels = new_voxels
        self._dirty = True
        self._cached_summary = None

    def points(self) -> np.ndarray:
        if not self._voxels:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([v[0] for v in self._voxels.values()])

    def summarize(self) -> ScanSummary:
        if not self._dirty and self._cached_summary is not None:
            return self._cached_summary

        P = self.points()
        if len(P) == 0:
            summary = ScanSummary(
                n_raw=0,
                n_voxels=0,
                height_m=0.0,
                length_m=0.0,
                width_m=0.0,
                yaw_rad=0.0,
                center_xy=np.zeros(2),
                hull_volume_m3=0.0,
                hull_area_m2=0.0,
                calipers_m=list(self.calipers_m),
                primitive_fit={"best": None, "weights": {"box": 0.0, "cylinder": 0.0, "sphere": 0.0}, "fits": {}},
                hull=None,
            )
            self._cached_summary = summary
            self._dirty = False
            return summary

        obb = minimum_area_obb_2d(P[:, :2])
        height_m = max(float(P[:, 2].max() - self.z_table), 0.0)

        # Inject 1 universal table-contact anchor directly under the horizontal centroid:
        #   - Guarantees exact vertical reach to `z_table` even if bottom wasn't scraped
        #   - Never bloats spheres into cylinders (unlike projecting equatorial points straight down)
        ground_anchor = np.array([[float(P[:, 0].mean()), float(P[:, 1].mean()), self.z_table]])
        P_hull = np.vstack([P, ground_anchor])
        hull = compute_convex_hull_3d(P_hull)
        prim = fit_table_primitives(P, self.z_table, obb)

        summary = ScanSummary(
            n_raw=self._n_raw,
            n_voxels=len(P),
            height_m=height_m,
            length_m=float(obb["length_m"]),
            width_m=float(obb["width_m"]),
            yaw_rad=float(obb["yaw_rad"]),
            center_xy=obb["center_xy"],
            hull_volume_m3=0.0 if hull is None else hull.volume_m3,
            hull_area_m2=0.0 if hull is None else hull.area_m2,
            calipers_m=list(self.calipers_m),
            primitive_fit=prim,
            hull=hull,
        )
        self._cached_summary = summary
        self._dirty = False
        return summary

    def export_ply(self, path: str | Path) -> None:
        p_out = Path(path)
        P = self.points() * 1000.0  # export in mm for MeshLab / CloudCompare compatibility
        P = P[np.all(np.isfinite(P), axis=1)] if len(P) else P
        lines = "\n".join(f"{r[0]:.2f} {r[1]:.2f} {r[2]:.2f}" for r in P)
        text = (
            f"ply\nformat ascii 1.0\nelement vertex {len(P)}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
            + (lines + "\n" if lines else "")
        )
        tmp = p_out.with_name(f"{p_out.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p_out)
