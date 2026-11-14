"""Object shapes: three parametric primitives (box, cylinder, sphere) and import of glove-made shapes.

Convention for every mesh returned here: metres, resting on the table, i.e. min z = 0, and the
horizontal bounding box centred on x = y = 0. The card's `pose` then places it in the robot frame.

Accepted JSON inputs (`load_shape_json`):
  1. touchmap_grasp `result.json` (+ `cloud.ply` next to it). Uses the fingertip cloud (convex hull) if
     the .ply is there, otherwise rebuilds the best-fit primitive from the fit parameters. Units: mm.
  2. {"vertices": [[x,y,z],...], "faces": [[i,j,k],...]}   a mesh
  3. {"points": [[x,y,z],...]}  or  {"points_mm": ...}      a point cloud -> convex hull
  4. {"shape": "box|cylinder|sphere", "params": {...}}       parametric description
  Optional key "units": "mm" | "m" (default: mm for touchmap files, m otherwise).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

SHAPES = ("box", "cylinder", "sphere")

# default parameters in metres
DEFAULTS = {
    "box": {"x": 0.060, "y": 0.060, "z": 0.100},
    "cylinder": {"radius": 0.033, "height": 0.115},
    "sphere": {"radius": 0.020},
}


@dataclass
class Shape:
    mesh: trimesh.Trimesh          # metres, min z = 0, xy-centred
    kind: str                      # box | cylinder | sphere | mesh
    params: dict
    source: str = "parametric"
    note: str = ""

    @property
    def extents(self) -> np.ndarray:
        return self.mesh.bounds[1] - self.mesh.bounds[0]

    @property
    def height(self) -> float:
        return float(self.extents[2])

    def volume(self) -> float:
        return float(self.mesh.convex_hull.volume)


def ground(mesh: trimesh.Trimesh, keep_xy: bool = False) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Return the mesh with min z = 0 and (unless keep_xy) the xy bounding-box centre at the origin,
    plus the translation that was applied."""
    lo, hi = mesh.bounds
    t = np.array([0.0 if keep_xy else -(lo[0] + hi[0]) / 2, 0.0 if keep_xy else -(lo[1] + hi[1]) / 2, -lo[2]])
    m = mesh.copy()
    m.apply_translation(t)
    return m, t


def make_shape(kind: str, **overrides) -> Shape:
    """Build a primitive. Pass dimensions in metres, e.g. make_shape("box", x=.05, y=.05, z=.14)."""
    if kind not in SHAPES:
        raise ValueError(f"unknown shape {kind!r}; choose from {SHAPES}")
    p = {**DEFAULTS[kind], **{k: float(v) for k, v in overrides.items() if v is not None}}
    unknown = set(p) - set(DEFAULTS[kind])
    if unknown:
        raise ValueError(f"{kind}: unknown parameter(s) {sorted(unknown)}; valid: {sorted(DEFAULTS[kind])}")
    if min(p.values()) <= 0:
        raise ValueError("dimensions must be positive")
    if kind == "box":
        m = trimesh.creation.box(extents=[p["x"], p["y"], p["z"]])
    elif kind == "cylinder":
        m = trimesh.creation.cylinder(radius=p["radius"], height=p["height"], sections=48)
    else:
        m = trimesh.creation.icosphere(subdivisions=3, radius=p["radius"])
    m, _ = ground(m)
    return Shape(m, kind, p)


# ---------------------------------------------------------------------------- JSON import
def _scale(doc: dict, default_units: str) -> float:
    return 1e-3 if str(doc.get("units", default_units)).lower() == "mm" else 1.0


def _hull(points: np.ndarray) -> trimesh.Trimesh:
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) < 4:
        raise ValueError(f"need at least 4 distinct points for a 3D hull, got {len(pts)}")
    hull = trimesh.convex.convex_hull(pts)
    if hull.volume < 1e-9:
        raise ValueError("points are (nearly) coplanar; cannot build a solid")
    return hull


def with_table_base(pts: np.ndarray, layer: float = 0.35) -> tuple[np.ndarray, str]:
    """Glove touches rarely reach the table, so a raw hull would float above it. Copy the lowest layer of points
    down to z = 0 (a flat base: right for boxes and upright cans, slightly generous for balls). Points below the
    table are clipped. Returns (points, note)."""
    pts = pts.copy()
    pts[:, 2] = np.maximum(pts[:, 2], 0.0)
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    if zmin <= 0.003 or zmax - zmin < 1e-3:
        return pts, ""
    low = pts[pts[:, 2] <= zmin + layer * (zmax - zmin)]
    base = low.copy()
    base[:, 2] = 0.0
    return np.vstack([pts, base]), f"base extended {zmin*1000:.0f} mm down to the table"


def _read_ply_xyz(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "end_header") + 1
    rows = [l.split()[:3] for l in lines[start:] if l.strip()]
    return np.array(rows, float).reshape(-1, 3)


def _touchmap_primitive(doc: dict, scale: float, pts: np.ndarray | None = None) -> Shape | None:
    models = doc.get("models") or {}
    name = doc.get("label") or doc.get("best_guess")
    if isinstance(name, dict):
        name = name.get("shape") or name.get("name")
    if name not in models:
        return None
    p = models[name]["params"]
    c = np.array(p["center_mm"], float) * 1e-3
    if name == "sphere":
        m = trimesh.creation.icosphere(subdivisions=3, radius=p["radius_mm"] * 1e-3)
    elif name == "cylinder":
        m = trimesh.creation.cylinder(radius=p["radius_mm"] * 1e-3, height=p["height_mm"] * 1e-3, sections=48)
        a = np.array(p["axis"], float)
        m.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], a / np.linalg.norm(a)))
    else:  # box: touchmap lists dims sorted largest-first but axes in fitted order; rebuild from the axes
        dims = np.array(p["dims_mm"], float) * 1e-3          # sorted largest first
        R = np.array(p["axes"], float)                       # rows = box axes, in the fitted (unsorted) order
        if pts is not None and len(pts) >= 4:
            # give the largest dimension to the axis along which the touches spread the most, and so on
            spread = np.ptp((pts - pts.mean(0)) @ R.T, axis=0)
            assigned = np.empty(3)
            assigned[np.argsort(-spread)] = dims
            dims = assigned
        m = trimesh.creation.box(extents=dims)
        T = np.eye(4)
        T[:3, :3] = R.T if abs(np.linalg.det(R) - 1) < 1e-3 else np.eye(3)
        m.apply_transform(T)
    m.apply_translation(c)
    return Shape(m, name, p, "touchmap fit", "rebuilt from fitted primitive")


def load_shape_json(path: str | Path, prefer: str = "cloud") -> tuple[Shape, np.ndarray | None]:
    """Load a glove-made shape. Returns (Shape grounded at the origin, original xy centre in metres or None).
    The second value is where the object stood in the source frame (used to propose a pose)."""
    path = Path(path)
    doc = json.loads(path.read_text())
    is_touchmap = "models" in doc or "probabilities" in doc
    scale = _scale(doc, "mm" if is_touchmap else "m")
    note = ""
    src = f"json:{path.name}"

    if "vertices" in doc and "faces" in doc:
        mesh = trimesh.Trimesh(np.array(doc["vertices"], float) * scale, np.array(doc["faces"], int), process=True)
        kind, params = "mesh", {}
    elif any(k in doc for k in ("points", "points_mm", "cloud")):
        pts = np.array(doc.get("points") or doc.get("points_mm") or doc.get("cloud"), float) * scale
        extra = ""
        if doc.get("ground", True):
            pts, extra = with_table_base(pts)
        mesh, kind, params = _hull(pts), "mesh", {}
        note = f"convex hull of {len(pts)} points" + (f"; {extra}" if extra else "")
    elif "shape" in doc and "params" in doc:
        s = make_shape(doc["shape"], **{k: v * scale for k, v in doc["params"].items()})
        s.source = src
        return s, None
    elif is_touchmap:
        ply = path.with_name("cloud.ply")
        shape = None
        if prefer == "cloud" and ply.exists() and len(_read_ply_xyz(ply)) >= 4:
            pts = _read_ply_xyz(ply) * 1e-3
            extra = ""
            if doc.get("ground", True):
                pts, extra = with_table_base(pts)
            mesh, kind, params = _hull(pts), "mesh", {}
            note = f"convex hull of {len(pts)} glove points (cloud.ply)" + (f"; {extra}" if extra else "")
            lbl = doc.get("label") or doc.get("best_guess")
            if lbl:
                note += f"; fit says {lbl}"
        else:
            cloud = _read_ply_xyz(ply) * 1e-3 if ply.exists() else None
            shape = _touchmap_primitive(doc, scale, cloud)
            if shape is None:
                raise ValueError(f"{path.name}: touchmap result has no usable fit and no cloud.ply next to it")
            mesh, kind, params, note = shape.mesh, shape.kind, shape.params, shape.note
    else:
        raise ValueError(f"{path.name}: unrecognised shape JSON (need vertices+faces, points, shape+params, or a touchmap result)")

    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{path.name}: empty or invalid mesh")
    mesh.fix_normals()
    lo, hi = mesh.bounds
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
    grounded, _ = ground(mesh)
    return Shape(grounded, kind, params, src, note), centre


def save_obj(shape: Shape, path: str | Path) -> Path:
    path = Path(path)
    shape.mesh.export(path)
    return path
