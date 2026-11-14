"""Match a 3D tactile scan against a known Object Catalog and build a validated Object Card.

Bridges the missing-weight gap on force-less mocap gloves:
  3D tactile geometry -> Catalog retrieval -> Known physics (`mass_kg`, `friction`, `max_grip_n`,
  `material`, `grasp`) -> Schema-validated `glove_twin` Object Card (`v1`).

Supports both:
  1. The default repo catalog (`glove_twin.objects.OBJECTS`: `redbull`, `profeel`, `wood_sphere`)
  2. Custom JSON catalog additions (`load_custom_catalog`) so teams can add any table object
     on-site with calipers + a kitchen scale in 30 seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from datetime import datetime, timezone

import numpy as np

from glove_twin.density import estimate_density
from glove_twin.materials import CONTAINER_MATERIALS, MATERIALS
from glove_twin.objects import OBJECTS, RefObject
from glove_twin.shape_fit import ShapeResult

try:
    from glove_twin.object_card import build_card as _gt_build_card, validate_card as _gt_validate_card
except ImportError:  # e.g. laptop with numpy only (before `pip install mujoco`)
    _gt_build_card = None
    _gt_validate_card = None

from .cloud_hull import ScanSummary


def load_catalog(extra_json_path: str | Path | None = None) -> dict[str, RefObject]:
    """Load `glove_twin.objects.OBJECTS` plus any extra items from a JSON file."""
    catalog: dict[str, RefObject] = dict(OBJECTS)
    if extra_json_path is not None:
        raw = json.loads(Path(extra_json_path).read_text(encoding="utf-8"))
        items = raw["objects"] if isinstance(raw, dict) and "objects" in raw else raw
        for item in items:
            obj = RefObject(
                key=str(item["key"]),
                label=str(item.get("label", item["key"])),
                shape=str(item["shape"]),
                dims={k: float(v) for k, v in item["dims"].items()},
                mass_kg=float(item["mass_kg"]),
                mass_note=str(item.get("mass_note", "measured for catalog")),
                material=str(item.get("material", "hard_plastic")),
                friction=float(item.get("friction", 0.45)),
                max_grip_n=float(item.get("max_grip_n", 60.0)),
                container=bool(item.get("container", False)),
            )
            catalog[obj.key] = obj
    return catalog


@dataclass
class CatalogMatchResult:
    best_key: str | None
    confident: bool
    probabilities: dict[str, float]
    scores_detail: dict[str, dict[str, float]]
    advice: str


def match_scan_to_catalog(
    scan: ScanSummary,
    catalog: dict[str, RefObject] | None = None,
    *,
    confidence_threshold: float = 0.80,
    min_voxels: int = 10,
) -> CatalogMatchResult:
    """Compute posterior match probabilities across `catalog` from metric 3D geometry."""
    cat = OBJECTS if catalog is None else catalog
    if scan.n_voxels < min_voxels:
        n = max(len(cat), 1)
        return CatalogMatchResult(
            best_key=None,
            confident=False,
            probabilities={k: round(1.0 / n, 3) for k in cat},
            scores_detail={},
            advice=f"Keep scanning ({scan.n_voxels}/{min_voxels} voxels): trace top, sides, and base seam.",
        )

    prim_weights = scan.primitive_fit.get("weights", {})
    prim_fits = scan.primitive_fit.get("fits", {})

    # Estimate horizontal (L, W) and height from both OBB and best primitive fit for robustness
    H_meas = scan.height_m
    L_meas = scan.length_m
    W_meas = scan.width_m

    log_scores: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}

    for key, ref in cat.items():
        ext = ref.extents()  # [x, y, z] upright bounding box in metres
        L_ref, W_ref = sorted([float(ext[0]), float(ext[1])], reverse=True)
        H_ref = float(ext[2])
        V_ref = float(ref.volume_m3)

        # If we have a template fit for `ref.shape`, blend its fitted extents with raw OBB
        t_fit = prim_fits.get(ref.shape)
        if t_fit is not None:
            td = t_fit["dims_m"]
            if ref.shape == "sphere":
                L_eff = W_eff = H_eff = 2.0 * td["radius"]
            elif ref.shape == "cylinder":
                L_eff = W_eff = 2.0 * td["radius"]
                H_eff = 0.65 * td["height"] + 0.35 * H_meas
            else:
                L_eff, W_eff = sorted([td["width"], td["depth"]], reverse=True)
                H_eff = 0.65 * td["height"] + 0.35 * H_meas
        else:
            L_eff, W_eff, H_eff = L_meas, W_meas, H_meas

        # Normalized squared residuals (metric tolerances ~9-11 mm)
        err_h = ((H_eff - H_ref) / 0.011) ** 2
        err_l = ((L_eff - L_ref) / 0.011) ** 2
        err_w = ((W_eff - W_ref) / 0.011) ** 2

        # Hull volume term (a partial scan hull typically captures ~80-100% of true volume)
        err_v = 0.0
        if scan.hull_volume_m3 > 1e-6:
            corrected_hull_v = scan.hull_volume_m3 / 0.88
            err_v = ((corrected_hull_v - V_ref) / max(0.30 * V_ref, 5e-5)) ** 2

        # Caliper pinch likelihood (if user took any thumb-index caliper widths)
        err_cal = 0.0
        if scan.calipers_m:
            ref_dims = np.array([L_ref, W_ref, H_ref])
            for c_val in scan.calipers_m:
                best_d = float(np.min(np.abs(ref_dims - c_val)))
                err_cal += (best_d / 0.007) ** 2

        # Shape-class BIC weight term (separates Red Bull cylinder from similar-size PROfeel box!)
        p_shape = max(float(prim_weights.get(ref.shape, 0.15)), 0.02)
        shape_nll = -2.2 * math.log(p_shape)

        total_d2 = err_h + err_l + err_w + 0.6 * err_v + err_cal + shape_nll
        log_scores[key] = -0.5 * total_d2
        details[key] = {
            "H_mm": round(H_eff * 1000.0, 1),
            "L_mm": round(L_eff * 1000.0, 1),
            "W_mm": round(W_eff * 1000.0, 1),
            "ref_H_mm": round(H_ref * 1000.0, 1),
            "ref_L_mm": round(L_ref * 1000.0, 1),
            "ref_W_mm": round(W_ref * 1000.0, 1),
            "shape_prob": round(p_shape, 3),
            "distance_sq": round(total_d2, 2),
        }

    mx = max(log_scores.values())
    exps = {k: math.exp(v - mx) for k, v in log_scores.items()}
    s = sum(exps.values())
    probs = {k: round(v / s, 4) for k, v in exps.items()}
    ranked = sorted(probs, key=probs.get, reverse=True)
    best_k = ranked[0]
    confident = probs[best_k] >= confidence_threshold and scan.n_voxels >= 14

    if confident:
        advice = f"Matched {cat[best_k].label} ({probs[best_k] * 100:.0f}%). Ready to grab & relocate!"
    elif len(ranked) >= 2:
        a, b = cat[ranked[0]], cat[ranked[1]]
        if a.shape != b.shape and {a.shape, b.shape} == {"box", "cylinder"}:
            advice = "Close sizes (box vs cylinder): trace across a corner/flat face or grasp at 45 deg!"
        else:
            advice = f"Top candidates: {a.key} vs {b.key} — sweep top & base seam for exact height/width."
    else:
        advice = "Continue tracing object surfaces."

    return CatalogMatchResult(
        best_key=best_k,
        confident=confident,
        probabilities=probs,
        scores_detail=details,
        advice=advice,
    )


def validate_card_schema(card: dict[str, Any]) -> list[str]:
    if _gt_validate_card is not None:
        return _gt_validate_card(card)
    problems = []
    for k in ("card_version", "object", "shape", "mass_kg", "material", "friction", "max_safe_grip_n", "grasp"):
        if k not in card:
            problems.append(f"missing {k}")
    if card.get("mass_kg", 0) <= 0:
        problems.append("mass must be positive")
    if "grasp" in card and card["grasp"].get("grip_force_n", 0) <= 0:
        problems.append("grip force must be positive")
    return problems


def build_matched_object_card(
    ref_obj: RefObject,
    scan: ScanSummary,
    match: CatalogMatchResult,
    *,
    final_pose_ws: dict[str, Any] | None = None,
    final_pose_robot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a schema-valid `glove_twin` Object Card (`v1`) enriched with live 3D scan & pose metadata."""
    mat_key = ref_obj.material if ref_obj.material in MATERIALS else "hard_plastic"
    mat_probs = {m: (0.93 if m == mat_key else round(0.07 / max(len(MATERIALS) - 1, 1), 4)) for m in MATERIALS}

    # Build a `ShapeResult` compatible with `glove_twin.object_card.build_card`
    if ref_obj.shape == "sphere":
        dims = {"radius": float(ref_obj.dims["radius"])}
    elif ref_obj.shape == "cylinder":
        dims = {"radius": float(ref_obj.dims["radius"]), "height": float(ref_obj.dims["height"])}
    else:
        w, d = sorted([float(ref_obj.dims["x"]), float(ref_obj.dims["y"])])
        dims = {"width": w, "depth": d, "height": float(ref_obj.dims["z"])}

    sigma = {k: 0.004 for k in dims}
    shape_res = ShapeResult(
        best=ref_obj.shape,
        fits={},
        weights=scan.primitive_fit.get("weights", {ref_obj.shape: 1.0}),
        ambiguous=not match.confident,
        reasons=[] if match.confident else [match.advice],
        dims=dims,
        sigma=sigma,
        coverage={"voxels": scan.n_voxels},
        touch_here="none" if match.confident else match.advice,
    )

    planned_force = round(
        min(
            max(1.5 * ref_obj.mass_kg * 9.81 / max(2.0 * ref_obj.friction, 1e-3), 1.0),
            0.80 * float(ref_obj.max_grip_n),
        ),
        2,
    )

    if _gt_build_card is not None:
        card = _gt_build_card(
            name=ref_obj.key,
            label=ref_obj.label,
            shape=shape_res,
            mass_kg=ref_obj.mass_kg,
            mass_source=f"catalog match ({match.probabilities.get(ref_obj.key, 0.0) * 100:.1f}%): {ref_obj.mass_note}",
            material_probs=mat_probs,
            friction=ref_obj.friction,
            n_contacts=scan.n_voxels,
            n_grasps=max(1, len(scan.calipers_m)),
        )
    else:
        vol = float(ref_obj.volume_m3)
        dens = estimate_density(mat_probs, ref_obj.mass_kg * 1000.0, vol * 1e6)
        if ref_obj.shape == "cylinder":
            grasp = {"type": "power_wrap", "approach": "side", "wrap_axis": "object long axis",
                     "grip_width_m": 2.0 * dims["radius"], "preshape": "open"}
        elif ref_obj.shape == "box":
            grasp = {"type": "box_wrap", "approach": "side", "wrap_axis": "object long axis",
                     "grip_width_m": dims["width"], "palm_on": "a face across the narrowest width", "preshape": "open"}
        else:
            grasp = {"type": "spherical_power", "approach": "side", "grip_width_m": 2.0 * dims["radius"],
                     "preshape": "sphere"}
        grasp["feasible"] = bool(grasp["grip_width_m"] <= 0.12)
        grasp["grip_force_n"] = planned_force
        grasp["grip_force_rule"] = "1.5 x weight / (2 x friction), capped at 80% of the safe squeeze"
        card = {
            "card_version": 1,
            "object": ref_obj.key,
            "label": ref_obj.label,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shape": {
                "primitive": ref_obj.shape,
                "dims_m": {k: round(v, 4) for k, v in dims.items()},
                "sigma_m": sigma,
                "unmeasured": [],
                "model_weights": shape_res.weights,
                "ambiguous": shape_res.ambiguous,
                "reasons": shape_res.reasons,
                "touch_here_next": shape_res.touch_here,
                "contacts": scan.n_voxels,
                "grasps": max(1, len(scan.calipers_m)),
                "follow_up_rounds": 0,
            },
            "mass_kg": round(ref_obj.mass_kg, 4),
            "mass_source": f"catalog match ({match.probabilities.get(ref_obj.key, 0.0) * 100:.1f}%): {ref_obj.mass_note}",
            "volume_m3": round(vol, 7),
            "bulk_density_kg_m3": round(dens.value, 1),
            "container": mat_key in CONTAINER_MATERIALS,
            "material": {"best": mat_key, "probabilities": mat_probs},
            "friction": round(ref_obj.friction, 3),
            "friction_source": "catalog prior",
            "max_safe_grip_n": float(ref_obj.max_grip_n),
            "fragile": float(ref_obj.max_grip_n) <= 20.0,
            "grasp": grasp,
        }

    card["max_safe_grip_n"] = float(ref_obj.max_grip_n)
    card["fragile"] = float(ref_obj.max_grip_n) <= 20.0
    card["grasp"]["grip_force_n"] = planned_force

    card["catalog_match"] = {
        "matched_key": ref_obj.key,
        "confidence": match.probabilities.get(ref_obj.key, 0.0),
        "probabilities": match.probabilities,
    }
    card["scan_3d"] = {
        "voxels": scan.n_voxels,
        "raw_points": scan.n_raw,
        "measured_height_m": round(scan.height_m, 4),
        "measured_length_m": round(scan.length_m, 4),
        "measured_width_m": round(scan.width_m, 4),
        "convex_hull_volume_cm3": round(scan.hull_volume_m3 * 1e6, 1),
        "calipers_m": scan.calipers_m,
    }
    if final_pose_ws is not None:
        card["pose_workspace"] = final_pose_ws
    if final_pose_robot is not None:
        card["pose_unitree_g1_base"] = final_pose_robot

    problems = validate_card_schema(card)
    if problems:
        raise ValueError(f"Generated invalid object card: {problems}")
    return card
