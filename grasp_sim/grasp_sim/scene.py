"""MuJoCo scene: Unitree G1 (torso fixed, right arm + right Dex3-style hand active), table, object."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
MENAGERIE = ROOT / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"

PELVIS_Z = -0.07         # pelvis height relative to the table surface (table top is z = 0 in the robot frame);
                         # lower shoulders = longer horizontal reach across the table
TABLE_X0 = 0.12          # table starts this far in front of the pelvis
FLOOR_Z = -0.74

ARM_JOINTS = [f"right_{n}_joint" for n in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
HAND_JOINTS = [f"right_hand_{n}_joint" for n in
               ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
WRIST_BODY = "right_wrist_yaw_link"


@dataclass
class SceneInfo:
    model: mujoco.MjModel
    data: mujoco.MjData
    arm_qadr: np.ndarray
    arm_dofadr: np.ndarray
    arm_act: np.ndarray
    hand_qadr: np.ndarray
    hand_act: np.ndarray
    wrist_id: int
    obj_body: int
    obj_geom: int
    hand_geoms: dict           # geom id -> "thumb" | "index" | "middle" | "palm" | "arm"
    table_geom: int
    stand_q: dict              # joint name -> value (home posture)


def _stand_pose() -> dict:
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(d.qpos[m.jnt_qposadr[j]])
    return out


def _group(name: str) -> str | None:
    if "right_hand_thumb" in name:
        return "thumb"
    if "right_hand_index" in name:
        return "index"
    if "right_hand_middle" in name:
        return "middle"
    if "right_hand_palm" in name or "right_wrist" in name:
        return "palm"
    if "right_elbow" in name or "right_shoulder" in name:
        return "arm"
    return None


def build_scene(mesh: trimesh.Trimesh, pos_m, quat_wxyz, mass_kg: float, mu: float, hollow: bool = False,
                pad_mu: float | None = None) -> SceneInfo:
    """`mesh` must already be rotated by the pose quaternion and have min z = 0 (see sim.prepare_object)."""
    stand = _stand_pose()
    spec = mujoco.MjSpec.from_file(str(MENAGERIE))
    spec.copy_during_attach = True

    # --- fix the base: drop the free joint and put the pelvis where we want it
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)
    pelvis = spec.body("pelvis")
    pelvis.pos = [0, 0, PELVIS_Z]

    # --- only the right arm and hand collide; everything else is a pose-holding ghost
    keep = {}
    for g in spec.geoms:
        grp = _group(g.name or "")
        parent = g.parent.name if g.parent else ""
        grp = grp or _group(parent)
        if grp is None:
            g.contype, g.conaffinity = 0, 0
        elif g.contype or g.conaffinity:
            # bit 2 = robot hand/arm: collides with table and object, never with itself
            g.contype, g.conaffinity = 2, 0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.timestep = 0.002

    # --- table (top surface at z = 0) and floor
    wb = spec.worldbody
    table = wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.6, 0.02],
                        pos=[TABLE_X0 + 0.5, 0, -0.02], rgba=[0.55, 0.42, 0.30, 1])
    table.friction = [0.6, 0.005, 0.0001]
    table.contype, table.conaffinity = 1, 3
    fl = wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[0, 0, FLOOR_Z],
                     rgba=[0.25, 0.28, 0.32, 1])
    fl.contype, fl.conaffinity = 1, 3
    wb.add_light(pos=[0.5, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    # --- the object: free body with a mesh geom
    m = spec.add_mesh()
    m.name = "obj_mesh"
    m.uservert = np.asarray(mesh.vertices, float).flatten()
    m.userface = np.asarray(mesh.faces, np.int32).flatten()
    if hollow and hasattr(mujoco, "mjtMeshInertia"):
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    ob = wb.add_body(name="object", pos=list(map(float, pos_m)))
    ob.add_freejoint()
    og = ob.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="obj_mesh",
                     mass=float(mass_kg), friction=[mu, 0.01, 0.001], condim=4,
                     rgba=[0.9, 0.35, 0.15, 1], solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    og.contype, og.conaffinity = 1, 2

    # --- hand pads get the pad friction; condim 4 gives torsional friction (helps round objects)
    pm = pad_mu if pad_mu is not None else mu
    for g in spec.geoms:
        grp = _group(g.name or "") or _group(g.parent.name if g.parent else "")
        if grp in ("thumb", "index", "middle", "palm") and g.contype:
            g.friction = [pm, 0.01, 0.001]
            g.condim = 4
            g.solref = [0.004, 1]
            g.solimp = [0.95, 0.99, 0.001, 0.5, 2]

    model = spec.compile()
    data = mujoco.MjData(model)

    def jid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

    for n, v in stand.items():
        j = jid(n)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = v

    def act_for(joint):
        j = jid(joint)
        for a in range(model.nu):
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == j:
                return a
        raise KeyError(joint)

    # hold every non-right-arm/hand joint at its home angle
    for a in range(model.nu):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
        if jn in stand:
            data.ctrl[a] = stand[jn]

    hand_geoms = {}
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        b = model.geom_bodyid[gid]
        grp = _group(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if grp:
            hand_geoms[gid] = grp

    info = SceneInfo(
        model=model, data=data,
        arm_qadr=np.array([model.jnt_qposadr[jid(n)] for n in ARM_JOINTS]),
        arm_dofadr=np.array([model.jnt_dofadr[jid(n)] for n in ARM_JOINTS]),
        arm_act=np.array([act_for(n) for n in ARM_JOINTS]),
        hand_qadr=np.array([model.jnt_qposadr[jid(n)] for n in HAND_JOINTS]),
        hand_act=np.array([act_for(n) for n in HAND_JOINTS]),
        wrist_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY),
        obj_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object"),
        obj_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"),
        hand_geoms=hand_geoms,
        table_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"),
        stand_q=stand,
    )
    # the stock arm torque limits are conservative for a pose-holding task; let the arm track the IK path
    model.actuator_forcerange[info.arm_act, 0] = [-80, -80, -80, -60, -30, -20, -20]
    model.actuator_forcerange[info.arm_act, 1] = [80, 80, 80, 60, 30, 20, 20]
    mujoco.mj_forward(model, data)
    return info
