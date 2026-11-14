"""Inverse kinematics for the right arm (damped least squares on position + orientation)."""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import SceneInfo


def rot_err(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) rotating R_cur onto R_tgt."""
    Re = R_tgt @ R_cur.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Re.flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def solve_ik(si: SceneInfo, pos, R, q0=None, iters=400, tol_p=1e-3, tol_r=2e-2, w_rot=0.5):
    """Return (q_arm, ok, pos_err, rot_err) for the wrist body to reach pos (3,) with rotation R (3,3)."""
    m = si.model
    d = mujoco.MjData(m)
    d.qpos[:] = si.data.qpos
    if q0 is not None:
        d.qpos[si.arm_qadr] = q0
    lo = m.jnt_range[[m.dof_jntid[a] for a in si.arm_dofadr]]
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    best = None
    for it in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        ep = np.asarray(pos) - d.xpos[si.wrist_id]
        er = rot_err(d.xmat[si.wrist_id].reshape(3, 3), R)
        n_p, n_r = np.linalg.norm(ep), np.linalg.norm(er)
        score = n_p + 0.05 * n_r
        if best is None or score < best[0]:
            best = (score, d.qpos[si.arm_qadr].copy(), n_p, n_r)
        if n_p < tol_p and n_r < tol_r:
            break
        mujoco.mj_jacBody(m, d, jp, jr, si.wrist_id)
        J = np.vstack([jp[:, si.arm_dofadr], w_rot * jr[:, si.arm_dofadr]])
        e = np.r_[ep, w_rot * er]
        lam = 1e-3 + 0.05 * np.linalg.norm(e)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        dq = np.clip(dq, -0.2, 0.2)
        q = np.clip(d.qpos[si.arm_qadr] + dq, lo[:, 0], lo[:, 1])
        d.qpos[si.arm_qadr] = q
    _, q, n_p, n_r = best
    return q, bool(n_p < tol_p * 3 and n_r < tol_r * 2), float(n_p), float(n_r)
