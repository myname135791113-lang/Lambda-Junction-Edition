"""Simulated lift: approach -> close -> squeeze to target force -> lift 10 cm -> hold 3 s."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .grasp import CLOSED, OPEN, Grasp
from .kin import solve_ik
from .scene import SceneInfo

DT_CTRL = 0.01
LIFT_M = 0.10
HOLD_S = 3.0
SLIP_LIMIT_M = 0.010        # more than 10 mm of movement relative to the hand = SLIPPED
DROP_MARGIN_M = 0.04        # object more than 4 cm below the lift target at the end = DROPPED


@dataclass
class SimResult:
    outcome: str
    peak_force_N: float
    max_force_N: float
    slip_mm: float
    target_force_N: float
    lift_height_mm: float
    notes: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    log: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"outcome": self.outcome, "peak_contact_force_N": round(self.peak_force_N, 2),
                "max_force_N": self.max_force_N, "slip_mm": round(self.slip_mm, 2),
                "target_grip_force_N": round(self.target_force_N, 2),
                "final_lift_height_mm": round(self.lift_height_mm, 1), "notes": self.notes}


class Runner:
    def __init__(self, si: SceneInfo, grasp: Grasp, target_force: float, max_force: float,
                 render: bool = False, size=(480, 640), fps=30):
        self.si, self.g = si, grasp
        self.m, self.d = si.model, si.data
        self.target, self.max_force = target_force, max_force
        self.fps = fps
        self.frames = []
        self.rend = None
        if render:
            self.rend = mujoco.Renderer(self.m, *size)
            self.cam = mujoco.MjvCamera()
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.85, 118, -28
        self._next_frame = 0.0
        self.peak = 0.0
        self.s = 0.0                       # closing parameter 0..1
        self.rel_ref = None                # object pose in the hand when the lift starts (slip reference)
        self.raised = False
        self.target0 = target_force
        self.hand_open = OPEN[grasp.kind]
        self.t_log, self.f_log, self.z_log = [], [], []

    # ---------------------------------------------------------------- low level
    def contacts(self):
        """(thumb_normal_sum, finger_normal_sum, peak single contact force, n_contacts) on the object."""
        m, d = self.m, self.d
        th = fi = peak = 0.0
        n = 0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if self.si.obj_geom not in (a, b):
                continue
            other = b if a == self.si.obj_geom else a
            grp = self.si.hand_geoms.get(other)
            if grp is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = max(f6[0], 0.0)
            peak = max(peak, float(np.linalg.norm(f6[:3])))
            n += 1
            if grp == "thumb":
                th += fn
            elif grp in ("index", "middle"):
                fi += fn
            else:
                th += 0.0
        return th, fi, peak, n

    def hand_ctrl(self, s=None):
        """Hand position target between the open pre-shape (s = 0) and fully closed (s = 1)."""
        s = self.s if s is None else s
        return self.hand_open + s * (CLOSED - self.hand_open)

    def step(self, arm_q, hand_q, n=None):
        """Advance one control period."""
        d, m = self.d, self.m
        d.ctrl[self.si.arm_act] = arm_q
        d.ctrl[self.si.hand_act] = hand_q
        for _ in range(n or int(round(DT_CTRL / m.opt.timestep))):
            mujoco.mj_step(m, d)
            _, _, pk, _ = self.contacts()
            self.peak = max(self.peak, pk)
            if self.rend is not None and d.time >= self._next_frame:
                self._next_frame += 1.0 / self.fps
                self.capture()

    def capture(self):
        d = self.d
        self.cam.lookat[:] = [0.26, -0.14, 0.10]
        self.rend.update_scene(d, self.cam)
        self.frames.append(self.rend.render().copy())

    def log(self):
        th, fi, pk, n = self.contacts()
        self.t_log.append(self.d.time)
        self.f_log.append((th, fi, pk))
        self.z_log.append(float(self.d.xpos[self.si.obj_body][2]))

    def move_arm(self, q_from, q_to, dur, hand_q=None, regulate=False):
        n = max(1, int(dur / DT_CTRL))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)          # smooth-step
            q = q_from + a * (q_to - q_from)
            if regulate:
                self.slip_watch()
                self.regulate()
            self.step(q, hand_q if hand_q is not None else self.hand_ctrl())
            self.log()

    def slip_watch(self):
        """If the object creeps in the hand, raise the squeeze target by 25 % (never above 85 % of the damage limit)."""
        if self.rel_ref is None:
            return
        d, si = self.d, self.si
        R = d.xmat[si.wrist_id].reshape(3, 3)
        rel = R.T @ (d.xpos[si.obj_body] - d.xpos[si.wrist_id])
        if np.linalg.norm(rel - self.rel_ref) > 0.003:
            cap = 0.85 * self.max_force
            if self.target < cap:
                self.target = min(self.target * 1.25, cap)
                self.raised = True
            self.rel_ref = rel

    def regulate(self):
        """Squeeze-force feedback: move the closing parameter so both sides carry the target force.
        The force measured is the mean of the thumb side and the finger side normal forces."""
        th, fi, pk, n = self.contacts()
        f = 0.5 * (th + fi) if (th > 0 and fi > 0) else 0.5 * max(th, fi)
        err = self.target - f
        # gentler steps for light / fragile objects: a position servo this stiff turns 1 mrad into ~10 N
        gain = 2e-4 * min(1.0, max(self.target, 1.0) / 8.0)
        lim = 6e-4 * min(1.0, max(self.target, 1.0) / 8.0)
        ds = float(np.clip(gain * err, -lim, lim))
        if pk > 0.92 * self.max_force:                  # never knowingly crush: back off
            ds = min(ds, -lim)
        self.s = float(np.clip(self.s + ds, 0.0, 1.0))

    # ---------------------------------------------------------------- the routine
    def run(self) -> SimResult:
        si, g, d, m = self.si, self.g, self.d, self.m
        notes = []
        # the robot starts in a "ready" pose: hand open, held above the table beside the object
        hover = g.pre_pos.copy()
        hover[2] = max(hover[2], 0.20)
        q_h, ok_h, *_ = solve_ik(si, hover, g.R, q0=g.q_pre)
        d.qpos[si.arm_qadr] = q_h
        d.qpos[si.hand_qadr] = self.hand_ctrl(0.0)
        mujoco.mj_forward(m, d)
        self.step(q_h, self.hand_ctrl(0.0), n=400)
        self.move_arm(q_h, g.q_pre, 1.2, hand_q=self.hand_ctrl(0.0))
        # straight-line approach to the grasp pose (IK at waypoints)
        qs = [g.q_pre]
        for a in np.linspace(0.25, 1.0, 4):
            q, *_ = solve_ik(si, g.pre_pos + a * (g.wrist_pos - g.pre_pos), g.R, q0=qs[-1])
            qs.append(q)
        for k in range(len(qs) - 1):
            self.move_arm(qs[k], qs[k + 1], 0.5, hand_q=self.hand_ctrl(0.0))
        self.step(qs[-1], self.hand_ctrl(0.0), n=100)
        obj_before = d.xpos[si.obj_body].copy()
        if np.linalg.norm(obj_before[:2] - np.array(self.obj_xy0)) > 0.03:
            notes.append("object was knocked during the approach")

        # close: fast until first contact, then force-regulated
        t_close = 0.0
        while t_close < 4.0:
            th, fi, pk, n = self.contacts()
            if n == 0:
                self.s = min(1.0, self.s + 0.004 * min(1.0, max(self.target, 1.0) / 8.0 + 0.4))
            else:
                self.regulate()
            self.step(qs[-1], self.hand_ctrl())
            self.log()
            t_close += DT_CTRL
        th, fi, pk, n = self.contacts()
        if n == 0:
            notes.append("fingers closed without touching the object")
        # squeeze hold
        for _ in range(60):
            self.regulate()
            self.step(qs[-1], self.hand_ctrl())
            self.log()

        # lift 10 cm (IK at the end, joint-space blend), keep regulating
        lift_target = g.wrist_pos + np.array([0, 0, LIFT_M])
        q_lift, ok_l, ep, er = solve_ik(si, lift_target, g.R, q0=qs[-1])
        if not ok_l:
            notes.append(f"lift pose IK error {ep*1000:.1f} mm")
        wrist_pos0 = d.xpos[si.wrist_id].copy()
        R0 = d.xmat[si.wrist_id].reshape(3, 3).copy()
        rel0 = R0.T @ (d.xpos[si.obj_body] - wrist_pos0)
        z0 = d.xpos[si.obj_body][2]
        self.rel_ref = rel0.copy()
        self.move_arm(qs[-1], q_lift, 2.0, regulate=True)
        # hold
        n = int(HOLD_S / DT_CTRL)
        for _ in range(n):
            self.slip_watch()
            self.regulate()
            self.step(q_lift, self.hand_ctrl())
            self.log()
        wp = d.xpos[si.wrist_id]
        R1 = d.xmat[si.wrist_id].reshape(3, 3)
        rel1 = R1.T @ (d.xpos[si.obj_body] - wp)
        slip = float(np.linalg.norm(rel1 - rel0))
        z1 = d.xpos[si.obj_body][2]
        lift_h = float(z1 - z0)
        th, fi, pk, n_c = self.contacts()

        if self.peak > self.max_force:
            outcome = "BROKEN"
        elif lift_h < LIFT_M - DROP_MARGIN_M or n_c == 0:
            outcome = "DROPPED"
        elif slip > SLIP_LIMIT_M:
            outcome = "SLIPPED"
        else:
            outcome = "SUCCESS"
        if self.raised:
            notes.append(f"slip detected: squeeze target raised {self.target0:.1f} -> {self.target:.1f} N")
        res = SimResult(outcome, self.peak, self.max_force, slip * 1000, self.target, lift_h * 1000, notes, self.frames,
                        {"t": self.t_log, "f": self.f_log, "z": self.z_log})
        return res
