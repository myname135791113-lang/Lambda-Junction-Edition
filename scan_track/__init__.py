"""scan_track: Live 3D tactile point-cloud reconstruction, Convex Hull + catalog matching,
single-grasp relocation tracking, MuJoCo twin verification, and Unitree G1 pick planning.

Supports:
  - Rokoko Smartgloves II standalone ("planted-wrist" mode & caliper/multi-grasp mode)
  - Rokoko Smartgloves II + Coil Pro (full 6-DoF room-scale scan & pick-and-place tracking)
  - Synthetic animated dual-hand exploration & relocation stream (zero hardware needed)
"""
from __future__ import annotations

__version__ = "0.1.0"
