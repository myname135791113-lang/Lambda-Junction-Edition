# grasp_sim: can the robot hand pick this up?

Implements `specs.txt` (14 Nov 2026). Unitree G1 with the Dex3-style 3-finger hand (MuJoCo Menagerie),
torso fixed, right arm (7 DoF) and hand (7 joints) active, MuJoCo on the CPU, no GPU.

1. **Feasibility check** (instant, no physics): `CAN_PICK`, or `MISSING_INPUT`, `TOO_FAR`, `TOO_BIG`, `TOO_HEAVY`, `NO_SAFE_FORCE`.
   It also returns the chosen grasp and the target grip force.
2. **Simulated lift**: approach, close, squeeze to the target force, lift 10 cm, hold 3 s.
   Reports `SUCCESS` / `SLIPPED` / `DROPPED` / `BROKEN`, peak contact force vs `max_force_N`, slip distance, and writes a video.

## Setup (already done on this laptop)
```
pip install mujoco trimesh imageio imageio-ffmpeg pillow scipy numpy
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # into this folder
```

## Use
```
python -m grasp_sim gui                       # window: pick shape, edit parameters, preview, check, run
python -m grasp_sim run --shape box --size 60 60 120 --mass 0.2 --max-force 30
python -m grasp_sim run --shape cylinder --radius 33 --height 115 --mass 0.35 --hollow
python -m grasp_sim run --shape sphere --radius 20 --mass 0.0027 --max-force 8 --mu 0.5
python -m grasp_sim run --json path\to\result.json            # glove-made shape (see below)
python -m grasp_sim run --json path\to\result.json --use-fit  # same, but the fitted primitive instead of the hull
python -m grasp_sim run --card card.json                      # the spec's object card
python -m grasp_sim run ... --check-only                      # feasibility only, no physics
python -m grasp_sim suite                                     # the spec's 8-object test set
python -m grasp_sim showcase                                  # test set + glove sessions + runs\showcase\index.html
```
Every run writes `runs/<time>_<name>/`: `video.mp4`, `preview.png`, `result.json`, `card.json`, `<name>.obj`.
Options: `--x --y --yaw` (object pose, robot frame, table top is z = 0), `--mass`, `--mu` (object vs finger-pad friction),
`--material`, `--max-force`, `--hollow`, `--attempts N` (different grasps to try, default 3), `--no-video`.

## Shapes
- **Built-in**: box (`--size X Y Z`, mm), cylinder (`--radius --height`), sphere (`--radius`). Defaults are in `shapes.py`.
- **Glove-made JSON** (`shapes.load_shape_json`), any of:
  - a `touchmap_grasp` session `result.json` with its `cloud.ply` beside it (what the Rokoko pipeline writes, mm).
    Default: convex hull of the fingertip cloud. Glove touches rarely reach the table, so the lowest 35 % of points
    are copied down to z = 0 (a flat base). `--use-fit` rebuilds the fitted box/cylinder/sphere instead.
    The object is placed where the glove saw it unless `--x/--y` are given.
  - `{"vertices": [...], "faces": [...]}` (a mesh), `{"points": [...]}` (cloud, hull), `{"shape": "box", "params": {...}}`.
    Add `"units": "mm"` or `"m"` when it is not the default (mm for touchmap, m otherwise).
- Card `mesh` files (`.obj/.stl/.ply`) are re-centred horizontally and rested on the table; `pose.quat_wxyz` rotates them.

## How it decides (all in `pipeline.py`, `grasp.py`)
- **Grasp families**: `side` (palm faces the object, fingers stacked vertically; tall things) and `top` (palm down;
  small or round things). For each approach yaw (-110..110 deg) the arm IK (`kin.py`) must reach both the grasp and
  a pre-grasp pose. No solution for any: `TOO_FAR`. Width across the closing direction must be at most 95 mm: else `TOO_BIG`.
- **Force**: friction on both sides must carry the weight with a safety factor of 2:
  `F_min = 2 m g / (2 mu)` per side. `NO_SAFE_FORCE` if `F_min > 0.85 * max_force_N`. The target is `1.25 F_min`,
  at least 2 N (the hand cannot regulate lower), at most `0.85 max_force_N`.
- **Squeeze**: force feedback on the measured thumb-side and finger-side normal forces; backs off above 92 % of the limit.
  During the lift, if the object creeps more than 3 mm in the hand, the target is raised 25 % (never above 85 % of the limit).
- **Outcome**: `BROKEN` if any contact force exceeded `max_force_N` at any moment; else `DROPPED` if it did not stay
  lifted; else `SLIPPED` if it moved more than 10 mm relative to the hand; else `SUCCESS`.
- With `--attempts 3` the sim tries up to 3 different grasps headless and keeps the first success (the video shows the one used).

## Results on the spec's test set (`python -m grasp_sim suite`)
| object | verdict | outcome |
|---|---|---|
| ping-pong ball (r 20 mm, 2.7 g) | CAN_PICK | SUCCESS |
| tennis ball (r 33.5 mm, 57 g) | CAN_PICK | SUCCESS |
| empty can (r 33, h 115 mm, 15 g) | CAN_PICK | SUCCESS |
| full can (350 g) | CAN_PICK | SUCCESS (needed a second grasp: the first overshot the force limit) |
| tissue box (65 x 65 x 90 mm, 70 g, yawed 30 deg) | CAN_PICK | SUCCESS |
| paper cup (cone, 80 mm top, 12 g) | CAN_PICK | SUCCESS |
| can placed at x = 0.95 m | TOO_FAR | not simulated |
| 800 g block | TOO_HEAVY | not simulated |

## Things to know (honest limits)
- **Sizes, masses, friction and break limits of the test objects are my assumptions** (`suite.py`). Replace with measured values.
  The friction gap from the spec is real: `mu_pad` is object vs robot pad. The card's `material` has a rough lookup (`card.py`).
- **The grasp offsets were tuned in this simulator** (`grasp.py`: `CENTERS`, `SIDE_MIN_Z`, `TOP_PALM_CLEAR`) on this object set.
  Results change with small geometry changes (for example the table height or the robot's distance to the table);
  that is why 3 attempts are the default.
- **Hand size window**: balls down to about 40 mm and objects up to about 80 mm wide work. Flat objects (under about 5 cm tall,
  e.g. a flat tissue pack) fail with both grasp families. The hand is too large to pinch them.
- **Reach**: with the pelvis 7 cm below the table top (`scene.py: PELVIS_Z`), side grasps work out to about x = 0.45 m and
  top grasps out to about x = 0.5 m, for y between -0.35 and +0.05. Beyond that you get `TOO_FAR`.
- **Glove meshes are rough.** The hull of about 30 noisy fingertip points is a lumpy shape, larger or smaller than the real
  object. In the 6 sample runs (`samples/touchmap_ref`, produced by the real `touchmap_grasp.py` in synthetic mode)
  the hull succeeded for the box and failed for the cylinder and sphere. The fitted primitive (`--use-fit`) succeeded for the cylinder
  and the sphere (when moved into reach) and failed for the box: touchmap's box fit lists dimensions sorted largest-first but axes in fitted order;
  `shapes.py` re-matches them using the point cloud, which is a guess. Treat both as inputs to look at in the video.
- **Hollow** uses MuJoCo's shell inertia on the mesh. **Convex only**: MuJoCo collides meshes as convex hulls, so a cup is
  a solid cone, not an open container.
- The slip/drop/break thresholds (10 mm, 4 cm, 3 mm creep trigger) and the 2 N floor are my choices.
- Not committed to git; nothing here touches the `Lambda-Junction-Edition` repository.

## Files
`grasp_sim/` : `shapes.py` (primitives + JSON import), `card.py`, `scene.py` (MuJoCo scene), `kin.py` (IK), `grasp.py` (planner),
`pipeline.py` (feasibility, force, attempts), `sim.py` (the lift), `run.py` (build + outputs + video captions),
`suite.py` (test set), `gui.py`, `gallery.py`, `__main__.py` (CLI). `samples/touchmap_ref/` : the touchmap script and 3 sessions used as test input.
