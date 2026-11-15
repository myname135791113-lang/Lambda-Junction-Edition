# Lambda - Junction Edition

**Scan an object by touch with Rokoko Smartgloves, recognise it, track it as you move it, and hand a verified pick plan to a Unitree G1 humanoid.**

The gloves have no force sensing, so the system builds a 3D point cloud from fingertip contacts, matches it against a catalog of known objects, and checks the grasp in a MuJoCo twin before it sends anything to the robot.

```
touch scan → 3D point cloud + hull → catalog match → object card → carry & place → MuJoCo check → G1 pick plan
```

## Quick start

```bash
pip install -r requirements.txt
python -m scan_track live --source synthetic     # live 3D dashboard at http://127.0.0.1:8765, no hardware
python -m glove_twin.visual                      # HTML report: touch mapping + robot grasps
python -m unittest discover -s tests -v          # tests
```

The synthetic demo loops by itself: scan → lock on at >90 % confidence → carry → place → MuJoCo lift check → G1 plan → restart with the next object.

With real gloves, stream from Rokoko Studio (Custom Streaming, UDP, JSON v4, port 14043):

```bash
python -m scan_track live --source rokoko --mode planted   # Smartgloves only: wrist planted next to the object
python -m scan_track live --source rokoko --mode coil      # with Coil Pro: room-scale tracking
```

### With or without the Coil Pro

The same program runs in two modes, chosen with `--mode` (or the mode dropdown in the dashboard):

| | `--mode planted` (Smartgloves only) | `--mode coil` (Smartgloves + Coil Pro) |
|---|---|---|
| Hand position in the room | Not known. The right wrist is treated as fixed at an anchor point next to the object | Tracked in 6-DoF by the Coil Pro |
| How to scan | Rest the wrist on the table, pivot it and sweep the index finger or fist over the object | Move the scanning hand freely around the object |
| Pick, carry and relocate | Limited, since the wrist anchor is assumed fixed | Full room-scale tracking |
| Robot alignment | Table-plane calibration only (`B`) | Optional 4-point Coil ↔ G1 calibration (`python -m scan_track calibrate`) |

Both modes use the same gestures, catalog matching and G1 plan output. See `scan_track/rokoko_stream.py` and `scan_track/calibration.py` for the exact behaviour, and [`scan_track/README.md`](scan_track/README.md) for setup and the G1 output format.

## Controls

The **left hand** drives the scan so the right (scanning) hand stays steady. Each sign has a keyboard fallback.

| Left hand | Key | Action |
|---|---|---|
| OK-pinch (hold) | `T` | Stream right index fingertip contacts |
| Fist (tap) | `G` | Snapshot all five right fingertips |
| Thumb–pinky pinch (tap) | `C` | Record thumb–index caliper width |
| Peace sign (tap) | `Space` | First tap: lock grasp and carry. Second tap: place, verify, emit plan |
| – | `B` / `R` / `U` | Level table plane / reset / undo |
| – | `O` `H` `V` / `F` / `0` | Toggle object, hull, points / follow hands / reset camera |

## Packages

| Package | What it does |
|---|---|
| [`scan_track/`](scan_track/README.md) | The live system: Rokoko stream, gestures, point cloud and hull, catalog matcher, relocation tracker, offline 3D dashboard, G1 bridge |
| `glove_twin/` | Object cards, object catalog, shape fitting (box / cylinder / sphere), tap-based material classification, MuJoCo hand and robot grasp, visual report |
| [`touchmap/`](touchmap/SPEC.md) | Standalone multi-grasp shape fitter with BIC scoring and an optional Rerun viewer |
| `firmware/tap_glove/` | Teensy sketch for the tap glove (FSR, piezo, microphone) |
| [`PLAN_3D_SCAN_AND_TRACKING.md`](PLAN_3D_SCAN_AND_TRACKING.md) | Design plan for 3D scanning and tracking |

## Simulators

There are two simulators, and neither is finished yet. They are meant to give a close-enough approximation of the real 3D mesh, not an exact reproduction of it.

- **Synthetic hands** (`scan_track`, `--source synthetic`): a scripted pair of hands that scans, carries and places a catalog object. It drives the dashboard demo without hardware.
- **MuJoCo twin** (`glove_twin`): the Rokoko-style hand model and the robot hand that grasp the object from its card. It also checks every placement before a G1 plan is emitted.

Real-glove behaviour (noise, slip, soft fingertips) is not fully modelled, so results from either simulator still need confirming on hardware.

## Supported shapes

The **box, cylinder and sphere** are the shapes we have perfected: fitting, recognition, rendering and grasp planning are tuned for these three. Other shapes are forced into the nearest one.

## Object catalog

| Object | Shape | Size | Mass |
|---|---|---|---|
| Red Bull 250 ml can | cylinder | Ø 53 × 133 mm | ≈ 270 g |
| Kantolan Suola Cracker box (lying down) | box | 195 × 66 × 45 mm | 153 g |
| Wooden sphere | sphere | r = 50 mm | ≈ 323 g |

Defined in `glove_twin/objects.py` with sources. Add your own with `--catalog extra.json` (measure with calipers and a scale).

## Outputs

- **Object card** (`glove_twin/object_card.py`): shape and dimensions with uncertainty, mass, material, friction, max safe grip force, grasp type and planned grip force.
- **Session folder** (`sessions/<id>/`): scan data and `unitree_g1_pick.json`, which holds the object pose in the G1 pelvis frame, the MuJoCo check result, four wrist waypoints and Dex3-1 / Inspire hand commands.

## Other tools

```bash
python -m glove_twin map --object redbull --simulate --robot   # shape from simulated touch → card → robot grasp
python touchmap/touchmap_grasp.py --bench 10                   # shape-fitter benchmark
python -m glove_twin demo                                      # tap sensing: material → density → MuJoCo twin
python -m glove_twin scan --object mug --taps 5 --port COM5    # tap a real object with the tap glove
python -m scan_track bench                                     # headless scan and relocation benchmark
```

## How shape recognition works

1. Fingertip contacts are accumulated into a world-frame point cloud (convex hull and oriented box computed live).
2. Box, cylinder and sphere templates are fitted with free pose and size, using a robust Cauchy error and a BIC penalty.
3. The result is matched to the catalog. At ≥ 90 % confidence the object locks and its card is built.
4. On pick-up, the object's pose follows the hand. On placement it snaps flat to the table and is refined against the finger contacts.

## Limits

- **Simulation only:** Both simulators are unfinished, and everything has been verified in them. Real-glove accuracy still needs measuring.
- **No Coil Pro:** The hand's position in the room is unknown, so scanning relies on a planted wrist. Carrying and relocating an object and aligning to the robot are limited. With a Coil Pro you get free scanning and full pick-and-place tracking, but it needs care: keep it on a non-metallic table and 40–50 cm from robot motors, power bricks and laptops, or its electromagnetic field distorts (see the plan doc, section 2.2).
- **Three shapes only:** Anything that isn't a box, cylinder or sphere is matched to the closest of the three.
- The gloves cannot sense force, so mass and friction come from the catalog, not the scan.
- Thumb–index caliper reach is about 169 mm, so longer objects (the 195 mm cracker box) cannot be measured directly.
- Tap-based material priors are order-of-magnitude estimates until calibrated on hardware.
