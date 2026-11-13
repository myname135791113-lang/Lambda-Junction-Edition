# touchmap_grasp: specification

Recognise the shape of a held object (box, cylinder or sphere) from fingertip contact points captured
over several grasps, and show the fit live in 3D.

## 1. Purpose and scope

A person grasps an object several times. Each grasp records 5 fingertip positions. All points go into one
world-frame point cloud. After every grasp the tool fits three parametric templates, scores them, and
reports a probability per shape, the fitted dimensions, and advice on where to grasp next.

In scope: shape class (3 templates), pose and size estimation, live visualisation, session logging.
Out of scope: material, mass or density (see `glove_twin/`), arbitrary meshes, moving objects.

## 2. Requirements

| ID | Requirement |
|---|---|
| R1 | Accept grasps from a synthetic generator or a live Rokoko Studio stream |
| R2 | A grasp is 5 points in mm, ordered thumb, index, middle, ring, pinky |
| R3 | Accumulate all points in one world frame. The object must not move between grasps |
| R4 | After each grasp, fit box, cylinder and sphere with free position, orientation and size |
| R5 | Output a probability per template and a label only when the top probability is at least 0.9 |
| R6 | Tolerate outliers (a finger that did not touch) without them deciding the result |
| R7 | Penalise model complexity so a box is not chosen merely because it has more parameters |
| R8 | Advise which grasp would best separate the two leading candidates |
| R9 | Persist every session to disk, with atomic writes of `result.json` |
| R10 | Run with no hardware and no viewer (`--source synthetic --no-view`) |

## 3. Inputs

### 3.1 Synthetic (`--source synthetic`, default)
A hidden object is placed on a table (centre x 300-500 mm, y within ±150 mm). Shapes and size ranges:

| Shape | Size ranges (mm) |
|---|---|
| box | 40-150 x 40-120 x 40-200 |
| cylinder | radius 25-60, height 60-200 |
| sphere | radius 30-70 |

Grasps are either a wrap grasp (thumb opposite the fingers, 75%) or a top-rim grasp (25%).
Per-point noise is 2 mm. Each finger misses with 10% probability and hovers 5-20 mm off the surface.
`--tilted` randomises orientation, `--shape` fixes the shape, and `--seed` makes runs repeatable.
`--phalanges` (benchmark only) adds middle-phalanx contacts that miss 30% of the time.

### 3.2 Rokoko (`--source rokoko`)
Rokoko Studio Custom Streaming over UDP, JSON v4, port `--port` (default 14043). LZ4-compressed frames
are detected by their magic number and need the `lz4` package. Fingertips are found by joint names that
contain the hand side (`--hand`, default right) and `tip`, matched loosely per finger. Use `--dump` once
to print the first packet and joint names, and adjust `FINGER_KEYS` if the names differ.
Coordinates are converted from left-handed Y-up metres to right-handed Z-up mm: `(x, y, z) -> 1000·(x, z, y)`.

### 3.3 Triggers
Enter in the terminal, `--auto N` (every N seconds), or `--serial PORT` (an Arduino that prints `B` at
115200 baud on a button press).

## 4. Method

**Templates.** Each exposes a signed-distance residual per point, starting guesses, bounds and a mesh.

| Template | Parameters | Free count |
|---|---|---|
| sphere | centre (3), radius | 4 |
| cylinder | centre (3), axis angles (2), radius, height | 7 |
| box | centre (3), rotation vector (3), dimensions (3) | 9 |

`--upright` fixes the cylinder axis to world z, and lets a box rotate only about z (yaw), which lowers the
free counts and reduces the grasps needed.

**Fitting.** `scipy.optimize.least_squares` with a Cauchy loss (`f_scale = SIGMA`, 3 mm), at most 300
evaluations per start. Several starts per template are tried and the lowest cost is kept: algebraic
sphere and circle fits, PCA principal axes (each as the up axis, with a few spins for the box), and world z.

**Model choice.**
- Deviance = Σ log(1 + (r/σ)²), a heavy-tailed likelihood where outliers add only logarithmically.
- BIC = deviance + k·ln(n), with k the free parameter count and n the number of points.
- Probability = softmax(−BIC/2) across the three templates.
- Label = the top template if its probability is at least 0.9, otherwise none.
- A template is fitted only when n ≥ k + 1. Until all three can be fitted no probabilities are given.

**Advice when undecided** (top two candidates):
- sphere vs cylinder: grasp much higher or lower than before
- box vs cylinder: grasp from a direction rotated about 45° (across an edge)
- box vs sphere: spread the fingers wider across one face

## 5. Outputs

Written to `sessions/<YYYYMMDD_HHMMSS>_<source>/`:

| File | Content |
|---|---|
| `grasps.jsonl` | One line per grasp: index, timestamp, finger order, points in mm |
| `result.json` | Counts, label, best guess, advice, probabilities, per-model parameters, RMS error, BIC. Written atomically. Includes `ground_truth` for synthetic runs |
| `cloud.ply` | All points, ASCII PLY |
| `view.rrd` | Rerun recording (only with `--save-rrd`) |

**Live view (Rerun).** A 3D panel shows the cloud (colour by grasp number, latest grasp enlarged and
joined by lines) and the three fitted meshes (best fit most opaque). Beside it are a probability time
series per grasp and a markdown summary. Coordinates are right-handed Z-up.

## 6. Command line

```
python touchmap_grasp.py [--source synthetic|rokoko] [--shape box|cylinder|sphere] [--tilted]
        [--seed N] [--upright] [--port 14043] [--hand right] [--dump] [--serial PORT]
        [--auto SECONDS] [--max-grasps N] [--out sessions] [--no-view] [--rrd FILE] [--save-rrd]
        [--bench TRIALS] [--bench-grasps 6] [--phalanges]
```

Interactive commands: Enter = grasp, `u` = undo last, `r` = reset, `q` = quit.
`--bench N` runs N trials per shape and prints accuracy against the number of grasps, how often a
confident label is right and wrong, a confusion matrix, and radius error.

## 7. Dependencies

Python 3.10+, numpy, scipy, rerun-sdk. Optional: pyserial (`--serial`), lz4 (compressed Rokoko frames).
Tested on Windows 11 with Python 3.12, numpy 2.5, scipy 1.18, rerun-sdk 0.43.2.

## 8. Performance

A fit after each grasp took roughly 0.4-2.7 s on the dev machine (3 templates, up to 40 points, default
free orientation), and grows with the point count. `--upright` is faster.

## 9. Known limits

- **Early confidence can be wrong.** In one synthetic run (hidden cylinder) the tool was 91% sure of a
  sphere after 3 grasps, then settled on cylinder (98%) by grasp 8. Do not trust a label before about 5
  varied grasps.
- Sphere and cylinder look alike unless grasps differ in height. Follow the advice line.
- Only three shapes. Anything else is forced into the nearest one.
- The object must stay fixed in the world frame. A moved object corrupts the cloud (use `r` to reset).
- Fingertip points are used as surface points. Pad offset is covered only by the 3 mm error model.
- Rokoko joint naming is matched heuristically and may need `FINGER_KEYS` changes.
- The Rokoko path and the serial button path have not been tested against hardware.
- `SIGMA` (3 mm) is a fixed assumption, not calibrated to a real sensor.
