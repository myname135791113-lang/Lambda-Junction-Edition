# touchmap_grasp: specification

Recognise the shape of a held object (box, cylinder or sphere) from fingertip contact points captured
over several grasps, and show the fit live in 3D.

## 1. Purpose and scope

A person grasps an object several times. Each grasp records 5 fingertip positions. All points go into one
world-frame point cloud. After every grasp the tool fits three parametric templates, scores them, and
reports a probability per shape, the fitted dimensions, and advice on where to grasp next.

In scope: shape class (3 templates), pose and size estimation, live visualisation, session logging.
Out of scope: material, mass or density (see `glove_twin/`), arbitrary meshes (see `scan_track/`).

## 2. Requirements

| ID | Requirement |
|---|---|
| R1 | Accept grasps from a synthetic generator or a live Rokoko Studio stream |
| R2 | A grasp is 5 points in mm, ordered thumb, index, middle, ring, pinky |
| R3 | Accumulate all points in one world frame. The object must not move between grasps |
| R4 | After each grasp, fit box, cylinder and sphere with free position, orientation and size |
| R5 | Output a probability per template and a label only when the top probability is at least 0.9 |
| R6 | Tolerate outliers (a finger that did not touch) and drop non-finite coordinates safely |
| R7 | Penalise model complexity with adaptive noise scale so real-sensor jitter does not cause false certainty |
| R8 | Advise which grasp would best separate the two leading candidates |
| R9 | Persist every session to disk without directory collisions, with atomic writes of `grasps.jsonl`, `result.json`, and `cloud.ply` |
| R10 | Run with no hardware and no viewer (`--source synthetic --no-view`) |
| R11 | Detect stale Rokoko streams, malformed UDP packets, and bad serial ports with explicit warnings |

## 3. Inputs

### 3.1 Synthetic (`--source synthetic`, default)
A hidden object is placed on a table (centre x 300-500 mm, y within ±150 mm). Shapes and size ranges:

| Shape | Size ranges (mm) |
|---|---|
| box | 40-150 x 40-120 x 40-200 |
| cylinder | radius 25-60, height 60-200 |
| sphere | radius 30-70 |

### 3.2 Rokoko (`--source rokoko`)
Rokoko Studio Custom Streaming over UDP, JSON v4, port `--port` (default 14043), with frame freshness
checking (`--max-age`, default 1.0 s). LZ4-compressed frames are detected by magic bytes and need `lz4`.
Joint names are tokenized across camelCase and snake_case (`rightIndexTip`, `right_index_tip`, etc.)
without ambiguous bare-digit matching. Use `--dump` once to inspect incoming packet joint names.

### 3.3 Triggers
Enter in the terminal, `--auto N` (every N seconds), or `--serial PORT` (an Arduino that prints `B` at
115200 baud on a button press).

## 4. Outputs

Written atomically to a unique directory `sessions/<YYYYMMDD_HHMMSS_ffffff>_<source>/`:
- `grasps.jsonl`
- `result.json`
- `cloud.ply`
- optional `view.rrd`
