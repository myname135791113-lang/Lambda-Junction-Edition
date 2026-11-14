# Plan: Coil Pro 3D Tactile Scanning, Catalog Matching & Relocation Tracking

_Architecture & implementation plan for the **Rokoko Smartgloves II + Coil Pro + Sim/Real Robot** pipeline (no FSR force pads required)._

---

## 1. Overview & Motivation

### Why We Adjusted the Scope
With **Rokoko Smartgloves II + Coil Pro** alone (without calibrated fingertip force/pressure sensors), we cannot directly weigh an object from hand pose. However, Smartgloves + Coil Pro give **metric 6-DoF hand & finger tracking at 83.33 Hz** in a shared 3D frame.

Instead of measuring weight directly on the glove, we close the physics loop via **3D tactile reconstruction + catalog retrieval**:

```text
[1. Tactile Scan on Table]  -->  Live 3D Point Cloud + Convex Hull (Visual Demo)
            |
            v
[2. Geometric Matching]     -->  Match (H, L, W, Hull Volume, Primitive Fit) to Known Catalog
            |                    Unlocks Object Card: mass, friction, fragility, grasp strategy
            v
[3. Free Pick & Relocate]   -->  User grabs object freely & sets it anywhere on the flat table
            |                    Single-grasp rigid transform + table-plane snap tracks new pose
            v
[4. Sim -> Real Execution]  -->  MuJoCo Digital Twin verifies grasp at new (x, y, yaw)
                                 -> Physical robot arm executes the verified pick
```

---

## 2. Hardware Setup & Coordinate Calibration

### 2.1 Hardware in the Loop
- **Both Rokoko Smartgloves II** (Right = scanning/grasping hand, Left = gesture trigger hand; ~1.6 mm local finger accuracy, 83.33 Hz).
- **Rokoko Coil Pro** (electromagnetic 6-DoF wrist position + orientation in world space).
- **Flat Workspace Table** (calibrated plane $z = z_{\text{table}}$).
- **MuJoCo Digital Twin** + **Physical Robot Arm**.

### 2.2 Coil Pro Placement Rules (Critical — Test First!)
Coil Pro uses an electromagnetic field that bends near ferromagnetic metal and motors:
1. **Non-metallic table:** Use a wooden or plastic table (check underneath for steel support beams).
2. **Keep distance from robot motors:** Place the Coil Pro **40–50+ cm away** from the physical robot base, servo motors, power bricks, and laptops.
3. **Workspace layout:** Put the shared **Scan & Drop-Off Zone** cleanly *between* the Coil Pro and the robot's reach envelope so neither the coil nor the robot base sits on top of the other.

### 2.3 Aligning Coil Pro Space to Robot Space ($T_{\text{robot}\leftarrow\text{coil}}$)
Coil Pro $(0,0,0)$ is at the coil hub; the robot's $(0,0,0)$ is at its base. They must agree within **~5–10 mm** so the physical gripper lands accurately on the relocated object.

**30-Second 4-Point Table Calibration:**
1. Stick **4–6 tape X marks** on the flat workspace table inside the robot's reachable area.
2. Jog the physical robot gripper tip (TCP) to touch each X mark → record $\mathbf{p}_i^{\text{robot}} \in \mathbb{R}^3$.
3. Touch the exact same X marks in order with the gloved **Right Index fingertip** (firing the left-hand trigger) → record $\mathbf{p}_i^{\text{coil}} \in \mathbb{R}^3$.
4. Solve rigid **Kabsch SVD alignment** to get $(R, \mathbf{t}) = T_{\text{robot}\leftarrow\text{coil}}$, and fit the flat table plane ($z = z_{\text{table}}$, unit normal $\hat{\mathbf{n}}_{\text{table}}$) from the same points.
   - _Optional mild-warp compensation:_ With 4–6 grid points on the table, fit a 2D affine correction in $(x, y)$ if mild magnetic distortion (>8 mm residual) is observed near the robot side of the table.

---

## 3. User Manual & Left-Hand Gesture Protocol

### Why Left-Hand Gestures Work Reliably
Rokoko finger curl angles are local to the glove (zero room-drift) and triggering with the **left hand** avoids shaking or tensing the **right (scanning) hand** at the moment of measurement.

### 3.1 Left-Hand Gesture Vocabulary
Because we do **not** have FSRs to detect which fingers are touching vs. floating in thin air, we separate **1-finger continuous tracing** from **5-finger snapshots**:

| Left-Hand Gesture | Detection Rule (Local Left Hand) | System Action |
|---|---|---|
| **Index–Thumb Pinch (Hold)** | `\|\|thumb_tip - index_tip\|\| < 25 mm` | **Stream Right Index Contact (83 Hz):** Continuously adds *only* the Right Index finger-pad contact point while the user slides/traces along the object. Zero risk of floating-finger outliers; yields hundreds of dense points in seconds. |
| **Closed Fist (Tap / Hold 0.2 s)** | All 4 fingertips curled within `< 45 mm` of palm | **5-Finger Grasp Snapshot:** Records *all 5* Right finger-pad points at once (used when the user intentionally rests all 5 fingertips on the object). |
| **Peace Sign (Index + Middle Extended)** | Index & middle straight, ring & pinky curled (rising edge) | **Phase Toggle (`LOCK_GRASP` → `PLACED`):** First trigger locks the object-to-hand transform right as the user grabs the object to move it; second trigger (right before releasing on the table) locks its final resting pose and hands off to the robot. |

*(Keyboard/CLI fallback keys `Space`, `G`, `P`, `R` must also be wired in parallel for easy debugging and single-glove testing.)*

### 3.2 Step-by-Step User Routine ("User Manual")
1. **Place on flat surface:** Set the object anywhere inside the workspace (use a thin non-slip silicone/rubber mat in the scan spot so light touches don't nudge light objects).
2. **Scan while stationary (~10–15 s):**
   - Keep the object still on the table during scanning (moving/spinning an untracked object mid-scan scrambles the point cloud).
   - Hold **Left Pinch** to trace the top, sides/corners, and **one quick finger sweep around the base seam** where the object meets the table; optionally use **Left Fist** for 5-finger enclosure snapshots from 3–4 angles (~60° apart).
   - Watch the live **3D Point Cloud + Convex Hull + Catalog Match %** lock onto the object on screen.
3. **Pick up freely & relocate:**
   - Grab the scanned object **any way you want** in a **single firm grasp**.
   - Trigger **`LOCK_GRASP`** as you hold it, lift it off the table, and carry it anywhere on the flat workspace *(do not spin/juggle/re-grasp the object in mid-air during transport)*.
4. **Place & hand off to robot:**
   - Set the object flat anywhere on the workspace surface.
   - Trigger **`PLACED`** *right before opening your fingers*, then pull your hands clear.
   - MuJoCo verifies the grasp at the new $(x, y, \theta_{\text{yaw}})$ → physical robot arm picks it up.

---

## 4. Algorithms & Mathematical Pipeline

### 4.1 Finger-Pad Skin Offset (~7 mm)
Rokoko reports skeletal joint/tip centers inside the finger, **not** the outer skin surface. Raw bone tips make reconstructed objects ~14–16 mm too wide on every axis.
- **Fix:** Offset each fingertip point inward along the distal phalanx palmar normal (opposite the fingernail/dorsal axis) by finger-pad radius $r_{\text{pad}} \approx 6.5\text{–}8.0\text{ mm}$ (reuse the palmar normal geometry already in `glove_twin/touch.py`).

### 4.2 Live Point Cloud & Table-Aware Convex Hull
1. **Filtering:** Reject points below $z_{\text{table}} - 3\text{ mm}$; clamp near-table touches within $[z_{\text{table}} - 3\text{ mm}, z_{\text{table}} + 4\text{ mm}]$ to $z_{\text{table}}$; voxel-downsample on a `3 mm` grid so hovering in one spot doesn't bias fits.
2. **The Convex Hull Bottom-Surface Pitfall (and Fix):**
   - **Problem 1 (Floating/chopped bottom):** If the user only touches the top and upper sides of a can or box, a raw `scipy.spatial.ConvexHull` stops mid-air at the lowest side touch instead of reaching the table:
     ```text
     Real can on table:             Raw Convex Hull of upper touches (*):
        +---------+                     *---------*   <-- touched top
        |         |                     |         |
        |         |                     *         *   <-- lowest side touches (4 cm up)
        |         |                                   <-- bottom 4 cm missing!
        +---------+
     =================              =================  table (z = z_table)
     ```
   - **Problem 2 (Why naive vertical projection breaks spheres):** Copying *all* $(x, y)$ points straight down onto $z_{\text{table}}$ gives cans/boxes a clean flat base, **but** turns the bottom half of a sphere into a flat-bottomed cylinder:
     ```text
     Table-projected CAN (Good):    Table-projected BALL (Bad!):
        *---------*                         * * *
        |         |                      *         *  <-- widest equator of ball
        *         *                      |         |  <-- projecting equator straight down
        o---------o  <-- projected       o---------o      turns ball into a CYLINDER!
     =================              =================  table (z = z_table)
     ```
   - **Clean Universal Solution:**
     1. Always inject **1 virtual ground contact point** directly beneath the cloud's horizontal centroid: $(\bar{x}, \bar{y}, z_{\text{table}})$. Every convex resting object (box, upright/side cylinder, or sphere) touches the table under its center, guaranteeing exact height $H = \max(z) - z_{\text{table}}$ without bloating spheres.
     2. Have the user do **one quick index-finger sweep around the base seam** where the object meets the table: on boxes/cans this naturally traces the wide flat base; on spheres the finger naturally tucks inward under the lower curve. Plain `ConvexHull` then produces a faithful 3D mesh for all shapes.

### 4.3 Catalog Matching → Known Object Card
From the accumulated point cloud $\mathcal{P}$ and Convex Hull $\mathcal{H}$:
1. **Metric descriptors:**
   - Height: $H = \max_{p \in \mathcal{P}}(p_z) - z_{\text{table}}$
   - Oriented 2D footprint $(L, W)$: minimum-area bounding rectangle of $(p_x, p_y)$ via rotating calipers on the 2D hull ($L \ge W$), plus initial yaw $\theta_0$.
   - Convex hull volume $V_{\text{hull}}$ and surface area $A_{\text{hull}}$.
   - Cauchy-loss primitive fit (`box`, `cylinder`, `sphere`) with BIC model weights $w_{\text{box}}, w_{\text{cyl}}, w_{\text{sph}}$ (reuse `touchmap_grasp.py` / `shape_fit.py`).
2. **Catalog scoring:**
   For each reference object $c \in \text{Catalog}$ (`glove_twin/objects.py`) with ground-truth $(H_c, L_c, W_c, V_c, \text{shape}_c)$:
   $$d^2(c) = \left(\frac{H - H_c}{\sigma_H}\right)^2 + \left(\frac{L - L_c}{\sigma_L}\right)^2 + \left(\frac{W - W_c}{\sigma_W}\right)^2 + \left(\frac{V_{\text{hull}} - V_c}{\sigma_V}\right)^2 - 2 \ln P(\text{shape}_c \mid \mathcal{P})$$
   with $\sigma_H \approx 8\text{ mm}$, $\sigma_{L,W} \approx 10\text{ mm}$, $\sigma_V \approx 0.25 V_c$, and $P(c \mid \mathcal{P}) = \text{softmax}\!\left(-\tfrac{1}{2} d^2(c)\right)$.
3. **Output:** Once top probability $\ge 0.85$ (and $N \ge 15$ distinct voxels), lock the matched `RefObject` and emit a validated `v1` **Object Card** (`glove_twin/object_card.py`) populated with the catalog's known `mass_kg`, `friction`, `max_grip_n`, `material`, and planned `grasp`.

> **Catalog Separation Note (`glove_twin/objects.py`):**
> The three catalog objects — `redbull` (upright cylinder, Ø53 × 133 mm), `kantolan_suola_cracker` (flat box lying down, 195 × 66 × 45 mm), and `wood_sphere` (sphere, Ø100 mm) — are separated by **> 20 mm** in height and footprint, plus distinct `cylinder` / `box` / `sphere` BIC weights.

### 4.4 Relocation Pose Tracking (Pick Up & Place Anywhere on Table)
Let $T_{\text{world}\leftarrow\text{obj}}(t_0)$ be the scanned object's initial table pose (centroid $(\bar{x}, \bar{y}, z_{\text{table}} + H/2)$ and footprint yaw $\theta_0$).

1. **At `LOCK_GRASP` ($t = t_0$):**
   Record the right wrist 6-DoF pose $T_{\text{world}\leftarrow\text{hand}}(t_0)$ from Coil Pro and freeze the rigid hand-to-object offset:
   $$T_{\text{hand}\leftarrow\text{obj}} = \bigl(T_{\text{world}\leftarrow\text{hand}}(t_0)\bigr)^{-1} \, T_{\text{world}\leftarrow\text{obj}}(t_0)$$
2. **During Carry ($t \in [t_0, t_1]$):**
   Stream live object pose in the 3D viewer as it moves with the user's hand:
   $$\hat{T}_{\text{world}\leftarrow\text{obj}}(t) = T_{\text{world}\leftarrow\text{hand}}(t) \, T_{\text{hand}\leftarrow\text{obj}}$$
3. **At `PLACED` ($t = t_1$, right before fingers open on the table):**
   - Start from propagated pose $\tilde{T} = T_{\text{world}\leftarrow\text{hand}}(t_1) \, T_{\text{hand}\leftarrow\text{obj}}$.
   - **Table-Plane Snap ($SE(3) \to SE(2)$ + discrete resting face):**
     - Snap the object's up-axis (or nearest stable face normal if the user turned a box onto its side) flush to $\hat{\mathbf{n}}_{\text{table}}$, zeroing out pitch/roll tilt error.
     - Clamp center height exactly to $z_{\text{final}} = z_{\text{table}} + H_{\text{resting}} / 2$ (eliminates vertical Coil Pro drift completely).
   - **Final Contact-Cage Refinement (anti-slip):**
     - Using the 5 right-hand finger-pad positions at $t_1$, run 10 iterations of 2D $(x, y, \theta_{\text{yaw}})$ least-squares refinement to keep the known catalog mesh centered inside the touching fingers while regularized toward $\tilde{T}$.
     - *(Note: For spheres and upright cylinders, rotational yaw is symmetric anyway—final $(x, y)$ inside the finger cage is accurate to a few millimeters even if minor in-hand rotation occurred.)*

---

## 5. Existing Repo Assets Across Branches (Do Not Reinvent!)

Before writing new code, note what already exists across branches of `Lambda-Junction-Edition`:

| Branch & File | What Already Exists | How to Reuse It |
|---|---|---|
| `origin/touchmap-grasp` → `touchmap/touchmap_grasp.py` | Rokoko Studio UDP v4 receiver (port `14043`, LZ4 decompression, Y-up m → Z-up mm conversion), Cauchy-loss `box`/`cylinder`/`sphere` fitter, BIC scoring, live **Rerun** 3D viewer, `.ply` export | Cherry-pick/port the Rokoko UDP receiver + Rerun hooks + 3D Cauchy primitive fitter directly; extend from 1-hand fingertips to **both hands + wrist 6-DoF + palmar skin offset**. |
| `origin/testing` → `glove_twin/objects.py` | `RefObject` dataclass & `OBJECTS` catalog (`redbull`, `kantolan_suola_cracker`, `wood_sphere`) with dimensions, `mass_kg`, `friction`, `max_grip_n`, `material`, `volume_m3` | Use directly as the **Known Object Catalog** (and add 2–3 well-separated demo objects measured with calipers + kitchen scale). |
| `origin/testing` → `glove_twin/object_card.py` | `build_card()`, `validate_card()`, `choose_grasp()`, JSON schema `v1` | Emit this exact `*.card.json` format (plus a `pose_robot_frame` block with `x, y, z, yaw_rad`) after catalog match + placement! |
| `origin/testing` → `glove_twin/touch.py` | MuJoCo kinematic Rokoko hand model + palmar pad normal/offset math (`pad_contacts`) | Reuse palmar normal offset concept (or compute directly from Rokoko proximal→distal bone vectors in wrist/world frame). |
| `origin/testing` → `glove_twin/robot_sim.py` & `robot_hand.py` | Parallel-jaw & multi-finger MuJoCo grasp simulation with planned grip force & drop/crush verification | Spawn the matched catalog object at its tracked `(x, y, z_table, yaw)` in MuJoCo to verify the grasp before sending waypoints to the real robot arm. |

---

## 6. Proposed Module Structure & Parallel Work Split

To let multiple teammates/agents build in parallel without merge conflicts, add a clean package (e.g., `scan_track/`) alongside `glove_twin/`:

```text
scan_track/
  __init__.py
  rokoko_stream.py      # Dual-hand UDP v4 + LZ4 parser (wrist 6-DoF + 5 tips + palmar normals) + synthetic replay
  calibration.py        # 4-point Kabsch SVD (Coil Pro <-> Robot frame) + table plane z_table fit -> calib.json
  gestures.py           # Left-hand state machine: IDLE | TRACE_INDEX | SNAPSHOT_5 | LOCK_GRASP | PLACED
  cloud_hull.py         # Voxel filter, table-centroid ground point, live scipy ConvexHull, OBB (H, L, W, V)
  catalog_matcher.py    # Score (H, L, W, V, primitive BIC) against glove_twin.objects.OBJECTS -> object_card.json
  relocation.py         # Rigid T_hand_obj lock during carry + SE(2) table snap + finger-cage refinement at drop-off
  viz_live.py           # Live 3D Rerun / web view: raw points, translucent Convex Hull, catalog bars, tracked mesh
  robot_bridge.py       # Step 1: verify pick at (x, y, yaw) in MuJoCo; Step 2: send pick pose + grip force to real arm
```

### Parallel Tracks & Acceptance Tests

1. **Track A — Hardware Stream, Calibration & Gestures (`rokoko_stream.py`, `calibration.py`, `gestures.py`)**
   - *First 30-min hardware go/no-go test:* Slide gloved right index finger along a straight wooden ruler on the table ~50 cm from Coil Pro. Check that plotted points form a straight line with `< 8 mm` bow and `< 5 mm` Z-height drift across the workspace.
   - Verify left-hand `Pinch`, `Fist`, and `Peace` gestures trigger cleanly with zero false positives while moving the left arm.

2. **Track B — Live Cloud, Convex Hull & Catalog Matcher (`cloud_hull.py`, `catalog_matcher.py`, `viz_live.py`)**
   - Build a `--synthetic` generator first (simulates tracing + grasping noisy catalog objects on $z = 0$, plus moving them to a random new $(x, y, \theta)$) so everything is testable offline without hardware.
   - Verify live `ConvexHull` updates at $\ge 30\text{ Hz}$ and top-1 catalog accuracy hits $\ge 95\%$ on synthetic & real scans.

3. **Track C — Pick-and-Place Pose Tracker (`relocation.py`)**
   - Test picking up a scanned box/can/sphere from spot A, placing it 30 cm away rotated by ~45° at spot B, and measuring physical ruler error between true spot B and predicted $(x, y, \theta_{\text{yaw}})$ in `F_robot`. Target: **`< 10 mm` $(x, y)$ error** (well inside gripper capture margin).

4. **Track D — Sim-to-Real Robot Execution (`robot_bridge.py`)**
   - Consume `{card, final_pose_robot}` → run headless/visual MuJoCo grasp check (`glove_twin/robot_sim.py`) → on `lifted == True`, execute pre-grasp $(x, y, z + 10\text{ cm}, \theta_{\text{yaw}})$ → descend → close with card's `grip_force_n` → lift on the physical robot arm.
