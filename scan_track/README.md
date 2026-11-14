# `scan_track` — Rokoko Smartgloves II (+ Coil Pro) 3D Cloud, Convex Hull & Unitree G1 Tracker

Ready-to-test package implementing [PLAN_3D_SCAN_AND_TRACKING.md](../PLAN_3D_SCAN_AND_TRACKING.md):
- Works **immediately with zero hardware** (`--source synthetic`)
- Works **right now with Smartgloves II alone** (`--source rokoko --mode planted` + caliper pinches + left-hand gestures)
- Upgrades **with one flag when Coil Pro arrives** (`--source rokoko --mode coil`)
- Verifies grasps in the **MuJoCo Digital Twin** before emitting **Unitree G1 Humanoid** (`Dex3-1` / `Inspire` + 7-DoF arm) pick trajectories.

---

## 1. Quickstart Commands

```bash
pip install -r requirements.txt
# Optional if Rokoko Studio streams LZ4-compressed JSON:
# pip install lz4 scipy
```

### A. Run the Full Animated Demo Right Now (No Hardware Required)
```bash
python -m scan_track live --source synthetic
```
Opens `http://127.0.0.1:8765` (100% offline 3D viewer—works even without internet Wi-Fi) showing both hands scanning the object, building the translucent 3D Convex Hull, matching the catalog, carrying the object to a new spot on the table, verifying in MuJoCo, and generating the Unitree G1 pick command.

### B. Test Real Rokoko Smartgloves II *Right Now* (Without Coil Pro)
1. In **Rokoko Studio**, pair your Smartgloves II and enable **Custom Streaming**:
   - Protocol: **UDP**, IP: `127.0.0.1`, Port: `14043`, Format: **JSON v4** (uncompressed JSON recommended, or LZ4 if `lz4` is installed).
2. Run a 10-second stream + gesture sanity check in your terminal:
   ```bash
   python -m scan_track dump --port 14043
   ```
3. Launch the live 3D dashboard in **Planted-Wrist Mode**:
   ```bash
   python -m scan_track live --source rokoko --mode planted --port 14043
   ```
   - **How to scan without Coil Pro:** Place the object on the flat table, rest your right wrist heel on one fixed spot right next to the object, press **`[B]`** (with palm flat) once to level the table plane, then pivot your wrist & sweep your right index finger over the object while holding a **Left-Hand OK-Pinch (`👌`)**!

### C. Upgrade to Full Room-Scale Pick & Relocate (Once Coil Pro Arrives)
1. *(Optional 30-second robot alignment)* Touch 4 tape X marks on the table to align Coil Pro coordinates with the Unitree G1 pelvis base frame:
   ```bash
   python -m scan_track calibrate --port 14043 --points 4 --warp --output calibration.json
   ```
2. Launch in **Coil Pro 6-DoF Mode** (and optionally stream pick JSON straight to the Unitree G1 controller over UDP):
   ```bash
   python -m scan_track live --source rokoko --mode coil --port 14043 --calibration calibration.json --g1-udp 192.168.123.164:15000
   ```

---

## 2. Left-Hand Gesture & Keyboard Controls

| Left-Hand Sign | Keyboard / UI Fallback | Action |
|---|---|---|
| **🖐️ Open Hand** | — | Idle / move scanning hand freely without recording |
| **👌 OK-Pinch** *(thumb+index touching, outer 3 fingers straight — hold)* | **`T`** | Stream **Right Index finger-pad** contacts continuously (83 Hz) |
| **✊ Closed Fist** *(all 4 fingers curled — tap)* | **`G`** | Capture **all 5 Right finger-pad** points at once |
| **🤏 Thumb–Pinky Pinch** *(tap)* | **`C`** | Record exact Right Thumb–Index **caliper width** |
| **✌️ Peace Sign** *(index+middle straight, ring+pinky curled — tap)* | **`Space` / `P`** | **1st tap:** `LOCK_GRASP` (pick up & move object)<br>**2nd tap:** `PLACED` (snap to table, verify in MuJoCo, emit Unitree G1 pick plan) |
| — | **`B`** | Calibrate flat table plane (`z_table = 0`) from current right hand |
| — | **`R`** | Reset scan for next object |
| — | **Right-Drag / `Shift`+Drag / `WASD` / Arrows** | Move 3D visual map across the horizontal **X–Z table plane** |
| — | **`Alt`+Drag / `Q` & `E`** | Adjust vertical view height |
| — | **`F`** *(or **Follow** toggle)* | Center 3D camera onto active object / live gloved hands (handles any Coil Pro offset) |
| — | **`O`** *(or **Object: Visible / See-Through**)* | Toggle predefined 3D object between solid and see-through (to inspect the 3D Convex Hull) |
| — | **`H` / `V` / `0`** | Cycle Convex Hull visibility / toggle tactile points / reset camera view |

---

## 3. Unitree G1 Humanoid Output Format

Every completed placement writes `sessions/<id>/unitree_g1_pick.json` (and optionally UDP-broadcasts it via `--g1-udp`):
- **`object_pose_g1_base`**: `[x, y, z_table]` in metres + `yaw_rad` in the **Unitree G1 pelvis frame** (`+X` forward, `+Y` left, `+Z` up).
- **`mujoco_twin_gate`**: Digital twin grasp check (`outcome == "lifted"`, `peak_squeeze_n`, `safe_to_execute`).
- **`waypoints`**: 4-stage wrist pose trajectory (`1_PRE_GRASP` → `2_DESCEND_TO_GRASP` → `3_FORCE_CLOSE` → `4_LIFT_OBJECT`) with `wrist_pos_m` and `wrist_quat_wxyz`.
- **`dex3_1_hand`**: Joint positions `q_open_rad`, `q_grasp_rad`, gains `kp`/`kd`, and force-scaled feedforward torques `tau_ff_nm` for `rt/dex3/{right|left}/cmd` (`thumb_0..2`, `middle_0..1`, `index_0..1`).
- **`inspire_hand`**: Alternative 6-DoF closure & force thresholds if your G1 uses 5-finger Inspire hands.

---

## 4. Headless Benchmark & Tests

```bash
python -m scan_track bench
python -m unittest discover -s tests -p "test_scan_track.py" -v
```
