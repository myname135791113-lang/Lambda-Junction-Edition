# Lambda - Junction Edition

**Glove tap sensing → material and density → MuJoCo digital twin.**

A person wearing the sensor glove taps an object lightly a few times. The glove works out what the
object is made of, estimates its density, friction and fragility, and writes a MuJoCo twin. A robot
can then plan on that twin and handle the same real object without the glove (real-to-sim-to-real).

## What a tap can and can't tell you

Tapping identifies the **material class**. Steel rings for a long time, glass rings sharp and bright,
wood thuds, and foam is dead. Each class carries a typical density, friction and fragility.

Tapping does **not** measure density directly: a hollow ceramic mug and a solid ceramic block sound
alike. For true bulk density, add the mass (FSR lift or a kitchen scale) and the volume (phone scan).
The tool then reports **solidity** (bulk density ÷ material density), which separates solid objects
from shells and containers.

| Signal | Sensor | Feature | Why it helps |
|---|---|---|---|
| Contact duration | FSR | `contact_ms` | Hard objects bounce the finger back fast |
| Impact sharpness | Fingertip piezo | `impact_centroid_hz`, `amp_per_n` | Harder means a sharper transient |
| Ring damping | Microphone | `q_factor` | Mostly a material property, independent of size: the strongest clue |
| Ring pitch | Microphone | `ring_hz` | Depends on size too, so it's weighted low |

**Why a microphone:** a fingertip piezo only senses while the finger touches the object, about 1 ms
for a light tap. The ringing that best identifies the material continues after the finger has left,
so it needs to be heard (the way people listen to a tap).

In simulation, a fingernail tap or a microphone is enough to identify all 8 material classes. A soft
finger-pad tap with no microphone gets only about half. **Tap with a fingernail or a hard thimble.**

## Quick start (no hardware)

```bash
pip install -r requirements.txt
python -m glove_twin demo                 # classify all materials and build a mug twin
python -m unittest discover -s tests -v   # 23 tests, incl. the Rokoko hand and robot grasp sims
```

## Visual test: Rokoko hand → twin → robot grasp

```bash
python -m glove_twin.visual            # ~15 s, writes reports/visual/index.html and opens it
python -m glove_twin.visual --viewer   # also replays every grasp live in the MuJoCo viewer
```

1. **A simulated Rokoko-glove hand taps** 8 reference blocks to calibrate, then taps 4 test objects
   (steel cube, glass vase, cardboard box, wood block) with its index finger. MuJoCo computes the
   fingertip contact force; the FSR, piezo and microphone readings are synthesised from it.
2. Each tap goes through the real pipeline: features → classifier → density → twin files.
3. **A robot gripper grasps each "real" object twice**, planning its grip force from
   A) default physics (density 1000, friction 1, as in RialTo) and B) the glove twin.

The report shows the tap animations (40× slow motion), the force, microphone and Rokoko traces,
material probabilities, and side-by-side grasp animations with grip-force and height charts.
Typical result: **glove twin 4/4 safe lifts, default physics 1/4** (the steel cube is dropped; the
vase and the box are crushed).

### The Rokoko hand model (`glove_twin/hand.py`)

Built from the [Rokoko Device SDK](https://sdk.rokoko.com) docs (hand-solving guide and Smartgloves
reference):

- **Skeleton:** forearm → wrist → five fingers. The thumb has metacarpal, proximal and distal
  joints; the other fingers add a medial joint. Bodies use the 26 OpenXR hand-joint names that the
  SDK solver emits with `--emit-openxr` (OpenXR "intermediate" = Rokoko "medial").
- **Sensors** (as sites named after the SDK sensor frames): `index_finger_sensor` and the other
  finger sensors on the back of each middle segment, `thumb_finger_sensor` on the thumb's proximal
  segment, `hub` on the back of the hand, and `arm_sensor` on the forearm.
- **What the glove can't do:** it streams pose at **83.33 Hz** and has **no force sensing**. A tap
  lasts under 1 ms, which falls between two 12 ms frames. That's why this project adds a fingertip
  FSR + piezo and a wrist microphone; the report shows the Rokoko frames missing the contact.
- The SDK sends exact bone lengths at runtime (`static_data`). The model uses typical size-M
  lengths until those are wired in.

## With the glove

1. Flash `firmware/tap_glove/tap_glove.ino` (wiring is in the header comment). A Teensy 4.x is
   recommended.
2. **Calibrate on your own taps.** Tap objects of known material; 8–10 taps each, 3+ materials:
   ```bash
   python -m glove_twin calibrate --material wood --taps 10 --port COM5
   python -m glove_twin calibrate --material steel --taps 10 --port COM5
   python -m glove_twin calibrate --material glass --taps 10 --port COM5
   ```
   Taps are saved to `calibration.json`. Until you calibrate, a prior built from the simulator is
   used. It's only a placeholder for real hardware.
3. **Scan an object:**
   ```bash
   python -m glove_twin scan --object mug --taps 5 --port COM5 --mass-g 350 --dims 0.09 0.09 0.10
   ```
   Add `--mesh mug.obj` to use the phone-scan mesh, and `--volume-cm3` for a true volume.
   Add `--no-mic` if the glove has no microphone.

Use `--simulate MATERIAL` instead of `--port` to try any command without hardware.

## Output (`twins/`)

- `<name>.properties.json`: material probabilities, density with range and source, solidity,
  friction prior, max safe grip force, median tap features
- `<name>.glove.xml`: MuJoCo twin with glove-measured physics
- `<name>.default.xml`: same geometry with MuJoCo default physics (density 1000, friction 1), the
  RialTo baseline for the A/B test (crush and drop rates)

## Code map

| File | Role |
|---|---|
| `glove_twin/tap.py` | Tap data structure, serial packet parsing, FSR → newtons |
| `glove_twin/features.py` | Signal processing: contact time, impact spectrum, ring frequency and Q |
| `glove_twin/classifier.py` | k-NN material classifier, trained on your calibration taps |
| `glove_twin/density.py` | Material-prior density, mass/volume density, solidity |
| `glove_twin/twin.py` | Properties JSON and MuJoCo MJCF writer |
| `glove_twin/simulate.py` | Synthetic taps / sensor models for testing without hardware |
| `glove_twin/hand.py` | MuJoCo Rokoko-glove hand (SDK skeleton + sensor layout) |
| `glove_twin/robot_sim.py` | Hand tap simulation and robot gripper A/B grasp |
| `glove_twin/visual.py`, `report.py` | Visual test runner and HTML report |
| `glove_twin/materials.py` | Material priors (density, friction, fragility, tap response) |
| `firmware/tap_glove/` | Microcontroller sketch: triggered capture, JSON over serial |

## Known limits

- All material priors are rough, order-of-magnitude values. Calibration replaces the tap-response
  priors, but not density, friction or fragility.
- The simulator is a simplified model. Accuracy on simulated taps shows the pipeline is consistent,
  not how well it will do on real objects. Measure that on hardware.
- FSRs lag and under-read short taps (hysteresis, drift). Consistent calibration matters more than
  absolute newtons.
- Objects outside the 8 classes are forced into the nearest one. Add classes in `materials.py` and
  calibrate them.
- Twins go stale when the object changes (a cup is filled, fruit ripens).
