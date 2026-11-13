# Lambda - Junction Edition

**Map an object by touch with a Rokoko glove → object card → a robot hand grabs the same kind of object.**

A person wearing the glove handles an object for 30–60 seconds. From touch alone the system works out
its **shape** (box, cylinder or sphere) and **dimensions**, and from light taps, a lift and a slide its
**material, mass and friction**. It writes an **object card** (JSON) that tells a robot hand how to
grasp that kind of object, and how hard to squeeze.

```bash
pip install -r requirements.txt
python -m glove_twin map --object redbull --simulate --robot   # one object, card + robot grasp
python -m glove_twin.visual                                    # full visual report, opens in the browser
python -m unittest discover -s tests -v                        # all tests (~5 min)
```

## Touch mapping: box, cylinder, sphere

### Model objects (ground truth, with sources)

| Object | Shape and size | Mass | Sources |
|---|---|---|---|
| Red Bull 250 ml can, full | cylinder, Ø 53 mm, H 133 mm | ≈ 270 g (estimate: 250 ml × 1.04 g/ml + ~10 g can; weigh it) | [Crown 250 ml slim](https://crowncork.com/beverage-packaging/products/beverage-cans/83oz-250ml-slim), [Ball design guide](https://printguides.ball.com/whats-available/Europe), [redbull.com](https://www.redbull.com/int-en/energydrink/red-bull-energy-drink) |
| Valio PROfeel 250 ml carton, full | box, 47.8 × 47.5 × 140 mm, sloped "Edge" top | 265.8 g | [Prisma.fi listing](https://www.prisma.fi/tuotteet/111163612/valio-profeel-proteiinijuoma-salted-caramel-250-ml-uht-laktoositon-111163612), [Valio + Tetra Brik Aseptic Edge](https://www.packaginginsights.com/news/valio-revitalises-dairy-products-with-tetra-brik-aseptic-edge.html) |
| Wooden sphere | sphere, r = 50 mm (V = 523.6 cm³) | ≈ 323 g: average of beech, paper birch and red pine | [USDA Wood Handbook, ch. 5](https://research.fs.usda.gov/treesearch/37427) (SG at 12 % moisture × 1.12) |

All values live in `glove_twin/objects.py`. The mapping pipeline never reads them; they are only used
to drive the simulator and score accuracy.

### What the glove can and can't do (Rokoko Smartgloves II datasheet)

- Finger positions **relative to the back of the hand**: 1.23 mm noise, **1.60 mm accuracy** (typical).
  Output rate is **83.33 Hz**. ([datasheet](https://sdk.rokoko.com/datasheets/Smartgloves_II_Technical_Specifications_v1.0.0_External.pdf))
- **No force sensing.** That's why cheap FSR pressure pads are added on the finger segments and palm.
- **No Coil Pro**, so the glove doesn't know where the hand is in the room between grasps.

### Gloves-only method

1. **Routine:** two side enclosures, a top grasp (palm on the top), and thumb-to-index **caliper pinches**
   across width, depth and height. This follows the exploratory procedures in
   [Lederman & Klatzky 1987](https://web.eecs.umich.edu/~stellayu/teach/2023action/papers/1987klatzkyHandMovements.pdf).
2. **Contacts:** each pressed pad gives a point (pad centre + pad radius along the palmar axis) and an
   outward normal (the segment's dorsal axis), computed from joint angles only (`touch.py`).
3. **Joint fit:** an object's size doesn't depend on where the hand is. So one Levenberg–Marquardt
   fit estimates the shape parameters plus an unknown pose for every grasp (`shape_fit.py`, numpy
   only). A pinch measures a width directly, because both contacts are in the same hand frame.
4. **Shape choice:** BIC over box / cylinder / sphere. A margin below 6 counts as ambiguous.
5. **Follow-up:** if the result is ambiguous or a dimension wasn't measured, the person is asked for one
   more enclosure and new pinches (up to two rounds). Unmeasured dimensions are reported as
   "not measured", never as a confident number.
6. **Calibration:** one session on the can sets the pad offsets (`calibrate_pad_offset`). Enclosures
   and pinches get separate offsets.

### Simulated accuracy

Ten sessions per object, each with a new hand size (±2 %), 1° joint noise and follow-ups enabled:
**30/30 shapes correct**.

- **Can:** radius within 5.6 mm, height within 10.3 mm.
- **Carton:** width and depth within 5 mm. One session's height pinch couldn't be done at that hand
  size, and was flagged rather than guessed.
- **Sphere:** within 2 mm.

The simulator's hand closes kinematically. The real glove has to confirm these numbers.

### Object card → robot hand

`object_card.py` writes, per object:
- primitive, dimensions ± uncertainty, BIC weights
- mass (from the lift test, never from the material), material (taps), friction (slide test)
- max safe squeeze, grasp type (power wrap / box wrap / spherical) and grip force

`robot_hand.py` grasps the real object with a multi-finger robot hand (Rokoko geometry, torque-driven
fingers, force-regulated squeeze). With the card's force it lifts all three objects. At 0.3× it drops
them, and above the safe limit it crushes the can and the carton.

### Real hardware

Record frames as JSON (`touch.save_frames`: joint angles from the Rokoko solver plus FSR readings per
grasp), then:

```bash
python -m glove_twin map --frames session.json --mass-g 270 --material filled_can \
    --offset-enclosure-mm 2.0 --offset-caliper-mm 2.8
```

A live Rokoko RGMP stream reader is the next step: TCP 12277 from `rkk-hand-solver`, per the SDK docs.

## Tap sensing: material and density

A person wearing the sensor glove taps an object lightly a few times. The glove works out what the
object is made of, estimates its density, friction and fragility, and writes a MuJoCo twin.

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
python -m unittest discover -s tests -v   # all tests, incl. the Rokoko hand and robot sims
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
| `glove_twin/visual.py`, `visual_mapping.py`, `report.py` | Visual test runner and HTML report |
| `glove_twin/assets/fonts/` | Self-hosted report fonts (Schibsted Grotesk, IBM Plex Mono; SIL OFL 1.1, licences alongside) |
| `glove_twin/objects.py` | Sourced catalog of the model objects (ground truth) |
| `glove_twin/shape_fit.py` | Gloves-only joint fit of box / cylinder / sphere, BIC model choice |
| `glove_twin/touch.py` | Glove frames (joint angles + FSR pads) to contact points and normals |
| `glove_twin/explore_sim.py` | Simulated exploration routine (kinematic closing, pinches, follow-ups) |
| `glove_twin/mapping.py` | Session orchestration, calibration, physical tests, card building |
| `glove_twin/object_card.py` | Object card schema, grasp choice, grip force |
| `glove_twin/robot_hand.py` | Multi-finger robot hand grasp from the card |
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
