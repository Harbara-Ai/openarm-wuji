# OpenArm + Wuji coordinated demonstrations

## Purpose

The coordinated recorder runs the existing scripted task unchanged and records one
unified embodiment for later behavior cloning:

```text
observation.images.front_t
observation.images.wrist_t
[7 arm actual q, 20 hand actual q]_t
    -> [7 arm position targets, 20 hand position targets]_t
```

Both `observation.state` and `action` are 27-D. The action is **not** the next measured
state and is **not** the three-dimensional Wuji synergy. It is read directly from the
27 MuJoCo position-actuator controls after arm clipping and after the synergy mapper has
generated its rate-limited 20-D hand target. Consequently, `action -
observation.state` preserves the controller preload that helps hold the cube.

The original 10-D `[7 arm targets, 3 hand synergies]` command remains in
`raw_script_action` only for audit and deterministic replay. This does not change the
existing 10-D online LeRobot robot-plugin action contract.

## Batch collection

Use the environment that already contains MuJoCo, PyArrow, and Pillow:

```powershell
$env:PYTHONPATH = "src"
./.venvs/lerobot-policy/Scripts/python.exe scripts/collect_coordinated_demos.py `
  --seed-start 0 --attempts 100 --max-successes 50
```

Explicit seeds are also supported:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/collect_coordinated_demos.py `
  --seeds 7,11,18 --image-height 240 --image-width 320
```

The collector never changes the approach, grasp, preload, Lift, or hold commands.
Every attempted episode is retained:

```text
outputs/coordinated_demos/
    raw/successful/*.npz
    raw/diagnostics/*.npz
    raw/diagnostics/*.json  # failures before the first control transition
    summary.json
    episodes.jsonl
    lerobot_staging/
```

An episode enters `raw/successful` only when Reach and grasp complete, the cube reaches
the Lift target, remains above the external 25 mm load-bearing floor, and completes the
configured 0.5-second post-trajectory hold. `grasp_stable` and the outcome taxonomy are
retained as diagnostics; the external 8 mm / 6 degree gate is not silently substituted
for the requested unsupported-Lift-and-hold definition.

## Raw episode schema

The policy-aligned arrays have one row per control transition and begin at reset frame
zero:

| Field | Shape | Meaning |
|---|---:|---|
| `timestamp` | `[T]` | Simulation time relative to the reset observation |
| `observation.state` | `[T,27]` | Actual arm then actual hand joint positions |
| `action` | `[T,27]` | Exact arm then hand position-controller targets |
| `raw_script_action` | `[T,10]` | Original arm + synergy command, audit only |
| `next_observation.state` | `[T,27]` | Measured result after `action` |
| `observation.images.front` | `[T,H,W,3]` | Synchronized front RGB, uint8 HWC |
| `observation.images.wrist` | `[T,H,W,3]` | Synchronized wrist RGB, uint8 HWC |
| `phase` | `[T]` | Semantic Reach/approach/grasp/preload/Lift/hold phase |
| `controller_phase` | `[T]` | Original state-machine phase |
| `telemetry.cube_pose_world` | `[T,7]` | Cube xyz + scalar-first wxyz |
| `telemetry.cube_pose_relative_to_palm` | `[T,7]` | Full relative SE(3) pose |
| `telemetry.contact_count` | `[T]` | Diagnostic hand–cube contact count |
| `telemetry.active_finger_mask` | `[T,5]` | Diagnostic contact mask |
| `telemetry.finger_normal_force_n` | `[T,5]` | Diagnostic normal forces |

Scalar and JSON metadata include seed, task text, outcome, success/failure, Reach/grasp/
Lift/hold booleans, cube poses at reset and before approach/grasp-close/Lift, and the
first hand–cube contact phase, finger, body, geom, position, and force. Telemetry is not
part of the default policy observation.

The current task calls its movement-and-hold loop `lift_s_curve`; the recorder derives
semantic `lift` and `hold` labels from the configured trajectory duration without
changing any controller action.

## LeRobot staging export

Collection automatically exports successful raw episodes, or it can be rebuilt with:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/export_coordinated_demos.py
```

The staging directory contains LeRobot-named Parquet columns, Hugging Face Image-style
PNG byte structs, `meta/info.json`, `episodes.jsonl`, `tasks.jsonl`, and state/action
statistics. It is intentionally marked `native_lerobot_dataset=false`.

One final conversion step remains: install LeRobot's optional `datasets` dependency,
write/import the staging rows through `LeRobotDataset` 0.6.2, and pass a DataLoader
smoke test. For a large collection, images should then be encoded as MP4 rather than
kept as PNG bytes in Parquet. No ACT or SmolVLA training should start until that native
dataset validation and train/validation split are complete.
