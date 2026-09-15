# Final staged ACT pickup report

Date: 2026-09-15 (Asia/Singapore)
Formal pipeline: `openarm_wuji_staged_act_pickup_v1`
Formal manifest: [`configs/staged_pipeline.json`](../configs/staged_pipeline.json)
Machine-readable benchmark: [`benchmarks/staged_pickup_fresh_seed9000_100.json`](../benchmarks/staged_pickup_fresh_seed9000_100.json)

## 1. Project objective and current result

The project coordinates OpenArm and the Wuji dexterous hand in MuJoCo to execute:

```text
Reach → Approach → Grasp → Lift → 1 s hold
```

The current system is a complete closed-loop staged pipeline. It uses current RGB
and measured robot state at every control step; it does not read an expert action or
inject an expert state. On 100 fresh deterministic simulator seeds (`9000–9099`) it
completed the full load-bearing task in **21/100 episodes**.

This milestone freezes and packages the working system. It does not claim that grasp
reliability is solved.

## 2. Why the project uses staged ACT

The first end-to-end behavior-cloning pipeline proved the complete data path:

```text
MuJoCo expert → LeRobotDataset v3 → ACT → MuJoCo controller
```

Its training loss and offline reconstruction error decreased, but a monolithic ACT
trained on the whole multi-phase task achieved `0/20` closed-loop Reach even on the
demonstration initial conditions. ImageNet initialization, chunk-size and
phase-balanced sampling ablations improved continuous offline metrics but did not
remove that failure.

The controlled Reach-only experiment changed the conclusion: the same core
observation/action interface learned stable pregrasp Reach on `19/20` training-demo
initial conditions. Removing Approach/Grasp/Lift therefore removed a major source of
phase and temporal ambiguity. The project consequently moved to separate policies
with explicit gates and a selective Recovery policy, rather than continuing to add
steps to the monolithic model.

## 3. Frozen architecture and interface

```text
fresh reset
  ↓
Reach ACT ── Reach gate
  ↓
Approach ACT ── selective_hysteresis router ── optional Recovery ACT
  ↓
Approach-stage success
  ↓
GraspSecure ACT
  ↓ finite terminal + 25 mm pre-Lift safety guard
scripted Lift
  ↓
unsupported terminal hold ≥ 1 s
  ↓
SUCCESS / FAILURE
```

All ACT stages use the same 30 Hz policy contract:

- Input: front and wrist RGB (`uint8`, `240×320`) plus 27D actual qpos
  (`7D OpenArm + 20D Wuji`).
- Output: 27D absolute target sent to the MuJoCo position controller
  (`7D OpenArm + 20D Wuji`).
- Execution horizon: `H_exec=1`; the policy re-observes after every control step.
- Cube pose, contacts, force, relative SE(3), phase and timestamps are telemetry and
  never enter the policy observation.

| Module | Input → output | Frozen artifact | Gate / role |
|---|---|---|---|
| Reach ACT | dual RGB + 27D state → 27D target | `outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model` | Pregrasp position error ≤12 mm for 5 consecutive frames |
| Approach ACT | same → same | `outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model` | Drives from pregrasp to the frozen grasp-start terminal region while monitoring cube motion |
| Recovery ACT | same → same | `outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model` | One optional correction selected by `configs/recovery_router.json` (`selective_hysteresis`) |
| GraspSecure ACT | same → same | `outputs/grasp_preload_act/act_train/checkpoints/001500/pretrained_model` | Forms grasp/preload and returns its real terminal simulator state; static gate remains diagnostic |
| scripted Lift | live simulator state + preserved current controller target → 27D targets | `openarm_wuji.tasks.scripted_lift_handoff` and `configs/reach_grasp_lift.json` | Existing bounded Lift trajectory, then unsupported 1 s hold |

The manifest also records SHA-256 hashes for the four `model.safetensors` files and
the compiled MuJoCo model. Those large artifacts are local and intentionally outside
Git.

## 4. Gate and Lift-admission semantics

The formal runner does **not** require the GraspSecure static gate to pass before
Lift. That gate combines contact/preload/terminal-hold features that are useful
diagnostics, but forced-Lift experiments showed both false positives and false
negatives. In the formal pipeline:

1. GraspSecure must return finite qpos, qvel, controller target and cube state.
2. The original hard safety rule rejects Lift when pre-Lift cube motion exceeds
   25 mm.
3. Otherwise the runner preserves the live Wuji controller target and starts the
   existing scripted Lift from the actual closed-loop terminal state.
4. Contact count/topology, force, preload and relative drift are recorded, not used
   as new hidden admission thresholds.

No retry or micro-lift probe is active. The earlier retry/probe implementation and
negative result remain in the repository for experimental provenance, but
`configs/staged_pipeline.json` explicitly disables them.

## 5. Formal fresh-seed benchmark

Protocol:

- 100 complete closed-loop rollouts, deterministic seeds `9000–9099`.
- Fixed policy inference seed `0`.
- Frozen checkpoints, router, controller, scene, task gates and scripted Lift.
- No expert action, state injection, policy training, retry or probe.

### Stage-wise results

| Quantity | Count | Probability |
|---|---:|---:|
| Reach success | 78/100 | 78.0% |
| Approach-stage success given Reach | 60/78 | 76.9% |
| Finite GraspSecure terminal given Approach | 60/60 | 100.0% |
| Static GraspSecure gate PASS (diagnostic) | 38/60 | 63.3% |
| Pre-Lift safety pass | 44/60 | 73.3% |
| Lift success given finite GraspSecure terminal | 21/60 | 35.0% |
| Lift success given an actual Lift attempt | 21/44 | 47.7% |
| **Full task success** | **21/100** | **21.0%** |

The stage-product is consistent with the measured full rate:
`0.78 × 0.769 × 1.00 × 0.35 = 0.21` (rounding aside). The 35% conditional value
includes the 16 safety-rejected terminals; the 47.7% value isolates actual Lift
attempts.

Recovery was triggered in `34/78` Reach-success episodes (43.6%) and succeeded in
`16/34` triggers (47.1%).

### Failure distribution

The mutually exclusive final failure stages were:

| Failure stage | Count |
|---|---:|
| Reach | 22 |
| Recovery / Approach-stage timeout | 18 |
| Pre-Lift terminal safety | 16 |
| Lift drop | 10 |
| Lift hold | 13 |

The 23 failed actual Lift attempts consisted of 10 drops, 11 environment-support
failures and 2 height-hold failures. Across all episodes the diagnostic timeout count
was 35 (22 Reach plus 13 downstream timeouts). Pre-Lift cube motion over the 60
GraspSecure terminals had median 7.67 mm and mean 35.04 mm; the mean is dominated by
large escape outliers. The maximum cube displacement diagnostic over those terminals
had median 81.57 mm.

### What the static gate predicted

Across 60 finite GraspSecure terminals:

| Static gate | Full Lift success | Not successful |
|---|---:|---:|
| PASS | 17 | 21 |
| FAIL | 4 | 18 |

Four full successes (`9030`, `9032`, `9033`, `9097`) were static-gate failures.
Therefore making this gate a hard Lift admission rule would discard real
load-bearing grasps. Conversely, many PASS terminals still failed. This supports the
frozen diagnostic-only semantics and confirms that preload/contact at a static
terminal is not a sufficient load-bearing criterion.

## 6. Representative visualizations

The formal runner deterministically replays and renders one example from each
required category. Each replay matched the benchmark success, failure stage and
failure reason.

| Category | Seed | Outcome | Local artifacts |
|---|---:|---|---|
| Complete success | 9002 | Reach → Approach → GraspSecure → Lift → Hold | `outputs/staged_pickup_formal_seed9000_100/representative_rollouts/success/` |
| Reach/Approach failure | 9000 | Reach timeout | `outputs/staged_pickup_formal_seed9000_100/representative_rollouts/reach_or_approach_failure/` |
| Grasp/Lift failure | 9003 | Drop during Lift | `outputs/staged_pickup_formal_seed9000_100/representative_rollouts/grasp_or_lift_failure/` |

Each directory contains a dual-camera `rollout.gif`, `phase_timeline.png`, compact
`summary.json`, and (where applicable) ignored raw 27D trajectory NPZ files. Media
and raw trajectories remain outside Git; the committed compact benchmark records
their paths and deterministic outcome match.

## 7. Reproduction

Follow the root [`README.md`](../README.md) to install the three pinned upstream
repositories, create the Python 3.12 LeRobot environment, restore the four frozen
checkpoint directories, and build the MuJoCo model. Validate all frozen artifacts:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/run_staged_pickup.py --check-only
```

One rollout:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/run_staged_pickup.py `
  --seed 9000 --output outputs/staged_pickup_single
```

Formal protocol:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/run_staged_pickup.py `
  --seed-start 9000 --num-runs 100 `
  --output outputs/staged_pickup_formal_seed9000_100 `
  --compact-summary benchmarks/staged_pickup_fresh_seed9000_100.json `
  --save-representatives
```

`progress.jsonl` makes the batch resumable. `summary.json` contains per-episode rows;
`summary.compact.json` contains the stage probabilities, failures and diagnostics;
the manifest snapshot records the exact run composition.

## 8. What was tried and rejected

The project reached this design through controlled negative results rather than a
single successful run:

- Early scripted power grasps could lift but showed off-center motion and large
  palm-frame drift.
- Freeze/settle, preload sweeps and gentler S-curve Lift improved control semantics
  but did not make grasp geometry reliably stable.
- Reward tuning and 20D/structured SAC did not discover persistent multi-contact;
  expert PCA5 improved contact acquisition but not stable topology.
- Monolithic ACT reconstructed demonstrations offline yet failed closed-loop Reach.
- Reach-only ACT isolated phase ambiguity and motivated staged policies.
- Mixing correction demonstrations into Approach degraded success; an independent
  Recovery ACT plus a matched selective router was better.
- A static GraspSecure gate and preload norm did not predict load reliably.
- A 15 mm micro-lift hold probe rejected all 60 evaluated terminals even though
  18 later succeeded under the original continuous Lift; it was not deployed.
- One frozen regrasp retry produced no rescue in that conditional experiment and is
  not part of this milestone.

The detailed evidence and links to every specialized report are preserved in
[`project_experiment_journal.md`](project_experiment_journal.md).

## 9. Tests and integrity checks

The release verification passed **103/103 tests**. It includes policy and dataset
loader contracts, state/action alignment, manifest/path/hash validation, staged
routing, GraspSecure and Lift handoff tests, plus reporting semantics. A separate
one-episode formal-runner smoke (seed 8999 when run as a standalone inference stream)
completed Reach → Recovery → GraspSecure → Lift → Hold. The 100-run fresh-seed
benchmark is the full end-to-end integration exercise.

## 10. Known limitations and frozen decision

- Full success is 21%, and actual scripted-Lift conditional success is 47.7%.
- GraspSecure static classification has limited load-bearing discrimination.
- Preload magnitude is correlated weakly but is not a sufficient criterion.
- No retry or verification probe is deployed.
- Lift is scripted; no Lift ACT has been trained.
- Large checkpoints and generated media are external local artifacts.
- Results are simulation-only; real-hardware calibration and validation remain open.

The formal decision is to **accept these limitations and freeze the reproducible
pipeline**. Future experiments may improve grasp verification or learn Lift, but
they must start from this manifest and benchmark rather than silently changing the
current result.
