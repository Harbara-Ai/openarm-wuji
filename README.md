# OpenArm + Wuji staged ACT pickup

This repository is a simulation-first robot-learning project that coordinates a
7-DoF OpenArm and a 20-DoF Wuji hand in MuJoCo. The current reproducible milestone
executes:

```text
fresh reset
→ Reach ACT
→ Approach ACT
→ selective_hysteresis router
→ optional Recovery ACT
→ GraspSecure ACT
→ scripted Lift
→ 1 s terminal hold
→ success/failure report
```

The learned interface is fixed at 30 Hz:

```text
observation = front RGB 240×320 + wrist RGB 240×320 + 27D actual qpos
action      = 27D absolute position-controller target
              [7D OpenArm target + 20D Wuji target]
H_exec      = 1
```

The formal fresh-seed benchmark (seeds 9000–9099) completes the full task in
**21/100 runs**. This is a reproducibility milestone, not a claim that grasping is
solved. See [the final staged-ACT report](docs/FINAL_STAGED_ACT_REPORT.md) for the
stage-wise results, limitations, and representative rollouts. The full experiment
history, including negative results, is in
[the experiment journal](docs/project_experiment_journal.md).

## Frozen pipeline

[`configs/staged_pipeline.json`](configs/staged_pipeline.json) is the single source
of truth. It fixes all policy checkpoints, the router, control rate, observation and
action contract, scripted Lift configuration, safety semantics, and SHA-256 hashes
for the large local artifacts.

| Stage | Frozen implementation |
|---|---|
| Reach | ACT step 2000 |
| Approach | ACT step 2000 |
| Recovery | ACT step 1500, selected by `selective_hysteresis` |
| GraspSecure | ACT step 1500 |
| Lift | existing scripted trajectory, followed by a 1 s hold |

The GraspSecure static gate is diagnostic only for Lift admission. A finite
GraspSecure terminal state may continue to Lift if the original 25 mm pre-Lift cube
motion safety guard passes. Contact topology, force, and relative drift remain
telemetry. Retry, micro-lift probe, online expert actions, and Lift ACT are disabled.

## Install on Windows

Python 3.12 and Git are required. Clone the pinned upstream projects into `vendor/`
(this directory is intentionally not committed):

```powershell
git clone https://github.com/enactic/openarm_mujoco.git vendor/openarm_mujoco
git -C vendor/openarm_mujoco checkout a8c979629f2591ad035d99d338ce114969e6cddc

git clone --recursive https://github.com/wuji-technology/wuji-retargeting.git vendor/wuji-retargeting
git -C vendor/wuji-retargeting checkout 531f6ed4250b475d2e9231f54e988fc9b1c5b4ea

git clone https://github.com/huggingface/lerobot.git vendor/lerobot
git -C vendor/lerobot checkout fa048804d05c1b965b0a5801cf205f97a9e8a3e8

./scripts/setup_lerobot_policy.ps1 -Python C:\path\to\python.exe
```

Pinned versions and upstream licenses are documented in
[`docs/upstream_versions.md`](docs/upstream_versions.md).

## Prepare frozen artifacts

The four ACT checkpoints total about 827 MB and the compiled MuJoCo model is about
90 MB. They are deliberately excluded from GitHub. Restore the preserved checkpoint
directories at the exact repo-relative paths recorded in the manifest:

```text
outputs/act_reach_only/act_train/checkpoints/002000/pretrained_model
outputs/act_staged/approach_only/act_train/checkpoints/002000/pretrained_model
outputs/staged_act_with_recovery/recovery_act_train/checkpoints/001500/pretrained_model
outputs/grasp_preload_act/act_train/checkpoints/001500/pretrained_model
```

Each directory must contain the LeRobot `config.json`, preprocessors,
postprocessors, `train_config.json`, and `model.safetensors`. The training procedure
for each asset is documented in
[`act_reach_only_diagnosis.md`](docs/act_reach_only_diagnosis.md),
[`staged_act_pipeline.md`](docs/staged_act_pipeline.md),
[`staged_act_with_recovery.md`](docs/staged_act_with_recovery.md), and
[`grasp_preload_act.md`](docs/grasp_preload_act.md).

Build the compiled scene from the pinned upstream MJCF assets:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/build_reach_grasp_lift_model.py `
  --project . `
  --config configs/reach_grasp_lift.json `
  --output outputs/reach_grasp_lift
```

Validate paths, file sizes, and SHA-256 hashes without loading a policy:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/run_staged_pickup.py --check-only
```

## Run the frozen system

Run one deterministic closed-loop rollout:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/run_staged_pickup.py `
  --seed 9000 `
  --output outputs/staged_pickup_single
```

Run the formal 100-seed protocol and render three representative rollouts:

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/run_staged_pickup.py `
  --seed-start 9000 `
  --num-runs 100 `
  --output outputs/staged_pickup_formal_seed9000_100 `
  --compact-summary benchmarks/staged_pickup_fresh_seed9000_100.json `
  --save-representatives
```

Use `--resume` to continue an interrupted output directory or `--overwrite` to
replace one intentionally. Every episode records explicit phase outcomes plus
`failure_stage` and `failure_reason`. The runner writes:

```text
outputs/<run>/progress.jsonl
outputs/<run>/summary.json
outputs/<run>/summary.compact.json
outputs/<run>/pipeline_manifest.snapshot.json
outputs/<run>/representative_rollouts/{success,...}/
```

Large output trajectories and media stay local. The small audited benchmark summary
is committed at
[`benchmarks/staged_pickup_fresh_seed9000_100.json`](benchmarks/staged_pickup_fresh_seed9000_100.json).

## Tests

```powershell
$env:PYTHONPATH = 'src;.'
./.venvs/lerobot-policy/Scripts/python.exe -B -m unittest discover `
  -s tests -p 'test_*.py' -v
```

The suite covers policy/dataset contracts, state-action alignment, staged routing,
GraspSecure/Lift handoff, reporting semantics, and the portable manifest. The formal
release run passed **103/103 tests**; a separate one-episode formal-runner smoke also
completed Reach → Recovery → GraspSecure → Lift → Hold successfully. The 100-run
benchmark is the full end-to-end integration exercise.

## Known limitations

- Actual scripted-Lift success after admission is 21/44 (47.7%).
- The GraspSecure static gate has limited load-bearing discrimination; preload alone
  is not sufficient.
- There is no deployed retry or micro-lift verification mechanism.
- Lift remains scripted; no Lift ACT has been trained.
- The four checkpoints are external artifacts, not Git objects.
- Results are MuJoCo-only. Real-hardware communication, tool-to-palm calibration,
  tactile calibration, and sim-to-real validation are not complete.

## Historical components

The repository also retains the scripted grasp, coordinated demonstration recorder,
LeRobotDataset v3 exporter, monolithic ACT ablations, SAC/action-prior experiments,
and rejected probe/retry experiments as documented evidence. They are not enabled by
the formal manifest. Start with the
[final report](docs/FINAL_STAGED_ACT_REPORT.md), then use the
[journal](docs/project_experiment_journal.md) as the detailed index.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
Third-party components retain their upstream licenses.
