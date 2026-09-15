# OpenArm + Wuji: 20-demo ACT end-to-end smoke test

Date: 2026-09-10
LeRobot: 0.6.2
Dataset codebase version: v3.0

## Result

The complete software path is operational:

```text
scripted MuJoCo expert
  -> synchronized 27-D + two-camera recording
  -> native LeRobotDataset v3.0
  -> ACT training and checkpoint
  -> checkpoint reload
  -> 27-D MuJoCo position-controller rollout
```

The 50-update checkpoint does **not** yet reproduce the complete expert
behavior. On training seeds 0, 7, and 11 it drives both OpenArm and Wuji, but
closes the hand too early, pushes the cube, and fails before a valid reach.
This is a behavioral/model-undertraining failure, not a broken action interface.

## Demonstration collection

- Requested successful demonstrations: 20
- Attempts required: 23
- Successful demonstrations: 20
- Diagnostic failures retained: 3 (seeds 4, 13, and 15)
- Training frames: 2,804
- Frequency: 30 Hz
- Resolution: 240 x 320 RGB for both cameras
- Successful training seeds: 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12,
  14, 16, 17, 18, 19, 20, 21, and 22

An episode enters the behavioral-cloning set only when the existing recorder
observes grasp, unsupported lift, and the required 15-frame hold. All three
failed attempts remain under `raw/diagnostics` and were excluded from the
native dataset. The successful set deliberately contains the recorder's task
successes even when the stricter diagnostic `grasp_stable` flag is false; that
matches the collection criterion for this smoke test and remains visible in
the raw metadata.

## Dataset schema and alignment

| Feature | Shape | Meaning |
| --- | ---: | --- |
| `observation.state` | `[27]` | 7 actual OpenArm qpos + 20 actual Wuji qpos |
| `action` | `[27]` | exact 7 + 20 MuJoCo position-actuator targets sent at that transition |
| `observation.images.front` | `[3, 240, 320]` | synchronized front RGB |
| `observation.images.wrist` | `[3, 240, 320]` | synchronized wrist RGB |
| `timestamp` | scalar | canonical `frame_index / 30` seconds within episode |

The dataset was written through `LeRobotDataset.create`, finalized, reloaded,
and read through a PyTorch DataLoader. The first batch shapes were state
`[8,27]`, action `[8,27]`, and both images `[8,3,240,320]`. Exact state/action
round-trip comparison passed. For all 20 episodes, global row ranges are
contiguous, local frame indices reset to zero, timestamps equal
`frame_index / fps`, and both camera tensors align with the same row. Image
features use native LeRobot PNG storage (`use_videos=False`); this avoids the
Windows FFmpeg/TorchCodec shared-library issue without changing policy data.

## ACT training

ACT used the LeRobot 0.6.2 defaults where practical: ResNet-18, 512 model
dimension, 8 attention heads, four encoder layers, one decoder layer, a
100-step action chunk, VAE latent dimension 32, KL weight 10, and AdamW at
`1e-5`. Necessary smoke-test compatibility changes were CPU execution,
`num_workers=0`, no Weights & Biases, PyAV dataset loading, and no downloaded
ImageNet backbone weights. A small Windows wrapper replaces the privileged
`checkpoints/last` symlink with `last_checkpoint.txt`.

Training ran for 50 optimizer updates (batch 8). Loss was noisy but decreased
normally:

| Step | Total loss |
| ---: | ---: |
| 1 | 68.460 |
| 10 | 11.320 |
| 25 | 6.978 |
| 37 | 4.852 |
| 47 | 4.368 |
| 50 | 5.237 |

The final checkpoint reload passed. Policy inputs are state `[B,27]`, front
`[B,3,240,320]`, and wrist `[B,3,240,320]`. The ACT network predicts an
internal `[B,100,27]` chunk; `select_action` exposes one unnormalized `[B,27]`
absolute controller target per control step.

## MuJoCo rollout

The new inference path sends the 27-D ACT result directly to the same MuJoCo
position actuators used to record `action`. It does not reinterpret the 20 hand
joints as the legacy three synergy values. All predicted values in the three
rollouts were finite and inside actuator ranges: maximum clipping was 0 rad.

| Training seed | Reach | Max contacts | Longest >=2 contacts | Peak cube lift | Cube displacement | Full sequence |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | no | 2 | 1 frame | 22.8 mm | 223.1 mm | no |
| 7 | no | 1 | 0 frames | 19.9 mm | 168.6 mm | no |
| 11 | no | 2 | 1 frame | 24.4 mm | 169.8 mm | no |

The earliest failure is reproducible: for seed 0 the first predicted action has
mean absolute error 0.496 rad against the expert first target (arm 0.159 rad,
hand 0.615 rad). The expert is still nearly open on frame 0, whereas ACT already
commands most hand joints around 0.7--0.85 rad. That premature closing explains
the large cube displacement and why none of the three runs reaches the
pregrasp tolerance.

## Diagnosis

- Data pipeline: passed (20 successes, 2,804 aligned frames, exact 27-D round trip).
- Checkpoint and inference pipeline: passed (reload, preprocessing,
  unnormalization, `[27]` output, direct controller execution).
- Learned behavior: failed (0/3 complete reach-approach-grasp-lift rollouts).
- Primary current blocker: model optimization/training budget. Fifty updates
  prove the pipeline and produce falling loss, but are far from convergence.
- Secondary risk after adequate training: only 20 highly scripted trajectories
  may not cover recovery from off-trajectory states. The present test does not
  yet isolate that risk because the checkpoint already has large error on the
  very first training frame.

Accordingly, the first-stage *software pipeline* is connected, but the stronger
criterion "ACT noticeably reproduces expert behavior" is not yet met. The clean
next experiment is to resume this exact checkpoint/config for substantially
more optimizer updates and evaluate fixed checkpoints on the same training
seeds; no grasp, reward, observation, or controller change is justified by this
smoke result.

## Reproducible entry points

```text
scripts/collect_coordinated_demos.py
scripts/export_native_lerobot_dataset.py
scripts/validate_native_lerobot_dataset.py
scripts/train_act_smoke.py
scripts/rollout_act_mujoco.py
```

Artifacts:

```text
outputs/act_e2e_smoke/coordinated_demos/
outputs/act_e2e_smoke/lerobot_dataset/
outputs/act_e2e_smoke/act_train/checkpoints/000050/pretrained_model/
outputs/act_e2e_smoke/rollouts/
```

Repository verification after adding the exporter, validator, absolute-target
controller path, and ACT rollout entry point: 58 unit/integration tests passed.

The follow-up exact-resume experiment through 500 optimizer steps, including
fixed-seed rollouts every 100 steps, is reported in
`docs/act_500_step_fixed_seed_rollouts.md`.
