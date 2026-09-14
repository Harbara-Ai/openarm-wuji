# Frozen staged ACT system v1

This release freezes the currently selected staged controller:

```text
Reach ACT step 2000
  -> Reach gate
Approach ACT step 2000
  -> selective_hysteresis router
Recovery ACT step 1500 (zero or one attempt)
```

The complete machine-readable inventory, interface contract, gates, validation
result, and SHA-256 checksums are in
`configs/frozen_staged_act_v1.json`. Model and normalization `.safetensors`
files are versioned with Git LFS; a clone must run `git lfs pull` before use.

## Frozen behavior

- Observation: front and wrist RGB at 240 x 320 plus 27-D actual qpos.
- Action: 27-D absolute target sent to the position controller.
- ACT: ImageNet-initialized ResNet-18, chunk 20, `n_action_steps=20`.
- Execution: strict closed loop, `H_exec=1`.
- Reach gate: position error at most 12 mm for five consecutive frames.
- Router: wait at least 12 Approach frames, require persistent failure evidence,
  permit at most one switch to Recovery.
- Grasp, Preload, and Lift are not part of this frozen learned controller.

On the matched router ablation (seeds 1400-1599), 155 episodes passed Reach.
Approach-stage success improved from 112/155 to 129/155; the router rescued 22
and regressed 5 episodes. Timeout stayed 20, while cube-safety failures fell
from 23 to 6. These are simulation results, not a real-robot guarantee.

## Reproduce the staged evaluation

After cloning the repository and installing its LeRobot/MuJoCo environment:

```powershell
git lfs pull
python scripts/evaluate_staged_with_recovery.py `
  --output outputs/frozen_staged_act_v1/evaluation `
  --rollouts 200 `
  --seed-start 1400
```

The evaluator defaults to the three frozen checkpoints and
`configs/recovery_router.json`. Use an explicit output directory; add
`--overwrite` only when replacing a previous local result intentionally.

Supporting evidence is documented in `docs/staged_act_pipeline.md`,
`docs/staged_act_with_recovery.md`, and `docs/recovery_router_ablation.md`.
