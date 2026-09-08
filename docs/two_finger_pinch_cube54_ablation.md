# Two-finger pinch cube-size ablation: 70 mm vs 54 mm

## Scope

This is a strict single-variable ablation of the existing fixed-palm thumb-index reachability audit. The only physics value changed is cube half-size, from `0.035 m` (70 mm edge) to `0.027 m` (54 mm edge). Cube mass (`0.12 kg`), center/orientation logic, friction/contact parameters, solver/timestep, Wuji collision geometry, actuators, controller, joint limits, passive-finger posture, reward/success definitions, search algorithm, seed, budget, and trajectory duration were kept unchanged. No RL training was run.

The 54 mm audit used the same `1 + 1600 + 30 * 3 * 4 = 1961` candidates, seed `7`, and the same full-candidate evaluation. The cube center logic was unchanged; the rebuilt model settled at `[0.48, 0.150000289, 0.426892244]`.

## Results

| Metric | 70 mm baseline | 54 mm ablation |
|---|---:|---:|
| Total candidates | 1961 | 1961 |
| Ever dual-contact | 1658 | 1291 |
| Dual-contact >= 0.10 s | 56 | 41 |
| Dual-contact >= 0.20 s | 0 | 0 |
| Dual-contact >= 0.30 s | 0 | 0 |
| Dual-contact >= 0.50 s | 0 | 0 |
| Strong pinch | 0 | 0 |
| Longest dual-contact (full review) | 0.1333 s | 0.1667 s |
| Mean slip of best | 13.99 mm/s | 32.47 mm/s |
| Max slip of best | 47.17 mm/s | 76.11 mm/s |
| Thumb-index normal angle of best | 90.36° | 95.37° |
| Thumb face / index face | -Y / +Z | +Z / -Y |
| Cube translation drift of best | 0.182 mm | 0.356 mm |
| Cube rotation drift of best | 0.039° | 0.457° |
| Penetration of best | 1.382 mm | 0.752 mm |
| Opposing normals > 90° | 591 | 519 |

The 70 mm values are the project's existing formal baseline (`outputs/two_finger_pinch/top_candidates.json` and `candidates.parquet`). Candidate counts for the 54 mm column come from the new audit parquet; the longest-duration value is from its full-review `top_candidates.json`.

## Best 54 mm candidate

```text
candidate_id: full_00_from_local_22_0.035_02
thumb: [1.0445917923281438, 0.866092637160105, 0.7307873117721497, 1.3405448749317606]
index: [1.1635204179494523, -0.37, 1.3399471706900687, 1.1438294161177518]
```

The thumb contacted cube face `+Z` and the index contacted `-Y`. Their mean contact-normal angle was `95.37°` (range `91.27°–98.03°`), still far from an antipodal `150°–180°` geometry. Mean tangential slip was `32.47 mm/s` and maximum slip `76.11 mm/s`. Penetration was `0.752 mm` and was not classified as an exploit. Low cube motion is not evidence of a hold because the cube remains table-supported.

Fixed-target validation over seeds 7–11 produced strong success `0/5` and formal pure thumb-index success `0/5`; longest dual-contact was `[0.1667, 0.1667, 0.0667, 0.1000, 0.1000] s`.

## Conclusion

This is **Case C: partial reachability improvement, stable grasp still absent**. The maximum continuous dual-contact increased only from `0.1333 s` to `0.1667 s` (about `1.25x`), with no candidate reaching `0.20 s`, `0.30 s`, or `0.50 s`; slip worsened and the contact-normal geometry moved only from `90.36°` to `95.37°`, not toward antipodal opposition. Therefore changing 70 mm to 54 mm alone is not sufficient to explain or solve the previous failure. The next diagnosis should prioritize the difference in palm-object geometry and the official Wuji collision/contact/controller setup, while keeping this size result as a controlled negative/partial ablation.

## Artifacts

The raw 54 mm artifacts were saved without overwriting the existing 70 mm outputs at:

`D:/yl/embodied ai/_cube54_ablation_runtime/audit/candidates.parquet`

`D:/yl/embodied ai/_cube54_ablation_runtime/audit/top_candidates.json`

`D:/yl/embodied ai/_cube54_ablation_runtime/audit/multi_seed_validation.json`

The machine-readable project summary is `outputs/two_finger_pinch_cube54_summary.json`.
