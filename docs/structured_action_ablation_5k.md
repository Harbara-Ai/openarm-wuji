# Structured 5-D action ablation, 5K SAC

Date: 2026-09-07. This is a controlled comparison against the 20-D Reward V2 run.
Observation (228-D), Reward V2, reset, 120-step horizon, success/failure criteria,
fixed palm, SAC hyperparameters, Lift exclusion, and ACT exclusion are byte-for-byte
equivalent in the two configurations. Only the action representation changes.

## Representation

The policy outputs five scalars in `[-1, 1]`, one for each finger. Within each finger,
the scalar multiplies an L2-normalized direction derived from the previously verified
scripted probe:

```text
d_i = sign(q_close_i - q_open_i)
delta_q_i = scale_i * a_i * d_i / ||d_i||_2
```

The thumb scale is 0.0700 rad and the other scales are 0.06062 rad. These are the
norms of the old four-joint/three-joint 0.035 rad delta vectors, so an all-ones 5-D
action expands to exactly the old 20-D scripted full-close command. Rate limiting,
smoothing, and joint-limit clipping occur as before. The observation retains its
original 20-D previous-joint-command field, populated by the expanded command, and
therefore remains 228-D.

## Sanity tests

Test A, `[1, 1, 1, 1, 1]`, ran from seed 7 for 120 steps:

| Metric | Result |
|---|---:|
| maximum simultaneous contacts | 5 |
| first contact | step 22 |
| deepest penetration | 0.512 mm |
| failure termination | none |

Every fingertip reached approximately zero signed cube distance at some point. This
reproduces the contact-rich reachability of the old 20-D scripted direction, although
it is not a stable grasp and ends with zero contacts.

Test B sent the five one-hot actions separately. In all five cases, the commanded
joint delta was exactly zero outside the selected finger and non-zero inside it.
Small passive motion in uncommanded joints after physics stepping is reported
separately and is not actuator-command leakage.

## 20-D V2 versus 5-D structured V2

Both SAC policies were initialized from scratch with seed 7 and trained for 5,000
environment steps (41 complete episodes and 4,900 gradient updates).

| Metric | 20-D V2 | 5-D structured V2 |
|---|---:|---:|
| mean episode return | -1.859 | -1.736 |
| first / last 10-episode return | -1.786 / -1.874 | -1.585 / -1.751 |
| episodes with any contact | 10/41 (24.4%) | 20/41 (48.8%) |
| maximum simultaneous contacts | 1 | 1 |
| episodes with >=2 contacts | 0 | 0 |
| episodes with >=3 contacts | 0 | 0 |
| held >=2 contacts for 0.1 s / 0.3 s | 0 / 0 | 0 / 0 |
| stable grasp success | 0 | 0 |

The higher contact frequency and slightly less-negative return are real but do not
meet the multi-contact acceptance criterion.

### Distance behavior

| Metric | 20-D V2 | 5-D structured V2 |
|---|---:|---:|
| third-closest, first 250 steps | 35.112 mm | 35.059 mm |
| third-closest, last 250 steps | 35.437 mm | 35.278 mm |
| within-run change | +0.325 mm | +0.219 mm |
| deterministic final third-closest, seeds 7-11 | 35.401 mm | 35.320 mm |

Lower is better. The 5-D result is only 0.081 mm better in deterministic evaluation,
and its own training curve becomes 0.219 mm worse. This is not meaningful evidence of
multi-finger approach.

The final deterministic mean finger distances across seeds 7-11 are:

| Finger | 20-D V2 | 5-D structured V2 |
|---|---:|---:|
| thumb / finger1 | 6.562 mm | 3.903 mm |
| finger2 | 35.401 mm | 35.320 mm |
| middle / finger3 | 40.051 mm | 40.066 mm |
| ring / finger4 | 36.751 mm | 36.749 mm |
| little / finger5 | 32.726 mm | 32.331 mm |

The geometry remains thumb-dominant; fingers 2-5 stay far from the cube.

## Exploration and deterministic evaluation

5-D stochastic training mean absolute actions are 0.517, 0.514, 0.512, 0.520, and
0.524 for fingers 1-5. Per-dimension standard deviations are 0.585-0.594. Final policy
latent standard deviations are 0.873-0.907, and entropy alpha changes from 1.0 to
0.230, essentially matching the 20-D run. Exploration does not collapse to one action.

The learned deterministic commands for all five fingers are small and predominantly
negative (opening rather than coordinated closing). Seeds 7-11 all finish with zero
contacts, zero multi-contact, zero hold, and zero success.

## Interpretation and decision

The action mapper itself is not kinematically incapable: a sustained all-positive
structured action reaches five contacts. Reducing 20 dimensions to five also doubles
the frequency of isolated contact. However, ordinary zero-mean SAC exploration still
does not discover and retain the temporally coordinated positive commands needed for
multi-contact.

Therefore:

1. The 5-D representation does not break the learning bottleneck in 5K.
2. Maximum simultaneous learned contact remains one.
3. Third-closest distance does not meaningfully decrease.
4. The 10K continuation gate is not met; training stops at 5K.
5. Structured-plus-20-D-residual is premature because the coarse structured policy has
   not learned multi-contact. The next controlled experiment should seed contact-rich
   experience (scripted replay/demonstration) or use a closer-to-contact reset
   curriculum before adding residual freedom.

Evidence:

- `outputs/rl_grasp_stage1/structured5_sanity.json`
- `outputs/rl_grasp_stage1/sac_v2_structured5_5000_steps.json`
- `outputs/rl_grasp_stage1/sac_grasp_stage1_v2_structured5_5000_steps.zip`
- 20-D baseline: `outputs/rl_grasp_stage1/sac_v2_5000_steps.json`
