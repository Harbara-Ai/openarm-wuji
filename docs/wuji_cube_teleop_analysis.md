# Wuji cube teleop: raw Parquet numerical analysis

This report uses the original LeRobot `observation.state`, `action`, and `timestamp`
columns from [yeeeiii111/wuji-pick-and-place](https://huggingface.co/datasets/yeeeiii111/wuji-pick-and-place),
pinned at revision `211283b732d73c3d10c005ebc8191af2d63673cf`. It analyzes
`cube_left` episodes 60–89 and `cube_right` episodes 90–119: 60 source Parquet
files, 20,769 frames, 30 Hz. No joint-motion value is inferred from video.

The extracted hand-only Parquet was read back and compared element-for-element with
the source files. `q_hand`, `a_hand`, and `timestamp` are bit-exact after float32
export. The combined file SHA-256 is
`650d93a253286352e28aa6fa5ddee0e0312b0b71e4ad487231b049d42ced7f76`.

## 54D layout and joint order

Both state and action use:

```text
left_arm[0:7] + right_arm[7:14] + left_hand[14:34] + right_hand[34:54]
```

Therefore the left-hand range is global indices **14–33** and the right-hand range
is **34–53** (zero-based, inclusive).

Important provenance boundary: dataset `info.json` stores `names: null` for both
54D arrays. The names below are not embedded in the Parquet schema; they apply the
[official Wuji flat-array convention](https://docs.wuji.tech/docs/en/wujihandros2/latest/configuration/):
finger 1–5 = thumb, index, middle, ring, pinky; joint 1–4 = proximal to distal;
flat index = `finger_id * 4 + joint_id`. This is the strongest public mapping, but
the dataset producer should still confirm that its recorder did not reorder channels.

| Hand-local | Left global/name | Right global/name | Finger |
|---:|---|---|---|
| 0 | 14 `left_finger1_joint1` | 34 `right_finger1_joint1` | Thumb |
| 1 | 15 `left_finger1_joint2` | 35 `right_finger1_joint2` | Thumb |
| 2 | 16 `left_finger1_joint3` | 36 `right_finger1_joint3` | Thumb |
| 3 | 17 `left_finger1_joint4` | 37 `right_finger1_joint4` | Thumb |
| 4 | 18 `left_finger2_joint1` | 38 `right_finger2_joint1` | Finger2/index |
| 5 | 19 `left_finger2_joint2` | 39 `right_finger2_joint2` | Finger2/index |
| 6 | 20 `left_finger2_joint3` | 40 `right_finger2_joint3` | Finger2/index |
| 7 | 21 `left_finger2_joint4` | 41 `right_finger2_joint4` | Finger2/index |
| 8 | 22 `left_finger3_joint1` | 42 `right_finger3_joint1` | Middle |
| 9 | 23 `left_finger3_joint2` | 43 `right_finger3_joint2` | Middle |
| 10 | 24 `left_finger3_joint3` | 44 `right_finger3_joint3` | Middle |
| 11 | 25 `left_finger3_joint4` | 45 `right_finger3_joint4` | Middle |
| 12 | 26 `left_finger4_joint1` | 46 `right_finger4_joint1` | Ring |
| 13 | 27 `left_finger4_joint2` | 47 `right_finger4_joint2` | Ring |
| 14 | 28 `left_finger4_joint3` | 48 `right_finger4_joint3` | Ring |
| 15 | 29 `left_finger4_joint4` | 49 `right_finger4_joint4` | Ring |
| 16 | 30 `left_finger5_joint1` | 50 `right_finger5_joint1` | Little/pinky |
| 17 | 31 `left_finger5_joint2` | 51 `right_finger5_joint2` | Little/pinky |
| 18 | 32 `left_finger5_joint3` | 52 `right_finger5_joint3` | Little/pinky |
| 19 | 33 `left_finger5_joint4` | 53 `right_finger5_joint4` | Little/pinky |

## Extraction and phase definition

For every episode the analysis directly extracts `q_hand(t)`, `a_hand(t)`, timestamps,
frame indices, task metadata, and length. Start is the median of the first 0.5 s.
Main closing onset is the first sustained crossing of `max(0.08 rad, 12% of the
episode's 20D command excursion)`. Main close is the first sustained crossing of
85% excursion. Grasp posture is the median over the following one-second window.

This is a numerical segmentation of `a_hand`; the dataset has no tactile/contact
column. Video is not used to calculate joint values, signs, ratios, onset order, or
PCA. Every episode's frames, phase indices, start/grasp vectors, 20D deltas, and
source SHA-256 are in `cube_episode_metrics.parquet` and the JSON report.

Trajectory length is 346.15 ± 51.07 frames over all 60 episodes (median 333.5,
range 273–520). Left is 347.53 ± 50.10; right is 344.77 ± 51.99.

## Per-finger state closing delta

Values are `Δq = median(q_hand in grasp window) - median(q_hand in first 0.5 s)`
in radians. Each cell is `[joint1, joint2, joint3, joint4]`.

| Side/finger | Mean Δq | Median Δq | Std Δq |
|---|---|---|---|
| All thumb | `[+0.5112,-0.0646,-0.2840,-0.0020]` | `[+0.5162,-0.0330,-0.2475,+0.0000]` | `[0.2754,0.1574,0.1940,0.0055]` |
| All finger2 | `[+0.0723,+0.0550,+0.3972,+0.2092]` | `[+0.1112,+0.0686,+0.4553,+0.2336]` | `[0.2618,0.0766,0.2617,0.1236]` |
| All middle | `[+0.2494,+0.0776,+0.1683,+0.3678]` | `[+0.3420,+0.0909,+0.1291,+0.3583]` | `[0.3643,0.0997,0.3161,0.1674]` |
| All ring | `[+0.2750,+0.0972,+0.2881,+0.2073]` | `[+0.3633,+0.1046,+0.2892,+0.2586]` | `[0.3080,0.1092,0.4236,0.1959]` |
| All little | `[+0.1419,+0.0476,+0.2676,+0.0919]` | `[+0.1552,+0.0424,+0.2318,+0.0793]` | `[0.1547,0.0624,0.3679,0.1104]` |
| Left thumb | `[+0.5048,+0.0362,-0.4151,+0.0000]` | `[+0.5409,+0.0430,-0.3756,+0.0000]` | `[0.3079,0.1103,0.1797,0.0000]` |
| Left finger2 | `[+0.0782,+0.0537,+0.4208,+0.2019]` | `[+0.0953,+0.0376,+0.5403,+0.2210]` | `[0.2558,0.0787,0.3200,0.1032]` |
| Left middle | `[+0.2253,+0.0184,+0.2327,+0.2734]` | `[+0.2842,+0.0045,+0.2631,+0.2973]` | `[0.3548,0.0678,0.3936,0.1058]` |
| Left ring | `[+0.2676,+0.0168,+0.4385,+0.0857]` | `[+0.3422,+0.0185,+0.5296,+0.0968]` | `[0.3327,0.0660,0.5316,0.1542]` |
| Left little | `[+0.1831,+0.0100,+0.4083,+0.1353]` | `[+0.2104,-0.0009,+0.4651,+0.1257]` | `[0.1862,0.0468,0.4667,0.1190]` |
| Right thumb | `[+0.5176,-0.1655,-0.1530,-0.0041]` | `[+0.5072,-0.1801,-0.1667,-0.0012]` | `[0.2383,0.1307,0.0927,0.0072]` |
| Right finger2 | `[+0.0664,+0.0563,+0.3736,+0.2164]` | `[+0.1208,+0.0771,+0.4120,+0.2536]` | `[0.2675,0.0744,0.1829,0.1406]` |
| Right middle | `[+0.2734,+0.1368,+0.1039,+0.4622]` | `[+0.3885,+0.1650,+0.0554,+0.4991]` | `[0.3720,0.0909,0.1916,0.1644]` |
| Right ring | `[+0.2824,+0.1776,+0.1376,+0.3290]` | `[+0.3753,+0.2012,+0.1064,+0.3720]` | `[0.2810,0.0810,0.1762,0.1528]` |
| Right little | `[+0.1006,+0.0852,+0.1269,+0.0485]` | `[+0.1333,+0.0919,+0.1086,+0.0534]` | `[0.0988,0.0528,0.1150,0.0803]` |

## Per-joint numerical statistics

`q start`, `q grasp`, and `Δq` are episode-balanced means. `action mean` and
`action std` are means of each episode's full-trajectory statistics. Min/max are
global extrema across all frames for that side. `d_action` is the corresponding
component of the per-finger normalized median action closing vector.

### Left hand (30 episodes)

| Joint | q start | q grasp | Δq | action mean | action std | action min | action max | d_action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1J1 | +0.6417 | +1.1464 | +0.5048 | +0.8511 | 0.2468 | +0.3265 | +1.5204 | +0.7870 |
| F1J2 | -0.0889 | -0.0527 | +0.0362 | -0.0909 | 0.1120 | -0.1659 | +0.2113 | +0.2671 |
| F1J3 | +1.1848 | +0.7697 | -0.4151 | +0.9904 | 0.1525 | +0.2132 | +1.3915 | -0.5199 |
| F1J4 | +0.0000 | +0.0000 | +0.0000 | -0.4442 | 0.0736 | -0.4932 | +0.0789 | +0.1974 |
| F2J1 | +0.5869 | +0.6651 | +0.0782 | +0.5665 | 0.1853 | +0.0580 | +1.0182 | +0.2636 |
| F2J2 | +0.0456 | +0.0992 | +0.0537 | +0.0435 | 0.0452 | -0.1562 | +0.3039 | +0.0645 |
| F2J3 | +0.4751 | +0.8959 | +0.4208 | +0.6559 | 0.2645 | +0.0188 | +1.2364 | +0.9003 |
| F2J4 | -0.2759 | -0.0740 | +0.2019 | -0.2260 | 0.1080 | -0.4930 | +0.2183 | +0.3404 |
| F3J1 | +0.5495 | +0.7748 | +0.2253 | +0.5388 | 0.2536 | -0.0106 | +1.0757 | +0.6253 |
| F3J2 | +0.1855 | +0.2039 | +0.0184 | +0.1852 | 0.0300 | +0.0165 | +0.3240 | +0.0041 |
| F3J3 | +0.4947 | +0.7274 | +0.2327 | +0.6191 | 0.2043 | -0.4907 | +1.1935 | +0.5214 |
| F3J4 | +0.2965 | +0.5699 | +0.2734 | +0.4050 | 0.1322 | +0.1909 | +0.9886 | +0.5807 |
| F4J1 | +0.4798 | +0.7474 | +0.2676 | +0.4988 | 0.2278 | -0.0806 | +1.0172 | +0.5341 |
| F4J2 | +0.1927 | +0.2095 | +0.0168 | +0.1925 | 0.0308 | -0.0207 | +0.2809 | +0.0234 |
| F4J3 | +0.5979 | +1.0365 | +0.4385 | +0.7642 | 0.3144 | -0.3553 | +1.5814 | +0.8288 |
| F4J4 | +0.0931 | +0.1788 | +0.0857 | +0.1269 | 0.0887 | -0.2131 | +0.6457 | +0.1653 |
| F5J1 | +0.0265 | +0.2096 | +0.1831 | +0.0750 | 0.1188 | -0.3207 | +0.4337 | +0.3994 |
| F5J2 | +0.3538 | +0.3638 | +0.0100 | +0.2651 | 0.0430 | +0.0120 | +0.4109 | +0.1404 |
| F5J3 | +0.4010 | +0.8093 | +0.4083 | +0.5248 | 0.2789 | -0.1105 | +1.5916 | +0.8746 |
| F5J4 | +0.3239 | +0.4593 | +0.1353 | +0.3796 | 0.0804 | -0.1565 | +1.0833 | +0.2365 |

The left F1J4 state is fixed at zero while its commanded action changes. It must not
be used as clean state supervision until the recorder/controller mapping is confirmed.

### Right hand (30 episodes)

| Joint | q start | q grasp | Δq | action mean | action std | action min | action max | d_action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1J1 | +0.1639 | +0.6815 | +0.5176 | +0.5091 | 0.2587 | -0.0448 | +1.1205 | +0.9340 |
| F1J2 | -0.0038 | -0.1692 | -0.1655 | -0.1116 | 0.0763 | -0.1659 | +0.2635 | -0.2583 |
| F1J3 | +1.5066 | +1.3537 | -0.1530 | +1.4717 | 0.0816 | +1.2051 | +1.6272 | -0.2467 |
| F1J4 | -0.4898 | -0.4939 | -0.0041 | -0.4928 | 0.0013 | -0.4932 | -0.4589 | -0.0000 |
| F2J1 | +0.9509 | +1.0173 | +0.0664 | +0.8605 | 0.1997 | +0.3607 | +1.2531 | +0.3222 |
| F2J2 | +0.0554 | +0.1116 | +0.0563 | +0.0728 | 0.0771 | -0.1087 | +0.3021 | +0.2351 |
| F2J3 | +0.5058 | +0.8794 | +0.3736 | +0.6361 | 0.1887 | +0.2362 | +1.1829 | +0.7880 |
| F2J4 | -0.2832 | -0.0668 | +0.2164 | -0.2343 | 0.1214 | -0.4413 | +0.1135 | +0.4690 |
| F3J1 | +0.9912 | +1.2646 | +0.2734 | +0.9383 | 0.2770 | +0.2210 | +1.5261 | +0.5825 |
| F3J2 | -0.1914 | -0.0546 | +0.1368 | -0.1553 | 0.1006 | -0.3358 | +0.0707 | +0.3218 |
| F3J3 | +0.3237 | +0.4276 | +0.1039 | +0.3495 | 0.0972 | +0.0393 | +1.1116 | +0.0809 |
| F3J4 | +0.2819 | +0.7441 | +0.4622 | +0.4127 | 0.2143 | +0.1003 | +0.9743 | +0.7420 |
| F4J1 | +0.7854 | +1.0678 | +0.2824 | +0.7702 | 0.2252 | +0.2192 | +1.2350 | +0.6510 |
| F4J2 | -0.2204 | -0.0428 | +0.1776 | -0.1685 | 0.0862 | -0.2989 | +0.0797 | +0.3547 |
| F4J3 | +0.4337 | +0.5713 | +0.1376 | +0.4765 | 0.0954 | +0.0541 | +1.2766 | +0.1851 |
| F4J4 | +0.4409 | +0.7699 | +0.3290 | +0.5298 | 0.1558 | +0.2894 | +0.9771 | +0.6451 |
| F5J1 | +0.3232 | +0.4238 | +0.1006 | +0.3350 | 0.0665 | +0.1221 | +0.6890 | +0.6404 |
| F5J2 | -0.3230 | -0.2378 | +0.0852 | -0.3029 | 0.0521 | -0.4802 | -0.1666 | +0.4497 |
| F5J3 | +0.8458 | +0.9727 | +0.1269 | +0.5705 | 0.0736 | +0.2084 | +0.9648 | +0.5644 |
| F5J4 | +0.4589 | +0.5074 | +0.0485 | +0.4558 | 0.0431 | +0.2781 | +0.6688 | +0.2630 |

## Fixed-ratio coupling and signs

The first uncentered SVD component energy measures how well one four-joint direction
explains each finger's episode deltas. Median cosine measures directional consistency.
These are computed from Parquet values only.

| Side/finger | State energy | State median cosine | Action energy | Action median cosine |
|---|---:|---:|---:|---:|
| Left thumb | 96.1% | 0.988 | 95.5% | 0.984 |
| Left finger2 | 84.5% | 0.950 | 88.0% | 0.971 |
| Left middle | 76.8% | 0.911 | 80.2% | 0.930 |
| Left ring | 92.2% | 0.959 | 92.0% | 0.956 |
| Left little | 94.3% | 0.983 | 93.9% | 0.981 |
| Right thumb | 93.6% | 0.973 | 95.1% | 0.981 |
| Right finger2 | 81.0% | 0.979 | 84.3% | 0.985 |
| Right middle | 83.3% | 0.978 | 83.6% | 0.982 |
| Right ring | 86.9% | 0.970 | 87.0% | 0.970 |
| Right little | 81.3% | 0.908 | 80.6% | 0.915 |

Conclusion: each finger is approximately, but not exactly, one-dimensional within a
given side. The fixed ratio is strongest for thumbs and several left fingers; middle
and right little retain meaningful residual variation. A single left/right-shared
ratio is inappropriate.

Action signs are unambiguous at the median:

- Left thumb: J1/J2/J4 positive, J3 negative. Left F2–F5 are positive where the
  median motion is material; middle J2 and ring J2 are near zero.
- Right thumb: J1 positive, J2/J3 negative, J4 near zero. Every material F2–F5
  component is positive.
- Thumb is clearly different from the four opposing fingers and is also strongly
  side-dependent.

## Finger onset and staged closing

Median onset offsets from full-hand onset are:

| Side | Thumb | Finger2 | Middle | Ring | Little |
|---|---:|---:|---:|---:|---:|
| Left | 0.100 s | 0.050 s | 0.017 s | 0.067 s | 0.217 s |
| Right | 0.033 s | 0.033 s | 0.100 s | 0.100 s | 0.133 s |

Strict first-mover counts are left `9/7/10/1/3` and right `13/7/3/0/7` for
thumb/F2/middle/ring/little. Across all 60, counts are `22/14/13/1/10`.
When ties within one 30 Hz frame are allowed, Finger2 is co-first most often (32/60),
then middle (26), thumb (23), little (21), and ring (13).

This is not fully synchronous. Per-episode onset spread has median 0.617 s; only
23.3% of episodes keep all detected finger onsets within 0.20 s, while 65.0% exceed
0.30 s. There is therefore clear staged closing in many trajectories, but no single
universal stage order. Left often starts with middle/thumb/F2; right often starts with
thumb/F2/little.

## PCA/SVD of the 20D closing delta

Primary PCA is centered PCA of the requested state `Δq`. Action-delta PCA is included
as a robustness check and as the safer intended-command source for action-prior design.

| Group/source | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 | PC9 | PC10 | dims 80/90/95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| All state | 61.35 | 20.36 | 5.64 | 3.99 | 2.17 | 1.83 | 1.20 | 0.70 | 0.59 | 0.53 | 2 / 4 / 6 |
| All action | 61.63 | 20.04 | 6.33 | 3.55 | 2.26 | 1.50 | 1.04 | 0.73 | 0.62 | 0.52 | 2 / 4 / 6 |
| Left state | 80.10 | 6.55 | 5.63 | 2.52 | 1.99 | 0.85 | 0.66 | 0.52 | 0.33 | 0.23 | 1 / 3 / 5 |
| Left action | 80.90 | 5.87 | 5.40 | 2.49 | 1.91 | 0.89 | 0.70 | 0.59 | 0.36 | 0.28 | 1 / 3 / 5 |
| Right state | 69.28 | 15.77 | 4.64 | 3.77 | 1.73 | 1.27 | 1.14 | 0.68 | 0.45 | 0.42 | 2 / 4 / 5 |
| Right action | 70.70 | 15.25 | 4.24 | 3.49 | 1.57 | 1.30 | 1.13 | 0.62 | 0.46 | 0.40 | 2 / 3 / 5 |

Numbers are percentages. PC signs are arbitrary mathematically; the report fixes each
sign deterministically for reproducibility. The pooled state PC1–PC5 vectors are:

| PC/finger | J1 | J2 | J3 | J4 |
|---|---:|---:|---:|---:|
| PC1 thumb | +0.2764 | +0.0866 | -0.1321 | +0.0008 |
| PC1 finger2 | +0.2496 | +0.0668 | +0.2739 | +0.0717 |
| PC1 middle | +0.3415 | +0.0470 | +0.3200 | -0.0001 |
| PC1 ring | +0.3247 | +0.0450 | +0.4711 | -0.0806 |
| PC1 little | +0.1582 | +0.0095 | +0.4066 | +0.0651 |
| PC2 thumb | +0.1073 | -0.0733 | +0.0652 | +0.0006 |
| PC2 finger2 | +0.3105 | +0.0791 | +0.0090 | +0.1340 |
| PC2 middle | +0.4797 | +0.1470 | -0.2878 | +0.2913 |
| PC2 ring | +0.3180 | +0.1635 | -0.3668 | +0.3308 |
| PC2 little | +0.0025 | +0.0824 | -0.2447 | -0.0151 |
| PC3 thumb | +0.2575 | -0.4489 | +0.4814 | -0.0094 |
| PC3 finger2 | -0.1531 | -0.0378 | +0.2984 | -0.1247 |
| PC3 middle | -0.1461 | +0.1231 | +0.3756 | +0.1722 |
| PC3 ring | -0.0810 | +0.2174 | +0.1106 | +0.1807 |
| PC3 little | -0.1375 | +0.1038 | -0.1825 | -0.0439 |
| PC4 thumb | +0.5876 | -0.0349 | -0.4127 | -0.0007 |
| PC4 finger2 | -0.1634 | +0.0580 | -0.1287 | +0.0538 |
| PC4 middle | -0.3195 | +0.0646 | -0.0955 | +0.2170 |
| PC4 ring | -0.1507 | -0.0395 | -0.1235 | +0.3588 |
| PC4 little | +0.1097 | +0.0953 | +0.2497 | +0.1399 |
| PC5 thumb | -0.0688 | +0.2540 | -0.2940 | +0.0026 |
| PC5 finger2 | +0.0502 | -0.0118 | +0.4454 | +0.0625 |
| PC5 middle | +0.0121 | -0.1264 | +0.0743 | +0.2183 |
| PC5 ring | -0.1955 | -0.1335 | +0.1464 | +0.0757 |
| PC5 little | -0.2327 | -0.0531 | -0.4631 | +0.4642 |

The machine-readable JSON contains all 20 PCs for pooled, left, and right state and
action deltas. Uncentered SVD gives one dominant raw closing direction: PC1 energy is
73.2% state / 74.3% action pooled, 83.2% / 84.7% left, and 82.0% / 83.8% right.

## Current scripted direction versus expert action

The current structured environment uses:

```text
thumb:   [+1,+1,+1,+1]
F2-F5:   [+1, 0,+1,+1]
```

After per-finger L2 normalization, cosine similarity to median expert action delta is:

| Side | Thumb | Finger2 | Middle | Ring | Little |
|---|---:|---:|---:|---:|---:|
| Left | 0.366 | 0.868 | 0.997 | 0.882 | 0.872 |
| Right | 0.215 | 0.912 | 0.811 | 0.855 | 0.847 |

Concrete errors:

- Thumb J3 has the wrong sign on both sides. Right thumb J2 also has the wrong sign.
- Right thumb J4 is effectively stationary, yet scripted closing drives it positive.
- Scripted F2–F5 J2 is always zero. Expert left F2/Little J2 and all right F2–F5 J2
  move materially positive; left middle/ring J2 are near zero.
- Equal weights are poor for left F2 (J3 dominates), left ring/little (J3 dominates),
  right middle (J1/J4 dominate), and right little (J4 is much smaller).
- Thumb is by far the largest mismatch. Among opposing fingers, left F2/little and
  right middle/little differ most from the scripted coupling.

## Five representative trajectories and replay files

Five medoids were selected from action-delta direction, magnitude, and closing duration,
with three left and two right clusters. This favors representative modes rather than
only the single trajectory nearest the global mean.

| Episode | Side | Frames | Task |
|---:|---|---:|---|
| 69 | left | 330 | sponge block left |
| 75 | left | 332 | sponge block left |
| 87 | left | 306 | sponge block left |
| 118 | right | 297 | sponge block right |
| 119 | right | 320 | sponge block right |

Each NPZ contains bit-exact float32 `q_hand`, `a_hand`, and `timestamp`, plus frame,
episode, task, side, joint names, phase indices, and dataset revision. All five exports
passed element-wise equality checks against their source Parquet arrays.

## Manifold conclusion and SAC action-prior recommendation

Yes: real Wuji cube-grasp deltas are clearly low-dimensional relative to 20D, but not
one-dimensional. Side-specific state/action results agree: roughly 1–2 centered PCs
explain 80%, 3–4 explain 90%, and 5 explain about 95%. The pooled 6D result is partly
the cost of mixing distinct left/right conventions.

For the current left-hand MuJoCo experiment, use expert **action** deltas as intended
commands and state deltas only for execution validation. A clean next action-prior
ablation is:

```text
Δq = alpha * d_mean_left + B_left[0:4]^T * z + lambda * residual_20D
```

where `d_mean_left` is the normalized median expert close direction, `B_left` is the
centered action-delta PCA basis, and `alpha,z` are bounded SAC outputs. Start without
the residual for attribution; later use a small `lambda` (for example 0.05–0.10).
This provides a 5D demonstration-derived whole-hand latent instead of five hand-written
finger scalars. If retaining the 5D per-finger interface for the immediate ablation,
at minimum replace every per-finger direction with the side-specific expert median
ratios above and keep independent amplitudes because staged onset is real.

Before MuJoCo replay, explicitly map hardware order, sign, zero offset, mirroring, and
joint limits. Do not copy absolute encoder positions directly into simulation.

## Reproduce

```powershell
./.venvs/lerobot-policy/Scripts/python.exe scripts/analyze_wuji_cube_teleop.py
```

Outputs:

- `outputs/wuji_teleop_analysis/cube_hand_analysis.json`: full machine-readable
  statistics, all per-episode metrics, PCA bases, explained variance, ratios, hashes.
- `outputs/wuji_teleop_analysis/action_prior_summary.json`: compact mean/median closing
  vectors, PCA bases, explained variance, per-finger ratios, and selected episode IDs.
- `outputs/wuji_teleop_analysis/cube_60_hand_trajectories.parquet`: every frame of all
  60 hand-only state/action trajectories.
- `outputs/wuji_teleop_analysis/cube_episode_metrics.parquet`: one row per episode.
- `outputs/wuji_teleop_analysis/selected_expert_trajectories/*.npz`: five replay files.

This analysis does not change SAC, Reward V2, success gates, Lift, ACT, or MuJoCo.
