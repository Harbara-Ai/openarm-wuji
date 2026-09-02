# Milestone 2 — Wuji Hand retargeting

Status: passed on 2026-09-01 under WSL2 Ubuntu 22.04.

- Upstream: `wuji-technology/wuji-retargeting` commit `531f6ed4250b475d2e9231f54e988fc9b1c5b4ea`, cloned recursively with LFS sample data.
- Variant: original Wuji Hand, left-hand configuration; not Wuji Hand 2.
- Config: official `example/config/adaptive_analytical_avp.yaml`.
- Input: all 2751 frames of official `example/data/avp1.pkl`, each shaped `21×3`.
- Output: `2751×20`, all finite; maximum adjacent-frame absolute joint change 0.334680 rad.
- Runtime: 12.99 seconds / 211.82 frames per second in batch mode.
- MuJoCo model: `hand/body/mjcf/left.xml`, `nq=20`, `nv=20`, `nu=20`; optimizer and actuator orders match exactly.
- Evidence: `outputs/wuji/retarget_report.json`, `outputs/wuji/avp1_left_retarget.npz`, and `outputs/wuji/wuji_retarget_smoke.png`.
- Reproduce: `scripts/run_wuji_retarget_wsl.ps1 -WslPython /path/to/wuji-retarget/bin/python -WujiSource /path/to/wuji-retargeting`.

The Linux-native source copy should live on the WSL ext4 filesystem because editable builds on `/mnt/d` may fail atomic metadata writes under DrvFS. The authoritative upstream commit is recorded in `docs/upstream_versions.md`; local vendor checkouts are not committed.
