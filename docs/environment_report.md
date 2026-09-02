# Environment report

Generated: 2026-09-01 (Asia/Singapore)

- Host: native Windows 11-compatible build 26200 / display version 25H2 (registry product label reports Windows 10 Education).
- WSL follow-up (2026-09-01): Ubuntu 22.04 is installed and running as WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`.
- WSL GPU/GUI: NVIDIA RTX 2000 Ada, driver 610.88 and 16,380 MiB visible inside WSL; WSLg exposes `DISPLAY=:0` and `WAYLAND_DISPLAY=wayland-0`.
- WSL Python: system Python 3.10.12 remains clean; project environments live in a user-local `.venvs` directory.
- CPU: Intel Family 6 Model 183, 32 logical processors. Full marketing name was inaccessible without elevation.
- Available memory during inspection: about 18.4 GiB. Total memory was inaccessible without elevation.
- Disk free: C: about 766 GiB; D: about 1.79 TiB.
- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU, 16,380 MiB VRAM.
- Driver: 610.88; NVIDIA-SMI reports CUDA UMD capability 13.3. A PyTorch CUDA build is not installed yet.
- Python on PATH: absent. Codex bundled Python 3.12.13 exists and is used only for bootstrap tests.
- Conda/Mamba/uv: absent. Git 2.53.0 and Git LFS 3.7.1 are bundled. Docker, CMake, Ninja: absent.
- MuJoCo/OpenGL GUI: not yet verifiable; WSLg is unavailable because WSL is absent.
- Workspace: initially empty and not a Git repository.

The CUDA version displayed by `nvidia-smi` is driver capability, not proof that a matching CUDA toolkit or PyTorch build is installed.
