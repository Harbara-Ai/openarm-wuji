# Known gaps

- WSL2 Ubuntu 22.04, GPU passthrough, and WSLg are operational.
- Official repositories, Wuji submodules, and its LFS sample are cloned and commit hashes recorded.
- OpenArm headless simulation, programmatic movement, joint inventory, and screenshot are complete. Interactive GUI/keyboard operation still needs a user-visible session.
- Wuji retargeting is operational in WSL. Reproducibility requires ABI pins `cmeel-urdfdom==4.0.1` and `cmeel-tinyxml2==10.0.0` with `pin==3.8.0`; unconstrained latest cmeel packages are incompatible at runtime.
- The combined MJCF mount is a visual simulation estimate; its real tool transform still requires measurement on the lab hardware.
- The Reach phase is implemented, but physical grasp closure, contact labels, lift success, and failure taxonomy are not yet implemented.
- LeRobotDataset creation, ACT training/rollout, and official async inference remain pending; the isolated LeRobot contract environment itself is operational.
- No real-hardware behavior is claimed; the customized arm protocol and parameters are unknown by design.
