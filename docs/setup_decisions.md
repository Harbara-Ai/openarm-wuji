# Setup decisions

## Selected target

Use WSL2 with Ubuntu 22.04 as the primary simulation/training environment, subject to user approval for the system-level installation. It is the closest available option to the upstream Linux workflows and is more suitable for the later robot-client split than native Windows.

Until approved, only dependency-free project code and tests run under the bundled Python. System Python is not modified.

The WSL installation was approved in-app and attempted twice, but the Codex process lacks a true Windows administrator token; Windows made no change. The user/IT must run the documented administrator command and reboot if requested.

## Isolation

Three environments remain separate: `openarm-sim`, `wuji-retarget`, and `lerobot-policy`. Python 3.11 is selected for ecosystem compatibility. Exact transitive locks and repository commit hashes will be generated only after official repositories are cloned and installations resolve successfully.

## Upstream verification

Official sources checked on 2026-09-01: OpenArm repositories, Wuji retargeting, and LeRobot documentation. Wuji requires recursive submodules; LeRobot async inference uses the `async` extra; ACT is included in base LeRobot. No blog installation recipe is used.
