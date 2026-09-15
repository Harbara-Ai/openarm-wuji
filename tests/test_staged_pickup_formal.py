import unittest
from pathlib import Path

from openarm_wuji.tasks.staged_pickup import (
    load_pipeline_manifest,
    phase_outcomes,
    resolve_pipeline_paths,
    select_representative_rollouts,
    summarize_staged_pickup,
    verify_artifact_integrity,
)


ROOT = Path(__file__).resolve().parents[1]


def row(*, seed, reach=True, approach=True, terminal=True,
        gate=True, safety=True, lift_attempted=True, lift=False,
        failure_stage="lift", failure_reason="drop", recovery=False):
    if lift:
        failure_stage = failure_reason = None
    return {
        "rollout_index": seed,
        "seed": seed,
        "reach_success": reach,
        "approach_stage_success": approach,
        "recovery_attempted": recovery,
        "recovery_success": False,
        "graspsecure_attempted": approach,
        "graspsecure_terminal_valid": terminal,
        "graspsecure_success": gate,
        "prelift_safety_pass": safety,
        "lift_attempted": lift_attempted,
        "lift_success": lift,
        "full_task_success": lift,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "maximum_cube_displacement_m": 0.04,
        "prelift_cube_motion_m": 0.01,
        "lift": ({
            "failure_reason": None if lift else failure_reason,
            "trajectory_complete": True,
        } if lift_attempted else None),
        "upstream": {},
    }


class StagedPickupFormalTest(unittest.TestCase):
    def test_manifest_is_portable_and_resolves_from_repo_root(self):
        manifest = load_pipeline_manifest(ROOT / "configs/staged_pipeline.json")
        paths = resolve_pipeline_paths(manifest, ROOT)
        self.assertEqual(paths["router"], ROOT / "configs/recovery_router.json")
        self.assertFalse(
            manifest["lift_admission"]["require_graspsecure_static_gate"]
        )
        if not all(path.exists() for path in paths.values()):
            self.skipTest("large frozen pipeline artifacts are not restored")
        integrity = verify_artifact_integrity(manifest, ROOT)
        self.assertEqual(len(integrity), 5)
        self.assertTrue(all(item["bytes_match"] for item in integrity))
        self.assertTrue(all(item["sha256_match"] for item in integrity))

    def test_summary_uses_terminal_validity_not_static_gate_for_lift(self):
        rows = [
            row(seed=1, gate=False, lift=True),
            row(seed=2, reach=False, approach=False, terminal=False,
                gate=False, safety=False, lift_attempted=False,
                failure_stage="reach", failure_reason="timeout"),
        ]
        summary = summarize_staged_pickup(
            rows, seed_start=1, requested_runs=2, pipeline_name="test"
        )
        self.assertEqual(summary["counts"]["graspsecure_terminal_valid"], 1)
        self.assertEqual(
            summary["counts"]["graspsecure_static_gate_pass_diagnostic"], 0
        )
        self.assertEqual(summary["counts"]["full_task_success"], 1)
        self.assertEqual(summary["probabilities"]["P_full_task_success"], 0.5)

    def test_phase_timeline_and_representative_selection(self):
        rows = [
            row(seed=1, lift=True),
            row(seed=2, reach=False, approach=False, terminal=False,
                gate=False, safety=False, lift_attempted=False,
                failure_stage="reach", failure_reason="timeout"),
            row(seed=3, lift=False),
        ]
        phases = [item["phase"] for item in phase_outcomes(rows[0])]
        self.assertEqual(
            phases,
            ["REACH", "APPROACH", "GRASP_SECURE", "LIFT", "HOLD", "SUCCESS"],
        )
        selected = select_representative_rollouts(rows)
        self.assertEqual(set(selected), {
            "success", "reach_or_approach_failure", "grasp_or_lift_failure"
        })

    def test_drop_does_not_claim_hold_was_reached(self):
        phases = phase_outcomes(row(seed=3, lift=False))
        by_name = {item["phase"]: item["status"] for item in phases}
        self.assertEqual(by_name["LIFT"], "complete")
        self.assertEqual(by_name["HOLD"], "failure")
        interrupted = row(seed=4, lift=False)
        interrupted["lift"]["trajectory_complete"] = False
        by_name = {
            item["phase"]: item["status"] for item in phase_outcomes(interrupted)
        }
        self.assertEqual(by_name["LIFT"], "failure")
        self.assertEqual(by_name["HOLD"], "not_reached")


if __name__ == "__main__":
    unittest.main()
