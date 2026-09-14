import unittest

from scripts.build_recovery_router_ablation_report import (
    _compact_closed_loop,
    _mcnemar_exact_p,
)


class RecoveryRouterAblationReportTest(unittest.TestCase):
    def test_compact_summary_preserves_matched_effect(self):
        raw = {
            "router": {"candidate_spec": {"name": "test"}},
            "reach_successes": 10,
            "approach_attempts": 8,
            "approach_direct_successes": 3,
            "recovery_attempts": 5,
            "recovery_successes": 4,
            "recovery_success_rate": 0.8,
            "mean_recovery_frames": 12.0,
            "baseline": {
                "approach_successes": 5,
                "approach_conditional_success_rate": 0.625,
                "timeouts": 2, "cube_safety_failures": 1,
                "maximum_cube_displacement_mm": {"median": 2.0, "p90": 9.0},
            },
            "staged_with_recovery": {
                "approach_stage_successes": 7,
                "approach_conditional_success_rate": 0.875,
                "timeouts": 1, "cube_safety_failures": 0,
                "maximum_cube_displacement_mm": {"median": 1.0, "p90": 4.0},
            },
            "matched_switch_effect": {
                "rescued_failures": 3, "regressed_successes": 1,
                "both_success": 1, "both_failure": 0,
            },
            "trigger_by_primary_type": {},
        }
        result = _compact_closed_loop("test", raw)
        self.assertEqual(result["net_benefit"], 2)
        self.assertEqual(result["candidate_successes"], 7)
        self.assertAlmostEqual(result["recovery_trigger_rate_given_reach"], 0.625)

    def test_exact_pair_test(self):
        self.assertLess(_mcnemar_exact_p(22, 5), 0.01)


if __name__ == "__main__":
    unittest.main()
