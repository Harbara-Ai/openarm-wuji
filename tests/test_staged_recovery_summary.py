import unittest
from pathlib import Path

from scripts.evaluate_staged_with_recovery import _summarize


def _metrics(success, outcome, cube):
    return {
        "success": success,
        "outcome": outcome,
        "frames": 10,
        "timeout": outcome == "timeout",
        "maximum_cube_displacement_m": cube,
    }


class StagedRecoverySummaryTest(unittest.TestCase):
    def test_counts_matched_rescues_and_router_regressions(self):
        rows = [
            {
                "seed": 1400,
                "reach_success": False,
                "approach_attempted": False,
                "triggered": False,
                "baseline_success": False,
                "baseline_outcome": "reach_failure",
                "new_success": False,
                "new_outcome": "reach_failure",
            },
            {
                "seed": 1401,
                "reach_success": True,
                "approach_attempted": True,
                "triggered": True,
                "trigger": {
                    "primary_type": "moving_away",
                    "active_types": ["moving_away"],
                },
                "baseline_success": False,
                "baseline_outcome": "timeout",
                "baseline": _metrics(False, "timeout", 0.004),
                "approach_direct_success": False,
                "recovery_attempted": True,
                "recovery_success": True,
                "recovery": _metrics(True, "success", 0.005),
                "new_success": True,
                "new_outcome": "success",
                "new": _metrics(True, "success", 0.005),
            },
            {
                "seed": 1402,
                "reach_success": True,
                "approach_attempted": True,
                "triggered": True,
                "trigger": {
                    "primary_type": "gate_stall",
                    "active_types": ["gate_stall", "terminal_plateau"],
                },
                "baseline_success": True,
                "baseline_outcome": "success",
                "baseline": _metrics(True, "success", 0.003),
                "approach_direct_success": False,
                "recovery_attempted": True,
                "recovery_success": False,
                "recovery": _metrics(False, "timeout", 0.006),
                "new_success": False,
                "new_outcome": "timeout",
                "new": _metrics(False, "timeout", 0.006),
            },
            {
                "seed": 1403,
                "reach_success": True,
                "approach_attempted": True,
                "triggered": False,
                "trigger": None,
                "baseline_success": True,
                "baseline_outcome": "success",
                "baseline": _metrics(True, "success", 0.002),
                "approach_direct_success": True,
                "recovery_attempted": False,
                "recovery_success": False,
                "recovery": None,
                "new_success": True,
                "new_outcome": "success",
                "new": _metrics(True, "success", 0.002),
            },
        ]
        summary = _summarize(
            rows,
            requested=4,
            seed_start=1400,
            reach_checkpoint=Path("reach"),
            approach_checkpoint=Path("approach"),
            recovery_checkpoint=Path("recovery"),
        )
        self.assertEqual(summary["reach_successes"], 3)
        self.assertEqual(summary["recovery_attempts"], 2)
        self.assertEqual(summary["recovery_successes"], 1)
        self.assertEqual(summary["baseline"]["approach_successes"], 2)
        self.assertEqual(
            summary["staged_with_recovery"]["approach_stage_successes"], 2
        )
        self.assertEqual(summary["matched_switch_effect"]["rescued_failures"], 1)
        self.assertEqual(
            summary["matched_switch_effect"]["regressed_successes"], 1
        )


if __name__ == "__main__":
    unittest.main()
