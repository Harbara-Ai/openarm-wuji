import unittest

from scripts.build_staged_recovery_report import (
    _decision,
    _mcnemar_exact_p,
    _select,
)


def _unseen(*, recovery_rate, rescues, regressions,
            baseline_success=10, staged_success=10,
            baseline_safety=2, staged_safety=2):
    return {
        "recovery_success_rate": recovery_rate,
        "baseline": {
            "approach_successes": baseline_success,
            "cube_safety_failures": baseline_safety,
        },
        "staged_with_recovery": {
            "approach_stage_successes": staged_success,
            "cube_safety_failures": staged_safety,
        },
        "matched_switch_effect": {
            "rescued_failures": rescues,
            "regressed_successes": regressions,
        },
    }


class StagedRecoveryReportTest(unittest.TestCase):
    def test_checkpoint_selection_uses_requested_priority(self):
        rows = [
            {
                "step": 500, "recovery_successes": 25,
                "cube_safety_failures": 2, "timeouts": 2,
                "terminal_error_mm": {"median": 5.0},
            },
            {
                "step": 1000, "recovery_successes": 25,
                "cube_safety_failures": 1, "timeouts": 8,
                "terminal_error_mm": {"median": 6.0},
            },
            {
                "step": 1500, "recovery_successes": 24,
                "cube_safety_failures": 0, "timeouts": 0,
                "terminal_error_mm": {"median": 1.0},
            },
        ]
        self.assertEqual(_select(rows)["step"], 1000)

    def test_decision_cases(self):
        case, _ = _decision(
            {"recovery_success_rate": 0.7},
            _unseen(recovery_rate=0.9, rescues=10, regressions=0),
        )
        self.assertEqual(case, "Case C")
        case, _ = _decision(
            {"recovery_success_rate": 0.9},
            _unseen(recovery_rate=0.5, rescues=10, regressions=0),
        )
        self.assertEqual(case, "Case B")
        case, _ = _decision(
            {"recovery_success_rate": 0.9},
            _unseen(
                recovery_rate=0.8, rescues=2, regressions=3,
                baseline_success=10, staged_success=9,
            ),
        )
        self.assertEqual(case, "Case D")
        case, _ = _decision(
            {"recovery_success_rate": 0.9},
            _unseen(
                recovery_rate=0.8, rescues=10, regressions=0,
                baseline_success=10, staged_success=20,
                baseline_safety=2, staged_safety=1,
            ),
        )
        self.assertEqual(case, "Case A")
        self.assertLess(_mcnemar_exact_p(10, 0), 0.05)

        case, _ = _decision(
            {"recovery_success_rate": 0.63},
            _unseen(
                recovery_rate=0.60, rescues=20, regressions=43,
                baseline_success=119, staged_success=96,
                baseline_safety=17, staged_safety=4,
            ),
        )
        self.assertEqual(case, "Case D")


if __name__ == "__main__":
    unittest.main()
