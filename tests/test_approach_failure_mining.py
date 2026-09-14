import unittest

import numpy as np

from scripts.build_approach_correction_dataset import _deduplicate
from scripts.mine_approach_near_failures import _trigger_scores


class TriggerTest(unittest.TestCase):
    def test_terminal_plateau_and_gate_stall(self):
        triggers = _trigger_scores(
            errors=[0.0150, 0.0149, 0.0148, 0.0147, 0.0146, 0.0145, 0.0144],
            cube_displacement_m=0.0,
            gate_frames_total=4,
            gate_consecutive=0,
            frame=20,
            timeout_frames=90,
            status_success=False,
        )
        self.assertIn("terminal_plateau", triggers)
        self.assertIn("gate_stall", triggers)

    def test_moving_away_and_cube_displacement(self):
        triggers = _trigger_scores(
            errors=[0.019, 0.018, 0.021, 0.025],
            cube_displacement_m=0.004,
            gate_frames_total=0,
            gate_consecutive=0,
            frame=25,
            timeout_frames=90,
            status_success=False,
        )
        self.assertIn("moving_away", triggers)
        self.assertIn("cube_displacement", triggers)


class DeduplicationTest(unittest.TestCase):
    def test_filters_unsafe_and_prefers_unique_rollouts(self):
        candidates = []
        for index in range(6):
            candidates.append({
                "path": f"snapshot_{index}.npz",
                "rollout_index": index,
                "seed": 1000 + index,
                "source_frame": 20,
                "trigger_type": "moving_away" if index % 2 else "gate_stall",
                "source_outcome": "timeout" if index < 3 else "success",
                "trigger_score": 1.0,
                "terminal_error_vector_m": np.asarray(
                    [0.005 * (index + 1), -0.002, -0.010]
                ),
                "terminal_error_m": 0.015 + index * 0.001,
                "cube_displacement_vector_m": np.asarray([0.001 * index, 0.0, 0.0]),
                "cube_displacement_m": 0.001 * index,
                "arm_qpos": np.linspace(0.0, 0.1, 7) + index * 0.01,
            })
        candidates.append({**candidates[0], "rollout_index": 99,
                           "cube_displacement_m": 0.026})
        selected, summary = _deduplicate(candidates, target_count=5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(summary["excluded_unsafe_or_nonterminal"], 1)
        self.assertEqual(len({item["rollout_index"] for item in selected}), 5)


if __name__ == "__main__":
    unittest.main()
