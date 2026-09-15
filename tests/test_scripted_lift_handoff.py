import unittest

import numpy as np

from openarm_wuji.tasks.scripted_lift_handoff import (
    cliffs_delta,
    compose_controller_target,
    load_bearing_success,
)


class ScriptedLiftHandoffTest(unittest.TestCase):
    def test_compose_controller_target_preserves_exact_20d_hand_target(self):
        arm = np.linspace(-0.3, 0.3, 7)
        hand = np.linspace(-1.0, 1.0, 20)
        result = compose_controller_target(arm, hand)
        self.assertEqual(result.shape, (27,))
        np.testing.assert_array_equal(result[:7], arm)
        np.testing.assert_array_equal(result[7:], hand)

    def test_compose_controller_target_rejects_synergy_sized_hand_input(self):
        with self.assertRaisesRegex(ValueError, "20-D"):
            compose_controller_target(np.zeros(7), np.zeros(3))

    def test_load_bearing_success_requires_full_unsupported_hold(self):
        self.assertTrue(load_bearing_success(
            trajectory_complete=True,
            terminal_heights_m=np.full(30, 0.081),
            terminal_environment_support=np.zeros(30, dtype=bool),
            required_hold_frames=30,
            success_height_m=0.08,
            dropped=False,
        ))
        self.assertFalse(load_bearing_success(
            trajectory_complete=True,
            terminal_heights_m=np.full(29, 0.081),
            terminal_environment_support=np.zeros(29, dtype=bool),
            required_hold_frames=30,
            success_height_m=0.08,
            dropped=False,
        ))

    def test_lift_gate_does_not_use_finger_count_or_topology(self):
        # The load-bearing hard gate intentionally has no contact-count input.
        self.assertTrue(load_bearing_success(
            trajectory_complete=True,
            terminal_heights_m=np.full(30, 0.09),
            terminal_environment_support=np.zeros(30, dtype=bool),
            required_hold_frames=30,
            success_height_m=0.08,
            dropped=False,
        ))

    def test_load_bearing_failure_modes(self):
        for failure in ("support", "height", "drop", "trajectory"):
            with self.subTest(failure=failure):
                support = np.zeros(30, dtype=bool)
                heights = np.full(30, 0.09)
                dropped = False
                complete = True
                if failure == "support":
                    support[-1] = True
                elif failure == "height":
                    heights[-1] = 0.079
                elif failure == "drop":
                    dropped = True
                elif failure == "trajectory":
                    complete = False
                self.assertFalse(load_bearing_success(
                    trajectory_complete=complete,
                    terminal_heights_m=heights,
                    terminal_environment_support=support,
                    required_hold_frames=30,
                    success_height_m=0.08,
                    dropped=dropped,
                ))

    def test_cliffs_delta_uses_success_minus_failure_sign(self):
        self.assertEqual(cliffs_delta([2.0, 3.0], [0.0, 1.0]), 1.0)
        self.assertEqual(cliffs_delta([0.0], [1.0]), -1.0)
        self.assertIsNone(cliffs_delta([], [1.0]))


if __name__ == "__main__":
    unittest.main()
