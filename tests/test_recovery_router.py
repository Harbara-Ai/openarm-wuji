import unittest

from openarm_wuji.policy.recovery_router import (
    detect_near_failure,
    primary_trigger,
)


class RecoveryRouterTest(unittest.TestCase):
    def test_primary_trigger_uses_fixed_safety_first_priority(self):
        triggers = {
            "gate_stall": 12.0,
            "moving_away": 0.01,
            "cube_displacement": 0.004,
        }
        self.assertEqual(primary_trigger(triggers), "cube_displacement")
        self.assertIsNone(primary_trigger({}))

    def test_detector_retains_mining_thresholds(self):
        triggers = detect_near_failure(
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


if __name__ == "__main__":
    unittest.main()
