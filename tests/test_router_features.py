import unittest

from openarm_wuji.policy.router_features import extract_router_features


class RouterFeaturesTest(unittest.TestCase):
    def test_temporal_features_use_only_current_and_past(self):
        errors = [0.020, 0.018, 0.017, 0.018, 0.019]
        common = [0.0] * len(errors)
        features = extract_router_features(
            errors_m=errors,
            cube_displacements_m=common,
            cube_radial_velocities_m_s=common,
            arm_velocity_norms_rad_s=common,
            arm_velocity_max_abs_rad_s=common,
            grasp_velocity_norms_m_s=common,
            grasp_velocity_toward_target_m_s=common,
            gate_stall_flags=[False, False, False, True, True],
            gate_consecutive=0,
            gate_frames_total=4,
            orientation_error_deg=1.0,
            active_triggers={"moving_away": 0.002},
            frame=4,
            timeout_frames=90,
        )
        self.assertEqual(features["moving_away_consecutive_frames"], 2)
        self.assertEqual(features["plateau_duration_frames"], 2)
        self.assertEqual(features["gate_stall_duration_frames"], 2)
        self.assertAlmostEqual(features["recent_error_min_5_m"], 0.017)
        self.assertAlmostEqual(
            features["distance_from_recent_min_5_m"], 0.002
        )
        self.assertEqual(features["remaining_timeout_frames"], 85)
        self.assertEqual(features["trigger_primary_type"], "moving_away")

    def test_history_lengths_must_match(self):
        with self.assertRaises(ValueError):
            extract_router_features(
                errors_m=[0.02, 0.01], cube_displacements_m=[0.0],
                cube_radial_velocities_m_s=[0.0, 0.0],
                arm_velocity_norms_rad_s=[0.0, 0.0],
                arm_velocity_max_abs_rad_s=[0.0, 0.0],
                grasp_velocity_norms_m_s=[0.0, 0.0],
                grasp_velocity_toward_target_m_s=[0.0, 0.0],
                gate_stall_flags=[False, False], gate_consecutive=0,
                gate_frames_total=0, orientation_error_deg=0.0,
                active_triggers={}, frame=1, timeout_frames=90,
            )


if __name__ == "__main__":
    unittest.main()
