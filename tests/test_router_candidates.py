import unittest

from openarm_wuji.policy.router_candidates import candidate_fires


class RouterCandidateTest(unittest.TestCase):
    def test_patience_trigger_and_condition(self):
        spec = {
            "minimum_approach_frames": 10,
            "branches": [{
                "trigger_any": ["moving_away"],
                "conditions": [{
                    "feature": "moving_away_consecutive_frames",
                    "operator": ">=", "value": 5,
                }],
            }],
        }
        features = {
            "approach_elapsed_frames": 12,
            "trigger_active_types": ["moving_away"],
            "moving_away_consecutive_frames": 5,
        }
        self.assertTrue(candidate_fires(spec, features))
        features["approach_elapsed_frames"] = 9
        self.assertFalse(candidate_fires(spec, features))
        features["approach_elapsed_frames"] = 12
        features["moving_away_consecutive_frames"] = 4
        self.assertFalse(candidate_fires(spec, features))

    def test_any_branch_can_fire(self):
        spec = {
            "branches": [
                {"trigger_any": ["cube_displacement"]},
                {"trigger_any": ["gate_stall"]},
            ],
        }
        self.assertTrue(candidate_fires(spec, {
            "approach_elapsed_frames": 1,
            "trigger_active_types": ["gate_stall"],
        }))


if __name__ == "__main__":
    unittest.main()
