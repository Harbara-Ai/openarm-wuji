import json
from pathlib import Path
import unittest

import numpy as np

from openarm_wuji.tasks.contact_grasp import (
    ContactRegionGraspOptimizer,
    discover_fingertips,
    inward_face_direction,
    project_to_face_region,
)
from openarm_wuji.tasks.finger_path_planner import (
    FINGER2_PATH_OUTCOMES,
    Finger2PathPlanner,
)


class ContactRegionTests(unittest.TestCase):
    def test_acquisition_contract_is_explicit_and_policy_independent(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs/contact_grasp_optimization.json").read_text()
        )
        acquisition = config["acquisition"]
        for key in (
            "pregrasp_clearance_m", "contact_confirm_steps", "hold_s",
            "squeeze_s", "max_cube_translation_before_topology_m",
            "max_cube_rotation_before_topology_deg",
            "pregrasp_ik_iterations", "pregrasp_ik_tolerance_m",
            "pregrasp_ik_damping", "pregrasp_ik_max_step_rad",
            "pregrasp_retreat_attempts", "pregrasp_retreat_step_m",
        ):
            self.assertIn(key, acquisition)
        self.assertGreaterEqual(acquisition["pregrasp_clearance_m"], 0.004)
        self.assertLessEqual(acquisition["pregrasp_clearance_m"], 0.006)
        self.assertEqual(acquisition["default_strategy"], "synchronized")

        planning = config["finger2_path_planning"]
        self.assertEqual(planning["finger"], "finger2")
        self.assertEqual(planning["target_face"], "+X")
        self.assertEqual(planning["path_samples_per_segment"], 21)
        self.assertGreater(planning["soft_non_tip_clearance_m"], 0.0)
        self.assertLessEqual(planning["soft_non_tip_clearance_m"], 0.002)
        self.assertGreaterEqual(planning["dynamic_segment_duration_s"], 3.0)

    def test_finger2_path_outcomes_are_explicit(self):
        self.assertEqual(set(FINGER2_PATH_OUTCOMES), {
            "contact_region_endpoint_infeasible",
            "midpoint_optimization_failed",
            "path_non_tip_collision",
            "path_feasible",
            "single_finger_tip_first",
            "single_finger_non_tip_first",
            "single_finger_path_pass",
            "finger2_required_contact_rejected",
        })
    def test_projection_penalizes_plane_and_out_of_region_error(self):
        target, error = project_to_face_region(
            [-0.050, 0.030, -0.010], "-X", [0.035] * 3, [0.018, 0.018]
        )
        np.testing.assert_allclose(target, [-0.035, 0.018, -0.010])
        np.testing.assert_allclose(error, [-0.015, 0.012, 0.0])

    def test_projection_inside_region_only_has_normal_error(self):
        target, error = project_to_face_region(
            [0.050, 0.010, -0.015], "+X", [0.035] * 3, [0.018, 0.018]
        )
        np.testing.assert_allclose(target, [0.035, 0.010, -0.015])
        np.testing.assert_allclose(error, [0.015, 0.0, 0.0])

    def test_inward_direction_sign(self):
        np.testing.assert_array_equal(inward_face_direction("-X"), [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(inward_face_direction("+Y"), [0.0, -1.0, 0.0])


class FingertipDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.model_path = cls.root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb"

    def test_discovers_five_distal_chains_without_geom_names(self):
        if not self.model_path.exists():
            self.skipTest("compiled reach-grasp-lift model is unavailable")
        import mujoco

        model = mujoco.MjModel.from_binary_path(str(self.model_path))
        config = json.loads(
            (self.root / "configs/contact_grasp_optimization.json").read_text()
        )
        tips = discover_fingertips(
            model,
            palm_site_name=config["palm_site_name"],
            finger_body_regex=config["finger_body_regex"],
        )
        self.assertEqual(set(tips), {f"finger{i}" for i in range(1, 6)})
        for finger, descriptor in tips.items():
            self.assertIn(finger, descriptor.distal_body_name)
            self.assertEqual(len(descriptor.joint_ids), 4)
            self.assertTrue(
                model.geom_contype[descriptor.tip_geom_id] != 0
                or model.geom_conaffinity[descriptor.tip_geom_id] != 0
            )

    def test_topologies_use_small_named_initialization_sets(self):
        config = json.loads(
            (self.root / "configs/contact_grasp_optimization.json").read_text()
        )
        expected = {
            "thumb_minus_x_fingers_plus_x": (
                ["finger1", "finger2", "finger3", "finger4"], ["finger5"]
            ),
            "thumb_minus_y_fingers_plus_y": (
                ["finger1", "finger2", "finger3", "finger4"], ["finger5"]
            ),
            "thumb_middle_ring_x_opposition": (
                ["finger1", "finger3", "finger4"], ["finger2", "finger5"]
            ),
        }
        self.assertEqual(
            {item["name"] for item in config["topologies"]}, set(expected)
        )
        for topology in config["topologies"]:
            required, optional = expected[topology["name"]]
            self.assertEqual(topology["required_fingers"], required)
            self.assertEqual(topology["optional_fingers"], optional)
            initializations = topology["initializations"]
            self.assertLessEqual(len(initializations), 3)
            self.assertEqual(
                len({item["name"] for item in initializations}),
                len(initializations),
            )
            for item in initializations:
                self.assertIn(item["hand_seed"], {
                    "current", "opposition", "power_half"
                })

    def test_finger2_planner_uses_five_semantic_regions(self):
        if not self.model_path.exists():
            self.skipTest("compiled reach-grasp-lift model is unavailable")
        import mujoco

        model = mujoco.MjModel.from_binary_path(str(self.model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        config = json.loads(
            (self.root / "configs/contact_grasp_optimization.json").read_text()
        )
        topology = config["topologies"][0]
        optimizer = ContactRegionGraspOptimizer(
            model, data, config, topology=topology
        )
        synergy = json.loads(
            (self.root / "configs/wuji_hand_left_synergies.json").read_text()
        )
        planner = Finger2PathPlanner(
            optimizer, config, open_hand=synergy["open_pose"]
        )
        self.assertEqual(
            [name for name, _ in planner.region_seeds],
            ["center", "upper", "lower", "plus_y_side", "minus_y_side"],
        )
        self.assertEqual(len(planner.local_ids), 4)


if __name__ == "__main__":
    unittest.main()
