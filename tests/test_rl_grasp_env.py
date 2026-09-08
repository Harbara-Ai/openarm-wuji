from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

try:
    from openarm_wuji.rl.grasp_env import (
        WujiStaticGraspEnv,
        bounded_slip_penalty,
        bounded_contact_slip_penalty,
        contact_interior_score,
        contact_persistence_score,
        contact_transition_terms,
        coverage_potential,
        multicontact_ramp,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"gymnasium", "mujoco"}:
        raise
    WujiStaticGraspEnv = None


@unittest.skipIf(WujiStaticGraspEnv is None, "RL optional dependencies unavailable")
class TestWujiStaticGraspEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        model = cls.root / "outputs/reach_grasp_lift/reach_grasp_lift.mjb"
        if not model.exists():
            raise unittest.SkipTest("compiled Reach-Grasp-Lift model unavailable")
        cls.env = WujiStaticGraspEnv.from_json(
            cls.root / "configs/rl/grasp_stage1.json", project_root=cls.root
        )
        cls.structured_env = WujiStaticGraspEnv.from_json(
            cls.root / "configs/rl/grasp_stage1_structured5.json",
            project_root=cls.root,
        )
        cls.pca_env = WujiStaticGraspEnv.from_json(
            cls.root / "configs/rl/grasp_stage1_expert_pca5.json",
            project_root=cls.root,
        )
        cls.latch_env = WujiStaticGraspEnv.from_json(
            cls.root / "configs/rl/grasp_stage1_expert_pca5_contact_latch.json",
            project_root=cls.root,
        )
        cls.v3_env = WujiStaticGraspEnv.from_json(
            cls.root / "configs/rl/grasp_stage1_expert_pca5_reward_v3.json",
            project_root=cls.root,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "env"):
            cls.env.close()
        if hasattr(cls, "structured_env"):
            cls.structured_env.close()
        if hasattr(cls, "pca_env"):
            cls.pca_env.close()
        if hasattr(cls, "latch_env"):
            cls.latch_env.close()
        if hasattr(cls, "v3_env"):
            cls.v3_env.close()

    def test_observation_and_action_contract(self):
        observation, info = self.env.reset(seed=7)
        self.assertEqual(observation.shape, (228,))
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(self.env.action_space.shape, (20,))
        self.assertEqual(len(self.env.action_mapping), 20)
        forbidden = ("rgb", "image", "timestamp", "telemetry")
        for name, _, _ in self.env.observation_layout:
            self.assertFalse(any(token in name for token in forbidden))
        self.assertLess(info["reset_cube_displacement_m"], 0.02)

    def test_zero_action_preserves_fixed_arm(self):
        self.env.reset(seed=8)
        locked = self.env._locked_arm_qpos.copy()
        observation, reward, terminated, truncated, info = self.env.step(
            np.zeros(20, dtype=np.float32)
        )
        np.testing.assert_allclose(
            self.env.data.qpos[self.env.arm_qpos_ids], locked, atol=1e-12
        )
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info["failure_reason"])
        required_reward_terms = {
            "r_approach_progress", "phi_coverage",
            "third_closest_finger_distance_m", "r_new_contact",
            "bonus_two_contact", "bonus_three_contact",
            "r_contact_maintain", "r_stability_raw", "r_hold_raw",
            "p_slip_raw", "beta_multicontact", "p_penetration",
            "p_cube_motion", "p_action", "p_delta_action", "total_reward",
        }
        self.assertTrue(required_reward_terms <= info["reward_terms"].keys())
        self.assertNotIn("proximity", info["reward_terms"])
        self.assertNotIn("thumb_bonus", info["reward_terms"])
        self.assertAlmostEqual(reward, info["reward_terms"]["total_reward"])

    def test_one_action_index_targets_only_its_joint(self):
        self.env.reset(seed=9)
        q_current = self.env.data.qpos[self.env.hand_qpos_ids].copy()
        action = np.zeros(20, dtype=np.float32)
        action[6] = 1.0
        _, _, _, _, info = self.env.step(action)
        target_delta = info["applied_hand_target_rad"] - q_current
        changed = np.flatnonzero(np.abs(target_delta) > 1e-12)
        np.testing.assert_array_equal(changed, [6])
        self.assertIn("finger2_joint3", self.env.action_mapping[6])

    def test_structured_action_preserves_observation_contract(self):
        observation, _ = self.structured_env.reset(seed=7)
        self.assertEqual(observation.shape, (228,))
        self.assertEqual(self.structured_env.action_space.shape, (5,))
        self.assertEqual(len(self.structured_env.action_mapping), 5)
        np.testing.assert_allclose(
            np.linalg.norm(self.structured_env.structured_directions, axis=1),
            np.ones(5),
        )

    def test_structured_ablation_keeps_environment_and_reward_fixed(self):
        baseline = json.loads((
            self.root / "configs/rl/grasp_stage1.json"
        ).read_text(encoding="utf-8"))
        structured = json.loads((
            self.root / "configs/rl/grasp_stage1_structured5.json"
        ).read_text(encoding="utf-8"))
        frozen_fields = (
            "reward", "observation", "success", "reset", "failure",
            "external_rigid_diagnostic", "control_hz", "max_episode_steps",
        )
        for field in frozen_fields:
            self.assertEqual(baseline[field], structured[field], field)

    def test_structured_full_close_matches_scripted_probe_direction(self):
        expanded = self.structured_env._expand_policy_action(
            np.ones(5), radians=True
        )
        scripted = 0.035 * np.sign(
            self.structured_env.robot.mapper.close_pose
            - self.structured_env.robot.mapper.open_pose
        )
        np.testing.assert_allclose(expanded, scripted, atol=1e-10)

    def test_structured_scalar_only_commands_its_finger(self):
        for finger_index in range(5):
            self.structured_env.reset(seed=7)
            q_current = self.structured_env.data.qpos[
                self.structured_env.hand_qpos_ids
            ].copy()
            action = np.zeros(5, dtype=np.float32)
            action[finger_index] = 1.0
            _, _, _, _, info = self.structured_env.step(action)
            target_delta = info["applied_hand_target_rad"] - q_current
            outside = np.ones(20, dtype=bool)
            outside[4 * finger_index:4 * (finger_index + 1)] = False
            np.testing.assert_allclose(target_delta[outside], 0.0, atol=1e-12)
            self.assertGreater(
                np.max(np.abs(target_delta[~outside])), 0.0
            )

    def test_expert_pca5_absolute_action_contract(self):
        observation, _ = self.pca_env.reset(seed=7)
        self.assertEqual(observation.shape, (228,))
        self.assertEqual(self.pca_env.action_space.shape, (5,))
        self.assertEqual(len(self.pca_env.action_mapping), 5)
        action = np.asarray([1.0, -1.0, 0.25, 0.0, -0.5])
        expected = self.pca_env.expert_pca5_mean + self.pca_env.expert_pca5_basis @ (
            self.pca_env.expert_pca5_latent_center
            + self.pca_env.expert_pca5_latent_half_range * action
        )
        np.testing.assert_allclose(
            self.pca_env._expand_policy_action(action, radians=True), expected
        )

    def test_expert_pca5_target_is_rate_limited_not_teleported(self):
        self.pca_env.reset(seed=7)
        q_current = self.pca_env.data.qpos[self.pca_env.hand_qpos_ids].copy()
        target = self.pca_env.expert_pca5_mean.copy()
        target += 10.0
        _, _, _, _, info = self.pca_env.step_absolute_target(target)
        applied = info["applied_hand_target_rad"]
        self.assertTrue(np.all(applied <= self.pca_env.hand_upper + 1e-12))
        self.assertTrue(np.all(applied >= self.pca_env.hand_lower - 1e-12))
        self.assertLess(np.max(np.abs(applied - q_current)), 1.0)
        self.assertEqual(info["action_representation"], "expert_pca5_absolute")

    def test_expert_pca5_keeps_frozen_environment_fields(self):
        baseline = json.loads((
            self.root / "configs/rl/grasp_stage1.json"
        ).read_text(encoding="utf-8"))
        pca = json.loads((
            self.root / "configs/rl/grasp_stage1_expert_pca5.json"
        ).read_text(encoding="utf-8"))
        for field in (
            "reward", "observation", "success", "reset", "failure",
            "external_rigid_diagnostic", "control_hz", "max_episode_steps",
        ):
            self.assertEqual(baseline[field], pca[field], field)

    def test_reward_v3_only_changes_reward_configuration(self):
        v2 = json.loads((
            self.root / "configs/rl/grasp_stage1_expert_pca5.json"
        ).read_text(encoding="utf-8"))
        v3 = json.loads((
            self.root / "configs/rl/grasp_stage1_expert_pca5_reward_v3.json"
        ).read_text(encoding="utf-8"))
        for field in (
            "action", "observation", "success", "reset", "failure",
            "external_rigid_diagnostic", "control_hz", "max_episode_steps",
            "model_path", "synergy_config", "task_config",
        ):
            self.assertEqual(v2[field], v3[field], field)
        observation, _ = self.v3_env.reset(seed=7)
        self.assertEqual(observation.shape, (228,))
        self.assertEqual(self.v3_env.action_space.shape, (5,))
        observation, reward, _, _, info = self.v3_env.step(np.zeros(5))
        self.assertEqual(observation.shape, (228,))
        self.assertTrue(np.isfinite(reward))
        self.assertIn("contact_quality", info)
        self.assertIn("r_contact_interior", info["reward_terms"])
        self.assertIn("r_contact_persistence", info["reward_terms"])
        self.assertIn("p_contact_slip", info["reward_terms"])
        self.assertNotIn("contact_quality", [name for name, _, _ in self.v3_env.observation_layout])

    def test_contact_latch_uses_three_fingers_for_tenth_second(self):
        self.latch_env.reset(seed=7)
        open_pose = self.latch_env.robot.mapper.open_pose
        close_pose = self.latch_env.robot.mapper.close_pose
        latched_target = None
        for step in range(120):
            target = open_pose + min(1.0, step / 60.0) * (close_pose - open_pose)
            _, _, _, _, info = self.latch_env.step_absolute_target(target)
            if info["post_contact_latched"]:
                latched_target = info["latched_q_target"].copy()
                break
        self.assertIsNotNone(latched_target)
        for _ in range(3):
            target = open_pose.copy()
            _, _, _, _, info = self.latch_env.step_absolute_target(target)
            np.testing.assert_allclose(info["desired_hand_target_rad"], latched_target)
            self.assertEqual(info["contact_count_at_established"], 3)


class TestRewardV2Functions(unittest.TestCase):
    def test_coverage_uses_third_closest_finger(self):
        potential, third = coverage_potential(
            [0.005, 0.010, 0.025, 0.080, 0.090],
            sigma_m=0.025, mean_weight=0.3, third_weight=0.7,
        )
        expected_phi = np.exp(-np.asarray([0.005, 0.010, 0.025, 0.080, 0.090]) / 0.025)
        self.assertAlmostEqual(third, 0.025)
        self.assertAlmostEqual(
            potential, 0.3 * np.mean(expected_phi) + 0.7 * expected_phi[2]
        )

    def test_contact_bonuses_are_transition_only(self):
        acquired = contact_transition_terms(
            3, 1, target_count=3, new_contact_bonus=0.25,
            two_contact_bonus=0.5, three_contact_bonus=0.5,
        )
        maintained = contact_transition_terms(
            3, 3, target_count=3, new_contact_bonus=0.25,
            two_contact_bonus=0.5, three_contact_bonus=0.5,
        )
        self.assertEqual(acquired, {
            "r_new_contact": 0.5,
            "bonus_two_contact": 0.5,
            "bonus_three_contact": 0.5,
        })
        self.assertEqual(sum(maintained.values()), 0.0)

    def test_slip_penalty_is_bounded(self):
        self.assertEqual(bounded_slip_penalty(
            0.0, 0.0, linear_scale=0.04, angular_scale=0.70
        ), 0.0)
        large = bounded_slip_penalty(
            100.0, 100.0, linear_scale=0.04, angular_scale=0.70
        )
        self.assertGreaterEqual(large, 0.0)
        self.assertLessEqual(large, 1.0)

    def test_multicontact_grace_and_ramp(self):
        self.assertEqual(multicontact_ramp(
            0.05, grace_s=0.10, ramp_s=0.20
        ), 0.0)
        self.assertAlmostEqual(multicontact_ramp(
            0.20, grace_s=0.10, ramp_s=0.20
        ), 0.5)
        self.assertAlmostEqual(multicontact_ramp(
            0.30, grace_s=0.10, ramp_s=0.20
        ), 1.0)

    def test_contact_quality_terms_are_bounded(self):
        interior = contact_interior_score(
            [0.0, 0.003, 0.030], sigma_m=0.003, target_fingers=3
        )
        slip = bounded_contact_slip_penalty(
            [0.001, 0.010, 0.016], sigma_m_s=0.010
        )
        persistence = contact_persistence_score(
            [0.1, 0.3, 1.0], target_duration_s=0.3, target_fingers=3
        )
        for value in (interior, slip, persistence):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_one_finger_cannot_farm_persistence_or_interior(self):
        self.assertLessEqual(contact_persistence_score(
            [100.0], target_duration_s=0.3, target_fingers=3
        ), 1.0 / 3.0)
        self.assertLessEqual(contact_interior_score(
            [100.0], sigma_m=0.003, target_fingers=3
        ), 1.0 / 3.0)

    def test_no_contact_has_zero_quality_reward_and_penalty(self):
        self.assertEqual(contact_interior_score(
            [], sigma_m=0.003, target_fingers=3
        ), 0.0)
        self.assertEqual(contact_persistence_score(
            [], target_duration_s=0.3, target_fingers=3
        ), 0.0)
        self.assertEqual(bounded_contact_slip_penalty(
            [], sigma_m_s=0.010
        ), 0.0)


if __name__ == "__main__":
    unittest.main()
