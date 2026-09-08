from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..simulation.mujoco_backend import MujocoOpenArmWuji
from ..tasks.contact_grasp import discover_fingertips
from ..tasks.reach_grasp_lift import ReachGraspLiftTask
from ..tasks.se3 import (
    pose_drift,
    quaternion_to_matrix,
    relative_pose,
)


FINGERS = tuple(f"finger{index}" for index in range(1, 6))


def _canonical_quaternion(value) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("zero quaternion in simulation state")
    result /= norm
    if result[0] < 0.0:
        result *= -1.0
    return result


def _clip_unit(value) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), -1.0, 1.0)


def coverage_potential(distances, *, sigma_m: float,
                       mean_weight: float, third_weight: float
                       ) -> tuple[float, float]:
    """Return multi-finger coverage potential and third-closest distance."""
    values = np.sort(np.asarray(tuple(distances), dtype=float))
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("coverage requires five finite finger distances")
    if sigma_m <= 0.0 or mean_weight < 0.0 or third_weight < 0.0:
        raise ValueError("coverage scales and weights must be non-negative")
    if not np.isclose(mean_weight + third_weight, 1.0):
        raise ValueError("coverage weights must sum to one")
    potentials = np.exp(-np.maximum(values, 0.0) / sigma_m)
    third_distance = float(values[2])
    potential = mean_weight * float(np.mean(potentials)) + (
        third_weight * float(potentials[2])
    )
    return potential, third_distance


def bounded_slip_penalty(linear_speed: float, angular_speed: float, *,
                         linear_scale: float, angular_scale: float) -> float:
    """Smooth slip cost in [0, 1], before the multi-contact ramp."""
    if min(linear_speed, angular_speed) < 0.0:
        raise ValueError("relative speeds must be non-negative")
    if min(linear_scale, angular_scale) <= 0.0:
        raise ValueError("slip scales must be positive")
    penalty = 0.5 * (
        1.0 - np.exp(-(linear_speed / linear_scale) ** 2)
        + 1.0 - np.exp(-(angular_speed / angular_scale) ** 2)
    )
    return float(np.clip(penalty, 0.0, 1.0))


def contact_interior_score(edge_margins, *, sigma_m: float,
                           target_fingers: int) -> float:
    """Finger-level face-interior quality in [0, 1].

    ``edge_margins`` contains one already-aggregated margin per contacting
    finger, so multiple collision geoms cannot multiply the reward. Dividing
    by the target finger count caps a lone finger at 1/target_fingers.
    """
    values = np.asarray(tuple(edge_margins), dtype=float)
    if sigma_m <= 0.0 or target_fingers < 1:
        raise ValueError("invalid contact-interior scale/target")
    if values.size == 0:
        return 0.0
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("edge margins must be finite and non-negative")
    quality = np.sum(1.0 - np.exp(-values / sigma_m)) / target_fingers
    return float(np.clip(quality, 0.0, 1.0))


def bounded_contact_slip_penalty(tangential_speeds, *, sigma_m_s: float) -> float:
    """Mean finger-level tangential slip penalty in [0, 1]."""
    values = np.asarray(tuple(tangential_speeds), dtype=float)
    if sigma_m_s <= 0.0:
        raise ValueError("contact-slip scale must be positive")
    if values.size == 0:
        return 0.0
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("tangential speeds must be finite and non-negative")
    penalty = np.mean(1.0 - np.exp(-np.square(values / sigma_m_s)))
    return float(np.clip(penalty, 0.0, 1.0))


def contact_persistence_score(durations_s, *, target_duration_s: float,
                              target_fingers: int) -> float:
    """Multi-finger persistence score in [0, 1], resistant to one-finger farming."""
    values = np.asarray(tuple(durations_s), dtype=float)
    if target_duration_s <= 0.0 or target_fingers < 1:
        raise ValueError("invalid persistence duration/target")
    if values.size == 0:
        return 0.0
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("contact durations must be finite and non-negative")
    score = np.sum(np.clip(values / target_duration_s, 0.0, 1.0)) / target_fingers
    return float(np.clip(score, 0.0, 1.0))


def contact_transition_terms(contact_count: int, previous_count: int, *,
                             target_count: int, new_contact_bonus: float,
                             two_contact_bonus: float,
                             three_contact_bonus: float) -> dict[str, float]:
    """One-step bonuses for acquiring new contact groups and milestones."""
    delta = max(int(contact_count) - int(previous_count), 0)
    return {
        "r_new_contact": float(new_contact_bonus) * delta,
        "bonus_two_contact": float(two_contact_bonus) * float(
            previous_count < 2 <= contact_count
        ),
        "bonus_three_contact": float(three_contact_bonus) * float(
            previous_count < target_count <= contact_count
        ),
    }


def multicontact_ramp(elapsed_s: float, *, grace_s: float,
                      ramp_s: float) -> float:
    """Gate stability, hold, and slip after a multi-contact grace period."""
    if elapsed_s < 0.0 or grace_s < 0.0 or ramp_s <= 0.0:
        raise ValueError("elapsed/grace must be non-negative and ramp positive")
    return float(np.clip((elapsed_s - grace_s) / ramp_s, 0.0, 1.0))


class WujiStaticGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """Stage-1 static grasp with a fixed palm and configurable hand actions.

    This is deliberately a privileged-state MLP environment.  It neither
    renders nor exposes images, timestamps, task telemetry dictionaries, or the
    old three-dimensional synergy action.
    """

    metadata = {"render_modes": []}
    HAND_JOINT_DIM = 20
    ACTION_DIM = 20

    def __init__(self, config: dict[str, Any], *, project_root: str | Path):
        super().__init__()
        self.config = deepcopy(config)
        self.root = Path(project_root).resolve()
        self._validate_config()
        self.control_hz = float(config["control_hz"])
        self.control_dt = 1.0 / self.control_hz
        self.max_episode_steps = int(config["max_episode_steps"])
        latch_cfg = config.get("post_contact_latch", {})
        self.post_contact_latch_enabled = bool(latch_cfg.get("enabled", False))
        self.post_contact_latch_min_fingers = int(
            latch_cfg.get("min_contact_fingers", 3)
        )
        self.post_contact_latch_window_s = float(
            latch_cfg.get("continuous_window_s", 0.10)
        )
        if self.post_contact_latch_min_fingers < 1:
            raise ValueError("post-contact latch finger threshold must be positive")
        if self.post_contact_latch_window_s <= 0.0:
            raise ValueError("post-contact latch window must be positive")

        self.robot = MujocoOpenArmWuji(
            self._path(config["model_path"]),
            self._path(config["synergy_config"]),
            arm_side="left",
            control_hz=self.control_hz,
            render=False,
        )
        self.robot.connect()
        task_config = json.loads(
            self._path(config["task_config"]).read_text(encoding="utf-8")
        )
        self.task = ReachGraspLiftTask(self.robot, task_config)
        self.model = self.robot.model
        self.data = self.robot.data

        self._setup_model_ids(task_config)
        self._setup_action_representation()
        self.observation_layout = self._make_observation_layout()
        self.observation_dim = self.observation_layout[-1][2]
        self.action_space = spaces.Box(-1.0, 1.0, (self.ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(
            -1.0, 1.0, (self.observation_dim,), np.float32
        )

        self._step_count = 0
        self._previous_action = np.zeros(self.ACTION_DIM)
        self._filtered_action = np.zeros(self.ACTION_DIM)
        self._previous_joint_action = np.zeros(self.HAND_JOINT_DIM)
        self._hand_target = self.robot.mapper.open_pose.copy()
        self._locked_arm_qpos = np.zeros(7)
        self._initial_cube_position = np.zeros(3)
        self._contact_anchor: tuple[np.ndarray, np.ndarray] | None = None
        self._contact_established_step: int | None = None
        hold_steps = max(2, int(np.ceil(
            float(config["success"]["sustain_s"]) * self.control_hz
        )))
        self._stability_window: deque[dict[str, Any]] = deque(maxlen=hold_steps)
        self._max_establishment_translation_m = 0.0
        self._max_establishment_rotation_deg = 0.0
        self._previous_coverage_potential = 0.0
        self._previous_contact_count = 0
        self._multicontact_reward_start_step: int | None = None
        self._post_contact_latched = False
        self._latched_q_target = self._hand_target.copy()
        self._post_contact_gate_frames = 0
        self._contact_established_gate_step: int | None = None
        self._contact_count_at_established: int | None = None
        self._finger_contact_frames = {finger: 0 for finger in FINGERS}
        self._closed = False

    @classmethod
    def from_json(cls, path: str | Path, *, project_root: str | Path | None = None):
        path = Path(path).resolve()
        if project_root is None:
            project_root = path.parents[2] if path.parent.name == "rl" else path.parent
        return cls(json.loads(path.read_text(encoding="utf-8")), project_root=project_root)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _validate_config(self) -> None:
        required = {
            "model_path", "synergy_config", "task_config", "control_hz",
            "max_episode_steps", "reset", "action", "observation", "reward",
            "success", "failure", "learning_milestones",
            "external_rigid_diagnostic", "reward_version",
        }
        missing = required - self.config.keys()
        if missing:
            raise KeyError(f"RL grasp config missing keys: {sorted(missing)}")
        if float(self.config["control_hz"]) <= 0.0:
            raise ValueError("control_hz must be positive")
        if int(self.config["max_episode_steps"]) < 1:
            raise ValueError("max_episode_steps must be positive")
        if self.config["reward_version"] not in {"v2", "v3"}:
            raise ValueError("WujiStaticGraspEnv requires reward_version v2 or v3")
        reward = self.config["reward"]
        if float(reward["multicontact_ramp_s"]) <= 0.0:
            raise ValueError("multicontact_ramp_s must be positive")
        if not np.isclose(
            float(reward["coverage_mean_weight"])
            + float(reward["coverage_third_weight"]), 1.0
        ):
            raise ValueError("coverage weights must sum to one")
        if self.config["reward_version"] == "v3":
            required_v3 = {
                "contact_interior_weight", "contact_persistence_weight",
                "contact_slip_weight", "edge_margin_sigma_m",
                "contact_slip_sigma_m_s", "persistence_target_s",
                "interior_margin_threshold_m",
            }
            missing_v3 = required_v3 - reward.keys()
            if missing_v3:
                raise KeyError(f"Reward V3 config missing keys: {sorted(missing_v3)}")
            if min(
                float(reward["edge_margin_sigma_m"]),
                float(reward["contact_slip_sigma_m_s"]),
                float(reward["persistence_target_s"]),
                float(reward["interior_margin_threshold_m"]),
            ) <= 0.0:
                raise ValueError("Reward V3 quality scales must be positive")
            if min(
                float(reward["contact_interior_weight"]),
                float(reward["contact_persistence_weight"]),
                float(reward["contact_slip_weight"]),
            ) < 0.0:
                raise ValueError("Reward V3 quality weights must be non-negative")
        representation = self.config["action"].get(
            "representation", "independent_joint"
        )
        if representation not in {
            "independent_joint", "structured_per_finger", "expert_pca5_absolute"
        }:
            raise ValueError(f"unknown action representation: {representation}")

    def _setup_model_ids(self, task_config: dict[str, Any]) -> None:
        import mujoco

        self.arm_qpos_ids = self.robot.arm_qpos_ids
        self.arm_qvel_ids = self.robot.arm_qvel_ids
        self.arm_actuator_ids = self.robot.arm_actuator_ids
        self.hand_qpos_ids = self.robot.hand_qpos_ids
        self.hand_qvel_ids = self.robot.hand_qvel_ids
        self.hand_actuator_ids = self.robot.hand_actuator_ids
        if len(self.hand_qpos_ids) != self.HAND_JOINT_DIM:
            raise ValueError(f"expected 20 hand joints, got {len(self.hand_qpos_ids)}")
        self.hand_joint_names = tuple(self.robot.mapper.joint_names)
        hand_joint_ids = np.asarray([
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"wuji_{name}"
            ) for name in self.hand_joint_names
        ], dtype=int)
        if np.any(hand_joint_ids < 0):
            raise KeyError("one or more configured Wuji joints are missing")
        joint_lower = self.model.jnt_range[hand_joint_ids, 0].copy()
        joint_upper = self.model.jnt_range[hand_joint_ids, 1].copy()
        ctrl_lower = self.model.actuator_ctrlrange[self.hand_actuator_ids, 0]
        ctrl_upper = self.model.actuator_ctrlrange[self.hand_actuator_ids, 1]
        self.hand_lower = np.maximum(joint_lower, ctrl_lower)
        self.hand_upper = np.minimum(joint_upper, ctrl_upper)
        self.hand_mid = 0.5 * (self.hand_lower + self.hand_upper)
        self.hand_half_range = np.maximum(
            0.5 * (self.hand_upper - self.hand_lower), 1e-6
        )

        self.palm_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE,
            task_config["scene"]["grasp_site_name"],
        )
        self.cube_body_id = self.task.cube_body_id
        cube_geoms = [
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) == self.cube_body_id
        ]
        cube_boxes = [
            geom for geom in cube_geoms
            if int(self.model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        ]
        if self.palm_site_id < 0 or not cube_boxes:
            raise KeyError("RL grasp scene requires the grasp site and cube box geom")
        self.cube_geom_id = cube_boxes[0]
        self.fingertips = discover_fingertips(
            self.model,
            palm_site_name=task_config["scene"]["grasp_site_name"],
            finger_body_regex=rf"wuji_{self.task.arm_side}_finger(\d+)_",
        )
        self.finger_geoms: dict[str, tuple[int, ...]] = {}
        self.pad_geom_ids: dict[str, int] = {}
        for finger, descriptor in self.fingertips.items():
            body_ids = set()
            cursor = descriptor.distal_body_id
            palm_body = int(self.model.site_bodyid[self.palm_site_id])
            while cursor > palm_body:
                body_ids.add(cursor)
                cursor = int(self.model.body_parentid[cursor])
            geoms = tuple(
                geom for geom in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom]) in body_ids
                and (int(self.model.geom_contype[geom]) != 0
                     or int(self.model.geom_conaffinity[geom]) != 0)
            )
            self.finger_geoms[finger] = geoms
            distal_non_tip = [
                geom for geom in geoms
                if int(self.model.geom_bodyid[geom]) == descriptor.distal_body_id
                and geom != descriptor.tip_geom_id
            ]
            self.pad_geom_ids[finger] = (
                max(distal_non_tip, key=lambda geom: float(np.linalg.norm(
                    self.model.geom_pos[geom]
                ))) if distal_non_tip else descriptor.tip_geom_id
            )

    def _setup_action_representation(self) -> None:
        action_cfg = self.config["action"]
        self.action_representation = action_cfg.get(
            "representation", "independent_joint"
        )
        if self.action_representation == "independent_joint":
            self.ACTION_DIM = self.HAND_JOINT_DIM
            self.structured_directions = np.eye(self.HAND_JOINT_DIM)
            self.structured_scales_rad = np.full(
                self.HAND_JOINT_DIM,
                float(action_cfg["delta_position_scale_rad"]),
            )
            self._joint_action_observation_directions = np.eye(
                self.HAND_JOINT_DIM
            )
            return

        if self.action_representation == "expert_pca5_absolute":
            prior_dir = Path(action_cfg["prior_dir"])
            if not prior_dir.is_absolute():
                prior_dir = self.root / prior_dir
            self.expert_pca5_mean = np.load(prior_dir / "mean.npy")
            self.expert_pca5_basis = np.load(prior_dir / "basis.npy")
            scaling = json.loads((prior_dir / "latent_scaling.json").read_text(
                encoding="utf-8"
            ))
            self.expert_pca5_latent_center = np.asarray(
                scaling["latent_center"], dtype=float
            )
            self.expert_pca5_latent_half_range = np.asarray(
                scaling["latent_half_range"], dtype=float
            )
            if (
                self.expert_pca5_mean.shape != (self.HAND_JOINT_DIM,)
                or self.expert_pca5_basis.shape != (self.HAND_JOINT_DIM, 5)
                or self.expert_pca5_latent_center.shape != (5,)
                or self.expert_pca5_latent_half_range.shape != (5,)
                or np.any(self.expert_pca5_latent_half_range <= 0.0)
            ):
                raise ValueError("expert_pca5 prior files have invalid shapes/scales")
            self.ACTION_DIM = 5
            self.structured_directions = self.expert_pca5_basis.T.copy()
            self.structured_scales_rad = self.expert_pca5_latent_half_range.copy()
            self._joint_action_observation_directions = np.zeros((5, 20))
            return

        if action_cfg.get("direction_source") != "scripted_sign_close_minus_open":
            raise ValueError(
                "structured actions require scripted_sign_close_minus_open"
            )
        raw = np.sign(
            self.robot.mapper.close_pose - self.robot.mapper.open_pose
        ).reshape(5, 4)
        norms = np.linalg.norm(raw, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("each structured finger direction must be non-zero")
        self.structured_directions = raw / norms[:, None]
        configured_scales = action_cfg.get("per_finger_delta_norm_scale_rad")
        if configured_scales is None:
            configured_scales = (
                float(action_cfg["delta_position_scale_rad"]) * norms
            )
        self.structured_scales_rad = np.asarray(configured_scales, dtype=float)
        if (self.structured_scales_rad.shape != (5,)
                or np.any(self.structured_scales_rad <= 0.0)):
            raise ValueError("structured per-finger scales must be five positive values")
        self._joint_action_observation_directions = raw
        self.ACTION_DIM = 5

    def _expand_policy_action(self, action: np.ndarray, *, radians: bool) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError("policy action has the wrong dimension")
        if self.action_representation == "expert_pca5_absolute":
            latent = self.expert_pca5_latent_center + (
                self.expert_pca5_latent_half_range * action
            )
            target = self.expert_pca5_mean + self.expert_pca5_basis @ latent
            return target if radians else _clip_unit(
                (target - self.hand_mid) / self.hand_half_range
            )
        if self.action_representation == "independent_joint":
            scale = self.structured_scales_rad if radians else 1.0
            return action * scale
        direction = (
            self.structured_directions * self.structured_scales_rad[:, None]
            if radians else self._joint_action_observation_directions
        )
        return (action[:, None] * direction).reshape(self.HAND_JOINT_DIM)

    @staticmethod
    def _make_observation_layout() -> tuple[tuple[str, int, int], ...]:
        fields = (
            ("hand_qpos", 20), ("hand_qvel", 20),
            ("hand_target", 20), ("previous_action", 20),
            ("palm_position", 3), ("palm_quaternion", 4),
            ("palm_linear_velocity", 3), ("palm_angular_velocity", 3),
            ("cube_position", 3), ("cube_quaternion", 4),
            ("cube_linear_velocity", 3), ("cube_angular_velocity", 3),
            ("cube_in_palm_position", 3), ("cube_in_palm_quaternion", 4),
            ("tip_and_pad_positions_relative_cube_and_palm", 60),
            ("per_finger_contact_features", 55),
        )
        result = []
        offset = 0
        for name, size in fields:
            result.append((name, offset, offset + size))
            offset += size
        return tuple(result)

    @property
    def action_mapping(self) -> tuple[str, ...]:
        if self.action_representation == "expert_pca5_absolute":
            return tuple(
                f"action[{index}] -> expert PCA5 absolute latent target z[{index}]"
                for index in range(5)
            )
        if self.action_representation == "structured_per_finger":
            return tuple(
                f"action[{index}] -> {finger} coordinated closing direction"
                for index, finger in enumerate(FINGERS)
            )
        return tuple(
            f"action[{index}] -> wuji_{name}_actuator delta target"
            for index, name in enumerate(self.hand_joint_names)
        )

    def _site_pose_velocity(self) -> tuple[np.ndarray, ...]:
        import mujoco

        palm_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(
            palm_quaternion, self.data.site_xmat[self.palm_site_id]
        )
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
            self.palm_site_id, velocity, 0,
        )
        return (
            self.data.site_xpos[self.palm_site_id].copy(),
            _canonical_quaternion(palm_quaternion),
            velocity[3:].copy(), velocity[:3].copy(),
        )

    def _cube_pose_velocity(self) -> tuple[np.ndarray, ...]:
        import mujoco

        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.cube_body_id, velocity, 0,
        )
        return (
            self.data.xpos[self.cube_body_id].copy(),
            _canonical_quaternion(self.data.xquat[self.cube_body_id]),
            velocity[3:].copy(), velocity[:3].copy(),
        )

    def _body_point_velocity_relative_cube(self, body_id: int,
                                           point_world: np.ndarray) -> np.ndarray:
        import mujoco

        finger_velocity = np.zeros(6)
        cube_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            int(body_id), finger_velocity, 0,
        )
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.cube_body_id, cube_velocity, 0,
        )
        finger_point_velocity = finger_velocity[3:] + np.cross(
            finger_velocity[:3], point_world - self.data.xpos[int(body_id)]
        )
        cube_point_velocity = cube_velocity[3:] + np.cross(
            cube_velocity[:3], point_world - self.data.xpos[self.cube_body_id]
        )
        return finger_point_velocity - cube_point_velocity

    def _body_point_speed_relative_cube(self, body_id: int,
                                        point_world: np.ndarray) -> float:
        return float(np.linalg.norm(self._body_point_velocity_relative_cube(
            body_id, point_world
        )))

    def _contact_quality(self, contacts: list[dict]) -> dict[str, Any]:
        """Aggregate valid contacts once per finger for Reward V3 telemetry."""
        cfg = self.config["reward"]
        sigma_edge = float(cfg.get("edge_margin_sigma_m", 0.003))
        sigma_slip = float(cfg.get("contact_slip_sigma_m_s", 0.010))
        persistence_target = float(cfg.get("persistence_target_s", 0.30))
        interior_threshold = float(cfg.get("interior_margin_threshold_m", 0.003))
        target_fingers = int(cfg["contact_target_fingers"])
        force_threshold = float(self.config["success"]["min_normal_force_n"])
        per_finger = {}
        margins = []
        tangential_speeds = []
        durations = []
        interior_count = 0
        for finger in FINGERS:
            items = [
                item for item in contacts
                if item["finger"] == finger
                and float(item["normal_force_n"]) >= force_threshold
            ]
            if not items:
                self._finger_contact_frames[finger] = 0
                per_finger[finger] = {
                    "valid_contact": False,
                    "contact_duration_s": 0.0,
                    "normal_force_n": 0.0,
                    "tangential_speed_m_s": 0.0,
                    "nearest_face": None,
                    "edge_margin_m": None,
                    "contact_region": "none",
                    "contact_geom": [],
                    "contact_point_world_m": [],
                    "interior_score": 0.0,
                    "slip_penalty": 0.0,
                }
                continue
            self._finger_contact_frames[finger] += 1
            forces = np.asarray([
                max(float(item["normal_force_n"]), 0.0) for item in items
            ])
            weights = forces / max(float(np.sum(forces)), 1e-12)
            item_margins = np.asarray([
                float(item["distance_to_nearest_edge_m"]) for item in items
            ])
            item_speeds = []
            for item in items:
                velocity = self._body_point_velocity_relative_cube(
                    int(item["other_body_id"]),
                    np.asarray(item["position_world_m"], dtype=float),
                )
                normal = np.asarray(item["normal_on_cube_world"], dtype=float)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                tangential = velocity - np.dot(velocity, normal) * normal
                item_speeds.append(float(np.linalg.norm(tangential)))
            edge_margin = float(np.sum(weights * item_margins))
            tangential_speed = float(np.sum(weights * np.asarray(item_speeds)))
            duration = self._finger_contact_frames[finger] * self.control_dt
            strongest = items[int(np.argmax(forces))]
            finger_interior = float(1.0 - np.exp(-edge_margin / sigma_edge))
            finger_slip = float(
                1.0 - np.exp(-(tangential_speed / sigma_slip) ** 2)
            )
            edge = bool(any(item["edge_contact"] for item in items))
            corner = bool(any(item["corner_contact"] for item in items))
            region = "corner" if corner else "edge" if edge else "face_interior"
            margins.append(edge_margin)
            tangential_speeds.append(tangential_speed)
            durations.append(duration)
            interior_count += int(edge_margin >= interior_threshold)
            per_finger[finger] = {
                "valid_contact": True,
                "contact_duration_s": float(duration),
                "normal_force_n": float(np.sum(forces)),
                "tangential_speed_m_s": tangential_speed,
                "nearest_face": strongest["cube_face"],
                "edge_margin_m": edge_margin,
                "contact_region": region,
                "contact_geom": sorted({
                    item.get("other_geom_name") or str(item["other_geom_id"])
                    for item in items
                }),
                "contact_point_world_m": [
                    np.asarray(item["position_world_m"], dtype=float).tolist()
                    for item in items
                ],
                "interior_score": finger_interior,
                "slip_penalty": finger_slip,
            }
        valid_count = len(margins)
        interior_reward = contact_interior_score(
            margins, sigma_m=sigma_edge, target_fingers=target_fingers
        )
        contact_slip = bounded_contact_slip_penalty(
            tangential_speeds, sigma_m_s=sigma_slip
        )
        persistence = contact_persistence_score(
            durations, target_duration_s=persistence_target,
            target_fingers=target_fingers,
        )
        return {
            "per_finger": per_finger,
            "valid_contact_count": valid_count,
            "interior_contact_count": interior_count,
            "interior_contact_fraction": (
                float(interior_count / valid_count) if valid_count else 0.0
            ),
            "mean_edge_margin_m": (
                float(np.mean(margins)) if margins else None
            ),
            "min_edge_margin_m": (
                float(np.min(margins)) if margins else None
            ),
            "mean_tangential_speed_m_s": (
                float(np.mean(tangential_speeds)) if tangential_speeds else 0.0
            ),
            "max_tangential_speed_m_s": (
                float(np.max(tangential_speeds)) if tangential_speeds else 0.0
            ),
            "persistence_score": persistence,
            "contact_interior_reward": interior_reward,
            "contact_slip_penalty": contact_slip,
        }

    def _contacts(self) -> tuple[list[dict], np.ndarray]:
        contacts = self.task.contact_monitor.contacts(0.0)
        scale = self.config["observation"]
        features = []
        for finger in FINGERS:
            items = [item for item in contacts if item["finger"] == finger]
            if not items:
                features.extend(np.zeros(11))
                continue
            forces = np.asarray([max(0.0, item["normal_force_n"]) for item in items])
            force_sum = float(np.sum(forces))
            weights = forces / force_sum if force_sum > 1e-9 else np.full(
                len(items), 1.0 / len(items)
            )
            normal = np.sum(np.asarray([
                item["normal_on_cube_cube"] for item in items
            ]) * weights[:, None], axis=0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1e-9:
                normal /= normal_norm
            tangential_force = 0.0
            relative_speed = 0.0
            for item in items:
                force = np.asarray(item["force_on_cube_cube_n"])
                contact_normal = np.asarray(item["normal_on_cube_cube"])
                tangential_force += float(np.linalg.norm(
                    force - np.dot(force, contact_normal) * contact_normal
                ))
                relative_speed = max(relative_speed,
                    self._body_point_speed_relative_cube(
                        item["other_body_id"], np.asarray(item["position_world_m"])
                    )
                )
            strongest = items[int(np.argmax(forces))]
            role = strongest["contact_role"]
            role_one_hot = [
                float(role == "designated_tip_geom"),
                float(role == "distal_non_tip_geom"),
                float(role == "other_finger_link_geom"),
            ]
            features.extend([
                1.0,
                min(len(items) / float(scale["contact_count_scale"]), 1.0),
                *_clip_unit(normal),
                min(force_sum / float(scale["normal_force_scale_n"]), 1.0),
                min(tangential_force / float(scale["tangential_force_scale_n"]), 1.0),
                min(relative_speed / float(
                    scale["relative_contact_speed_scale_m_s"]
                ), 1.0),
                *role_one_hot,
            ])
        return contacts, np.asarray(features, dtype=float)

    def _observation(self, contact_features: np.ndarray | None = None) -> np.ndarray:
        obs_cfg = self.config["observation"]
        palm_pos, palm_quat, palm_linear, palm_angular = self._site_pose_velocity()
        cube_pos, cube_quat, cube_linear, cube_angular = self._cube_pose_velocity()
        relative_pos, relative_quat = relative_pose(
            palm_pos, palm_quat, cube_pos, cube_quat
        )
        if contact_features is None:
            _, contact_features = self._contacts()
        qpos = self.data.qpos[self.hand_qpos_ids]
        qvel = self.data.qvel[self.hand_qvel_ids]
        world_center = np.asarray(obs_cfg["world_position_center_m"])
        world_scale = float(obs_cfg["world_position_scale_m"])
        relative_scale = float(obs_cfg["relative_position_scale_m"])
        tip_pad = []
        for finger in FINGERS:
            tip = self.fingertips[finger].tip_geom_id
            pad = self.pad_geom_ids[finger]
            tip_pad.extend(_clip_unit(
                (self.data.geom_xpos[tip] - cube_pos) / relative_scale
            ))
            tip_pad.extend(_clip_unit(
                (self.data.geom_xpos[pad] - cube_pos) / relative_scale
            ))
            tip_pad.extend(_clip_unit(
                (self.data.geom_xpos[tip] - palm_pos) / relative_scale
            ))
            tip_pad.extend(_clip_unit(
                (self.data.geom_xpos[pad] - palm_pos) / relative_scale
            ))
        observation = np.concatenate([
            _clip_unit((qpos - self.hand_mid) / self.hand_half_range),
            _clip_unit(qvel / float(obs_cfg["joint_velocity_scale_rad_s"])),
            _clip_unit((self._hand_target - self.hand_mid) / self.hand_half_range),
            _clip_unit(self._previous_joint_action),
            _clip_unit((palm_pos - world_center) / world_scale), palm_quat,
            _clip_unit(palm_linear / float(obs_cfg["linear_velocity_scale_m_s"])),
            _clip_unit(palm_angular / float(obs_cfg["angular_velocity_scale_rad_s"])),
            _clip_unit((cube_pos - world_center) / world_scale), cube_quat,
            _clip_unit(cube_linear / float(obs_cfg["linear_velocity_scale_m_s"])),
            _clip_unit(cube_angular / float(obs_cfg["angular_velocity_scale_rad_s"])),
            _clip_unit(relative_pos / relative_scale), relative_quat,
            np.asarray(tip_pad), contact_features,
        ]).astype(np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"observation layout mismatch: {observation.shape} != "
                f"{(self.observation_dim,)}"
            )
        if not np.isfinite(observation).all():
            raise FloatingPointError("non-finite RL observation")
        return observation

    def _finger_cube_distances(self) -> dict[str, float]:
        import mujoco

        result = {}
        from_to = np.zeros(6)
        for finger, geoms in self.finger_geoms.items():
            distances = [float(mujoco.mj_geomDistance(
                self.model, self.data, geom, self.cube_geom_id, 0.25, from_to
            )) for geom in geoms]
            result[finger] = min(distances) if distances else 0.25
        return result

    def _deepest_penetration(self) -> float:
        distances = self._finger_cube_distances()
        return max(0.0, -min(distances.values()))

    def _relative_state(self) -> dict[str, Any]:
        palm_pos, palm_quat, palm_linear, palm_angular = self._site_pose_velocity()
        cube_pos, cube_quat, cube_linear, cube_angular = self._cube_pose_velocity()
        relative_pos, relative_quat = relative_pose(
            palm_pos, palm_quat, cube_pos, cube_quat
        )
        delta_world = cube_pos - palm_pos
        relative_linear_world = (
            cube_linear - palm_linear - np.cross(palm_angular, delta_world)
        )
        relative_angular_world = cube_angular - palm_angular
        return {
            "position": relative_pos,
            "quaternion": relative_quat,
            "linear_speed": float(np.linalg.norm(relative_linear_world)),
            "angular_speed": float(np.linalg.norm(relative_angular_world)),
            "cube_position": cube_pos,
            "cube_linear_speed": float(np.linalg.norm(cube_linear)),
        }

    def _contact_group_count(self, contacts: list[dict]) -> int:
        threshold = float(self.config["success"]["min_normal_force_n"])
        return len({item["finger"] for item in contacts
                    if item["normal_force_n"] >= threshold})

    def _update_post_contact_latch(self, contact_count: int) -> None:
        """Latch the current absolute target after sustained valid contact.

        This trigger is intentionally separate from ``_update_success``:
        stable-grasp success still uses the configured formal criterion, while
        this ablation only tests whether continued target motion is the source
        of post-contact instability.
        """
        if not self.post_contact_latch_enabled or self._post_contact_latched:
            return
        required_frames = max(
            1, int(np.ceil(
                self.post_contact_latch_window_s * self.control_hz
            ))
        )
        if contact_count >= self.post_contact_latch_min_fingers:
            self._post_contact_gate_frames += 1
        else:
            self._post_contact_gate_frames = 0
        if self._post_contact_gate_frames >= required_frames:
            self._post_contact_latched = True
            self._contact_established_gate_step = self._step_count
            self._contact_count_at_established = int(contact_count)
            self._latched_q_target = self._hand_target.copy()

    def _reward(self, action: np.ndarray, relative: dict[str, Any],
                distances: dict[str, float], deepest_penetration: float,
                contact_count: int, success: bool,
                contact_quality: dict[str, Any],
                ) -> tuple[float, dict[str, float]]:
        cfg = self.config["reward"]
        phi_coverage, third_distance = coverage_potential(
            distances.values(), sigma_m=float(cfg["coverage_sigma_m"]),
            mean_weight=float(cfg["coverage_mean_weight"]),
            third_weight=float(cfg["coverage_third_weight"]),
        )
        approach_progress = phi_coverage - self._previous_coverage_potential
        transition = contact_transition_terms(
            contact_count, self._previous_contact_count,
            target_count=int(cfg["contact_target_fingers"]),
            new_contact_bonus=float(cfg["new_contact_bonus_per_finger"]),
            two_contact_bonus=float(cfg["two_contact_milestone_bonus"]),
            three_contact_bonus=float(cfg["three_contact_milestone_bonus"]),
        )
        contact_transition = sum(transition.values())
        contact_maintain = min(
            contact_count / float(cfg["contact_target_fingers"]), 1.0
        )
        minimum_multicontact = int(self.config["success"]["min_contact_fingers"])
        if contact_count >= minimum_multicontact:
            if (self._previous_contact_count < minimum_multicontact
                    or self._multicontact_reward_start_step is None):
                self._multicontact_reward_start_step = self._step_count
            elapsed = (
                self._step_count - self._multicontact_reward_start_step
            ) * self.control_dt
            beta = multicontact_ramp(
                elapsed,
                grace_s=float(cfg["multicontact_grace_s"]),
                ramp_s=float(cfg["multicontact_ramp_s"]),
            )
        else:
            self._multicontact_reward_start_step = None
            beta = 0.0
        stability_raw = 0.5 * (
            np.exp(-(relative["linear_speed"] /
                     float(cfg["stability_linear_scale_m_s"])) ** 2)
            + np.exp(-(relative["angular_speed"] /
                       float(cfg["stability_angular_scale_rad_s"])) ** 2)
        )
        hold_translation = hold_rotation_rad = 0.0
        if self._contact_anchor is not None:
            hold_translation, hold_rotation_deg = pose_drift(
                self._contact_anchor[0], self._contact_anchor[1],
                relative["position"], relative["quaternion"],
            )
            hold_rotation_rad = float(np.radians(hold_rotation_deg))
        hold_raw = 0.5 * (
            np.exp(-(hold_translation /
                     float(cfg["hold_translation_scale_m"])) ** 2)
            + np.exp(-(hold_rotation_rad /
                       float(cfg["hold_rotation_scale_rad"])) ** 2)
        )
        slip_raw = bounded_slip_penalty(
            relative["linear_speed"], relative["angular_speed"],
            linear_scale=float(cfg["slip_linear_scale_m_s"]),
            angular_scale=float(cfg["slip_angular_scale_rad_s"]),
        )
        penetration = (
            max(0.0, deepest_penetration - float(cfg["penetration_tolerance_m"]))
            / float(cfg["penetration_scale_m"])
        ) ** 2
        displacement = float(np.linalg.norm(
            relative["cube_position"] - self._initial_cube_position
        ))
        cube_motion = (
            max(0.0, displacement - float(cfg["cube_displacement_allowance_m"]))
            / float(cfg["cube_displacement_scale_m"])
            + max(0.0, relative["cube_linear_speed"] -
                  float(cfg["cube_speed_allowance_m_s"]))
            / float(cfg["cube_speed_scale_m_s"])
        )
        action_magnitude = float(np.mean(np.square(action)))
        action_rate = float(np.mean(np.square(action - self._previous_action)))
        weighted = {
            "approach_progress": (
                float(cfg["approach_progress_weight"]) * approach_progress
            ),
            "contact_transition": (
                float(cfg["contact_transition_weight"]) * contact_transition
            ),
            "contact_maintain": (
                float(cfg["contact_maintain_weight"]) * contact_maintain
            ),
            "stability": float(cfg["stability_weight"]) * beta * stability_raw,
            "hold": float(cfg["hold_region_weight"]) * beta * hold_raw,
            "slip": -float(cfg["slip_weight"]) * beta * slip_raw,
            "penetration": -float(cfg["penetration_weight"]) * penetration,
            "cube_motion": -float(cfg["cube_motion_weight"]) * cube_motion,
            "action": -float(cfg["action_magnitude_weight"]) * action_magnitude,
            "delta_action": -float(cfg["action_rate_weight"]) * action_rate,
            "success": float(cfg["success_bonus"]) * float(success),
        }
        if self.config["reward_version"] == "v3":
            weighted.update({
                "contact_interior": (
                    float(cfg["contact_interior_weight"])
                    * float(contact_quality["contact_interior_reward"])
                ),
                "contact_persistence": (
                    float(cfg["contact_persistence_weight"])
                    * float(contact_quality["persistence_score"])
                ),
                "contact_slip": (
                    -float(cfg["contact_slip_weight"])
                    * float(contact_quality["contact_slip_penalty"])
                ),
            })
        total_reward = float(sum(weighted.values()))
        terms = {
            "r_approach_progress": approach_progress,
            "phi_coverage": phi_coverage,
            "third_closest_finger_distance_m": third_distance,
            **transition,
            "r_contact_transition": contact_transition,
            "r_contact_maintain": contact_maintain,
            "r_stability_raw": stability_raw,
            "r_hold_raw": hold_raw,
            "p_slip_raw": slip_raw,
            "beta_multicontact": beta,
            "p_penetration": penetration,
            "p_cube_motion": cube_motion,
            "p_action": action_magnitude,
            "p_delta_action": action_rate,
            "r_contact_interior": float(
                contact_quality["contact_interior_reward"]
            ),
            "r_contact_persistence": float(
                contact_quality["persistence_score"]
            ),
            "p_contact_slip": float(
                contact_quality["contact_slip_penalty"]
            ),
            **{f"weighted_{name}": value for name, value in weighted.items()},
            "total_reward": total_reward,
        }
        self._previous_coverage_potential = phi_coverage
        self._previous_contact_count = contact_count
        terms = {name: float(value) for name, value in terms.items()}
        return total_reward, terms

    def _update_success(self, contact_count: int,
                        relative: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        cfg = self.config["success"]
        enough_contacts = contact_count >= int(cfg["min_contact_fingers"])
        if enough_contacts and self._contact_anchor is None:
            self._contact_anchor = (
                relative["position"].copy(), relative["quaternion"].copy()
            )
            self._contact_established_step = self._step_count
        if not enough_contacts:
            self._stability_window.clear()
        else:
            self._stability_window.append({
                "position": relative["position"].copy(),
                "quaternion": relative["quaternion"].copy(),
                "linear_speed": relative["linear_speed"],
                "angular_speed": relative["angular_speed"],
            })
        establishment_translation = establishment_rotation = 0.0
        if self._contact_anchor is not None:
            establishment_translation, establishment_rotation = pose_drift(
                self._contact_anchor[0], self._contact_anchor[1],
                relative["position"], relative["quaternion"],
            )
            self._max_establishment_translation_m = max(
                self._max_establishment_translation_m, establishment_translation
            )
            self._max_establishment_rotation_deg = max(
                self._max_establishment_rotation_deg, establishment_rotation
            )
        window_translation = window_rotation = float("inf")
        if self._stability_window:
            anchor = self._stability_window[0]
            drifts = [pose_drift(
                anchor["position"], anchor["quaternion"],
                item["position"], item["quaternion"],
            ) for item in self._stability_window]
            window_translation = max(value[0] for value in drifts)
            window_rotation = max(value[1] for value in drifts)
        grace_steps = int(np.ceil(float(cfg["settle_grace_s"]) * self.control_hz))
        grace_complete = (
            self._contact_established_step is not None
            and self._step_count - self._contact_established_step >= grace_steps
        )
        full_window = len(self._stability_window) == self._stability_window.maxlen
        velocities_ok = bool(full_window and all(
            item["linear_speed"] <= float(cfg["max_relative_linear_speed_m_s"])
            and np.degrees(item["angular_speed"])
            <= float(cfg["max_relative_angular_speed_deg_s"])
            for item in self._stability_window
        ))
        displacement = float(np.linalg.norm(
            relative["cube_position"] - self._initial_cube_position
        ))
        success = bool(
            enough_contacts and grace_complete and full_window and velocities_ok
            and window_translation <= float(cfg["max_window_translation_drift_m"])
            and window_rotation <= float(cfg["max_window_rotation_drift_deg"])
            and displacement <= float(cfg["max_cube_displacement_m"])
        )
        rigid = self.config["external_rigid_diagnostic"]
        rigid_success = bool(
            success
            and self._max_establishment_translation_m
            < float(rigid["max_relative_translation_drift_m"])
            and self._max_establishment_rotation_deg
            < float(rigid["max_relative_rotation_drift_deg"])
        )
        return success, {
            "contact_fingers": contact_count,
            "establishment_translation_drift_m": establishment_translation,
            "establishment_rotation_drift_deg": establishment_rotation,
            "max_establishment_translation_drift_m": self._max_establishment_translation_m,
            "max_establishment_rotation_drift_deg": self._max_establishment_rotation_deg,
            "window_translation_drift_m": window_translation,
            "window_rotation_drift_deg": window_rotation,
            "relative_linear_speed_m_s": relative["linear_speed"],
            "relative_angular_speed_deg_s": float(np.degrees(relative["angular_speed"])),
            "success_hold_frames": len(self._stability_window),
            "static_grasp_success": success,
            "rigid_success_diagnostic": rigid_success,
        }

    def _lock_arm_and_step(self) -> None:
        import mujoco

        physics_steps = max(1, int(round(
            self.control_dt / float(self.model.opt.timestep)
        )))
        self.data.ctrl[self.arm_actuator_ids] = self._locked_arm_qpos
        self.data.ctrl[self.hand_actuator_ids] = self._hand_target
        for _ in range(physics_steps):
            self.data.qpos[self.arm_qpos_ids] = self._locked_arm_qpos
            self.data.qvel[self.arm_qvel_ids] = 0.0
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.arm_qpos_ids] = self._locked_arm_qpos
        self.data.qvel[self.arm_qvel_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        reset_cfg = self.config["reset"]
        if seed is None:
            seed = int(self.np_random.integers(
                int(reset_cfg["seed_min"]), int(reset_cfg["seed_max"]) + 1
            ))
        reset_info = self.task.reset(int(seed))
        cube_before = self.task._object_position()
        target = cube_before + np.asarray(
            self.task.config["grasp"]["target_offset_m"], dtype=float
        ) + np.asarray(reset_cfg["pregrasp_extra_offset_m"], dtype=float)
        arm_goal, residual, orientation_residual = self.task._solve_arm_goal(
            target, phase_config=self.task.config["grasp"]
        )
        if (residual > float(self.task.config["grasp"]["ik_solve_tolerance_m"])
                or orientation_residual > float(
                    self.task.config["reach"]["ik_orientation_solve_tolerance_deg"]
                )):
            raise RuntimeError(
                f"pregrasp reset IK failed: {residual=:.6g}, "
                f"{orientation_residual=:.6g}"
            )
        self._locked_arm_qpos = arm_goal.copy()
        self.data.qpos[self.arm_qpos_ids] = self._locked_arm_qpos
        self.data.qvel[self.arm_qvel_ids] = 0.0
        self._hand_target = self.robot.mapper.open_pose.copy()
        self.data.qpos[self.hand_qpos_ids] = self._hand_target
        self.data.qvel[self.hand_qvel_ids] = 0.0
        self.data.ctrl[self.arm_actuator_ids] = self._locked_arm_qpos
        self.data.ctrl[self.hand_actuator_ids] = self._hand_target
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(reset_cfg["post_place_settle_steps"])):
            self.data.qpos[self.arm_qpos_ids] = self._locked_arm_qpos
            self.data.qvel[self.arm_qvel_ids] = 0.0
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[self.arm_qpos_ids] = self._locked_arm_qpos
        self.data.qvel[self.arm_qvel_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)
        cube_after = self.task._object_position()
        reset_displacement = float(np.linalg.norm(cube_after - cube_before))
        if reset_displacement > float(reset_cfg["max_reset_cube_displacement_m"]):
            raise RuntimeError(
                f"pregrasp reset displaced cube by {reset_displacement:.4f} m"
            )

        self._initial_cube_position = cube_after.copy()
        self._step_count = 0
        self._previous_action.fill(0.0)
        self._filtered_action.fill(0.0)
        self._previous_joint_action.fill(0.0)
        self._contact_anchor = None
        self._contact_established_step = None
        self._stability_window.clear()
        self._max_establishment_translation_m = 0.0
        self._max_establishment_rotation_deg = 0.0
        self._post_contact_latched = False
        self._latched_q_target = self._hand_target.copy()
        self._post_contact_gate_frames = 0
        self._contact_established_gate_step = None
        self._contact_count_at_established = None
        self._finger_contact_frames = {finger: 0 for finger in FINGERS}
        contacts, contact_features = self._contacts()
        initial_distances = self._finger_cube_distances()
        self._previous_coverage_potential, _ = coverage_potential(
            initial_distances.values(),
            sigma_m=float(self.config["reward"]["coverage_sigma_m"]),
            mean_weight=float(self.config["reward"]["coverage_mean_weight"]),
            third_weight=float(self.config["reward"]["coverage_third_weight"]),
        )
        self._previous_contact_count = self._contact_group_count(contacts)
        self._multicontact_reward_start_step = None
        observation = self._observation(contact_features)
        return observation, {
            "seed": int(seed),
            "reset_cube_position_m": cube_after.copy(),
            "reset_cube_displacement_m": reset_displacement,
            "pregrasp_ik_residual_m": float(residual),
            "pregrasp_orientation_residual_deg": float(orientation_residual),
            "initial_contact_fingers": self._contact_group_count(contacts),
            "post_contact_latched": False,
            "contact_established_time_s": None,
            "contact_count_at_established": None,
        }

    def _advance_with_target(
        self,
        desired_target: np.ndarray,
        reward_action: np.ndarray,
        *,
        apply_joint_rate_limit: bool,
    ):
        """Advance one simulator step toward an absolute 20D hand target.

        ``expert_pca5_absolute`` is an absolute posture controller, while the
        legacy representations pass a target obtained by adding a delta to the
        current posture.  Keeping the simulator/reward bookkeeping here makes
        the two semantics explicit and prevents PCA latent actions from being
        accidentally interpreted as joint deltas.
        """
        if self._closed:
            raise RuntimeError("cannot step a closed environment")
        desired_target = np.asarray(desired_target, dtype=float)
        reward_action = np.asarray(reward_action, dtype=float)
        if desired_target.shape != (self.HAND_JOINT_DIM,) or not np.isfinite(
            desired_target
        ).all():
            raise ValueError("desired_target must be a finite 20-D vector")
        if reward_action.shape != (self.ACTION_DIM,) or not np.isfinite(
            reward_action
        ).all():
            raise ValueError(
                f"reward_action must be a finite {self.ACTION_DIM}-D vector"
            )
        reward_action = np.clip(reward_action, -1.0, 1.0)
        requested_target = desired_target.copy()
        if self._post_contact_latched:
            desired_target = self._latched_q_target.copy()
        q_current = self.data.qpos[self.hand_qpos_ids].copy()
        if apply_joint_rate_limit:
            mapper_velocity = np.asarray(
                getattr(self.robot.mapper, "max_velocity", 0.0), dtype=float
            )
            if mapper_velocity.shape == (self.HAND_JOINT_DIM,) and np.all(
                mapper_velocity > 0.0
            ):
                max_delta = mapper_velocity * self.control_dt
            else:
                max_velocity = float(np.max(mapper_velocity)) if mapper_velocity.size else 0.0
                if max_velocity <= 0.0:
                    max_velocity = float(
                        self.config["action"].get("max_joint_velocity_rad_s", 8.0)
                    )
                max_delta = max_velocity * self.control_dt
            target_delta = np.clip(
                desired_target - q_current, -max_delta, max_delta
            )
            self._hand_target = np.clip(
                q_current + target_delta, self.hand_lower, self.hand_upper
            )
        else:
            self._hand_target = np.clip(
                desired_target, self.hand_lower, self.hand_upper
            )
        self._lock_arm_and_step()
        self._step_count += 1

        contacts, contact_features = self._contacts()
        relative = self._relative_state()
        contact_count = self._contact_group_count(contacts)
        contact_quality = self._contact_quality(contacts)
        self._update_post_contact_latch(contact_count)
        success, diagnostics = self._update_success(contact_count, relative)
        finger_distances = self._finger_cube_distances()
        deepest_penetration = max(0.0, -min(finger_distances.values()))
        reward, reward_terms = self._reward(
            reward_action, relative, finger_distances, deepest_penetration,
            contact_count, success, contact_quality
        )
        cube_displacement = float(np.linalg.norm(
            relative["cube_position"] - self._initial_cube_position
        ))
        failure_cfg = self.config["failure"]
        failure_reason = None
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            failure_reason = "non_finite_simulation"
        elif deepest_penetration > float(failure_cfg["max_penetration_m"]):
            failure_reason = "penetration_exploit"
        elif cube_displacement > float(failure_cfg["max_cube_displacement_m"]):
            failure_reason = "cube_excessive_motion"
        elif relative["cube_position"][2] < (
            self._initial_cube_position[2] - float(failure_cfg["drop_margin_m"])
        ):
            failure_reason = "drop"
        terminated = bool(success or failure_reason is not None)
        truncated = bool(
            not terminated and self._step_count >= self.max_episode_steps
        )
        observation = self._observation(contact_features)
        milestone_cfg = self.config["learning_milestones"]
        short_hold_frames = int(np.ceil(
            float(milestone_cfg["contact_hold_short_s"]) * self.control_hz
        ))
        long_hold_frames = int(np.ceil(
            float(milestone_cfg["contact_hold_long_s"]) * self.control_hz
        ))
        info = {
            **diagnostics,
            "failure_reason": failure_reason,
            "cube_displacement_m": cube_displacement,
            "deepest_finger_cube_penetration_m": deepest_penetration,
            "reward_terms": reward_terms,
            "contact_quality": contact_quality,
            "finger_distances_m": finger_distances,
            "multi_contact_acquired": bool(
                contact_count >= int(self.config["success"]["min_contact_fingers"])
            ),
            "three_contact_acquired": bool(
                contact_count >= int(self.config["reward"]["contact_target_fingers"])
            ),
            "contact_held_0p1s": bool(
                len(self._stability_window) >= short_hold_frames
            ),
            "contact_held_0p3s": bool(
                len(self._stability_window) >= long_hold_frames
            ),
            "applied_hand_target_rad": self._hand_target.copy(),
            "filtered_action": self._filtered_action.copy(),
            "pca_latent_action": (
                reward_action.copy()
                if self.action_representation == "expert_pca5_absolute"
                else None
            ),
            "desired_hand_target_rad": desired_target.copy(),
            "requested_hand_target_rad": requested_target,
            "action_representation": self.action_representation,
            "post_contact_latched": bool(self._post_contact_latched),
            "contact_established_time_s": (
                None if self._contact_established_gate_step is None
                else float(self._contact_established_gate_step * self.control_dt)
            ),
            "contact_count_at_established": self._contact_count_at_established,
            "latched_q_target": self._latched_q_target.copy(),
            "post_contact_gate_frames": int(self._post_contact_gate_frames),
            "expanded_joint_action": (
                _clip_unit((desired_target - self.hand_mid) / self.hand_half_range)
                if self.action_representation == "expert_pca5_absolute"
                else self._expand_policy_action(self._filtered_action, radians=False)
            ),
            "applied_joint_delta_rad": (
                self._hand_target - q_current
            ).copy(),
        }
        if self.action_representation == "expert_pca5_absolute":
            self._previous_joint_action = _clip_unit(
                (desired_target - self.hand_mid) / self.hand_half_range
            )
        else:
            self._previous_joint_action = self._expand_policy_action(
                reward_action, radians=False
            )
        self._previous_action = reward_action.copy()
        return observation, reward, terminated, truncated, info

    def step(self, action: np.ndarray):
        if self._closed:
            raise RuntimeError("cannot step a closed environment")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                f"action must be a finite {self.ACTION_DIM}-D vector"
            )
        action = np.clip(action, -1.0, 1.0)
        action_cfg = self.config["action"]
        max_change = float(action_cfg["max_action_change_per_step"])
        rate_limited = self._previous_action + np.clip(
            action - self._previous_action, -max_change, max_change
        )
        alpha = float(action_cfg["smoothing_alpha"])
        self._filtered_action = (
            alpha * rate_limited + (1.0 - alpha) * self._filtered_action
        )
        q_current = self.data.qpos[self.hand_qpos_ids].copy()
        if self.action_representation == "expert_pca5_absolute":
            desired_target = self._expand_policy_action(
                self._filtered_action, radians=True
            )
            return self._advance_with_target(
                desired_target, action, apply_joint_rate_limit=True
            )
        joint_delta_rad = self._expand_policy_action(
            self._filtered_action, radians=True
        )
        return self._advance_with_target(
            q_current + joint_delta_rad,
            action,
            apply_joint_rate_limit=False,
        )

    def step_absolute_target(
        self, target: np.ndarray, reward_action: np.ndarray | None = None
    ):
        """Replay a 20D absolute hand posture through the same reward path.

        This is intentionally a public deterministic-replay hook rather than a
        second environment.  It preserves the fixed-palm reset, PD stepping,
        rate limiting, contacts, termination, and Reward V2 bookkeeping.
        """
        target = np.asarray(target, dtype=float)
        if target.shape != (self.HAND_JOINT_DIM,) or not np.isfinite(target).all():
            raise ValueError("target must be a finite 20-D vector")
        if reward_action is None:
            reward_action = np.zeros(self.ACTION_DIM, dtype=float)
        reward_action = np.asarray(reward_action, dtype=float)
        if reward_action.shape != (self.ACTION_DIM,):
            raise ValueError(f"reward_action must have shape ({self.ACTION_DIM},)")
        reward_action = np.clip(reward_action, -1.0, 1.0)
        self._filtered_action = reward_action.copy()
        return self._advance_with_target(
            target, reward_action, apply_joint_rate_limit=True
        )

    def close(self) -> None:
        if not self._closed:
            self.robot.disconnect()
            self._closed = True
