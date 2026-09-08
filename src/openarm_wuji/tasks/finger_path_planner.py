from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .contact_grasp import ContactRegionGraspOptimizer, _actuators_for_joints
from .se3 import pose_drift


FINGER2_PATH_OUTCOMES = (
    "contact_region_endpoint_infeasible",
    "midpoint_optimization_failed",
    "path_non_tip_collision",
    "path_feasible",
    "single_finger_tip_first",
    "single_finger_non_tip_first",
    "single_finger_path_pass",
    "finger2_required_contact_rejected",
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class FingerPathCandidate:
    name: str
    contact_region_center_cube_m: tuple[float, float]
    contact_region_half_extent_m: float
    outcome: str
    endpoint_feasible: bool
    endpoint_error_m: float
    contact_point_cube_m: tuple[float, ...]
    q_pre: tuple[float, ...]
    q_mid: tuple[float, ...]
    q_star: tuple[float, ...]
    midpoint_found: bool
    path_feasible: bool
    geom53_min_clearance_m: float | None
    geom55_min_clearance_m: float | None
    overall_non_tip_min_clearance_m: float | None
    min_clearance_segment: str | None
    min_clearance_alpha: float | None
    predicted_first_contact_geom: int | None
    endpoint_diagnostics: dict
    pregrasp_diagnostics: dict
    path_diagnostics: dict

    def to_dict(self) -> dict:
        return _plain(asdict(self))


class Finger2PathPlanner:
    """Finite contact-region and one-waypoint planner for finger2 only.

    Palm, cube, and every other finger remain fixed.  The planner searches five
    semantic regions on the cube +X face; each region is continuous within a
    small rectangle and is therefore an optimizer initialization, not a fixed
    contact point or a broad brute-force scan.
    """

    def __init__(self, optimizer: ContactRegionGraspOptimizer, config: dict,
                 *, open_hand):
        self.optimizer = optimizer
        self.model = optimizer.model
        self.data = optimizer.data
        self.mujoco = optimizer.mujoco
        self.config = config
        self.cfg = config["finger2_path_planning"]
        self.penetration_tolerance = float(
            config["optimization"]["max_kinematic_penetration_m"]
        )
        self.finger = self.cfg.get("finger", "finger2")
        if self.finger != "finger2":
            raise ValueError("the first path-aware planner is intentionally finger2-only")
        if self.cfg.get("target_face") != "+X":
            raise ValueError("finger2 path planner currently supports only the +X face")
        self.descriptor = optimizer.tips[self.finger]
        self.tip_geom_id = int(self.descriptor.tip_geom_id)
        self.finger_geoms = tuple(optimizer.finger_collision_geoms[self.finger])
        self.non_tip_geoms = tuple(
            geom for geom in self.finger_geoms if geom != self.tip_geom_id
        )
        self.finger_body_ids = {
            int(self.model.geom_bodyid[geom]) for geom in self.finger_geoms
        }
        self.hand_body_ids = {
            int(self.model.geom_bodyid[geom])
            for geoms in optimizer.finger_collision_geoms.values()
            for geom in geoms
        }
        self.arm_count = len(optimizer.arm_joint_ids)
        self.finger_index = 1
        start = self.arm_count + 4 * self.finger_index
        self.local_ids = np.arange(start, start + 4, dtype=int)
        self.joint_ids = np.asarray(self.descriptor.joint_ids, dtype=int)
        self.lower = optimizer.lower[self.local_ids]
        self.upper = optimizer.upper[self.local_ids]
        self.ranges = np.maximum(self.upper - self.lower, 1e-6)
        self.open_hand = np.asarray(open_hand, dtype=float)
        if self.open_hand.shape != (len(optimizer.hand_joint_ids),):
            raise ValueError("open_hand must contain the 20 independent hand joints")
        extent = np.asarray(config["target_region"]["tangential_half_extent_m"], dtype=float)
        offset = min(float(self.cfg["contact_region_seed_offset_m"]),
                     float(np.min(extent)) * 0.75)
        self.region_seeds = (
            ("center", (0.0, 0.0)),
            ("upper", (0.0, offset)),
            ("lower", (0.0, -offset)),
            ("plus_y_side", (offset, 0.0)),
            ("minus_y_side", (-offset, 0.0)),
        )

    def _set_q(self, q) -> None:
        self.optimizer._set_q(np.asarray(q, dtype=float))

    def _geom_distance(self, geom: int) -> tuple[float, np.ndarray]:
        segment = np.zeros(6)
        distance = float(self.mujoco.mj_geomDistance(
            self.model, self.data, int(geom), self.optimizer.cube_geom_id,
            1.0, segment,
        ))
        return distance, segment

    def _cube_point(self, world_point) -> np.ndarray:
        return self.optimizer._cube_frame(world_point)

    def _nearest_face(self, cube_point) -> str:
        cube_point = np.asarray(cube_point, dtype=float)
        axis = int(np.argmax(
            np.abs(cube_point) / np.maximum(self.optimizer.cube_half_size, 1e-12)
        ))
        return f"{'+' if cube_point[axis] >= 0 else '-'}{'XYZ'[axis]}"

    def _self_collision_penetration(self) -> float:
        penetration = 0.0
        other_hand_bodies = self.hand_body_ids - self.finger_body_ids
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if ((body1 in self.finger_body_ids and body2 in other_hand_bodies)
                    or (body2 in self.finger_body_ids and body1 in other_hand_bodies)):
                penetration += max(0.0, -float(contact.dist))
        return penetration

    def _state_diagnostics(self, q) -> dict:
        self._set_q(q)
        tip_distance, tip_segment = self._geom_distance(self.tip_geom_id)
        contact_point = self._cube_point(tip_segment[3:])
        non_tip = {}
        for geom in self.non_tip_geoms:
            distance, segment = self._geom_distance(geom)
            non_tip[str(geom)] = {
                "signed_distance_m": distance,
                "nearest_cube_point_world_m": segment[3:].copy(),
                "nearest_face": self._nearest_face(self._cube_point(segment[3:])),
            }
        tangent_margin = np.asarray(self.optimizer.cube_half_size[1:]) - np.abs(
            contact_point[1:]
        )
        return {
            "tip_signed_distance_m": tip_distance,
            "tip_nearest_face": self._nearest_face(contact_point),
            "contact_point_cube_m": contact_point,
            "distance_to_nearest_edge_m": float(np.min(tangent_margin)),
            "non_tip": non_tip,
            "overall_non_tip_min_clearance_m": min(
                (item["signed_distance_m"] for item in non_tip.values()),
                default=float("inf"),
            ),
            "self_collision_penetration_m": self._self_collision_penetration(),
        }

    def _pose_residual(self, q, *, region_center, desired_gap_m,
                       nominal_q) -> np.ndarray:
        self._set_q(q)
        cfg = self.cfg
        distance, segment = self._geom_distance(self.tip_geom_id)
        cube_point = self._cube_point(segment[3:])
        half_extent = float(cfg["contact_region_half_extent_m"])
        center = np.asarray(region_center, dtype=float)
        tangential_error = np.sign(cube_point[1:] - center) * np.maximum(
            np.abs(cube_point[1:] - center) - half_extent, 0.0
        )
        measurement = self.optimizer._tip_measurement(self.finger)
        desired_direction = np.array([-1.0, 0.0, 0.0])
        direction_error = (
            measurement["effective_inward_direction_cube"] - desired_direction
        )
        margin = float(cfg["soft_non_tip_clearance_m"])
        residual = [
            (distance - float(desired_gap_m))
            / float(cfg["endpoint_distance_scale_m"]),
            (cube_point[0] - self.optimizer.cube_half_size[0]) / 0.002,
            *(tangential_error / 0.006),
            *(0.2 * direction_error),
        ]
        for geom in self.non_tip_geoms:
            geom_distance, _ = self._geom_distance(geom)
            residual.append(
                np.sqrt(float(cfg["path_clearance_weight"]))
                * max(0.0, margin - geom_distance) / max(margin, 1e-6)
            )
        residual.append(
            np.sqrt(float(cfg["path_self_collision_weight"]))
            * self._self_collision_penetration() / 0.001
        )
        local = np.asarray(q)[self.local_ids]
        residual.extend(0.04 * (local - nominal_q) / self.ranges)
        joint_margin = 0.04 * self.ranges
        residual.extend(0.1 * np.maximum(
            0.0, self.lower + joint_margin - local
        ) / joint_margin)
        residual.extend(0.1 * np.maximum(
            0.0, local - (self.upper - joint_margin)
        ) / joint_margin)
        return np.asarray(residual, dtype=float)

    def _optimize_pose(self, initial_q, *, region_center, desired_gap_m,
                       nominal_q) -> tuple[np.ndarray, dict]:
        q = np.asarray(initial_q, dtype=float).copy()
        cfg = self.cfg
        damping = float(cfg["pose_damping"])
        epsilon = float(cfg["pose_finite_difference_rad"])
        max_step = float(cfg["pose_max_joint_step_rad"])
        residual = self._pose_residual(
            q, region_center=region_center, desired_gap_m=desired_gap_m,
            nominal_q=nominal_q,
        )
        cost = 0.5 * float(np.dot(residual, residual))
        initial_cost = cost
        iterations = 0
        for iterations in range(1, int(cfg["pose_max_iterations"]) + 1):
            jacobian = np.empty((len(residual), 4))
            for column, local_id in enumerate(self.local_ids):
                trial = q.copy()
                trial[local_id] = min(self.upper[column], trial[local_id] + epsilon)
                actual = trial[local_id] - q[local_id]
                if actual <= 1e-12:
                    trial[local_id] = max(self.lower[column], q[local_id] - epsilon)
                    actual = trial[local_id] - q[local_id]
                jacobian[:, column] = (
                    self._pose_residual(
                        trial, region_center=region_center,
                        desired_gap_m=desired_gap_m, nominal_q=nominal_q,
                    ) - residual
                ) / actual
            lhs = jacobian.T @ jacobian + damping * np.eye(4)
            step = np.linalg.solve(lhs, -jacobian.T @ residual)
            step = np.clip(step, -max_step, max_step)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                trial = q.copy()
                trial[self.local_ids] = np.clip(
                    q[self.local_ids] + scale * step, self.lower, self.upper
                )
                trial_residual = self._pose_residual(
                    trial, region_center=region_center,
                    desired_gap_m=desired_gap_m, nominal_q=nominal_q,
                )
                trial_cost = 0.5 * float(np.dot(trial_residual, trial_residual))
                if trial_cost < cost:
                    improvement = cost - trial_cost
                    q, residual, cost = trial, trial_residual, trial_cost
                    damping = max(float(cfg["pose_damping"]), damping * 0.7)
                    accepted = True
                    break
            if not accepted:
                damping *= 5.0
            elif improvement < 1e-8:
                break
        diagnostics = self._state_diagnostics(q)
        diagnostics.update({
            "iterations": iterations,
            "initial_cost": initial_cost,
            "final_cost": cost,
            "desired_tip_gap_m": float(desired_gap_m),
            "region_center_cube_yz_m": list(region_center),
        })
        return q, diagnostics

    def _sample_path(self, q_pre, q_mid, q_star) -> dict:
        sample_count = max(2, int(self.cfg["path_samples_per_segment"]))
        records = []
        minimum = None
        first_penetration = None
        for segment_name, start, end in (
            ("pre_to_mid", q_pre, q_mid),
            ("mid_to_star", q_mid, q_star),
        ):
            for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, sample_count)):
                if segment_name == "mid_to_star" and sample_index == 0:
                    continue
                q = np.asarray(start) + float(alpha) * (
                    np.asarray(end) - np.asarray(start)
                )
                self._set_q(q)
                tip_distance, _ = self._geom_distance(self.tip_geom_id)
                geom_distances = {}
                for geom in self.non_tip_geoms:
                    distance, _ = self._geom_distance(geom)
                    geom_distances[str(geom)] = distance
                    candidate = {
                        "segment": segment_name,
                        "alpha": float(alpha),
                        "geom_id": int(geom),
                        "signed_distance_m": distance,
                    }
                    if minimum is None or distance < minimum["signed_distance_m"]:
                        minimum = candidate
                    if distance <= 0.0 and first_penetration is None:
                        first_penetration = candidate
                records.append({
                    "segment": segment_name,
                    "alpha": float(alpha),
                    "tip_signed_distance_m": tip_distance,
                    "non_tip_signed_distances_m": geom_distances,
                    "minimum_non_tip_clearance_m": min(
                        geom_distances.values(), default=float("inf")
                    ),
                    "self_collision_penetration_m": self._self_collision_penetration(),
                })
        endpoint = self._state_diagnostics(q_star)
        non_tip_collision = any(
            item["minimum_non_tip_clearance_m"] <= 0.0 for item in records
        )
        early_tip_contact = any(
            item["tip_signed_distance_m"] <= 0.0 for item in records[:-1]
        )
        self_collision = any(
            item["self_collision_penetration_m"] > self.penetration_tolerance
            for item in records
        )
        predicted = first_penetration
        if predicted is None:
            tip_candidate = {
                "segment": "mid_to_star", "alpha": 1.0,
                "geom_id": self.tip_geom_id,
                "signed_distance_m": endpoint["tip_signed_distance_m"],
            }
            predicted = min(
                [tip_candidate] + ([minimum] if minimum is not None else []),
                key=lambda item: item["signed_distance_m"],
            )
        by_geom = {
            str(geom): min(
                item["non_tip_signed_distances_m"][str(geom)] for item in records
            ) for geom in self.non_tip_geoms
        }
        return {
            "samples_per_segment": sample_count,
            "curve": records,
            "per_geom_min_clearance_m": by_geom,
            "overall_non_tip_min_clearance_m": (
                None if minimum is None else minimum["signed_distance_m"]
            ),
            "minimum_clearance_segment": None if minimum is None else minimum["segment"],
            "minimum_clearance_alpha": None if minimum is None else minimum["alpha"],
            "minimum_clearance_geom": None if minimum is None else minimum["geom_id"],
            "predicted_first_contact_geom": predicted["geom_id"],
            "path_non_tip_collision": non_tip_collision,
            "early_tip_contact": early_tip_contact,
            "path_self_collision": self_collision,
            "path_feasible": bool(
                not non_tip_collision and not early_tip_contact and not self_collision
            ),
        }

    def _path_objective(self, q_mid, q_pre, q_star) -> tuple[float, dict]:
        path = self._sample_path(q_pre, q_mid, q_star)
        cfg = self.cfg
        margin = float(cfg["soft_non_tip_clearance_m"])
        clearance_loss = 0.0
        hard_collisions = 0
        self_penetration = 0.0
        for sample_index, item in enumerate(path["curve"]):
            for distance in item["non_tip_signed_distances_m"].values():
                clearance_loss += max(0.0, margin - distance) ** 2
                hard_collisions += int(distance <= 0.0)
            # The designated tip is allowed to make contact at the final q_star.
            if sample_index < len(path["curve"]) - 1:
                hard_collisions += int(item["tip_signed_distance_m"] <= 0.0)
            self_penetration += item["self_collision_penetration_m"] ** 2
        local_pre = np.asarray(q_pre)[self.local_ids]
        local_mid = np.asarray(q_mid)[self.local_ids]
        local_star = np.asarray(q_star)[self.local_ids]
        smooth = float(np.sum(((local_mid - local_pre) / self.ranges) ** 2)
                       + np.sum(((local_star - local_mid) / self.ranges) ** 2))
        joint_margin = 0.04 * self.ranges
        joint_limit = float(np.sum(
            np.maximum(0.0, self.lower + joint_margin - local_mid) ** 2
            + np.maximum(0.0, local_mid - (self.upper - joint_margin)) ** 2
        ))
        cost = (
            float(cfg["path_clearance_weight"]) * clearance_loss / max(margin ** 2, 1e-12)
            + float(cfg["path_hard_collision_penalty"]) * hard_collisions
            + float(cfg["path_smoothness_weight"]) * smooth
            + float(cfg["path_joint_limit_weight"]) * joint_limit
            + float(cfg["path_self_collision_weight"]) * self_penetration / 1e-6
        )
        return cost, path

    def _optimize_midpoint(self, q_pre, q_star, *, rng) -> tuple[np.ndarray, dict]:
        base = np.asarray(q_pre) + 0.5 * (
            np.asarray(q_star) - np.asarray(q_pre)
        )
        starts = [base]
        for _ in range(max(0, int(self.cfg["midpoint_restarts"]) - 1)):
            trial = base.copy()
            trial[self.local_ids] = np.clip(
                trial[self.local_ids] + rng.normal(0.0, 0.10, size=4),
                self.lower, self.upper,
            )
            starts.append(trial)
        best_q = base.copy()
        best_cost, best_path = self._path_objective(best_q, q_pre, q_star)
        for start in starts:
            q = start.copy()
            cost, path = self._path_objective(q, q_pre, q_star)
            step = float(self.cfg["midpoint_initial_step_rad"])
            for _ in range(int(self.cfg["midpoint_iterations"])):
                improved = False
                for column, local_id in enumerate(self.local_ids):
                    for direction in (-1.0, 1.0):
                        trial = q.copy()
                        trial[local_id] = np.clip(
                            trial[local_id] + direction * step,
                            self.lower[column], self.upper[column],
                        )
                        trial_cost, trial_path = self._path_objective(
                            trial, q_pre, q_star
                        )
                        if trial_cost < cost:
                            q, cost, path = trial, trial_cost, trial_path
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < float(self.cfg["midpoint_min_step_rad"]):
                        break
            if cost < best_cost:
                best_q, best_cost, best_path = q, cost, path
            if best_path["path_feasible"]:
                break
        return best_q, {"objective": best_cost, **best_path}

    def plan(self, base_q, *, seed: int = 7) -> list[FingerPathCandidate]:
        base_q = np.asarray(base_q, dtype=float).copy()
        if base_q.shape != self.optimizer.nominal.shape:
            raise ValueError("base_q must match the optimizer decision variables")
        # Single-finger planning fixes every non-finger2 digit in the open pose.
        # This prevents their Stage-1 endpoint contacts from contaminating the
        # isolated path feasibility and subsequent dynamics validation.
        for finger_index in (0, 2, 3, 4):
            start = self.arm_count + 4 * finger_index
            hand_start = 4 * finger_index
            base_q[start:start + 4] = self.open_hand[hand_start:hand_start + 4]
        rng = np.random.default_rng(seed)
        results = []
        for name, region_center in self.region_seeds:
            endpoint_nominal = base_q[self.local_ids].copy()
            q_star, endpoint = self._optimize_pose(
                base_q, region_center=region_center,
                desired_gap_m=float(self.cfg["endpoint_gap_m"]),
                nominal_q=endpoint_nominal,
            )
            endpoint_error = abs(
                endpoint["tip_signed_distance_m"]
                - float(self.cfg["endpoint_gap_m"])
            )
            edge_margin = float(self.config["acquisition"]["edge_margin_m"])
            endpoint_feasible = bool(
                endpoint_error <= float(self.cfg["endpoint_contact_tolerance_m"])
                and endpoint["tip_nearest_face"] == "+X"
                and endpoint["distance_to_nearest_edge_m"] >= edge_margin
                and endpoint["overall_non_tip_min_clearance_m"] > 0.0
                and endpoint["self_collision_penetration_m"] <= self.penetration_tolerance
            )
            if not endpoint_feasible:
                results.append(self._candidate(
                    name, region_center, "contact_region_endpoint_infeasible",
                    False, endpoint_error, endpoint, base_q, base_q, q_star,
                    False, {},
                ))
                continue
            q_pre, pregrasp = self._optimize_pose(
                q_star, region_center=region_center,
                desired_gap_m=float(self.cfg["pregrasp_clearance_m"]),
                nominal_q=q_star[self.local_ids],
            )
            pregrasp_feasible = bool(
                pregrasp["tip_signed_distance_m"] > 0.0
                and pregrasp["overall_non_tip_min_clearance_m"] > 0.0
                and pregrasp["self_collision_penetration_m"] <= self.penetration_tolerance
            )
            if not pregrasp_feasible:
                results.append(self._candidate(
                    name, region_center, "midpoint_optimization_failed",
                    True, endpoint_error, endpoint, q_pre, q_pre, q_star,
                    False, {"pregrasp": pregrasp},
                ))
                continue
            q_mid, path = self._optimize_midpoint(q_pre, q_star, rng=rng)
            path_feasible = bool(path["path_feasible"])
            outcome = "path_feasible" if path_feasible else "path_non_tip_collision"
            results.append(self._candidate(
                name, region_center, outcome, True, endpoint_error, endpoint,
                q_pre, q_mid, q_star, True, {"pregrasp": pregrasp, **path},
            ))
        return results

    def _candidate(self, name, region_center, outcome, endpoint_feasible,
                   endpoint_error, endpoint, q_pre, q_mid, q_star,
                   midpoint_found, path) -> FingerPathCandidate:
        curve = path.get("curve", [])
        by_geom = path.get("per_geom_min_clearance_m", {})
        return FingerPathCandidate(
            name=name,
            contact_region_center_cube_m=tuple(float(v) for v in region_center),
            contact_region_half_extent_m=float(
                self.cfg["contact_region_half_extent_m"]
            ),
            outcome=outcome,
            endpoint_feasible=bool(endpoint_feasible),
            endpoint_error_m=float(endpoint_error),
            contact_point_cube_m=tuple(float(v) for v in endpoint["contact_point_cube_m"]),
            q_pre=tuple(float(v) for v in q_pre),
            q_mid=tuple(float(v) for v in q_mid),
            q_star=tuple(float(v) for v in q_star),
            midpoint_found=bool(midpoint_found),
            path_feasible=bool(path.get("path_feasible", False)),
            geom53_min_clearance_m=by_geom.get("53"),
            geom55_min_clearance_m=by_geom.get("55"),
            overall_non_tip_min_clearance_m=path.get(
                "overall_non_tip_min_clearance_m"
            ),
            min_clearance_segment=path.get("minimum_clearance_segment"),
            min_clearance_alpha=path.get("minimum_clearance_alpha"),
            predicted_first_contact_geom=path.get("predicted_first_contact_geom"),
            endpoint_diagnostics=_plain(endpoint),
            pregrasp_diagnostics=_plain(path.get("pregrasp", {})),
            path_diagnostics=_plain({
                key: value for key, value in path.items() if key != "pregrasp"
            }),
        )


def execute_finger2_waypoint_path(robot, task, candidate: FingerPathCandidate,
                                  optimizer: ContactRegionGraspOptimizer,
                                  config: dict) -> dict:
    """Slowly execute q_pre -> q_mid -> q_star with only finger2 moving."""
    import mujoco

    if not candidate.path_feasible:
        return {"outcome": "path_non_tip_collision", "single_finger_path_pass": False}
    cfg = config["finger2_path_planning"]
    acq = config["acquisition"]
    model, data = robot.model, robot.data
    q_pre = np.asarray(candidate.q_pre, dtype=float)
    q_mid = np.asarray(candidate.q_mid, dtype=float)
    q_star = np.asarray(candidate.q_star, dtype=float)
    joint_ids = np.asarray(optimizer.joint_ids, dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    actuator_ids = _actuators_for_joints(model, joint_ids)
    finger_ids = np.arange(len(optimizer.arm_joint_ids) + 4,
                           len(optimizer.arm_joint_ids) + 8, dtype=int)
    fixed_ids = np.asarray([
        index for index in range(len(joint_ids)) if index not in set(finger_ids)
    ], dtype=int)
    data.qpos[qpos_ids] = q_pre
    data.qvel.fill(0.0)
    data.ctrl[actuator_ids] = q_pre
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    duration = float(cfg["dynamic_segment_duration_s"])
    control_hz = float(acq.get("control_hz", 30.0))
    steps = max(2, int(round(duration * control_hz)))
    physics_steps = max(1, int(round(1.0 / (control_hz * dt))))

    def lock_fixed_state():
        data.qpos[qpos_ids[fixed_ids]] = q_pre[fixed_ids]
        data.qvel[dof_ids[fixed_ids]] = 0.0
        data.ctrl[actuator_ids[fixed_ids]] = q_pre[fixed_ids]

    def lock_pregrasp_state():
        """Hold the complete planned q_pre while the free cube settles."""
        data.qpos[qpos_ids] = q_pre
        data.qvel[dof_ids] = 0.0
        data.ctrl[actuator_ids] = q_pre

    # Clear transients from teleporting into the isolated pregrasp.  Only the
    # cube remains dynamically free; palm/wrist and four non-tested digits are
    # numerically locked at every MuJoCo physics step.
    pre_settle_steps = max(0, int(round(
        float(cfg["dynamic_pre_settle_s"]) / dt
    )))
    for _ in range(pre_settle_steps):
        lock_pregrasp_state()
        mujoco.mj_step(model, data)
        lock_pregrasp_state()
        mujoco.mj_forward(model, data)

    initial = task.task_telemetry()
    anchor_position = initial["cube_position_m"].copy()
    anchor_quaternion = initial["cube_quaternion_wxyz"].copy()
    first_contact = None
    non_tip_before_tip = False
    first_tip_contact = None
    first_non_tip_contact = None
    max_translation = 0.0
    max_rotation = 0.0
    time_series = []
    tip_geom_id = int(optimizer.tips["finger2"].tip_geom_id)
    finger_geoms = tuple(optimizer.finger_collision_geoms["finger2"])
    non_tip_geoms = tuple(geom for geom in finger_geoms if geom != tip_geom_id)
    min_signed_distance = {str(geom): float("inf") for geom in finger_geoms}
    min_non_tip_before_tip = {str(geom): float("inf") for geom in non_tip_geoms}
    min_force = float(acq["min_normal_force_n"])
    start_time = float(data.time)
    active_command = q_pre.copy()
    motion_segments = [
        ("pre_to_mid", q_pre, q_mid),
        ("mid_to_star", q_mid, q_star),
    ]
    hold_steps = max(0, int(round(
        float(cfg["dynamic_contact_hold_s"]) * control_hz
    )))
    if hold_steps:
        motion_segments.append(("star_hold", q_star, q_star))

    def contact_event(contact):
        geom_id = int(contact.get("other_geom_id"))
        signed_distance = float(mujoco.mj_geomDistance(
            model, data, geom_id, optimizer.cube_geom_id, 1.0, np.zeros(6)
        ))
        return {
            "time_s": float(data.time - start_time),
            "geom_id": contact.get("other_geom_id"),
            "geom_name": contact.get("other_geom_name"),
            "body_name": contact.get("other_body_name"),
            "is_designated_tip": contact.get("is_designated_tip"),
            "is_distal_non_tip": contact.get("is_distal_non_tip"),
            "face": contact.get("cube_face"),
            "edge_contact": contact.get("edge_contact"),
            "corner_contact": contact.get("corner_contact"),
            "position_cube_m": _plain(contact.get("position_cube_m")),
            "normal_force_n": float(contact.get("normal_force_n", 0.0)),
            "signed_distance_m": signed_distance,
            "actual_finger_qpos": _plain(data.qpos[qpos_ids[finger_ids]]),
            "commanded_finger_qpos": _plain(active_command[finger_ids]),
        }

    for segment_name, start, end in motion_segments:
        segment_steps = hold_steps if segment_name == "star_hold" else steps
        for step in range(1, segment_steps + 1):
            alpha = step / segment_steps
            command = start + alpha * (end - start)
            active_command = command
            data.ctrl[actuator_ids[finger_ids]] = command[finger_ids]
            for _ in range(physics_steps):
                lock_fixed_state()
                mujoco.mj_step(model, data)
                lock_fixed_state()
                mujoco.mj_forward(model, data)
                for geom in finger_geoms:
                    distance = float(mujoco.mj_geomDistance(
                        model, data, int(geom), optimizer.cube_geom_id,
                        1.0, np.zeros(6),
                    ))
                    min_signed_distance[str(geom)] = min(
                        min_signed_distance[str(geom)], distance
                    )
                    if first_tip_contact is None and geom != tip_geom_id:
                        min_non_tip_before_tip[str(geom)] = min(
                            min_non_tip_before_tip[str(geom)], distance
                        )
                physics_contacts = [
                    item for item in task.contact_monitor.contacts(0.0)
                    if item["finger"] == "finger2"
                    and item["normal_force_n"] >= min_force
                ]
                tip_contacts = [
                    item for item in physics_contacts
                    if item.get("is_designated_tip", False)
                ]
                non_tip_contacts = [
                    item for item in physics_contacts
                    if not item.get("is_designated_tip", False)
                ]
                if non_tip_contacts and first_non_tip_contact is None:
                    first_non_tip_contact = contact_event(max(
                        non_tip_contacts, key=lambda item: item["normal_force_n"]
                    ))
                if tip_contacts and first_tip_contact is None:
                    first_tip_contact = contact_event(max(
                        tip_contacts, key=lambda item: item["normal_force_n"]
                    ))
                if first_contact is None and physics_contacts:
                    # Simultaneous tip/non-tip within one 2 ms physics step is
                    # conservatively classified as non-tip-first.
                    selected = max(
                        non_tip_contacts or tip_contacts,
                        key=lambda item: item["normal_force_n"],
                    )
                    first_contact = contact_event(selected)
                    non_tip_before_tip = bool(non_tip_contacts)
            telemetry = task.task_telemetry()
            contacts = [
                item for item in telemetry["contacts"]
                if item["finger"] == "finger2"
            ]
            translation, rotation = pose_drift(
                anchor_position, anchor_quaternion,
                telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"],
            )
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
            time_series.append({
                "segment": segment_name,
                "alpha": alpha,
                "time_s": float(data.time - start_time),
                "finger2_contacts": _plain(contacts),
                "cube_translation_m": translation,
                "cube_rotation_deg": rotation,
            })
    tip_first = bool(first_contact and first_contact["is_designated_tip"])
    face_ok = bool(first_contact and first_contact["face"] == "+X")
    interior = bool(first_contact and not first_contact["edge_contact"]
                    and not first_contact["corner_contact"])
    small_motion = bool(
        max_translation <= float(acq["max_cube_translation_before_topology_m"])
        and max_rotation <= float(acq["max_cube_rotation_before_topology_deg"])
    )
    penetration_tolerance = float(
        config["optimization"]["max_kinematic_penetration_m"]
    )
    no_non_tip_penetration = all(
        value >= -penetration_tolerance
        for key, value in min_signed_distance.items()
        if int(key) != tip_geom_id
    )
    no_penetration = bool(
        no_non_tip_penetration
        and min_signed_distance[str(tip_geom_id)] >= -penetration_tolerance
    )
    non_tip_clear_before_tip = all(
        value > 0.0 for value in min_non_tip_before_tip.values()
    )
    prediction_matches_actual = bool(
        first_contact and candidate.predicted_first_contact_geom
        == first_contact["geom_id"]
    )
    passed = bool(tip_first and face_ok and interior and not non_tip_before_tip
                  and non_tip_clear_before_tip and prediction_matches_actual
                  and small_motion and no_penetration)
    return {
        "outcome": "single_finger_path_pass" if passed else (
            "single_finger_tip_first" if tip_first else
            "single_finger_non_tip_first" if first_contact is not None else
            "single_finger_no_contact"
        ),
        "single_finger_path_pass": passed,
        "first_contact": first_contact,
        "first_tip_contact": first_tip_contact,
        "first_non_tip_contact": first_non_tip_contact,
        "predicted_first_contact_geom": candidate.predicted_first_contact_geom,
        "prediction_matches_actual": prediction_matches_actual,
        "max_cube_translation_m": max_translation,
        "max_cube_rotation_deg": max_rotation,
        "minimum_signed_distance_by_geom_m": min_signed_distance,
        "minimum_non_tip_distance_before_tip_by_geom_m": min_non_tip_before_tip,
        "non_tip_clear_before_tip": non_tip_clear_before_tip,
        "no_non_tip_penetration": no_non_tip_penetration,
        "no_penetration": no_penetration,
        "physics_contact_time_resolution_s": dt,
        "time_series": time_series,
    }
