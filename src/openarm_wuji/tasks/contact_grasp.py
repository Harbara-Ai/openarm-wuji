from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Sequence

import numpy as np

from .ik import DampedLeastSquaresIK
from .se3 import (
    normalize_quaternion,
    pose_drift,
    quaternion_multiply,
    rotation_vector_to_quaternion,
)


_FACE_AXIS = {"X": 0, "Y": 1, "Z": 2}


def project_to_face_region(point_cube_m, face: str, half_size_m,
                           tangential_half_extent_m) -> tuple[np.ndarray, np.ndarray]:
    """Project a point onto a rectangular region centered on a box face."""
    point = np.asarray(point_cube_m, dtype=float)
    half_size = np.asarray(half_size_m, dtype=float)
    extent = np.asarray(tangential_half_extent_m, dtype=float)
    if point.shape != (3,) or half_size.shape != (3,) or extent.shape != (2,):
        raise ValueError("point/half-size/region extent shapes must be 3/3/2")
    if len(face) != 2 or face[0] not in "+-" or face[1] not in _FACE_AXIS:
        raise ValueError(f"invalid box face: {face!r}")
    if np.any(half_size <= 0.0) or np.any(extent <= 0.0):
        raise ValueError("box and region extents must be positive")
    axis = _FACE_AXIS[face[1]]
    tangent = [index for index in range(3) if index != axis]
    target = point.copy()
    target[axis] = (1.0 if face[0] == "+" else -1.0) * half_size[axis]
    target[tangent] = np.clip(target[tangent], -extent, extent)
    return target, point - target


def inward_face_direction(face: str) -> np.ndarray:
    """Return the desired finger-to-object direction for a box face."""
    if len(face) != 2 or face[0] not in "+-" or face[1] not in _FACE_AXIS:
        raise ValueError(f"invalid box face: {face!r}")
    result = np.zeros(3)
    result[_FACE_AXIS[face[1]]] = -1.0 if face[0] == "+" else 1.0
    return result


@dataclass(frozen=True)
class FingertipDescriptor:
    finger: str
    distal_body_id: int
    distal_body_name: str
    tip_geom_id: int
    tip_geom_name: str | None
    joint_ids: tuple[int, ...]
    joint_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContactGraspSolution:
    topology: str
    initialization: str
    success: bool
    iterations: int
    initial_cost: float
    final_cost: float
    variable_joint_ids: tuple[int, ...]
    variable_joint_names: tuple[str, ...]
    arm_joint_ids: tuple[int, ...]
    hand_joint_ids: tuple[int, ...]
    qpos: tuple[float, ...]
    palm_position_world_m: tuple[float, ...]
    palm_quaternion_world_wxyz: tuple[float, ...]
    fingertip_diagnostics: dict
    collision_diagnostics: dict
    joint_limit_hits: tuple[str, ...]
    required_fingers: tuple[str, ...]
    required_fingers_satisfied: tuple[str, ...]
    optional_fingers: tuple[str, ...]
    initialization_diagnostics: dict
    stage_history: tuple[dict, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _name(mujoco, model, kind, index: int) -> str | None:
    return mujoco.mj_id2name(model, kind, int(index))


def _descendants(model, root_body_id: int) -> set[int]:
    result = set()
    for body_id in range(model.nbody):
        cursor = body_id
        while cursor > 0:
            if cursor == root_body_id:
                result.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    result.add(int(root_body_id))
    return result


def discover_fingertips(model, *, palm_site_name: str,
                        finger_body_regex: str) -> dict[str, FingertipDescriptor]:
    """Find distal finger bodies and their outermost collision-enabled geom."""
    import mujoco

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, palm_site_name
    )
    if site_id < 0:
        raise KeyError(f"palm site is missing: {palm_site_name}")
    palm_body = int(model.site_bodyid[site_id])
    hand_bodies = _descendants(model, palm_body)
    pattern = re.compile(finger_body_regex)
    grouped: dict[str, list[int]] = {}
    for body_id in hand_bodies:
        body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        match = pattern.search(body_name)
        if match:
            grouped.setdefault(f"finger{match.group(1)}", []).append(body_id)
    descriptors = {}
    for finger in [f"finger{index}" for index in range(1, 6)]:
        bodies = grouped.get(finger, [])
        if not bodies:
            raise KeyError(f"could not discover a body chain for {finger}")
        body_set = set(bodies)
        distal = [body for body in bodies if not any(
            int(model.body_parentid[child]) == body for child in body_set
        )]
        if len(distal) != 1:
            raise ValueError(f"ambiguous distal body for {finger}: {distal}")
        body_id = distal[0]
        geoms = [index for index in range(model.ngeom)
                 if int(model.geom_bodyid[index]) == body_id
                 and (int(model.geom_contype[index]) != 0
                      or int(model.geom_conaffinity[index]) != 0)]
        if not geoms:
            raise KeyError(f"no collision-enabled distal geom for {finger}")
        tip_geom = max(geoms, key=lambda index: float(np.linalg.norm(
            model.geom_pos[index]
        )))
        joints = []
        cursor = body_id
        while cursor != palm_body and cursor > 0:
            start = int(model.body_jntadr[cursor])
            count = int(model.body_jntnum[cursor])
            joints.extend(range(start, start + count))
            cursor = int(model.body_parentid[cursor])
        joints.reverse()
        descriptors[finger] = FingertipDescriptor(
            finger=finger,
            distal_body_id=body_id,
            distal_body_name=_name(
                mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or f"body_{body_id}",
            tip_geom_id=tip_geom,
            tip_geom_name=_name(
                mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, tip_geom
            ),
            joint_ids=tuple(joints),
            joint_names=tuple(
                _name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, joint) or
                f"joint_{joint}" for joint in joints
            ),
        )
    return descriptors


class ContactRegionGraspOptimizer:
    """Numerical whole-hand IK for box-face contact regions.

    The decision variables are the actuated arm joints (which determine palm
    SE(3)) and all independent finger joints. It deliberately does not use the
    three-dimensional policy synergy action.
    """

    def __init__(self, model, source_data, config: dict, *, topology: dict):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = source_data.qpos
        self.data.qvel[:] = source_data.qvel
        self.data.ctrl[:] = source_data.ctrl
        mujoco.mj_forward(model, self.data)
        self.config = config
        self.topology = topology
        self.tips = discover_fingertips(
            model,
            palm_site_name=config["palm_site_name"],
            finger_body_regex=config["finger_body_regex"],
        )
        self.palm_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, config["palm_site_name"]
        )
        self.cube_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, config["cube_body_name"]
        )
        if self.cube_body_id < 0:
            raise KeyError(f"cube body is missing: {config['cube_body_name']}")
        cube_geoms = [index for index in range(model.ngeom)
                      if int(model.geom_bodyid[index]) == self.cube_body_id]
        boxes = [index for index in cube_geoms
                 if int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
        if not boxes:
            raise KeyError("cube body has no box geom")
        self.cube_geom_id = boxes[0]
        self.cube_half_size = model.geom_size[self.cube_geom_id].copy()
        palm_body = int(model.site_bodyid[self.palm_site_id])
        ancestor_joints = []
        cursor = palm_body
        while cursor > 0:
            start = int(model.body_jntadr[cursor])
            count = int(model.body_jntnum[cursor])
            ancestor_joints.extend(range(start, start + count))
            cursor = int(model.body_parentid[cursor])
        ancestor_joints.reverse()
        self.arm_joint_ids = tuple(joint for joint in ancestor_joints
                                   if int(model.jnt_type[joint])
                                   == int(mujoco.mjtJoint.mjJNT_HINGE))
        self.hand_joint_ids = tuple(
            joint for finger in self.tips.values() for joint in finger.joint_ids
        )
        self.joint_ids = self.arm_joint_ids + self.hand_joint_ids
        if len(self.arm_joint_ids) != 7 or len(self.hand_joint_ids) != 20:
            raise ValueError(
                f"expected 7 arm and 20 hand joints, got "
                f"{len(self.arm_joint_ids)} and {len(self.hand_joint_ids)}"
            )
        self.qpos_ids = np.asarray([
            int(model.jnt_qposadr[joint]) for joint in self.joint_ids
        ])
        self.nominal = self.data.qpos[self.qpos_ids].copy()
        self.lower = np.asarray([
            model.jnt_range[joint, 0] for joint in self.joint_ids
        ])
        self.upper = np.asarray([
            model.jnt_range[joint, 1] for joint in self.joint_ids
        ])
        self.ranges = np.maximum(self.upper - self.lower, 1e-6)
        tip_geoms = {tip.tip_geom_id for tip in self.tips.values()}
        hand_bodies = _descendants(model, palm_body)
        self.non_tip_hand_geoms = tuple(
            geom for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) in hand_bodies
            and geom not in tip_geoms
            and (int(model.geom_contype[geom]) != 0
                 or int(model.geom_conaffinity[geom]) != 0)
        )
        self._finger_pattern = re.compile(config["finger_body_regex"])
        self.finger_collision_geoms = {}
        for finger, descriptor in self.tips.items():
            body_ids = {descriptor.distal_body_id}
            # Include the complete finger chain, not only the distal body.
            cursor = descriptor.distal_body_id
            while cursor > palm_body:
                body_ids.add(cursor)
                cursor = int(model.body_parentid[cursor])
            self.finger_collision_geoms[finger] = tuple(
                geom for geom in range(model.ngeom)
                if int(model.geom_bodyid[geom]) in body_ids
                and (int(model.geom_contype[geom]) != 0
                     or int(model.geom_conaffinity[geom]) != 0)
            )

    def swept_path_diagnostics(self, q_pre, q_star, *, samples=None,
                               path_duration_s=None) -> dict:
        """Measure every finger collision geom over q_pre -> q_star.

        Distances are signed MuJoCo geom distances.  TTC is estimated from the
        finite-difference normal velocity of each geom along the sampled path;
        it is reported only while the geom is moving toward the cube.
        """
        q_pre = np.asarray(q_pre, dtype=float)
        q_star = np.asarray(q_star, dtype=float)
        count = int(samples or self.config.get("acquisition", {}).get(
            "path_samples", 21
        ))
        count = max(2, count)
        duration = float(path_duration_s or self.config.get("acquisition", {}).get(
            "pregrasp_move_s", 0.8
        ))
        alphas = np.linspace(0.0, 1.0, count)
        dt_path = duration / max(count - 1, 1)
        records = {finger: [] for finger in self.topology.get(
            "required_fingers", self.tips
        )}
        previous = {}
        for alpha in alphas:
            q = np.clip((1.0 - alpha) * q_pre + alpha * q_star,
                        self.lower, self.upper)
            self._set_q(q)
            for finger in records:
                tip_id = self.tips[finger].tip_geom_id
                for geom in self.finger_collision_geoms[finger]:
                    segment = np.zeros(6)
                    distance = float(self.mujoco.mj_geomDistance(
                        self.model, self.data, geom, self.cube_geom_id,
                        1.0, segment,
                    ))
                    direction = segment[3:] - segment[:3]
                    norm = float(np.linalg.norm(direction))
                    normal = direction / max(norm, 1e-12)
                    point = self.data.geom_xpos[geom].copy()
                    velocity = np.zeros(3)
                    if geom in previous:
                        velocity = (point - previous[geom]) / max(dt_path, 1e-12)
                    v_n = float(np.dot(velocity, normal))
                    ttc = None if v_n >= -1e-8 else max(distance, 0.0) / (-v_n)
                    body_id = int(self.model.geom_bodyid[geom])
                    body_name = _name(
                        self.mujoco, self.model,
                        self.mujoco.mjtObj.mjOBJ_BODY, body_id,
                    ) or ""
                    item = {
                        "alpha": float(alpha), "geom_id": int(geom),
                        "geom_name": _name(
                            self.mujoco, self.model,
                            self.mujoco.mjtObj.mjOBJ_GEOM, geom,
                        ),
                        "body_name": body_name,
                        "is_designated_tip": bool(geom == tip_id),
                        "is_distal_non_tip": bool(
                            geom != tip_id and body_id == self.tips[finger].distal_body_id
                        ),
                        "contact_role": (
                            "designated_tip_geom" if geom == tip_id else
                            "distal_non_tip_geom" if body_id == self.tips[finger].distal_body_id else
                            "other_finger_link_geom"
                        ),
                        "signed_distance_m": distance,
                        "nearest_cube_point_world_m": segment[3:].copy(),
                        "nearest_face": self._cube_face_from_world_point(segment[3:]),
                        "geom_position_world_m": point,
                        "normal_world": normal,
                        "normal_velocity_m_s": v_n,
                        "ttc_s": ttc,
                    }
                    records[finger].append(item)
                    previous[geom] = point
        summary = {}
        safe_margin = float(self.config.get("acquisition", {}).get(
            "path_clearance_safe_m", 0.001
        ))
        for finger, items in records.items():
            non_tip = [item for item in items if not item["is_designated_tip"]]
            tip = [item for item in items if item["is_designated_tip"]]
            min_path = min(items, key=lambda item: item["signed_distance_m"], default=None)
            min_non_tip = min(non_tip, key=lambda item: item["signed_distance_m"],
                              default=None)
            min_ttc = min(
                (item for item in non_tip if item["ttc_s"] is not None),
                key=lambda item: item["ttc_s"], default=None,
            )
            predicted = min(
                (item for item in items if item["signed_distance_m"] <= 0.0),
                key=lambda item: item["alpha"], default=None,
            )
            if predicted is None:
                predicted = min(
                    (item for item in items if item["ttc_s"] is not None),
                    key=lambda item: item["ttc_s"], default=min_non_tip,
                )
            summary[finger] = {
                "min_path_clearance_m": None if min_path is None else min_path["signed_distance_m"],
                "min_path_clearance_alpha": None if min_path is None else min_path["alpha"],
                "min_path_clearance_geom": None if min_path is None else min_path["geom_id"],
                "min_non_tip_path_clearance_m": None if min_non_tip is None else min_non_tip["signed_distance_m"],
                "min_non_tip_clearance_alpha": None if min_non_tip is None else min_non_tip["alpha"],
                "min_non_tip_ttc_s": None if min_ttc is None else min_ttc["ttc_s"],
                "designated_tip_ttc_s": min(
                    (item["ttc_s"] for item in tip if item["ttc_s"] is not None),
                    default=None,
                ),
                "predicted_first_contact_geom": None if predicted is None else predicted["geom_id"],
                "predicted_first_contact_role": None if predicted is None else predicted["contact_role"],
                "path_infeasible": bool(any(
                    item["signed_distance_m"] <= 0.0 for item in non_tip
                )),
                "soft_clearance_violation": bool(
                    min_non_tip is not None and min_non_tip["signed_distance_m"] < safe_margin
                ),
                "samples": records[finger],
            }
        self._set_q(q_star)
        return {
            "samples": count,
            "path_duration_s": duration,
            "safe_clearance_m": safe_margin,
            "per_finger": summary,
        }

    def _cube_face_from_world_point(self, world_point) -> str:
        """Return the nearest cube face for a world-space point."""
        point = self._cube_frame(world_point)
        axis = int(np.argmax(np.abs(point) / np.maximum(self.cube_half_size, 1e-12)))
        sign = "+" if point[axis] >= 0.0 else "-"
        return f"{sign}{'XYZ'[axis]}"

    def make_initial_q(self, initialization: dict, *, hand_seed) -> tuple[np.ndarray, dict]:
        """Construct one semantic palm/hand initialization and enforce reachability."""
        self._set_q(self.nominal)
        hand_seed = np.asarray(hand_seed, dtype=float)
        if hand_seed.shape != (len(self.hand_joint_ids),):
            raise ValueError("hand seed must contain one value per independent joint")
        palm_position = self.data.site_xpos[self.palm_site_id].copy()
        palm_quaternion = np.zeros(4)
        self.mujoco.mju_mat2Quat(
            palm_quaternion, self.data.site_xmat[self.palm_site_id]
        )
        cube_rotation = np.asarray(
            self.data.xmat[self.cube_body_id]
        ).reshape(3, 3)
        translation_cube = np.asarray(
            initialization["palm_translation_cube_m"], dtype=float
        )
        rotation_cube = np.asarray(
            initialization["palm_rotation_cube_rotvec_rad"], dtype=float
        )
        if translation_cube.shape != (3,) or rotation_cube.shape != (3,):
            raise ValueError("palm initialization deltas must be 3-D")
        target_position = palm_position + cube_rotation @ translation_cube
        rotation_world = cube_rotation @ rotation_cube
        target_quaternion = quaternion_multiply(
            rotation_vector_to_quaternion(rotation_world),
            normalize_quaternion(palm_quaternion),
        )
        arm_names = [
            _name(
                self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint
            ) or f"joint_{joint}" for joint in self.arm_joint_ids
        ]
        ik_cfg = self.config["optimization"]["palm_seed_ik"]
        ik = DampedLeastSquaresIK(
            self.model,
            self.data,
            site_name=self.config["palm_site_name"],
            joint_names=arm_names,
            damping=float(ik_cfg["damping"]),
            max_joint_step=float(ik_cfg["max_joint_step_rad"]),
        )
        arm_q, position_error, orientation_error, iterations = ik.solve_pose(
            target_position,
            target_quaternion,
            max_iterations=int(ik_cfg["max_iterations"]),
            position_tolerance=float(ik_cfg["position_tolerance_m"]),
            orientation_tolerance_deg=float(ik_cfg["orientation_tolerance_deg"]),
            orientation_weight=float(ik_cfg["orientation_weight"]),
        )
        q = self.nominal.copy()
        q[:len(self.arm_joint_ids)] = arm_q
        q[len(self.arm_joint_ids):] = np.clip(
            hand_seed,
            self.lower[len(self.arm_joint_ids):],
            self.upper[len(self.arm_joint_ids):],
        )
        self._set_q(q)
        return q, {
            "name": initialization["name"],
            "hand_seed": initialization["hand_seed"],
            "palm_seed_ik_position_error_m": position_error,
            "palm_seed_ik_orientation_error_deg": orientation_error,
            "palm_seed_ik_iterations": iterations,
            "palm_seed_ik_reached": bool(
                position_error <= float(ik_cfg["position_tolerance_m"])
                and orientation_error <= float(ik_cfg["orientation_tolerance_deg"])
            ),
        }

    def _set_q(self, q) -> None:
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel.fill(0.0)
        self.mujoco.mj_forward(self.model, self.data)

    def build_pregrasp(self, q_star, *, open_hand, clearance_m=None,
                       search_steps=None, required_fingers=None
                       ) -> tuple[np.ndarray, dict]:
        """Construct a collision-free Cartesian-normal pre-contact state.

        Each designated fingertip is given a target on its assigned cube face,
        offset outward by the configured clearance.  A damped point-Jacobian IK
        moves only that finger while the solved palm and the other fingers stay
        fixed.  The result is still checked against every hand/cube and
        inter-finger collision; this is intentionally not a blind joint-space
        retreat.
        """
        q_star = np.asarray(q_star, dtype=float)
        open_hand = np.asarray(open_hand, dtype=float)
        arm_count = len(self.arm_joint_ids)
        if q_star.shape != self.nominal.shape:
            raise ValueError("q_star must match optimizer decision variables")
        if open_hand.shape != (len(self.hand_joint_ids),):
            raise ValueError("open_hand must match independent hand joints")
        acq = self.config.get("acquisition", {})
        clearance = float(acq.get("pregrasp_clearance_m", 0.004)
                           if clearance_m is None else clearance_m)
        count = int(acq.get("pregrasp_search_steps", 32)
                    if search_steps is None else search_steps)
        q_pre = q_star.copy()
        # Start from an open hand and independently construct each required
        # finger's Cartesian pre-contact pose.  Beginning at q_star lets
        # distal links collide with the cube and with neighboring fingers
        # before the normal-retreat solver has a chance to act.
        q_pre[arm_count:] = np.clip(
            open_hand, self.lower[arm_count:], self.upper[arm_count:]
        )
        required_fingers = set(
            self.topology.get(
                "required_fingers", [f"finger{index}" for index in range(1, 6)]
            ) if required_fingers is None else required_fingers
        )
        diagnostics = {
            "method": "cartesian_normal_precontact",
            "clearance_m": clearance,
            "fingers": {},
        }
        for index in range(5):
            finger = f"finger{index + 1}"
            section = slice(arm_count + 4 * index, arm_count + 4 * (index + 1))
            target = q_star[section].copy()
            source = open_hand[4 * index:4 * (index + 1)]
            if finger not in required_fingers:
                diagnostics["fingers"][finger] = {
                    "skipped": True,
                    "reason": "optional_finger_left_open",
                    "selected_qpos": q_pre[section].tolist(),
                }
                continue
            self._set_q(q_pre)
            before = self._tip_measurement(finger)
            face = self.topology["faces"][finger]
            axis = _FACE_AXIS[face[1]]
            sign = 1.0 if face[0] == "+" else -1.0
            cube_rotation = np.asarray(self.data.xmat[self.cube_body_id]).reshape(3, 3)
            surface_cube, _ = project_to_face_region(
                before["representative_point_cube_m"], face, self.cube_half_size,
                self.config["target_region"]["tangential_half_extent_m"],
            )
            standoff = before["representative_standoff_m"]
            target_cube = surface_cube.copy()
            target_cube[axis] += sign * (clearance + standoff)
            outward_world = cube_rotation[:, axis] * sign
            target_world = self.data.xpos[self.cube_body_id] + cube_rotation @ target_cube
            ik_cfg = self.config.get("acquisition", {})
            iterations = int(ik_cfg.get("pregrasp_ik_iterations", max(8, count)))
            damping = float(ik_cfg.get("pregrasp_ik_damping", 1e-3))
            tolerance = float(ik_cfg.get("pregrasp_ik_tolerance_m", 0.0005))
            max_step = float(ik_cfg.get("pregrasp_ik_max_step_rad", 0.06))
            dof_ids = np.asarray([
                int(self.model.jnt_dofadr[joint])
                for joint in self.tips[finger].joint_ids
            ], dtype=int)
            local_ids = np.asarray([
                arm_count + 4 * index + offset for offset in range(4)
            ], dtype=int)
            reached_error = float("inf")
            retreat_attempts = max(1, int(ik_cfg.get(
                "pregrasp_retreat_attempts", 4
            )))
            retreat_step = float(ik_cfg.get("pregrasp_retreat_step_m", 0.003))
            retreat_extra = 0.0
            retry_count = 0
            for retry_count in range(retreat_attempts):
                retreat_extra = retry_count * retreat_step
                retry_target = target_world + retreat_extra * outward_world
                for _ in range(iterations):
                    self._set_q(q_pre)
                    point = self.data.geom_xpos[self.tips[finger].tip_geom_id].copy()
                    error = retry_target - point
                    reached_error = float(np.linalg.norm(error))
                    if reached_error <= tolerance:
                        break
                    jacp = np.zeros((3, self.model.nv))
                    jacr = np.zeros((3, self.model.nv))
                    self.mujoco.mj_jacGeom(
                        self.model, self.data, jacp, jacr,
                        self.tips[finger].tip_geom_id,
                    )
                    jac = jacp[:, dof_ids]
                    step = jac.T @ np.linalg.solve(
                        jac @ jac.T + damping * np.eye(3), error
                    )
                    step = np.clip(step, -max_step, max_step)
                    trial = q_pre.copy()
                    trial[local_ids] = np.clip(
                        trial[local_ids] + step,
                        self.lower[local_ids], self.upper[local_ids],
                    )
                    q_pre = trial
                self._set_q(q_pre)
                measurement = self._tip_measurement(finger)
                collision = self._collision_diagnostics()
                cube_contact = any(
                    self.cube_body_id in (
                        int(self.model.geom_bodyid[self.data.contact[index].geom1]),
                        int(self.model.geom_bodyid[self.data.contact[index].geom2]),
                    ) for index in range(self.data.ncon)
                )
                if (measurement["distance_to_cube_m"] >= clearance
                        and collision["max_non_tip_actual_penetration_m"] <= 1e-9
                        and not cube_contact):
                    break
                # Distal links can be closer than the representative geom
                # center predicts.  Increase the Cartesian-normal standoff and
                # resolve; the configured 4--6 mm remains the requested margin,
                # while the adaptive extra is reported for diagnosis.
            self._set_q(q_pre)
            after = self._tip_measurement(finger)
            collision = self._collision_diagnostics()
            diagnostics["fingers"][finger] = {
                "target_face": face,
                "target_surface_cube_m": surface_cube.tolist(),
                "target_precontact_cube_m": target_cube.tolist(),
                "target_precontact_world_m": target_world.tolist(),
                "requested_clearance_m": clearance,
                "adaptive_retreat_extra_m": retreat_extra,
                "precontact_retry_count": retry_count,
                "precontact_ik_iterations": iterations,
                "precontact_ik_reached_error_m": reached_error,
                "distance_to_cube_m": after["distance_to_cube_m"],
                "tip_penetration_m": after["tip_penetration_m"],
                "max_non_tip_actual_penetration_m": collision[
                    "max_non_tip_actual_penetration_m"
                ],
                "selected_qpos": q_pre[section].tolist(),
            }
        self._set_q(q_pre)
        final_collision = self._collision_diagnostics()
        palm_body = int(self.model.site_bodyid[self.palm_site_id])
        hand_bodies = _descendants(self.model, palm_body)
        cube_contacts = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if ((body1 == self.cube_body_id and body2 in hand_bodies)
                    or (body2 == self.cube_body_id and body1 in hand_bodies)):
                cube_contacts += 1
        diagnostics["final_collision"] = final_collision
        diagnostics["cube_hand_contact_count"] = cube_contacts
        diagnostics["no_cube_contact"] = bool(cube_contacts == 0)
        diagnostics["q_pre"] = q_pre.tolist()
        self._set_q(q_star)
        return q_pre, diagnostics

    def _cube_frame(self, world_point) -> np.ndarray:
        rotation = np.asarray(self.data.xmat[self.cube_body_id]).reshape(3, 3)
        return rotation.T @ (
            np.asarray(world_point) - self.data.xpos[self.cube_body_id]
        )

    def _tip_measurement(self, finger: str) -> dict:
        descriptor = self.tips[finger]
        segment = np.zeros(6)
        distance = float(self.mujoco.mj_geomDistance(
            self.model, self.data, descriptor.tip_geom_id,
            self.cube_geom_id, 1.0, segment,
        ))
        tip_surface_world = segment[:3].copy()
        cube_surface_world = segment[3:].copy()
        representative_world = self.data.geom_xpos[descriptor.tip_geom_id].copy()
        body_origin = self.data.xpos[descriptor.distal_body_id]
        distal_axis_world = representative_world - body_origin
        distal_axis_world = distal_axis_world / max(
            float(np.linalg.norm(distal_axis_world)), 1e-12
        )
        direction_world = cube_surface_world - tip_surface_world
        direction_norm = float(np.linalg.norm(direction_world))
        if direction_norm <= 1e-9:
            direction_world = distal_axis_world
        else:
            direction_world /= direction_norm
        rotation = np.asarray(self.data.xmat[self.cube_body_id]).reshape(3, 3)
        direction_cube = rotation.T @ direction_world
        point_cube = self._cube_frame(representative_world)
        surface_point_cube = self._cube_frame(tip_surface_world)
        face = self.topology["faces"][finger]
        target, error = project_to_face_region(
            point_cube, face, self.cube_half_size,
            self.config["target_region"]["tangential_half_extent_m"],
        )
        axis = _FACE_AXIS[face[1]]
        sign = 1.0 if face[0] == "+" else -1.0
        standoff = (
            float(self.config["target_region"][
                "representative_standoff_fraction_of_geom_rbound"
            ])
            * float(self.model.geom_rbound[descriptor.tip_geom_id])
        )
        target[axis] += sign * standoff
        error = point_cube - target
        desired = inward_face_direction(face)
        angle = float(np.degrees(np.arccos(np.clip(
            np.dot(direction_cube, desired), -1.0, 1.0
        ))))
        tangent = [index for index in range(3) if index != axis]
        edge_distance = float(np.min(
            self.cube_half_size[tangent] - np.abs(target[tangent])
        ))
        return {
            "target_face": face,
            "tip_surface_world_m": tip_surface_world,
            "tip_surface_cube_m": surface_point_cube,
            "representative_point_world_m": representative_world,
            "representative_point_cube_m": point_cube,
            "target_representative_point_cube_m": target,
            "representative_standoff_m": standoff,
            "position_error_vector_m": error,
            "position_error_m": float(np.linalg.norm(error)),
            "distance_to_cube_m": distance,
            "tip_penetration_m": max(0.0, -distance),
            "effective_inward_direction_cube": direction_cube,
            "distal_axis_direction_cube": rotation.T @ distal_axis_world,
            "target_inward_direction_cube": desired,
            "normal_alignment_angle_deg": angle,
            "target_edge_distance_m": edge_distance,
        }

    def _collision_diagnostics(self) -> dict:
        cfg = self.config["optimization"]
        clearance = float(cfg["non_tip_collision_clearance_m"])
        distances = {}
        clearance_violations = []
        actual_penetrations = []
        for geom in self.non_tip_hand_geoms:
            distance = float(self.mujoco.mj_geomDistance(
                self.model, self.data, geom, self.cube_geom_id, 0.1, None
            ))
            body_id = int(self.model.geom_bodyid[geom])
            distances[str(geom)] = {
                "geom_name": _name(
                    self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom
                ),
                "body_name": _name(
                    self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
                ),
                "distance_m": distance,
            }
            clearance_violations.append(max(0.0, clearance - distance))
            actual_penetrations.append(max(0.0, -distance))
        self_penetration = 0.0
        self_contacts = 0
        self_collision_pairs = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            name1 = _name(
                self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_BODY, body1
            ) or ""
            name2 = _name(
                self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_BODY, body2
            ) or ""
            match1 = self._finger_pattern.search(name1)
            match2 = self._finger_pattern.search(name2)
            if match1 and match2 and match1.group(1) != match2.group(1):
                self_contacts += 1
                penetration = max(0.0, -float(contact.dist))
                self_penetration += penetration
                self_collision_pairs.append({
                    "body1": name1,
                    "body2": name2,
                    "penetration_m": penetration,
                })
        return {
            "non_tip_cube_distances_m": distances,
            "max_non_tip_clearance_violation_m": max(
                clearance_violations, default=0.0
            ),
            "max_non_tip_actual_penetration_m": max(
                actual_penetrations, default=0.0
            ),
            "self_collision_contact_count": self_contacts,
            "self_collision_penetration_sum_m": self_penetration,
            "self_collision_pairs": self_collision_pairs,
        }

    def _residual(self, q, *, diagnostics: bool = False, stage: dict | None = None):
        self._set_q(q)
        cfg = dict(self.config["optimization"])
        if stage is not None:
            cfg.update(stage)
        result = []
        tip_diagnostics = {}
        optional = set(self.topology.get("optional_fingers", []))
        for finger in [f"finger{index}" for index in range(1, 6)]:
            item = self._tip_measurement(finger)
            tip_diagnostics[finger] = item
            finger_weight = (
                float(cfg["optional_finger_weight"])
                if finger in optional else 1.0
            )
            result.extend(
                np.sqrt(finger_weight * float(cfg["position_weight"]))
                * item["position_error_vector_m"]
                / float(cfg["position_scale_m"])
            )
            dot = float(np.dot(
                item["effective_inward_direction_cube"],
                item["target_inward_direction_cube"],
            ))
            result.append(
                np.sqrt(finger_weight * float(cfg["normal_weight"])) * (1.0 - dot)
                / float(cfg["normal_scale"])
            )
            result.append(
                np.sqrt(finger_weight * float(cfg["collision_weight"]))
                * item["tip_penetration_m"]
                / float(cfg["collision_scale_m"])
            )
        collision = self._collision_diagnostics()
        clearance = float(cfg["non_tip_collision_clearance_m"])
        collision_scale = float(cfg["collision_scale_m"])
        for item in collision["non_tip_cube_distances_m"].values():
            distance = item["distance_m"]
            result.append(
                np.sqrt(float(cfg["collision_weight"]))
                * max(0.0, clearance - distance) / collision_scale
            )
        result.append(
            np.sqrt(float(cfg["self_collision_weight"]))
            * collision["self_collision_penetration_sum_m"] / collision_scale
        )
        result.extend(
            np.sqrt(float(cfg["joint_delta_weight"]))
            * (np.asarray(q) - self.nominal) / self.ranges
        )
        margin = float(cfg["joint_limit_margin_fraction"]) * self.ranges
        limit_weight = np.sqrt(float(cfg["joint_limit_weight"]))
        result.extend(limit_weight * np.maximum(
            0.0, self.lower + margin - q
        ) / margin)
        result.extend(limit_weight * np.maximum(
            0.0, q - (self.upper - margin)
        ) / margin)
        residual = np.asarray(result, dtype=float)
        if diagnostics:
            return residual, tip_diagnostics, collision
        return residual

    def _optimize_stage(self, q: np.ndarray, stage: dict
                        ) -> tuple[np.ndarray, int, float, float]:
        cfg = dict(self.config["optimization"])
        cfg.update(stage)
        residual = self._residual(q, stage=stage)
        initial_cost = 0.5 * float(np.dot(residual, residual))
        cost = initial_cost
        damping = float(cfg["damping"])
        epsilon = float(cfg["finite_difference_rad"])
        iterations = 0
        for iterations in range(1, int(cfg["max_iterations"]) + 1):
            jacobian = np.empty((len(residual), len(q)))
            for column in range(len(q)):
                perturbed = q.copy()
                perturbed[column] = min(self.upper[column], q[column] + epsilon)
                actual_step = perturbed[column] - q[column]
                if actual_step <= 1e-12:
                    perturbed[column] = max(self.lower[column], q[column] - epsilon)
                    actual_step = perturbed[column] - q[column]
                jacobian[:, column] = (
                    self._residual(perturbed, stage=stage) - residual
                ) / actual_step
            lhs = jacobian.T @ jacobian + damping * np.eye(len(q))
            step = np.linalg.solve(lhs, -jacobian.T @ residual)
            arm_count = len(self.arm_joint_ids)
            step[:arm_count] = np.clip(
                step[:arm_count], -float(cfg["max_arm_step_rad"]),
                float(cfg["max_arm_step_rad"]),
            )
            step[arm_count:] = np.clip(
                step[arm_count:], -float(cfg["max_finger_step_rad"]),
                float(cfg["max_finger_step_rad"]),
            )
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                trial = np.clip(q + scale * step, self.lower, self.upper)
                trial_residual = self._residual(trial, stage=stage)
                trial_cost = 0.5 * float(np.dot(trial_residual, trial_residual))
                if trial_cost < cost:
                    improvement = cost - trial_cost
                    q, residual, cost = trial, trial_residual, trial_cost
                    accepted = True
                    damping = max(float(cfg["damping"]), damping * 0.7)
                    break
            if not accepted:
                damping *= 10.0
            elif improvement < 1e-7:
                break
        return q, iterations, initial_cost, cost

    def solve(self, *, initial_q=None, initialization_name: str = "current_grasp",
              initialization_diagnostics: dict | None = None
              ) -> ContactGraspSolution:
        cfg = self.config["optimization"]
        q = self.nominal.copy() if initial_q is None else np.asarray(initial_q, dtype=float)
        if q.shape != self.nominal.shape or not np.isfinite(q).all():
            raise ValueError("initial_q must be finite and match the decision variables")
        q = np.clip(q, self.lower, self.upper)
        initial_residual = self._residual(q)
        initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
        stage_history = []
        iterations = 0
        for stage in cfg.get("stages", [{
            "name": "joint_contact_solve",
            "max_iterations": cfg["max_iterations"],
        }]):
            q, stage_iterations, stage_initial, stage_final = self._optimize_stage(
                q, stage
            )
            iterations += stage_iterations
            stage_history.append({
                "name": stage["name"],
                "iterations": stage_iterations,
                "initial_cost": stage_initial,
                "final_cost": stage_final,
            })
        residual, tips, collision = self._residual(q, diagnostics=True)
        tolerance = float(cfg["contact_position_tolerance_m"])
        normal_tolerance = float(cfg["normal_alignment_tolerance_deg"])
        edge_clearance = float(
            self.config["target_region"]["required_edge_clearance_m"]
        )
        penetration_tolerance = float(cfg["max_kinematic_penetration_m"])
        required = tuple(self.topology.get(
            "required_fingers", [f"finger{index}" for index in range(1, 6)]
        ))
        optional = tuple(self.topology.get("optional_fingers", []))
        satisfied = tuple(finger for finger in required if (
            tips[finger]["position_error_m"] <= tolerance
            and tips[finger]["normal_alignment_angle_deg"] <= normal_tolerance
            and tips[finger]["target_edge_distance_m"] >= edge_clearance
            and tips[finger]["tip_penetration_m"] <= penetration_tolerance
        ))
        success = len(satisfied) == len(required) and (
            collision["max_non_tip_actual_penetration_m"]
            <= penetration_tolerance
            and collision["self_collision_penetration_sum_m"]
            <= penetration_tolerance
        )
        margin = float(cfg["joint_limit_margin_fraction"]) * self.ranges
        hits = tuple(
            _name(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint)
            or f"joint_{joint}"
            for index, joint in enumerate(self.joint_ids)
            if q[index] <= self.lower[index] + margin[index]
            or q[index] >= self.upper[index] - margin[index]
        )
        palm_quaternion = np.zeros(4)
        self.mujoco.mju_mat2Quat(
            palm_quaternion, self.data.site_xmat[self.palm_site_id]
        )
        plain_tips = {
            finger: {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in item.items()}
            for finger, item in tips.items()
        }
        return ContactGraspSolution(
            topology=self.topology["name"],
            initialization=initialization_name,
            success=bool(success),
            iterations=iterations,
            initial_cost=initial_cost,
            final_cost=0.5 * float(np.dot(residual, residual)),
            variable_joint_ids=tuple(self.joint_ids),
            variable_joint_names=tuple(
                _name(self.mujoco, self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint)
                or f"joint_{joint}" for joint in self.joint_ids
            ),
            arm_joint_ids=tuple(self.arm_joint_ids),
            hand_joint_ids=tuple(self.hand_joint_ids),
            qpos=tuple(float(value) for value in q),
            palm_position_world_m=tuple(
                float(value) for value in self.data.site_xpos[self.palm_site_id]
            ),
            palm_quaternion_world_wxyz=tuple(float(value) for value in palm_quaternion),
            fingertip_diagnostics=plain_tips,
            collision_diagnostics=collision,
            joint_limit_hits=hits,
            required_fingers=required,
            required_fingers_satisfied=satisfied,
            optional_fingers=optional,
            initialization_diagnostics=(
                {} if initialization_diagnostics is None
                else initialization_diagnostics
            ),
            stage_history=tuple(stage_history),
        )


def _actuators_for_joints(model, joint_ids: Sequence[int]) -> np.ndarray:
    result = []
    for joint in joint_ids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matches) != 1:
            raise ValueError(f"joint {joint} does not have exactly one actuator")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=int)


def _smoothstep(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def execute_static_grasp_test(robot, task, solution: ContactGraspSolution,
                              config: dict, *, close_direction,
                              apply_candidate: bool = True) -> dict:
    """Execute independent finger targets, squeeze, then remove table support."""
    import mujoco

    model, data = robot.model, robot.data
    execution = config["execution"]
    static = config["static_test"]
    joint_ids = np.asarray(solution.variable_joint_ids, dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    actuator_ids = _actuators_for_joints(model, joint_ids)
    arm_count = len(solution.arm_joint_ids)
    hand_count = len(solution.hand_joint_ids)
    start = data.qpos[qpos_ids].copy()
    target = np.asarray(solution.qpos, dtype=float) if apply_candidate else start.copy()
    dt = float(model.opt.timestep)

    def drive(destination, duration_s):
        origin = data.ctrl[actuator_ids].copy()
        steps = max(1, int(round(float(duration_s) / dt)))
        for step in range(1, steps + 1):
            alpha = _smoothstep(step / steps)
            data.ctrl[actuator_ids] = origin + alpha * (destination - origin)
            mujoco.mj_step(model, data)

    if apply_candidate:
        drive(target, execution["move_to_candidate_s"])
        squeeze = target.copy()
        direction = np.asarray(close_direction, dtype=float)
        if direction.shape != (hand_count,):
            raise ValueError("close_direction must match independent hand joints")
        increment = float(execution["squeeze_increment_rad"])
        for finger_index in range(5):
            finger = f"finger{finger_index + 1}"
            gain = float(execution["squeeze_gain"][finger])
            section = slice(arm_count + 4 * finger_index,
                            arm_count + 4 * (finger_index + 1))
            squeeze[section] += gain * increment * direction[4 * finger_index:4 * (finger_index + 1)]
        lower = model.jnt_range[joint_ids, 0]
        upper = model.jnt_range[joint_ids, 1]
        squeeze = np.clip(squeeze, lower, upper)
        drive(squeeze, execution["squeeze_s"])
        drive(squeeze, execution["settle_s"])

    before = task.task_telemetry()
    lock_arm_qpos = data.qpos[qpos_ids[:arm_count]].copy()
    anchor_position = before["object_relative_position_m"].copy()
    anchor_quaternion = before["object_relative_quaternion_wxyz"].copy()
    palm_start = before["grasp_center_position_m"].copy()
    table_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, config["table_geom_name"]
    )
    table_contype = int(model.geom_contype[table_id])
    table_conaffinity = int(model.geom_conaffinity[table_id])
    model.geom_contype[table_id] = 0
    model.geom_conaffinity[table_id] = 0
    mujoco.mj_forward(model, data)
    arm_qpos_ids = qpos_ids[:arm_count]
    arm_dof_ids = dof_ids[:arm_count]
    arm_actuators = actuator_ids[:arm_count]
    hold_steps = int(round(float(static["hold_s"]) / dt))
    max_translation = max_rotation = max_palm_drift = 0.0
    first_complete_loss = None
    min_groups = 5
    contact_counts = []
    try:
        for step in range(hold_steps):
            data.qpos[arm_qpos_ids] = lock_arm_qpos
            data.qvel[arm_dof_ids] = 0.0
            data.ctrl[arm_actuators] = lock_arm_qpos
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            data.qpos[arm_qpos_ids] = lock_arm_qpos
            data.qvel[arm_dof_ids] = 0.0
            mujoco.mj_forward(model, data)
            telemetry = task.task_telemetry()
            translation, rotation = pose_drift(
                anchor_position, anchor_quaternion,
                telemetry["object_relative_position_m"],
                telemetry["object_relative_quaternion_wxyz"],
            )
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
            max_palm_drift = max(max_palm_drift, float(np.linalg.norm(
                telemetry["grasp_center_position_m"] - palm_start
            )))
            groups = sum(
                force >= float(static["min_normal_force_n"])
                for force in telemetry["finger_normal_forces_n"].values()
            )
            min_groups = min(min_groups, groups)
            if step % max(1, round((1.0 / 30.0) / dt)) == 0:
                contact_counts.append(groups)
            if groups == 0 and first_complete_loss is None:
                first_complete_loss = (step + 1) * dt
        after = task.task_telemetry()
    finally:
        model.geom_contype[table_id] = table_contype
        model.geom_conaffinity[table_id] = table_conaffinity
    final_groups = sum(
        force >= float(static["min_normal_force_n"])
        for force in after["finger_normal_forces_n"].values()
    )
    survival = (float(static["hold_s"]) if first_complete_loss is None
                else first_complete_loss)
    static_success = bool(
        first_complete_loss is None
        and final_groups >= int(static["min_finger_groups"])
        and max_translation < float(static["max_relative_translation_drift_m"])
        and max_rotation < float(static["max_relative_rotation_drift_deg"])
    )
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value
    def enriched_contacts(contacts):
        result = []
        for contact in contacts:
            item = plain(contact)
            force = np.asarray(contact["force_on_cube_world_n"], dtype=float)
            normal = np.asarray(contact["normal_on_cube_world"], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            tangent = force - np.dot(force, normal) * normal
            item["tangential_force_world_n"] = tangent.tolist()
            item["tangential_force_magnitude_n"] = float(np.linalg.norm(tangent))
            result.append(item)
        return result
    return {
        "static_success": static_success,
        "survival_time_s": survival,
        "complete_contact_loss_time_s": first_complete_loss,
        "max_relative_translation_drift_m": max_translation,
        "max_relative_rotation_drift_deg": max_rotation,
        "max_palm_translation_drift_m": max_palm_drift,
        "minimum_finger_groups": min_groups,
        "final_finger_groups": final_groups,
        "contact_groups_30hz": contact_counts,
        "pre_removal_contacts": enriched_contacts(before["contacts"]),
        "pre_removal_wrench": {
            "net_force_world_n": plain(before["contact_resultant_force_world_n"]),
            "horizontal_force_n": float(np.linalg.norm(
                before["contact_resultant_force_world_n"][:2]
            )),
            "net_moment_world_nm": plain(
                before["contact_resultant_moment_about_cube_world_nm"]
            ),
            "net_moment_magnitude_nm": float(np.linalg.norm(
                before["contact_resultant_moment_about_cube_world_nm"]
            )),
        },
        "final_contacts": enriched_contacts(after["contacts"]),
    }


def execute_contact_acquisition(robot, task, solution: ContactGraspSolution,
                                optimizer: ContactRegionGraspOptimizer,
                                config: dict, *, open_hand, close_direction,
                                strategy: str = "synchronized",
                                allow_squeeze: bool = True,
                                lock_inactive_fingers: bool = False,
                                smooth_control_substeps: bool = False) -> dict:
    """Execute q_pre -> per-finger contact acquisition -> hold -> light squeeze.

    This is intentionally a contact-topology experiment.  The table stays
    enabled, the arm is locked after pregrasp, and no lift or gravity-only test
    is performed.  Independent finger targets are used only by this diagnostic
    executor; the LeRobot policy observation/action contracts are untouched.
    """
    import mujoco

    if strategy not in {"synchronized", "thumb_first", "fingers_first"}:
        raise ValueError(f"unknown acquisition strategy: {strategy}")
    acq = config["acquisition"]
    model, data = robot.model, robot.data
    q_star = np.asarray(solution.qpos, dtype=float)
    q_pre, pregrasp_diag = optimizer.build_pregrasp(
        q_star, open_hand=open_hand,
        required_fingers=solution.required_fingers,
    )
    path_diag = optimizer.swept_path_diagnostics(q_pre, q_star)
    joint_ids = np.asarray(solution.variable_joint_ids, dtype=int)
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    actuator_ids = _actuators_for_joints(model, joint_ids)
    arm_count = len(solution.arm_joint_ids)
    hand_count = len(solution.hand_joint_ids)
    dt = float(model.opt.timestep)
    control_hz = float(acq.get("control_hz", 30.0))
    control_steps = max(1, int(round(1.0 / (control_hz * dt))))
    direction = np.asarray(close_direction, dtype=float)
    if direction.shape != (hand_count,):
        raise ValueError("close_direction must match independent hand joints")
    open_hand = np.asarray(open_hand, dtype=float)

    def lock_arm():
        data.qpos[qpos_ids[:arm_count]] = q_star[:arm_count]
        data.qvel[dof_ids[:arm_count]] = 0.0
        data.ctrl[actuator_ids[:arm_count]] = q_star[:arm_count]

    def lock_fingers(q_command, active_fingers):
        if not lock_inactive_fingers:
            return
        active_fingers = set(active_fingers)
        for index in range(5):
            finger = f"finger{index + 1}"
            if finger in active_fingers:
                continue
            section = slice(arm_count + 4 * index, arm_count + 4 * (index + 1))
            data.qpos[qpos_ids[section]] = q_command[section]
            data.qvel[dof_ids[section]] = 0.0
            data.ctrl[actuator_ids[section]] = q_command[section]

    def step_command(q_command, *, active_fingers):
        start_command = data.ctrl[actuator_ids].copy()
        for substep in range(control_steps):
            alpha = ((substep + 1) / control_steps
                     if smooth_control_substeps else 1.0)
            subcommand = start_command + alpha * (q_command - start_command)
            data.ctrl[actuator_ids] = subcommand
            lock_arm()
            lock_fingers(subcommand, active_fingers)
            mujoco.mj_step(model, data)
            lock_arm()
            lock_fingers(subcommand, active_fingers)
            mujoco.mj_forward(model, data)

    def drive_pregrasp():
        # q_pre is the explicit collision-free pre-contact state.  Place the
        # simulator at this state before stepping the acquisition controller;
        # this keeps the experiment focused on contact acquisition rather than
        # attributing an earlier closed-grasp retreat path to a strategy.
        data.qpos[qpos_ids] = q_pre
        # The preceding seed grasp can leave the free cube with residual
        # velocity.  A contact-acquisition trial starts from a stationary cube
        # on the table, so clear all generalized velocities after teleporting.
        data.qvel.fill(0.0)
        data.ctrl[actuator_ids] = q_pre
        mujoco.mj_forward(model, data)

    def sync_optimizer():
        optimizer.data.qpos[:] = data.qpos
        optimizer.data.qvel[:] = data.qvel
        optimizer.data.ctrl[:] = data.ctrl
        mujoco.mj_forward(optimizer.model, optimizer.data)

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    def contacts_by_finger():
        telemetry = task.task_telemetry()
        groups = {}
        for contact in telemetry["contacts"]:
            finger = contact["finger"]
            # Prefer the designated fingertip whenever it is present.  A
            # stronger distal-pad contact must remain visible in telemetry but
            # cannot mask a valid fingertip contact for the acquisition gate.
            if finger not in groups:
                groups[finger] = contact
            elif contact.get("is_designated_tip", False) and not groups[finger].get("is_designated_tip", False):
                groups[finger] = contact
            elif contact.get("is_designated_tip", False) == groups[finger].get("is_designated_tip", False) and contact["normal_force_n"] > groups[finger]["normal_force_n"]:
                groups[finger] = contact
        return telemetry, groups

    def valid_contact(finger, contact):
        if contact is None:
            return False, "no_contact", None
        if not bool(contact.get("is_designated_tip", False)):
            reason = "non_tip_contact"
            if bool(contact.get("is_distal_non_tip", False)):
                reason = "non_tip_contact"
            return False, reason, {
                "geom_id": contact.get("other_geom_id"),
                "geom_name": contact.get("other_geom_name"),
                "contact_role": contact.get("contact_role", "distal_non_tip_geom"),
            }
        target_face = solution.fingertip_diagnostics.get(finger, {}).get(
            "target_face", optimizer.topology["faces"][finger]
        )
        actual_face = contact.get("cube_face")
        edge = bool(contact.get("edge_contact", False))
        edge_distance = float(contact.get("distance_to_nearest_edge_m", 0.0))
        normal = np.asarray(contact.get("normal_on_cube_cube", [0.0, 0.0, 0.0]), dtype=float)
        desired = inward_face_direction(target_face)
        alignment = float(np.degrees(np.arccos(np.clip(
            np.dot(normal, desired) / max(np.linalg.norm(normal), 1e-12), -1.0, 1.0
        ))))
        if actual_face != target_face:
            return False, "tip_wrong_face", {"actual_face": actual_face, "alignment_deg": alignment}
        if edge or edge_distance < float(acq["edge_margin_m"]):
            return False, "tip_edge_contact", {"actual_face": actual_face, "edge_distance_m": edge_distance, "alignment_deg": alignment}
        if alignment > float(acq["normal_alignment_max_deg"]):
            return False, "tip_wrong_face", {"actual_face": actual_face, "alignment_deg": alignment}
        if float(contact.get("normal_force_n", 0.0)) < float(acq["min_normal_force_n"]):
            return False, "tip_contact_candidate", {"actual_face": actual_face, "alignment_deg": alignment}
        return True, "tip_contact_acquired", {"actual_face": actual_face, "edge_distance_m": edge_distance, "alignment_deg": alignment}

    required = tuple(solution.required_fingers)
    states = {finger: "PREGRASP" for finger in required}
    confirm = {finger: 0 for finger in required}
    q_command = q_pre.copy()
    drive_pregrasp()
    telemetry, groups = contacts_by_finger()
    infeasible_fingers = [
        finger for finger, item in path_diag["per_finger"].items()
        if finger in solution.required_fingers and item["path_infeasible"]
    ]
    if groups:
        # A pregrasp that already touches the cube cannot be used to attribute
        # the first push/contact to the approach controller.
        pregrasp_non_tip = any(
            not bool(item.get("is_designated_tip", False))
            for item in telemetry.get("contacts", [])
        )
        return plain({
            "strategy": strategy,
            "outcome": "path_infeasible" if infeasible_fingers else ("non_tip_contact" if pregrasp_non_tip else "pregrasp_failed"),
            "kinematic_candidate_valid": bool(solution.success),
            "q_pre": q_pre,
            "q_star": q_star,
            "pregrasp_diagnostics": {
                **pregrasp_diag,
                "swept_path": path_diag,
                "cube_hand_contact_count_after_move": len(groups),
            },
            "states": {finger: "PREGRASP" for finger in solution.required_fingers},
            "first_contact": None,
            "topology_acquired": False,
            "topology_held": False,
            "acquisition_disturbed": False,
            "failure_reason": (
                f"non_tip_path_penetration:{','.join(infeasible_fingers)}"
                if infeasible_fingers else
                "non_tip_contact" if pregrasp_non_tip else "pregrasp_contact"
            ),
            "max_cube_translation_m": 0.0,
            "max_cube_rotation_deg": 0.0,
            "max_cube_linear_velocity_m_s": 0.0,
            "max_cube_angular_velocity_deg_s": 0.0,
            "final_required_contact_count": 0,
            "time_series": [],
        })
    if infeasible_fingers:
        return plain({
            "strategy": strategy,
            "outcome": "path_infeasible",
            "kinematic_candidate_valid": bool(solution.success),
            "q_pre": q_pre, "q_star": q_star,
            "pregrasp_diagnostics": {**pregrasp_diag, "swept_path": path_diag},
            "states": {finger: "PREGRASP" for finger in solution.required_fingers},
            "first_contact": None, "topology_acquired": False,
            "topology_held": False, "acquisition_disturbed": False,
            "failure_reason": f"non_tip_path_penetration:{','.join(infeasible_fingers)}",
            "max_cube_translation_m": 0.0, "max_cube_rotation_deg": 0.0,
            "max_cube_linear_velocity_m_s": 0.0,
            "max_cube_angular_velocity_deg_s": 0.0,
            "final_required_contact_count": 0, "time_series": [],
        })
    anchor_position = telemetry["cube_position_m"].copy()
    anchor_quaternion = telemetry["cube_quaternion_wxyz"].copy()
    acquisition_start = float(data.time)
    rows = []
    first_contact = None
    failure_reason = None
    acquisition_disturbed = False
    max_translation = max_rotation = 0.0
    max_linear_velocity = max_angular_velocity = 0.0

    def row(step_index, phase, telemetry, groups):
        nonlocal max_translation, max_rotation, max_linear_velocity, max_angular_velocity
        translation, rotation = pose_drift(
            anchor_position, anchor_quaternion,
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"],
        )
        linear = float(np.linalg.norm(telemetry["cube_linear_velocity_world_m_s"]))
        angular = float(np.degrees(np.linalg.norm(telemetry["cube_angular_velocity_world_rad_s"])))
        max_translation = max(max_translation, translation)
        max_rotation = max(max_rotation, rotation)
        max_linear_velocity = max(max_linear_velocity, linear)
        max_angular_velocity = max(max_angular_velocity, angular)
        per_finger = {}
        sync_optimizer()
        for finger in required:
            measurement = optimizer._tip_measurement(finger)
            contact = groups.get(finger)
            per_finger[finger] = {
                "state": states[finger],
                "target_face": optimizer.topology["faces"][finger],
                "actual_face": None if contact is None else contact.get("cube_face"),
                "other_geom_id": None if contact is None else contact.get("other_geom_id"),
                "other_geom_name": None if contact is None else contact.get("other_geom_name"),
                "other_body_name": None if contact is None else contact.get("other_body_name"),
                "is_distal_link": None if contact is None else contact.get("is_distal_link"),
                "is_designated_tip": None if contact is None else contact.get("is_designated_tip"),
                "is_distal_non_tip": None if contact is None else contact.get("is_distal_non_tip"),
                "contact_role": None if contact is None else contact.get("contact_role"),
                "contact_point_world_m": None if contact is None else plain(contact.get("position_world_m")),
                "edge_distance_m": None if contact is None else contact.get("distance_to_nearest_edge_m"),
                "normal_world": None if contact is None else plain(contact.get("normal_on_cube_world")),
                "normal_force_n": 0.0 if contact is None else contact.get("normal_force_n", 0.0),
                "position_error_m": measurement["position_error_m"],
                "tip_position_world_m": plain(measurement["representative_point_world_m"]),
                "qpos_rad": q_command[arm_count + 4 * int(finger[-1]) - 4:arm_count + 4 * int(finger[-1])].tolist(),
                "qtarget_rad": q_star[arm_count + 4 * int(finger[-1]) - 4:arm_count + 4 * int(finger[-1])].tolist(),
            }
        return {
            "step": step_index, "time_s": float(data.time - acquisition_start),
            "phase": phase, "cube_translation_from_start_m": translation,
            "cube_rotation_from_start_deg": rotation,
            "cube_linear_velocity_world_m_s": plain(telemetry["cube_linear_velocity_world_m_s"]),
            "cube_angular_velocity_world_rad_s": plain(telemetry["cube_angular_velocity_world_rad_s"]),
            "required_contact_count": sum(finger in groups and groups[finger]["normal_force_n"] >= float(acq["min_normal_force_n"]) for finger in required),
            "face_topology": {finger: groups[finger].get("cube_face") for finger in groups},
            "net_force_world_n": plain(telemetry["contact_resultant_force_world_n"]),
            "net_force_magnitude_n": float(np.linalg.norm(telemetry["contact_resultant_force_world_n"])),
            "net_moment_world_nm": plain(telemetry["contact_resultant_moment_about_cube_world_nm"]),
            "net_moment_magnitude_nm": float(np.linalg.norm(telemetry["contact_resultant_moment_about_cube_world_nm"])),
            "fingers": per_finger,
        }

    def disturbed(telemetry):
        translation, rotation = pose_drift(
            anchor_position, anchor_quaternion,
            telemetry["cube_position_m"], telemetry["cube_quaternion_wxyz"],
        )
        return (
            translation > float(acq["max_cube_translation_before_topology_m"])
            or rotation > float(acq["max_cube_rotation_before_topology_deg"])
            or np.linalg.norm(telemetry["cube_linear_velocity_world_m_s"]) > float(acq["max_cube_linear_velocity_m_s"])
            or np.degrees(np.linalg.norm(telemetry["cube_angular_velocity_world_rad_s"])) > float(acq["max_cube_angular_velocity_deg_s"])
        )

    max_steps = int(acq["max_acquisition_steps"])
    for step_index in range(max_steps):
        telemetry, groups = contacts_by_finger()
        if groups and first_contact is None:
            first_finger = sorted(groups)[0]
            first_item = groups[first_finger]
            first_contact = {
                "finger": first_finger,
                "time_s": float(data.time - acquisition_start),
                "force_n": float(first_item.get("normal_force_n", 0.0)),
                "face": first_item.get("cube_face"),
                "other_geom_id": first_item.get("other_geom_id"),
                "other_geom_name": first_item.get("other_geom_name"),
                "other_body_name": first_item.get("other_body_name"),
                "is_distal_link": first_item.get("is_distal_link"),
                "is_designated_tip": first_item.get("is_designated_tip"),
                "is_distal_non_tip": first_item.get("is_distal_non_tip"),
                "contact_role": first_item.get("contact_role"),
                "valid_for_target": False,
                "classification": "contact_before_confirmation",
            }
        if disturbed(telemetry) and len([f for f in required if states[f] in {"CONTACT_ACQUIRED", "CONTACT_HOLD", "SQUEEZE"}]) < len(required):
            acquisition_disturbed = True
            non_tip_present = any(
                bool(item.get("is_distal_non_tip", False))
                or item.get("contact_role") == "other_finger_link_geom"
                for item in telemetry.get("contacts", [])
            )
            failure_reason = "non_tip_unilateral_push" if non_tip_present else "unilateral_push"
            break
        active = list(required)
        if strategy == "thumb_first" and states["finger1"] not in {"CONTACT_ACQUIRED", "CONTACT_HOLD"}:
            active = ["finger1"]
        elif strategy == "thumb_first":
            active = [finger for finger in required if finger != "finger1"]
        elif strategy == "fingers_first" and any(states[finger] not in {"CONTACT_ACQUIRED", "CONTACT_HOLD"} for finger in required if finger != "finger1"):
            active = [finger for finger in required if finger != "finger1"]
        elif strategy == "fingers_first":
            active = ["finger1"]
        for finger in required:
            contact = groups.get(finger)
            valid, reason, _ = valid_contact(finger, contact)
            if contact is not None and first_contact is None:
                first_contact = {
                    "finger": finger,
                    "time_s": float(data.time - acquisition_start),
                    "force_n": float(contact.get("normal_force_n", 0.0)),
                    "face": contact.get("cube_face"),
                    "other_geom_id": contact.get("other_geom_id"),
                    "other_geom_name": contact.get("other_geom_name"),
                    "other_body_name": contact.get("other_body_name"),
                    "is_distal_link": contact.get("is_distal_link"),
                    "valid_for_target": bool(valid),
                    "classification": reason,
                }
            if valid:
                confirm[finger] += 1
                if states[finger] in {"PREGRASP", "APPROACH", "CONTACT_CANDIDATE"}:
                    states[finger] = "CONTACT_CANDIDATE"
                if confirm[finger] >= int(acq["contact_confirm_steps"]):
                    states[finger] = "CONTACT_ACQUIRED"
                    q_command[arm_count + 4 * int(finger[-1]) - 4:arm_count + 4 * int(finger[-1])] = data.qpos[qpos_ids[arm_count + 4 * int(finger[-1]) - 4:arm_count + 4 * int(finger[-1])]]
                    if first_contact is None:
                        first_contact = {
                            "finger": finger,
                            "time_s": float(data.time - acquisition_start),
                            "force_n": contact["normal_force_n"],
                            "face": contact["cube_face"],
                            "other_geom_id": contact.get("other_geom_id"),
                            "other_geom_name": contact.get("other_geom_name"),
                            "other_body_name": contact.get("other_body_name"),
                            "is_designated_tip": contact.get("is_designated_tip"),
                            "is_distal_non_tip": contact.get("is_distal_non_tip"),
                            "contact_role": contact.get("contact_role"),
                            "valid_for_target": True,
                            "classification": reason,
                        }
            else:
                confirm[finger] = 0
                if contact is not None and float(contact.get("normal_force_n", 0.0)) >= float(acq["min_normal_force_n"]):
                    states[finger] = "CONTACT_CANDIDATE"
                    if failure_reason is None and reason in {"tip_wrong_face", "tip_edge_contact", "non_tip_contact"}:
                        failure_reason = reason
                elif states[finger] not in {"CONTACT_ACQUIRED", "CONTACT_HOLD"}:
                    states[finger] = "APPROACH"
        for finger in active:
            if states[finger] in {"CONTACT_ACQUIRED", "CONTACT_HOLD"}:
                continue
            index = int(finger[-1]) - 1
            section = slice(arm_count + 4 * index, arm_count + 4 * (index + 1))
            sync_optimizer()
            distance = optimizer._tip_measurement(finger)["position_error_m"]
            scale = float(np.clip(distance / 0.01, 0.15, 1.0))
            step_rad = float(acq["approach_step_rad"]) * scale
            if strategy == "synchronized":
                step_rad *= 0.75 + 0.25 * scale
            q_command[section] = np.clip(
                q_command[section] + np.clip(q_star[section] - q_command[section], -step_rad, step_rad),
                model.jnt_range[joint_ids[arm_count + 4 * index:arm_count + 4 * (index + 1)], 0],
                model.jnt_range[joint_ids[arm_count + 4 * index:arm_count + 4 * (index + 1)], 1],
            )
            if states[finger] == "PREGRASP":
                states[finger] = "APPROACH"
        dynamic_fingers = set(active)
        dynamic_fingers.update(
            finger for finger in required
            if states[finger] in {"CONTACT_ACQUIRED", "CONTACT_HOLD", "SQUEEZE"}
        )
        step_command(q_command, active_fingers=dynamic_fingers)
        telemetry, groups = contacts_by_finger()
        rows.append(row(step_index, "approach", telemetry, groups))
        if all(states[finger] == "CONTACT_ACQUIRED" for finger in required):
            break

    topology_acquired = all(states[finger] == "CONTACT_ACQUIRED" for finger in required)
    hold_steps = max(1, int(round(float(acq["hold_s"]) * control_hz)))
    if topology_acquired:
        for finger in required:
            states[finger] = "CONTACT_HOLD"
        for step_index in range(hold_steps):
            step_command(q_command, active_fingers=required)
            telemetry, groups = contacts_by_finger()
            if any(not valid_contact(finger, groups.get(finger))[0] for finger in required):
                failure_reason = "contact_lost"
                topology_acquired = False
                break
            rows.append(row(max_steps + step_index, "contact_hold", telemetry, groups))
    topology_held = bool(topology_acquired)
    if topology_held and allow_squeeze:
        squeeze_target = q_command.copy()
        increment = float(acq["squeeze_increment_rad"])
        gains = acq["squeeze_gain"]
        for index in range(5):
            finger = f"finger{index + 1}"
            section = slice(arm_count + 4 * index, arm_count + 4 * (index + 1))
            squeeze_target[section] = np.clip(
                squeeze_target[section] + float(gains.get(finger, 0.0)) * increment * direction[4 * index:4 * (index + 1)],
                model.jnt_range[joint_ids[arm_count + 4 * index:arm_count + 4 * (index + 1)], 0],
                model.jnt_range[joint_ids[arm_count + 4 * index:arm_count + 4 * (index + 1)], 1],
            )
        squeeze_steps = max(1, int(round(float(acq["squeeze_s"]) * control_hz)))
        for step_index in range(squeeze_steps):
            alpha = (step_index + 1) / squeeze_steps
            q_step = q_command + alpha * (squeeze_target - q_command)
            step_command(q_step, active_fingers=[f"finger{index}" for index in range(1, 6)])
            telemetry, groups = contacts_by_finger()
            q_command = q_step
            if any(not valid_contact(finger, groups.get(finger))[0] for finger in required) or disturbed(telemetry):
                failure_reason = "squeeze_destabilized"
                topology_held = False
                break
            for finger in required:
                states[finger] = "SQUEEZE"
            rows.append(row(max_steps + hold_steps + step_index, "squeeze", telemetry, groups))
        if topology_held:
            squeeze_hold_steps = max(1, int(round(float(acq["squeeze_hold_s"]) * control_hz)))
            for step_index in range(squeeze_hold_steps):
                step_command(q_command, active_fingers=[f"finger{index}" for index in range(1, 6)])
                telemetry, groups = contacts_by_finger()
                if any(not valid_contact(finger, groups.get(finger))[0] for finger in required) or disturbed(telemetry):
                    failure_reason = "squeeze_destabilized"
                    topology_held = False
                    break
                rows.append(row(max_steps + hold_steps + squeeze_steps + step_index, "squeeze_hold", telemetry, groups))
    outcome = "topology_held" if topology_held else (
        "topology_acquired" if topology_acquired else (failure_reason or "contact_timeout")
    )
    return plain({
        "strategy": strategy,
        "outcome": outcome,
        "kinematic_candidate_valid": bool(solution.success),
        "q_pre": q_pre,
        "q_star": q_star,
        "pregrasp_diagnostics": pregrasp_diag,
        "swept_path_diagnostics": path_diag,
        "states": states,
        "first_contact": first_contact,
        "prediction_vs_actual": {
            finger: {
                "predicted_first_contact_geom": item["predicted_first_contact_geom"],
                "actual_first_contact_geom": (
                    first_contact["other_geom_id"] if first_contact and first_contact["finger"] == finger else None
                ),
                "prediction_matches": bool(
                    first_contact and first_contact["finger"] == finger
                    and item["predicted_first_contact_geom"] == first_contact["other_geom_id"]
                ),
            }
            for finger, item in path_diag["per_finger"].items()
        },
        "topology_acquired": topology_acquired,
        "topology_held": topology_held,
        "acquisition_disturbed": acquisition_disturbed,
        "failure_reason": failure_reason,
        "max_cube_translation_m": max_translation,
        "max_cube_rotation_deg": max_rotation,
        "max_cube_linear_velocity_m_s": max_linear_velocity,
        "max_cube_angular_velocity_deg_s": max_angular_velocity,
        "final_required_contact_count": sum(states[finger] in {"CONTACT_HOLD", "SQUEEZE"} for finger in required),
        "time_series": rows,
    })


def execute_single_finger_path_test(robot, task, solution: ContactGraspSolution,
                                    optimizer: ContactRegionGraspOptimizer,
                                    config: dict, *, finger: str,
                                    open_hand, close_direction) -> dict:
    """Run one finger toward its target while all other fingers stay locked."""
    if finger not in solution.required_fingers:
        raise ValueError(f"{finger} is not a required finger")
    single = replace(
        solution,
        required_fingers=(finger,),
        required_fingers_satisfied=(finger,) if finger in solution.required_fingers_satisfied else (),
    )
    result = execute_contact_acquisition(
        robot, task, single, optimizer, config,
        open_hand=open_hand, close_direction=close_direction,
        strategy="synchronized", allow_squeeze=False,
        lock_inactive_fingers=True,
        smooth_control_substeps=True,
    )
    result["single_finger"] = finger
    result["outcome"] = (
        "single_finger_path_pass"
        if result.get("topology_held") or result.get("topology_acquired")
        else result.get("outcome", "contact_timeout")
    )
    return result
