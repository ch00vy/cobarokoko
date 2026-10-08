import math

import mujoco
import numpy as np

from smartglove.config import DISTAL_COUPLING, FINGER_JOINTS, IK_ITERATIONS
from smartglove.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_error,
    quaternion_multiply,
    quaternion_relative,
)


class HandQposMapper:
    def __init__(
        self,
        model,
        data,
        flexion_gain: float = 1.0,
        invert_thumb_abduction: bool = False,
    ) -> None:
        if not math.isfinite(flexion_gain) or flexion_gain <= 0.0:
            raise ValueError("flexion_gain harus finite dan lebih besar dari nol.")
        self.model = model
        self.data = data
        self.neutral_qpos = data.qpos.copy()
        self.flexion_gain = flexion_gain
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
        self.fingers = {}

        for finger, specification in FINGER_JOINTS.items():
            joint_ids = [
                self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore[attr-defined]
                for name in specification["joints"]
            ]
            self.fingers[finger] = {
                "qpos": [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids],
                "dof": [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids],
                "joint_ids": joint_ids,
                "flexion_slots": specification["flexion_slots"],
                "flexion_closed_limit": specification.get(
                    "flexion_closed_limit", "high"
                ),
                "invert_abduction": specification.get(
                    "invert_abduction", False
                )
                and (finger != "THUMB" or invert_thumb_abduction),
                "body": self._name_id(
                    mujoco.mjtObj.mjOBJ_BODY,  # type: ignore[attr-defined]
                    specification["body"],
                ),
                "neutral_orientation": normalize_quaternion(
                    data.xquat[
                        self._name_id(
                            mujoco.mjtObj.mjOBJ_BODY,  # type: ignore[attr-defined]
                            specification["body"],
                        )
                    ]
                ),
            }

        self.sensor_to_model = None
        self.sensor_open = None
        self.flexion_calibration = None

    def _name_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)  # type: ignore[attr-defined]
        if object_id < 0:
            raise ValueError(f"Objek MuJoCo tidak ditemukan: {name}")
        return object_id

    def calibrate(self, sensor_neutral: dict[str, np.ndarray]) -> None:
        self.sensor_to_model = {}
        self.sensor_open = {
            finger: normalize_quaternion(quaternion)
            for finger, quaternion in sensor_neutral.items()
        }
        self.flexion_calibration = None
        for finger, sensor_quaternion in sensor_neutral.items():
            model_quaternion = self.fingers[finger]["neutral_orientation"]
            self.sensor_to_model[finger] = normalize_quaternion(
                quaternion_multiply(
                    model_quaternion,
                    quaternion_conjugate(sensor_quaternion),
                )
            )

    def calibrate_grip(self, sensor_grip: dict[str, np.ndarray]) -> list[str]:
        if self.sensor_to_model is None or self.sensor_open is None:
            raise RuntimeError("Kalibrasi pose terbuka harus dilakukan lebih dulu.")

        calibration = {}
        skipped_fingers = []
        for finger, sensor_quaternion in sensor_grip.items():
            relative = quaternion_relative(
                self.sensor_open[finger], sensor_quaternion
            )
            axis_magnitude = float(np.linalg.norm(relative[1:]))
            closed_angle = 2.0 * math.atan2(axis_magnitude, float(relative[0]))
            if closed_angle < math.radians(5.0):
                calibration[finger] = None
                skipped_fingers.append(finger)
                continue
            closure_axis = relative[1:] / axis_magnitude

            finger_data = self.fingers[finger]
            finger_calibration = {}
            for slot in finger_data["flexion_slots"]:
                qpos_id = finger_data["qpos"][slot]
                joint_id = finger_data["joint_ids"][slot]
                if not self.model.jnt_limited[joint_id]:
                    raise ValueError(
                        f"Joint {finger} slot {slot} tidak memiliki batas sudut."
                    )
                low, high = self.model.jnt_range[joint_id]
                target_closed = (
                    low
                    if finger_data["flexion_closed_limit"] == "low"
                    else high
                )
                finger_calibration[qpos_id] = (
                    float(self.neutral_qpos[qpos_id]),
                    float(target_closed),
                    joint_id,
                )
            calibration[finger] = (closure_axis, closed_angle, finger_calibration)

        if len(skipped_fingers) == len(sensor_grip):
            raise ValueError(
                "Tidak ada jari yang menunjukkan perubahan cukup antara pose "
                "terbuka dan genggam."
            )
        self.flexion_calibration = calibration
        return skipped_fingers

    def _bounds(self, finger_data: dict) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(3, -np.inf)
        upper = np.full(3, np.inf)
        joint_ids = finger_data["joint_ids"]
        qpos_ids = finger_data["qpos"]

        for variable, joint_id in enumerate(joint_ids[:3]):
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                neutral = self.neutral_qpos[qpos_ids[variable]]
                lower[variable] = low - neutral
                upper[variable] = high - neutral

        distal_id = joint_ids[3]
        if self.model.jnt_limited[distal_id]:
            low, high = self.model.jnt_range[distal_id]
            neutral = self.neutral_qpos[qpos_ids[3]]
            lower[2] = max(lower[2], (low - neutral) / DISTAL_COUPLING)
            upper[2] = min(upper[2], (high - neutral) / DISTAL_COUPLING)

        return lower, upper

    def _set_finger_variables(
        self,
        finger_data: dict,
        variables: np.ndarray,
    ) -> None:
        qpos_ids = finger_data["qpos"]
        self.data.qpos[qpos_ids[0]] = self.neutral_qpos[qpos_ids[0]] + variables[0]
        self.data.qpos[qpos_ids[1]] = self.neutral_qpos[qpos_ids[1]] + variables[1]
        self.data.qpos[qpos_ids[2]] = self.neutral_qpos[qpos_ids[2]] + variables[2]
        self.data.qpos[qpos_ids[3]] = (
            self.neutral_qpos[qpos_ids[3]] + DISTAL_COUPLING * variables[2]
        )

    def _fit_finger(
        self,
        finger_data: dict,
        target_orientation: np.ndarray,
        initial_variables: np.ndarray,
    ) -> np.ndarray:
        lower, upper = self._bounds(finger_data)
        variables = np.clip(initial_variables.copy(), lower, upper)
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))

        for _ in range(IK_ITERATIONS):
            self._set_finger_variables(finger_data, variables)
            mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
            current_orientation = normalize_quaternion(
                self.data.xquat[finger_data["body"]]
            )
            error = quaternion_error(target_orientation, current_orientation)

            if np.linalg.norm(error) < 1e-5:
                break

            mujoco.mj_jacBody(  # type: ignore[attr-defined]
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                finger_data["body"],
            )
            dof_ids = finger_data["dof"]
            jacobian = np.column_stack(
                (
                    jacobian_rotation[:, dof_ids[0]],
                    jacobian_rotation[:, dof_ids[1]],
                    jacobian_rotation[:, dof_ids[2]]
                    + DISTAL_COUPLING * jacobian_rotation[:, dof_ids[3]],
                )
            )
            damped_system = jacobian @ jacobian.T + 1e-5 * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(damped_system, error)
            delta = np.clip(delta, -0.15, 0.15)
            variables = np.clip(variables + delta, lower, upper)

        self._set_finger_variables(finger_data, variables)
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        return variables

    def map_frame(self, sensor_quaternions: dict[str, np.ndarray]) -> np.ndarray:
        if self.sensor_to_model is None:
            raise RuntimeError("Mapper harus dikalibrasi sebelum memetakan qpos.")

        for finger, sensor_quaternion in sensor_quaternions.items():
            finger_data = self.fingers[finger]
            target_orientation = normalize_quaternion(
                quaternion_multiply(
                    self.sensor_to_model[finger],
                    sensor_quaternion,
                )
            )
            qpos_ids = finger_data["qpos"]
            initial_variables = np.array(
                [
                    self.data.qpos[qpos_ids[index]]
                    - self.neutral_qpos[qpos_ids[index]]
                    for index in range(3)
                ]
            )
            if finger_data["invert_abduction"]:
                initial_variables[0] *= -1.0
            self._fit_finger(
                finger_data,
                target_orientation,
                initial_variables,
            )
            if finger_data["invert_abduction"]:
                abduction_qpos = qpos_ids[0]
                neutral = self.neutral_qpos[abduction_qpos]
                value = 2.0 * neutral - self.data.qpos[abduction_qpos]
                joint_id = finger_data["joint_ids"][0]
                if self.model.jnt_limited[joint_id]:
                    low, high = self.model.jnt_range[joint_id]
                    value = float(np.clip(value, low, high))
                self.data.qpos[abduction_qpos] = value

        fitted_qpos = self.data.qpos.copy()

        if self.flexion_calibration is not None:
            if self.sensor_open is None:
                raise RuntimeError("Pose terbuka belum dikalibrasi.")
            for finger, finger_calibration_data in self.flexion_calibration.items():
                if finger_calibration_data is None:
                    continue
                closure_axis, closed_angle, finger_calibration = (
                    finger_calibration_data
                )
                relative = quaternion_relative(
                    self.sensor_open[finger], sensor_quaternions[finger]
                )
                progress_angle = 2.0 * math.atan2(
                    float(np.dot(relative[1:], closure_axis)),
                    float(relative[0]),
                )
                progress = float(
                    np.clip(
                        progress_angle / closed_angle,
                        0.0,
                        1.0,
                    )
                )
                for qpos_id, (target_open, target_closed, joint_id) in finger_calibration.items():
                    value = target_open + progress * self.flexion_gain * (target_closed - target_open)
                    low, high = self.model.jnt_range[joint_id]
                    self.data.qpos[qpos_id] = float(np.clip(value, low, high))
        elif self.flexion_gain != 1.0:
            for finger_data in self.fingers.values():
                for slot in finger_data["flexion_slots"]:
                    joint_id = finger_data["joint_ids"][slot]
                    qpos_id = int(self.model.jnt_qposadr[joint_id])
                    value = self.neutral_qpos[qpos_id] + self.flexion_gain * (
                        fitted_qpos[qpos_id] - self.neutral_qpos[qpos_id]
                    )
                    if self.model.jnt_limited[joint_id]:
                        low, high = self.model.jnt_range[joint_id]
                        value = float(np.clip(value, low, high))
                    self.data.qpos[qpos_id] = value

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        return self.data.qpos.copy()
