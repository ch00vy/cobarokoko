import mujoco
import numpy as np

from smartglove.config import IK_ITERATIONS
from smartglove.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_error,
    quaternion_multiply,
)


class WristQposMapper:
    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data
        self.sensor_to_model: dict[str, np.ndarray] | None = None
        self.bodies = {
            "FOREARM": self._name_id(
                mujoco.mjtObj.mjOBJ_BODY, "radius"  # type: ignore[attr-defined]
            ),
            "HAND": self._name_id(
                mujoco.mjtObj.mjOBJ_BODY, "lunate"  # type: ignore[attr-defined]
            ),
        }
        self.joints = [
            self._name_id(
                mujoco.mjtObj.mjOBJ_JOINT, name  # type: ignore[attr-defined]
            )
            for name in ("pro_sup", "deviation", "flexion")
        ]
        self.qpos_ids = [int(model.jnt_qposadr[joint]) for joint in self.joints]
        self.dof_ids = [int(model.jnt_dofadr[joint]) for joint in self.joints]
        self.lower = np.array(
            [
                model.jnt_range[joint][0]
                if model.jnt_limited[joint]
                else -np.inf
                for joint in self.joints
            ],
            dtype=float,
        )
        self.upper = np.array(
            [
                model.jnt_range[joint][1]
                if model.jnt_limited[joint]
                else np.inf
                for joint in self.joints
            ],
            dtype=float,
        )

    def _name_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(  # type: ignore[attr-defined]
            self.model,
            object_type,
            name,
        )
        if object_id < 0:
            raise ValueError(f"Objek MuJoCo tidak ditemukan: {name}")
        return object_id

    def calibrate(
        self,
        forearm_quaternion: np.ndarray,
        hand_quaternion: np.ndarray,
    ) -> None:
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        self.sensor_to_model = {
            sensor: normalize_quaternion(
                quaternion_multiply(
                    self.data.xquat[body],
                    quaternion_conjugate(normalize_quaternion(sensor_quaternion)),
                )
            )
            for sensor, body, sensor_quaternion in (
                ("FOREARM", self.bodies["FOREARM"], forearm_quaternion),
                ("HAND", self.bodies["HAND"], hand_quaternion),
            )
        }

    def _fit_orientations(
        self,
        targets: dict[str, np.ndarray],
    ) -> None:
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        variables = np.clip(self.data.qpos[self.qpos_ids], self.lower, self.upper)

        for _ in range(IK_ITERATIONS):
            self.data.qpos[self.qpos_ids] = variables
            mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
            errors = []
            jacobians = []

            for sensor, body in self.bodies.items():
                current = normalize_quaternion(self.data.xquat[body])
                errors.append(quaternion_error(targets[sensor], current))
                mujoco.mj_jacBody(  # type: ignore[attr-defined]
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    body,
                )
                jacobians.append(jacobian_rotation[:, self.dof_ids].copy())

            error = np.concatenate(errors)
            if np.linalg.norm(error) < 1e-5:
                break

            jacobian = np.vstack(jacobians)
            damped_system = jacobian @ jacobian.T + 1e-5 * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(damped_system, error)
            variables = np.clip(
                variables + np.clip(delta, -0.15, 0.15),
                self.lower,
                self.upper,
            )

        self.data.qpos[self.qpos_ids] = variables
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]

    def map_frame(
        self,
        sensor_quaternions: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.sensor_to_model is None:
            raise RuntimeError("Mapper pergelangan harus dikalibrasi lebih dulu.")
        missing = set(self.bodies) - sensor_quaternions.keys()
        if missing:
            raise ValueError(
                f"Quaternion sensor pergelangan tidak tersedia: "
                f"{', '.join(sorted(missing))}."
            )

        targets = {
            sensor: normalize_quaternion(
                quaternion_multiply(
                    self.sensor_to_model[sensor],
                    sensor_quaternions[sensor],
                )
            )
            for sensor in self.bodies
        }
        self._fit_orientations(targets)
        return self.data.qpos.copy()
