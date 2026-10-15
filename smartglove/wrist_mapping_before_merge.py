import math
from dataclasses import dataclass

import mujoco
import numpy as np

from smartglove.quaternion import normalize_quaternion, quaternion_relative

WRIST_RANGE_FRACTION = 0.6


@dataclass(frozen=True)
class WristOutputCalibration:
    sensor_weights: tuple[float, float, float, float, float, float]
    negative_range: float
    positive_range: float
    deadband: float


# Ridge-regression weights use [FOREARM xyz, HAND xyz] capture features.
# Other wrist gestures and jari_only_baru were zero-output samples; deadbands
# suppress their residual output.
WRIST_OUTPUT_CALIBRATIONS: dict[str, WristOutputCalibration] = {
    "pro_sup": WristOutputCalibration(
        (0.42720504, -0.03778274, -0.06045433, -0.01117321, -0.03975597, 0.06283068),
        negative_range=0.0546,
        positive_range=0.1233,
        deadband=0.04,
    ),
    "deviation": WristOutputCalibration(
        (0.36327936, 0.09689331, -0.61295967, 0.54404526, 0.03535099, 0.00849656),
        negative_range=0.4979,
        positive_range=0.5537,
        deadband=0.0410,
    ),
    "flexion": WristOutputCalibration(
        (-0.10659294, 0.71833708, -0.34625890, 0.16114529, 0.05399674, -0.16053558),
        negative_range=0.2807,
        positive_range=0.4371,
        deadband=0.0520,
    ),
}


class WristQposMapper:
    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data
        self.sensors = ("FOREARM", "HAND")
        self.sensor_neutral: dict[str, np.ndarray] | None = None
        self.neutral_qpos: np.ndarray | None = None
        self.joints = [
            self._name_id(
                mujoco.mjtObj.mjOBJ_JOINT, name  # type: ignore[attr-defined]
            )
            for name in WRIST_OUTPUT_CALIBRATIONS
        ]
        self.qpos_ids = [int(model.jnt_qposadr[joint]) for joint in self.joints]
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
        self.sensor_weights = {
            name: np.asarray(calibration.sensor_weights, dtype=float)
            for name, calibration in WRIST_OUTPUT_CALIBRATIONS.items()
        }

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
        self.sensor_neutral = {
            sensor: normalize_quaternion(quaternion)
            for sensor, quaternion in (
                ("FOREARM", forearm_quaternion),
                ("HAND", hand_quaternion),
            )
        }
        self.neutral_qpos = self.data.qpos[self.qpos_ids].copy()

    @staticmethod
    def _rotation_vector(quaternion: np.ndarray) -> np.ndarray:
        vector_norm = float(np.linalg.norm(quaternion[1:]))
        if vector_norm < 1e-10:
            return np.zeros(3)
        angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
        return quaternion[1:] * (angle / vector_norm)

    def map_frame(
        self,
        sensor_quaternions: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.sensor_neutral is None or self.neutral_qpos is None:
            raise RuntimeError("Mapper pergelangan harus dikalibrasi lebih dulu.")
        missing = set(self.sensors) - sensor_quaternions.keys()
        if missing:
            raise ValueError(
                f"Quaternion sensor pergelangan tidak tersedia: "
                f"{', '.join(sorted(missing))}."
            )

        features = np.concatenate(
            [
                self._rotation_vector(
                    quaternion_relative(
                        self.sensor_neutral[sensor],
                        normalize_quaternion(sensor_quaternions[sensor]),
                    )
                )
                for sensor in self.sensors
            ]
        )
        for index, (joint_name, calibration) in enumerate(
            WRIST_OUTPUT_CALIBRATIONS.items()
        ):
            signal = float(np.dot(self.sensor_weights[joint_name], features))
            if signal > calibration.deadband:
                fraction = np.clip(
                    (signal - calibration.deadband)
                    / (calibration.positive_range - calibration.deadband),
                    0.0,
                    1.0,
                )
                output_range = (
                    self.upper[index] - self.neutral_qpos[index]
                ) * WRIST_RANGE_FRACTION
                delta = output_range * fraction
            elif signal < -calibration.deadband:
                fraction = np.clip(
                    (-signal - calibration.deadband)
                    / (calibration.negative_range - calibration.deadband),
                    0.0,
                    1.0,
                )
                output_range = (
                    self.neutral_qpos[index] - self.lower[index]
                ) * WRIST_RANGE_FRACTION
                delta = -output_range * fraction
            else:
                delta = 0.0
            self.data.qpos[self.qpos_ids[index]] = np.clip(
                self.neutral_qpos[index] + delta,
                self.lower[index],
                self.upper[index],
            )

        mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        return self.data.qpos.copy()
