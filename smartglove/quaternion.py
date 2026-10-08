import math
from typing import Sequence

import numpy as np

from smartglove.config import FILTER_CUTOFF_HZ


def normalize_quaternion(
    quaternion: np.ndarray | tuple[float, float, float, float],
) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Quaternion harus berisi empat nilai finite dan bukan nol.")
    return result / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.array((1.0, -1.0, -1.0, -1.0))


def quaternion_relative(start: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = normalize_quaternion(
        quaternion_multiply(quaternion_conjugate(start), current)
    )
    return -relative if relative[0] < 0.0 else relative


def quaternion_slerp(
    start: np.ndarray,
    target: np.ndarray,
    amount: float,
) -> np.ndarray:
    start = normalize_quaternion(start)
    target = normalize_quaternion(target)
    dot = float(np.dot(start, target))

    if dot < 0.0:
        target = -target
        dot = -dot

    amount = float(np.clip(amount, 0.0, 1.0))
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(start + amount * (target - start))

    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    start_weight = math.sin((1.0 - amount) * angle) / sin_angle
    target_weight = math.sin(amount * angle) / sin_angle
    return normalize_quaternion(start_weight * start + target_weight * target)


class QuaternionLowPassFilter:
    def __init__(self, cutoff_hz: float = FILTER_CUTOFF_HZ) -> None:
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError("filter_cutoff_hz harus finite dan lebih besar dari nol.")
        self.cutoff_hz = cutoff_hz
        self.filtered: dict[str, np.ndarray] = {}
        self.last_time: float | None = None

    def reset(
        self,
        orientations: dict[str, np.ndarray],
        timestamp: float,
    ) -> None:
        self.filtered = {
            finger: normalize_quaternion(orientation)
            for finger, orientation in orientations.items()
        }
        self.last_time = timestamp

    def update(
        self,
        orientations: dict[str, np.ndarray],
        timestamp: float,
    ) -> dict[str, np.ndarray]:
        if self.last_time is None:
            self.reset(orientations, timestamp)
            return {finger: value.copy() for finger, value in self.filtered.items()}

        elapsed = max(0.0, timestamp - self.last_time)
        amount = -math.expm1(-2.0 * math.pi * self.cutoff_hz * elapsed)
        for finger, orientation in orientations.items():
            normalized = normalize_quaternion(orientation)
            previous = self.filtered.get(finger)
            self.filtered[finger] = (
                normalized
                if previous is None
                else quaternion_slerp(previous, normalized, amount)
            )

        self.last_time = timestamp
        return {finger: value.copy() for finger, value in self.filtered.items()}


def quaternion_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    error = normalize_quaternion(
        quaternion_multiply(target, quaternion_conjugate(current))
    )
    if error[0] < 0.0:
        error = -error

    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1e-10:
        return 2.0 * error[1:]

    angle = 2.0 * math.atan2(vector_norm, float(error[0]))
    return error[1:] * (angle / vector_norm)


def average_quaternions(quaternions: Sequence[np.ndarray]) -> np.ndarray:
    reference = quaternions[0]
    aligned = [
        quaternion if np.dot(reference, quaternion) >= 0.0 else -quaternion
        for quaternion in quaternions
    ]
    return normalize_quaternion(np.mean(aligned, axis=0))


def sensor_quaternion_to_wxyz(
    quaternion: np.ndarray | tuple[float, float, float, float],
    order: str,
) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,):
        raise ValueError("Quaternion harus berisi tepat empat nilai.")
    if order == "xyzw":
        return normalize_quaternion((values[3], values[0], values[1], values[2]))
    if order == "wxyz":
        return normalize_quaternion(values)
    raise ValueError(f"Urutan quaternion tidak didukung: {order}")
