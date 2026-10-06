import argparse
import importlib
import math
import socket
import time
from collections import deque
from typing import Sequence

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from control.smartglove_muscle_control import SmartGloveMuscleController
from smartglove_realtime import (
    HOST,
    PORT,
    RATE_WINDOW_INTERVALS,
    calculate_packet_rate,
    parse_packet_with_error,
)

importlib.import_module("myosuite")


ENV_NAME = "myoHandPoseFixed-v0"
CALIBRATION_PACKETS = 30
IK_ITERATIONS = 20
FILTER_CUTOFF_HZ = 12.0
# Couple the two flexion joints because one distal quaternion cannot resolve both.
DISTAL_COUPLING = 2.0 / 3.0

FINGER_JOINTS = {
    "THUMB": {
        "joints": (
            "cmc_abduction",
            "cmc_flexion",
            "mp_flexion",
            "ip_flexion",
        ),
        "flexion_slots": (1, 2, 3),
        "body": "distal_thumb",
        "flexion_closed_limit": "low",
        "invert_abduction": True,
    },
    "INDEX": {
        "joints": (
            "mcp2_flexion",
            "mcp2_abduction",
            "pm2_flexion",
            "md2_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph2",
    },
    "MIDDLE": {
        "joints": (
            "mcp3_flexion",
            "mcp3_abduction",
            "pm3_flexion",
            "md3_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph3",
    },
    "RING": {
        "joints": (
            "mcp4_flexion",
            "mcp4_abduction",
            "pm4_flexion",
            "md4_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph4",
    },
    "PINKY": {
        "joints": (
            "mcp5_flexion",
            "mcp5_abduction",
            "pm5_flexion",
            "md5_flexion",
        ),
        "flexion_slots": (0, 2, 3),
        "body": "distph5",
    },
}


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


def drain_udp_queue(sock: socket.socket) -> bytes | None:
    latest_packet = None
    sock.setblocking(False)
    try:
        while True:
            try:
                latest_packet, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
    finally:
        sock.settimeout(0.01)
    return latest_packet


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


def capture_sensor_quaternions(parsed: dict, order: str) -> dict[str, np.ndarray]:
    return {
        finger: sensor_quaternion_to_wxyz(values["quaternion"], order)
        for finger, values in parsed.items()
    }


def run(
    quaternion_order: str,
    flexion_gain: float,
    filter_cutoff_hz: float,
) -> None:
    env = gym.make(ENV_NAME)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        unwrapped = env.unwrapped
        model = getattr(unwrapped, "mj_model")
        data = getattr(unwrapped, "mj_data")
        data.qpos[:] = getattr(unwrapped, "init_qpos")
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
        mapper = HandQposMapper(model, data, flexion_gain)
        muscle_controller = SmartGloveMuscleController(
            model,
            data,
            FINGER_JOINTS,
        )
        quaternion_filter = QuaternionLowPassFilter(filter_cutoff_hz)

        print(f"[OK] MyoSuite model : {ENV_NAME} (nq={model.nq})")
        print(f"[OK] Muscle control: {model.nu} actuator aktif di viewer")
        print(f"[OK] Flexion gain  : {flexion_gain:.2f}x")
        print(f"[OK] Filter cutoff : {filter_cutoff_hz:.1f} Hz")
        print(
            "Atur glove ke pose netral yang ingin dianggap sebagai tangan terbuka: "
            "jari lurus, ibu jari terbuka, pergelangan rileks."
        )
        input("Tahan pose tersebut, lalu tekan Enter untuk mulai kalibrasi...")

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        sock.settimeout(0.01)
        sock.bind((HOST, PORT))

        print(f"[OK] UDP listener  : {HOST}:{PORT}")
        print(
            f"[WAIT] Tahan tangan di pose netral. "
            f"Kalibrasi {CALIBRATION_PACKETS} paket..."
        )

        calibration_samples = {finger: [] for finger in FINGER_JOINTS}
        grip_calibration_samples = {finger: [] for finger in FINGER_JOINTS}
        packet_intervals = deque(maxlen=RATE_WINDOW_INTERVALS)
        last_packet_time = None
        frame_number = 0
        last_status_time = 0.0
        calibrated = False
        grip_calibrated = False

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                try:
                    packet, _ = sock.recvfrom(65535)
                except socket.timeout:
                    viewer.sync()
                    continue

                latest_packet = drain_udp_queue(sock)
                if latest_packet is not None:
                    packet = latest_packet

                now = time.perf_counter()
                parsed, parse_error = parse_packet_with_error(packet)
                if parsed is None:
                    print(
                        f"[WARNING] Paket diabaikan: {parse_error}"
                    )
                    continue

                sensor_quaternions = capture_sensor_quaternions(
                    parsed,
                    quaternion_order,
                )
                frame_number += 1

                if not calibrated:
                    for finger, quaternion in sensor_quaternions.items():
                        calibration_samples[finger].append(quaternion)

                    sample_count = len(calibration_samples["THUMB"])
                    if sample_count < CALIBRATION_PACKETS:
                        if now - last_status_time >= 0.5:
                            print(
                                f"[CALIBRATION] {sample_count}/"
                                f"{CALIBRATION_PACKETS} paket netral",
                                end="\r",
                            )
                            last_status_time = now
                        viewer.sync()
                        continue

                    sensor_neutral = {
                        finger: average_quaternions(samples)
                        for finger, samples in calibration_samples.items()
                    }
                    mapper.calibrate(sensor_neutral)
                    quaternion_filter.reset(sensor_neutral, now)
                    calibrated = True
                    print("\n[OK] Pose terbuka tersimpan.")
                    input(
                        "Sekarang genggam tangan dengan nyaman, tahan posisi, "
                        "lalu tekan Enter untuk kalibrasi genggaman..."
                    )
                    drain_udp_queue(sock)
                    last_status_time = 0.0
                    continue

                if not grip_calibrated:
                    for finger, quaternion in sensor_quaternions.items():
                        grip_calibration_samples[finger].append(quaternion)

                    sample_count = len(grip_calibration_samples["THUMB"])
                    if sample_count < CALIBRATION_PACKETS:
                        if now - last_status_time >= 0.5:
                            print(
                                f"[CALIBRATION GRIP] {sample_count}/"
                                f"{CALIBRATION_PACKETS} paket genggam",
                                end="\r",
                            )
                            last_status_time = now
                        viewer.sync()
                        continue

                    sensor_grip = {
                        finger: average_quaternions(samples)
                        for finger, samples in grip_calibration_samples.items()
                    }
                    try:
                        skipped_fingers = mapper.calibrate_grip(sensor_grip)
                    except ValueError as error:
                        print(f"\n[WARNING] Kalibrasi genggam gagal: {error}")
                        input(
                            "Atur ulang genggaman, tahan, lalu tekan Enter "
                            "untuk mencoba lagi..."
                        )
                        drain_udp_queue(sock)
                        grip_calibration_samples = {
                            finger: [] for finger in FINGER_JOINTS
                        }
                        last_status_time = 0.0
                        continue

                    for finger in skipped_fingers:
                        print(
                            f"[WARNING] Kalibrasi genggam {finger} tidak menangkap "
                            "perubahan; jari ini tetap menggunakan IK."
                        )
                    quaternion_filter.reset(sensor_grip, now)
                    grip_calibrated = True
                    print("\n[OK] Kalibrasi terbuka dan genggam selesai.")

                sensor_quaternions = quaternion_filter.update(
                    sensor_quaternions,
                    now,
                )
                qpos = mapper.map_frame(sensor_quaternions)
                muscle_controller.update(qpos)

                if last_packet_time is not None:
                    interval = now - last_packet_time
                    if interval > 0:
                        packet_intervals.append(interval)
                last_packet_time = now

                if now - last_status_time >= 0.5:
                    rate = calculate_packet_rate(packet_intervals)
                    finger_flexion = " | ".join(
                        f"{finger}:"
                        f"{np.round(np.degrees([qpos[mapper.fingers[finger]['qpos'][slot]] for slot in mapper.fingers[finger]['flexion_slots']])).astype(int).tolist()}"
                        for finger in FINGER_JOINTS
                    )
                    print(
                        f"Frame {frame_number:>7} | "
                        f"Rate {rate:5.1f} Hz | "
                        f"fleksi MCP/PIP/DIP (thumb CMC/MP/IP): {finger_flexion}",
                        end="\r",
                    )
                    print()
                    last_status_time = now

                viewer.sync()
    except KeyboardInterrupt:
        print("\n[STOP] Smartglove MuJoCo dihentikan.")
    finally:
        sock.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Rokoko Smartglove UDP quaternions to MyoSuite hand qpos."
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Urutan komponen quaternion pada paket Rokoko (default: xyzw).",
    )
    parser.add_argument(
        "--flexion-gain",
        type=float,
        default=1.0,
        help="Pengali fleksi jari (default: 1.0).",
    )
    parser.add_argument(
        "--filter-cutoff-hz",
        type=float,
        default=FILTER_CUTOFF_HZ,
        help="Cutoff filter quaternion dalam Hz; lebih rendah lebih halus (default: 12).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.flexion_gain) or args.flexion_gain <= 0.0:
        parser.error("--flexion-gain harus finite dan lebih besar dari nol.")
    if not math.isfinite(args.filter_cutoff_hz) or args.filter_cutoff_hz <= 0.0:
        parser.error("--filter-cutoff-hz harus finite dan lebih besar dari nol.")
    run(args.quaternion_order, args.flexion_gain, args.filter_cutoff_hz)


if __name__ == "__main__":
    main()
