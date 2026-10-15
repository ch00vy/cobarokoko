import importlib
import socket
import time
from collections import deque

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from control.smartglove_muscle_control import SmartGloveMuscleController
from smartglove.config import CALIBRATION_PACKETS, ENV_NAME, FINGER_JOINTS
from smartglove.mapping import HandQposMapper
from smartglove.quaternion import (
    QuaternionLowPassFilter,
    average_quaternions,
    sensor_quaternion_to_wxyz,
)
from smartglove.wrist_mapping import WristQposMapper
from smartglove_realtime import (
    HOST,
    PORT,
    RATE_WINDOW_INTERVALS,
    calculate_packet_rate,
    parse_packet_with_error,
)

importlib.import_module("myosuite")


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
        wrist_mapper = WristQposMapper(model, data, quaternion_order)
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
        wrist_calibration_samples = {
            sensor: [] for sensor in ("FOREARM", "HAND")
        }
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
                        if finger in calibration_samples:
                            calibration_samples[finger].append(quaternion)
                        elif finger in wrist_calibration_samples:
                            wrist_calibration_samples[finger].append(quaternion)

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
                    sensor_wrist_neutral = {
                        sensor: average_quaternions(samples)
                        for sensor, samples in wrist_calibration_samples.items()
                    }
                    wrist_mapper.calibrate(
                        sensor_wrist_neutral["FOREARM"],
                        sensor_wrist_neutral["HAND"],
                    )
                    quaternion_filter.reset(
                        {**sensor_neutral, **sensor_wrist_neutral},
                        now,
                    )
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
                        if finger in grip_calibration_samples:
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
                    quaternion_filter.reset(sensor_quaternions, now)
                    grip_calibrated = True
                    print("\n[OK] Kalibrasi terbuka dan genggam selesai.")

                sensor_quaternions = quaternion_filter.update(
                    sensor_quaternions,
                    now,
                )
                finger_quaternions = {
                    finger: sensor_quaternions[finger]
                    for finger in FINGER_JOINTS
                }
                # Pergelangan dipetakan lebih dulu agar IK jari (abduksi) melihat
                # pose wrist pada frame yang sama, bukan frame sebelumnya.
                wrist_mapper.map_frame(sensor_quaternions)
                qpos = mapper.map_frame(finger_quaternions)
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
                    wrist_text = " ".join(
                        f"{name}:{angle:+.0f}{'*' if saturated else ''}"
                        for name, angle, saturated in zip(
                            ("ps", "dev", "flex"),
                            wrist_mapper.last_angles_deg,
                            wrist_mapper.saturated,
                        )
                    )
                    print(
                        f"Frame {frame_number:>7} | "
                        f"Rate {rate:5.1f} Hz | "
                        f"fleksi MCP/PIP/DIP (thumb CMC/MP/IP): {finger_flexion} | "
                        f"wrist deg (* = di batas ROM) {wrist_text}",
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
