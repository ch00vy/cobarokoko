import contextlib
import importlib
import socket
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from control.smartglove_muscle_control import SmartGloveMuscleController
from smartglove.calibration import CalibrationSession, frame_quaternions
from smartglove.config import (
    CALIBRATION_PACKETS,
    DISTAL_COUPLING,
    ENV_NAME,
    FILTER_CUTOFF_HZ,
    FINGER_JOINTS,
    IK_ITERATIONS,
)
from smartglove.frame import (
    Event,
    Frame,
    RecordingError,
    frame_from_parsed,
    glove_time_seconds,
)
from smartglove.mapping import HandQposMapper
from smartglove.quaternion import (
    QuaternionLowPassFilter,
    sensor_quaternion_to_wxyz,
)
from smartglove.recorder import (
    JsonlRecorder,
    build_meta,
    describe_axes_file,
    read_recording,
)
from smartglove.wrist_alignment import DEFAULT_AXES_FILE
from smartglove.wrist_mapping import WristQposMapper
from smartglove_realtime import (
    HOST,
    PORT,
    RATE_WINDOW_INTERVALS,
    calculate_packet_rate,
    parse_packet_header,
    parse_packet_with_error,
)

importlib.import_module("myosuite")

# Selisih maksimum (per komponen quaternion) yang masih dianggap sama saat
# membandingkan kalibrasi hasil hitung ulang dengan yang tersimpan di rekaman.
CALIBRATION_MATCH_TOLERANCE = 1e-9


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


def drain_udp_packets(sock: socket.socket) -> list[bytes]:
    """Ambil SEMUA paket yang sedang mengantre, dari yang tertua ke terbaru."""
    packets: list[bytes] = []
    sock.setblocking(False)
    try:
        while True:
            try:
                packet, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            packets.append(packet)
    finally:
        sock.settimeout(0.01)
    return packets


class LiveCapture:
    """Penomoran frame live dan penulisan rekaman.

    mode "processed": hanya paket yang diproses mapper yang direkam.
    mode "all": setiap paket valid yang dibaca direkam, termasuk paket yang
    dibuang drain (ditandai ``proc: false``). Pemrosesan tidak berubah: tetap
    paket terbaru dari tiap gelombang baca.
    """

    def __init__(self, recorder: JsonlRecorder | None, record_all: bool) -> None:
        self.recorder = recorder
        self.record_all = bool(record_all and recorder is not None)
        self.frame_number = 0
        self.processed = 0
        self.invalid_skipped = 0

    def record_unprocessed(self, packets: list[bytes], t: float) -> None:
        if not self.record_all:
            return
        for packet in packets:
            parsed, _ = parse_packet_with_error(packet)
            if parsed is None:
                self.invalid_skipped += 1
                continue
            self.frame_number += 1
            glove_tick, glove_seq = parse_packet_header(packet)
            self.recorder.write_frame(  # type: ignore[union-attr]
                frame_from_parsed(
                    parsed, self.frame_number, t,
                    glove_seq=glove_seq, glove_tick=glove_tick, processed=False,
                )
            )

    def new_frame(self, parsed: dict, packet: bytes, t: float) -> Frame:
        self.frame_number += 1
        self.processed += 1
        glove_tick, glove_seq = parse_packet_header(packet)
        frame = frame_from_parsed(
            parsed, self.frame_number, t, glove_seq=glove_seq, glove_tick=glove_tick
        )
        if self.recorder is not None:
            self.recorder.write_frame(frame)
        return frame


def capture_sensor_quaternions(parsed: dict, order: str) -> dict[str, np.ndarray]:
    return {
        finger: sensor_quaternion_to_wxyz(values["quaternion"], order)
        for finger, values in parsed.items()
    }


# ---------------------------------------------------------------- pipeline


@dataclass
class Pipeline:
    model: object
    data: object
    mapper: HandQposMapper
    wrist_mapper: WristQposMapper
    muscle_controller: SmartGloveMuscleController
    quaternion_filter: QuaternionLowPassFilter
    calibration: CalibrationSession


def build_pipeline(
    env,
    quaternion_order: str,
    flexion_gain: float,
    filter_cutoff_hz: float,
    emit=None,
) -> Pipeline:
    unwrapped = env.unwrapped
    model = getattr(unwrapped, "mj_model")
    data = getattr(unwrapped, "mj_data")
    data.qpos[:] = getattr(unwrapped, "init_qpos")
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
    mapper = HandQposMapper(model, data, flexion_gain)
    wrist_mapper = WristQposMapper(model, data, quaternion_order)
    muscle_controller = SmartGloveMuscleController(model, data, FINGER_JOINTS)
    quaternion_filter = QuaternionLowPassFilter(filter_cutoff_hz)
    calibration = CalibrationSession(
        mapper, wrist_mapper, quaternion_filter, quaternion_order, emit=emit
    )
    return Pipeline(
        model, data, mapper, wrist_mapper, muscle_controller,
        quaternion_filter, calibration,
    )


def map_frame(pipeline: Pipeline, quats: dict[str, np.ndarray], t: float) -> np.ndarray:
    """Langkah per-frame yang sama untuk live dan replay: filter -> wrist -> jari -> otot."""
    filtered = pipeline.quaternion_filter.update(quats, t)
    finger_quaternions = {finger: filtered[finger] for finger in FINGER_JOINTS}
    # Pergelangan dipetakan lebih dulu agar IK jari (abduksi) melihat
    # pose wrist pada frame yang sama, bukan frame sebelumnya.
    pipeline.wrist_mapper.map_frame(filtered)
    qpos = pipeline.mapper.map_frame(finger_quaternions)
    pipeline.muscle_controller.update(qpos)
    return qpos


def format_status(
    frame_number: int, rate: float, qpos: np.ndarray, pipeline: Pipeline
) -> str:
    mapper, wrist_mapper = pipeline.mapper, pipeline.wrist_mapper
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
    return (
        f"Frame {frame_number:>7} | "
        f"Rate {rate:5.1f} Hz | "
        f"fleksi MCP/PIP/DIP (thumb CMC/MP/IP): {finger_flexion} | "
        f"wrist deg (* = di batas ROM) {wrist_text}"
    )


def session_params(flexion_gain: float, filter_cutoff_hz: float) -> dict:
    return {
        "env": ENV_NAME,
        "flexion_gain": flexion_gain,
        "filter_cutoff_hz": filter_cutoff_hz,
        "calibration_packets": CALIBRATION_PACKETS,
        "ik_iterations": IK_ITERATIONS,
        "distal_coupling": DISTAL_COUPLING,
    }


# -------------------------------------------------------------------- live


def run(
    quaternion_order: str,
    flexion_gain: float,
    filter_cutoff_hz: float,
    record_path: str | Path | None = None,
    record_all: bool = False,
) -> None:
    if record_all and record_path is None:
        raise ValueError("record_all membutuhkan record_path.")
    env = gym.make(ENV_NAME)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recorder: JsonlRecorder | None = None
    capture: LiveCapture | None = None
    session_start: float | None = None
    try:
        if record_path is not None:
            recorder = JsonlRecorder(
                record_path,
                build_meta(
                    quat_order=quaternion_order,
                    source=f"{HOST}:{PORT}",
                    params=session_params(flexion_gain, filter_cutoff_hz),
                    axes_file=DEFAULT_AXES_FILE,
                    capture="all" if record_all else "processed",
                ),
            )
        capture = LiveCapture(recorder, record_all)
        pipeline = build_pipeline(
            env,
            quaternion_order,
            flexion_gain,
            filter_cutoff_hz,
            emit=recorder.write_event if recorder is not None else None,
        )
        model, data = pipeline.model, pipeline.data
        calibration = pipeline.calibration

        print(f"[OK] MyoSuite model : {ENV_NAME} (nq={model.nq})")  # type: ignore[attr-defined]
        print(f"[OK] Muscle control: {model.nu} actuator aktif di viewer")  # type: ignore[attr-defined]
        print(f"[OK] Flexion gain  : {flexion_gain:.2f}x")
        print(f"[OK] Filter cutoff : {filter_cutoff_hz:.1f} Hz")
        if recorder is not None:
            mode_text = "semua paket" if record_all else "frame yang diproses"
            print(f"[OK] Rekaman JSONL : {recorder.path} ({mode_text})")
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

        packet_intervals = deque(maxlen=RATE_WINDOW_INTERVALS)
        last_packet_time = None
        last_status_time = 0.0
        session_start = time.perf_counter()
        calibration.start_open(0, 0.0)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                try:
                    packet, _ = sock.recvfrom(65535)
                except socket.timeout:
                    viewer.sync()
                    continue

                queued = drain_udp_packets(sock)
                earlier = [packet, *queued[:-1]] if queued else []
                if queued:
                    packet = queued[-1]

                now = time.perf_counter()
                capture.record_unprocessed(earlier, now - session_start)
                parsed, parse_error = parse_packet_with_error(packet)
                if parsed is None:
                    capture.invalid_skipped += 1
                    print(
                        f"[WARNING] Paket diabaikan: {parse_error}"
                    )
                    continue

                frame = capture.new_frame(parsed, packet, now - session_start)
                sensor_quaternions = frame_quaternions(frame, quaternion_order)

                if calibration.phase == "open":
                    result = calibration.feed(frame, sensor_quaternions)
                    if result == "collecting_open":
                        if now - last_status_time >= 0.5:
                            print(
                                f"[CALIBRATION] {calibration.sample_count}/"
                                f"{CALIBRATION_PACKETS} paket netral",
                                end="\r",
                            )
                            last_status_time = now
                        viewer.sync()
                        continue

                    print("\n[OK] Pose terbuka tersimpan.")
                    input(
                        "Sekarang genggam tangan dengan nyaman, tahan posisi, "
                        "lalu tekan Enter untuk kalibrasi genggaman..."
                    )
                    capture.record_unprocessed(drain_udp_packets(sock), frame.t)
                    last_status_time = 0.0
                    calibration.start_grip(capture.frame_number, frame.t)
                    continue

                if calibration.phase == "grip":
                    result = calibration.feed(frame, sensor_quaternions)
                    if result == "collecting_grip":
                        if now - last_status_time >= 0.5:
                            print(
                                f"[CALIBRATION GRIP] {calibration.sample_count}/"
                                f"{CALIBRATION_PACKETS} paket genggam",
                                end="\r",
                            )
                            last_status_time = now
                        viewer.sync()
                        continue

                    if result == "grip_failed":
                        print(f"\n[WARNING] Kalibrasi genggam gagal: {calibration.last_error}")
                        input(
                            "Atur ulang genggaman, tahan, lalu tekan Enter "
                            "untuk mencoba lagi..."
                        )
                        capture.record_unprocessed(drain_udp_packets(sock), frame.t)
                        last_status_time = 0.0
                        calibration.start_grip(capture.frame_number, frame.t)
                        continue

                    for finger in calibration.skipped_fingers:
                        print(
                            f"[WARNING] Kalibrasi genggam {finger} tidak menangkap "
                            "perubahan; jari ini tetap menggunakan IK."
                        )
                    print("\n[OK] Kalibrasi terbuka dan genggam selesai.")

                qpos = map_frame(pipeline, sensor_quaternions, frame.t)

                if last_packet_time is not None:
                    interval = now - last_packet_time
                    if interval > 0:
                        packet_intervals.append(interval)
                last_packet_time = now

                if now - last_status_time >= 0.5:
                    rate = calculate_packet_rate(packet_intervals)
                    print(format_status(capture.frame_number, rate, qpos, pipeline), end="\r")
                    print()
                    last_status_time = now

                viewer.sync()
    except KeyboardInterrupt:
        print("\n[STOP] Smartglove MuJoCo dihentikan.")
    finally:
        if recorder is not None:
            try:
                if capture is not None and session_start is not None:
                    recorder.write_event(
                        "session_end",
                        capture.frame_number,
                        time.perf_counter() - session_start,
                        capture="all" if record_all else "processed",
                        frames=capture.frame_number,
                        processed=capture.processed,
                        invalid_skipped=capture.invalid_skipped,
                    )
            finally:
                recorder.close()
        sock.close()
        env.close()


# ------------------------------------------------------------------ replay


@dataclass
class ReplayResult:
    qpos: np.ndarray
    warnings: list[str] = field(default_factory=list)
    frames_total: int = 0
    frames_mapped: int = 0
    meta: dict = field(default_factory=dict)


def extract_stored_calibration(recording) -> tuple[dict, dict, int]:
    """Payload kalibrasi buka dan genggam terakhir yang berhasil, plus nomor
    frame penutup kalibrasi genggam (frame itu ikut dipetakan)."""
    open_payload = grip_payload = None
    resume_n = None
    for event in recording.events:
        if event.name == "calib_open_done":
            open_payload = event.payload
        elif event.name == "calib_grip_done" and event.payload.get("ok"):
            grip_payload = event.payload
            resume_n = event.n
    if open_payload is None or grip_payload is None or resume_n is None:
        raise RecordingError(
            "rekaman tidak memuat event kalibrasi buka dan genggam yang berhasil"
        )
    return open_payload, grip_payload, resume_n


def _compare_calibration(session: CalibrationSession, event: Event) -> str | None:
    from smartglove.calibration import payload_max_difference

    if event.name == "calib_open_done":
        computed, key = session.last_open_payload, "neutral"
    else:
        computed, key = session.last_grip_payload, "grip"
    if computed is None:
        return (
            f"{event.name} (frame {event.n}): frame kalibrasi tidak lengkap, "
            "tidak bisa dibandingkan dengan nilai tersimpan"
        )
    if event.name == "calib_grip_done" and bool(computed.get("ok")) != bool(
        event.payload.get("ok")
    ):
        return f"{event.name} (frame {event.n}): hasil berhasil/gagal berbeda dari rekaman"
    difference = payload_max_difference(computed, event.payload, key)
    if difference > CALIBRATION_MATCH_TOLERANCE:
        return (
            f"{event.name} (frame {event.n}): hasil hitung ulang berbeda dari rekaman "
            f"(selisih maks {difference:.3g}); kode kalibrasi atau mapping berubah"
        )
    return None


def replay(
    path: str | Path,
    *,
    quaternion_order: str | None = None,
    flexion_gain: float | None = None,
    filter_cutoff_hz: float | None = None,
    mode: str = "recompute",
    calib_from: str | Path | None = None,
    show_viewer: bool = True,
    realtime: bool = True,
    frames: str = "processed",
) -> ReplayResult:
    """Putar ulang rekaman JSONL lewat pipeline yang sama dengan sesi live.

    mode="recompute": kalibrasi dihitung ulang dari frame kalibrasi di rekaman
    (hasilnya dibandingkan dengan nilai tersimpan; selisih jadi peringatan).
    mode="stored": memakai rata-rata quaternion yang tersimpan di event.
    ``calib_from`` meminjam kalibrasi dari rekaman lain (otomatis mode stored).

    frames="processed": hanya frame yang diproses saat live, berbasis waktu
    penerima ``t``; identik dengan sesi live.
    frames="all": setiap frame (termasuk yang dibuang saat live) dipetakan
    dengan waktu jam glove (``glove_tick``). Kalibrasi tetap memakai hanya
    frame yang diproses, sama seperti live, dan duplikat ``glove_seq``
    berurutan dilewati.
    """
    if mode not in ("recompute", "stored"):
        raise ValueError(f"mode replay tidak dikenal: {mode!r}")
    if frames not in ("processed", "all"):
        raise ValueError(f"frames replay tidak dikenal: {frames!r}")
    recording = read_recording(path)
    meta = recording.meta
    params = meta.get("params") or {}
    order = quaternion_order or meta["quat_order"]
    gain = flexion_gain if flexion_gain is not None else params.get("flexion_gain", 1.0)
    cutoff = (
        filter_cutoff_hz
        if filter_cutoff_hz is not None
        else params.get("filter_cutoff_hz", FILTER_CUTOFF_HZ)
    )
    env_name = params.get("env", ENV_NAME)

    warnings: list[str] = []
    if recording.truncated:
        warnings.append("baris terakhir rekaman terpotong dan dibuang")
    if frames == "all" and recording.capture != "all":
        warnings.append(
            "rekaman hanya berisi frame yang diproses saat live (capture=processed); "
            "mode 'all' tidak menambah data"
        )
    if meta.get("capture") is not None and not recording.ended_cleanly:
        warnings.append(
            "rekaman tidak ditutup dengan event session_end (sesi berhenti tidak normal?)"
        )
    recorded_axes = (meta.get("wrist_axes") or {}).get("sha256")
    current_axes = (describe_axes_file(DEFAULT_AXES_FILE) or {}).get("sha256")
    if recorded_axes and current_axes and recorded_axes != current_axes:
        warnings.append(
            "wrist_axes.json saat ini berbeda dari yang dipakai saat merekam; "
            "hasil pemetaan wrist bisa berbeda"
        )

    use_stored = mode == "stored" or calib_from is not None
    stored_open = stored_grip = None
    resume_n = 0
    if use_stored:
        source = read_recording(calib_from) if calib_from is not None else recording
        if source.meta["quat_order"] != order:
            raise RecordingError(
                "urutan quaternion rekaman sumber kalibrasi "
                f"({source.meta['quat_order']}) berbeda dari replay ({order})"
            )
        stored_open, stored_grip, source_resume = extract_stored_calibration(source)
        if calib_from is None:
            resume_n = source_resume
        else:
            try:
                resume_n = extract_stored_calibration(recording)[2]
            except RecordingError:
                resume_n = 0

    env = gym.make(env_name)
    try:
        pipeline = build_pipeline(env, order, gain, cutoff)
        session = pipeline.calibration
        if use_stored:
            session.apply_stored(stored_open, stored_grip)  # type: ignore[arg-type]

        items = recording.items
        if frames == "all":
            glove_times = iter(glove_time_seconds(recording.frames))
            items = [
                replace(item, t=next(glove_times)) if isinstance(item, Frame) else item
                for item in items
            ]

        qpos_log: list[np.ndarray] = []
        frames_total = 0
        last_seq: int | None = None
        filter_primed = not use_stored
        viewer_context = (
            mujoco.viewer.launch_passive(pipeline.model, pipeline.data)
            if show_viewer
            else contextlib.nullcontext()
        )
        previous_t: float | None = None
        with viewer_context as viewer:
            for item in items:
                if viewer is not None and not viewer.is_running():
                    break
                if isinstance(item, Event):
                    if use_stored:
                        continue
                    if item.name == "calib_open_start":
                        session.start_open(item.n, item.t)
                    elif item.name == "calib_grip_start":
                        session.start_grip(item.n, item.t, item.payload.get("attempt"))
                    elif item.name in ("calib_open_done", "calib_grip_done"):
                        problem = _compare_calibration(session, item)
                        if problem is not None:
                            warnings.append(problem)
                    continue

                frames_total += 1
                is_duplicate = (
                    frames == "all"
                    and item.glove_seq is not None
                    and item.glove_seq == last_seq
                )
                last_seq = item.glove_seq
                if use_stored and item.n < resume_n:
                    continue
                if frames == "processed" and not item.processed:
                    continue
                if session.phase in ("open", "grip"):
                    # Kalibrasi memakai hanya frame yang diproses saat live,
                    # sehingga rata-ratanya sama dengan sesi aslinya.
                    if not item.processed:
                        continue
                    quats = frame_quaternions(item, order)
                    if session.feed(item, quats) != "grip_done":
                        continue
                elif not session.ready or is_duplicate:
                    continue
                else:
                    quats = frame_quaternions(item, order)

                if not filter_primed:
                    # Mode stored: jalur live dan recompute melakukan reset(frame
                    # penutup) lalu update pada waktu yang sama. Meniru urutan
                    # operasi itu persis; inisialisasi malas oleh update() tidak
                    # setara bit-per-bit (normalisasi ulang tidak selalu idempoten).
                    pipeline.quaternion_filter.reset(quats, item.t)
                    filter_primed = True

                qpos_log.append(map_frame(pipeline, quats, item.t))
                if viewer is not None:
                    if realtime and previous_t is not None:
                        time.sleep(max(0.0, item.t - previous_t))
                    previous_t = item.t
                    viewer.sync()
    finally:
        env.close()

    if not qpos_log:
        raise RecordingError(
            "tidak ada frame yang dipetakan: rekaman tidak memuat kalibrasi lengkap. "
            "Gunakan calib_from untuk meminjam kalibrasi dari rekaman lain."
        )
    return ReplayResult(
        qpos=np.array(qpos_log),
        warnings=warnings,
        frames_total=frames_total,
        frames_mapped=len(qpos_log),
        meta=meta,
    )
