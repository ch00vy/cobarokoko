"""Kalibrasi pose terbuka dan genggam, dipakai bersama oleh jalur live dan replay.

Sebelumnya logika ini tertanam di loop ``runtime.run`` bersama ``input()`` dan
soket. Di sini hanya ada mesin keadaan murni yang menerima ``Frame``:

    idle --start_open--> open --(N frame)--> await_grip
    await_grip --start_grip--> grip --(N frame)--> ready      (berhasil)
                                     \\--> await_grip           (gagal, ulangi)

Perilaku disamakan dengan versi lama: kalibrasi selesai pada frame ke-N
(frame itu ikut dikonsumsi kalibrasi), sedangkan frame penutup kalibrasi
genggam yang berhasil juga langsung dipetakan oleh pemanggil.

Hasil tiap tahap dipancarkan sebagai event rekaman (``calib_open_start``,
``calib_open_done``, ``calib_grip_start``, ``calib_grip_done``) lewat callback
``emit``. Payload ``*_done`` memuat quaternion rata-rata sehingga replay dapat
memakainya langsung tanpa menghitung ulang (mode ``stored``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from smartglove.config import CALIBRATION_PACKETS, FINGER_JOINTS
from smartglove.frame import Frame
from smartglove.quaternion import average_quaternions, sensor_quaternion_to_wxyz

FINGERS = tuple(FINGER_JOINTS)
WRIST_SENSORS = ("FOREARM", "HAND")

Emit = Callable[..., None]


def frame_quaternions(frame: Frame, order: str) -> dict[str, np.ndarray]:
    """Quaternion semua sensor sebagai wxyz ternormalisasi (setara
    ``runtime.capture_sensor_quaternions`` pada keluaran parser)."""
    return {
        name: sensor_quaternion_to_wxyz(sample.q, order)
        for name, sample in frame.sensors.items()
    }


def wxyz_to_order(quaternion: np.ndarray, order: str) -> list[float]:
    w, x, y, z = (float(c) for c in quaternion)
    if order == "xyzw":
        return [x, y, z, w]
    if order == "wxyz":
        return [w, x, y, z]
    raise ValueError(f"Urutan quaternion tidak didukung: {order}")


def order_to_wxyz(values: list[float], order: str) -> np.ndarray:
    """Kebalikan ``wxyz_to_order``. Hanya permutasi, tanpa normalisasi ulang,
    agar nilai tersimpan kembali persis."""
    if len(values) != 4:
        raise ValueError("Quaternion harus berisi empat nilai.")
    if order == "xyzw":
        x, y, z, w = values
        return np.array((w, x, y, z), dtype=float)
    if order == "wxyz":
        return np.array(values, dtype=float)
    raise ValueError(f"Urutan quaternion tidak didukung: {order}")


def payload_max_difference(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    """Selisih komponen terbesar antara dua payload (``neutral`` atau ``grip``)."""
    a, b = left.get(key) or {}, right.get(key) or {}
    if set(a) != set(b):
        return float("inf")
    worst = 0.0
    for name in a:
        worst = max(worst, float(np.max(np.abs(np.asarray(a[name]) - np.asarray(b[name])))))
    return worst


class CalibrationSession:
    def __init__(
        self,
        mapper,
        wrist_mapper,
        quaternion_filter,
        quat_order: str,
        packets: int = CALIBRATION_PACKETS,
        emit: Emit | None = None,
    ) -> None:
        self.mapper = mapper
        self.wrist_mapper = wrist_mapper
        self.quaternion_filter = quaternion_filter
        self.quat_order = quat_order
        self.packets = packets
        self._emit = emit
        self.phase = "idle"
        self.attempt = 0
        self.skipped_fingers: list[str] = []
        self.last_error: str | None = None
        self.last_open_payload: dict[str, Any] | None = None
        self.last_grip_payload: dict[str, Any] | None = None
        self._open_samples: dict[str, list[np.ndarray]] = {}
        self._grip_samples: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------ status

    @property
    def ready(self) -> bool:
        return self.phase == "ready"

    @property
    def sample_count(self) -> int:
        samples = self._open_samples if self.phase == "open" else self._grip_samples
        return len(samples.get("THUMB", ()))

    def _emit_event(self, name: str, n: int, t: float, **payload: Any) -> None:
        if self._emit is not None:
            self._emit(name, n, t, **payload)

    # ------------------------------------------------------------------ fase

    def start_open(self, n: int, t: float) -> None:
        self.phase = "open"
        self._open_samples = {name: [] for name in (*FINGERS, *WRIST_SENSORS)}
        self._emit_event("calib_open_start", n, t)

    def start_grip(self, n: int, t: float, attempt: int | None = None) -> None:
        self.phase = "grip"
        self.attempt = self.attempt + 1 if attempt is None else int(attempt)
        self._grip_samples = {name: [] for name in FINGERS}
        self._emit_event("calib_grip_start", n, t, attempt=self.attempt)

    def feed(self, frame: Frame, quats: dict[str, np.ndarray]) -> str:
        """Konsumsi satu frame pada fase ``open`` atau ``grip``.

        Mengembalikan ``collecting_open``, ``open_done``, ``collecting_grip``,
        ``grip_failed`` atau ``grip_done``.
        """
        if self.phase == "open":
            return self._feed_open(frame, quats)
        if self.phase == "grip":
            return self._feed_grip(frame, quats)
        raise RuntimeError(f"feed() dipanggil pada fase '{self.phase}'.")

    def _feed_open(self, frame: Frame, quats: dict[str, np.ndarray]) -> str:
        for name, quaternion in quats.items():
            if name in self._open_samples:
                self._open_samples[name].append(quaternion)
        if self.sample_count < self.packets:
            return "collecting_open"

        neutral = {
            name: average_quaternions(self._open_samples[name]) for name in FINGERS
        }
        wrist_neutral = {
            name: average_quaternions(self._open_samples[name]) for name in WRIST_SENSORS
        }
        self.mapper.calibrate(neutral)
        self.wrist_mapper.calibrate(wrist_neutral["FOREARM"], wrist_neutral["HAND"])
        self.quaternion_filter.reset({**neutral, **wrist_neutral}, frame.t)

        self.last_open_payload = {
            "samples": self.packets,
            "neutral": {
                name: wxyz_to_order(quaternion, self.quat_order)
                for name, quaternion in {**neutral, **wrist_neutral}.items()
            },
        }
        self.phase = "await_grip"
        self._emit_event("calib_open_done", frame.n, frame.t, **self.last_open_payload)
        return "open_done"

    def _feed_grip(self, frame: Frame, quats: dict[str, np.ndarray]) -> str:
        for name, quaternion in quats.items():
            if name in self._grip_samples:
                self._grip_samples[name].append(quaternion)
        if self.sample_count < self.packets:
            return "collecting_grip"

        grip = {name: average_quaternions(self._grip_samples[name]) for name in FINGERS}
        payload: dict[str, Any] = {
            "attempt": self.attempt,
            "samples": self.packets,
            "grip": {
                name: wxyz_to_order(quaternion, self.quat_order)
                for name, quaternion in grip.items()
            },
        }
        try:
            skipped = self.mapper.calibrate_grip(grip)
        except ValueError as error:
            self.last_error = str(error)
            self.last_grip_payload = {
                **payload, "ok": False, "skipped": [], "error": self.last_error,
            }
            self.phase = "await_grip"
            self._emit_event("calib_grip_done", frame.n, frame.t, **self.last_grip_payload)
            return "grip_failed"

        self.skipped_fingers = list(skipped)
        self.last_error = None
        self.quaternion_filter.reset(quats, frame.t)
        self.last_grip_payload = {
            **payload,
            "ok": True,
            "skipped": self.skipped_fingers,
            "error": None,
            "mapping_calibration": self._mapping_calibration_payload(),
        }
        self.phase = "ready"
        self._emit_event("calib_grip_done", frame.n, frame.t, **self.last_grip_payload)
        return "grip_done"

    def _mapping_calibration_payload(self) -> dict[str, Any]:
        finger_calibration = self.mapper.flexion_calibration
        if finger_calibration is None or self.mapper.sensor_to_model is None:
            raise RuntimeError("Kalibrasi pemetaan jari belum lengkap.")

        fingers: dict[str, Any] = {}
        for finger, value in finger_calibration.items():
            if value is None:
                fingers[finger] = None
                continue
            closure_axis, closed_angle, targets = value
            joint_names = dict(
                zip(
                    self.mapper.fingers[finger]["qpos"],
                    FINGER_JOINTS[finger]["joints"],
                )
            )
            fingers[finger] = {
                "closure_axis": closure_axis.tolist(),
                "closed_angle_rad": float(closed_angle),
                "targets": [
                    {
                        "joint": joint_names[qpos_id],
                        "qpos_index": int(qpos_id),
                        "open": float(target_open),
                        "closed": float(target_closed),
                    }
                    for qpos_id, (target_open, target_closed, _) in targets.items()
                ],
            }

        wrist = self.wrist_mapper
        return {
            "hand_sensor_to_model": {
                name: quaternion.tolist()
                for name, quaternion in self.mapper.sensor_to_model.items()
            },
            "finger_flexion": fingers,
            "wrist": {
                "alignment": wrist.alignment.tolist(),
                "alignment_source": wrist.alignment_source,
                "twist_axis": (
                    None if wrist.twist_axis is None else wrist.twist_axis.tolist()
                ),
            },
        }

    # ---------------------------------------------------------- mode "stored"

    def apply_stored(
        self,
        open_payload: dict[str, Any],
        grip_payload: dict[str, Any],
    ) -> None:
        """Pasang kalibrasi dari payload event tanpa mengonsumsi frame.

        Filter tidak di-reset di sini. Pemanggil harus memanggil
        ``quaternion_filter.reset(quaternion frame pertama, waktunya)`` tepat
        sebelum memetakan frame pertama, seperti pada frame penutup kalibrasi
        genggam di jalur live. Inisialisasi malas oleh ``update()`` TIDAK setara
        bit-per-bit: reset lalu update pada waktu yang sama menormalisasi ulang
        quaternion, dan normalisasi ulang tidak selalu idempoten (pada rekaman
        nyata ~0,6% quaternion berubah 1 ulp).
        """
        if not grip_payload.get("ok", False):
            raise ValueError("Payload genggam bukan percobaan yang berhasil.")
        neutral = {
            name: order_to_wxyz(values, self.quat_order)
            for name, values in open_payload["neutral"].items()
        }
        self.mapper.calibrate({name: neutral[name] for name in FINGERS})
        self.wrist_mapper.calibrate(neutral["FOREARM"], neutral["HAND"])
        grip = {
            name: order_to_wxyz(values, self.quat_order)
            for name, values in grip_payload["grip"].items()
        }
        self.skipped_fingers = list(self.mapper.calibrate_grip(grip))
        self.last_open_payload = open_payload
        self.last_grip_payload = grip_payload
        self.phase = "ready"
