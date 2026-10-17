"""Skema frame internal Smartglove: ``smartglove.frame/1`` (JSON Lines).

Satu file rekaman berisi baris-baris JSON dengan tiga jenis ``type``:

* ``meta``  - baris pertama; parameter sesi (urutan quaternion, parameter
  mapping, isi wrist_axes.json, commit kode).
* ``frame`` - satu paket UDP valid yang benar-benar diproses mapper.
* ``event`` - penanda tahap kerja (kalibrasi terbuka/genggam, catatan).

Prinsip desain:

* Runtime hanya mengonsumsi ``Frame``. Jalur live dan replay membuat objek yang
  sama, termasuk pembulatan waktu dan posisi di ``frame_from_parsed``, sehingga
  hasil replay identik bit-per-bit dengan sesi live.
* Quaternion (``q``) disimpan LOSSLESS dalam urutan kabel sensor (xyzw), sudah
  dinormalisasi oleh parser. ``meta.quat_order`` adalah interpretasi urutan
  yang dipakai sesi (flag ``--quaternion-order``).
* ``p`` (posisi mentah) satuannya belum diketahui dan tidak dipakai mapper.
  Disimpan 9 digit signifikan.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from smartglove_realtime import SENSOR_INDEX

SCHEMA = "smartglove.frame/1"
SCHEMA_PREFIX = "smartglove.frame/"
SCHEMA_MAJOR = 1

# Urutan sensor dan indeksnya mengikuti parser biner.
SENSOR_SLOTS: dict[str, int] = dict(SENSOR_INDEX)
SENSOR_NAMES: tuple[str, ...] = tuple(SENSOR_SLOTS)

QUAT_ORDERS = ("xyzw", "wxyz")
QUAT_NORM_TOLERANCE = 1e-3
TIME_DECIMALS = 6  # mikrodetik


class RecordingError(ValueError):
    """Rekaman atau baris JSON tidak valid. Pesan memuat nomor baris bila ada."""

    def __init__(self, message: str, line_number: int | None = None) -> None:
        self.line_number = line_number
        prefix = f"baris {line_number}: " if line_number is not None else ""
        super().__init__(prefix + message)


@dataclass(frozen=True)
class SensorSample:
    q: tuple[float, float, float, float]
    p: tuple[float, float, float]
    id: int


@dataclass(frozen=True)
class Frame:
    n: int
    t: float
    sensors: dict[str, SensorSample]
    glove_seq: int | None = None
    glove_tick: int | None = None


@dataclass(frozen=True)
class Event:
    name: str
    n: int
    t: float
    payload: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ pembuat


def _round_significant(value: float, digits: int = 9) -> float:
    return float(f"{value:.{digits}g}")


def frame_from_parsed(
    parsed: dict,
    n: int,
    t: float,
    glove_seq: int | None = None,
    glove_tick: int | None = None,
) -> Frame:
    """Bentuk ``Frame`` dari keluaran ``parse_packet_with_error``.

    ``t`` dibulatkan ke mikrodetik dan ``p`` ke 9 digit signifikan di sini,
    bukan saat menulis, supaya sesi live memakai angka yang persis sama dengan
    yang nanti dibaca replay.
    """
    sensors: dict[str, SensorSample] = {}
    for name in SENSOR_NAMES:
        if name not in parsed:
            raise ValueError(f"Sensor {name} tidak ada pada paket yang diparse.")
        values = parsed[name]
        sensors[name] = SensorSample(
            q=tuple(float(c) for c in values["quaternion"]),  # type: ignore[arg-type]
            p=tuple(_round_significant(float(c)) for c in values["position"]),  # type: ignore[arg-type]
            id=int(values["id"]),
        )
    return Frame(
        n=int(n),
        t=round(float(t), TIME_DECIMALS),
        sensors=sensors,
        glove_seq=glove_seq,
        glove_tick=glove_tick,
    )


def frame_to_parsed(frame: Frame) -> dict[str, dict[str, Any]]:
    """Kebalikan ``frame_from_parsed``: bentuk sama dengan parser biner."""
    return {
        name: {
            "id": sample.id,
            "position": tuple(sample.p),
            "quaternion": tuple(sample.q),
        }
        for name, sample in frame.sensors.items()
    }


# ------------------------------------------------------------- serialisasi


def frame_to_record(frame: Frame) -> dict[str, Any]:
    return {
        "type": "frame",
        "n": frame.n,
        "t": frame.t,
        "glove_seq": frame.glove_seq,
        "glove_tick": frame.glove_tick,
        "s": {
            name: {
                "q": list(sample.q),
                "p": list(sample.p),
                "id": f"0x{sample.id:08X}",
            }
            for name, sample in frame.sensors.items()
        },
    }


def event_to_record(event: Event) -> dict[str, Any]:
    reserved = {"type", "name", "n", "t"}
    clash = reserved & set(event.payload)
    if clash:
        raise ValueError(f"Payload event memakai kunci tercadang: {sorted(clash)}")
    return {"type": "event", "name": event.name, "n": event.n, "t": event.t, **event.payload}


def dumps(record: dict[str, Any]) -> str:
    """Satu baris JSON ringkas. NaN/Infinity ditolak (bukan JSON valid)."""
    return json.dumps(record, separators=(",", ":"), allow_nan=False)


# ------------------------------------------------------------------ validasi


def _require(condition: bool, message: str, line_number: int | None) -> None:
    if not condition:
        raise RecordingError(message, line_number)


def check_meta(meta: dict[str, Any], line_number: int | None = None) -> None:
    _require(meta.get("type") == "meta", "baris pertama harus bertipe 'meta'", line_number)
    schema = meta.get("schema")
    _require(
        isinstance(schema, str) and schema.startswith(SCHEMA_PREFIX),
        f"schema tidak dikenal: {schema!r}",
        line_number,
    )
    try:
        major = int(schema[len(SCHEMA_PREFIX):].split(".")[0])  # type: ignore[index]
    except ValueError:
        raise RecordingError(f"versi schema tidak valid: {schema!r}", line_number) from None
    _require(
        major == SCHEMA_MAJOR,
        f"versi mayor schema {major} tidak didukung (didukung: {SCHEMA_MAJOR})",
        line_number,
    )
    _require(
        meta.get("quat_order") in QUAT_ORDERS,
        f"quat_order harus salah satu dari {QUAT_ORDERS}",
        line_number,
    )
    slots = meta.get("sensor_slots")
    _require(
        slots == SENSOR_SLOTS,
        f"sensor_slots tidak cocok dengan parser (diharapkan {SENSOR_SLOTS})",
        line_number,
    )


def record_to_frame(record: dict[str, Any], line_number: int | None = None) -> Frame:
    sensors_raw = record.get("s")
    _require(isinstance(sensors_raw, dict), "frame tanpa objek 's'", line_number)
    _require(
        set(sensors_raw) == set(SENSOR_NAMES),  # type: ignore[arg-type]
        f"frame harus berisi tepat sensor {list(SENSOR_NAMES)}",
        line_number,
    )
    n, t = record.get("n"), record.get("t")
    _require(isinstance(n, int) and not isinstance(n, bool), "'n' harus integer", line_number)
    _require(
        isinstance(t, (int, float)) and math.isfinite(t), "'t' harus angka finite", line_number
    )

    sensors: dict[str, SensorSample] = {}
    for name in SENSOR_NAMES:
        raw = sensors_raw[name]  # type: ignore[index]
        q, p, ident = raw.get("q"), raw.get("p"), raw.get("id")
        _require(
            isinstance(q, list) and len(q) == 4 and all(
                isinstance(c, (int, float)) and math.isfinite(c) for c in q
            ),
            f"{name}: 'q' harus empat angka finite",
            line_number,
        )
        norm = math.sqrt(sum(c * c for c in q))
        _require(
            abs(norm - 1.0) <= QUAT_NORM_TOLERANCE,
            f"{name}: norma quaternion {norm:.6f} jauh dari 1",
            line_number,
        )
        _require(
            isinstance(p, list) and len(p) == 3 and all(
                isinstance(c, (int, float)) and math.isfinite(c) for c in p
            ),
            f"{name}: 'p' harus tiga angka finite",
            line_number,
        )
        try:
            ident_value = int(ident, 16) if isinstance(ident, str) else int(ident)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise RecordingError(f"{name}: 'id' tidak valid: {ident!r}", line_number) from None
        sensors[name] = SensorSample(
            q=tuple(float(c) for c in q),  # type: ignore[arg-type]
            p=tuple(float(c) for c in p),  # type: ignore[arg-type]
            id=ident_value,
        )

    def optional_int(key: str) -> int | None:
        value = record.get(key)
        if value is None:
            return None
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"'{key}' harus integer atau null",
            line_number,
        )
        return value

    return Frame(
        n=n,  # type: ignore[arg-type]
        t=float(t),  # type: ignore[arg-type]
        sensors=sensors,
        glove_seq=optional_int("glove_seq"),
        glove_tick=optional_int("glove_tick"),
    )


def record_to_event(record: dict[str, Any], line_number: int | None = None) -> Event:
    name, n, t = record.get("name"), record.get("n"), record.get("t")
    _require(isinstance(name, str) and name != "", "event tanpa 'name'", line_number)
    _require(isinstance(n, int) and not isinstance(n, bool), "'n' harus integer", line_number)
    _require(
        isinstance(t, (int, float)) and math.isfinite(t), "'t' harus angka finite", line_number
    )
    payload = {k: v for k, v in record.items() if k not in ("type", "name", "n", "t")}
    return Event(name=name, n=n, t=float(t), payload=payload)  # type: ignore[arg-type]


def count_glove_seq_gaps(frames: list[Frame]) -> int:
    """Jumlah nomor urut glove yang hilang di antara frame berurutan.

    Lompatan bisa berasal dari paket yang dibuang ``drain_udp_queue`` atau
    hilang di jaringan; keduanya tidak bisa dibedakan dari data ini saja.
    """
    gaps = 0
    previous: int | None = None
    for frame in frames:
        if frame.glove_seq is None:
            previous = None
            continue
        if previous is not None and frame.glove_seq > previous + 1:
            gaps += frame.glove_seq - previous - 1
        previous = frame.glove_seq
    return gaps
