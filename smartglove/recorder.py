"""Penulis dan pembaca rekaman JSONL ``smartglove.frame/1`` (lihat frame.py)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from smartglove.frame import (
    SCHEMA,
    SENSOR_SLOTS,
    Event,
    Frame,
    RecordingError,
    check_meta,
    dumps,
    event_to_record,
    frame_to_record,
    record_to_event,
    record_to_frame,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKET_BYTES = 624


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def describe_axes_file(path: Path | None) -> dict[str, Any] | None:
    """Hash dan isi wrist_axes.json, karena WristQposMapper memuatnya saat kalibrasi."""
    if path is None or not Path(path).exists():
        return None
    raw = Path(path).read_bytes()
    try:
        content = json.loads(raw)
    except json.JSONDecodeError:
        content = None
    return {"sha256": hashlib.sha256(raw).hexdigest(), "content": content}


def build_meta(
    *,
    quat_order: str,
    source: str | None,
    params: dict[str, Any],
    axes_file: Path | None = None,
    hand: str | None = None,
    device: str = "rokoko-smartglove-v1",
    packet_bytes: int = DEFAULT_PACKET_BYTES,
    capture: str = "processed",
) -> dict[str, Any]:
    meta = {
        "type": "meta",
        "schema": SCHEMA,
        "created": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "device": device,
        "hand": hand,
        "packet_bytes": packet_bytes,
        "quat_order": quat_order,
        "quat_normalized": True,
        "capture": capture,
        "sensor_slots": dict(SENSOR_SLOTS),
        "source": source,
        "app_commit": _git_commit(),
        "params": dict(params),
        "wrist_axes": describe_axes_file(axes_file),
    }
    check_meta(meta)
    return meta


class JsonlRecorder:
    """Menulis meta, frame, dan event ke satu file JSONL.

    Frame harus datang dengan ``n`` yang naik ketat; ini menjaga file tetap
    valid untuk pembaca. Buffer di-flush tiap ``flush_every`` frame dan pada
    setiap event, sehingga crash hanya menghilangkan paling banyak satu
    segmen pendek (pembaca menoleransi baris terakhir yang terpotong).
    """

    def __init__(
        self,
        path: str | Path,
        meta: dict[str, Any],
        flush_every: int = 50,
    ) -> None:
        check_meta(meta)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        self._flush_every = max(1, int(flush_every))
        self._since_flush = 0
        self._last_n: int | None = None
        self.frames_written = 0
        self._file.write(dumps(meta) + "\n")
        self._file.flush()

    def write_frame(self, frame: Frame) -> None:
        if self._last_n is not None and frame.n <= self._last_n:
            raise ValueError(
                f"Nomor frame harus naik: {frame.n} setelah {self._last_n}."
            )
        self._file.write(dumps(frame_to_record(frame)) + "\n")
        self._last_n = frame.n
        self.frames_written += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self.flush()

    def write_event(self, name: str, n: int, t: float, **payload: Any) -> None:
        event = Event(name=name, n=int(n), t=round(float(t), 6), payload=payload)
        self._file.write(dumps(event_to_record(event)) + "\n")
        self.flush()

    def flush(self) -> None:
        self._file.flush()
        self._since_flush = 0

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "JsonlRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class Recording:
    meta: dict[str, Any]
    items: list[Frame | Event] = field(default_factory=list)
    truncated: bool = False  # baris terakhir terpotong dan dibuang

    @property
    def frames(self) -> list[Frame]:
        return [item for item in self.items if isinstance(item, Frame)]

    @property
    def events(self) -> list[Event]:
        return [item for item in self.items if isinstance(item, Event)]

    @property
    def capture(self) -> str:
        return self.meta.get("capture", "processed")

    @property
    def ended_cleanly(self) -> bool:
        """True bila rekaman ditutup dengan event ``session_end`` (tidak crash)."""
        events = self.events
        return bool(events) and events[-1].name == "session_end"


def read_recording(path: str | Path, tolerate_truncated_tail: bool = True) -> Recording:
    """Baca dan validasi seluruh file.

    Baris terakhir yang bukan JSON utuh dibuang bila ``tolerate_truncated_tail``
    (akibat crash saat merekam). Kerusakan di baris lain selalu ditolak.
    """
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    numbered = [(i + 1, text) for i, text in enumerate(lines) if text.strip()]
    if not numbered:
        raise RecordingError("file rekaman kosong")

    recording: Recording | None = None
    last_n: int | None = None
    last_index = len(numbered) - 1

    for index, (line_number, text) in enumerate(numbered):
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            if tolerate_truncated_tail and index == last_index and index > 0:
                assert recording is not None
                recording.truncated = True
                break
            raise RecordingError(f"bukan JSON valid ({error.msg})", line_number) from None
        if not isinstance(record, dict):
            raise RecordingError("setiap baris harus objek JSON", line_number)

        if recording is None:
            check_meta(record, line_number)
            recording = Recording(meta=record)
            continue

        kind = record.get("type")
        if kind == "frame":
            frame = record_to_frame(record, line_number)
            if last_n is not None and frame.n <= last_n:
                raise RecordingError(
                    f"nomor frame tidak naik ({frame.n} setelah {last_n})", line_number
                )
            last_n = frame.n
            recording.items.append(frame)
        elif kind == "event":
            recording.items.append(record_to_event(record, line_number))
        elif kind == "meta":
            raise RecordingError("meta hanya boleh ada di baris pertama", line_number)
        else:
            raise RecordingError(f"type tidak dikenal: {kind!r}", line_number)

    assert recording is not None
    return recording
