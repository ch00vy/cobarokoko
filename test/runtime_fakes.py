"""Harness tanpa glove, tanpa jendela viewer, dan tanpa jam nyata.

Menjalankan ``smartglove.runtime.run`` (jalur live) dengan paket UDP dari
capture di repo. Dipakai untuk golden master: qpos tiap frame yang diproses
direkam lewat pengendali otot palsu, lalu dibandingkan antar versi kode.

Titik tambal yang harus dipertahankan oleh runtime:
``smartglove.runtime.socket``, ``smartglove.runtime.time``,
``smartglove.runtime.mujoco.viewer.launch_passive``,
``smartglove.runtime.SmartGloveMuscleController``, dan ``builtins.input``.
"""

from __future__ import annotations

import builtins
import socket as real_socket
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN_PATH = Path(__file__).with_name("data") / "golden_runtime_qpos.npz"

OPEN_DIR = REPO / "smartglove_capture_netral"
GRIP_AND_MOTION_DIR = REPO / "smartglove_capture_jari_only"
OPEN_PACKETS = 30
GRIP_PACKETS = 30
MOTION_PACKETS = 100
CLOCK_STEP = 0.015625  # 2**-6: tepat di biner DAN 6 desimal (Frame.t dibulatkan ke 6 desimal)


def load_packets(directory: Path, start: int, count: int) -> list[bytes]:
    files = sorted(directory.glob("packet_*.bin"))[start : start + count]
    if len(files) != count:
        raise RuntimeError(f"{directory} hanya punya {len(files)} paket")
    return [f.read_bytes() for f in files]


def session_packets_with_failed_grip() -> list[bytes]:
    """Genggam pertama gagal (tangan tidak berubah dari pose terbuka), lalu diulang."""
    return (
        load_packets(OPEN_DIR, 0, OPEN_PACKETS)
        + load_packets(OPEN_DIR, OPEN_PACKETS, GRIP_PACKETS)  # tidak ada perubahan -> gagal
        + load_packets(GRIP_AND_MOTION_DIR, 0, GRIP_PACKETS + MOTION_PACKETS)
    )


def session_packets() -> list[bytes]:
    """Urutan: pose terbuka, pose genggam, lalu gerakan (kalibrasi menelan 30+30)."""
    return load_packets(OPEN_DIR, 0, OPEN_PACKETS) + load_packets(
        GRIP_AND_MOTION_DIR, 0, GRIP_PACKETS + MOTION_PACKETS
    )


def renumber_packets(
    packets: list[bytes], start_seq: int = 1000, start_tick: int = 50_000, step_ms: int = 12
) -> list[bytes]:
    """Tulis ulang seq (0x24) dan tick (0x20) di header agar aliran sintetis dari
    beberapa capture menjadi satu aliran kontinu seperti glove sungguhan."""
    out = []
    for index, packet in enumerate(packets):
        buffer = bytearray(packet)
        struct.pack_into("<II", buffer, 0x20, start_tick + index * step_ms, start_seq + index)
        out.append(bytes(buffer))
    return out


def batch_sizes_for(total: int, pattern: tuple[int, ...] = (4, 3, 2, 3, 1)) -> list[int]:
    """Ukuran gelombang baca (paket per iterasi loop) yang jumlahnya tepat ``total``."""
    sizes: list[int] = []
    cursor = 0
    while sum(sizes) < total:
        sizes.append(pattern[cursor % len(pattern)])
        cursor += 1
    sizes[-1] -= sum(sizes) - total
    return sizes


class FakeSocket:
    """Meniru soket UDP.

    ``batch_sizes`` menentukan berapa paket yang sudah mengantre tiap kali loop
    membaca: satu dikembalikan recvfrom biasa, sisanya oleh drain (non-blocking).
    Default: satu paket per iterasi, sehingga drain selalu melihat antrean kosong.
    """

    def __init__(
        self,
        packets: list[bytes],
        state: SimpleNamespace,
        batch_sizes: list[int] | None = None,
    ) -> None:
        self._packets = list(packets)
        self._index = 0
        self._state = state
        self._nonblocking = False
        self._pending: list[bytes] = []
        self._batch_sizes = list(batch_sizes) if batch_sizes is not None else None
        self._batch_index = 0
        state.sockets.append(self)

    def inject(self, packets: list[bytes]) -> None:
        """Paket yang menumpuk di buffer OS, misalnya saat menunggu Enter."""
        self._pending.extend(packets)

    def setsockopt(self, *args: object) -> None: ...
    def bind(self, *args: object) -> None: ...
    def close(self) -> None: ...

    def setblocking(self, flag: bool) -> None:
        self._nonblocking = not flag

    def settimeout(self, value: float | None) -> None:
        self._nonblocking = False

    def recvfrom(self, size: int):
        if self._nonblocking:
            if self._pending:
                return self._pending.pop(0), ("127.0.0.1", 14041)
            raise BlockingIOError
        if not self._pending:
            if self._index >= len(self._packets):
                self._state.exhausted = True
                raise real_socket.timeout
            if self._batch_sizes is None:
                count = 1
            else:
                count = self._batch_sizes[self._batch_index]
                self._batch_index += 1
            self._pending = self._packets[self._index : self._index + count]
            self._index += len(self._pending)
        return self._pending.pop(0), ("127.0.0.1", 14041)


class FakeViewer:
    def __init__(self, state: SimpleNamespace) -> None:
        self._state = state

    def is_running(self) -> bool:
        return not self._state.exhausted

    def sync(self) -> None: ...

    def __enter__(self) -> "FakeViewer":
        return self

    def __exit__(self, *exc_info: object) -> None: ...


class RecordingController:
    """Pengganti SmartGloveMuscleController: hanya menyimpan qpos tiap frame."""

    qpos_log: list[np.ndarray] = []

    def __init__(self, model, data, finger_joints) -> None: ...

    def update(self, qpos: np.ndarray) -> None:
        RecordingController.qpos_log.append(np.array(qpos, dtype=float).copy())


def run_fake_live_session(
    packets: list[bytes] | None = None,
    quaternion_order: str = "xyzw",
    flexion_gain: float = 1.0,
    filter_cutoff_hz: float = 12.0,
    batch_sizes: list[int] | None = None,
    stale_packets: list[bytes] | None = None,
    **run_kwargs,
) -> np.ndarray:
    """Jalankan ``smartglove.runtime.run`` dan kembalikan qpos (frame x nq).

    ``stale_packets`` disuntik ke buffer soket saat prompt Enter kedua (sebelum
    kalibrasi genggam), seperti paket yang menumpuk selama pengguna menunggu.
    """
    from smartglove import runtime

    state = SimpleNamespace(exhausted=False, sockets=[])
    prompts = {"count": 0}

    def fake_input(*args: object) -> str:
        prompts["count"] += 1
        if prompts["count"] == 2 and stale_packets:
            state.sockets[0].inject(list(stale_packets))
        return ""

    packets = session_packets() if packets is None else packets
    clock = {"t": 1000.0}

    def perf_counter() -> float:
        clock["t"] += CLOCK_STEP
        return clock["t"]

    fake_socket_module = SimpleNamespace(
        socket=lambda *a, **k: FakeSocket(packets, state, batch_sizes),
        AF_INET=real_socket.AF_INET,
        SOCK_DGRAM=real_socket.SOCK_DGRAM,
        SOL_SOCKET=real_socket.SOL_SOCKET,
        SO_RCVBUF=real_socket.SO_RCVBUF,
        timeout=real_socket.timeout,
    )
    fake_time = SimpleNamespace(perf_counter=perf_counter, sleep=lambda s: None)
    fake_viewer_module = SimpleNamespace(launch_passive=lambda model, data: FakeViewer(state))

    RecordingController.qpos_log = []
    with (
        mock.patch.object(runtime, "socket", fake_socket_module),
        mock.patch.object(runtime, "time", fake_time),
        mock.patch.object(runtime.mujoco, "viewer", fake_viewer_module),
        mock.patch.object(runtime, "SmartGloveMuscleController", RecordingController),
        mock.patch.object(builtins, "input", fake_input),
    ):
        runtime.run(quaternion_order, flexion_gain, filter_cutoff_hz, **run_kwargs)
    return np.array(RecordingController.qpos_log)


def save_golden() -> None:
    qpos = run_fake_live_session()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, qpos=qpos)
    print(f"golden: {qpos.shape} -> {GOLDEN_PATH}")


if __name__ == "__main__":
    save_golden()
