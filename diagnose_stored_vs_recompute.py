"""Diagnostik: mengapa replay 'stored' dan 'recompute' (mode all) tidak identik bit-per-bit?

Jalankan dari ROOT repo (folder yang berisi cli.py), di venv yang sama dengan pytest:

    python diagnose_stored_vs_recompute.py

Skrip ini hanya membaca dan menjalankan kode repo; tidak mengubah file apa pun
(rekaman sementara ditulis ke folder temp). Hapus setelah selesai.
"""

import contextlib
import io
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "test")
import runtime_fakes as fakes  # noqa: E402
from test_runtime_replay import RecordAllTests, replay_quiet  # noqa: E402

from smartglove import runtime  # noqa: E402

import mujoco  # noqa: E402

print("python", platform.python_version(), "|", platform.platform())
print("numpy", np.__version__, "| mujoco", mujoco.__version__)

plan = RecordAllTests.build_plan()
path = Path(tempfile.mkdtemp()) / "all.jsonl"
with contextlib.redirect_stdout(io.StringIO()):
    live = fakes.run_fake_live_session(
        packets=plan["main"], batch_sizes=plan["sizes"],
        stale_packets=plan["stale"], record_path=path, record_all=True,
    )

# Toleransi 0: selisih kalibrasi sekecil apa pun antara hasil hitung ulang dan
# nilai yang tersimpan akan muncul sebagai peringatan lengkap dengan besarnya.
runtime.CALIBRATION_MATCH_TOLERANCE = 0.0

for frames in ("processed", "all"):
    print(f"\n=== frames={frames}")
    first = replay_quiet(path, frames=frames, mode="recompute")
    second = replay_quiet(path, frames=frames, mode="recompute")
    stored = replay_quiet(path, frames=frames, mode="stored")
    print("recompute dua kali identik          :", np.array_equal(first.qpos, second.qpos))
    if frames == "processed":
        print("recompute == sesi live              :", np.array_equal(first.qpos, live))
    print("recompute == stored                 :", np.array_equal(first.qpos, stored.qpos))
    diff = np.abs(first.qpos - stored.qpos).max(axis=1)
    bad = np.flatnonzero(diff > 0)
    print(f"frame berbeda                       : {len(bad)} dari {len(diff)}")
    if len(bad):
        print("indeks frame berbeda pertama        :", [int(i) for i in bad[:8]])
        print("selisih di frame-frame itu (rad)    :", [float(f"{diff[i]:.2e}") for i in bad[:8]])
        print(f"selisih maksimum                    : {diff.max():.3e} rad")
    print("peringatan recompute (toleransi 0)  :", first.warnings or "tidak ada")
