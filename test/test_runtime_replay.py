"""Regresi jalur live (golden master) dan replay JSONL.

Butuh MyoSuite 2.x (nama joint tanpa akhiran ``_r``, seperti yang diharapkan
mapper). Tes dilewati bila model tidak cocok.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import runtime_fakes as fakes

try:
    from smartglove import runtime
    from smartglove.frame import RecordingError
    from smartglove.recorder import JsonlRecorder, read_recording
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest(f"dependensi runtime tidak ada: {error}")


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def replay_quiet(path, **kwargs):
    """Replay headless dengan pengendali otot palsu (tidak mengubah qpos)."""
    with mock.patch.object(runtime, "SmartGloveMuscleController", fakes.RecordingController):
        return quiet(runtime.replay, path, show_viewer=False, **kwargs)


class LiveAndReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._directory.cleanup)
        cls.record_path = Path(cls._directory.name) / "session.jsonl"
        try:
            cls.live_qpos = quiet(
                fakes.run_fake_live_session, record_path=cls.record_path
            )
        except ValueError as error:
            if "Objek MuJoCo tidak ditemukan" in str(error):
                raise unittest.SkipTest(
                    "model MyoSuite tidak cocok (butuh 2.x dengan nama joint tanpa '_r')"
                ) from error
            raise
        cls.recording = read_recording(cls.record_path)

    # ------------------------------------------------------------- golden

    def test_live_path_matches_golden_master(self) -> None:
        # Golden dibuat di Linux; di platform lain selisih ~1e-9 itu wajar.
        # Toleransi 1e-6 jauh di atas noise tapi jauh di bawah lompatan 1e-3.
        golden = np.load(fakes.GOLDEN_PATH)["qpos"]
        self.assertEqual(self.live_qpos.shape, golden.shape)
        np.testing.assert_allclose(self.live_qpos, golden, rtol=0, atol=1e-6)

    # ------------------------------------------------------------ rekaman

    def test_recording_has_every_processed_frame_and_calibration_events(self) -> None:
        total = fakes.OPEN_PACKETS + fakes.GRIP_PACKETS + fakes.MOTION_PACKETS
        frames = self.recording.frames
        self.assertEqual(len(frames), total)
        self.assertEqual([f.n for f in frames], list(range(1, total + 1)))
        self.assertTrue(all(f.glove_seq is not None for f in frames))
        names = [e.name for e in self.recording.events]
        self.assertEqual(
            names,
            ["calib_open_start", "calib_open_done", "calib_grip_start", "calib_grip_done"],
        )
        open_done = self.recording.events[1]
        self.assertEqual(open_done.n, fakes.OPEN_PACKETS)
        self.assertEqual(
            set(open_done.payload["neutral"]),
            {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY", "FOREARM", "HAND"},
        )
        grip_done = self.recording.events[3]
        self.assertTrue(grip_done.payload["ok"])
        self.assertEqual(grip_done.n, fakes.OPEN_PACKETS + fakes.GRIP_PACKETS)
        self.assertEqual(set(grip_done.payload["grip"]), {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY"})

    def test_meta_records_session_parameters(self) -> None:
        meta = self.recording.meta
        self.assertEqual(meta["params"]["flexion_gain"], 1.0)
        self.assertEqual(meta["params"]["filter_cutoff_hz"], 12.0)
        self.assertEqual(meta["params"]["calibration_packets"], 30)
        self.assertEqual(meta["quat_order"], "xyzw")
        self.assertIn("sha256", meta["wrist_axes"])

    # -------------------------------------------------------------- replay

    def test_replay_recompute_equals_live_bit_for_bit(self) -> None:
        result = replay_quiet(self.record_path)
        self.assertTrue(np.array_equal(result.qpos, self.live_qpos))
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.frames_total, len(self.recording.frames))
        self.assertEqual(result.frames_mapped, len(self.live_qpos))

    def test_replay_twice_is_deterministic(self) -> None:
        first = replay_quiet(self.record_path).qpos
        second = replay_quiet(self.record_path).qpos
        self.assertTrue(np.array_equal(first, second))

    def test_replay_stored_mode_matches_live(self) -> None:
        result = replay_quiet(self.record_path, mode="stored")
        self.assertEqual(result.qpos.shape, self.live_qpos.shape)
        self.assertTrue(np.array_equal(result.qpos, self.live_qpos))

    def test_calib_from_lets_a_calibration_free_recording_replay(self) -> None:
        grip_done_n = self.recording.events[3].n
        motion_only = Path(self._directory.name) / "motion_only.jsonl"
        with JsonlRecorder(motion_only, self.recording.meta) as recorder:
            for frame in self.recording.frames:
                if frame.n >= grip_done_n:
                    recorder.write_frame(frame)
        with self.assertRaisesRegex(RecordingError, "calib_from"):
            replay_quiet(motion_only)
        result = replay_quiet(motion_only, calib_from=self.record_path)
        self.assertEqual(result.qpos.shape, self.live_qpos.shape)
        np.testing.assert_allclose(result.qpos, self.live_qpos, rtol=0, atol=1e-9)

    def test_tampered_stored_calibration_produces_a_warning(self) -> None:
        lines = self.record_path.read_text().split("\n")
        for index, line in enumerate(lines):
            if '"calib_open_done"' in line:
                record = json.loads(line)
                record["neutral"]["INDEX"][0] += 0.01
                lines[index] = json.dumps(record)
        tampered = Path(self._directory.name) / "tampered.jsonl"
        tampered.write_text("\n".join(lines))
        result = replay_quiet(tampered)
        self.assertTrue(any("calib_open_done" in w for w in result.warnings), result.warnings)
        # hasil tetap berasal dari frame yang direkam, bukan dari nilai yang diubah
        self.assertTrue(np.array_equal(result.qpos, self.live_qpos))

    def test_changed_wrist_axes_hash_produces_a_warning(self) -> None:
        lines = self.record_path.read_text().split("\n")
        meta = json.loads(lines[0])
        meta["wrist_axes"]["sha256"] = "0" * 64
        lines[0] = json.dumps(meta)
        changed = Path(self._directory.name) / "axes_changed.jsonl"
        changed.write_text("\n".join(lines))
        result = replay_quiet(changed)
        self.assertTrue(any("wrist_axes" in w for w in result.warnings), result.warnings)

    def test_quaternion_order_mismatch_with_calib_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(RecordingError, "urutan quaternion"):
            replay_quiet(self.record_path, calib_from=self.record_path, quaternion_order="wxyz")


class FailedGripRetryTests(unittest.TestCase):
    def test_failed_grip_is_recorded_and_replay_uses_the_successful_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.jsonl"
            try:
                live = quiet(
                    fakes.run_fake_live_session,
                    packets=fakes.session_packets_with_failed_grip(),
                    record_path=path,
                )
            except ValueError as error:
                if "Objek MuJoCo tidak ditemukan" in str(error):
                    self.skipTest("model MyoSuite tidak cocok")
                raise
            recording = read_recording(path)
            events = [(e.name, e.payload.get("attempt"), e.payload.get("ok")) for e in recording.events]
            self.assertEqual(
                events,
                [
                    ("calib_open_start", None, None),
                    ("calib_open_done", None, None),
                    ("calib_grip_start", 1, None),
                    ("calib_grip_done", 1, False),
                    ("calib_grip_start", 2, None),
                    ("calib_grip_done", 2, True),
                ],
            )
            self.assertIn("error", recording.events[3].payload)
            result = replay_quiet(path)
            self.assertTrue(np.array_equal(result.qpos, live))
            self.assertEqual(result.warnings, [])
            stored = replay_quiet(path, mode="stored")
            self.assertTrue(np.array_equal(stored.qpos, live))


if __name__ == "__main__":
    unittest.main()
