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
        names = [
            e.name for e in self.recording.events if e.name != "mapping_frame"
        ]
        self.assertEqual(
            names,
            [
                "calib_open_start", "calib_open_done", "calib_grip_start",
                "calib_grip_done", "session_end",
            ],
        )
        self.assertTrue(self.recording.ended_cleanly)
        end = self.recording.events[-1].payload
        self.assertEqual((end["frames"], end["processed"], end["invalid_skipped"]), (total, total, 0))
        self.assertEqual(end["capture"], "processed")
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
        mapping_calibration = grip_done.payload["mapping_calibration"]
        self.assertEqual(
            set(mapping_calibration["hand_sensor_to_model"]),
            {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY"},
        )
        self.assertTrue(
            all(
                target["joint"] and "qpos_index" in target
                for target in mapping_calibration["finger_flexion"]["INDEX"][
                    "targets"
                ]
            )
        )
        self.assertIn("alignment", mapping_calibration["wrist"])
        mapping_frames = [
            event for event in self.recording.events if event.name == "mapping_frame"
        ]
        self.assertEqual(len(mapping_frames), len(self.live_qpos))
        first_mapping = mapping_frames[0].payload
        np.testing.assert_array_equal(first_mapping["qpos"], self.live_qpos[0])
        self.assertIn("joint_positions", first_mapping)
        self.assertIn("actuator_controls", first_mapping)
        self.assertIn("filtered_quaternions", first_mapping)
        self.assertEqual(set(first_mapping["wrist"]), {"angles_deg", "saturated"})

    def test_meta_records_session_parameters(self) -> None:
        meta = self.recording.meta
        self.assertEqual(meta["params"]["flexion_gain"], 1.0)
        self.assertEqual(meta["params"]["filter_cutoff_hz"], 12.0)
        self.assertEqual(meta["params"]["calibration_packets"], 30)
        self.assertEqual(meta["quat_order"], "xyzw")
        self.assertIn("sha256", meta["wrist_axes"])

    def test_live_listener_binds_to_all_interfaces_by_default(self) -> None:
        from smartglove_realtime import DEFAULT_BIND_HOST, PORT

        self.assertEqual(
            fakes.FakeSocket.last_bound_address,
            (DEFAULT_BIND_HOST, PORT),
        )

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
            events = [
                (e.name, e.payload.get("attempt"), e.payload.get("ok"))
                for e in recording.events
                if e.name != "mapping_frame"
            ]
            self.assertEqual(
                events,
                [
                    ("calib_open_start", None, None),
                    ("calib_open_done", None, None),
                    ("calib_grip_start", 1, None),
                    ("calib_grip_done", 1, False),
                    ("calib_grip_start", 2, None),
                    ("calib_grip_done", 2, True),
                    ("session_end", None, None),
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


class RecordAllTests(unittest.TestCase):
    """Mode --record-all: semua paket valid direkam, pemrosesan tidak berubah."""

    OPEN_BATCHES = 30
    GRIP_BATCHES = 30
    BATCH = 3

    @classmethod
    def build_plan(cls) -> dict:
        n_open = cls.OPEN_BATCHES * cls.BATCH
        n_grip = cls.GRIP_BATCHES * cls.BATCH
        n_stale, n_motion = 40, 110
        ordered = fakes.renumber_packets(
            fakes.load_packets(fakes.OPEN_DIR, 0, n_open)
            + fakes.load_packets(fakes.OPEN_DIR, n_open, n_stale)
            + fakes.load_packets(fakes.GRIP_AND_MOTION_DIR, 0, n_grip + n_motion)
        )
        open_ = ordered[:n_open]
        stale = ordered[n_open : n_open + n_stale]
        rest = ordered[n_open + n_stale :]
        main = open_ + rest
        sizes = (
            [cls.BATCH] * cls.OPEN_BATCHES
            + [cls.BATCH] * cls.GRIP_BATCHES
            + fakes.batch_sizes_for(n_motion)
        )
        assert sum(sizes) == len(main)
        # urutan waktu sebenarnya: open, stale, lalu sisanya
        time_order_mask = []
        cursor = 0
        for index, size in enumerate(sizes):
            mask = [False] * (size - 1) + [True]
            time_order_mask.extend(mask)
            cursor += size
            if index == cls.OPEN_BATCHES - 1:
                time_order_mask.extend([False] * n_stale)
        last_of_batch = []
        cursor = 0
        for size in sizes:
            last_of_batch.append(main[cursor + size - 1])
            cursor += size
        return dict(
            main=main, stale=stale, sizes=sizes, processed_mask=time_order_mask,
            reference=last_of_batch, total=len(main) + n_stale,
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._directory.cleanup)
        directory = Path(cls._directory.name)
        cls.plan = cls.build_plan()
        cls.path_all = directory / "all.jsonl"
        cls.path_processed = directory / "processed.jsonl"
        plan = cls.plan
        try:
            cls.qpos_all = quiet(
                fakes.run_fake_live_session, packets=plan["main"],
                batch_sizes=plan["sizes"], stale_packets=plan["stale"],
                record_path=cls.path_all, record_all=True,
            )
        except ValueError as error:
            if "Objek MuJoCo tidak ditemukan" in str(error):
                raise unittest.SkipTest("model MyoSuite tidak cocok") from error
            raise
        cls.qpos_processed = quiet(
            fakes.run_fake_live_session, packets=plan["main"],
            batch_sizes=plan["sizes"], stale_packets=plan["stale"],
            record_path=cls.path_processed,
        )
        cls.qpos_reference = quiet(
            fakes.run_fake_live_session, packets=plan["reference"]
        )
        cls.rec_all = read_recording(cls.path_all)
        cls.rec_processed = read_recording(cls.path_processed)

    # ------------------------------------------------------------ pemrosesan

    def test_recording_everything_does_not_change_the_live_output(self) -> None:
        self.assertTrue(np.array_equal(self.qpos_all, self.qpos_processed))
        self.assertTrue(np.array_equal(self.qpos_all, self.qpos_reference))

    # --------------------------------------------------------------- rekaman

    def test_every_packet_is_recorded_in_order_without_gaps(self) -> None:
        frames = self.rec_all.frames
        self.assertEqual(len(frames), self.plan["total"])
        self.assertEqual([f.n for f in frames], list(range(1, self.plan["total"] + 1)))
        from smartglove.frame import count_glove_seq_gaps
        self.assertEqual(count_glove_seq_gaps(frames), 0)
        ticks = [f.glove_tick for f in frames]
        self.assertEqual({b - a for a, b in zip(ticks, ticks[1:])}, {12})
        self.assertEqual(self.rec_all.capture, "all")

    def test_only_the_last_packet_of_each_batch_is_marked_processed(self) -> None:
        flags = [f.processed for f in self.rec_all.frames]
        self.assertEqual(flags, self.plan["processed_mask"])
        self.assertEqual(sum(flags), len(self.plan["sizes"]))

    def test_packets_that_piled_up_during_the_prompt_are_recorded(self) -> None:
        items = self.rec_all.items
        names = [(i, it.name) for i, it in enumerate(items) if hasattr(it, "name")]
        open_done = next(i for i, n in names if n == "calib_open_done")
        grip_start = next(i for i, n in names if n == "calib_grip_start")
        between = items[open_done + 1 : grip_start]
        self.assertEqual(len(between), len(self.plan["stale"]))
        self.assertTrue(all(not f.processed for f in between))
        self.assertEqual(items[grip_start].n, self.OPEN_BATCHES * self.BATCH + len(self.plan["stale"]))

    def test_processed_only_recording_of_the_same_session_stays_small(self) -> None:
        frames = self.rec_processed.frames
        self.assertEqual(len(frames), len(self.plan["sizes"]))
        self.assertTrue(all(f.processed for f in frames))
        self.assertEqual(self.rec_processed.capture, "processed")

    def test_session_end_reports_counts(self) -> None:
        end = self.rec_all.events[-1]
        self.assertEqual(end.name, "session_end")
        self.assertEqual(end.payload["capture"], "all")
        self.assertEqual(end.payload["frames"], self.plan["total"])
        self.assertEqual(end.payload["processed"], len(self.plan["sizes"]))
        self.assertEqual(end.payload["invalid_skipped"], 0)
        mapping_frames = [
            event for event in self.rec_all.events if event.name == "mapping_frame"
        ]
        self.assertEqual(len(mapping_frames), len(self.qpos_all))

    # ---------------------------------------------------------------- replay

    def test_replay_processed_on_a_record_all_file_equals_live(self) -> None:
        result = replay_quiet(self.path_all, frames="processed")
        self.assertTrue(np.array_equal(result.qpos, self.qpos_all))
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.frames_total, self.plan["total"])

    def test_replay_all_maps_every_frame_after_calibration(self) -> None:
        grip_done = next(e for e in self.rec_all.events if e.name == "calib_grip_done")
        expected = self.plan["total"] - grip_done.n + 1
        result = replay_quiet(self.path_all, frames="all")
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.frames_mapped, expected)
        self.assertGreater(result.frames_mapped, len(self.qpos_all))
        # frame penutup kalibrasi dipetakan sama persis seperti saat live
        self.assertTrue(np.array_equal(result.qpos[0], self.qpos_all[0]))

    def test_replay_all_is_deterministic_and_stored_matches_recompute(self) -> None:
        first = replay_quiet(self.path_all, frames="all").qpos
        second = replay_quiet(self.path_all, frames="all").qpos
        stored = replay_quiet(self.path_all, frames="all", mode="stored").qpos
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first, stored))

    def test_stored_mode_primes_the_filter_exactly_like_the_live_path(self) -> None:
        """Struktural (tidak bergantung pada angka floating-point platform).

        Live dan recompute memanggil reset(frame penutup) lalu update() pada
        waktu yang sama. Mode stored harus menjalankan urutan operasi filter
        yang sama persis pada frame pertama yang dipetakan, bukan membiarkan
        update() menginisialisasi secara malas (itu tidak setara bit-per-bit).
        """
        from smartglove.quaternion import QuaternionLowPassFilter

        def filter_calls(**kwargs) -> list[tuple]:
            calls: list[tuple] = []
            original_reset = QuaternionLowPassFilter.reset
            original_update = QuaternionLowPassFilter.update

            def spy_reset(self, orientations, timestamp):
                calls.append(("reset", timestamp))
                return original_reset(self, orientations, timestamp)

            def spy_update(self, orientations, timestamp):
                calls.append(("update", timestamp, self.last_time is None))
                return original_update(self, orientations, timestamp)

            with (
                mock.patch.object(QuaternionLowPassFilter, "reset", spy_reset),
                mock.patch.object(QuaternionLowPassFilter, "update", spy_update),
            ):
                replay_quiet(self.path_all, **kwargs)
            return calls

        for frames in ("processed", "all"):
            recompute = filter_calls(frames=frames)
            stored = filter_calls(frames=frames, mode="stored")
            first_mapped = recompute.index(
                next(c for c in recompute if c[0] == "update" and c[2] is False)
            )
            # urutan di sekitar frame penutup genggam: reset(t) lalu update(t) dengan filter sudah terisi
            self.assertEqual(recompute[first_mapped - 1][0], "reset", frames)
            tail = recompute[first_mapped - 1 : first_mapped + 1]
            self.assertEqual(stored[:2], tail, frames)
            # dan seterusnya identik: hanya update biasa pada waktu yang sama
            self.assertEqual(stored[2:], recompute[first_mapped + 1 :], frames)

    def test_replay_all_skips_consecutive_duplicate_sequence_numbers(self) -> None:
        from dataclasses import replace
        from smartglove.frame import Frame

        target = self.plan["total"] - 20
        duplicated = Path(self._directory.name) / "duplicated.jsonl"
        with JsonlRecorder(duplicated, self.rec_all.meta) as recorder:
            for item in self.rec_all.items:
                if isinstance(item, Frame):
                    if item.n > target:
                        recorder.write_frame(replace(item, n=item.n + 1))
                    else:
                        recorder.write_frame(item)
                    if item.n == target:
                        recorder.write_frame(replace(item, n=item.n + 1))
                else:
                    recorder.write_event(item.name, item.n, item.t, **item.payload)
        baseline = replay_quiet(self.path_all, frames="all")
        result = replay_quiet(duplicated, frames="all")
        self.assertEqual(result.frames_total, baseline.frames_total + 1)
        self.assertEqual(result.frames_mapped, baseline.frames_mapped)
        self.assertTrue(np.array_equal(result.qpos, baseline.qpos))

    def test_replay_all_on_a_processed_only_recording_warns(self) -> None:
        result = replay_quiet(self.path_processed, frames="all")
        self.assertTrue(any("capture=processed" in w for w in result.warnings), result.warnings)
        self.assertEqual(result.frames_mapped, len(self.qpos_processed))

    def test_recording_without_session_end_produces_a_warning(self) -> None:
        lines = [l for l in self.path_all.read_text().split("\n") if '"session_end"' not in l]
        crashed = Path(self._directory.name) / "crashed.jsonl"
        crashed.write_text("\n".join(lines))
        result = replay_quiet(crashed)
        self.assertTrue(any("session_end" in w for w in result.warnings), result.warnings)


class RecordAllInvalidPacketTests(unittest.TestCase):
    def test_invalid_packets_are_counted_and_a_bad_last_packet_skips_only_processing(self) -> None:
        plan = RecordAllTests.build_plan()
        main, sizes = list(plan["main"]), list(plan["sizes"])
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        motion_batches = [i for i, size in enumerate(sizes) if i >= 60 and size >= 3]
        middle_batch, last_batch = motion_batches[0], motion_batches[1]
        main[starts[middle_batch] + 1] = main[starts[middle_batch] + 1][:100]
        main[starts[last_batch] + sizes[last_batch] - 1] = main[starts[last_batch] + sizes[last_batch] - 1][:100]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            try:
                qpos = quiet(
                    fakes.run_fake_live_session, packets=main, batch_sizes=sizes,
                    stale_packets=plan["stale"], record_path=path, record_all=True,
                )
            except ValueError as error:
                if "Objek MuJoCo tidak ditemukan" in str(error):
                    self.skipTest("model MyoSuite tidak cocok")
                raise
            recording = read_recording(path)
            end = recording.events[-1].payload
            self.assertEqual(end["invalid_skipped"], 2)
            self.assertEqual(len(recording.frames), plan["total"] - 2)
            self.assertEqual(end["processed"], len(sizes) - 1)
            # batch dengan paket terakhir rusak: paket valid sebelumnya tetap terekam
            self.assertEqual(
                sum(1 for f in recording.frames if not f.processed),
                (plan["total"] - len(sizes)) - 2 + 1,
            )
            baseline_rows = len(plan["sizes"]) - 60 + 1
            self.assertEqual(len(qpos), baseline_rows - 1)
            from smartglove.frame import count_glove_seq_gaps
            self.assertEqual(count_glove_seq_gaps(recording.frames), 2)
