import json
import math
import tempfile
import unittest
from pathlib import Path

from smartglove.frame import (
    SENSOR_NAMES,
    Event,
    RecordingError,
    count_glove_seq_gaps,
    dumps,
    frame_from_parsed,
    frame_to_parsed,
    frame_to_record,
    record_to_frame,
)
from smartglove.recorder import JsonlRecorder, build_meta, read_recording
from smartglove_realtime import parse_packet_header, parse_packet_with_error

CAPTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "capture_finger"
    / "smartglove_capture_pinky"
)


def synthetic_parsed(seed: float = 0.0) -> dict:
    parsed = {}
    for index, name in enumerate(SENSOR_NAMES):
        raw = (0.1 + seed + index, -0.3, 0.7 + 0.01 * index, 0.5)
        norm = math.sqrt(sum(c * c for c in raw))
        parsed[name] = {
            "id": index << 8,
            "position": (123.456789012 + index, -0.000123456789, 1e-9 * (index + 1)),
            "quaternion": tuple(c / norm for c in raw),
        }
    return parsed


def real_packets(limit: int | None = None) -> list[bytes]:
    files = sorted(CAPTURE_DIR.glob("packet_*.bin"))
    if limit is not None:
        files = files[:limit]
    return [f.read_bytes() for f in files]


def make_meta() -> dict:
    return build_meta(
        quat_order="xyzw",
        source="127.0.0.1:14041",
        params={"flexion_gain": 1.0, "filter_cutoff_hz": 12.0},
    )


class FrameTests(unittest.TestCase):
    def test_parsed_round_trip_keeps_quaternions_exactly(self) -> None:
        parsed = synthetic_parsed()
        frame = frame_from_parsed(parsed, n=1, t=0.1234567891, glove_seq=5, glove_tick=60)
        back = frame_to_parsed(frame)
        for name in SENSOR_NAMES:
            self.assertEqual(back[name]["quaternion"], parsed[name]["quaternion"])
            self.assertEqual(back[name]["id"], parsed[name]["id"])
        self.assertEqual(frame.t, 0.123457)  # dibulatkan ke mikrodetik

    def test_json_round_trip_is_bit_exact(self) -> None:
        frame = frame_from_parsed(synthetic_parsed(0.37), n=7, t=3.25, glove_seq=1, glove_tick=2)
        line = dumps(frame_to_record(frame))
        self.assertNotIn("\n", line)
        self.assertEqual(record_to_frame(json.loads(line)), frame)

    def test_position_keeps_nine_significant_digits(self) -> None:
        frame = frame_from_parsed(synthetic_parsed(), n=1, t=0.0)
        self.assertEqual(frame.sensors["THUMB"].p[0], 123.456789)  # 9 digit signifikan

    def test_nan_is_rejected_when_serialising(self) -> None:
        record = frame_to_record(frame_from_parsed(synthetic_parsed(), n=1, t=0.0))
        record["s"]["THUMB"]["q"][0] = float("nan")
        with self.assertRaises(ValueError):
            dumps(record)

    def test_reader_rejects_missing_sensor(self) -> None:
        record = frame_to_record(frame_from_parsed(synthetic_parsed(), n=1, t=0.0))
        del record["s"]["HAND"]
        with self.assertRaises(RecordingError):
            record_to_frame(record, 3)

    def test_reader_rejects_non_unit_quaternion(self) -> None:
        record = frame_to_record(frame_from_parsed(synthetic_parsed(), n=1, t=0.0))
        record["s"]["INDEX"]["q"] = [2.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(RecordingError, "norma"):
            record_to_frame(record)

    def test_glove_seq_gap_counting(self) -> None:
        frames = [
            frame_from_parsed(synthetic_parsed(), n=i + 1, t=i * 0.01, glove_seq=seq)
            for i, seq in enumerate((10, 11, 14, 15, 20))
        ]
        self.assertEqual(count_glove_seq_gaps(frames), 2 + 4)


class RealPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packets = real_packets()
        if not cls.packets:
            raise unittest.SkipTest("capture_finger/smartglove_capture_pinky tidak ada")

    def test_header_counters_advance_one_seq_per_packet(self) -> None:
        headers = [parse_packet_header(p) for p in self.packets]
        ticks = [h[0] for h in headers]
        seqs = [h[1] for h in headers]
        self.assertEqual({b - a for a, b in zip(seqs, seqs[1:])}, {1})
        self.assertEqual({b - a for a, b in zip(ticks, ticks[1:])}, {12})

    def test_header_of_short_packet_is_none(self) -> None:
        self.assertEqual(parse_packet_header(b"\x00" * 10), (None, None))

    def test_binary_to_json_to_parsed_matches_binary_parser(self) -> None:
        for number, packet in enumerate(self.packets, start=1):
            parsed, error = parse_packet_with_error(packet)
            self.assertIsNone(error)
            tick, seq = parse_packet_header(packet)
            frame = frame_from_parsed(parsed, n=number, t=number * 0.012, glove_seq=seq, glove_tick=tick)
            restored = record_to_frame(json.loads(dumps(frame_to_record(frame))))
            again = frame_to_parsed(restored)
            for name in SENSOR_NAMES:
                self.assertEqual(again[name]["quaternion"], parsed[name]["quaternion"])
                self.assertEqual(again[name]["id"], parsed[name]["id"])
            self.assertEqual(restored.glove_seq, seq)

    def test_recording_of_all_real_packets_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.jsonl"
            written = []
            with JsonlRecorder(path, make_meta()) as recorder:
                for number, packet in enumerate(self.packets, start=1):
                    parsed, _ = parse_packet_with_error(packet)
                    tick, seq = parse_packet_header(packet)
                    frame = frame_from_parsed(parsed, number, number * 0.012, seq, tick)
                    recorder.write_frame(frame)
                    written.append(frame)
            recording = read_recording(path)
            self.assertEqual(recording.frames, written)
            self.assertFalse(recording.truncated)
            self.assertEqual(count_glove_seq_gaps(recording.frames), 0)


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "nested" / "rec.jsonl"

    def write_sample(self, frames: int = 5) -> None:
        with JsonlRecorder(self.path, make_meta(), flush_every=2) as recorder:
            recorder.write_event("calib_open_start", 0, 0.0)
            for i in range(1, frames + 1):
                recorder.write_frame(frame_from_parsed(synthetic_parsed(i * 0.01), i, i * 0.01, i, i * 12))
            recorder.write_event("calib_open_done", frames, frames * 0.01, samples=frames,
                                 neutral={"THUMB": [0.0, 0.0, 0.0, 1.0]})

    def test_meta_carries_required_fields(self) -> None:
        meta = make_meta()
        self.assertEqual(meta["schema"], "smartglove.frame/1")
        self.assertEqual(meta["quat_order"], "xyzw")
        self.assertEqual(meta["packet_bytes"], 624)
        self.assertIn("THUMB", meta["sensor_slots"])

    def test_events_and_frames_keep_order_and_payload(self) -> None:
        self.write_sample()
        recording = read_recording(self.path)
        kinds = [type(item).__name__ for item in recording.items]
        self.assertEqual(kinds, ["Event"] + ["Frame"] * 5 + ["Event"])
        done = recording.events[-1]
        self.assertEqual(done.name, "calib_open_done")
        self.assertEqual(done.payload["samples"], 5)
        self.assertEqual(done.payload["neutral"]["THUMB"], [0.0, 0.0, 0.0, 1.0])

    def test_file_is_valid_jsonl_and_flushed_on_events(self) -> None:
        with JsonlRecorder(self.path, make_meta(), flush_every=1000) as recorder:
            recorder.write_frame(frame_from_parsed(synthetic_parsed(), 1, 0.0))
            recorder.write_event("note", 1, 0.0, text="set 3")
            # tanpa close: event memaksa flush, jadi semua sudah ada di disk
            lines = self.path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 3)
            for line in lines:
                json.loads(line)

    def test_writer_rejects_non_increasing_frame_number(self) -> None:
        with JsonlRecorder(self.path, make_meta()) as recorder:
            recorder.write_frame(frame_from_parsed(synthetic_parsed(), 5, 0.0))
            with self.assertRaises(ValueError):
                recorder.write_frame(frame_from_parsed(synthetic_parsed(), 5, 0.1))

    def test_truncated_last_line_is_tolerated(self) -> None:
        self.write_sample()
        text = self.path.read_text()
        self.path.write_text(text[: len(text) - 40])  # potong baris event terakhir
        recording = read_recording(self.path)
        self.assertTrue(recording.truncated)
        self.assertEqual(len(recording.frames), 5)
        with self.assertRaises(RecordingError):
            read_recording(self.path, tolerate_truncated_tail=False)

    def test_corruption_in_the_middle_is_rejected_with_line_number(self) -> None:
        self.write_sample()
        lines = self.path.read_text().split("\n")
        lines[3] = lines[3][:20]
        self.path.write_text("\n".join(lines))
        with self.assertRaises(RecordingError) as caught:
            read_recording(self.path)
        self.assertEqual(caught.exception.line_number, 4)

    def test_non_increasing_frame_number_in_file_is_rejected(self) -> None:
        self.write_sample()
        lines = self.path.read_text().split("\n")
        # baris: 0 meta, 1 event, 2..6 frame 1..5 -> tukar frame 3 dan 4
        lines[4], lines[5] = lines[5], lines[4]
        self.path.write_text("\n".join(lines))
        with self.assertRaisesRegex(RecordingError, "tidak naik"):
            read_recording(self.path)

    def test_unknown_schema_major_is_rejected(self) -> None:
        self.write_sample()
        text = self.path.read_text().replace("smartglove.frame/1", "smartglove.frame/2", 1)
        self.path.write_text(text)
        with self.assertRaisesRegex(RecordingError, "mayor"):
            read_recording(self.path)

    def test_minor_version_and_unknown_fields_are_accepted(self) -> None:
        self.write_sample()
        lines = self.path.read_text().split("\n")
        meta = json.loads(lines[0])
        meta["schema"] = "smartglove.frame/1.3"
        meta["field_from_the_future"] = True
        lines[0] = json.dumps(meta)
        frame = json.loads(lines[2])
        frame["extra"] = 1
        lines[2] = json.dumps(frame)
        self.path.write_text("\n".join(lines))
        self.assertEqual(len(read_recording(self.path).frames), 5)

    def test_second_meta_line_and_unknown_type_are_rejected(self) -> None:
        self.write_sample()
        lines = self.path.read_text().split("\n")
        lines.insert(3, lines[0])
        self.path.write_text("\n".join(lines))
        with self.assertRaisesRegex(RecordingError, "meta"):
            read_recording(self.path)

    def test_empty_file_is_rejected(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text("")
        with self.assertRaises(RecordingError):
            read_recording(self.path)

    def test_event_payload_cannot_shadow_reserved_keys(self) -> None:
        with JsonlRecorder(self.path, make_meta()) as recorder:
            with self.assertRaises(ValueError):
                recorder.write_event("x", 0, 0.0, type="frame")


if __name__ == "__main__":
    unittest.main()
