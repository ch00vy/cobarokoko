import argparse
import math

from smartglove.config import FILTER_CUTOFF_HZ
from smartglove.runtime import replay, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Rokoko Smartglove UDP quaternions to MyoSuite hand qpos."
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("xyzw", "wxyz"),
        default=None,
        help=(
            "Urutan komponen quaternion pada paket Rokoko (live default: xyzw; "
            "replay default: sesuai meta rekaman)."
        ),
    )
    parser.add_argument(
        "--flexion-gain",
        type=float,
        default=None,
        help="Pengali fleksi jari (live default: 1.0; replay default: dari rekaman).",
    )
    parser.add_argument(
        "--filter-cutoff-hz",
        type=float,
        default=None,
        help=(
            "Cutoff filter quaternion dalam Hz; lebih rendah lebih halus "
            f"(live default: {FILTER_CUTOFF_HZ:g}; replay default: dari rekaman)."
        ),
    )
    parser.add_argument(
        "--record",
        metavar="PATH.jsonl",
        help="Rekam sesi live ke JSONL (frame yang diproses + event kalibrasi).",
    )
    parser.add_argument(
        "--record-all",
        action="store_true",
        help=(
            "Dengan --record: rekam SEMUA paket valid (sekitar 83 Hz), termasuk "
            "yang dibuang saat live. Sekitar 6 MB per menit."
        ),
    )
    parser.add_argument(
        "--replay",
        metavar="PATH.jsonl",
        help="Putar ulang rekaman JSONL tanpa glove memakai pipeline yang sama.",
    )
    parser.add_argument(
        "--replay-calib",
        choices=("recompute", "stored"),
        default="recompute",
        help=(
            "recompute: hitung ulang kalibrasi dari frame di rekaman (default); "
            "stored: pakai nilai kalibrasi yang tersimpan di event."
        ),
    )
    parser.add_argument(
        "--replay-frames",
        choices=("processed", "all"),
        default="processed",
        help=(
            "processed: hanya frame yang diproses saat live (default, identik "
            "dengan sesi live); all: semua frame dengan waktu jam glove."
        ),
    )
    parser.add_argument(
        "--calib-from",
        metavar="PATH.jsonl",
        help="Pinjam kalibrasi dari rekaman lain (untuk rekaman tanpa kalibrasi).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Replay tanpa jendela viewer (headless).",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Replay secepat mungkin, tidak mengikuti waktu rekaman.",
    )
    args = parser.parse_args()

    if args.flexion_gain is not None and (
        not math.isfinite(args.flexion_gain) or args.flexion_gain <= 0.0
    ):
        parser.error("--flexion-gain harus finite dan lebih besar dari nol.")
    if args.filter_cutoff_hz is not None and (
        not math.isfinite(args.filter_cutoff_hz) or args.filter_cutoff_hz <= 0.0
    ):
        parser.error("--filter-cutoff-hz harus finite dan lebih besar dari nol.")

    if args.replay is None:
        replay_only = [
            flag
            for flag, used in (
                ("--calib-from", args.calib_from is not None),
                ("--replay-calib", args.replay_calib != "recompute"),
                ("--replay-frames", args.replay_frames != "processed"),
                ("--no-viewer", args.no_viewer),
                ("--no-realtime", args.no_realtime),
            )
            if used
        ]
        if replay_only:
            parser.error(f"{', '.join(replay_only)} hanya berlaku bersama --replay.")
        if args.record_all and args.record is None:
            parser.error("--record-all membutuhkan --record PATH.jsonl.")
        run(
            args.quaternion_order or "xyzw",
            1.0 if args.flexion_gain is None else args.flexion_gain,
            FILTER_CUTOFF_HZ if args.filter_cutoff_hz is None else args.filter_cutoff_hz,
            record_path=args.record,
            record_all=args.record_all,
        )
        return

    if args.record is not None or args.record_all:
        parser.error("--record dan --record-all hanya untuk sesi live, tidak bisa bersama --replay.")
    result = replay(
        args.replay,
        quaternion_order=args.quaternion_order,
        flexion_gain=args.flexion_gain,
        filter_cutoff_hz=args.filter_cutoff_hz,
        mode=args.replay_calib,
        calib_from=args.calib_from,
        show_viewer=not args.no_viewer,
        realtime=not args.no_realtime,
        frames=args.replay_frames,
    )
    print(
        f"[OK] Replay selesai: {result.frames_mapped} frame dipetakan "
        f"dari {result.frames_total} frame di rekaman."
    )
    for warning in result.warnings:
        print(f"[WARNING] {warning}")


if __name__ == "__main__":
    main()
