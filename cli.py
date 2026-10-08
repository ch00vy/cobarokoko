import argparse
import math

from smartglove.config import FILTER_CUTOFF_HZ
from smartglove.runtime import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map Rokoko Smartglove UDP quaternions to MyoSuite hand qpos."
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Urutan komponen quaternion pada paket Rokoko (default: xyzw).",
    )
    parser.add_argument(
        "--flexion-gain",
        type=float,
        default=1.0,
        help="Pengali fleksi jari (default: 1.0).",
    )
    parser.add_argument(
        "--filter-cutoff-hz",
        type=float,
        default=FILTER_CUTOFF_HZ,
        help="Cutoff filter quaternion dalam Hz; lebih rendah lebih halus (default: 12).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.flexion_gain) or args.flexion_gain <= 0.0:
        parser.error("--flexion-gain harus finite dan lebih besar dari nol.")
    if not math.isfinite(args.filter_cutoff_hz) or args.filter_cutoff_hz <= 0.0:
        parser.error("--filter-cutoff-hz harus finite dan lebih besar dari nol.")
    run(args.quaternion_order, args.flexion_gain, args.filter_cutoff_hz)


if __name__ == "__main__":
    main()
