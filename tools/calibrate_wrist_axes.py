"""Estimasi sumbu anatomis wrist di frame sensor dari rekaman sapuan gerak.

Contoh (memakai capture yang sudah ada di repo):

    python tools/calibrate_wrist_axes.py \
        --flexion smartglove_capture_atas_bawah \
        --deviation smartglove_capture_kanan_kiri \
        --pronation smartglove_capture_putar

Tiap rekaman = folder berisi packet_*.bin (paket UDP 624 byte). Gerakkan
HANYA satu sumbu per rekaman, lengan bawah diam, mulai dari pose netral.
Hasil disimpan ke smartglove/wrist_axes.json per urutan quaternion.

Tanda sumbu: sisi simpangan terbesar dianggap positif. Itu sesuai arah sendi
model untuk deviasi (ulnar positif, ROM model -10..+25 derajat), tetapi untuk
fleksi dan pronasi tanda tidak dapat ditentukan dari data; verifikasi sekali
secara visual di viewer, lalu balik dengan --flip-flexion / --flip-pronation.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartglove.quaternion import sensor_quaternion_to_wxyz  # noqa: E402
from smartglove.wrist_alignment import (  # noqa: E402
    DEFAULT_AXES_FILE,
    axes_quality,
    predicted_pro_sup_axis,
    principal_axis,
    save_axes,
    sweep_rotvecs,
)
from smartglove_realtime import parse_packet_with_error  # noqa: E402


def load_sweep(folder: str, order: str):
    forearm, hand = [], []
    for path in sorted(glob.glob(str(Path(folder) / "packet_*.bin"))):
        parsed, _ = parse_packet_with_error(Path(path).read_bytes())
        if parsed is None:
            continue
        forearm.append(sensor_quaternion_to_wxyz(parsed["FOREARM"]["quaternion"], order))
        hand.append(sensor_quaternion_to_wxyz(parsed["HAND"]["quaternion"], order))
    if len(hand) < 20:
        raise SystemExit(f"{folder}: hanya {len(hand)} paket valid (minimal 20).")
    return forearm, hand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--flexion", required=True, help="Folder sapuan fleksi/ekstensi.")
    parser.add_argument("--deviation", required=True, help="Folder sapuan deviasi radial/ulnar.")
    parser.add_argument(
        "--pronation",
        help="Folder sapuan pronasi/supinasi (opsional; hanya pengecekan silang).",
    )
    parser.add_argument("--order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--out", default=str(DEFAULT_AXES_FILE))
    parser.add_argument("--flip-flexion", action="store_true")
    parser.add_argument("--flip-deviation", action="store_true")
    parser.add_argument("--flip-pronation", action="store_true")
    args = parser.parse_args()

    sources = {"flexion": args.flexion, "deviation": args.deviation}
    if args.pronation:
        sources["pro_sup"] = args.pronation
    flips = {
        "flexion": args.flip_flexion,
        "deviation": args.flip_deviation,
        "pro_sup": args.flip_pronation,
    }

    axes = {}
    for name, folder in sources.items():
        forearm, hand = load_sweep(folder, args.order)
        info = principal_axis(sweep_rotvecs(forearm, hand))
        if flips[name]:
            info["axis"] = -info["axis"]
            info["min_deg"], info["max_deg"] = -info["max_deg"], -info["min_deg"]
        axes[name] = info
        a = info["axis"]
        print(
            f"{name:<10} sumbu=[{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}] "
            f"kemurnian={info['explained'] * 100:5.1f}% "
            f"simpangan={info['min_deg']:+6.1f}..{info['max_deg']:+6.1f} deg"
        )
        if info["explained"] < 0.9:
            print(f"  [PERINGATAN] {name}: gerak tidak satu-sumbu (<90%); ulangi rekaman.")

    flexion_deviation = axes_quality(
        {"flexion": axes["flexion"]["axis"], "deviation": axes["deviation"]["axis"]}
    )["flexion-deviation"]
    error = abs(flexion_deviation - 90.0)
    print(
        f"Sudut fleksi-deviasi terukur: {flexion_deviation:.1f} deg (ideal 90; "
        f"selisih {error:.1f}). Sumbu fleksi dipetakan eksak, deviasi dikoreksi "
        "ke tegak lurus."
    )
    if error > 20.0:
        print("  [PERINGATAN] selisih >20 deg: rekaman deviasi kemungkinan "
              "bercampur fleksi/pronasi; ulangi dengan lengan bawah diam.")

    if "pro_sup" in axes:
        predicted = predicted_pro_sup_axis(
            {"flexion": axes["flexion"]["axis"], "deviation": axes["deviation"]["axis"]}
        )
        measured = axes["pro_sup"]["axis"]
        cosine = abs(float(np.dot(predicted, measured)))
        print(
            "Pengecekan silang pronasi: sudut sumbu terukur vs prediksi = "
            f"{np.degrees(np.arccos(np.clip(cosine, 0, 1))):.1f} deg (baik bila < 20)"
        )

    save_axes(Path(args.out), args.order, axes)
    print(f"[OK] Disimpan ke {args.out} (urutan quaternion: {args.order})")


if __name__ == "__main__":
    main()
