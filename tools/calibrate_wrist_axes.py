"""Estimasi sumbu anatomis wrist di frame sensor dari rekaman sapuan gerak.

Tiga rekaman, masing-masing HANYA satu gerakan, lengan bawah bertumpu di meja,
mulai dari pose netral (tangan lurus dengan lengan bawah):

    python tools/calibrate_wrist_axes.py \
        --flexion DIR_FLEKSI_EKSTENSI --deviation DIR_DEVIASI \
        --pronation DIR_PUTAR_LENGAN_BAWAH --order xyzw

Tiap rekaman = folder berisi packet_*.bin (paket UDP 624 byte).

PENTING - nama gerakan bergantung pada pose netral:
  * telapak MENGHADAP BAWAH : naik-turun = fleksi/ekstensi, kiri-kanan = deviasi.
  * telapak MENYAMPING (jabat tangan, ibu jari di atas):
    kiri-kanan = fleksi/ekstensi, naik-turun = deviasi radial/ulnar.
Berikan folder ke --flexion / --deviation sesuai gerak anatominya, bukan
sesuai nama foldernya.

--pronation: lengan bawah diputar (telapak menyamping -> bawah -> menyamping
-> atas -> menyamping). Sumbu twist diukur dari sensor FOREARM saja dan
disimpan sebagai "forearm_twist"; tanpa itu pro_sup tidak dapat dibaca.

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

from smartglove.quaternion import (  # noqa: E402
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotvec,
    sensor_quaternion_to_wxyz,
)
from smartglove.wrist_alignment import (  # noqa: E402
    DEFAULT_AXES_FILE,
    axes_quality,
    forearm_twist_rotvecs,
    principal_axis,
    save_axes,
    sweep_rotvecs,
)
from smartglove_realtime import parse_packet_with_error  # noqa: E402

MIN_EXPLAINED = 0.90
MIN_TWIST_EXCURSION_DEG = 40.0
MAX_COMMON_ROLL_DEG = 25.0


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


def world_axis(quaternions):
    """Sumbu dominan rotasi sensor di frame dunia (q(t) * conj(q0))."""
    reference = quaternions[0]
    rotvecs = np.array(
        [
            quaternion_to_rotvec(quaternion_multiply(q, quaternion_conjugate(reference)))
            for q in quaternions
        ]
    )
    _, _, vt = np.linalg.svd(rotvecs - rotvecs.mean(axis=0), full_matrices=False)
    return vt[0]


def flip(info: dict) -> None:
    info["axis"] = -info["axis"]
    info["min_deg"], info["max_deg"] = -info["max_deg"], -info["min_deg"]


def describe(name: str, info: dict) -> None:
    a = info["axis"]
    print(
        f"{name:<14} sumbu=[{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}] "
        f"kemurnian={info['explained'] * 100:5.1f}% "
        f"simpangan={info['min_deg']:+6.1f}..{info['max_deg']:+6.1f} deg"
    )
    if info["explained"] < MIN_EXPLAINED:
        print(f"  [PERINGATAN] {name}: gerak tidak satu-sumbu (<90%); ulangi rekaman.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--flexion", required=True, help="Folder sapuan fleksi/ekstensi.")
    parser.add_argument("--deviation", required=True, help="Folder sapuan deviasi radial/ulnar.")
    parser.add_argument("--pronation", help="Folder sapuan putar lengan bawah (pronasi/supinasi).")
    parser.add_argument("--order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--out", default=str(DEFAULT_AXES_FILE))
    parser.add_argument("--flip-flexion", action="store_true")
    parser.add_argument("--flip-deviation", action="store_true")
    parser.add_argument("--flip-pronation", action="store_true")
    args = parser.parse_args()

    axes = {}
    for name, folder, flipped in (
        ("flexion", args.flexion, args.flip_flexion),
        ("deviation", args.deviation, args.flip_deviation),
    ):
        forearm, hand = load_sweep(folder, args.order)
        info = principal_axis(sweep_rotvecs(forearm, hand))
        if flipped:
            flip(info)
        axes[name] = info
        describe(name, info)

    angle = axes_quality(
        {"flexion": axes["flexion"]["axis"], "deviation": axes["deviation"]["axis"]}
    )["flexion-deviation"]
    print(
        f"Sudut fleksi-deviasi terukur: {angle:.1f} deg (ideal 90; selisih "
        f"{abs(angle - 90.0):.1f}). Sumbu fleksi dipetakan eksak, deviasi "
        "dikoreksi ke tegak lurus."
    )
    if abs(angle - 90.0) > 20.0:
        print("  [PERINGATAN] selisih >20 deg: rekaman deviasi kemungkinan bercampur; "
              "ulangi dengan lengan bawah diam.")

    if args.pronation:
        forearm, hand = load_sweep(args.pronation, args.order)
        info = principal_axis(forearm_twist_rotvecs(forearm))
        if args.flip_pronation:
            flip(info)
        axes["forearm_twist"] = info
        describe("forearm_twist", info)
        excursion = info["max_deg"] - info["min_deg"]
        if excursion < MIN_TWIST_EXCURSION_DEG:
            print(f"  [PERINGATAN] total simpangan putar hanya {excursion:.0f} deg "
                  f"(<{MIN_TWIST_EXCURSION_DEG:.0f}); putar lebih jauh agar sumbu stabil.")
        a_f, a_h = world_axis(forearm), world_axis(hand)
        common = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a_f, a_h)))))))
        print(f"Sumbu putar FOREARM vs HAND di frame dunia: {common:.1f} deg "
              f"(baik bila < {MAX_COMMON_ROLL_DEG:.0f}; kedua sensor harus berputar bersama).")
        if common > MAX_COMMON_ROLL_DEG:
            print("  [PERINGATAN] sensor tidak berputar pada sumbu yang sama; "
                  "kemungkinan bukan putaran lengan bawah murni.")
    else:
        print("[INFO] --pronation tidak diberikan: pro_sup belum bisa dibaca "
              "(mapper memakai rotasi relatif saja).")

    save_axes(Path(args.out), args.order, axes)
    print(f"[OK] Disimpan ke {args.out} (urutan quaternion: {args.order})")


if __name__ == "__main__":
    main()
