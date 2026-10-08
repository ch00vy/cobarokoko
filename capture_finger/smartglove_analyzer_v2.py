import argparse
import csv
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

PACKET_SIZE = 624
GROUP_A_START = 0x48
GROUP_A_END = 0x108
RECORD_SIZE = 32
FLOAT_COUNT = 7


def load_packets(folder):
    folder = Path(folder)
    files = sorted(folder.glob("packet_*.bin"))
    packets = []
    for path in files:
        data = path.read_bytes()
        if len(data) == PACKET_SIZE:
            packets.append((path.name, data))
    return packets


def quaternion_norm(values):
    q = values[3:7]
    return math.sqrt(sum(v * v for v in q))


def looks_like_quaternion(values, tolerance=0.01):
    norm = quaternion_norm(values)
    return abs(norm - 1.0) <= tolerance


def scan_records(data):
    """
    Scan every 4-byte aligned offset for:
        4-byte identifier + 7 float32
    where the last four floats form a near-unit quaternion.
    """
    candidates = []

    for offset in range(0, len(data) - 32 + 1, 4):
        try:
            identifier = struct.unpack_from("<I", data, offset)[0]
            values = struct.unpack_from("<7f", data, offset + 4)
        except struct.error:
            continue

        if not all(math.isfinite(v) for v in values):
            continue

        if looks_like_quaternion(values):
            x, y, z, q1, q2, q3, q4 = values
            candidates.append({
                "offset": offset,
                "id": identifier,
                "x": x,
                "y": y,
                "z": z,
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "q4": q4,
                "qnorm": quaternion_norm(values),
            })

    return candidates


def analyze_capture(folder, output_dir):
    packets = load_packets(folder)

    if not packets:
        print(f"[ERROR] Tidak ada packet valid di: {folder}")
        return

    print("=" * 80)
    print(" SMARTGLOVE ROBUST ANALYZER V2")
    print("=" * 80)
    print(f"Folder           : {folder}")
    print(f"Valid packets    : {len(packets)}")
    print(f"Packet size      : {len(packets[0][1])} bytes")
    print()

    # Collect candidates by (offset, id)
    observations = defaultdict(list)
    offset_counter = Counter()
    id_counter = Counter()

    for filename, data in packets:
        candidates = scan_records(data)

        # Avoid duplicate candidates with same offset/id in one packet
        seen = set()
        for c in candidates:
            key = (c["offset"], c["id"])
            if key in seen:
                continue
            seen.add(key)

            observations[key].append(c)
            offset_counter[c["offset"]] += 1
            id_counter[c["id"]] += 1

    # Only keep candidates appearing in at least 80% of packets.
    min_count = max(1, int(len(packets) * 0.80))
    stable = {
        key: values
        for key, values in observations.items()
        if len(values) >= min_count
    }

    print(f"Minimum stable occurrence: {min_count}/{len(packets)} packets")
    print(f"Stable ID/offset pairs   : {len(stable)}")
    print()

    print("-" * 80)
    print("STABLE CANDIDATES")
    print("-" * 80)
    print(
        f"{'Offset':>8} {'ID':>12} {'Count':>7} "
        f"{'X std':>10} {'Y std':>10} {'Z std':>10} "
        f"{'Q1 std':>10} {'Q2 std':>10} {'Q3 std':>10} {'Q4 std':>10}"
    )

    summary_rows = []

    for (offset, identifier), values in sorted(
        stable.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        def std(key):
            xs = [v[key] for v in values]
            if len(xs) < 2:
                return 0.0
            mean = sum(xs) / len(xs)
            return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))

        row = {
            "offset": offset,
            "id": f"0x{identifier:08X}",
            "id_decimal": identifier,
            "count": len(values),
            "x_std": std("x"),
            "y_std": std("y"),
            "z_std": std("z"),
            "q1_std": std("q1"),
            "q2_std": std("q2"),
            "q3_std": std("q3"),
            "q4_std": std("q4"),
            "qnorm_max_error": max(abs(v["qnorm"] - 1.0) for v in values),
        }

        summary_rows.append(row)

        print(
            f"0x{offset:04X} {identifier:#012x} {len(values):7d} "
            f"{row['x_std']:10.5f} {row['y_std']:10.5f} {row['z_std']:10.5f} "
            f"{row['q1_std']:10.5f} {row['q2_std']:10.5f} "
            f"{row['q3_std']:10.5f} {row['q4_std']:10.5f}"
        )

    print()

    # Save summary
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "robust_quaternion_candidates.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys() if summary_rows else [
            "offset", "id", "id_decimal", "count",
            "x_std", "y_std", "z_std",
            "q1_std", "q2_std", "q3_std", "q4_std",
            "qnorm_max_error"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save every detected stable record from every packet
    raw_csv = output_dir / "robust_records_all_packets.csv"
    with raw_csv.open("w", newline="") as f:
        fields = [
            "packet", "offset", "id",
            "x", "y", "z", "q1", "q2", "q3", "q4", "qnorm"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for filename, data in packets:
            for c in scan_records(data):
                key = (c["offset"], c["id"])
                if key in stable:
                    writer.writerow({
                        "packet": filename,
                        "offset": c["offset"],
                        "id": f"0x{c['id']:08X}",
                        "x": c["x"],
                        "y": c["y"],
                        "z": c["z"],
                        "q1": c["q1"],
                        "q2": c["q2"],
                        "q3": c["q3"],
                        "q4": c["q4"],
                        "qnorm": c["qnorm"],
                    })

    print(f"[OK] Summary : {csv_path}")
    print(f"[OK] Raw data: {raw_csv}")


def compare_captures(folder_a, folder_b, output_dir):
    packets_a = load_packets(folder_a)
    packets_b = load_packets(folder_b)

    def stable_map(packets):
        obs = defaultdict(list)

        for _, data in packets:
            seen = set()
            for c in scan_records(data):
                key = (c["offset"], c["id"])
                if key not in seen:
                    obs[key].append(c)
                    seen.add(key)

        min_count = max(1, int(len(packets) * 0.80))
        return {
            key: values
            for key, values in obs.items()
            if len(values) >= min_count
        }

    A = stable_map(packets_a)
    B = stable_map(packets_b)

    common = sorted(set(A) & set(B))

    print("=" * 80)
    print(" CAPTURE COMPARISON")
    print("=" * 80)
    print(f"A: {folder_a} -> {len(packets_a)} packets")
    print(f"B: {folder_b} -> {len(packets_b)} packets")
    print(f"Common stable records: {len(common)}")
    print()

    print(
        f"{'Offset':>8} {'ID':>12} "
        f"{'A motion':>12} {'B motion':>12} {'B/A ratio':>12}"
    )

    rows = []

    for key in common:
        offset, identifier = key

        def motion(values):
            # Quaternion component standard deviation sum.
            qstd = []
            for i in range(3, 7):
                xs = [v[f"q{i-3+1}"] for v in values]
                mean = sum(xs) / len(xs)
                qstd.append(math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)))
            return sum(qstd)

        ma = motion(A[key])
        mb = motion(B[key])
        ratio = mb / ma if ma > 1e-12 else float("inf")

        print(
            f"0x{offset:04X} {identifier:#012x} "
            f"{ma:12.6f} {mb:12.6f} {ratio:12.3f}"
        )

        rows.append({
            "offset": offset,
            "id": f"0x{identifier:08X}",
            "A_quaternion_motion": ma,
            "B_quaternion_motion": mb,
            "B_A_ratio": ratio,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "capture_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
            "offset", "id", "A_quaternion_motion",
            "B_quaternion_motion", "B_A_ratio"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"[OK] Comparison saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Robust analyzer for Rokoko Smartglove raw UDP packets."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze one capture folder")
    p_analyze.add_argument("folder")
    p_analyze.add_argument(
        "--output",
        default="smartglove_analysis",
        help="Output folder"
    )

    p_compare = sub.add_parser(
        "compare",
        help="Compare two capture folders"
    )
    p_compare.add_argument("folder_a")
    p_compare.add_argument("folder_b")
    p_compare.add_argument(
        "--output",
        default="smartglove_analysis",
        help="Output folder"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_capture(args.folder, args.output)

    elif args.command == "compare":
        compare_captures(
            args.folder_a,
            args.folder_b,
            args.output
        )


if __name__ == "__main__":
    main()