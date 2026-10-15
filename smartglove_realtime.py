import math
import socket
import struct
import time
from collections import deque
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "192.168.3.101"
PORT = 14041

PACKET_SIZE = 624
RATE_WINDOW_INTERVALS = 30

# Record layout:
# ID      : 4 bytes
# Position: 3 x float32
# Quaternion: 4 x float32
#
# Total = 4 + 12 + 16 = 32 bytes

FINGERS = {
    "THUMB":  0x28,
    "INDEX":  0x48,
    "MIDDLE": 0x68,
    "RING":   0x88,
    "PINKY":  0xA8,
}

WRIST_SENSORS = {
    "FOREARM": 0xC8,
    "HAND": 0xE8,
}

FINGER_IDS = {
    "THUMB": 0x00000000,
    "INDEX": 0x00000100,
    "MIDDLE": 0x00000200,
    "RING": 0x00000300,
    "PINKY": 0x00000400,
}

FINGER_ID_ALIASES = {
    0x00010000: "THUMB",
    0x02000000: "THUMB",
    0x03000000: "THUMB",
    0x03010000: "THUMB",
    0x00010200: "MIDDLE",
    0x01000300: "RING",
}

# Byte ke-1 (bit 8-15) dari ID record adalah indeks sensor yang stabil:
# 0=THUMB 1=INDEX 2=MIDDLE 3=RING 4=PINKY 5=FOREARM 6=HAND. Byte lainnya
# berubah-ubah (mis. 0x03010000, 0x01010500) sehingga tabel alias ID penuh
# tidak andal. Terverifikasi pada 2800 paket capture di repo ini.
SENSOR_INDEX = {
    "THUMB": 0,
    "INDEX": 1,
    "MIDDLE": 2,
    "RING": 3,
    "PINKY": 4,
    "FOREARM": 5,
    "HAND": 6,
}


def sensor_index_from_id(record_id: int) -> int:
    return (record_id >> 8) & 0xFF


# ============================================================
# PARSER
# ============================================================

def parse_finger(data, offset):
    """
    Parse one 32-byte finger record.

    Structure:
        +0x00 : uint32 ID
        +0x04 : float32 X
        +0x08 : float32 Y
        +0x0C : float32 Z
        +0x10 : float32 Q1
        +0x14 : float32 Q2
        +0x18 : float32 Q3
        +0x1C : float32 Q4
    """

    if len(data) < offset + 32:
        return None

    record = data[offset:offset + 32]

    finger_id = struct.unpack_from("<I", record, 0)[0]

    x, y, z = struct.unpack_from("<3f", record, 4)

    q1, q2, q3, q4 = struct.unpack_from("<4f", record, 16)
    quaternion = (q1, q2, q3, q4)
    norm = math.hypot(*quaternion)

    if not math.isfinite(norm) or norm == 0:
        return None

    quaternion = tuple(component / norm for component in quaternion)

    return {
        "id": finger_id,
        "position": (x, y, z),
        "quaternion": quaternion,
    }


def parse_packet(data):
    """
    Parse the 624-byte Smartglove UDP packet.
    """

    parsed, _ = parse_packet_with_error(data)
    return parsed


def parse_packet_with_error(data):
    """Parse a packet and return a reason when its contents are invalid.

    Setiap slot (offset) harus membawa indeks sensor yang sesuai dengan
    posisinya, termasuk FOREARM dan HAND yang sebelumnya tidak divalidasi.
    """
    if len(data) != PACKET_SIZE:
        return None, f"ukuran {len(data)} bytes (diharapkan {PACKET_SIZE})"

    result = {}
    slots = {**FINGERS, **WRIST_SENSORS}

    for sensor, offset in slots.items():
        parsed = parse_finger(data, offset)

        if parsed is None:
            record_id = struct.unpack_from("<I", data, offset)[0]
            return None, (
                f"quaternion tidak valid pada sensor {sensor} "
                f"offset 0x{offset:02X} (ID 0x{record_id:08X})"
            )

        index = sensor_index_from_id(parsed["id"])
        if index != SENSOR_INDEX[sensor]:
            return None, (
                f"ID 0x{parsed['id']:08X} pada offset 0x{offset:02X} membawa "
                f"indeks sensor {index}, diharapkan {SENSOR_INDEX[sensor]} "
                f"({sensor})"
            )

        result[sensor] = parsed

    return result, None


def calculate_packet_rate(intervals):
    elapsed = sum(intervals)
    return len(intervals) / elapsed if elapsed > 0 else 0.0


# ============================================================
# DISPLAY
# ============================================================

def print_finger_data(frame_number, data, packet_size, timestamp):
    print("\033[2J\033[H", end="")

    print("=" * 75)
    print(" ROKOKO SMARTGLOVE REALTIME PARSER")
    print("=" * 75)

    print(f"Packet       : {frame_number}")
    print(f"Packet size  : {packet_size} bytes")
    print(f"Source       : {data}")
    print(f"Timestamp    : {timestamp}")

    print("-" * 75)
    print(
        f"{'FINGER':<10}"
        f"{'ID':<12}"
        f"{'Q1':>10}"
        f"{'Q2':>10}"
        f"{'Q3':>10}"
        f"{'Q4':>10}"
    )
    print("-" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        1024 * 1024
    )

    sock.bind((HOST, PORT))

    print("=" * 75)
    print(" ROKOKO SMARTGLOVE REALTIME PARSER")
    print("=" * 75)
    print()
    print(f"[OK] UDP listener : {HOST}:{PORT}")
    print("[WAIT] Menunggu Smartglove...")
    print()

    frame_number = 0
    packet_intervals = deque(maxlen=RATE_WINDOW_INTERVALS)
    last_packet_time = None

    try:

        while True:

            packet, addr = sock.recvfrom(65535)

            now = time.perf_counter()
            timestamp = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )

            parsed = parse_packet(packet)

            if parsed is None:
                print(
                    f"[WARNING] Packet tidak valid "
                    f"(ukuran {len(packet)} bytes atau quaternion)"
                )
                continue

            frame_number += 1

            # ------------------------------------------------
            # Moving-average packet rate
            # ------------------------------------------------

            if last_packet_time is not None:
                interval = now - last_packet_time
                if interval > 0:
                    packet_intervals.append(interval)
            last_packet_time = now

            rate = calculate_packet_rate(packet_intervals)

            # ------------------------------------------------
            # Clear terminal
            # ------------------------------------------------

            print("\033[2J\033[H", end="")

            print("=" * 75)
            print(" ROKOKO SMARTGLOVE REALTIME PARSER")
            print("=" * 75)

            print(f"Frame        : {frame_number}")
            print(f"Packet size  : {len(packet)} bytes")
            print(f"Source       : {addr[0]}:{addr[1]}")
            print(f"Timestamp    : {timestamp}")
            print(
                f"Rate         : {rate:.1f} Hz "
                f"(moving average, {len(packet_intervals)} intervals)"
            )

            print("-" * 75)

            print(
                f"{'FINGER':<10}"
                f"{'ID':<12}"
                f"{'Q1':>10}"
                f"{'Q2':>10}"
                f"{'Q3':>10}"
                f"{'Q4':>10}"
            )

            print("-" * 75)

            # ------------------------------------------------
            # Display quaternion
            # ------------------------------------------------

            for finger, values in parsed.items():

                finger_id = values["id"]
                q1, q2, q3, q4 = values["quaternion"]

                print(
                    f"{finger:<10}"
                    f"0x{finger_id:08X} "
                    f"{q1:>10.5f}"
                    f"{q2:>10.5f}"
                    f"{q3:>10.5f}"
                    f"{q4:>10.5f}"
                )

            print("-" * 75)
            print("Tekan Ctrl+C untuk berhenti.")

    except KeyboardInterrupt:

        print()
        print("[STOP] Parser dihentikan.")

    finally:

        sock.close()


if __name__ == "__main__":
    main()