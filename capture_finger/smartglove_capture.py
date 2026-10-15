import socket
import struct
from pathlib import Path
import time

HOST = "192.168.3.101"
PORT = 14041

N_PACKETS = 200

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print("=" * 60)
print(" ROKOKO SMARTGLOVE RAW UDP CAPTURE")
print("=" * 60)
print(f"[OK] Listening UDP {HOST}:{PORT}")
print(f"[OK] Target packets : {N_PACKETS}")
print()
print("Gerakan bebas dulu.")
print("Tekan Ctrl+C untuk berhenti.")
print()

packets = []
timestamps = []

try:
    while len(packets) < N_PACKETS:
        data, addr = sock.recvfrom(65535)

        now = time.time()

        packets.append(data)
        timestamps.append(now)

        print(
            f"[{len(packets):03d}/{N_PACKETS}] "
            f"{addr} | {len(data)} bytes"
        )

except KeyboardInterrupt:
    print("\n[STOP] Capture dihentikan.")

finally:
    sock.close()


if not packets:
    print("[ERROR] Tidak ada packet.")
    raise SystemExit


# ============================================================
# SAVE RAW PACKETS
# ============================================================

capture_name = input("Nama capture (contoh: index): ").strip()

out_dir = Path(f"smartglove_capture_{capture_name}")
out_dir.mkdir(exist_ok=True)

for i, data in enumerate(packets):
    filename = out_dir / f"packet_{i:04d}.bin"
    filename.write_bytes(data)


# ============================================================
# SAVE FIRST PACKET AS HEX
# ============================================================

first = packets[0]

with open(out_dir / "packet_0000_hex.txt", "w") as f:

    for i in range(0, len(first), 16):

        chunk = first[i:i + 16]

        hex_part = " ".join(
            f"{b:02X}" for b in chunk
        )

        ascii_part = "".join(
            chr(b) if 32 <= b <= 126 else "."
            for b in chunk
        )

        f.write(
            f"{i:04X}  "
            f"{hex_part:<47}  "
            f"{ascii_part}\n"
        )


# ============================================================
# BASIC FLOAT ANALYSIS
# ============================================================

print()
print("=" * 60)
print(" BASIC ANALYSIS")
print("=" * 60)

print(f"Packets captured : {len(packets)}")
print(f"Packet size      : {len(first)} bytes")

if len(packets) >= 2:

    intervals = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
    ]

    avg_interval = sum(intervals) / len(intervals)

    if avg_interval > 0:
        fps = 1 / avg_interval

        print(f"Average interval : {avg_interval * 1000:.2f} ms")
        print(f"Approx. rate     : {fps:.2f} Hz")


# ============================================================
# COMPARE PACKET 1 VS PACKET 200
# ============================================================

if len(packets) >= 2:

    a = packets[0]
    b = packets[-1]

    print()
    print("Changed byte offsets:")
    print()

    changed = []

    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            changed.append(i)

    print(f"Total changed bytes: {len(changed)}")

    print()
    print("First 100 changed offsets:")

    print(changed[:100])


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print(" CAPTURE SELESAI")
print("=" * 60)
print()
print(f"Data tersimpan di:")
print(f"  {out_dir.resolve()}")
print()
print("Kirim folder/file hasil capture tersebut untuk analisis.")