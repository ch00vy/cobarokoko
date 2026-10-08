import socket

HOST = "0.0.0.0"
PORT = 14041

print("=" * 50)
print(" ROKOKO SMARTGLOVE UDP NETWORK TEST")
print("=" * 50)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"[OK] UDP server aktif di {HOST}:{PORT}")
print("[WAIT] Menunggu data Smartglove...\n")

try:
    while True:
        data, addr = sock.recvfrom(65535)

        print(f"[+] DATA dari {addr}")
        print(f"    Length : {len(data)} bytes")
        print(f"    HEX    : {data[:64].hex(' ')}")
        print()

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    sock.close()