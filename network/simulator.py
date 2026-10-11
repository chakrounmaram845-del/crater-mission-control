import random
import socket

HOST = "127.0.0.1"
INPUT_PORT = 5005
OUTPUT_PORT = 5006
LOSS_RATE = 0.20

receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receiver.bind((HOST, INPUT_PORT))
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Network simulator started!")
print(f"Packet loss: {LOSS_RATE * 100:.0f}%")

received = dropped = 0

try:
    while True:
        packet, _ = receiver.recvfrom(4096)
        received += 1

        if random.random() < LOSS_RATE:
            dropped += 1
            print(f"DROPPED   ({dropped}/{received})")
            continue

        sender.sendto(packet, (HOST, OUTPUT_PORT))
        print(f"FORWARDED {len(packet)} octets ({dropped}/{received} perdus)")

except KeyboardInterrupt:
    print("\nNetwork simulator stopped.")
finally:
    receiver.close()
    sender.close()
