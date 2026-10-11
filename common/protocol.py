import struct
import time

CLASSES = ["ROCK", "CRATER", "OBSTACLE", "OTHER"]
FMT = ">HBhhBH"                      # seq, type, x, y, confiance, ts (big-endian)
BODY_SIZE = struct.calcsize(FMT)     # 10 octets
PACKET_SIZE = BODY_SIZE + 1          # + 1 octet CRC = 11


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode(seq, type_name, x_cm, y_cm, confidence):
    type_id = CLASSES.index(type_name) if type_name in CLASSES else CLASSES.index("OTHER")
    x = max(-32768, min(32767, int(x_cm)))
    y = max(-32768, min(32767, int(y_cm)))
    conf = max(0, min(255, round(confidence * 255)))
    ts = int(time.time() * 1000) & 0xFFFF
    body = struct.pack(FMT, seq & 0xFFFF, type_id, x, y, conf, ts)
    return body + bytes([crc8(body)])


def decode(packet: bytes):
    """Retourne un dict, ou None si le paquet est invalide/corrompu."""
    if len(packet) != PACKET_SIZE:
        return None
    body, crc = packet[:-1], packet[-1]
    if crc8(body) != crc:
        return None
    seq, type_id, x, y, conf, ts = struct.unpack(FMT, body)
    return {
        "seq": seq,
        "type": CLASSES[type_id] if type_id < len(CLASSES) else "OTHER",
        "x_cm": x,
        "y_cm": y,
        "confidence": round(conf * 100 / 255),
        "ts": ts,
    }
