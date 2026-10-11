import argparse
import socket
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.camera import IMAGE_EXT, intrinsics, pixel_to_ground, shrink
from common.protocol import encode
from rover.perception import detect_classic, detect_yolo

HOST = "127.0.0.1"
PORT = 5005
MAX_PER_FRAME = 5                 # budget de télémétrie par image


def frames(path):
    """Génère des images en boucle depuis une image fixe ou une vidéo."""
    if path.suffix.lower() in IMAGE_EXT:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Impossible de lire cette image.")
        while True:
            yield image
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Impossible d'ouvrir cette vidéo.")
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # on reboucle
                continue
            yield frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="image ou vidéo locale")
    parser.add_argument("--detector", choices=["classic", "yolo"], default="classic")
    parser.add_argument("--period", type=float, default=1.0,
                        help="secondes entre deux images traitées")
    parser.add_argument("--debug", action="store_true",
                        help="enregistre debug.jpg avec les détections dessinées")
    args = parser.parse_args()

    path = Path(args.source)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    model = None
    if args.detector == "yolo":
        from ultralytics import YOLO
        print("Loading YOLO model...")
        model = YOLO("yolo11n.pt")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    print(f"Local perception started ({args.detector}) on {path.name}")

    try:
        for frame in frames(path):
            frame = shrink(frame)
            height, width = frame.shape[:2]
            cx, cy, focal = intrinsics(width, height)

            t0 = time.perf_counter()
            if args.detector == "yolo":
                dets = detect_yolo(frame, model)
            else:
                dets = detect_classic(frame)
            dt = time.perf_counter() - t0

            # les plus gros obstacles d'abord (YOLO n'a pas de "area" : on retombe sur la confiance)
            dets.sort(key=lambda d: (d.get("area", 0), d["conf"]), reverse=True)

            if args.debug:
                vis = frame.copy()
                for d in dets:
                    color = (0, 255, 0) if d["kind"] == "ROCK" else (0, 0, 255)
                    cv2.circle(vis, (int(d["u"]), int(d["v"])), 6, color, -1)
                cv2.imwrite("debug.jpg", vis)

            sent = 0
            for d in dets:
                if sent >= MAX_PER_FRAME:
                    break
                ground = pixel_to_ground(d["u"], d["v"], cx, cy, focal)
                if ground is None:
                    continue
                seq = (seq + 1) & 0xFFFF
                packet = encode(seq, d["kind"], ground[0], ground[1], d["conf"])
                sock.sendto(packet, (HOST, PORT))
                sent += 1
                print(f"SENT seq={seq} {d['kind']} x={ground[0]} y={ground[1]} "
                      f"conf={d['conf']:.2f} ({len(packet)} octets)")

            print(f"[perception] {len(dets)} détections, {dt*1000:.1f} ms "
                  f"-> {1/dt:.0f} FPS | {sent} envoyées")
            time.sleep(args.period)

    except KeyboardInterrupt:
        print("\nPerception stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
