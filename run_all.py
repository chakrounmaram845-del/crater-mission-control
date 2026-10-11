"""Lance tout le pipeline C.R.A.T.E.R. en une commande.

    python run_all.py --source data/lune.jpg
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Sous Windows, chaque programme a son propre groupe : Ctrl + C ne touche que run_all,
# qui arrête ensuite proprement les trois programmes.
NEW_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0


def start(script, *args):
    return subprocess.Popen([sys.executable, str(ROOT / script), *args],
                            cwd=ROOT, creationflags=NEW_GROUP)


def main():
    parser = argparse.ArgumentParser(description="Lance le pipeline C.R.A.T.E.R.")
    parser.add_argument("--source", required=True, help="image ou vidéo locale")
    parser.add_argument("--detector", choices=["classic", "yolo"], default="classic")
    parser.add_argument("--period", default="1.0", help="secondes entre deux images")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not Path(args.source).is_file():
        sys.exit(f"Fichier introuvable : {args.source}")

    procs = {}
    try:
        procs["network"] = start("network/simulator.py")
        time.sleep(1)                       # le temps d'ouvrir les ports
        procs["dashboard"] = start("control/dashboard.py", "--image", args.source)
        time.sleep(1)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:8000")
        procs["rover"] = start("rover/rover.py", "--source", args.source,
                               "--detector", args.detector, "--period", args.period)
        print("[run_all] Pipeline lancé. Ctrl + C pour tout arrêter.")

        while True:
            for name, p in procs.items():
                if p.poll() is not None:
                    print(f"[run_all] '{name}' s'est arrêté (code {p.returncode}). "
                          "Ports déjà utilisés ? Essaie : taskkill /IM python.exe /F")
                    return
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run_all] Arrêt demandé.")
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        print("[run_all] Tout est arrêté.")


if __name__ == "__main__":
    main()
