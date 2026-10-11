import cv2
import numpy as np

HORIZON_FRAC = 0.25      # on ignore le haut de l'image (ciel / relief lointain)
BLOCK = 101              # fenêtre de comparaison locale (impaire) : doit dépasser la taille d'un gros rocher
OFFSET = 12              # un pixel est "sombre" s'il est 12 niveaux sous la moyenne locale
MIN_AREA = 150
MAX_AREA = 20000
MIN_CONTRAST = 20
MAX_ROCKS = 10
ENABLE_CRATERS = False   # la détection de cercles donne trop de faux positifs ici


def detect_classic(frame, max_rocks=MAX_ROCKS):
    """Détection locale par vision classique.
    Retourne une liste de dicts : {u, v, kind, conf} (u, v = point de contact au sol).
    conf est un score heuristique (contraste + taille), pas une probabilité."""
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y0 = int(gray_full.shape[0] * HORIZON_FRAC)
    gray = gray_full[y0:]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    results = []

    # --- ROCHERS : zones plus sombres que leur voisinage ---
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, BLOCK, OFFSET)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rocks = []
    pad = 8
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)

        xa, ya = max(x - pad, 0), max(y - pad, 0)
        xb, yb = min(x + w + pad, gray.shape[1]), min(y + h + pad, gray.shape[0])
        patch = gray[ya:yb, xa:xb]
        mask = np.zeros(patch.shape, np.uint8)
        shifted = (c - np.array([xa, ya])).astype(np.int32)
        cv2.drawContours(mask, [shifted], -1, 255, -1)
        ring = cv2.dilate(mask, np.ones((9, 9), np.uint8)) - mask

        contrast = cv2.mean(patch, mask=ring)[0] - cv2.mean(patch, mask=mask)[0]
        if contrast < MIN_CONTRAST:
            continue

        conf = max(0.1, min(0.95, 0.2 + contrast / 150 + min(area, 3000) / 6000))
        rocks.append({"u": x + w / 2, "v": y0 + y + h, "kind": "ROCK",
                      "conf": conf, "area": area})

    # les plus gros obstacles d'abord : ce sont les plus dangereux
    rocks.sort(key=lambda d: d["area"], reverse=True)
    results += rocks[:max_rocks]

    if ENABLE_CRATERS:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
            param1=100, param2=60, minRadius=30, maxRadius=150)
        if circles is not None:
            for cx, cy, r in circles[0][:3]:
                results.append({"u": float(cx), "v": float(y0 + cy),
                                "kind": "CRATER", "conf": 0.5, "area": 0})
    return results


def detect_yolo(frame, model):
    results = []
    out = model.predict(frame, verbose=False)[0]
    for box in out.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[int(box.cls[0].item())].upper()
        kind = "CRATER" if "CRATER" in name else "ROCK" if "ROCK" in name else "OBSTACLE"
        results.append({"u": (x1 + x2) / 2, "v": y2, "kind": kind,
                        "conf": float(box.conf[0].item())})
    return results
