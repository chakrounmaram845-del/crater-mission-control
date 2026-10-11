"""Paramètres de la caméra et géométrie, partagés par le rover et Mission Control.

Une seule source de vérité : si on change une hypothèse ici, le rover et le
dashboard restent cohérents (les zones dessinées retombent au bon endroit).
"""
import math

import cv2

CAM_H = 0.40                     # hauteur de la caméra au-dessus du sol (m)
PITCH = math.radians(25)         # inclinaison de la caméra vers le sol
HFOV = math.radians(60)          # champ de vision horizontal
TARGET_WIDTH = 640               # largeur de travail (pixels)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def shrink(frame):
    """Réduit l'image pour tenir la contrainte de calcul embarqué."""
    h, w = frame.shape[:2]
    if w <= TARGET_WIDTH:
        return frame
    scale = TARGET_WIDTH / w
    return cv2.resize(frame, (TARGET_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def intrinsics(width, height):
    """Centre optique (cx, cy) et focale en pixels."""
    return width / 2, height / 2, (width / 2) / math.tan(HFOV / 2)


def pixel_to_ground(u, v, cx, cy, focal):
    """Pixel (u, v) -> (x, y) en cm sur le sol, repère centré sous le rover.
    x = droite, y = devant. None si le point est au-dessus de l'horizon."""
    xc = (u - cx) / focal
    yc = (v - cy) / focal
    denom = yc * math.cos(PITCH) + math.sin(PITCH)
    if denom <= 0:
        return None
    t = CAM_H / denom
    x = t * xc
    y = t * (math.cos(PITCH) - yc * math.sin(PITCH))
    return round(x * 100), round(y * 100)


def ground_to_pixel(x_cm, y_cm, cx, cy, focal):
    """Opération inverse : point du sol (x, y) en cm -> pixel (u, v).
    None si le point est derrière la caméra."""
    x, y = x_cm / 100, y_cm / 100
    z = y * math.cos(PITCH) + CAM_H * math.sin(PITCH)        # profondeur le long de l'axe optique
    if z <= 1e-6:
        return None
    down = -y * math.sin(PITCH) + CAM_H * math.cos(PITCH)    # composante vers le bas de l'image
    return cx + focal * x / z, cy + focal * down / z
