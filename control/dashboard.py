"""Mission Control C.R.A.T.E.R. : réception de la télémétrie (UDP) + dashboard web local.

    python control/dashboard.py --image data/lune.jpg

Seule la télémétrie (11 octets par objet) arrive par la liaison radio émulée.
La photo affichée est une image de RÉFÉRENCE chargée localement ; les zones de
danger sont reconstruites à partir des coordonnées reçues (projection inverse).
"""
import argparse
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.camera import IMAGE_EXT, ground_to_pixel, intrinsics, shrink
from common.protocol import PACKET_SIZE, decode

HOST = "127.0.0.1"
UDP_PORT = 5006
HTTP_PORT = 8000

# --- Règles de danger (à justifier au jury) ---
CORRIDOR_HALF_WIDTH_CM = 40   # demi-largeur du couloir de passage du rover
CORRIDOR_LENGTH_CM = 150      # distance surveillée devant le rover
KEEP_OUT_CM = 15              # rayon de la zone d'exclusion autour d'un obstacle
CELL_CM = 10                  # deux détections dans la même case de 10 cm = même objet
STALE_S = 10                  # au-delà, la donnée est considérée comme périmée
FORGET_S = 120                # un objet non revu depuis 2 min est oublié
LINK_OK_S = 3                 # lien OK si un paquet est arrivé il y a moins de 3 s
LINK_DEGRADED_S = 10          # sinon dégradé jusqu'à 10 s, puis NO SIGNAL

lock = threading.Lock()
objects = {}                          # objets uniques (clé = type + case de 10 cm)
packet_log = deque(maxlen=8)          # derniers paquets reçus
arrivals = deque(maxlen=500)          # (heure, taille JSON équivalente, latence ms)
stats = {"received": 0, "lost": 0, "corrupted": 0, "last_seq": None}
state = {"last_rx": None, "next_id": 1}
START = time.time()

REF = {"jpeg": None, "w": 640, "h": 480}   # photo de référence (déjà réduite comme sur le rover)


def load_reference(path):
    """Charge la photo de référence : une image, ou la 1re image d'une vidéo."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"[dashboard] image introuvable : {p}")
        return None
    if p.suffix.lower() in IMAGE_EXT:
        frame = cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(p))
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    return shrink(frame) if frame is not None else None


def receive_telemetry(sock):
    print(f"[dashboard] télémétrie sur UDP {UDP_PORT}")
    while True:
        packet, _ = sock.recvfrom(4096)
        data = decode(packet)
        if data is None:
            with lock:
                stats["corrupted"] += 1
            print("[dashboard] paquet corrompu rejeté")
            continue

        now = time.time()
        latency_ms = (int(now * 1000) - data["ts"]) & 0xFFFF
        if latency_ms > 60000:
            latency_ms = 0
        json_len = len(json.dumps(
            {"seq": data["seq"], "type": data["type"], "x_cm": data["x_cm"],
             "y_cm": data["y_cm"], "confidence": data["confidence"],
             "timestamp": round(now, 2)}, separators=(",", ":")))

        key = (data["type"], round(data["x_cm"] / CELL_CM), round(data["y_cm"] / CELL_CM))
        with lock:
            last = stats["last_seq"]
            if last is not None:
                gap = (data["seq"] - last) & 0xFFFF
                if 1 < gap < 1000:               # trou dans la numérotation = paquets perdus
                    stats["lost"] += gap - 1
            stats["last_seq"] = data["seq"]
            stats["received"] += 1
            state["last_rx"] = now
            arrivals.append((now, json_len, latency_ms))

            obj = objects.get(key)
            if obj is None:
                obj = {"id": f"H{state['next_id']}", "hits": 0}
                state["next_id"] += 1
                objects[key] = obj
            obj.update(type=data["type"], x_cm=data["x_cm"], y_cm=data["y_cm"],
                       confidence=data["confidence"], seq=data["seq"],
                       last_rx=now, latency_ms=latency_ms)
            obj["hits"] += 1

            packet_log.appendleft({"seq": data["seq"], "type": data["type"],
                                   "x_cm": data["x_cm"], "y_cm": data["y_cm"],
                                   "confidence": data["confidence"],
                                   "latency_ms": latency_ms, "rx_at": now})


def snapshot():
    now = time.time()
    cx, cy, focal = intrinsics(REF["w"], REF["h"])
    with lock:
        for k in [k for k, o in objects.items() if now - o["last_rx"] > FORGET_S]:
            del objects[k]
        objs = [dict(o) for o in objects.values()]
        log = [dict(e) for e in packet_log]
        received, lost, corrupted = stats["received"], stats["lost"], stats["corrupted"]
        recent = [a for a in arrivals if now - a[0] <= 10]
        last_rx = state["last_rx"]

    # --- état de la liaison ---
    last_age = None if last_rx is None else now - last_rx
    if last_age is None:
        link = "WAITING"
    elif last_age < LINK_OK_S:
        link = "OK"
    elif last_age < LINK_DEGRADED_S:
        link = "DEGRADED"
    else:
        link = "NO SIGNAL"

    window = max(1.0, min(10.0, now - START))
    rate = len(recent) / window
    avg_json = sum(a[1] for a in recent) / len(recent) if recent else 0
    avg_latency = sum(a[2] for a in recent) / len(recent) if recent else 0
    total = received + lost

    # --- objets : position, danger, projection sur la photo ---
    items = []
    for o in objs:
        age = now - o["last_rx"] + o["latency_ms"] / 1000
        in_path = abs(o["x_cm"]) <= CORRIDOR_HALF_WIDTH_CM and o["y_cm"] <= CORRIDOR_LENGTH_CM
        px = ground_to_pixel(o["x_cm"], o["y_cm"], cx, cy, focal)
        zone = []
        for k in range(24):
            a = 2 * math.pi * k / 24
            p = ground_to_pixel(o["x_cm"] + KEEP_OUT_CM * math.cos(a),
                                o["y_cm"] + KEEP_OUT_CM * math.sin(a), cx, cy, focal)
            if p:
                zone.append([round(p[0], 1), round(p[1], 1)])
        items.append({
            "id": o["id"], "type": o["type"], "x_cm": o["x_cm"], "y_cm": o["y_cm"],
            "dist_cm": round(math.hypot(o["x_cm"], o["y_cm"]), 1),
            "confidence": o["confidence"], "age_s": round(age, 2), "hits": o["hits"],
            "level": "red" if in_path else "green",
            "status": "IN PATH" if in_path else "OFF PATH",
            "u": round(px[0], 1) if px else None, "v": round(px[1], 1) if px else None,
            "zone": zone,
        })
    items.sort(key=lambda i: i["dist_cm"])

    # --- couloir du rover projeté sur la photo ---
    ys = list(range(20, CORRIDOR_LENGTH_CM + 1, 10))
    edge = ([ground_to_pixel(-CORRIDOR_HALF_WIDTH_CM, y, cx, cy, focal) for y in ys] +
            [ground_to_pixel(CORRIDOR_HALF_WIDTH_CM, y, cx, cy, focal) for y in reversed(ys)])
    corridor = [[round(p[0], 1), round(p[1], 1)] for p in edge if p]

    # --- alarme ---
    fresh_red = [i for i in items if i["level"] == "red" and i["age_s"] <= STALE_S]
    if link == "WAITING":
        alarm = {"level": "amber", "title": "WAITING FOR ROVER", "detail": "no telemetry received yet"}
    elif link == "NO SIGNAL":
        alarm = {"level": "amber", "title": "NO SIGNAL",
                 "detail": f"last packet {last_age:.0f} s ago: displayed data is aging"}
    elif fresh_red:
        alarm = {"level": "red", "title": "HAZARD IN PATH",
                 "detail": f"{len(fresh_red)} obstacle(s), nearest at {fresh_red[0]['dist_cm']:.0f} cm"}
    else:
        alarm = {"level": "green", "title": "PATH CLEAR",
                 "detail": f"no fresh hazard in the {2 * CORRIDOR_HALF_WIDTH_CM} cm x "
                           f"{CORRIDOR_LENGTH_CM} cm corridor"}

    max_dist = max((i["dist_cm"] for i in items), default=0)
    return {
        "mission_s": now - START,
        "link": link,
        "last_packet_age": None if last_age is None else round(last_age, 2),
        "frame": {"w": REF["w"], "h": REF["h"]},
        "has_image": REF["jpeg"] is not None,
        "corridor": corridor,
        "corridor_cm": {"half_width": CORRIDOR_HALF_WIDTH_CM, "length": CORRIDOR_LENGTH_CM,
                        "keep_out": KEEP_OUT_CM, "stale_s": STALE_S},
        "map_range_cm": max(200, math.ceil(max_dist * 1.3 / 50) * 50),
        "stats": {
            "received": received, "lost": lost, "corrupted": corrupted,
            "loss_pct": 100 * lost / total if total else 0,
            "rate_pps": rate, "packet_bytes": PACKET_SIZE,
            "bytes_per_s": rate * PACKET_SIZE,
            "json_bytes_per_s": sum(a[1] for a in recent) / window,
            "avg_json_len": avg_json, "avg_latency_ms": avg_latency,
        },
        "alarm": alarm,
        "objects": items,
        "log": [{**e, "age_s": round(now - e["rx_at"], 1)} for e in log],
    }


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C.R.A.T.E.R. Mission Control</title>
<style>
:root { --bg:#06121c; --panel:#0c2233; --line:#1f5671; --cyan:#59d8ff; --txt:#d7ecf7;
        --dim:#7fa3b8; --red:#ff3d3d; --green:#2bdc6b; --amber:#ffb020; --yel:#ffd400; }
* { box-sizing:border-box; }
html, body { margin:0; background:var(--bg); color:var(--txt);
             font-family:Consolas,"Courier New",monospace; font-size:13px; }
body { display:flex; flex-direction:column; gap:8px; padding:8px; min-height:100vh; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px 10px; }
.panel h2 { margin:0 0 6px; font-size:11px; letter-spacing:.14em; color:var(--cyan);
            text-transform:uppercase; border-bottom:1px solid var(--line); padding-bottom:4px; }
.note { float:right; color:var(--dim); font-weight:normal; letter-spacing:0; text-transform:none; }
.top { display:flex; flex-wrap:wrap; gap:24px; align-items:center; }
.top .title { font-size:20px; font-weight:bold; color:var(--cyan); letter-spacing:.06em; }
.top .item small { display:block; color:var(--dim); font-size:10px; letter-spacing:.12em; }
.top .item .v { font-size:16px; font-weight:bold; }
.ok { color:var(--green); } .warn { color:var(--amber); } .bad { color:var(--red); }
.main { display:grid; grid-template-columns:270px minmax(0,1fr) 350px; gap:8px;
        height:calc(100vh - 215px); min-height:560px; }
.col { display:flex; flex-direction:column; gap:8px; min-height:0; }
.cam { display:flex; flex-direction:column; min-height:0; }
.cam-body, .map-body { position:relative; flex:1; min-height:200px; }
.cam-body svg, .map-body svg { position:absolute; inset:0; width:100%; height:100%; }
.map-body { flex:none; height:300px; }
.grow { flex:1; overflow:auto; }
.alarm { padding:12px 10px; border-radius:3px; text-align:center; }
.alarm b { display:block; font-size:18px; letter-spacing:.08em; }
.alarm span { font-size:11px; }
.alarm.red { background:var(--red); color:#fff; animation:blink 1s infinite; }
.alarm.green { background:#12502c; color:#bff5d3; border:1px solid var(--green); }
.alarm.amber { background:#5a3c00; color:#ffe2a8; border:1px solid var(--amber); }
@keyframes blink { 50% { background:#8f1f1f; } }
table { width:100%; border-collapse:collapse; }
th, td { padding:3px 5px; text-align:left; border-bottom:1px solid #153a52; white-space:nowrap; }
th { color:var(--dim); font-weight:normal; font-size:10px; letter-spacing:.1em; }
.kv td:last-child { text-align:right; color:var(--cyan); }
.chip { padding:1px 6px; border-radius:2px; font-size:11px; font-weight:bold; color:#04110a; }
.chip.red { background:var(--red); color:#fff; } .chip.green { background:var(--green); }
.legend div { margin:4px 0; font-size:12px; }
.sw { display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:-2px; border-radius:2px; }
@media (max-width:1150px) { .main { grid-template-columns:1fr; height:auto; }
                            .cam-body { min-height:420px; } }
</style>
</head>
<body>
<div class="panel top">
  <div class="title">C.R.A.T.E.R. MISSION CONTROL</div>
  <div class="item"><small>MISSION TIME</small><span class="v" id="mtime">00:00:00</span></div>
  <div class="item"><small>LINK</small><span class="v" id="link">-</span></div>
  <div class="item"><small>LAST PACKET</small><span class="v" id="lastpkt">-</span></div>
  <div class="item"><small>HAZARDS TRACKED</small><span class="v" id="nobj">0</span></div>
</div>

<div class="main">
  <div class="col">
    <div class="panel"><h2>Alarm</h2>
      <div id="alarm" class="alarm amber"><b id="alarm-title">WAITING</b><span id="alarm-detail"></span></div>
    </div>
    <div class="panel"><h2>Link quality</h2><table class="kv" id="linkq"></table></div>
    <div class="panel legend"><h2>Legend</h2><div id="legend"></div></div>
  </div>

  <div class="panel cam">
    <h2>Camera view &mdash; hazard zones <span class="note">reference photo &middot; zones rebuilt from telemetry coordinates</span></h2>
    <div class="cam-body">
      <svg id="cam" preserveAspectRatio="xMidYMid meet" viewBox="0 0 640 480">
        <rect id="bg" x="0" y="0" width="640" height="480" fill="#02080d"/>
        <image id="photo" x="0" y="0" width="640" height="480"/>
        <g id="ov"></g>
      </svg>
    </div>
  </div>

  <div class="col">
    <div class="panel"><h2>Map &mdash; top view (cm)</h2>
      <div class="map-body"><svg id="map" preserveAspectRatio="xMidYMax meet"></svg></div>
    </div>
    <div class="panel grow"><h2>Hazard list</h2><table id="haz"></table></div>
  </div>
</div>

<div class="panel"><h2>Telemetry log (last packets)</h2><table id="log"></table></div>

<script>
const RED = '#ff3d3d', GREEN = '#2bdc6b', YEL = '#ffd400';
const $ = id => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined) ? '-' : Number(v).toFixed(d);
const pts = a => a.map(p => p[0] + ',' + p[1]).join(' ');
const opacityFor = a => Math.max(0.3, 1 - a / 40);
const ageClass = a => a < 3 ? 'ok' : (a < 10 ? 'warn' : 'bad');

function hms(s) {
  s = Math.floor(s);
  const p = n => String(n).padStart(2, '0');
  return p(Math.floor(s / 3600)) + ':' + p(Math.floor(s % 3600 / 60)) + ':' + p(s % 60);
}

let camKey = '';
function render(d) {
  const C = d.corridor_cm;

  // --- en-tête ---
  $('mtime').textContent = hms(d.mission_s);
  const L = $('link');
  L.textContent = d.link;
  L.className = 'v ' + (d.link === 'OK' ? 'ok' : d.link === 'DEGRADED' ? 'warn' : 'bad');
  $('lastpkt').textContent = d.last_packet_age === null ? '-' : fmt(d.last_packet_age) + ' s';
  $('nobj').textContent = d.objects.length;

  // --- alarme ---
  $('alarm').className = 'alarm ' + d.alarm.level;
  $('alarm-title').textContent = d.alarm.title;
  $('alarm-detail').textContent = d.alarm.detail;

  // --- qualité du lien ---
  const s = d.stats;
  $('linkq').innerHTML = [
    ['Received', s.received],
    ['Lost (seq gaps)', s.lost + ' (' + fmt(s.loss_pct, 0) + '%)'],
    ['Corrupted (CRC)', s.corrupted],
    ['Rate', fmt(s.rate_pps) + ' pkt/s'],
    ['Bandwidth used', fmt(s.bytes_per_s, 0) + ' B/s'],
    ['Same data as JSON', fmt(s.json_bytes_per_s, 0) + ' B/s'],
    ['Packet size', s.packet_bytes + ' B vs ~' + fmt(s.avg_json_len, 0) + ' B'],
    ['Avg latency', fmt(s.avg_latency_ms, 0) + ' ms'],
  ].map(r => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('');

  // --- légende ---
  $('legend').innerHTML =
    '<div><span class="sw" style="background:' + RED + '"></span>RED: hazard inside the rover corridor (|x| &le; ' + C.half_width + ' cm, y &le; ' + C.length + ' cm)</div>' +
    '<div><span class="sw" style="background:' + GREEN + '"></span>GREEN: hazard detected, outside the corridor</div>' +
    '<div><span class="sw" style="background:' + YEL + ';opacity:.5"></span>Yellow band: rover corridor</div>' +
    '<div>Circle radius = ' + C.keep_out + ' cm keep-out zone</div>' +
    '<div>Faded / dashed = aging data (&gt; ' + C.stale_s + ' s)</div>';

  // --- vue caméra ---
  const cam = $('cam');
  const key = d.frame.w + 'x' + d.frame.h + ':' + d.has_image;
  if (key !== camKey) {
    camKey = key;
    cam.setAttribute('viewBox', '0 0 ' + d.frame.w + ' ' + d.frame.h);
    ['bg', 'photo'].forEach(id => { $(id).setAttribute('width', d.frame.w); $(id).setAttribute('height', d.frame.h); });
    if (d.has_image) $('photo').setAttribute('href', '/image');
  }
  let ov = '';
  if (d.corridor.length) {
    ov += '<polygon points="' + pts(d.corridor) + '" fill="' + YEL + '" fill-opacity=".14" stroke="' + YEL +
          '" stroke-width="1.5" stroke-dasharray="6 4"/>';
  }
  const placed = [];
  d.objects.slice().sort((a, b) => b.v - a.v).forEach(o => {
    if (o.u === null) return;
    const col = o.level === 'red' ? RED : GREEN;
    const op = opacityFor(o.age_s);
    const dash = o.age_s > C.stale_s ? ' stroke-dasharray="5 4"' : '';
    const tx = Math.min(Math.max(o.u, 62), d.frame.w - 62);
    let ty = Math.max(o.v - 50, 4);
    while (placed.some(q => Math.abs(q.x - tx) < 120 && Math.abs(q.y - ty) < 36) && ty > 40) ty -= 36;
    placed.push({ x: tx, y: ty });
    ov += '<g opacity="' + op + '">' +
      (o.zone.length ? '<polygon points="' + pts(o.zone) + '" fill="' + col + '" fill-opacity=".30" stroke="' + col + '" stroke-width="2"' + dash + '/>' : '') +
      '<circle cx="' + o.u + '" cy="' + o.v + '" r="4" fill="' + col + '"/>' +
      '<line x1="' + o.u + '" y1="' + (ty + 32) + '" x2="' + o.u + '" y2="' + o.v + '" stroke="' + col + '" stroke-width="1"/>' +
      '<rect x="' + (tx - 58) + '" y="' + ty + '" width="116" height="32" rx="3" fill="#000" fill-opacity=".72" stroke="' + col + '"/>' +
      '<text x="' + tx + '" y="' + (ty + 13) + '" text-anchor="middle" fill="' + col + '" font-size="12" font-weight="bold">' +
        o.id + ' ' + o.type + ' ' + Math.round(o.dist_cm) + ' cm</text>' +
      '<text x="' + tx + '" y="' + (ty + 26) + '" text-anchor="middle" fill="#d7ecf7" font-size="11">x ' + o.x_cm + '  y ' + o.y_cm + ' cm</text>' +
      '</g>';
  });
  if (!d.has_image) {
    ov += '<text x="' + (d.frame.w / 2) + '" y="30" text-anchor="middle" fill="#7fa3b8" font-size="16">NO REFERENCE IMAGE (start with --image)</text>';
  }
  $('ov').innerHTML = ov;

  // --- carte vue de dessus ---
  const R = d.map_range_cm, Xr = Math.round(R * 0.75);
  let m = '';
  for (let r = 50; r <= R; r += 50) {
    m += '<circle cx="0" cy="0" r="' + r + '" fill="none" stroke="#1f5671" stroke-width="1"' + (r % 100 ? ' stroke-dasharray="3 4"' : '') + '/>' +
         '<text x="4" y="' + (-r + 13) + '" fill="#7fa3b8" font-size="13">' + r + '</text>';
  }
  m += '<line x1="0" y1="0" x2="0" y2="' + (-R) + '" stroke="#1f5671"/>';
  m += '<rect x="' + (-C.half_width) + '" y="' + (-C.length) + '" width="' + (2 * C.half_width) + '" height="' + C.length +
       '" fill="' + YEL + '" fill-opacity=".14" stroke="' + YEL + '" stroke-dasharray="6 4"/>';
  d.objects.forEach(o => {
    const col = o.level === 'red' ? RED : GREEN;
    const op = opacityFor(o.age_s);
    const dash = o.age_s > C.stale_s ? ' stroke-dasharray="4 3"' : '';
    m += '<g opacity="' + op + '"><circle cx="' + o.x_cm + '" cy="' + (-o.y_cm) + '" r="' + C.keep_out +
         '" fill="' + col + '" fill-opacity=".25" stroke="' + col + '" stroke-width="2"' + dash + '/>' +
         '<circle cx="' + o.x_cm + '" cy="' + (-o.y_cm) + '" r="4" fill="' + col + '"/>' +
         '<text x="' + (o.x_cm + 8) + '" y="' + (-o.y_cm - 8) + '" fill="#d7ecf7" font-size="14">' + o.id + '</text></g>';
  });
  m += '<polygon points="0,-16 -9,-4 9,-4" fill="#59d8ff"/><rect x="-14" y="-4" width="28" height="32" rx="3" fill="#59d8ff" fill-opacity=".85"/>';
  $('map').setAttribute('viewBox', (-Xr) + ' ' + (-R) + ' ' + (2 * Xr) + ' ' + (R + 40));
  $('map').innerHTML = m;

  // --- liste des dangers ---
  $('haz').innerHTML = '<tr><th>ID</th><th>TYPE</th><th>X</th><th>Y</th><th>DIST</th><th>CONF</th><th>AGE</th><th>STATUS</th></tr>' +
    d.objects.map(o => '<tr><td>' + o.id + '</td><td>' + o.type + '</td><td>' + o.x_cm + '</td><td>' + o.y_cm +
      '</td><td>' + Math.round(o.dist_cm) + '</td><td>' + o.confidence + '%</td>' +
      '<td class="' + ageClass(o.age_s) + '">' + fmt(o.age_s) + ' s</td>' +
      '<td><span class="chip ' + o.level + '">' + o.status + '</span></td></tr>').join('');

  // --- journal des paquets ---
  $('log').innerHTML = '<tr><th>SEQ</th><th>TYPE</th><th>X (cm)</th><th>Y (cm)</th><th>CONF</th><th>LATENCY</th><th>AGE</th></tr>' +
    d.log.map(e => '<tr><td>' + e.seq + '</td><td>' + e.type + '</td><td>' + e.x_cm + '</td><td>' + e.y_cm +
      '</td><td>' + e.confidence + '%</td><td>' + e.latency_ms + ' ms</td>' +
      '<td class="' + ageClass(e.age_s) + '">' + fmt(e.age_s) + ' s</td></tr>').join('');
}

async function update() {
  try {
    const r = await fetch('/data', { cache: 'no-store' });
    render(await r.json());
  } catch (e) {
    const L = $('link');
    L.textContent = 'DASHBOARD OFFLINE';
    L.className = 'v bad';
  }
}
setInterval(update, 500);
update();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body, cache="no-store"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # le navigateur a fermé la connexion

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/data":
            self._send(200, "application/json", json.dumps(snapshot()).encode("utf-8"))
        elif path == "/image" and REF["jpeg"] is not None:
            self._send(200, "image/jpeg", REF["jpeg"], cache="max-age=60")
        elif path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        else:
            self._send(404, "text/plain", b"not found")

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="photo de référence (image ou vidéo) affichée sur la vue caméra")
    args = parser.parse_args()

    frame = load_reference(args.image)
    if frame is not None:
        REF["h"], REF["w"] = frame.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            REF["jpeg"] = buf.tobytes()
    if REF["jpeg"] is None:
        print("[dashboard] pas de photo de référence : lance avec --image data/lune.jpg")

    # on ouvre le port UDP ici : s'il est déjà pris, le programme s'arrête tout de suite avec une erreur claire
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, UDP_PORT))
    threading.Thread(target=receive_telemetry, args=(sock,), daemon=True).start()

    server = ThreadingHTTPServer((HOST, HTTP_PORT), Handler)
    print(f"[dashboard] http://{HOST}:{HTTP_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
