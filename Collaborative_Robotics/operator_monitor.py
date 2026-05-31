"""
Operator Monitor — unified face recognition + fatigue detection.

Single camera, single MediaPipe Face Mesh call per frame, two analyses:
  • Who is in front of the camera?  (matched against photos in operators/)
  • Are they getting drowsy?         (EAR, PERCLOS, yawn-count, micro-sleep)

Press T to toggle a task-dashboard side panel that lists the identified
operator's tasks from operators.csv.

How to enroll a person:
  1) Drop a clear, well-lit front-facing photo into operators/
     e.g.  operators/Justin.jpg   →   identified name will be "Justin"
  2) Add a matching row to operators.csv with role + tasks.
  3) Run this script — enrollment happens automatically at startup.

Run:    python3 operator_monitor.py
Keys:   [T] tasks panel    [R] reset counters    [Q] quit
"""

import csv
import platform
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import face_recognition       # dlib-based deep face recognition

# ── Paths ──────────────────────────────────────────────────
HERE          = Path(__file__).parent
OPERATORS_DIR = HERE / "operators"
OPERATORS_CSV = HERE / "operators.csv"

# ── MediaPipe Face Mesh setup ──────────────────────────────
mp_face = mp.solutions.face_mesh

# Landmark indices reused from fatigue_monitor
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_TOP, MOUTH_BOT   = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308

# ── Fatigue thresholds (less sensitive — was triggering too easily) ──
EAR_CLOSED         = 0.16    # was 0.21 — eyes must be more clearly closed
EAR_HISTORY_SEC    = 60.0
PERCLOS_DROWSY_PCT = 32.0    # was 20% — need more sustained eye closure
MICROSLEEP_SEC     = 1.8     # was 1.0s — need a longer closed-eye event
MAR_YAWN           = 0.62    # was 0.55 — only count clearly open mouths
YAWN_MIN_SEC       = 0.8     # was 0.6 — must hold the yawn longer
YAWN_WINDOW_SEC    = 60.0
YAWN_DROWSY_COUNT  = 5       # was 5 — need more yawns to trip drowsy
RECOVERY_ALERT_SEC = 5.0
VOICE_COOLDOWN_SEC = 6.0       # don't repeat a voice alert faster than this
SNOOZE_AFTER_CLICK_SEC = 20.0  # click anywhere to silence alerts for this long

# ── Display size ──────────────────────────────────────────
# The captured frame is upscaled to this width before drawing overlays.
# 1280 ≈ half a 2560-wide laptop screen.
DISPLAY_WIDTH       = 1280

# ── Face recognition (dlib via face_recognition) ───────────
# face_recognition returns L2 distance between 128-D encodings.
# Library default for "same person" is 0.6. We use 0.45 (strict) because the
# enrolled photos sit close in embedding space and we'd rather show
# "Unknown" than misidentify someone.
ID_MAX_DISTANCE     = 0.45
# Re-run detection every N frames (identity doesn't flip frame-to-frame).
ID_FRAME_STRIDE     = 4
# Need N consecutive matches of the same name before committing — kills jitter.
ID_COMMIT_FRAMES    = 4
# Downscale factor for live face_locations call (speeds detection up ~4x).
ID_DETECT_SCALE     = 0.5


# ── Helpers ────────────────────────────────────────────────
def _speak(text):
    """Non-blocking voice alert on macOS via `say`, fallback to console bell."""
    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["say", "-v", "Samantha", text],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            pass
    print("\a", end="", flush=True)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _eye_aspect_ratio(pts_px, idx):
    p = [pts_px[i] for i in idx]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)


def _mouth_aspect_ratio(pts_px):
    return (_dist(pts_px[MOUTH_TOP], pts_px[MOUTH_BOT]) /
            (_dist(pts_px[MOUTH_LEFT], pts_px[MOUTH_RIGHT]) + 1e-6))


# ── Deep face recognition via dlib (face_recognition library) ──
class FaceID:
    """
    Thin wrapper around the `face_recognition` library.
    `encode_from_bgr(bgr_image)` → 128-D dlib encoding of the largest face, or None.
    `encode_live(bgr_frame)`     → faster live-frame version that downscales
                                    for detection but encodes at original res.
    """
    def __init__(self):
        # Sanity: confirm the library actually loads (dlib must be present).
        _ = face_recognition.__version__

    @staticmethod
    def _largest(locations):
        """Pick the (top, right, bottom, left) box with the biggest area."""
        return max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))

    def encode_from_bgr(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return None
        loc = self._largest(locs)
        encs = face_recognition.face_encodings(rgb, known_face_locations=[loc])
        return encs[0] if encs else None

    def encode_live(self, bgr_frame):
        """Optimised path for the live camera loop."""
        small = cv2.resize(bgr_frame, (0, 0),
                           fx=ID_DETECT_SCALE, fy=ID_DETECT_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs_small = face_recognition.face_locations(rgb_small, model="hog")
        if not locs_small:
            return None, None
        # Scale boxes back up to full-resolution coords
        inv = 1.0 / ID_DETECT_SCALE
        locs = [(int(t*inv), int(r*inv), int(b*inv), int(l*inv))
                for (t, r, b, l) in locs_small]
        loc = self._largest(locs)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb, known_face_locations=[loc])
        if not encs:
            return None, None
        return encs[0], loc


# ───────────────────────────────────────────────────────────
# Operator database (enrolls faces from operators/ + reads CSV)
# ───────────────────────────────────────────────────────────
class OperatorDB:
    def __init__(self, faceid: 'FaceID'):
        self.faceid = faceid
        self.operators = {}    # name -> {"vec":..., "thumb":..., "role":..., "tasks":[...]}
        self._load_csv_then_photos()

    def _load_csv_then_photos(self):
        # CSV first (role + tasks)
        meta = {}
        if OPERATORS_CSV.exists():
            # Read whole file, decode bytes with "replace" so any stray
            # non-UTF-8 byte (Excel/Word edits, smart quotes, etc.) becomes
            # the unicode replacement char instead of crashing the parser.
            raw = OPERATORS_CSV.read_bytes()
            text = raw.decode("utf-8", errors="replace").lstrip("﻿")
            import io
            try:
                for row in csv.DictReader(io.StringIO(text)):
                    nm = (row.get("name") or "").strip()
                    if not nm:
                        continue
                    tasks_raw = row.get("tasks", "") or ""
                    meta[nm] = {
                        "role":  (row.get("role") or "").strip(),
                        "tasks": [t.strip() for t in tasks_raw.split(";") if t.strip()],
                    }
            except csv.Error as e:
                print(f"[OperatorDB] CSV parse warning: {e} — continuing.")
            print(f"[OperatorDB] CSV: loaded {len(meta)} role/task entries")

        # Photos — enroll each into a SFace embedding
        if not OPERATORS_DIR.exists():
            print(f"[OperatorDB] No operators/ folder — no faces enrolled.")
            return

        for path in sorted(OPERATORS_DIR.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"[OperatorDB]  ✗ couldn't read {path.name}")
                continue
            emb = self.faceid.encode_from_bgr(img)
            if emb is None:
                print(f"[OperatorDB]  ✗ no face found in {path.name}")
                continue
            name = path.stem
            self.operators[name] = {
                "vec":   emb,
                "thumb": _make_thumb(img),
                "role":  meta.get(name, {}).get("role",  ""),
                "tasks": meta.get(name, {}).get("tasks", []),
            }
            print(f"[OperatorDB]  ✓ enrolled {name}")

        if not self.operators:
            print("[OperatorDB] No faces enrolled. Add photos to operators/.")

    def identify(self, vec):
        """Returns (best_name, similarity_score, confident).
        Always returns the CLOSEST enrolled operator — never None when any
        operators are enrolled. `confident` is True only when the distance
        is below the strict threshold; the UI can use it to soften the
        display (e.g. show 'closest: Justin' instead of just 'Justin').
        """
        if vec is None or not self.operators:
            return None, 0.0, False
        names = list(self.operators.keys())
        knowns = np.array([self.operators[n]["vec"] for n in names])
        dists = face_recognition.face_distance(knowns, vec)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        sim = max(0.0, 1.0 - dist)
        confident = dist <= ID_MAX_DISTANCE
        return names[idx], sim, confident


def _make_thumb(img):
    """Square 100x100 thumbnail (center crop)."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (100, 100))


# ───────────────────────────────────────────────────────────
# Combined per-frame analyzer (face id + fatigue, one FaceMesh pass)
# ───────────────────────────────────────────────────────────
class OperatorMonitor:
    def __init__(self, db: OperatorDB, faceid: FaceID, voice=True):
        self.db = db
        self.faceid = faceid
        self.face = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self._frame_n = 0
        # Fatigue state
        self._ear_hist     = deque()
        self._yawn_times   = deque()
        self._closed_since = None
        self._yawn_started = None
        self._smoothed_ear = None
        self.yawn_count       = 0
        self.microsleep_count = 0
        # Voice + reset
        self.voice_enabled = voice
        self._last_voice_at = 0.0
        self._alert_since   = None
        self._has_reset     = False
        # Click-to-snooze: when set, suppresses voice + flash until this time
        self._snoozed_until = 0.0
        # ID debouncing
        self._id_buffer = deque(maxlen=ID_COMMIT_FRAMES)
        self.committed_name = None
        self.committed_score = 0.0
        self.committed_confident = False

    def reset_counters(self):
        self.yawn_count = 0
        self.microsleep_count = 0
        self._ear_hist.clear()
        self._yawn_times.clear()
        self._has_reset = False

    def snooze(self):
        """Called when the user clicks the camera window to dismiss a warning."""
        self._snoozed_until = time.time() + SNOOZE_AFTER_CLICK_SEC
        print(f"[snooze] alerts silenced for {SNOOZE_AFTER_CLICK_SEC:.0f}s")

    def process(self, frame):
        h, w = frame.shape[:2]
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.face.process(rgb)

        info = {
            "face_found":       False,
            "operator":         None,
            "operator_score":   0.0,
            "ear":              None,
            "perclos":          0.0,
            "status":           "NO FACE",
            "status_color":     (120, 120, 120),
            "microsleep":       False,
            "yawning":          False,
            "yawn_count":       self.yawn_count,
            "yawns_recent":     len(self._yawn_times),
            "microsleep_count": self.microsleep_count,
        }

        if not res.multi_face_landmarks:
            self._closed_since = None
            self._yawn_started = None
            self._id_buffer.clear()
            return info

        info["face_found"] = True
        lms = res.multi_face_landmarks[0].landmark
        pts_px = [(int(l.x * w), int(l.y * h)) for l in lms]

        # ── Face ID via dlib (stride-throttled, downscaled detect) ──
        self._frame_n += 1
        if self._frame_n % ID_FRAME_STRIDE == 0:
            emb, _loc = self.faceid.encode_live(frame)
            cand, score, confident = self.db.identify(emb)
            self._id_buffer.append(cand)
            # Commit instantly if confident; otherwise wait for debounce
            if cand is not None:
                if confident or len(self._id_buffer) == self._id_buffer.maxlen \
                        and all(n == self._id_buffer[0] for n in self._id_buffer):
                    self.committed_name      = cand
                    self.committed_score     = score
                    self.committed_confident = confident
        info["operator"]            = self.committed_name
        info["operator_score"]      = self.committed_score
        info["operator_confident"]  = self.committed_confident

        # ── Fatigue (same formulas as fatigue_monitor.py) ─
        ear_l = _eye_aspect_ratio(pts_px, LEFT_EYE_IDX)
        ear_r = _eye_aspect_ratio(pts_px, RIGHT_EYE_IDX)
        ear   = (ear_l + ear_r) / 2.0
        mar   = _mouth_aspect_ratio(pts_px)
        self._smoothed_ear = ear if self._smoothed_ear is None \
            else 0.6 * self._smoothed_ear + 0.4 * ear
        info["ear"] = self._smoothed_ear

        # PERCLOS
        self._ear_hist.append((now, ear))
        cutoff = now - EAR_HISTORY_SEC
        while self._ear_hist and self._ear_hist[0][0] < cutoff:
            self._ear_hist.popleft()
        closed_frames = sum(1 for _, e in self._ear_hist if e < EAR_CLOSED)
        info["perclos"] = 100.0 * closed_frames / max(1, len(self._ear_hist))

        # Micro-sleep
        if ear < EAR_CLOSED:
            if self._closed_since is None:
                self._closed_since = now
            elif now - self._closed_since >= MICROSLEEP_SEC:
                if not info["microsleep"]:
                    self.microsleep_count += 1
                    info["microsleep_count"] = self.microsleep_count
                info["microsleep"] = True
        else:
            self._closed_since = None

        # Yawn
        if mar > MAR_YAWN:
            if self._yawn_started is None:
                self._yawn_started = now
            elif now - self._yawn_started >= YAWN_MIN_SEC:
                info["yawning"] = True
        else:
            if self._yawn_started and (now - self._yawn_started) >= YAWN_MIN_SEC:
                self.yawn_count += 1
                self._yawn_times.append(now)
                info["yawn_count"] = self.yawn_count
            self._yawn_started = None
        cutoff = now - YAWN_WINDOW_SEC
        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()
        info["yawns_recent"] = len(self._yawn_times)

        # Aggregate status (microsleep collapses into DROWSY for a shorter pill label)
        if info["microsleep"]:
            info["status"]       = "DROWSY"
            info["status_color"] = (0, 0, 255)
        elif info["perclos"] > PERCLOS_DROWSY_PCT or info["yawns_recent"] >= YAWN_DROWSY_COUNT:
            info["status"]       = "DROWSY"
            info["status_color"] = (0, 0, 255)
        elif info["perclos"] > PERCLOS_DROWSY_PCT * 0.5 or info["yawns_recent"] >= 3:
            info["status"]       = "TIRED"
            info["status_color"] = (0, 165, 255)
        else:
            info["status"]       = "ALERT"
            info["status_color"] = (0, 200, 0)

        # Are we currently snoozed?
        info["snoozed"] = now < self._snoozed_until

        # Voice — personalized with name when available, skipped while snoozed
        if (self.voice_enabled and not info["snoozed"]
                and (now - self._last_voice_at) > VOICE_COOLDOWN_SEC):
            who = self.committed_name or "Operator"
            msg = None
            if info["microsleep"]:
                msg = f"{who}, wake up. Eyes closed."
            elif info["status"] == "DROWSY":
                msg = f"{who}, you look drowsy. Take a break."
            elif info["status"] == "TIRED" and self._has_reset:
                msg = f"{who}, signs of fatigue detected."
            if msg:
                _speak(msg)
                self._last_voice_at = now
                self._has_reset = False

        # 5s sustained ALERT → clear history
        if info["status"] == "ALERT":
            if self._alert_since is None:
                self._alert_since = now
            elif (not self._has_reset
                  and now - self._alert_since >= RECOVERY_ALERT_SEC):
                self._ear_hist.clear()
                self._has_reset = True
        else:
            self._alert_since = None

        return info


# ───────────────────────────────────────────────────────────
# Drawing helpers — rounded shapes, anti-aliased text
# ───────────────────────────────────────────────────────────
AA = cv2.LINE_AA          # anti-aliased line type for everything

# UI animation + input state
_ui_state = {
    "task_progress": 0.0,
    "last_t":        None,
    "button_rects":  [],         # list of (rect, action_tuple) for hit-test
    "panel_open":    False,
    "editing_mode":  None,       # None | "add" | "edit"
    "editing_idx":   -1,
    "editing_text":  "",
    "editing_for":   None,       # operator name being edited
}
# Open / close durations in seconds. Time-based so frame rate doesn't matter.
TASKS_OPEN_SEC  = 0.22
TASKS_CLOSE_SEC = 0.18

def _ease_out(t):
    """Cubic ease-out: starts fast, decelerates."""
    return 1 - (1 - t) ** 3

def _alpha_blend(frame, x, y, w, h, color, alpha):
    sub = frame[y:y+h, x:x+w]
    overlay = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)

def _rounded_fill(frame, x1, y1, x2, y2, color, radius=10, alpha=1.0):
    """Filled rounded rectangle with optional alpha blending.
    Only the bounding rectangle is touched (avoids whole-frame copy)."""
    if alpha >= 1.0:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1, AA)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1, AA)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(frame, (cx, cy), radius, color, -1, AA)
        return
    # Sub-region alpha blend
    H, W = frame.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    sub = frame[sy1:sy2, sx1:sx2]
    overlay = sub.copy()
    # shape coords in sub-region space
    rx1, ry1, rx2, ry2 = x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1
    cv2.rectangle(overlay, (rx1 + radius, ry1), (rx2 - radius, ry2), color, -1, AA)
    cv2.rectangle(overlay, (rx1, ry1 + radius), (rx2, ry2 - radius), color, -1, AA)
    for cx, cy in [(rx1 + radius, ry1 + radius), (rx2 - radius, ry1 + radius),
                   (rx1 + radius, ry2 - radius), (rx2 - radius, ry2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1, AA)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)

def _rounded_outline(frame, x1, y1, x2, y2, color, radius=10, thickness=2):
    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness, AA)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness, AA)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness, AA)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness, AA)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius),
                180, 0, 90, color, thickness, AA)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius),
                270, 0, 90, color, thickness, AA)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius),
                 90, 0, 90, color, thickness, AA)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius),
                  0, 0, 90, color, thickness, AA)

def save_operators_csv(db, csv_path):
    """Write the in-memory operator tasks back to operators.csv."""
    import io
    # Preserve existing column order + any rows we didn't load (defensive)
    fieldnames = ["name", "role", "tasks"]
    existing = []
    if csv_path.exists():
        raw = csv_path.read_bytes().decode("utf-8", errors="replace").lstrip("﻿")
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            existing.append(row)
        if reader.fieldnames:
            # Keep any extra columns the user added
            for f in reader.fieldnames:
                if f and f not in fieldnames:
                    fieldnames.append(f)

    seen = set()
    rows_out = []
    for row in existing:
        name = (row.get("name") or "").strip()
        if name in db.operators:
            op = db.operators[name]
            row["role"]  = op.get("role", row.get("role", ""))
            row["tasks"] = ";".join(op.get("tasks", []))
        rows_out.append(row)
        seen.add(name)
    # Any in-memory operators not in CSV → append them
    for name, op in db.operators.items():
        if name in seen:
            continue
        rows_out.append({
            "name":  name,
            "role":  op.get("role", ""),
            "tasks": ";".join(op.get("tasks", [])),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _round_photo_corners(photo, radius=14):
    """Returns the photo with rounded corners (areas outside the rounded
    box are darkened so they blend cleanly when pasted onto a dark UI)."""
    h, w = photo.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1, AA)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1, AA)
    for cx, cy in [(radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)]:
        cv2.circle(mask, (cx, cy), radius, 255, -1, AA)
    out = photo.copy()
    out[mask == 0] = (18, 20, 24)        # corners match top-bar bg
    return out

def _metric_color(value, good_below, warn_below):
    """Pick a green/amber/red based on thresholds.
    Returns BGR tuple."""
    if value < good_below:
        return (110, 220, 130)           # green
    if value < warn_below:
        return (90, 200, 255)            # amber
    return (90, 90, 245)                 # red


def draw_main(frame, info, db, show_tasks):
    h, w = frame.shape[:2]

    # ── Top status bar — slim, dark, with a soft bottom rule ──
    BAR_H = 140
    _alpha_blend(frame, 0, 0, w, BAR_H, (15, 17, 22), 0.92)
    # A subtle accent line in the operator's status colour
    cv2.line(frame, (0, BAR_H), (w, BAR_H),
             info["status_color"], 1, AA)

    # ── ID photo with rounded corners + soft border in status color ──
    PHOTO = 110
    px, py = 18, (BAR_H - PHOTO) // 2
    op = db.operators.get(info["operator"]) if info["operator"] else None
    if op and op.get("thumb") is not None:
        thumb = cv2.resize(op["thumb"], (PHOTO, PHOTO))
        rounded = _round_photo_corners(thumb, radius=14)
        frame[py:py+PHOTO, px:px+PHOTO] = rounded
    else:
        _rounded_fill(frame, px, py, px + PHOTO, py + PHOTO,
                      (50, 50, 55), radius=14)
        cv2.putText(frame, "?", (px + 38, py + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (130, 130, 130), 4, AA)
    _rounded_outline(frame, px - 2, py - 2, px + PHOTO + 2, py + PHOTO + 2,
                     info["status_color"], radius=16, thickness=2)

    # ── Name + role + match (cleaner type hierarchy) ────
    text_x = px + PHOTO + 22
    name = info["operator"] or "Searching..."
    cv2.putText(frame, name, (text_x, py + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 248), 2, AA)

    role = ""
    if info["operator"]:
        role = db.operators.get(info["operator"], {}).get("role", "")
    if role:
        cv2.putText(frame, role.upper(), (text_x, py + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 150, 165), 1, AA)

    if info["operator"]:
        confident = info.get("operator_confident", False)
        bar_w  = int(80 * min(1.0, max(0.0, info['operator_score'])))
        # Small horizontal match-confidence bar
        cv2.rectangle(frame, (text_x, py + 84), (text_x + 80, py + 88),
                      (50, 55, 65), -1, AA)
        cv2.rectangle(frame, (text_x, py + 84), (text_x + bar_w, py + 88),
                      (120, 200, 130) if confident else (110, 140, 170),
                      -1, AA)
        label = "MATCH" if confident else "CLOSEST"
        cv2.putText(frame, f"{label}  {info['operator_score']:.2f}",
                    (text_x + 88, py + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (120, 200, 130) if confident else (130, 160, 190), 1, AA)

    # ── Rounded status pill (right side) ───────────────
    if not show_tasks:
        PILL_W, PILL_H = 240, 74
        pillx, pilly = w - PILL_W - 18, (BAR_H - PILL_H) // 2
        _rounded_fill(frame, pillx, pilly, pillx + PILL_W, pilly + PILL_H,
                      info["status_color"], radius=14, alpha=0.92)
        # Inner status indicator dot
        dot_x, dot_y = pillx + 22, pilly + PILL_H // 2
        cv2.circle(frame, (dot_x, dot_y), 6, (255, 255, 255), -1, AA)
        cv2.circle(frame, (dot_x, dot_y), 6, (255, 255, 255), 2, AA)
        cv2.putText(frame, "STATUS", (pillx + 40, pilly + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, AA)
        cv2.putText(frame, info["status"], (pillx + 40, pilly + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, AA)

    # ── Fatigue metrics card (bottom-right) ─────────────
    if info["ear"] is not None and not show_tasks:
        cw, ch = 270, 152
        cx, cy = w - cw - 16, h - ch - 16
        _rounded_fill(frame, cx, cy, cx + cw, cy + ch,
                      (16, 18, 24), radius=12, alpha=0.92)
        _rounded_outline(frame, cx, cy, cx + cw, cy + ch,
                         (55, 60, 72), radius=12, thickness=1)

        FONT  = cv2.FONT_HERSHEY_SIMPLEX
        SCALE = 0.55
        THICK = 1
        left_edge  = cx + 16
        right_edge = cx + cw - 16

        # Color thresholds for live coloring
        ear_color = (200, 220, 235) if info['ear'] is None else (
            _metric_color(0.30 - info['ear'], 0.10, 0.17))    # invert: bigger EAR = better
        # Actually treat low EAR as bad. Use closed-eye fraction:
        ear_color = _metric_color(max(0.0, 0.30 - info['ear']), 0.10, 0.17)
        perclos_color = _metric_color(info['perclos'], 16.0, 32.0)
        yawn_color    = _metric_color(info['yawns_recent'], 3, 5)

        rows = [
            ("Eyelid Height", f"{info['ear']:.2f}",          ear_color),
            ("% Eye Closed",  f"{info['perclos']:.1f}%",     perclos_color),
            ("Yawns / min",   f"{info['yawns_recent']} / 5", yawn_color),
        ]
        row_y = cy + 30
        for label, value, vcolor in rows:
            cv2.putText(frame, label, (left_edge, row_y),
                        FONT, SCALE, (150, 158, 172), THICK, AA)
            (vw, _), _ = cv2.getTextSize(value, FONT, SCALE + 0.05, THICK + 1)
            cv2.putText(frame, value, (right_edge - vw, row_y),
                        FONT, SCALE + 0.05, vcolor, THICK + 1, AA)
            row_y += 26

        # PERCLOS progress bar at bottom of card
        bar_y = cy + ch - 22
        bar_x1, bar_x2 = left_edge, right_edge
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 6),
                      (40, 45, 55), -1, AA)
        fill_w = int((bar_x2 - bar_x1) * min(1.0, info['perclos'] / 100.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y),
                          (bar_x1 + fill_w, bar_y + 6),
                          perclos_color, -1, AA)

    # ── Flash alert (suppressed while snoozed) ─────────
    if not info.get("snoozed"):
        if info["microsleep"]:
            _flash(frame, "WAKE UP")
        elif info["status"] == "DROWSY":
            _flash(frame, "TAKE A BREAK")
    elif info["microsleep"] or info["status"] == "DROWSY":
        _rounded_fill(frame, 16, h - 36, 360, h - 12,
                      (35, 35, 42), radius=10, alpha=0.85)
        cv2.putText(frame, "alerts snoozed — click to dismiss",
                    (28, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (160, 165, 175), 1, AA)

    # ── Bottom-left hint strip (keyboard shortcuts) ────
    _draw_key_hints(frame, show_tasks)

    # ── Tasks side panel (time-based slide-in) ─────────
    now_t = time.time()
    if _ui_state["last_t"] is None:
        _ui_state["last_t"] = now_t
    dt = now_t - _ui_state["last_t"]
    _ui_state["last_t"] = now_t
    target = 1.0 if show_tasks else 0.0
    cur = _ui_state["task_progress"]
    duration = TASKS_OPEN_SEC if target > cur else TASKS_CLOSE_SEC
    step = dt / max(0.001, duration)
    if target > cur:
        cur = min(target, cur + step)
    else:
        cur = max(target, cur - step)
    _ui_state["task_progress"] = cur

    if cur > 0.001:
        draw_tasks_panel(frame, info, db, progress=cur)
    if cur < 0.999:
        _hint_button(frame, "[T] Tasks", fade=1.0 - cur)


def _draw_key_hints(frame, show_tasks):
    """Small unobtrusive keyboard shortcut row at the bottom-left."""
    h, w = frame.shape[:2]
    hints = [("T", "Hide" if show_tasks else "Tasks"),
             ("R", "Reset"), ("Q", "Quit")]
    x = 16
    y = h - 28
    for key, label in hints:
        kw = 22
        _rounded_fill(frame, x, y, x + kw, y + 20,
                      (40, 44, 54), radius=4, alpha=0.85)
        cv2.putText(frame, key, (x + 6, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 230), 1, AA)
        cv2.putText(frame, label, (x + kw + 6, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 145, 158), 1, AA)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x += kw + 6 + lw + 18


def _flash(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    alpha = 0.14 + 0.06 * abs(np.sin(time.time() * 4))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Centered pill behind the text for legibility
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.3, 5)
    pad_x, pad_y = 50, 26
    px1 = (w - tw) // 2 - pad_x
    px2 = px1 + tw + 2 * pad_x
    py1 = (h - th) // 2 - pad_y
    py2 = py1 + th + 2 * pad_y
    _rounded_fill(frame, px1, py1, px2, py2, (0, 0, 0), radius=18, alpha=0.55)
    _rounded_outline(frame, px1, py1, px2, py2, (255, 255, 255),
                     radius=18, thickness=2)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.3, (255, 255, 255), 5, AA)


def _hint_button(frame, label, fade=1.0):
    """Floating Tasks button. `fade` (0..1) lets it disappear smoothly
    as the side panel slides in."""
    if fade <= 0.02:
        return
    h, w = frame.shape[:2]
    bw, bh = 150, 42
    # Slide off-screen to the right slightly as it fades
    offset = int((1 - fade) * 30)
    bx, by = w - bw - 16 + offset, 156
    _rounded_fill(frame, bx, by, bx + bw, by + bh, (35, 38, 48),
                  radius=12, alpha=0.92 * fade)
    _rounded_outline(frame, bx, by, bx + bw, by + bh,
                     (int(110 * fade), int(115 * fade), int(130 * fade)),
                     radius=12, thickness=1)
    color = (int(235 * fade), int(235 * fade), int(240 * fade))
    cv2.putText(frame, label, (bx + 18, by + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, AA)


def draw_tasks_panel(frame, info, db, progress=1.0):
    """Slide-in glassy tasks panel. `progress` 0..1 controls how far in it is."""
    h, w = frame.shape[:2]
    PW = 520                                          # final panel width
    eased = _ease_out(max(0.0, min(1.0, progress)))
    visible = int(PW * eased)                         # how much is on-screen
    if visible <= 4:
        return
    # Reset button rect registry (filled while drawing this panel)
    _ui_state["button_rects"] = []
    fully_open = progress >= 0.999
    panel_x = w - visible                             # left edge of panel
    panel_y = 14
    panel_h = h - 28
    panel_x2 = w - 14
    panel_y2 = panel_h + 14

    # Panel content is rendered into a temp ROI so the rounded mask cleanly
    # composites onto whatever's behind (camera feed, mesh, etc.).
    # ── Layered glassy background ─────────────────────
    # Dark base
    _rounded_fill(frame, panel_x, panel_y, panel_x2, panel_y2,
                  (14, 16, 22), radius=22, alpha=0.78 * eased)
    # Subtle gradient overlay — lighter at top
    _rounded_fill(frame, panel_x, panel_y, panel_x2, panel_y + 120,
                  (28, 32, 42), radius=22, alpha=0.35 * eased)
    # Thin highlight along the top + left edges
    _rounded_outline(frame, panel_x, panel_y, panel_x2, panel_y2,
                     (90, 100, 120), radius=22, thickness=1)

    # All inner content positions are in-panel coordinates
    cx = panel_x + 32
    inner_w = visible - 64

    # ── Header — uppercase kicker + thin accent dot ───
    cv2.putText(frame, "TASKS FOR TODAY", (cx, panel_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 188, 205), 1, AA)
    # Small status dot next to the header
    cv2.circle(frame, (cx + 220, panel_y + 46), 4, (90, 200, 130), -1, AA)
    # Hairline divider
    cv2.line(frame, (cx, panel_y + 78), (panel_x2 - 32, panel_y + 78),
             (45, 50, 62), 1, AA)

    op_name = info["operator"]
    if not op_name or op_name not in db.operators:
        cv2.putText(frame, "No operator identified",
                    (cx, panel_y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 185, 195), 1, AA)
        cv2.putText(frame, "Once a face is matched, their tasks will appear here.",
                    (cx, panel_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 125, 135), 1, AA)
        _draw_panel_footer(frame, cx, panel_x2, panel_y2)
        return

    op = db.operators[op_name]

    # ── Operator name + role ──────────────────────────
    cv2.putText(frame, op_name, (cx, panel_y + 128),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 246, 250), 2, AA)
    if op["role"]:
        cv2.putText(frame, op["role"].upper(), (cx, panel_y + 156),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130, 140, 158), 1, AA)

    # Task count chip
    tasks = op["tasks"]
    chip_text = f"{len(tasks)} ASSIGNED"
    (cw, _), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    chip_x1 = panel_x2 - 32 - cw - 18
    chip_y1 = panel_y + 110
    _rounded_fill(frame, chip_x1, chip_y1, chip_x1 + cw + 18, chip_y1 + 24,
                  (50, 55, 70), radius=10, alpha=0.8)
    cv2.putText(frame, chip_text, (chip_x1 + 9, chip_y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1, AA)

    # ── Hairline divider before tasks ────────────────
    cv2.line(frame, (cx, panel_y + 188), (panel_x2 - 32, panel_y + 188),
             (35, 40, 52), 1, AA)

    # ── "+ Add Task" pill button ──────────────────────
    ab_x1, ab_y1 = cx, panel_y + 200
    ab_x2, ab_y2 = cx + 138, ab_y1 + 30
    _rounded_fill(frame, ab_x1, ab_y1, ab_x2, ab_y2,
                  (40, 60, 90), radius=10, alpha=0.95)
    _rounded_outline(frame, ab_x1, ab_y1, ab_x2, ab_y2,
                     (110, 165, 230), radius=10, thickness=1)
    cv2.putText(frame, "+  ADD TASK", (ab_x1 + 14, ab_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 210, 245), 1, AA)
    if fully_open:
        _ui_state["button_rects"].append(
            ((ab_x1, ab_y1, ab_x2, ab_y2), ("add",)))

    # ── Tasks list ────────────────────────────────────
    y = panel_y + 264
    if not tasks:
        cv2.putText(frame, "No tasks yet — press + ADD TASK above.",
                    (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 145, 155), 1, AA)
    else:
        for i, t in enumerate(tasks, start=1):
            y = _draw_task_line(frame, t, i, cx, y, inner_w - 16,
                                fully_open=fully_open,
                                panel_right=panel_x2 - 32)
            y += 28

    _draw_panel_footer(frame, cx, panel_x2, panel_y2)


def draw_text_input_modal(frame):
    """In-window text input overlay shown while editing/adding a task."""
    if not _ui_state.get("editing_mode"):
        return
    h, w = frame.shape[:2]
    # Dim the whole frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Centered card
    bw, bh = min(820, w - 80), 220
    bx = (w - bw) // 2
    by = (h - bh) // 2
    _rounded_fill(frame, bx, by, bx + bw, by + bh, (22, 26, 36),
                  radius=18, alpha=1.0)
    _rounded_outline(frame, bx, by, bx + bw, by + bh,
                     (110, 165, 230), radius=18, thickness=2)

    # Header
    mode = _ui_state["editing_mode"]
    title = "ADD NEW TASK" if mode == "add" else "EDIT TASK"
    cv2.putText(frame, title, (bx + 30, by + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 195, 215), 1, AA)
    for_name = _ui_state.get("editing_for") or ""
    if for_name:
        cv2.putText(frame, f"for {for_name}",
                    (bx + 30, by + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 140, 155), 1, AA)

    # Hairline divider
    cv2.line(frame, (bx + 30, by + 84), (bx + bw - 30, by + 84),
             (50, 56, 70), 1, AA)

    # Input field — rounded inset with the current buffer + blinking caret
    fx1, fy1 = bx + 30, by + 104
    fx2, fy2 = bx + bw - 30, by + 154
    _rounded_fill(frame, fx1, fy1, fx2, fy2, (12, 14, 20),
                  radius=10, alpha=1.0)
    _rounded_outline(frame, fx1, fy1, fx2, fy2, (60, 75, 100),
                     radius=10, thickness=1)
    text = _ui_state.get("editing_text", "")
    # Blinking caret
    caret = "|" if int(time.time() * 2) % 2 == 0 else " "
    cv2.putText(frame, text + caret, (fx1 + 14, fy2 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 235, 245), 1, AA)

    # Hints
    cv2.putText(frame, "ENTER  save",
                (bx + 30, by + bh - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 170, 190), 1, AA)
    cv2.putText(frame, "ESC  cancel",
                (bx + 160, by + bh - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 170, 190), 1, AA)


def _draw_panel_footer(frame, cx, panel_x2, panel_y2):
    """Bottom-of-panel rounded chip with the hide hint."""
    fy1 = panel_y2 - 42
    fy2 = panel_y2 - 14
    fw  = 110
    _rounded_fill(frame, cx, fy1, cx + fw, fy2, (28, 32, 42),
                  radius=10, alpha=0.85)
    _rounded_outline(frame, cx, fy1, cx + fw, fy2, (55, 62, 78),
                     radius=10, thickness=1)
    cv2.putText(frame, "T  hide", (cx + 14, fy2 - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, AA)


def _draw_task_line(frame, text, idx, x, y, max_w,
                    fully_open=True, panel_right=None):
    """Smooth task row: tiny accent dot + clean text + edit/delete chips
    at the right edge."""
    # Tiny rounded accent square (acts like a bullet)
    dot_size = 18
    cy = y - dot_size + 2
    _rounded_fill(frame, x, cy, x + dot_size, cy + dot_size,
                  (35, 40, 54), radius=5)
    # Place the small index number centered inside the dot
    idx_str = f"{idx:02d}"
    (iw, ih), _ = cv2.getTextSize(idx_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, idx_str,
                (x + (dot_size - iw) // 2, cy + (dot_size + ih) // 2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 180, 235), 1, AA)

    # Word wrap, lighter weight than before
    words = text.split()
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 1)
        if tw < max_w - 40:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    text_x = x + dot_size + 14
    LINE_GAP = 26
    for i, line in enumerate(lines):
        ly = y + i * LINE_GAP
        cv2.putText(frame, line, (text_x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (228, 232, 240), 1, AA)
    # Faint divider below the task — gives the list its rhythm
    last_y = y + max(0, (len(lines) - 1)) * LINE_GAP
    cv2.line(frame,
             (text_x, last_y + 14),
             (text_x + min(int(max_w * 0.75), 320), last_y + 14),
             (32, 36, 48), 1, AA)

    # ── Edit + Delete chips at the right edge of the row ──
    if panel_right is not None and fully_open:
        chip_w, chip_h = 28, 22
        gap = 6
        del_x2 = panel_right
        del_x1 = del_x2 - chip_w
        edt_x2 = del_x1 - gap
        edt_x1 = edt_x2 - chip_w
        chip_y1 = y - 14
        chip_y2 = chip_y1 + chip_h
        # Edit chip
        _rounded_fill(frame, edt_x1, chip_y1, edt_x2, chip_y2,
                      (40, 50, 64), radius=6, alpha=0.95)
        _rounded_outline(frame, edt_x1, chip_y1, edt_x2, chip_y2,
                         (110, 165, 230), radius=6, thickness=1)
        cv2.putText(frame, "E", (edt_x1 + 9, chip_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (175, 210, 245), 1, AA)
        # Delete chip
        _rounded_fill(frame, del_x1, chip_y1, del_x2, chip_y2,
                      (60, 36, 40), radius=6, alpha=0.95)
        _rounded_outline(frame, del_x1, chip_y1, del_x2, chip_y2,
                         (230, 110, 110), radius=6, thickness=1)
        cv2.putText(frame, "X", (del_x1 + 9, chip_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 175, 175), 1, AA)

        _ui_state["button_rects"].append(
            ((edt_x1, chip_y1, edt_x2, chip_y2), ("edit", idx - 1)))
        _ui_state["button_rects"].append(
            ((del_x1, chip_y1, del_x2, chip_y2), ("delete", idx - 1)))

    return last_y


# ───────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────
def main(camera_index=0):
    t0 = time.time()
    print("→ Loading dlib face recognition ...", flush=True)
    try:
        faceid = FaceID()
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure face_recognition is installed:  pip install face_recognition")
        return
    print(f"  done ({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    print(f"→ Enrolling operators from {OPERATORS_DIR}/ ...", flush=True)
    db = OperatorDB(faceid)
    print(f"  done ({time.time()-t1:.1f}s) — {list(db.operators.keys()) or '(none)'}",
          flush=True)

    t2 = time.time()
    print("→ Opening camera + MediaPipe ...", flush=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera at index {camera_index}")
        return
    # Short warm-up: most cameras deliver a frame within a few tries.
    for _ in range(8):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.05)
    monitor = OperatorMonitor(db, faceid)
    print(f"  done ({time.time()-t2:.1f}s)", flush=True)
    print(f"\nReady in {time.time()-t0:.1f}s.", flush=True)
    print("Click the window to dismiss fatigue warnings for 30s.\n", flush=True)

    WINDOW_NAME = "Operator Monitor"
    cv2.namedWindow(WINDOW_NAME)

    def _begin_add():
        name = monitor.committed_name
        if not name or name not in db.operators:
            print("[add] No operator identified.")
            return
        _ui_state["editing_mode"] = "add"
        _ui_state["editing_idx"]  = -1
        _ui_state["editing_text"] = ""
        _ui_state["editing_for"]  = name

    def _begin_edit(idx):
        name = monitor.committed_name
        if not name or name not in db.operators:
            return
        tasks = db.operators[name]["tasks"]
        if not (0 <= idx < len(tasks)):
            return
        _ui_state["editing_mode"] = "edit"
        _ui_state["editing_idx"]  = idx
        _ui_state["editing_text"] = tasks[idx]
        _ui_state["editing_for"]  = name

    def _do_delete(idx):
        name = monitor.committed_name
        if not name or name not in db.operators:
            return
        tasks = db.operators[name]["tasks"]
        if 0 <= idx < len(tasks):
            removed = tasks.pop(idx)
            save_operators_csv(db, OPERATORS_CSV)
            print(f"[del] Removed: {removed}")

    def _commit_edit():
        mode = _ui_state.get("editing_mode")
        name = _ui_state.get("editing_for")
        text = _ui_state.get("editing_text", "").strip()
        if name and text and name in db.operators:
            tasks = db.operators[name]["tasks"]
            if mode == "add":
                tasks.append(text)
                print(f"[add] Added: {text}")
            elif mode == "edit":
                idx = _ui_state.get("editing_idx", -1)
                if 0 <= idx < len(tasks):
                    tasks[idx] = text
                    print(f"[edit] Updated task {idx+1}: {text}")
            save_operators_csv(db, OPERATORS_CSV)
        _cancel_edit()

    def _cancel_edit():
        _ui_state["editing_mode"] = None
        _ui_state["editing_text"] = ""
        _ui_state["editing_idx"]  = -1
        _ui_state["editing_for"]  = None

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # If editing, swallow clicks (text-input is keyboard only)
        if _ui_state.get("editing_mode"):
            return
        # Hit-test panel buttons first
        for rect, action in _ui_state.get("button_rects", []):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                kind = action[0]
                if kind == "add":
                    _begin_add()
                elif kind == "edit":
                    _begin_edit(action[1])
                elif kind == "delete":
                    _do_delete(action[1])
                return
        # Fallback: snooze fatigue alerts
        monitor.snooze()
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)
    show_tasks = False

    print("Running. Keys:  T tasks | R reset | Q quit  (use + / E / X buttons in the panel)")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        info = monitor.process(frame)
        if frame.shape[1] != DISPLAY_WIDTH:
            new_h = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])
            display = cv2.resize(frame, (DISPLAY_WIDTH, new_h),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = frame
        draw_main(display, info, db, show_tasks)
        # Text input modal renders over everything when active
        draw_text_input_modal(display)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        # ── Edit mode captures all keys (so Q etc. don't quit) ──
        if _ui_state.get("editing_mode"):
            if key == 27:                # ESC — cancel
                _cancel_edit()
            elif key in (13, 10):        # ENTER — commit
                _commit_edit()
            elif key in (8, 127):        # BACKSPACE
                _ui_state["editing_text"] = _ui_state["editing_text"][:-1]
            elif 32 <= key <= 126:       # printable ASCII
                _ui_state["editing_text"] += chr(key)
            continue                     # skip normal key handling

        if key == ord('q'):
            break
        elif key == ord('t'):
            show_tasks = not show_tasks
        elif key == ord('r'):
            monitor.reset_counters()
            print("Counters reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
