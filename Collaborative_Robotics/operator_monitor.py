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
Keys:   [T] tasks panel    [R] reset counters    [V] toggle voice    [Q] quit
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
VOICE_COOLDOWN_SEC = 6.0
RECOVERY_ALERT_SEC = 5.0
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
# Drawing — main view + collapsible tasks panel
# ───────────────────────────────────────────────────────────
def _alpha_blend(frame, x, y, w, h, color, alpha):
    sub = frame[y:y+h, x:x+w]
    overlay = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def draw_main(frame, info, db, show_tasks):
    h, w = frame.shape[:2]

    # ── Top status bar (smaller photo + smaller text) ─────
    BAR_H = 140
    _alpha_blend(frame, 0, 0, w, BAR_H, (18, 20, 24), 0.88)
    cv2.line(frame, (0, BAR_H), (w, BAR_H), (60, 60, 70), 2)

    # ID photo — ~2/3 of previous (160 → 110)
    PHOTO = 110
    px, py = 14, (BAR_H - PHOTO) // 2
    op = db.operators.get(info["operator"]) if info["operator"] else None
    if op and op.get("thumb") is not None:
        thumb = cv2.resize(op["thumb"], (PHOTO, PHOTO))
        frame[py:py+PHOTO, px:px+PHOTO] = thumb
    else:
        _alpha_blend(frame, px, py, PHOTO, PHOTO, (50, 50, 55), 1.0)
        cv2.putText(frame, "?", (px + 38, py + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (130, 130, 130), 4)
    cv2.rectangle(frame, (px - 2, py - 2),
                  (px + PHOTO + 2, py + PHOTO + 2),
                  info["status_color"], 3)

    # Name + role + match (smaller text)
    text_x = px + PHOTO + 18
    name = info["operator"] or "Searching..."
    cv2.putText(frame, name, (text_x, py + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (240, 240, 240), 2)

    role = ""
    if info["operator"]:
        role = db.operators.get(info["operator"], {}).get("role", "")
    if role:
        cv2.putText(frame, role, (text_x, py + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 175), 1)

    if info["operator"]:
        confident = info.get("operator_confident", False)
        label = f"match {info['operator_score']:.2f}"
        if not confident:
            label = f"closest · {label}"
        cv2.putText(frame, label, (text_x, py + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 120, 125) if confident else (110, 140, 170), 1)

    # Status pill — only draw if tasks panel isn't covering it
    if not show_tasks:
        PILL_W, PILL_H = 280, 74
        pillx, pilly = w - PILL_W - 14, (BAR_H - PILL_H) // 2
        _alpha_blend(frame, pillx, pilly, PILL_W, PILL_H,
                     info["status_color"], 0.9)
        cv2.rectangle(frame, (pillx, pilly), (pillx + PILL_W, pilly + PILL_H),
                      (255, 255, 255), 2)
        cv2.putText(frame, "STATUS", (pillx + 16, pilly + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, info["status"], (pillx + 16, pilly + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    # Fatigue metrics card (bottom-right) — 2/3 of previous size
    if info["ear"] is not None and not show_tasks:
        # (label, value) pairs — labels flush LEFT, values flush RIGHT
        rows = [
            ("Eyelid Height", f"{info['ear']:.2f}"),
            ("%EyeClose",     f"{info['perclos']:.1f}"),
            ("Yawns/s",       f"{info['yawns_recent']}/5"),
        ]
        row_h = 30
        cw, ch = 260, row_h * len(rows) + 22
        cx, cy = w - cw - 16, h - ch - 16
        _alpha_blend(frame, cx, cy, cw, ch, (15, 15, 15), 0.88)
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (60, 60, 70), 1)
        FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1
        left_edge  = cx + 16
        right_edge = cx + cw - 16
        for i, (label, value) in enumerate(rows):
            y = cy + 30 + i * row_h
            cv2.putText(frame, label, (left_edge, y),
                        FONT, SCALE, (170, 170, 175), THICK)
            (vw, _), _ = cv2.getTextSize(value, FONT, SCALE, THICK)
            cv2.putText(frame, value, (right_edge - vw, y),
                        FONT, SCALE, (230, 230, 235), THICK)

    # Flash alert (suppressed while snoozed)
    if not info.get("snoozed"):
        if info["microsleep"]:
            _flash(frame, "WAKE UP")
        elif info["status"] == "DROWSY":
            _flash(frame, "TAKE A BREAK")
    elif info["microsleep"] or info["status"] == "DROWSY":
        # Tiny indicator that we're snoozed (so it's not invisible)
        cv2.putText(frame, "(alerts snoozed - click to re-enable later)",
                    (20, frame.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 160), 1)

    # Tasks side panel
    if show_tasks:
        draw_tasks_panel(frame, info, db)
    else:
        _hint_button(frame, "[T] Tasks")


def _flash(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    alpha = 0.16 + 0.08 * abs(np.sin(time.time() * 4))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)


def _hint_button(frame, label):
    h, w = frame.shape[:2]
    bw, bh = 160, 48
    bx, by = w - bw - 14, 156          # just below the BAR_H=140 top bar
    _alpha_blend(frame, bx, by, bw, bh, (45, 45, 55), 0.92)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 210), 2)
    cv2.putText(frame, label, (bx + 16, by + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 245), 2)


def draw_tasks_panel(frame, info, db):
    h, w = frame.shape[:2]
    pw = 560                                  # tasks panel width
    px, py = w - pw, 0
    # Fully opaque — kills any bleed-through from the status pill underneath.
    _alpha_blend(frame, px, py, pw, h, (12, 14, 18), 1.0)
    cv2.line(frame, (px, 0), (px, h), (70, 70, 80), 2)

    # Header
    cv2.putText(frame, "TASKS FOR TODAY", (px + 28, py + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 230), 2)
    cv2.line(frame, (px + 28, py + 78), (px + pw - 28, py + 78),
             (80, 80, 95), 2)

    op_name = info["operator"]
    if not op_name or op_name not in db.operators:
        cv2.putText(frame, "No operator identified",
                    (px + 28, py + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        cv2.putText(frame, "(no tasks to show)",
                    (px + 28, py + 196),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (130, 130, 130), 2)
        cv2.putText(frame, "[T] hide panel",
                    (px + 28, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
        return

    op = db.operators[op_name]
    # Operator name
    cv2.putText(frame, op_name, (px + 28, py + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (245, 245, 245), 3)
    if op["role"]:
        cv2.putText(frame, op["role"], (px + 28, py + 184),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (170, 170, 180), 2)

    # Tasks list
    y = py + 260
    tasks = op["tasks"]
    if not tasks:
        cv2.putText(frame, "(no tasks listed in operators.csv)",
                    (px + 28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2)
    else:
        for i, t in enumerate(tasks, start=1):
            y = _draw_task_line(frame, t, i, px + 28, y, pw - 56)
            y += 64         # generous vertical spacing between tasks

    cv2.putText(frame, "[T] hide panel",
                (px + 28, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 175), 2)


def _draw_task_line(frame, text, idx, x, y, max_w):
    """Returns the y-coordinate of the last drawn line."""
    # Bigger number circle
    cv2.circle(frame, (x + 20, y - 14), 20, (0, 165, 255), 3)
    num_x = x + 12 if idx < 10 else x + 6
    cv2.putText(frame, str(idx), (num_x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)

    # Word wrap to as many lines as needed
    words = text.split()
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        if tw < max_w - 60:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    text_x = x + 54
    LINE_GAP = 36
    for i, line in enumerate(lines):
        ly = y + i * LINE_GAP
        cv2.putText(frame, line, (text_x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (230, 230, 235), 2)
    return y + max(0, (len(lines) - 1)) * LINE_GAP


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

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            monitor.snooze()
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)
    show_tasks = False

    print("Running. Keys:  T tasks panel | R reset | V toggle voice | Q quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # Real orientation — no mirroring.
        # Run analysis on the ORIGINAL frame (faster).
        info = monitor.process(frame)
        # Upscale to display size so overlays look big and readable.
        if frame.shape[1] != DISPLAY_WIDTH:
            new_h = int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])
            display = cv2.resize(frame, (DISPLAY_WIDTH, new_h),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = frame
        draw_main(display, info, db, show_tasks)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_tasks = not show_tasks
        elif key == ord('r'):
            monitor.reset_counters()
            print("Counters reset.")
        elif key == ord('v'):
            monitor.voice_enabled = not monitor.voice_enabled
            print(f"Voice: {'ON' if monitor.voice_enabled else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
