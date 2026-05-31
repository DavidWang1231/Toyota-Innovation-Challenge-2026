"""
Operator Fatigue Monitor — laptop webcam.

Runs MediaPipe Face Mesh and computes:
  • EAR (Eye Aspect Ratio)        — closes-when-tired metric
  • PERCLOS                       — % of last 60s with eyes closed (drowsiness)
  • Sustained closure detection   — micro-sleep alert (>1s eyes shut)
  • Yawn count (Mouth Aspect Ratio)

Designed to share a window with a future operator_id module.

Two ways to use:
  1) Standalone:        `python3 fatigue_monitor.py`
  2) Importable:        `from fatigue_monitor import FatigueAnalyzer`
                        Call `analyzer.process(frame)` per frame, returns
                        a dict with the current metrics + status string.
"""

import platform
import subprocess
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# ── Voice alert (non-blocking) ─────────────────────────────
# On macOS, uses the built-in `say` command (no extra deps).
# On other systems it falls back to a console beep.
def _speak(text):
    if platform.system() == "Darwin":
        try:
            subprocess.Popen(["say", "-v", "Samantha", text],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)
        print(f"[VOICE] {text}")

# ── MediaPipe Face Mesh setup ──────────────────────────────
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Face Mesh landmark indices for eyes (refined eye contour from MediaPipe).
# Order chosen so EAR formula  ( |p2-p6| + |p3-p5| ) / (2*|p1-p4|)  works.
LEFT_EYE_IDX  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
# Mouth landmarks for yawn / MAR
MOUTH_TOP    = 13   # upper inner lip
MOUTH_BOT    = 14   # lower inner lip
MOUTH_LEFT   = 78   # left corner
MOUTH_RIGHT  = 308  # right corner

# ── Thresholds ─────────────────────────────────────────────
EAR_CLOSED         = 0.21   # eye is "closed" below this
EAR_HISTORY_SEC    = 60.0   # PERCLOS window length
PERCLOS_DROWSY_PCT = 20.0   # >this% of last window with eyes closed -> drowsy
MICROSLEEP_SEC     = 1.0    # eyes shut continuously this long -> alert
MAR_YAWN           = 0.55   # mouth aspect ratio above this -> mid-yawn
YAWN_MIN_SEC       = 0.6    # must hold open this long to count as a yawn
YAWN_WINDOW_SEC    = 60.0   # rolling window over which we count yawns
YAWN_DROWSY_COUNT  = 5      # this many yawns in the window -> escalate to DROWSY

# Voice + reset behavior
VOICE_COOLDOWN_SEC  = 6.0    # don't repeat a voice alert faster than this
RECOVERY_ALERT_SEC  = 5.0    # if ALERT this long in a row, clear warning state


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _eye_aspect_ratio(landmarks_px, idx):
    p = [landmarks_px[i] for i in idx]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)


def _mouth_aspect_ratio(landmarks_px):
    top    = landmarks_px[MOUTH_TOP]
    bottom = landmarks_px[MOUTH_BOT]
    left   = landmarks_px[MOUTH_LEFT]
    right  = landmarks_px[MOUTH_RIGHT]
    return _dist(top, bottom) / (_dist(left, right) + 1e-6)


class FatigueAnalyzer:
    """
    Per-frame fatigue computation. Stateful (keeps a 60s history of EAR samples).
    Returns a dict the caller can render however it wants.
    """

    def __init__(self, voice=True):
        self.face = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,           # needed for accurate eye points
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # (timestamp, ear) history for PERCLOS
        self._ear_hist = deque()
        # Micro-sleep tracking
        self._closed_since = None
        # Yawn tracking
        self._yawn_started = None
        self.yawn_count = 0
        self._yawn_times = deque()   # timestamps of recent yawns (for rolling window)
        self.microsleep_count = 0
        # Smoothed EAR for display
        self._smoothed_ear = None
        # Voice + auto-reset state
        self.voice_enabled = voice
        self._last_voice_at = 0.0
        self._alert_since   = None   # when ALERT state began
        self._has_reset     = False  # have we already cleared since last warning?

    def process(self, frame):
        h, w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face.process(rgb)

        info = {
            "face_found":     False,
            "ear":            None,
            "mar":            None,
            "perclos":        0.0,
            "status":         "NO FACE",
            "status_color":   (120, 120, 120),
            "microsleep":     False,
            "yawning":        False,
            "yawn_count":     self.yawn_count,
            "microsleep_count": self.microsleep_count,
            "eye_pts":        None,
            "mouth_pts":      None,
        }

        if not result.multi_face_landmarks:
            self._closed_since = None
            self._yawn_started = None
            return info

        info["face_found"] = True
        lm = result.multi_face_landmarks[0].landmark
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        ear_l = _eye_aspect_ratio(pts, LEFT_EYE_IDX)
        ear_r = _eye_aspect_ratio(pts, RIGHT_EYE_IDX)
        ear   = (ear_l + ear_r) / 2.0
        mar   = _mouth_aspect_ratio(pts)

        # Smooth for display
        self._smoothed_ear = ear if self._smoothed_ear is None \
            else 0.6 * self._smoothed_ear + 0.4 * ear

        info["ear"] = self._smoothed_ear
        info["mar"] = mar
        info["eye_pts"] = ([pts[i] for i in LEFT_EYE_IDX],
                           [pts[i] for i in RIGHT_EYE_IDX])
        info["mouth_pts"] = (pts[MOUTH_TOP], pts[MOUTH_BOT],
                             pts[MOUTH_LEFT], pts[MOUTH_RIGHT])

        # ── EAR history / PERCLOS ─────────────────────────
        self._ear_hist.append((now, ear))
        cutoff = now - EAR_HISTORY_SEC
        while self._ear_hist and self._ear_hist[0][0] < cutoff:
            self._ear_hist.popleft()
        closed_frames = sum(1 for _, e in self._ear_hist if e < EAR_CLOSED)
        total = len(self._ear_hist) or 1
        info["perclos"] = 100.0 * closed_frames / total

        # ── Sustained eye-closure (micro-sleep) ───────────
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

        # ── Yawn ──────────────────────────────────────────
        if mar > MAR_YAWN:
            if self._yawn_started is None:
                self._yawn_started = now
            elif now - self._yawn_started >= YAWN_MIN_SEC:
                if not info["yawning"]:
                    info["yawning"] = True
        else:
            if self._yawn_started and (now - self._yawn_started) >= YAWN_MIN_SEC:
                self.yawn_count += 1
                self._yawn_times.append(now)
                info["yawn_count"] = self.yawn_count
            self._yawn_started = None

        # Trim yawns outside the rolling window
        cutoff = now - YAWN_WINDOW_SEC
        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()
        yawns_recent = len(self._yawn_times)
        info["yawns_recent"] = yawns_recent

        # ── Aggregate status ──────────────────────────────
        if info["microsleep"]:
            info["status"]       = "ALERT — EYES CLOSED"
            info["status_color"] = (0, 0, 255)
        elif info["perclos"] > PERCLOS_DROWSY_PCT or yawns_recent >= YAWN_DROWSY_COUNT:
            info["status"]       = "DROWSY"
            info["status_color"] = (0, 0, 255)
        elif info["perclos"] > PERCLOS_DROWSY_PCT * 0.5 or yawns_recent >= 3:
            info["status"]       = "TIRED"
            info["status_color"] = (0, 165, 255)
        else:
            info["status"]       = "ALERT"
            info["status_color"] = (0, 200, 0)

        # ── Voice alert (throttled) ──────────────────────
        if self.voice_enabled and (now - self._last_voice_at) > VOICE_COOLDOWN_SEC:
            msg = None
            if info["microsleep"]:
                msg = "Wake up. Eyes closed."
            elif info["status"] == "DROWSY":
                msg = "You look drowsy. Please take a break."
            elif info["status"] == "TIRED" and self._has_reset:
                # Only nudge for "tired" right after a recovery, not constantly
                msg = "Heads up, signs of fatigue detected."
            if msg:
                _speak(msg)
                self._last_voice_at = now
                self._has_reset = False    # block reset until next recovery

        # ── Recovery: 5s sustained ALERT clears warning state ──
        if info["status"] == "ALERT":
            if self._alert_since is None:
                self._alert_since = now
            elif (not self._has_reset
                    and now - self._alert_since >= RECOVERY_ALERT_SEC):
                # Clear the PERCLOS history so the operator gets a fresh start
                self._ear_hist.clear()
                self._has_reset = True
                info["just_reset"] = True
        else:
            self._alert_since = None

        return info


# ─────────────────────────────────────────────────────────────
# Drawing helpers — used by the standalone window, and reusable
# later by the combined fatigue + operator_id UI.
# ─────────────────────────────────────────────────────────────
def draw_overlay(frame, info):
    h, w = frame.shape[:2]

    # Eye + mouth landmarks (debug viz)
    if info["eye_pts"]:
        for eye in info["eye_pts"]:
            for p in eye:
                cv2.circle(frame, p, 2, (0, 255, 255), -1)
            cv2.polylines(frame, [np.array(eye)], True, (0, 255, 255), 1)
    if info["mouth_pts"]:
        for p in info["mouth_pts"]:
            cv2.circle(frame, p, 2, (255, 200, 0), -1)

    # Status header
    cv2.rectangle(frame, (0, 0), (w, 50), (25, 25, 25), -1)
    cv2.putText(frame, f"Operator Fatigue Monitor", (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, info["status"], (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["status_color"], 2)

    # Metrics panel (right-aligned)
    metrics = []
    if info["ear"] is not None:
        metrics.append(f"EAR:      {info['ear']:.2f}")
        metrics.append(f"PERCLOS:  {info['perclos']:.1f}%")
        metrics.append(f"Yawns/60: {info.get('yawns_recent', 0)}/5")
        metrics.append(f"Total:    {info['yawn_count']}")
        metrics.append(f"Micro:    {info['microsleep_count']}")
    else:
        metrics.append("No face detected")

    cv2.rectangle(frame, (w - 220, 60), (w - 10, 60 + 22 * len(metrics) + 12),
                  (25, 25, 25), -1)
    for i, line in enumerate(metrics):
        cv2.putText(frame, line, (w - 210, 80 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Big visual alert when drowsy / micro-sleep
    if info["microsleep"]:
        _flash_alert(frame, "WAKE UP", (0, 0, 255))
    elif info["status"] == "DROWSY":
        _flash_alert(frame, "TAKE A BREAK", (0, 0, 255))


def _flash_alert(frame, text, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    alpha = 0.18 + 0.08 * abs(np.sin(time.time() * 4))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 5)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)


# ─────────────────────────────────────────────────────────────
# Standalone runner — only fires when this file is executed directly.
# When operator_id.py is added later, it will share this main loop.
# ─────────────────────────────────────────────────────────────
def _run_standalone(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera at index {camera_index}")
        return

    # Warm-up for macOS camera-permission negotiation
    for _ in range(30):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.1)

    analyzer = FatigueAnalyzer()
    print("Fatigue monitor running. Press Q to quit, R to reset counters.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # Do NOT mirror — keep real orientation (matches workplace policy)

        info = analyzer.process(frame)
        draw_overlay(frame, info)

        cv2.imshow("Operator Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            analyzer.yawn_count = 0
            analyzer.microsleep_count = 0
            analyzer._ear_hist.clear()
            print("Counters reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_standalone()
