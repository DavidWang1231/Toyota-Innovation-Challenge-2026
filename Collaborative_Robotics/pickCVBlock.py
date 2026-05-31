import os
import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import hand_detection as hd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

"""CONSTANTS"""
Z_SAFE = 40
# Z_PICK = gripper height when grabbing a block off the table.
# More negative = closer to the table. The Dobot Magician can safely reach
# about -70 at typical XY range. If the gripper still hovers, drop this in
# 5mm steps. If it slams the table, raise it.
Z_PICK = -55
STABILITY_LIMIT = 60

# Robot speed: full when clear, slow when a hand is in the WARNING zone.
FULL_VEL, FULL_ACC = 100, 80
SLOW_VEL, SLOW_ACC = 30, 30

# Wave gesture (peace sign) settings.
WAVE_CYCLES   = 3       # how many left-right swings
WAVE_AMPL_DEG = 18      # how far the base joint swings each way (degrees)
WAVE_BIAS_DEG = 15      # lean the wave toward the side the hand came from
# If the wave goes the WRONG way relative to the worker, flip this to -1.
WAVE_DIR_SIGN = 1

# Camera index. FORCED (no auto-fallback) so we never accidentally use the
# laptop webcam, which would invalidate the homography matrix.
# 0 = laptop built-in. 1, 2, 3 = external/USB. Run hand_detection.py to find it.
CAMERA_INDEX = 1

# ---- Red object detection ----
# Tuned for the actual targets (see assets/IMG_6708.jpeg): LARGE red foam
# pieces (strips + stacked gear-shapes) on a warm wood table.
#
# Two complementary tests are AND'd together:
#   1. HSV hue ring (red wraps around H=0 and H=180)
#   2. LAB a* channel > LAB_A_MIN  (a* is OpenCV's red-vs-green axis,
#      0-255 with 128 = neutral. Wood sits at ~135-145, real red foam at
#      170+. This is the single most effective wood-rejector.)
RED_HSV_LOWER_1 = (0,   70,  50)
RED_HSV_UPPER_1 = (12,  255, 255)
RED_HSV_LOWER_2 = (168, 70,  50)
RED_HSV_UPPER_2 = (180, 255, 255)
# LAB a-channel floor — the workhorse for rejecting wood. Raise toward 170
# if wood gets through. Drop toward 140 if dim red objects get missed.
# 145 catches small/dim reds while still rejecting wood (wood a* ≈ 130-140).
LAB_A_MIN = 145
# Min area: covers a range from ~17x17 small cube up to large foam gears.
# Bump UP if you see false hits, DOWN if small reds are missed.
TARGET_MIN_AREA = 300
TARGET_MAX_AREA = 80000      # cap rejects giant background reds (clothing, walls)
# After two centroids land within this many ROBOT MILLIMETRES of each other,
# treat them as the same piece (keeps the bigger one).
DEDUPE_DIST_MM = 30
# Cap on targets per run.
MAX_TARGETS = 6
# Show "Red Mask (debug)" + "LAB a (debug)" windows so you can SEE what each
# gate is passing. Invaluable for tuning. Set False to hide.
RED_DEBUG_MASK = True

# ---- Tray detection (drop target) ----
# Tray = the most circular bright blob inside the CAUTION zone. If none is
# found within the timeout, we fall back to a fixed drop coordinate so the
# demo never stalls.
USE_TRAY_DETECTION     = True
TRAY_BRIGHTNESS_THRESH = 140      # 0..255 (lower if tray missed, raise if false hits)
TRAY_MIN_AREA          = 1500
TRAY_MAX_AREA          = 40000
TRAY_MIN_CIRCULARITY   = 0.55     # 1.0 = perfect circle
TRAY_DEBUG_MASK        = True      # show "Tray Mask (debug)" window while searching
DROP_FALLBACK_ROBOT_XY = (250, 0)  # used if tray not found

# ── Zones (scaled for 640x480 camera) ──
# Defaults — these are the STARTING positions. After target detection,
# phase_setup_zones() lets the user redraw or drag any zone with the same
# keys hand_detection.py uses (D / C / H / M). The redrawn values overwrite
# these module-level globals at runtime.
# Coordinates are (x1, y1, x2, y2) in pixels.
HANDOFF_ZONE = (450, 280, 640, 480)   # bottom-right, worker reaches here
CAUTION_ZONE = (40,  80, 600, 470)    # whole workspace; outside = ignored
DANGER_ZONE  = (80, 150, 340, 460)    # tight box around the Dobot body

# Handoff Zone robot coordinates - adjust on competition day
HANDOFF_ROBOT_X = 180
HANDOFF_ROBOT_Y = -80

machine_state = "scanning target"  # skipping plate detection

# --- CAMERA & ROBOT INIT ---
print("[INIT] Loading Dobot API...")
api = dType.load()

print(f"[INIT] Opening camera at FORCED index {CAMERA_INDEX} (no fallback)...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"[ERROR] Camera index {CAMERA_INDEX} could not be opened.")
    print("  - Is the USB camera plugged in?")
    print("  - Try a different CAMERA_INDEX (0,1,2,3) at the top of this file.")
    print("  - If you switch cameras you MUST re-run getTransformationMatrix.py.")
    exit()
print(f"[INIT] Camera opened on index {CAMERA_INDEX}.")

print("[INIT] Loading calibration files...")
H_matrix = np.load(os.path.join(SCRIPT_DIR, "HomographyMatrix.npy"))
data = np.load(os.path.join(SCRIPT_DIR, "camera_params.npz"))
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

print("[INIT] Grabbing first camera frame...")
ret, frame = cap.read()
if not ret or frame is None:
    print("[ERROR] Camera opened but returned no frame. Check the USB camera.")
    cap.release()
    exit()
h, w = frame.shape[:2]
print(f"[INIT] Frame OK: {w}x{h}")
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

# ─────────────────────────────────────────
# ZONE / HAND DETECTION  (delegates to hand_detection.py)
# Hands-only (no pose) - the white Dobot was triggering pose's "forearm"
# detector, which made check_hands constantly return DANGER and froze the
# pick loop. The hand_detection module avoids this entirely.
# ─────────────────────────────────────────
def detect_zone(frame):
    return hd.detect_zone(frame, DANGER_ZONE, CAUTION_ZONE, HANDOFF_ZONE, draw=True)

def check_hands(frame):
    zone, _ = hd.detect_zone(frame, DANGER_ZONE, CAUTION_ZONE, HANDOFF_ZONE, draw=False)
    return zone

# Track current speed so we only send a speed command when it actually changes.
_current_speed = None
def set_speed(vel, acc):
    global _current_speed
    if _current_speed == (vel, acc):
        return
    dType.SetPTPCommonParams(api, vel, acc, isQueued=0)
    _current_speed = (vel, acc)

def do_wave(api, side):
    """Pause, then wave the arm a few times, leaning toward `side`."""
    print(f"[GESTURE] Peace sign detected - pausing and waving toward {side}.")
    set_speed(FULL_VEL, FULL_ACC)
    pose = dType.GetPose(api)             # [x, y, z, rHead, j1, j2, j3, j4]
    j1, j2, j3, j4 = pose[4], pose[5], pose[6], pose[7]
    lean = WAVE_BIAS_DEG * WAVE_DIR_SIGN * (1 if side == "left" else -1)
    center = j1 + lean
    for _ in range(WAVE_CYCLES):
        dobotArm.move_joint_angles(api, center + WAVE_AMPL_DEG, j2, j3, j4)
        dobotArm.move_joint_angles(api, center - WAVE_AMPL_DEG, j2, j3, j4)
    dobotArm.move_joint_angles(api, j1, j2, j3, j4)   # return to where we were
    print("[GESTURE] Wave complete - resuming task.")

def monitor_humans(frame):
    """
    Single safety + gesture check on `frame`. Reacts immediately:
      - peace sign -> pause + wave
      - DANGER zone -> stop and wait until clear
      - WARNING zone -> slow the robot down
      - otherwise   -> full speed
    Returns the zone string so callers can branch (e.g. HANDOFF).
    """
    zone, peace, side, _ = hd.detect_zone_and_gesture(
        frame, DANGER_ZONE, CAUTION_ZONE, HANDOFF_ZONE, draw=False)
    if peace:
        do_wave(api, side)
    if zone == "DANGER":
        wait_for_hand_clear(cap, map1, map2)
    elif zone == "WARNING":
        set_speed(SLOW_VEL, SLOW_ACC)
    else:
        set_speed(FULL_VEL, FULL_ACC)
    return zone

def draw_zones(frame):
    cx1, cy1, cx2, cy2 = CAUTION_ZONE
    dx1, dy1, dx2, dy2 = DANGER_ZONE
    hx1, hy1, hx2, hy2 = HANDOFF_ZONE
    cv2.rectangle(frame, (cx1,cy1), (cx2,cy2), (0,165,255), 2)
    cv2.putText(frame, "CAUTION", (cx1, cy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
    cv2.rectangle(frame, (dx1,dy1), (dx2,dy2), (0,0,255), 2)
    cv2.putText(frame, "DANGER", (dx1, dy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.rectangle(frame, (hx1,hy1), (hx2,hy2), (0,255,0), 2)
    cv2.putText(frame, "HANDOFF", (hx1, hy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(frame, "Hana | Collaborative Robot", (10, fh_disp-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

fh_disp = 480  # for draw_zones text position

def wait_for_hand_clear(cap, map1, map2):
    print("DANGER - Robot paused! Remove hand/arm.")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        zone, frame = detect_zone(frame)
        draw_zones(frame)
        cv2.putText(frame, "DANGER - ROBOT PAUSED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Detection", frame)
        cv2.waitKey(1)
        if zone not in ["DANGER", "WARNING"]:
            break
    print("Clear. Resuming in 3s...")
    for i in range(3,0,-1):
        print(f"  {i}...")
        time.sleep(1)

# ─────────────────────────────────────────
# COORDINATE TRANSFORM
# ─────────────────────────────────────────
def pixel_to_robot(u, v, H):
    p  = np.array([u, v, 1.0])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

# ─────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────
def next_state():
    global machine_state
    if machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning target"

# ─────────────────────────────────────────
# RED-OBJECT DETECTION (shared helper)
# ─────────────────────────────────────────
def _work_area_mask(shape):
    """1-channel mask: white inside (CAUTION zone) MINUS (DANGER zone)."""
    fh, fw = shape[:2]
    m = np.zeros((fh, fw), dtype=np.uint8)
    cx1, cy1, cx2, cy2 = CAUTION_ZONE
    m[max(0,cy1):min(fh,cy2), max(0,cx1):min(fw,cx2)] = 255
    dx1, dy1, dx2, dy2 = DANGER_ZONE
    m[max(0,dy1):min(fh,dy2), max(0,dx1):min(fw,dx2)] = 0
    return m

def find_red_objects(frame, draw_on=None):
    """
    Find red objects in the frame (the foam strips/gear-stacks).

    Pipeline (tuned for IMG_6708 — red foam on wood):
      1. Light blur to suppress sensor noise.
      2. HSV gate: hue in the two red bands AND S,V above floors.
      3. LAB gate: a* channel > LAB_A_MIN. This is the WOOD-REJECTOR —
         wood's a* sits well below 160, real red foam pushes past 170.
      4. AND both gates together (a pixel must satisfy BOTH to count).
      5. OPEN 3x3 to kill specks.
      6. CLOSE 15x15 to fill the gear-stack's internal cutouts so one
         object stays one contour.
      7. Dilate 5x5 to merge adjacent stacked pieces that touch.
      8. Drop tiny/huge contours, dedupe by robot-frame distance, keep
         only the MAX_TARGETS biggest.

    Returns list of (robot_x, robot_y).
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # ── HSV gate (red wraps around H=0 and H=180) ──────────────────────
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv_mask = (cv2.inRange(hsv, np.array(RED_HSV_LOWER_1),
                                  np.array(RED_HSV_UPPER_1))
                + cv2.inRange(hsv, np.array(RED_HSV_LOWER_2),
                                    np.array(RED_HSV_UPPER_2)))

    # ── LAB a* gate (red-vs-green axis) — the wood rejector ────────────
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    _, a_chan, _ = cv2.split(lab)
    lab_mask = cv2.inRange(a_chan, LAB_A_MIN, 255)

    # ── Combine: a pixel must be red in BOTH color spaces ──────────────
    mask = cv2.bitwise_and(hsv_mask, lab_mask)

    # OPEN first to delete salt-and-pepper specks. Use a 3x3 to preserve
    # small targets (a 30x30 px cube barely survives anything bigger).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    # CLOSE 9x9 — bridges small gaps without erasing small objects. Good
    # compromise between "fill foam gear cutouts" and "keep a small cube".
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    if RED_DEBUG_MASK:
        cv2.imshow("Red Mask (debug)", mask)
        cv2.imshow("LAB a (debug)", a_chan)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # First pass: collect (area, centroid_px, centroid_robot, contour) for
    # every contour that passes the size gate.
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < TARGET_MIN_AREA or area > TARGET_MAX_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        rx, ry = pixel_to_robot(cx, cy, H_matrix)
        candidates.append({"area": area, "px": (cx, cy),
                           "robot": (rx, ry), "cnt": cnt})

    # Sort by area DESC — bigger blobs win when we dedupe near-duplicates.
    candidates.sort(key=lambda c: c["area"], reverse=True)

    kept = []
    for c in candidates:
        rx, ry = c["robot"]
        is_dup = False
        for k in kept:
            kx, ky = k["robot"]
            if (rx - kx) ** 2 + (ry - ky) ** 2 < DEDUPE_DIST_MM ** 2:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
        if len(kept) >= MAX_TARGETS:
            break

    result = []
    for c in kept:
        cx, cy = c["px"]
        rx, ry = c["robot"]
        result.append((rx, ry))
        if draw_on is not None:
            cv2.drawContours(draw_on, [c["cnt"]], -1, (0, 255, 0), 2)
            cv2.circle(draw_on, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(draw_on, f"RED({int(c['area'])})",
                        (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result

# ─────────────────────────────────────────
# TRAY SELECTION (drop target) — MANUAL ONLY
# Auto-detection was unreliable on silver trays with washed-out cameras
# (tried Hough circles, then brightness+circularity contour). We now ask
# the user to click-and-drag a circle around the tray on the Detection
# window before the pick loop starts. Only the circle CENTER is used —
# it's converted via homography to a robot (x, y). The radius is purely
# visual feedback.
# ─────────────────────────────────────────
# Mouse-callback state. center=(x,y) on first mousedown; radius grows
# while you drag; commit=True after the user presses Enter/Space.
_sel = {"center": None, "radius": 0, "dragging": False, "commit": False}

def _tray_mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _sel["center"]   = (x, y)
        _sel["radius"]   = 0
        _sel["dragging"] = True
        _sel["commit"]   = False
    elif event == cv2.EVENT_MOUSEMOVE and _sel["dragging"]:
        cx, cy = _sel["center"]
        _sel["radius"] = int(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)
    elif event == cv2.EVENT_LBUTTONUP and _sel["dragging"]:
        _sel["dragging"] = False
        cx, cy = _sel["center"]
        _sel["radius"] = max(_sel["radius"],
                             int(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5))

def manual_select_tray():
    """
    Block until the user draws a circle around the tray and confirms.
    Controls (on the "Detection" window):
        Click + drag : draw circle (1st click = center, drag out = radius)
        Enter / Space: confirm the current circle
        R            : reset and draw again
    No cancel — the program cannot continue without a drop point.
    Returns [(robot_x, robot_y)] for the circle's center.
    """
    print("\n[PHASE 1] MANUAL TRAY SELECTION")
    print("  In the 'Detection' window: click the tray's center and drag")
    print("  outward to set its radius. Press ENTER or SPACE to confirm.")
    print("  Press R to redo.")
    cv2.namedWindow("Detection")
    cv2.setMouseCallback("Detection", _tray_mouse_cb)
    _sel["center"] = None
    _sel["radius"] = 0
    _sel["dragging"] = False
    _sel["commit"] = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = frame.copy()
        # Zones intentionally NOT drawn here — they'd just clutter the tray
        # picking UI. They come on after target detection (zone editor phase).

        if _sel["center"] is not None and _sel["radius"] > 0:
            cv2.circle(display, _sel["center"], _sel["radius"],
                       (255, 0, 255), 2)
            cv2.circle(display, _sel["center"], 5, (255, 0, 255), -1)

        if _sel["center"] is None:
            msg = "Click the tray center and drag out to set radius"
        elif _sel["dragging"]:
            msg = f"Radius: {_sel['radius']} px  (release to lock)"
        else:
            msg = "ENTER/SPACE = confirm    R = redo"
        cv2.putText(display, msg, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imshow("Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):                       # Enter or Space
            if _sel["center"] is not None and _sel["radius"] > 0 \
                    and not _sel["dragging"]:
                cx, cy = _sel["center"]
                rx, ry = pixel_to_robot(cx, cy, H_matrix)
                print(f"Tray locked at pixel ({cx},{cy}) "
                      f"-> robot ({rx:.1f}, {ry:.1f}).")
                # Clear our callback so it doesn't fight later windows.
                cv2.setMouseCallback("Detection", lambda *a, **k: None)
                return [(rx, ry)]
            else:
                print("No circle drawn yet — draw one first.")
        elif key in (ord('r'), ord('R')):
            _sel["center"]   = None
            _sel["radius"]   = 0
            _sel["dragging"] = False
            print("Tray selection reset.")

def phase_detect_tray():
    """Manual selection only — no auto-detection."""
    return manual_select_tray()

# ─────────────────────────────────────────
# PHASE 2: DETECT TARGETS
# ─────────────────────────────────────────
def phase_detect_targets():
    """PHASE 2 — find the red blocks. Hand detection is OFF here on purpose:
    we just want a clean view of the scene so blocks lock in faster, and no
    MediaPipe inference is wasted while the arm is still idle."""
    print("\n[PHASE 2] Scanning for targets... (hand detection OFF)")
    stability_counter = 0
    last_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        # Zones intentionally NOT drawn here — they'd just clutter the
        # detection UI. They come on after target lock (zone editor phase).

        current_list = find_red_objects(frame, draw_on=display_frame)

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter/STABILITY_LIMIT)*100)
        cv2.putText(display_frame,
                    f"LOCKING TARGETS: {progress}%  ({len(current_list)} found)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "HAND DETECTION: OFF",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} targets.")
            return current_list


# ─────────────────────────────────────────
# ZONE EDITOR (Phase 2.5) — interactive
# Ports the D/C/H/M keybindings from hand_detection.py's standalone demo
# so the operator can resize or drag the safety zones live, on top of a
# running hand-detection preview. The same window is reused.
# ─────────────────────────────────────────
# Mouse-callback state (lives at module scope so the cv2 callback can write
# to it). Two modes:
#   edit_target  None | "DANGER" | "CAUTION" | "HANDOFF"
#       → click two opposite corners to redraw the zone
#   move_target  same set
#       → click+drag inside the zone to move it without resizing
_zedit = {
    "edit_target": None,
    "move_target": None,
    "corner1":     None,    # first click during a redraw
    "mouse_pos":   (0, 0),  # latest cursor location (for live preview)
    "moving":      False,   # mouse currently held during a move
    "grab_off":    (0, 0),  # offset from zone top-left to grab point
}

def _zone_for(target):
    if target == "DANGER":  return DANGER_ZONE
    if target == "CAUTION": return CAUTION_ZONE
    if target == "HANDOFF": return HANDOFF_ZONE
    return None

def _set_zone(target, rect):
    """Overwrite the global tuple for `target`."""
    global DANGER_ZONE, CAUTION_ZONE, HANDOFF_ZONE
    if target == "DANGER":  DANGER_ZONE  = rect
    if target == "CAUTION": CAUTION_ZONE = rect
    if target == "HANDOFF": HANDOFF_ZONE = rect

def _zone_editor_mouse_cb(event, x, y, flags, param):
    z = _zedit
    z["mouse_pos"] = (x, y)

    # ── MOVE mode ─────────────────────────────────────────────────
    if z["move_target"] is not None:
        zx1, zy1, zx2, zy2 = _zone_for(z["move_target"])
        if event == cv2.EVENT_LBUTTONDOWN and zx1 <= x <= zx2 and zy1 <= y <= zy2:
            z["moving"]   = True
            z["grab_off"] = (x - zx1, y - zy1)
        elif event == cv2.EVENT_MOUSEMOVE and z["moving"]:
            w_box, h_box = zx2 - zx1, zy2 - zy1
            nx1, ny1 = x - z["grab_off"][0], y - z["grab_off"][1]
            _set_zone(z["move_target"],
                      (nx1, ny1, nx1 + w_box, ny1 + h_box))
        elif event == cv2.EVENT_LBUTTONUP and z["moving"]:
            z["moving"]     = False
            z["move_target"] = None
        return

    # ── REDRAW (resize) mode ──────────────────────────────────────
    if z["edit_target"] is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if z["corner1"] is None:
            z["corner1"] = (x, y)
        else:
            x1, y1 = z["corner1"]
            rect = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            if rect[2] - rect[0] > 15 and rect[3] - rect[1] > 15:
                _set_zone(z["edit_target"], rect)
            z["edit_target"] = None
            z["corner1"]     = None

def phase_setup_zones():
    """
    [PHASE 2.5] Show zones for the first time + let the user edit them
    AND warm up MediaPipe so hand detection is live before the arm moves.

    Keys (same as hand_detection.py's standalone demo, plus ENTER):
        D       → redraw DANGER  zone (then click two opposite corners)
        C       → redraw CAUTION zone (then click two opposite corners)
        H       → redraw HANDOFF zone (then click two opposite corners)
        M       → MOVE HANDOFF (drag the box without resizing it)
        ESC     → cancel any in-progress edit/move
        ENTER / SPACE → finish setup, start picking
    """
    print("\n[PHASE 2.5] Zone setup + arming hand detection.")
    print("  D / C / H = redraw DANGER / CAUTION / HANDOFF (click 2 corners)")
    print("  M = move HANDOFF (drag the box)")
    print("  ESC = cancel an edit/move    ENTER or SPACE = done, start picking")

    cv2.namedWindow("Detection")
    cv2.setMouseCallback("Detection", _zone_editor_mouse_cb)
    _zedit["edit_target"] = None
    _zedit["move_target"] = None
    _zedit["corner1"]     = None
    _zedit["moving"]      = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = frame.copy()

        # Live hand detection so MediaPipe warms up AND the user sees zones
        # responding to their hand in real time.
        zone, _, _, display = hd.detect_zone_and_gesture(
            display, DANGER_ZONE, CAUTION_ZONE, HANDOFF_ZONE, draw=True)

        draw_zones(display)

        # ── Edit-mode preview (yellow box from first corner to cursor) ──
        if _zedit["edit_target"] is not None:
            if _zedit["corner1"] is None:
                hint = f"EDIT {_zedit['edit_target']}: click FIRST corner   (ESC cancels)"
            else:
                hint = f"EDIT {_zedit['edit_target']}: click SECOND corner   (ESC cancels)"
                cv2.rectangle(display, _zedit["corner1"],
                              _zedit["mouse_pos"], (255, 255, 0), 2)
            cv2.putText(display, hint, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif _zedit["move_target"] is not None:
            cv2.putText(display,
                        f"MOVE {_zedit['move_target']}: click inside the box and drag  (ESC cancels)",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        else:
            cv2.putText(display,
                        "[D]anger  [C]aution  [H]andoff = redraw   [M] move HANDOFF",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display,
                        "ENTER / SPACE = done, start picking",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(display, f"hands: {zone}", (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        if key in (13, 32):                    # Enter / Space → done
            cv2.setMouseCallback("Detection", lambda *a, **k: None)
            print(f"[PHASE 2.5] Zones locked. DANGER={DANGER_ZONE} "
                  f"CAUTION={CAUTION_ZONE} HANDOFF={HANDOFF_ZONE}")
            return
        elif key == 27:                        # Esc → cancel any in-progress edit
            _zedit["edit_target"] = None
            _zedit["move_target"] = None
            _zedit["corner1"]     = None
            _zedit["moving"]      = False
        elif key == ord('d'):
            _zedit["edit_target"] = "DANGER";  _zedit["corner1"] = None
        elif key == ord('c'):
            _zedit["edit_target"] = "CAUTION"; _zedit["corner1"] = None
        elif key == ord('h'):
            _zedit["edit_target"] = "HANDOFF"; _zedit["corner1"] = None
        elif key == ord('m'):
            _zedit["move_target"] = "HANDOFF"; _zedit["edit_target"] = None

# Backwards-compat alias: anything that used to call phase_arm_hand_detection
# now gets the zone editor (which also warms up MediaPipe).
phase_arm_hand_detection = phase_setup_zones

def phase_detect_targets_quick(frame):
    return find_red_objects(frame)

# ─────────────────────────────────────────
# PHASE 3: PICK / PLACE
# ─────────────────────────────────────────
def phase_execute_batch(api, pick_list, drop_list):
    time.sleep(0.5)
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("Missing targets or drop zones, aborting.")
        return False
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]
        print(f"Task {i+1}: ({pick_x:.1f},{pick_y:.1f}) -> ({drop_x:.1f},{drop_y:.1f})")

        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        monitor_humans(frame)   # peace->wave, DANGER->stop, WARNING->slow

        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        monitor_humans(frame)

        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
        dobotArm.close_gripper(api)
        time.sleep(0.5)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        # Jidoka pick check
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        remaining = phase_detect_targets_quick(frame)
        pick_failed = any(abs(rx-pick_x)<20 and abs(ry-pick_y)<20 for rx,ry in remaining)
        if pick_failed:
            print("JIDOKA - Pick failed, retrying...")
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
            dobotArm.close_gripper(api)
            time.sleep(0.5)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        # Handoff or default drop
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        zone = monitor_humans(frame)   # also handles DANGER/WARNING/peace
        if zone == "HANDOFF":
            print("Worker ready - delivering to handoff zone!")
            dobotArm.move_to_xyz(api, HANDOFF_ROBOT_X, HANDOFF_ROBOT_Y, Z_SAFE)
            dobotArm.move_to_xyz(api, HANDOFF_ROBOT_X, HANDOFF_ROBOT_Y, Z_PICK)
            time.sleep(1.0)
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            dobotArm.move_to_xyz(api, HANDOFF_ROBOT_X, HANDOFF_ROBOT_Y, Z_SAFE)
            print("Part delivered!")
        else:
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("\nBatch complete.")
    return True

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
print("[INIT] Connecting to Dobot and homing (this takes 20-40s)...")
dobotArm.initialize_robot(api)
print("[INIT] Dobot ready. Opening gripper.")
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

# Find where to drop (tray, or fixed fallback).
drop_zone = phase_detect_tray()

while machine_state == "scanning target":
    pick_target = phase_detect_targets()
    if pick_target is not None:
        # Now that we have the blocks locked, turn hand detection ON
        # before the arm starts moving.
        phase_arm_hand_detection()
        next_state()

while machine_state == "pick place":
    completed = phase_execute_batch(api, pick_target, drop_zone)
    if completed:
        next_state()
    else:
        break

cap.release()
cv2.destroyAllWindows()
