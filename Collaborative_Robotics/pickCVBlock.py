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
Z_PICK = -25
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

# ---- Red object detection (tuned for distance) ----
RED_HSV_LOWER_1 = (0,   70, 40)
RED_HSV_UPPER_1 = (10, 255, 255)
RED_HSV_LOWER_2 = (170, 70, 40)
RED_HSV_UPPER_2 = (180, 255, 255)
TARGET_MIN_AREA = 20         # ~5x5 px - small/distant objects still counted
TARGET_MAX_AREA = 20000      # rejects huge reds (clothing, big background items)

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
# Adjust these rectangles to match where your robot actually sits in the
# camera view. Coordinates are (x1, y1, x2, y2) in pixels.
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
    Find red blobs inside (CAUTION zone MINUS DANGER zone).
    Looser HSV + OPEN+CLOSE morphology = works at distance.
    Returns list of (robot_x, robot_y).
    """
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(RED_HSV_LOWER_1), np.array(RED_HSV_UPPER_1)) + \
           cv2.inRange(hsv, np.array(RED_HSV_LOWER_2), np.array(RED_HSV_UPPER_2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.bitwise_and(mask, _work_area_mask(frame.shape))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
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
        result.append((rx, ry))
        if draw_on is not None:
            cv2.drawContours(draw_on, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(draw_on, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(draw_on, f"RED({int(area)})", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result

# ─────────────────────────────────────────
# TRAY DETECTION (drop target)
# ─────────────────────────────────────────
def _tray_mask(frame):
    """Bright-pixel mask restricted to CAUTION zone. Exposed for debug window."""
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, mask = cv2.threshold(gray, TRAY_BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
    work = np.zeros_like(mask)
    cx1, cy1, cx2, cy2 = CAUTION_ZONE
    work[max(0,cy1):min(fh,cy2), max(0,cx1):min(fw,cx2)] = 255
    return cv2.bitwise_and(mask, work)

def find_tray(frame, draw_on=None):
    """Most circular bright blob in CAUTION zone. Returns (robot_x, robot_y) or None."""
    mask = _tray_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (TRAY_MIN_AREA <= area <= TRAY_MAX_AREA):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < TRAY_MIN_CIRCULARITY:
            continue
        score = area * circ
        if score > best_score:
            best_score, best = score, (cnt, circ)
    if best is None:
        return None
    cnt, circ = best
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    rx, ry = pixel_to_robot(cx, cy, H_matrix)
    if draw_on is not None:
        cv2.circle(draw_on, (cx, cy), int(radius), (200, 200, 200), 2)
        cv2.circle(draw_on, (cx, cy), 6, (255, 0, 255), -1)
        cv2.putText(draw_on, f"TRAY ({circ:.2f})", (cx - 50, cy - int(radius) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    return (rx, ry)

def phase_detect_tray():
    """Lock the tray over a few stable frames; fall back to fixed drop on timeout."""
    if not USE_TRAY_DETECTION:
        print(f"[PHASE 1] Tray detection OFF. Using fixed drop {DROP_FALLBACK_ROBOT_XY}.")
        return [DROP_FALLBACK_ROBOT_XY]
    print("\n[PHASE 1] Looking for the tray...")
    stable, last_xy, attempts = 0, None, 0
    while attempts < 150:                     # ~5s at 30fps
        attempts += 1
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = frame.copy()
        draw_zones(display)
        if TRAY_DEBUG_MASK:
            cv2.imshow("Tray Mask (debug)", _tray_mask(frame))
        tray = find_tray(frame, draw_on=display)
        if tray is not None:
            rx, ry = tray
            if last_xy and abs(rx-last_xy[0]) < 15 and abs(ry-last_xy[1]) < 15:
                stable += 1
            else:
                stable = 0
            last_xy = (rx, ry)
            cv2.putText(display, f"TRAY LOCK: {stable}/15", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            if stable >= 15:
                print(f"Tray locked at robot ({rx:.1f}, {ry:.1f}).")
                return [(rx, ry)]
        else:
            cv2.putText(display, "Searching for tray...", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow("Detection", display)
        cv2.waitKey(1)
    print(f"No tray locked. Using fallback drop {DROP_FALLBACK_ROBOT_XY}.")
    return [DROP_FALLBACK_ROBOT_XY]

# ─────────────────────────────────────────
# PHASE 2: DETECT TARGETS
# ─────────────────────────────────────────
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets...")
    stability_counter = 0
    last_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        draw_zones(display_frame)

        current_list = find_red_objects(frame, draw_on=display_frame)

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter/STABILITY_LIMIT)*100)
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} targets.")
            return current_list

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
        next_state()

while machine_state == "pick place":
    completed = phase_execute_batch(api, pick_target, drop_zone)
    if completed:
        next_state()
    else:
        break

cap.release()
cv2.destroyAllWindows()
