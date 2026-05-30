import cv2
import numpy as np
import mediapipe as mp
# ── Initialize MediaPipe Hand + Pose Detection ────────────
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=4,
                           model_complexity=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
# Pose is used ONLY to get elbow + wrist positions (the arm/forearm).
# Face, torso, and legs are intentionally ignored to reduce visual noise
# and false detections.
pose = mp_pose.Pose(model_complexity=0,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)
# ── Define Zones ──────────────────────────────────────────
# Handoff Zone is fixed (a spot the worker reaches into).
HANDOFF_ZONE = (380, 260, 640, 480)

# CAUTION_ZONE: LARGER static rectangle covering the full robot + arm reach.
# Tune these (x1, y1, x2, y2) to match where the robot sits in your camera view.
# Based on your screenshot the robot is centered around (700-1100, 400-1000).
CAUTION_ZONE = (550, 350, 1250, 1050)

# DANGER_ZONE: SMALLER dynamic rectangle, follows the white robot arm.
# This static value is only the fallback if arm detection fails.
DANGER_ZONE  = (700, 500, 1100, 950)

# ── Dynamic arm detection settings ────────────────────────
# The Dobot Magician's arm is mostly WHITE/light gray. We threshold for
# bright, low-saturation pixels and only search INSIDE the CAUTION zone
# so other white things (keyboards, paper) in the frame don't trip us.
ARM_HSV_LOWER = (0,   0,   170)     # bright pixels
ARM_HSV_UPPER = (180, 60,  255)     # low saturation (whitish)
ARM_MIN_AREA  = 2000                # smallest blob (in px) to count as the arm
DANGER_PADDING = 25                 # tight buffer around the arm tip
BBOX_EMA_ALPHA = 0.3                # smoothing factor (higher = follows faster)
smoothed_danger_bbox = None

# ── Calibration settings ──────────────────────────────────
CALIBRATION_SECONDS = 5.0           # auto-fit zones for the first N seconds
CAUTION_PAD = 120                   # how much bigger CAUTION is vs the robot footprint
DANGER_FALLBACK_PAD = 40            # padding for the fallback danger box

# ── Manual-edit state (driven by mouse + keyboard) ────────
# edit_target: None | "CAUTION" | "DANGER". When set, click the two opposite
# corners of the new rectangle. corner1 stores the first click; the second
# click commits the box.
edit_target = None
corner1 = None             # first clicked corner
mouse_pos = (0, 0)         # current cursor position (for live preview)
# Move mode: drag a whole zone without resizing it.
move_target = None         # None | "HANDOFF" | "DANGER" | "CAUTION"
moving = False             # True while the mouse button is held during a move
move_grab_off = (0, 0)     # offset from box top-left to the grab point

def _zone_for(target):
    return {"CAUTION": CAUTION_ZONE, "DANGER": DANGER_ZONE,
            "HANDOFF": HANDOFF_ZONE}[target]

def on_mouse(event, x, y, flags, param):
    global edit_target, corner1, mouse_pos
    global move_target, moving, move_grab_off
    global CAUTION_ZONE, DANGER_ZONE, HANDOFF_ZONE

    mouse_pos = (x, y)

    # ── MOVE MODE: drag the whole box ─────────────────────
    if move_target is not None:
        zx1, zy1, zx2, zy2 = _zone_for(move_target)
        if event == cv2.EVENT_LBUTTONDOWN and zx1 <= x <= zx2 and zy1 <= y <= zy2:
            moving = True
            move_grab_off = (x - zx1, y - zy1)
        elif event == cv2.EVENT_MOUSEMOVE and moving:
            w_box, h_box = zx2 - zx1, zy2 - zy1
            nx1, ny1 = x - move_grab_off[0], y - move_grab_off[1]
            new_rect = (nx1, ny1, nx1 + w_box, ny1 + h_box)
            if move_target == "CAUTION":
                CAUTION_ZONE = new_rect
            elif move_target == "DANGER":
                DANGER_ZONE = new_rect
            elif move_target == "HANDOFF":
                HANDOFF_ZONE = new_rect
        elif event == cv2.EVENT_LBUTTONUP and moving:
            moving = False
            move_target = None       # finish move after one drag
        return

    # ── RESIZE/REDRAW MODE: click two corners ─────────────
    if edit_target is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if corner1 is None:
            corner1 = (x, y)                 # first corner
        else:
            x1, y1 = corner1                 # second corner → commit
            rect = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            if rect[2] - rect[0] > 15 and rect[3] - rect[1] > 15:
                if edit_target == "CAUTION":
                    CAUTION_ZONE = rect
                elif edit_target == "DANGER":
                    DANGER_ZONE = rect
                elif edit_target == "HANDOFF":
                    HANDOFF_ZONE = rect
            edit_target = None
            corner1 = None

def compute_arm_mask(frame):
    """Binary mask (uint8 0/255) of white robot pixels across the whole frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ARM_HSV_LOWER, ARM_HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),  np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask

def inflate_bbox(bbox, pad, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    return (max(0, x1 - pad),
            max(0, y1 - pad),
            min(frame_w, x2 + pad),
            min(frame_h, y2 + pad))

def clamp_bbox(bbox, container):
    """Constrain `bbox` to lie within `container`."""
    x1, y1, x2, y2 = bbox
    cx1, cy1, cx2, cy2 = container
    return (max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2))

def ema_bbox(old, new, alpha):
    if old is None:
        return new
    return tuple(int(alpha * n + (1 - alpha) * o) for o, n in zip(old, new))

def union_bbox(a, b):
    """Smallest box containing both a and b. Either may be None."""
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
# Pose landmarks we care about: ONLY elbows + wrists (the forearm).
# We deliberately exclude shoulders so the face/torso area doesn't get tracked.
# MediaPipe Pose indices: 13 = left elbow, 14 = right elbow,
#                         15 = left wrist, 16 = right wrist.
POSE_BODY_LANDMARKS = [13, 14, 15, 16]
# Forearm bones to draw: elbow → wrist on each side.
POSE_FOREARM_PAIRS = [(13, 15), (14, 16)]
# Minimum visibility for a pose landmark to be trusted.
POSE_VIS_THRESHOLD = 0.7
# Fingertip landmarks from MediaPipe Hands (thumb, index, middle, ring, pinky).
HAND_FINGERTIPS = [4, 8, 12, 16, 20]
# Which fingertip is the "primary" pointer we draw and report on.
POINTER_FINGERTIP = 8  # index finger tip
# Warning Zone: anywhere else in frame — robot SLOWS DOWN
# (no rectangle needed, it's the default when hand is detected outside danger zone)
def get_zone(cx, cy):
    """
    Returns which zone the given pixel is in.
    Priority: HANDOFF > DANGER (tight, around arm) > WARNING (caution box around robot) > NONE
    """
    hx1, hy1, hx2, hy2 = HANDOFF_ZONE
    dx1, dy1, dx2, dy2 = DANGER_ZONE
    cx1, cy1, cx2, cy2 = CAUTION_ZONE
    if hx1 < cx < hx2 and hy1 < cy < hy2:
        return "HANDOFF"
    if dx1 < cx < dx2 and dy1 < cy < dy2:
        return "DANGER"
    if cx1 < cx < cx2 and cy1 < cy < cy2:
        return "WARNING"
    return "NONE"
# ── Robot speed control (placeholder for real robot integration) ──
def set_robot_speed(zone):
    """
    In the final system, this will send commands to the Dobot arm.
    For now, it just prints the action so we can verify the logic.
    """
    if zone == "NONE":
        print("Robot: FULL SPEED")
    elif zone == "WARNING":
        print("Robot: SLOWING DOWN (50% speed)")
    elif zone == "DANGER":
        print("Robot: STOPPED")
    elif zone == "HANDOFF":
        print("Robot: DELIVERING TO WORKER")
# ── Open Camera ───────────────────────────────────────────
# Try indices 0 and 1 so it works whether the laptop's built-in webcam
# or an external camera (e.g. Orbbec) is the default.
cap = None
for idx in (0, 1):
    test = cv2.VideoCapture(idx)
    if test.isOpened():
        cap = test
        print(f"Camera opened at index {idx}.")
        break
    test.release()

if cap is None:
    print("ERROR: No camera could be opened (tried indices 0 and 1).")
    raise SystemExit(1)

# Warm up — first few reads after open often fail on macOS while the OS
# negotiates camera access. Retry briefly before giving up.
import time as _time
warmup_ok = False
for _ in range(30):
    ret, _frame = cap.read()
    if ret and _frame is not None:
        warmup_ok = True
        break
    _time.sleep(0.1)

if not warmup_ok:
    print("ERROR: Camera opened but no frames received.")
    print("  → On macOS, grant camera permission to your terminal app:")
    print("    System Settings → Privacy & Security → Camera → enable your terminal/IDE,")
    print("    then fully quit and reopen it.")
    cap.release()
    raise SystemExit(1)

print("Camera started.")
print("First 5s: auto-drawing zones around the robot.")
print("After that, controls (click two opposite corners to redraw a zone):")
print("  [D] DANGER zone   [C] CAUTION zone   [H] HANDOFF zone")
print("  [R] recalibrate   [Q] quit")

WINDOW = "Hand Detection — Safety Zone System"
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)

prev_zone = "NONE"

# Calibration state
phase = "CALIBRATING"
calib_start = _time.time()
calib_accum = None        # float accumulator of arm-mask pixels over time
calib_frames = 0          # how many frames we've accumulated
# A pixel is considered "robot" if it was detected as arm in at least this
# fraction of calibration frames — rejects transient noise/glare.
CALIB_PERSIST_FRAC = 0.35

# Whole-frame undistort? (the search detection happens inside CAUTION already)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ── CALIBRATION PHASE: auto-fit zones to the robot ──────
    if phase == "CALIBRATING":
        elapsed = _time.time() - calib_start

        # Accumulate the arm mask each frame (robust to one-off noise)
        mask = compute_arm_mask(frame)
        if calib_accum is None:
            calib_accum = np.zeros((h, w), dtype=np.float32)
        calib_accum += (mask > 0)
        calib_frames += 1

        # Live preview: pixels seen as robot in >=35% of frames so far
        if calib_frames > 0:
            persist = (calib_accum >= max(1, int(CALIB_PERSIST_FRAC * calib_frames)))
            persist_u8 = persist.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(persist_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            big = [c for c in cnts if cv2.contourArea(c) >= ARM_MIN_AREA]
            if big:
                xs = np.concatenate([c[:, 0, 0] for c in big])
                ys = np.concatenate([c[:, 0, 1] for c in big])
                preview = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                cv2.rectangle(frame, preview[:2], preview[2:], (0, 255, 255), 2)

        secs_left = max(0.0, CALIBRATION_SECONDS - elapsed)
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.putText(frame, f"CALIBRATING... {secs_left:0.1f}s", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "Move the robot arm through its range of motion",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if elapsed >= CALIBRATION_SECONDS:
            robot_bbox = None
            if calib_frames > 0 and calib_accum is not None:
                persist = (calib_accum >= max(1, int(CALIB_PERSIST_FRAC * calib_frames)))
                persist_u8 = persist.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(persist_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                big = [c for c in cnts if cv2.contourArea(c) >= ARM_MIN_AREA]
                if big:
                    xs = np.concatenate([c[:, 0, 0] for c in big])
                    ys = np.concatenate([c[:, 0, 1] for c in big])
                    robot_bbox = (int(xs.min()), int(ys.min()),
                                  int(xs.max()), int(ys.max()))

            if robot_bbox is not None:
                DANGER_ZONE  = inflate_bbox(robot_bbox, DANGER_FALLBACK_PAD, w, h)
                CAUTION_ZONE = inflate_bbox(robot_bbox, CAUTION_PAD, w, h)
                print(f"Calibrated over {calib_frames} frames. "
                      f"DANGER={DANGER_ZONE}  CAUTION={CAUTION_ZONE}")
            else:
                print("Calibration: no stable robot region found, keeping defaults.")
            print("Auto-drawing stopped. Press D / C / H to redraw a zone by hand.")
            phase = "RUNNING"
        continue

    # ── Big static CAUTION zone (around full robot footprint) ──────
    cx1, cy1, cx2, cy2 = CAUTION_ZONE
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
    cv2.putText(frame, "CAUTION ZONE", (cx1, cy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # ── Static DANGER zone (set during calibration, editable by mouse) ─
    dx1, dy1, dx2, dy2 = DANGER_ZONE

    # Red filled DANGER box (with transparency)
    overlay = frame.copy()
    cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
    cv2.putText(frame, "DANGER ZONE", (dx1, max(dy1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    # ── Draw Handoff Zone (green box) ─────────────────────
    hx1, hy1, hx2, hy2 = HANDOFF_ZONE
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
    cv2.putText(frame, "HANDOFF ZONE", (hx1, hy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    # ── Hand + Pose Detection ─────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    zone = "NONE"
    status_text = "Workspace Clear - Robot Running"
    status_color = (0, 255, 0)
    all_zones = []
    person_detected = False

    # --- Pose: ONLY use forearm landmarks (elbow + wrist) ---
    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks.landmark

        # Draw forearm bones (elbow → wrist) if both endpoints are confident
        for elbow_i, wrist_i in POSE_FOREARM_PAIRS:
            e, wr = lms[elbow_i], lms[wrist_i]
            if e.visibility < POSE_VIS_THRESHOLD or wr.visibility < POSE_VIS_THRESHOLD:
                continue
            ex, ey = int(e.x * w),  int(e.y * h)
            wx, wy = int(wr.x * w), int(wr.y * h)
            cv2.line(frame, (ex, ey), (wx, wy), (0, 200, 255), 3)
            person_detected = True

        # Zone check on elbow + wrist points
        for idx in POSE_BODY_LANDMARKS:
            plm = lms[idx]
            if plm.visibility < POSE_VIS_THRESHOLD:
                continue
            px, py = int(plm.x * w), int(plm.y * h)
            all_zones.append(get_zone(px, py))
            cv2.circle(frame, (px, py), 8, (0, 200, 255), -1)

    # --- Hands: detect each fingertip ---
    if hand_result.multi_hand_landmarks:
        for lm in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            # Check every fingertip for zone membership (not just one point)
            for tip_idx in HAND_FINGERTIPS:
                tx = int(lm.landmark[tip_idx].x * w)
                ty = int(lm.landmark[tip_idx].y * h)
                all_zones.append(get_zone(tx, ty))
                # Highlight the primary pointer (index fingertip) larger
                if tip_idx == POINTER_FINGERTIP:
                    cv2.circle(frame, (tx, ty), 12, (255, 255, 255), -1)
                    cv2.circle(frame, (tx, ty),  6, (0, 0, 255),   -1)
                else:
                    cv2.circle(frame, (tx, ty), 5, (255, 255, 255), -1)

    # Resolve the most-dangerous zone seen this frame
    if "DANGER" in all_zones:
        zone = "DANGER"
    elif "HANDOFF" in all_zones:
        zone = "HANDOFF"
    elif "WARNING" in all_zones:
        zone = "WARNING"

    # Display feedback
    if zone == "HANDOFF":
        status_text  = "FINGER IN HANDOFF ZONE - Delivering to worker"
        status_color = (0, 255, 255)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 3)
    elif zone == "DANGER":
        status_text  = "DANGER - Person/finger in workspace - Robot STOPPED"
        status_color = (0, 0, 255)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 4)
    elif zone == "WARNING":
        status_text  = "WARNING - Person detected - Robot Slowing"
        status_color = (0, 165, 255)
    elif person_detected:
        status_text = "Person detected (outside zones) - Robot Running"
        status_color = (0, 200, 0)
    # Only print to terminal when zone changes (avoid spam)
    if zone != prev_zone:
        set_robot_speed(zone)
        prev_zone = zone
    # ── Status bar at top ─────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    # ── Edit mode: show prompt + live preview of the new box ──
    if edit_target is not None:
        if corner1 is None:
            hint = f"EDIT {edit_target}: click FIRST corner  (ESC cancels)"
        else:
            hint = f"EDIT {edit_target}: click SECOND corner  (ESC cancels)"
            # live preview from first corner to cursor
            cv2.rectangle(frame, corner1, mouse_pos, (255, 255, 0), 2)
        cv2.putText(frame, hint, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ── Move mode prompt ──────────────────────────────────
    if move_target is not None:
        cv2.putText(frame, f"MOVE {move_target}: click inside the box and drag  (ESC cancels)",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ── Controls hint + robot name at bottom ──────────────
    cv2.putText(frame, "[D]anger [C]aution [H]andoff redraw | [M]ove handoff | [R]ecal [Q]uit",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "Hana | Collaborative Robot | 2026 Toyota Innovation Challenge",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        edit_target = "CAUTION"
        corner1 = None
    elif key == ord('d'):
        edit_target = "DANGER"
        corner1 = None
    elif key == ord('h'):
        edit_target = "HANDOFF"
        corner1 = None
    elif key == ord('m'):
        move_target = "HANDOFF"
        edit_target = None
        corner1 = None
    elif key == ord('r'):
        phase = "CALIBRATING"
        calib_start = _time.time()
        calib_accum = None
        calib_frames = 0
        print("Recalibrating...")
    elif key == 27:  # ESC cancels an edit
        edit_target = None
        corner1 = None
        move_target = None
        moving = False

cap.release()
cv2.destroyAllWindows()
print("Program exited.")
