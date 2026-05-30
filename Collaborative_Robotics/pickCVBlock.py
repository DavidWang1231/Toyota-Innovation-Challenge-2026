import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import mediapipe as mp

"""CONSTANTS"""
Z_SAFE = 40
Z_PICK = -25
STABILITY_LIMIT = 60

# ── Zones (scaled for 640x480 camera) ──
# HANDOFF: bottom-right corner, worker reaches here
HANDOFF_ZONE = (400, 300, 620, 460)
# CAUTION: large area around robot, robot SLOWS
CAUTION_ZONE = (120, 80, 520, 440)
# DANGER: tight area at robot center, robot STOPS
DANGER_ZONE  = (220, 150, 420, 360)

# Handoff Zone robot coordinates - adjust on competition day
HANDOFF_ROBOT_X = 180
HANDOFF_ROBOT_Y = -80

machine_state = "scanning target"  # skipping plate detection

# --- CAMERA & ROBOT INIT ---
api = dType.load()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

ret, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

# --- MEDIAPIPE INIT (hands + pose) ---
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=4, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose  = mp_pose.Pose(model_complexity=0,
                     min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Pose: elbows + wrists only (forearm). 13=L elbow,14=R elbow,15=L wrist,16=R wrist
POSE_BODY_LANDMARKS = [13, 14, 15, 16]
POSE_FOREARM_PAIRS  = [(13, 15), (14, 16)]
POSE_VIS_THRESHOLD  = 0.7
# Fingertips: thumb,index,middle,ring,pinky
HAND_FINGERTIPS  = [4, 8, 12, 16, 20]
POINTER_FINGERTIP = 8

# ─────────────────────────────────────────
# ZONE LOGIC
# ─────────────────────────────────────────
def get_zone(cx, cy):
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

def detect_zone(frame):
    """
    Full hand + forearm + fingertip detection.
    Returns (zone_string, annotated_frame).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    all_zones = []
    fh, fw = frame.shape[:2]

    # Pose forearm (elbow -> wrist)
    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks.landmark
        for elbow_i, wrist_i in POSE_FOREARM_PAIRS:
            e, wr = lms[elbow_i], lms[wrist_i]
            if e.visibility < POSE_VIS_THRESHOLD or wr.visibility < POSE_VIS_THRESHOLD:
                continue
            ex, ey = int(e.x*fw), int(e.y*fh)
            wx, wy = int(wr.x*fw), int(wr.y*fh)
            cv2.line(frame, (ex,ey), (wx,wy), (0,200,255), 3)
        for idx in POSE_BODY_LANDMARKS:
            plm = lms[idx]
            if plm.visibility < POSE_VIS_THRESHOLD:
                continue
            px, py = int(plm.x*fw), int(plm.y*fh)
            all_zones.append(get_zone(px, py))
            cv2.circle(frame, (px,py), 8, (0,200,255), -1)

    # Hands fingertips
    if hand_result.multi_hand_landmarks:
        for lm in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            for tip_idx in HAND_FINGERTIPS:
                tx = int(lm.landmark[tip_idx].x * fw)
                ty = int(lm.landmark[tip_idx].y * fh)
                all_zones.append(get_zone(tx, ty))
                if tip_idx == POINTER_FINGERTIP:
                    cv2.circle(frame, (tx,ty), 10, (255,255,255), -1)
                    cv2.circle(frame, (tx,ty), 5, (0,0,255), -1)
                else:
                    cv2.circle(frame, (tx,ty), 4, (255,255,255), -1)

    if "DANGER" in all_zones:
        return "DANGER", frame
    elif "HANDOFF" in all_zones:
        return "HANDOFF", frame
    elif "WARNING" in all_zones:
        return "WARNING", frame
    return "NONE", frame

def check_hands(frame):
    """Lightweight wrapper returning just the zone string."""
    zone, _ = detect_zone(frame)
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

        hsv  = cv2.cvtColor(cv2.GaussianBlur(frame,(3,3),0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,100,50]),   np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,100,50]), np.array([180,255,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    rx, ry = pixel_to_robot(cx, cy, H_matrix)
                    current_list.append((rx, ry))
                    cv2.drawContours(display_frame, [cnt], -1, (0,255,0), 2)
                    cv2.circle(display_frame, (cx,cy), 5, (0,0,255), -1)

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
    hsv  = cv2.cvtColor(cv2.GaussianBlur(frame,(3,3),0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,100,50]),   np.array([10,255,255])) + \
           cv2.inRange(hsv, np.array([170,100,50]), np.array([180,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                rx, ry = pixel_to_robot(cx, cy, H_matrix)
                result.append((rx, ry))
    return result

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
        if check_hands(frame) == "DANGER":
            wait_for_hand_clear(cap, map1, map2)

        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if check_hands(frame) == "DANGER":
            wait_for_hand_clear(cap, map1, map2)

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
        zone = check_hands(frame)
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
            if zone == "DANGER":
                wait_for_hand_clear(cap, map1, map2)
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    print("\nBatch complete.")
    return True

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

# Fixed drop zone (skip plate detection)
drop_zone = [(250, 0)]

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