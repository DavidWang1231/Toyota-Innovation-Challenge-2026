import cv2
import mediapipe as mp
# ── Initialize MediaPipe Hand Detection ───────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
# ── Define Zones ──────────────────────────────────────────
# Handoff Zone: bottom-right corner (worker reaches here to receive part)
HANDOFF_ZONE = (400, 300, 620, 460)
# Danger Zone: center of frame (near robot working area) — robot STOPS
DANGER_ZONE = (200, 150, 440, 390)
# Warning Zone: anywhere else in frame — robot SLOWS DOWN
# (no rectangle needed, it's the default when hand is detected outside danger zone)
def get_zone(cx, cy):
    """
    Returns which zone the palm center is in.
    Priority: HANDOFF > DANGER > WARNING > NONE
    """
    hx1, hy1, hx2, hy2 = HANDOFF_ZONE
    dx1, dy1, dx2, dy2 = DANGER_ZONE
    if hx1 < cx < hx2 and hy1 < cy < hy2:
        return "HANDOFF"
    elif dx1 < cx < dx2 and dy1 < cy < dy2:
        return "DANGER"
    else:
        return "WARNING"
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
cap = cv2.VideoCapture(0)
print("Camera started. Press Q to quit.")
prev_zone = "NONE"
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    # ── Draw Danger Zone (red dashed box) ─────────────────
    dx1, dy1, dx2, dy2 = DANGER_ZONE
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
    cv2.putText(frame, "DANGER ZONE", (dx1, dy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    # ── Draw Handoff Zone (green box) ─────────────────────
    hx1, hy1, hx2, hy2 = HANDOFF_ZONE
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
    cv2.putText(frame, "HANDOFF ZONE", (hx1, hy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    # ── Hand Detection ────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    zone = "NONE"
    status_text = "No Hand - Robot Running"
    status_color = (0, 255, 0) # Green = all clear
    if result.multi_hand_landmarks:
        # Collect all zones from all detected hands
        all_zones = []

        for lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            cx = int(lm.landmark[9].x * w)
            cy = int(lm.landmark[9].y * h)
            all_zones.append(get_zone(cx, cy))
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)

        # Pick the most dangerous zone
        if "DANGER" in all_zones:
            zone = "DANGER"
        elif "WARNING" in all_zones:
            zone = "WARNING"
        elif "HANDOFF" in all_zones:
            zone = "HANDOFF"

        # Update display based on final zone
        if zone == "HANDOFF":
            status_text  = "HAND IN HANDOFF ZONE - Delivering to worker"
            status_color = (0, 255, 255)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 3)
        elif zone == "DANGER":
            status_text  = "DANGER - Robot STOPPED"
            status_color = (0, 0, 255)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 4)
        elif zone == "WARNING":
            status_text  = "WARNING - Robot Slowing Down"
            status_color = (0, 165, 255)
    # Only print to terminal when zone changes (avoid spam)
    if zone != prev_zone:
        set_robot_speed(zone)
        prev_zone = zone
    # ── Status bar at top ─────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    # ── Robot name at bottom ──────────────────────────────
    cv2.putText(frame, "Hana | Collaborative Robot | 2026 Toyota Innovation Challenge",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.imshow("Hand Detection — Safety Zone System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Program exited.")