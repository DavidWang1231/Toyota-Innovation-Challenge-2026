"""
manualControl.py
-----------------
Keyboard manual control for the Toyota Innovation Challenge arm.

This matches YOUR project's API:
    import dobotArm
    import lib.DobotDllType as dType
    api = dType.load()
    dobotArm.move_to_xyz(api, x, y, z, rHead)   # absolute move
    dType.GetPose(api) -> [x, y, z, rHead, ...] # read position

HOW TO USE
  While any detection phase is running, press 'M' to drop into manual mode.
  Drive the arm with the keyboard. Press 'M' again to return to auto, or
  Esc to quit the whole program.

KEY LAYOUT
  M      -> exit manual mode (back to auto)
  W / S  -> move forward / backward   (X axis)
  A / D  -> move left / right         (Y axis)
  Q / E  -> move up / down            (Z axis)
  Z / C  -> rotate gripper - / +      (rHead, clamped to -90..90)
  Space  -> open / close gripper
  H      -> go to home position
  Esc    -> quit program
"""

import dobotArm
import lib.DobotDllType as dType
import cv2


# ---------------------------------------------------------------------------
# How far the arm moves per key press. Start small for safety.
# ---------------------------------------------------------------------------
STEP_XY = 30     # mm per press for W/S/A/D
STEP_Z  = 30     # mm per press for Q/E
STEP_R  = 45     # degrees per press for Z/C

# ---------------------------------------------------------------------------
# Soft limits - stop the arm from being driven somewhere that hits the table
# or exceeds its reach. Widen these if the arm refuses to go where you want.
# (Defaults are conservative for a Dobot Magician on a table.)
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 100, 320
Y_MIN, Y_MAX = -150, 200
Z_MIN, Z_MAX = -45, 120
R_MIN, R_MAX = -90, 90


def _clamp(value, low, high):
    return max(low, min(high, value))


def _read_pose(api):
    """Return current (x, y, z, rHead) or None on failure."""
    try:
        pose = dType.GetPose(api)
        return pose[0], pose[1], pose[2], pose[3]
    except Exception as err:
        print(f"[ERROR reading pose] {err}")
        return None


def _draw_overlay(frame, x, y, z, r, gripper_closed):
    """Draw the manual-mode banner, live coordinates, and key hints."""
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
    cv2.putText(frame, "MANUAL CONTROL", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Live position readout (top right)
    grip = "CLOSED" if gripper_closed else "OPEN"
    pos_text = f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f} R:{r:.0f}  Grip:{grip}"
    cv2.putText(frame, pos_text, (w - 360, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Key hints (bottom)
    hints = [
        "W/S fwd/back   A/D left/right   Q/E up/down",
        "Z/C rotate   SPACE grip   H home   M auto   ESC quit",
    ]
    cv2.rectangle(frame, (0, h - 50), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, hints[0], (12, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, hints[1], (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return frame


def manual_control_loop(api, cap, window_name="Detection"):
    """
    Take over control of the arm with the keyboard.

    Returns:
        "auto" -> user pressed M, caller should resume automatic mode
        "quit" -> user pressed Esc, caller should shut everything down
    """
    print("\n[MANUAL] Manual control ON. WASD/QE/ZC to move, M to exit, Esc to quit.")

    gripper_closed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            # If the camera frame fails, keep going with a blank image so
            # the keyboard still works.
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pose = _read_pose(api)
        if pose is None:
            x = y = z = r = 0.0
        else:
            x, y, z, r = pose

        _draw_overlay(frame, x, y, z, r, gripper_closed)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10) & 0xFF   # 10ms wait = responsive but not maxed out

        if key == 255:                 # no key pressed this frame
            continue

        # ---- Mode / program keys ----
        if key == ord('m'):
            print("[MANUAL] Returning to AUTO mode.")
            return "auto"
        if key == 27:                  # Esc
            print("[MANUAL] Esc pressed - quitting.")
            return "quit"
        if key == ord('h'):
            print("[MANUAL] Going home.")
            dobotArm.move_to_home(api)
            continue

        # ---- Movement keys (need a valid pose) ----
        if pose is None:
            print("[MANUAL] Skipping move - could not read arm position.")
            continue

        if key == ord('w'):
            dobotArm.move_to_xyz(api, _clamp(x + STEP_XY, X_MIN, X_MAX), y, z, r)
        elif key == ord('s'):
            dobotArm.move_to_xyz(api, _clamp(x - STEP_XY, X_MIN, X_MAX), y, z, r)
        elif key == ord('a'):
            dobotArm.move_to_xyz(api, x, _clamp(y + STEP_XY, Y_MIN, Y_MAX), z, r)
        elif key == ord('d'):
            dobotArm.move_to_xyz(api, x, _clamp(y - STEP_XY, Y_MIN, Y_MAX), z, r)
        elif key == ord('q'):
            dobotArm.move_to_xyz(api, x, y, _clamp(z + STEP_Z, Z_MIN, Z_MAX), r)
        elif key == ord('e'):
            dobotArm.move_to_xyz(api, x, y, _clamp(z - STEP_Z, Z_MIN, Z_MAX), r)
        elif key == ord('z'):
            dobotArm.move_to_xyz(api, x, y, z, _clamp(r - STEP_R, R_MIN, R_MAX))
        elif key == ord('c'):
            dobotArm.move_to_xyz(api, x, y, z, _clamp(r + STEP_R, R_MIN, R_MAX))
        elif key == ord(' '):          # spacebar toggles gripper
            gripper_closed = not gripper_closed
            if gripper_closed:
                dobotArm.close_gripper(api)
                print("[MANUAL] Gripper CLOSED")
            else:
                dobotArm.open_gripper(api)
                dobotArm.stop_pump(api)
                print("[MANUAL] Gripper OPEN")
