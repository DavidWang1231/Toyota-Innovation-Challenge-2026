"""
manualControl.py
-----------------
Keyboard manual control for the Toyota Innovation Challenge arm.

WHY THE REWRITE
  The old version called the BLOCKING dobotArm.move_to_xyz on every key
  press. Each press waited ~0.5-1s for the move to finish, so quick taps
  were dropped and the arm felt laggy and unresponsive.

  This version uses QUEUED, NON-BLOCKING moves:
    - Each keypress sends a small PTP command with isQueued=1 and returns
      immediately. The Dobot's internal queue executes them back-to-back.
    - We track a "target pose" (tgt_x/y/z/r) so taps add to the target
      rather than reading the lagging actual pose.
    - The queue is capped at MAX_PENDING moves. Extra keys are dropped so
      the arm doesn't keep moving for a long time after you stop pressing.
    - Steps are SMALL so each tap is a small nudge.

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
import numpy as np


# ---------------------------------------------------------------------------
# How far the arm moves per key press. SMALL so each tap is a nudge — the
# queued model means held-down keys still travel quickly.
# ---------------------------------------------------------------------------
STEP_XY = 6      # mm per press for W/S/A/D
STEP_Z  = 6      # mm per press for Q/E
STEP_R  = 10     # degrees per press for Z/C

# Cap on how many moves can sit in the Dobot's queue. If you tap faster
# than the arm can execute, additional keys are DROPPED until the queue
# drains. Without this cap the arm would overshoot badly after you let go.
MAX_PENDING = 3

# ---------------------------------------------------------------------------
# Soft limits — stop the arm from being driven somewhere that hits the table
# or exceeds its reach. Widen these if the arm refuses to go where you want.
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


def _pending(api, last_sent_index):
    """How many of our queued moves haven't executed yet."""
    if last_sent_index is None:
        return 0
    try:
        current = dType.GetQueuedCmdCurrentIndex(api)[0]
    except Exception:
        return 0
    return max(0, last_sent_index - current)


def _queue_move(api, x, y, z, r):
    """Send a non-blocking PTP move. Returns the queue index of the command."""
    # PTPMOVJXYZMode matches what dobotArm.move_to_xyz uses, but isQueued=1
    # so we don't wait for it to finish before accepting the next key.
    idx = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                          x, y, z, r, isQueued=1)[0]
    return idx


def _draw_overlay(frame, x, y, z, r, gripper_closed, pending):
    """Draw the manual-mode banner, live coordinates, queue depth, key hints."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
    cv2.putText(frame, "MANUAL CONTROL", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    grip = "CLOSED" if gripper_closed else "OPEN"
    pos_text = f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f} R:{r:.0f}  Grip:{grip}  Q:{pending}"
    cv2.putText(frame, pos_text, (w - 420, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

    # Make sure the queue is actually executing — we send isQueued=1 commands.
    try:
        dType.SetQueuedCmdStartExec(api)
    except Exception as err:
        print(f"[MANUAL] Could not start queued exec: {err}")

    # Seed target pose from where the arm actually is right now. After this
    # we increment the TARGET (not the live pose) on each keypress, so taps
    # don't get lost while a previous move is still in flight.
    pose = _read_pose(api)
    if pose is None:
        print("[MANUAL] Could not read starting pose. Aborting manual mode.")
        return "auto"
    tgt_x, tgt_y, tgt_z, tgt_r = pose

    gripper_closed = False
    last_sent_index = None

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        live = _read_pose(api)
        if live is not None:
            lx, ly, lz, lr = live
        else:
            lx, ly, lz, lr = tgt_x, tgt_y, tgt_z, tgt_r

        pending = _pending(api, last_sent_index)
        _draw_overlay(frame, lx, ly, lz, lr, gripper_closed, pending)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF        # 1ms = maximum responsiveness

        if key == 255:                     # no key pressed this frame
            continue

        # ---- Mode / program keys (always respected) ----
        if key == ord('m'):
            print("[MANUAL] Returning to AUTO mode.")
            return "auto"
        if key == 27:                      # Esc
            print("[MANUAL] Esc pressed — quitting.")
            return "quit"
        if key == ord('h'):
            print("[MANUAL] Going home.")
            dobotArm.move_to_home(api)     # blocking is fine for H
            # After homing, resync target to the real pose.
            new_pose = _read_pose(api)
            if new_pose is not None:
                tgt_x, tgt_y, tgt_z, tgt_r = new_pose
            last_sent_index = None
            continue
        if key == ord(' '):
            gripper_closed = not gripper_closed
            if gripper_closed:
                dobotArm.close_gripper(api)
                print("[MANUAL] Gripper CLOSED")
            else:
                dobotArm.open_gripper(api)
                dobotArm.stop_pump(api)
                print("[MANUAL] Gripper OPEN")
            continue

        # ---- Movement keys: drop the press if the queue is full ----
        if pending >= MAX_PENDING:
            # Silently drop the key — prevents arm from chasing taps after
            # you let go. Uncomment to debug:
            # print(f"[MANUAL] Queue full ({pending}), key dropped.")
            continue

        moved = True
        if key == ord('w'):
            tgt_x = _clamp(tgt_x + STEP_XY, X_MIN, X_MAX)
        elif key == ord('s'):
            tgt_x = _clamp(tgt_x - STEP_XY, X_MIN, X_MAX)
        elif key == ord('a'):
            tgt_y = _clamp(tgt_y + STEP_XY, Y_MIN, Y_MAX)
        elif key == ord('d'):
            tgt_y = _clamp(tgt_y - STEP_XY, Y_MIN, Y_MAX)
        elif key == ord('q'):
            tgt_z = _clamp(tgt_z + STEP_Z, Z_MIN, Z_MAX)
        elif key == ord('e'):
            tgt_z = _clamp(tgt_z - STEP_Z, Z_MIN, Z_MAX)
        elif key == ord('z'):
            tgt_r = _clamp(tgt_r - STEP_R, R_MIN, R_MAX)
        elif key == ord('c'):
            tgt_r = _clamp(tgt_r + STEP_R, R_MIN, R_MAX)
        else:
            moved = False

        if moved:
            try:
                last_sent_index = _queue_move(api, tgt_x, tgt_y, tgt_z, tgt_r)
            except Exception as err:
                print(f"[MANUAL] Failed to queue move: {err}")
