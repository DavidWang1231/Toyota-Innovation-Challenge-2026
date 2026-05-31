"""
manualControl.py
-----------------
Keyboard manual control for the Toyota Innovation Challenge arm.

WHY THE REWRITE (v3 - JOG mode)
  v1 used blocking PTP — laggy, dropped keys.
  v2 used queued PTP — every step still accel/decel ramps, felt stuck.
  v3 uses JOG — the Dobot's native continuous-jog API. You tell the arm
  "start moving +X at velocity V" and it cruises smoothly until told to
  stop. No per-step acceleration ramp = truly smooth motion.

  OpenCV gives us key-press events with OS auto-repeat (~30 Hz) but NO
  key-release. So we use a WATCHDOG: every keypress resets a timer; if
  no key arrives for KEY_HOLD_TIMEOUT seconds (default 0.12s) we assume
  the user released and send a JOG-stop. Auto-repeat at 30Hz delivers a
  key every ~33ms, so 120ms is a safe cushion.

HOW TO USE
  Press 'M' to drop into manual mode. Drive with the keyboard. Press 'M'
  again to return to auto, or Esc to quit.

KEY LAYOUT
  M      -> exit manual mode (back to auto)
  W / S  -> jog forward / backward     (X axis)
  A / D  -> jog left / right           (Y axis)
  Q / E  -> jog up / down              (Z axis)
  Z / C  -> rotate gripper - / +       (rHead)
  Space  -> open / close gripper
  H      -> go to home position
  Esc    -> quit program
"""

import time
import cv2
import numpy as np

import dobotArm
import lib.DobotDllType as dType


# ---------------------------------------------------------------------------
# Jog velocities (mm/s for XYZ, deg/s for R). Bump UP for snappier feel,
# DOWN for finer hand-positioning.
# ---------------------------------------------------------------------------
JOG_VEL_XY = 120     # mm/s for X and Y jog
JOG_VEL_Z  = 100     # mm/s for Z jog
JOG_VEL_R  = 90      # deg/s for R jog
# Acceleration in mm/s^2 (deg/s^2 for R). High = snaps to cruise speed
# quickly. The Magician can handle 200 comfortably.
JOG_ACC_XY = 200
JOG_ACC_Z  = 200
JOG_ACC_R  = 200

# Overall velocity/accel ratio (0..100 %). Acts like a master throttle on
# top of the per-axis values above.
JOG_VEL_RATIO = 100
JOG_ACC_RATIO = 100

# Watchdog: if no key event arrives for this many seconds, send JOG stop.
# Must be > the OS key-repeat interval (~33 ms at "Fast" repeat rate).
# Larger = arm coasts longer after release. Smaller = risk of stutter
# during a held key if the OS misses a repeat.
KEY_HOLD_TIMEOUT = 0.12

# Map from key character → (isJoint, JOG cmd code).
# isJoint=0 means coordinate (Cartesian) jog.
# cmd codes from the Dobot manual:
#   1=+X 2=-X 3=+Y 4=-Y 5=+Z 6=-Z 7=+R 8=-R
KEY_TO_JOG = {
    ord('w'): (0, 1),   # +X (forward, away from base)
    ord('s'): (0, 2),   # -X
    ord('a'): (0, 3),   # +Y (left)
    ord('d'): (0, 4),   # -Y
    ord('q'): (0, 5),   # +Z (up)
    ord('e'): (0, 6),   # -Z
    ord('z'): (0, 7),   # +R
    ord('c'): (0, 8),   # -R
}

JOG_STOP_CMD = 0        # tells the arm to stop the current jog


def _read_pose(api):
    try:
        pose = dType.GetPose(api)
        return pose[0], pose[1], pose[2], pose[3]
    except Exception as err:
        print(f"[ERROR reading pose] {err}")
        return None


def _draw_overlay(frame, x, y, z, r, gripper_closed, jog_label):
    """Top banner + live pose + active-jog indicator + key hints."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
    cv2.putText(frame, "MANUAL CONTROL (JOG)", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    grip = "CLOSED" if gripper_closed else "OPEN"
    pos_text = f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f} R:{r:.0f}  Grip:{grip}  JOG:{jog_label}"
    cv2.putText(frame, pos_text, (w - 480, 27),
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


def _configure_jog(api):
    """One-time setup of jog speed/accel before any SetJOGCmd."""
    # Per-axis Cartesian jog speeds (X, Y, Z, R)
    dType.SetJOGCoordinateParams(api,
                                 JOG_VEL_XY, JOG_ACC_XY,
                                 JOG_VEL_XY, JOG_ACC_XY,
                                 JOG_VEL_Z,  JOG_ACC_Z,
                                 JOG_VEL_R,  JOG_ACC_R,
                                 isQueued=0)
    # Per-joint jog speeds — set even though we use coordinate jog, so the
    # firmware is fully primed.
    dType.SetJOGJointParams(api,
                            JOG_VEL_R, JOG_ACC_R,
                            JOG_VEL_R, JOG_ACC_R,
                            JOG_VEL_R, JOG_ACC_R,
                            JOG_VEL_R, JOG_ACC_R,
                            isQueued=0)
    # Master throttle (0..100 %)
    dType.SetJOGCommonParams(api, JOG_VEL_RATIO, JOG_ACC_RATIO, isQueued=0)


def _jog_stop(api):
    """Send the all-stop jog command. Safe to call any time."""
    try:
        dType.SetJOGCmd(api, 0, JOG_STOP_CMD, isQueued=0)
    except Exception as err:
        print(f"[MANUAL] jog stop failed: {err}")


def _jog_start(api, is_joint, cmd_code):
    try:
        dType.SetJOGCmd(api, is_joint, cmd_code, isQueued=0)
    except Exception as err:
        print(f"[MANUAL] jog start failed: {err}")


def manual_control_loop(api, cap, window_name="Detection"):
    """
    Take over control of the arm with the keyboard via JOG.

    Returns:
        "auto" -> user pressed M, caller should resume automatic mode
        "quit" -> user pressed Esc, caller should shut everything down
    """
    print("\n[MANUAL] JOG mode ON. WASD/QE/ZC to drive, M to exit, Esc to quit.")

    # Make sure no leftover queued PTP commands run while we jog.
    try:
        dType.SetQueuedCmdStopExec(api)
        dType.SetQueuedCmdClear(api)
    except Exception as err:
        print(f"[MANUAL] queue clear failed: {err}")

    _configure_jog(api)
    _jog_stop(api)

    gripper_closed = False
    active_cmd = None          # currently-running (isJoint, cmd_code) or None
    last_key_time = 0.0        # wall time of last movement keypress
    last_key = None            # which key is currently being held

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            live = _read_pose(api)
            if live is not None:
                lx, ly, lz, lr = live
            else:
                lx = ly = lz = lr = 0.0

            jog_label = "—" if active_cmd is None else f"cmd{active_cmd[1]}"
            _draw_overlay(frame, lx, ly, lz, lr, gripper_closed, jog_label)
            cv2.imshow(window_name, frame)

            # 10 ms poll — fast enough to react instantly, slow enough that
            # the watchdog timer has meaningful granularity.
            key = cv2.waitKey(10) & 0xFF

            # ── Watchdog: if user released the key, stop the jog ─────────
            if active_cmd is not None:
                if time.time() - last_key_time > KEY_HOLD_TIMEOUT:
                    _jog_stop(api)
                    active_cmd = None
                    last_key = None

            if key == 255:
                continue   # no key this poll → just keep the overlay alive

            # ── Mode / program keys (always respected, stop jog first) ───
            if key == ord('m'):
                _jog_stop(api)
                print("[MANUAL] Returning to AUTO mode.")
                return "auto"
            if key == 27:
                _jog_stop(api)
                print("[MANUAL] Esc pressed — quitting.")
                return "quit"
            if key == ord('h'):
                _jog_stop(api)
                active_cmd = None
                print("[MANUAL] Going home.")
                dobotArm.move_to_home(api)   # blocking, fine for H
                continue
            if key == ord(' '):
                _jog_stop(api)
                active_cmd = None
                gripper_closed = not gripper_closed
                if gripper_closed:
                    dobotArm.close_gripper(api)
                    print("[MANUAL] Gripper CLOSED")
                else:
                    dobotArm.open_gripper(api)
                    dobotArm.stop_pump(api)
                    print("[MANUAL] Gripper OPEN")
                continue

            # ── Movement keys: start/refresh a jog ───────────────────────
            if key in KEY_TO_JOG:
                new_cmd = KEY_TO_JOG[key]
                # If we're already jogging in the SAME direction, just
                # refresh the watchdog (the arm is already moving — don't
                # restart it, that would cause a micro-stutter).
                if active_cmd != new_cmd:
                    # Different direction (or none active) → stop the old,
                    # start the new.
                    if active_cmd is not None:
                        _jog_stop(api)
                    _jog_start(api, new_cmd[0], new_cmd[1])
                    active_cmd = new_cmd
                last_key = key
                last_key_time = time.time()
    finally:
        # Whatever happens (return, exception, KeyboardInterrupt) make sure
        # the arm isn't left running.
        _jog_stop(api)
