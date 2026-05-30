"""
manualControl.py
-----------------
Adds keyboard manual control to the Toyota Innovation Challenge arm project.

WHAT IT DOES
  Press 'M' to switch the robot between AUTO mode (it runs the normal
  pick-and-place loop) and MANUAL mode (you drive the arm with the keyboard).

KEY LAYOUT
  M      -> toggle AUTO / MANUAL
  W / S  -> move arm forward / backward   (X axis)
  A / D  -> move arm left / right         (Y axis)
  Q / E  -> move arm up / down            (Z axis)
  Z / C  -> rotate gripper left / right   (R axis)
  Space  -> open / close gripper
  H      -> send arm to home position
  Esc    -> stop the program

HOW IT WORKS
  Your main script (pickCVBlock.py) already opens an OpenCV window and calls
  cv2.waitKey() every frame. This module reads that same key press and turns it
  into an arm command. No new libraries are needed.

  >>> IMPORTANT <<<
  The arm-control method names below (move_to, pose, open_gripper, close_gripper,
  home) must match the methods in YOUR dobotArm.py. If your file uses different
  names, change them ONLY inside this file, in the marked section near the bottom.
"""

# ---------------------------------------------------------------------------
# How far the arm moves on each key press.
# Start small and safe. Increase if movement feels too slow.
# ---------------------------------------------------------------------------
STEP_XY = 10    # millimetres moved per press for W/S/A/D
STEP_Z  = 10    # millimetres moved per press for Q/E
STEP_R  = 10    # degrees rotated per press for Z/C


class ManualController:
    def __init__(self, arm):
        """
        arm: your connected Dobot object (the same one pickCVBlock.py uses).
        """
        self.arm = arm
        self.manual_mode = False      # start in AUTO mode
        self.gripper_closed = False   # track gripper state for the toggle

    # -----------------------------------------------------------------------
    # MODE SWITCHING
    # -----------------------------------------------------------------------
    def toggle_mode(self):
        self.manual_mode = not self.manual_mode
        mode = "MANUAL" if self.manual_mode else "AUTO"
        print(f"[MODE] Switched to {mode}")
        return self.manual_mode

    # -----------------------------------------------------------------------
    # MAIN ENTRY POINT - call this every frame from pickCVBlock.py
    # Returns:
    #   "quit"  -> user pressed Esc, you should break out of the loop
    #   True    -> the key was a manual command and was handled
    #   False   -> the key was not one of ours (let auto logic continue)
    # -----------------------------------------------------------------------
    def handle_key(self, key):
        # cv2.waitKey returns -1 when no key is pressed
        if key == -1 or key == 255:
            return False

        # Esc quits no matter what mode we are in
        if key == 27:  # 27 is the Esc key
            self._safe_stop()
            return "quit"

        # M toggles mode no matter what mode we are in
        if key == ord('m'):
            self.toggle_mode()
            return True

        # H sends arm home in either mode
        if key == ord('h'):
            self._home()
            return True

        # Everything below only works in MANUAL mode
        if not self.manual_mode:
            return False

        # Read the arm's current position so we can nudge it
        x, y, z, r = self._get_xyzr()
        if x is None:
            print("[WARN] Could not read arm position.")
            return True

        if key == ord('w'):
            self._move(x + STEP_XY, y, z, r)
        elif key == ord('s'):
            self._move(x - STEP_XY, y, z, r)
        elif key == ord('a'):
            self._move(x, y + STEP_XY, z, r)
        elif key == ord('d'):
            self._move(x, y - STEP_XY, z, r)
        elif key == ord('q'):
            self._move(x, y, z + STEP_Z, r)
        elif key == ord('e'):
            self._move(x, y, z - STEP_Z, r)
        elif key == ord('z'):
            self._move(x, y, z, r - STEP_R)
        elif key == ord('c'):
            self._move(x, y, z, r + STEP_R)
        elif key == ord(' '):           # spacebar
            self._toggle_gripper()
        else:
            return False                # not one of our keys

        return True

    # -----------------------------------------------------------------------
    # ON-SCREEN STATUS - call this every frame to draw the mode on the camera
    # window so you (and the judges) can see which mode the robot is in.
    # -----------------------------------------------------------------------
    def draw_status(self, frame):
        import cv2
        if self.manual_mode:
            text, color = "MANUAL CONTROL", (0, 165, 255)   # orange
        else:
            text, color = "AUTO MODE", (0, 200, 0)          # green
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (40, 40, 40), -1)
        cv2.putText(frame, text, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    # =======================================================================
    # ADAPTER SECTION
    # If your dobotArm.py uses different method names, fix them HERE ONLY.
    # =======================================================================
    def _get_xyzr(self):
        """Return current (x, y, z, r). Adjust if your method name differs."""
        try:
            pose = self.arm.pose()          # pydobot returns a tuple/list
            return pose[0], pose[1], pose[2], pose[3]
        except Exception as err:
            print(f"[ERROR reading pose] {err}")
            return None, None, None, None

    def _move(self, x, y, z, r):
        """Move arm to absolute coordinates. Adjust if your method name differs."""
        try:
            self.arm.move_to(x, y, z, r)
        except Exception as err:
            print(f"[ERROR moving] {err}")

    def _toggle_gripper(self):
        self.gripper_closed = not self.gripper_closed
        try:
            if self.gripper_closed:
                self.arm.close_gripper()    # or self.arm.grip(True)
                print("[GRIPPER] closed")
            else:
                self.arm.open_gripper()     # or self.arm.grip(False)
                print("[GRIPPER] open")
        except Exception as err:
            print(f"[ERROR gripper] {err}")

    def _home(self):
        try:
            self.arm.home()
            print("[HOME] arm returning home")
        except Exception as err:
            print(f"[ERROR home] {err}")

    def _safe_stop(self):
        try:
            if hasattr(self.arm, "stop"):
                self.arm.stop()
        except Exception:
            pass
        print("[STOP] Esc pressed - exiting")
