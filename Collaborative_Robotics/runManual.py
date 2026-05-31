import cv2
import dobotArm
import lib.DobotDllType as dType
import manualControl

# Connect + home the arm
api = dType.load()
dobotArm.initialize_robot(api)

# --- ADD THESE LINES TO MAXIMIZE SPEED ---
# dType.SetPTPCommonParams(api, velocity, acceleration, isQueued)
# Values range from 0 to 100 (percentage of max speed/accel)
dType.SetPTPCommonParams(api, 100, 100, 1) 
# -----------------------------------------


print("Arm connected and homed.")

# Open the camera (try built-in index 0, then external index 1)
cap = None
for idx in (0, 1):
    test = cv2.VideoCapture(idx)
    if test.isOpened():
        cap = test
        print(f"Camera opened at index {idx}.")
        break
    test.release()

if cap is None:
    print("WARNING: No camera found. Keys still control the arm; window is blank.")
    cap = cv2.VideoCapture(0)

# Run manual control
result = manualControl.manual_control_loop(api, cap, window_name="Manual Control")
print(f"Manual control ended ({result}).")

cap.release()
cv2.destroyAllWindows()
