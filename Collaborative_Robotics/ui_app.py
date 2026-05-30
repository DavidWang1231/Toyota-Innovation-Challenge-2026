import tkinter as tk
import cv2
from PIL import Image, ImageTk

# Try to import the team's zone detection. If not ready yet, run in demo mode.
try:
    from hand_detection import detect_zone
    HAS_DETECT = True
except Exception:
    HAS_DETECT = False

# ---- Shared state (the "whiteboard") ----
# The robot logic only needs to update this dict. The UI only reads it.
# Add new fields later (alarm, reaction_time, etc.) without breaking anything.
robot_state = {
    "status": "IDLE",      # IDLE / PICKING / STOPPED / HANDOFF
    "zone": "NONE",        # NONE / WARNING / DANGER / HANDOFF
    "picks": 0,            # completed pick count
    "stops": 0,            # safety stop count
    "alarm": False,        # reserved for future sound/light alarm
    "reaction_time": 0.0,  # reserved for future metrics (ms)
}

# Zones for 640x480 camera (x1, y1, x2, y2)
HANDOFF_ZONE = (400, 300, 620, 460)
CAUTION_ZONE = (120, 80, 520, 440)
DANGER_ZONE  = (220, 150, 420, 360)

# Color theme
BG = "#1e1e2e"
PANEL = "#2a2a3a"
TEXT = "#ffffff"
ACCENT = "#e60012"   # Toyota red

STATUS_COLORS = {
    "IDLE":    "#4caf50",
    "PICKING": "#2196f3",
    "STOPPED": "#f44336",
    "HANDOFF": "#ffc107",
}
LIGHT_COLORS = {
    "NONE":    "#4caf50",  # green - safe
    "WARNING": "#ff9800",  # orange
    "HANDOFF": "#ffc107",  # yellow
    "DANGER":  "#f44336",  # red
}


class RobotUI:
    def __init__(self, root):
        self.root = root
        root.title("Hana - Collaborative Robot Control Panel")
        root.configure(bg=BG)

        # ---- Left: camera video feed ----
        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, padx=12, pady=12)
        tk.Label(left, text="LIVE CAMERA", bg=BG, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.video_label = tk.Label(left, bg="#000000")
        self.video_label.pack()

        # ---- Right: status panel ----
        right = tk.Frame(root, bg=PANEL, padx=18, pady=18)
        right.grid(row=0, column=1, padx=12, pady=12, sticky="n")

        tk.Label(right, text="SYSTEM STATUS", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 12))

        # Safety light
        self.canvas = tk.Canvas(right, width=120, height=120,
                                bg=PANEL, highlightthickness=0)
        self.canvas.pack()
        self.light = self.canvas.create_oval(15, 15, 105, 105,
                                             fill="#4caf50", outline="#ffffff", width=3)
        tk.Label(right, text="SAFETY", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 10)).pack()

        # Status text
        self.status_lbl = tk.Label(right, text="STATUS: IDLE", bg=PANEL, fg=TEXT,
                                   font=("Segoe UI", 14, "bold"))
        self.status_lbl.pack(anchor="w", pady=(18, 4))
        self.zone_lbl = tk.Label(right, text="ZONE: NONE", bg=PANEL, fg=TEXT,
                                 font=("Segoe UI", 12))
        self.zone_lbl.pack(anchor="w", pady=4)

        # Stats
        tk.Frame(right, bg=ACCENT, height=2, width=200).pack(pady=12)
        self.picks_lbl = tk.Label(right, text="Picks completed: 0", bg=PANEL, fg=TEXT,
                                  font=("Segoe UI", 12))
        self.picks_lbl.pack(anchor="w", pady=2)
        self.stops_lbl = tk.Label(right, text="Safety stops: 0", bg=PANEL, fg=TEXT,
                                  font=("Segoe UI", 12))
        self.stops_lbl.pack(anchor="w", pady=2)

        # Buttons (demo controls - simulate states without robot)
        tk.Frame(right, bg=ACCENT, height=2, width=200).pack(pady=12)
        btns = tk.Frame(right, bg=PANEL)
        btns.pack()
        tk.Button(btns, text="Simulate PICK", width=14,
                  command=self.sim_pick).grid(row=0, column=0, pady=3)
        tk.Button(btns, text="Simulate STOP", width=14,
                  command=self.sim_stop).grid(row=1, column=0, pady=3)
        tk.Button(btns, text="Reset", width=14,
                  command=self.reset).grid(row=2, column=0, pady=3)

        # Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 = laptop built-in camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_frame()

    # ---- Demo button actions (replace with real robot logic later) ----
    def sim_pick(self):
        robot_state["status"] = "PICKING"
        robot_state["picks"] += 1

    def sim_stop(self):
        robot_state["status"] = "STOPPED"
        robot_state["stops"] += 1

    def reset(self):
        robot_state["status"] = "IDLE"
        robot_state["picks"] = 0
        robot_state["stops"] = 0

    def extract_zone(self, result, frame):
        """Safely pull a zone string out of whatever detect_zone returns."""
        valid = ("NONE", "WARNING", "DANGER", "HANDOFF")
        # Case 1: it already returned a clean string
        if isinstance(result, str):
            up = result.upper()
            if up == "CAUTION":
                return "WARNING"
            return up if up in valid else "NONE"
        # Case 2: it returned a tuple/list (e.g. zone + image)
        if isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, str):
                    up = item.upper()
                    if up == "CAUTION":
                        return "WARNING"
                    if up in valid:
                        return up
        # Case 3: anything else (numpy array, None, etc.) -> safe default
        return "NONE"

    # ---- Main loop driven by Tkinter's after() (no freezing) ----
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Update zone from real detection if available
            if HAS_DETECT:
                result = detect_zone(frame, DANGER_ZONE, CAUTION_ZONE,
                                     HANDOFF_ZONE, draw=True)
                zone = self.extract_zone(result, frame)
                robot_state["zone"] = zone
                # Auto safety logic: hand in danger -> stop
                if zone == "DANGER":
                    robot_state["status"] = "STOPPED"
                elif zone == "HANDOFF":
                    robot_state["status"] = "HANDOFF"

            # Convert BGR -> RGB for Tkinter
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.refresh_panel()
        self.root.after(30, self.update_frame)  # ~33 fps, no freeze

    def refresh_panel(self):
        s = robot_state
        self.status_lbl.config(text=f"STATUS: {s['status']}",
                               fg=STATUS_COLORS.get(s["status"], TEXT))
        self.zone_lbl.config(text=f"ZONE: {s['zone']}")
        self.picks_lbl.config(text=f"Picks completed: {s['picks']}")
        self.stops_lbl.config(text=f"Safety stops: {s['stops']}")
        self.canvas.itemconfig(self.light,
                               fill=LIGHT_COLORS.get(s["zone"], "#4caf50"))

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RobotUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()