"""
Hana — Collaborative Robotics Control Center (Web UI)

Tiny local web server using only the Python standard library — no Flask, no
extra deps. Launches scripts as subprocesses, tracks them, kills them on
request. The UI is a single self-contained HTML page.

Run:  python3 launcher.py
Then your browser opens at http://127.0.0.1:8765
"""
import json
import sys
import threading
import time
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = Path(__file__).parent.resolve()
PORT = 8765

# ── Tools catalog ─────────────────────────────────────────
# Each card occupies one grid area. Cards can have multiple actions
# (separate launch buttons sharing the same card body).
TOOLS = [
    {
        "id":      "calibration",
        "area":    "cal",
        "name":    "Robot Calibration",
        "desc":    "Map the camera's pixels to the robot's coordinate frame.",
        "color":   "amber",
        "actions": [
            {"label": "Launch Calibration", "file": "getTransformationMatrix.py"},
            {"label": "Test Connection",    "file": "testDobot.py"},
        ],
    },
    {
        "id":      "automation",
        "area":    "auto",
        "name":    "Start Robot Automation",
        "desc":    "Run the collaborative pick-and-place demo or the standalone hand-detection tool.",
        "color":   "red",
        "actions": [
            {"label": "Pick & Place",   "file": "pickCVBlock.py"},
            {"label": "Hand Detection", "file": "hand_detection.py"},
        ],
    },
    {
        "id":      "operator",
        "area":    "op",
        "name":    "Operator Monitor",
        "desc":    "Identify operators and detect fatigue on the laptop webcam.",
        "color":   "amber",
        "actions": [
            {"label": "Launch", "file": "operator_monitor.py"},
        ],
    },
    {
        "id":      "manual",
        "area":    "man",
        "name":    "Manual Control",
        "desc":    "Drive the robot arm manually to find or recover positions.",
        "color":   "green",
        "actions": [
            {"label": "Launch", "file": "manualControl.py"},
        ],
    },
]

def _find_action(filename):
    """Find the action dict for a given script filename."""
    for t in TOOLS:
        for a in t["actions"]:
            if a["file"] == filename:
                return a, t
    return None, None

# ── Process tracking (thread-safe) ────────────────────────
_proc_lock = threading.Lock()
_processes: dict[str, subprocess.Popen] = {}     # file -> Popen

def _is_running(filename: str) -> bool:
    with _proc_lock:
        p = _processes.get(filename)
        return p is not None and p.poll() is None

def _launch_file(filename) -> tuple[bool, str]:
    action, _tool = _find_action(filename)
    if action is None:
        return False, f"Unknown file: {filename}"
    path = SCRIPT_DIR / action["file"]
    if not path.exists():
        return False, f"Missing file: {action['file']}"
    if _is_running(filename):
        return False, "Already running"
    cmd = [sys.executable, str(path)] + action.get("args", [])
    try:
        proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR))
        with _proc_lock:
            _processes[filename] = proc
        return True, "Launched"
    except Exception as e:
        return False, f"Failed: {e}"

def _stop(filename: str) -> tuple[bool, str]:
    with _proc_lock:
        proc = _processes.get(filename)
    if not proc:
        return False, "Not running"
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            return False, f"Error stopping: {e}"
    with _proc_lock:
        _processes.pop(filename, None)
    return True, "Stopped"

def _stop_all():
    with _proc_lock:
        files = list(_processes.keys())
    for f in files:
        _stop(f)

def _running_files() -> list[str]:
    with _proc_lock:
        return [f for f, p in _processes.items() if p.poll() is None]

# Clean up dead processes periodically
def _reaper():
    while True:
        with _proc_lock:
            for f in list(_processes.keys()):
                if _processes[f].poll() is not None:
                    _processes.pop(f, None)
        threading.Event().wait(0.5)


# ── Robot connection probe (non-motion) ───────────────────
# Periodically checks whether the Dobot arm is reachable. The result is
# cached and exposed via /api/status so the frontend can show a live
# colored indicator without itself slowing things down.
_robot_status_lock = threading.Lock()
_robot_status = {
    "connected": False,
    "message":   "Checking...",
    "last_check": 0.0,
}
ROBOT_CHECK_INTERVAL = 8.0   # seconds between probes

def _probe_robot_once():
    """Quick non-motion check. Loads lib + searches port + handshakes. Returns dict."""
    try:
        import lib.DobotDllType as dType
    except Exception as e:
        return {"connected": False, "message": f"Library import failed"}
    try:
        api = dType.load()
    except OSError:
        return {"connected": False, "message": "Driver not installed on this OS"}
    except Exception as e:
        return {"connected": False, "message": f"Driver error: {e}"}
    try:
        com_port = dType.SearchDobot(api)[0]
    except Exception:
        return {"connected": False, "message": "Port scan failed"}
    if "COM" not in com_port:
        return {"connected": False, "message": "Arm not detected"}
    # Try a quick handshake. "Occupied" means another script is using the
    # arm — which still means the hardware IS connected.
    try:
        state = dType.ConnectDobot(api, com_port, 115200)[0]
    except Exception as e:
        return {"connected": False, "message": f"Connect error"}
    if state == dType.DobotConnect.DobotConnect_NoError:
        try:
            dType.DisconnectDobot(api)
        except Exception:
            pass
        return {"connected": True, "message": f"Connected · {com_port}"}
    if state == dType.DobotConnect.DobotConnect_Occupied:
        return {"connected": True, "message": f"In use · {com_port}"}
    return {"connected": False, "message": "Handshake failed"}

def _robot_probe_loop():
    """Background thread: probe the robot at a steady interval."""
    while True:
        try:
            res = _probe_robot_once()
        except Exception as e:
            res = {"connected": False, "message": f"Probe error: {e}"}
        with _robot_status_lock:
            _robot_status["connected"]  = res["connected"]
            _robot_status["message"]    = res["message"]
            _robot_status["last_check"] = time.time()
        threading.Event().wait(ROBOT_CHECK_INTERVAL)

def _get_robot_status():
    with _robot_status_lock:
        return dict(_robot_status)


# ── HTML / CSS / JS ────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Collaborative Robotics Control Center</title>
<style>
  :root {
    --bg:        #0e1015;
    --panel:     #1a1d24;
    --panel-2:   #232730;
    --border:    #2c313a;
    --text:      #e6e8ec;
    --text-dim:  #8b919c;
    --accent:    #e53935;
    --accent-h:  #ff5252;
    --red:       #e53935;
    --amber:     #f9a825;
    --green:     #43a047;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
               font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                            "Segoe UI", Helvetica, Arial, sans-serif; }

  /* ── Splash / connect screen ──────────────────────────── */
  #splash { position: fixed; inset: 0; z-index: 100; overflow: hidden;
            background: #07090d;
            display: flex; flex-direction: column; align-items: center;
            justify-content: center;
            transition: opacity .4s ease; }
  #splash.gone { display: none; }

  /* ── Curtain blinds: close in from top & bottom, then open up & down ── */
  #curtainTop, #curtainBottom {
      position: fixed; left: 0; right: 0;
      height: 52vh; z-index: 200;
      background: #07090d;
      pointer-events: none;
  }
  #curtainTop    { top: 0;    transform: translateY(-100%); }
  #curtainBottom { bottom: 0; transform: translateY(100%);  }

  /* Subtle red glow on each curtain's inner edge */
  #curtainTop::after,
  #curtainBottom::after {
      content: "";
      position: absolute; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg,
          transparent 0%,
          rgba(229,57,53,0.3) 15%,
          rgba(255,80,80,1)   50%,
          rgba(229,57,53,0.3) 85%,
          transparent 100%);
      box-shadow: 0 0 18px rgba(229,57,53,0.85),
                  0 0 48px rgba(229,57,53,0.4);
      opacity: 0;
      transition: opacity 0.2s ease;
  }
  #curtainTop::after    { bottom: 0; }
  #curtainBottom::after { top: 0;    }
  #curtainTop.closing::after,
  #curtainBottom.closing::after { opacity: 1; }

  /* Closing: slide IN to meet at center */
  #curtainTop.closing,
  #curtainBottom.closing {
      transform: translateY(0);
      transition: transform 0.45s cubic-bezier(0.7, 0, 0.3, 1);
  }
  /* Opening: slide OUT to top + bottom edges, revealing dashboard */
  #curtainTop.opening    { transform: translateY(-100%);
                            transition: transform 0.7s cubic-bezier(0.7, 0, 0.3, 1); }
  #curtainBottom.opening { transform: translateY(100%);
                            transition: transform 0.7s cubic-bezier(0.7, 0, 0.3, 1); }

  /* Dashboard wrapper — fades in once curtains start opening */
  .dashboard { opacity: 0;
               transition: opacity 0.45s ease 0.1s; }
  .dashboard.show { opacity: 1; }

  /* Glitch state: brief stutter + RGB-split + chroma flashes before implode */
  #splash.glitching { animation: splashGlitch 0.55s steps(11, end) forwards; }
  #splash.glitching .hero,
  #splash.glitching .meta {
      animation: contentGlitch 0.55s steps(11, end) forwards;
  }
  @keyframes splashGlitch {
      0%   { transform: translate(0,0)        skewX(0deg);  filter: none; }
      10%  { transform: translate(-5px, 2px)  skewX(-2deg); filter: hue-rotate(15deg) saturate(1.5) contrast(1.2); }
      20%  { transform: translate(7px, -2px)  skewX(3deg);  filter: hue-rotate(-20deg) saturate(2); }
      30%  { transform: translate(-3px, 4px)  skewX(0deg);  filter: hue-rotate(25deg) blur(1px); }
      40%  { transform: translate(0,0)        skewX(-4deg); filter: invert(0.05) hue-rotate(-30deg); }
      50%  { transform: translate(6px, 0)     skewX(0deg);  filter: blur(2px) saturate(0.5); }
      60%  { transform: translate(-4px, -3px) skewX(2deg);  filter: hue-rotate(40deg) brightness(1.4); }
      70%  { transform: translate(2px, 5px)   skewX(0deg);  filter: hue-rotate(-15deg); }
      80%  { transform: translate(0, -2px)    skewX(-1deg); filter: blur(0.5px); }
      90%  { transform: translate(0,0)        skewX(0deg);  filter: brightness(1.6); }
      100% { transform: translate(0,0)        skewX(0deg);  filter: none; }
  }
  @keyframes contentGlitch {
      0%, 100% { text-shadow: none;
                 transform: translate(0,0); }
      15%      { text-shadow: -3px 0 rgba(255,0,80,0.9),  3px 0 rgba(0,180,255,0.9);
                 transform: translate(-2px, 1px); }
      30%      { text-shadow:  4px 0 rgba(255,0,80,0.8), -4px 0 rgba(0,255,200,0.8);
                 transform: translate(3px, -1px); }
      45%      { text-shadow: -2px 0 rgba(255,50,50,0.9), 2px 0 rgba(50,200,255,0.9);
                 transform: translate(-1px, 2px); }
      60%      { text-shadow:  5px 0 rgba(255,0,0,0.7),  -5px 0 rgba(0,255,255,0.7);
                 transform: translate(0, -3px); }
      75%      { text-shadow: -1px 0 rgba(255,255,255,0.6);
                 transform: translate(2px, 0); }
  }

  /* Layer 1 — animated radial gradient mesh (brighter, more visible) */
  #splash .bg-mesh { position: absolute; inset: -20%; z-index: 0;
                      background:
                        radial-gradient(circle at 22% 28%, rgba(229,57,53,0.30), transparent 45%),
                        radial-gradient(circle at 80% 72%, rgba(229,57,53,0.22), transparent 50%),
                        radial-gradient(circle at 50% 50%, rgba(40,50,80,0.30), transparent 60%);
                      animation: meshDrift 22s ease-in-out infinite alternate;
                      filter: blur(20px); }
  @keyframes meshDrift {
      0%   { transform: translate(0, 0)        rotate(0deg);   }
      50%  { transform: translate(60px, -40px)  rotate(6deg);   }
      100% { transform: translate(-30px, 50px) rotate(-4deg);  }
  }

  /* Layer 1b — diagonal scanning light sweep */
  #splash .scanlight { position: absolute; inset: 0; z-index: 0;
                        background: linear-gradient(115deg,
                          transparent 35%,
                          rgba(229,57,53,0.07) 50%,
                          transparent 65%);
                        background-size: 220% 220%;
                        animation: scan 14s linear infinite;
                        pointer-events: none; }
  @keyframes scan {
      from { background-position: 120% 120%; }
      to   { background-position: -20% -20%; }
  }

  /* Layer 1c — vignette to darken edges */
  #splash .vignette { position: absolute; inset: 0; z-index: 0;
                       background: radial-gradient(ellipse at center,
                         transparent 35%, rgba(0,0,0,0.55) 100%);
                       pointer-events: none; }

  /* Layer 2 — subtle grid, masked to fade at edges */
  #splash .grid { position: absolute; inset: 0; z-index: 0; opacity: .35;
                   background-image:
                      linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
                   background-size: 64px 64px;
                   -webkit-mask-image: radial-gradient(ellipse 70% 60% at center, #000 30%, transparent 80%);
                           mask-image: radial-gradient(ellipse 70% 60% at center, #000 30%, transparent 80%); }

  /* Layer 3 — floating red particles */
  #splash .particles { position: absolute; inset: 0; z-index: 0; pointer-events: none; }
  #splash .particles span {
      position: absolute; bottom: -10px;
      width: 3px; height: 3px; border-radius: 50%;
      background: rgba(229,57,53,0.55);
      box-shadow: 0 0 8px rgba(229,57,53,0.6);
      animation: floatUp linear infinite;
      opacity: 0;
  }
  @keyframes floatUp {
      0%   { transform: translateY(0)      translateX(0);    opacity: 0; }
      10%  { opacity: 1; }
      50%  { transform: translateY(-50vh)  translateX(20px); opacity: 0.9; }
      90%  { opacity: 1; }
      100% { transform: translateY(-110vh) translateX(-10px); opacity: 0; }
  }

  /* Layer 4 — slow radar pulse, subtle */
  #splash .pulse-wrap { position: absolute; inset: 0; z-index: 0;
                         display: flex; align-items: center; justify-content: center;
                         pointer-events: none; overflow: hidden; }
  #splash .pulse-wrap .ring {
      position: absolute; top: 50%; left: 50%;
      width: 220px; height: 220px;
      margin-top: -110px; margin-left: -110px;
      border: 1px solid rgba(229,57,53,0.5);
      border-radius: 50%; opacity: 0;
      animation: radar 6s cubic-bezier(0.16, 1, 0.3, 1) infinite;
  }
  #splash .pulse-wrap .ring:nth-child(2) { animation-delay: 2s; }
  #splash .pulse-wrap .ring:nth-child(3) { animation-delay: 4s; }
  @keyframes radar {
      0%   { transform: scale(0.2);  opacity: 0.7; }
      85%  { opacity: 0.04; }
      100% { transform: scale(14);   opacity: 0; }
  }

  /* All content above background layers */
  #splash > *:not(.bg-mesh):not(.grid):not(.particles):not(.pulse-wrap) {
      position: relative; z-index: 2;
  }

  /* Hero — staggered reveal animations */
  #splash .hero { display: flex; flex-direction: column; align-items: center;
                  gap: 40px; }

  /* Kicker — centered above the logo, part of hero column */
  #splash .hero .kicker { color: var(--text-dim); font-size: 11px;
                          letter-spacing: .45em; text-transform: uppercase;
                          opacity: 0;
                          animation: fadeIn 0.8s ease 0.1s forwards;
                          margin-bottom: -12px; }

  #splash .hero .lockup { display: flex; flex-direction: row;
                          align-items: center; gap: 18px; }
  #splash .hero img.tmmc-logo { height: 160px; width: auto;
                                 filter: drop-shadow(0 6px 30px rgba(229,57,53,0.35));
                                 opacity: 0;
                                 animation: logoIn 1.1s cubic-bezier(0.16,1,0.3,1) 0.25s both; }
  #splash .hero .brand-text { color: #fff; font-weight: 600;
                              font-size: 36px; line-height: 1.08;
                              letter-spacing: -0.01em; text-align: left; }
  #splash .hero .brand-text .line {
      display: block; opacity: 0;
      animation: lineIn 0.9s cubic-bezier(0.16,1,0.3,1) forwards;
  }
  #splash .hero .brand-text .line:nth-child(1) { animation-delay: 0.55s; }
  #splash .hero .brand-text .line:nth-child(2) { animation-delay: 0.70s; }
  #splash .hero .brand-text .line:nth-child(3) { animation-delay: 0.85s; }
  @keyframes logoIn { from { opacity:0; transform: scale(0.94) translateY(14px); }
                      to   { opacity:1; transform: scale(1)    translateY(0);    } }
  @keyframes lineIn { from { opacity:0; transform: translateX(-18px); }
                      to   { opacity:1; transform: translateX(0);     } }

  /* Logo image fallback */
  #splash .hero .lockup.broken img.tmmc-logo { display: none; }
  #splash .hero .lockup .fallback { display: none; background: var(--accent);
                                     color: #fff; font-weight: 800;
                                     font-size: 30px; letter-spacing: .12em;
                                     padding: 18px 26px; border-radius: 10px;
                                     box-shadow: 0 8px 32px rgba(229,57,53,0.45); }
  #splash .hero .lockup.broken .fallback { display: block; }

  /* CONNECT button — entrance + breathe + magnetic hover + sheen */
  #connectBtn { position: relative; overflow: hidden;
                background: var(--accent); color: #fff; border: 0;
                padding: 18px 64px; font-size: 14px; font-weight: 700;
                letter-spacing: .32em; border-radius: 8px; cursor: pointer;
                display: inline-flex; align-items: center; gap: 14px;
                box-shadow: 0 6px 24px rgba(229,57,53,0.35);
                opacity: 0;
                animation: btnIn 1s cubic-bezier(0.16,1,0.3,1) 1.2s both,
                           btnBreathe 3.6s ease-in-out 2.2s infinite;
                transition: transform .25s cubic-bezier(0.16,1,0.3,1),
                            box-shadow .25s ease,
                            letter-spacing .25s ease,
                            filter .15s ease; }
  #connectBtn .arrow { display: inline-block;
                       transition: transform .25s cubic-bezier(0.16,1,0.3,1); }
  #connectBtn::before { content: ""; position: absolute; inset: 0;
                        background: linear-gradient(120deg,
                          transparent 0%, rgba(255,255,255,0.18) 50%,
                          transparent 100%);
                        transform: translateX(-100%);
                        transition: transform .6s ease; }
  #connectBtn:hover { transform: translateY(-2px);
                       letter-spacing: .38em;
                       box-shadow: 0 14px 44px rgba(229,57,53,0.55); }
  #connectBtn:hover::before { transform: translateX(100%); }
  #connectBtn:hover .arrow  { transform: translateX(6px); }
  #connectBtn:active { transform: translateY(0); filter: brightness(0.95); }
  @keyframes btnIn { from { opacity:0; transform: translateY(20px); }
                     to   { opacity:1; transform: translateY(0);    } }
  @keyframes btnBreathe {
      0%, 100% { box-shadow: 0 6px 24px rgba(229,57,53,0.35),
                             0 0 0  0   rgba(229,57,53,0.0); }
      50%      { box-shadow: 0 6px 24px rgba(229,57,53,0.35),
                             0 0 0 18px rgba(229,57,53,0.0); }
  }

  /* Bottom-center status — clipped well below the bottom edge */
  #splash .meta { position: absolute; bottom: -22px; left: 0; right: 0;
                  color: var(--text-dim); font-size: 10px;
                  letter-spacing: .35em; text-transform: uppercase;
                  display: flex; align-items: center; justify-content: center;
                  gap: 10px;
                  opacity: 0;
                  animation: fadeIn 0.8s ease 1.6s forwards;
                  pointer-events: none; }
  #splash .meta .status-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: #43a047; box-shadow: 0 0 8px rgba(67,160,71,0.8);
      animation: dotPulse 2s ease-in-out infinite;
  }
  @keyframes dotPulse { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
  @keyframes fadeIn   { to { opacity: 1; } }

  /* ── Dashboard animated background (matches splash) ───── */
  .dashboard-bg { position: fixed; inset: 0; z-index: 0;
                   overflow: hidden; pointer-events: none; }
  .dashboard, header, main, footer { position: relative; z-index: 1; }
  .dashboard-bg .mesh {
      position: absolute; inset: -20%;
      background:
        radial-gradient(circle at 22% 28%, rgba(229,57,53,0.22), transparent 45%),
        radial-gradient(circle at 80% 72%, rgba(229,57,53,0.16), transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(40,50,80,0.22), transparent 60%);
      animation: meshDrift 22s ease-in-out infinite alternate;
      filter: blur(24px);
  }
  .dashboard-bg .scan {
      position: absolute; inset: 0;
      background: linear-gradient(115deg,
          transparent 35%, rgba(229,57,53,0.05) 50%, transparent 65%);
      background-size: 220% 220%;
      animation: scan 14s linear infinite;
  }
  .dashboard-bg .grid {
      position: absolute; inset: 0; opacity: .25;
      background-image:
         linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
         linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
      background-size: 64px 64px;
      -webkit-mask-image: radial-gradient(ellipse 80% 70% at center, #000 30%, transparent 90%);
              mask-image: radial-gradient(ellipse 80% 70% at center, #000 30%, transparent 90%);
  }
  .dashboard-bg .vignette {
      position: absolute; inset: 0;
      background: radial-gradient(ellipse at center,
          transparent 40%, rgba(0,0,0,0.55) 100%);
  }
  /* html keeps the dark fallback; body is transparent so .dashboard-bg shows */
  html { background: #07090d; }
  body { background: transparent; }
  .card { background: rgba(26, 29, 36, 0.85); backdrop-filter: blur(6px);
          -webkit-backdrop-filter: blur(6px); }

  /* ── Main dashboard ───────────────────────────────────── */
  header { padding: 22px 36px 16px; border-bottom: 1px solid var(--border); }
  header .tmmc-banner { display: flex; align-items: center; gap: 18px; }
  header .tmmc-banner .logo { height: 44px; width: auto; flex-shrink: 0;
                               background: #fff; padding: 4px 8px;
                               border-radius: 6px; }
  header .tmmc-banner .title-block { flex: 1; }
  header .tmmc-banner .kicker { color: var(--text-dim); font-size: 10px;
                                 letter-spacing: .22em; text-transform: uppercase;
                                 margin-bottom: 2px; }
  header .tmmc-banner h1 { margin: 0; font-size: 22px; font-weight: 700;
                            color: var(--text); }
  header .tmmc-banner .meta { color: var(--text-dim); font-size: 11px;
                               letter-spacing: .14em; text-transform: uppercase; }
  header p { color: var(--text-dim); margin: 10px 0 0; font-size: 12px;
             padding-left: 62px; }

  /* Robot connection indicator (top header) */
  .robot-status { display: inline-flex; align-items: center; gap: 8px;
                  font-size: 11px; letter-spacing: .18em;
                  text-transform: uppercase; font-weight: 600;
                  padding: 6px 12px; border-radius: 999px;
                  border: 1px solid rgba(255,255,255,0.07);
                  background: rgba(255,255,255,0.03);
                  margin-right: 14px; }
  .robot-status .rs-dot { width: 8px; height: 8px; border-radius: 50%;
                          background: var(--text-dim);
                          box-shadow: 0 0 0 0 rgba(255,255,255,0); }
  .robot-status.ok { color: #66d171; border-color: rgba(67,160,71,0.45); }
  .robot-status.ok .rs-dot { background: #43a047;
                              box-shadow: 0 0 10px rgba(67,160,71,0.9);
                              animation: rsPulse 2s ease-in-out infinite; }
  .robot-status.bad { color: #ff7979; border-color: rgba(229,57,53,0.45); }
  .robot-status.bad .rs-dot { background: #e53935;
                               box-shadow: 0 0 10px rgba(229,57,53,0.9); }
  .robot-status.checking { color: var(--text-dim); }
  @keyframes rsPulse { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }

  main { padding: 24px 28px 80px;
         display: grid;
         grid-template-columns: 1fr 1fr;
         grid-template-areas:
             "cal  cal"
             "auto op"
             "man  man";
         gap: 18px;
         max-width: 1280px;
         margin: 0 auto; }
  .card[data-area="cal"]  { grid-area: cal;  }
  .card[data-area="auto"] { grid-area: auto; }
  .card[data-area="op"]   { grid-area: op;   }
  .card[data-area="man"]  { grid-area: man;  }

  /* Multi-action button row inside a card body */
  .card-body .actions { display: flex; flex-wrap: wrap; gap: 10px; }

  /* ── Card: 3D glass panel with depth & parallax content ──── */
  /* Grid parent has perspective so the cards can tilt in 3D */
  main { perspective: 1800px; }

  .card { position: relative;
          background:
              /* Top rim highlight + bottom shadow gradient for "domed" look */
              linear-gradient(180deg,
                  rgba(255,255,255,0.12) 0%,
                  rgba(255,255,255,0.03) 8%,
                  rgba(255,255,255,0)    35%,
                  rgba(0,0,0,0.08)       70%,
                  rgba(0,0,0,0.25)       100%),
              rgba(22, 25, 32, 0.38);
          backdrop-filter: blur(22px) saturate(150%);
          -webkit-backdrop-filter: blur(22px) saturate(150%);
          border: 1px solid rgba(255, 255, 255, 0.07);
          border-radius: 18px;
          overflow: visible;            /* allow content to pop forward in 3D */
          --accent-glow: 229, 57, 53;
          transform-origin: center center;
          transform-style: preserve-3d; /* enable 3D children */
          will-change: transform;
          /* Multi-layer shadows: inset rim lighting + outer drop + ambient */
          box-shadow:
              inset 0  1px 0 rgba(255,255,255,0.18),
              inset 0 -1px 0 rgba(0,0,0,0.45),
              inset 1px 0 0 rgba(255,255,255,0.04),
              inset -1px 0 0 rgba(0,0,0,0.15),
              0 12px 24px rgba(0,0,0,0.40),
              0 32px 60px rgba(0,0,0,0.30);
          transition: transform .35s cubic-bezier(0.16, 1, 0.3, 1),
                      border-color .35s ease,
                      box-shadow .35s ease; }

  /* Subtle accent line at the very top — sits forward in 3D space */
  .card .accent-line {
      position: absolute; top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg,
          transparent 0%,
          rgba(var(--accent-glow), 0.55) 30%,
          rgba(var(--accent-glow), 0.95) 50%,
          rgba(var(--accent-glow), 0.55) 70%,
          transparent 100%);
      opacity: 0.55;
      transform: translateZ(8px);          /* slight parallax forward */
      transition: opacity .35s ease, height .35s ease,
                  box-shadow .35s ease, transform .35s ease;
      pointer-events: none;
      z-index: 3;
  }
  .card:hover .accent-line {
      opacity: 1; height: 2px;
      transform: translateZ(20px);
      box-shadow: 0 0 14px rgba(var(--accent-glow), 0.8);
  }

  /* Inner top highlight — "lit from above" glass feel */
  .card::after { content: ""; position: absolute; inset: 0;
                  border-radius: inherit; pointer-events: none;
                  background: linear-gradient(180deg,
                      rgba(255,255,255,0.09) 0%,
                      rgba(255,255,255,0)    80px);
                  z-index: 1; }

  /* Cursor-following spotlight */
  .card::before { content: ""; position: absolute; inset: 0;
                   border-radius: inherit; pointer-events: none;
                   opacity: 0; transition: opacity .35s ease;
                   background: radial-gradient(circle 460px
                       at var(--mx, 50%) var(--my, 50%),
                       rgba(var(--accent-glow), 0.24),
                       transparent 60%);
                   z-index: 1; }
  .card:hover::before { opacity: 1; }

  /* 3D tilt + lift on hover — driven by --rx, --ry set in JS */
  .card:hover {
      transform: rotateX(var(--rx, 0deg))
                 rotateY(var(--ry, 0deg))
                 translateY(-8px) translateZ(60px);
      border-color: rgba(var(--accent-glow), 0.50);
      box-shadow:
          /* Stronger inset rim lighting on hover */
          inset 0  1px 0 rgba(255,255,255,0.25),
          inset 0 -1px 0 rgba(0,0,0,0.55),
          inset 1px 0 0 rgba(255,255,255,0.06),
          inset -1px 0 0 rgba(0,0,0,0.2),
          /* Deeper drop shadows */
          0 24px 48px rgba(0,0,0,0.55),
          0 48px 96px rgba(0,0,0,0.45),
          /* Accent halo */
          0 0 60px rgba(var(--accent-glow), 0.28);
  }

  /* Per-card accent colour */
  .card[data-area="auto"] { --accent-glow: 229, 57, 53; }
  .card[data-area="op"],
  .card[data-area="cal"]  { --accent-glow: 249, 168, 37; }
  .card[data-area="man"]  { --accent-glow: 67, 160, 71; }

  .card-body { position: relative; z-index: 2; }

  /* Staggered entry — slight 3D rotateX so cards "lift" into place */
  .card { opacity: 0;
          transform: translateY(40px) rotateX(-12deg); }
  body.entered .card {
      opacity: 1;
      transform: translateY(0) rotateX(0);
      animation: cardIn 0.85s cubic-bezier(0.16, 1, 0.3, 1) backwards;
      animation-delay: calc(0.35s + var(--idx, 0) * 0.11s);
  }
  @keyframes cardIn {
      from { opacity: 0; transform: translateY(40px) rotateX(-12deg); }
      to   { opacity: 1; transform: translateY(0)    rotateX(0);     }
  }

  /* ── Card body ─ floats forward in 3D so it parallax-tilts ── */
  .card-body { padding: 26px 26px 24px;
                transform-style: preserve-3d;
                transform: translateZ(0);
                transition: transform .35s cubic-bezier(0.16, 1, 0.3, 1); }
  .card:hover .card-body { transform: translateZ(30px); }
  .card-body h2 { margin: 0 0 10px; font-size: 22px; font-weight: 700;
                   letter-spacing: -0.015em; color: var(--text);
                   transform: translateZ(0);
                   transition: transform .35s cubic-bezier(0.16, 1, 0.3, 1),
                               text-shadow .35s ease; }
  .card:hover .card-body h2 {
       transform: translateZ(18px);
       text-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
  }
  .card-body p  { margin: 0 0 22px; font-size: 13.5px; line-height: 1.55;
                   color: var(--text-dim); max-width: 90%; }
  .card-body .actions { transform: translateZ(0);
                         transition: transform .35s cubic-bezier(0.16, 1, 0.3, 1); }
  .card:hover .card-body .actions { transform: translateZ(24px); }

  /* ── Buttons: dimensional, glowing on hover ────────────── */
  .btn { display: inline-flex; align-items: center; gap: 8px;
         padding: 11px 20px; border: 1px solid rgba(255,255,255,0.10);
         border-radius: 9px;
         font-size: 13px; font-weight: 700; cursor: pointer;
         color: #fff; letter-spacing: .04em;
         background: linear-gradient(180deg,
             rgba(255,255,255,0.12) 0%,
             rgba(255,255,255,0) 50%,
             rgba(0,0,0,0.08) 100%), var(--btn-bg, #555c66);
         box-shadow: inset 0 1px 0 rgba(255,255,255,0.20),
                     0 2px 8px rgba(0,0,0,0.30);
         transition: transform .15s cubic-bezier(0.16, 1, 0.3, 1),
                     filter .2s ease,
                     box-shadow .25s ease; }
  .btn:hover  { transform: translateY(-1px); filter: brightness(1.10);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.30),
                            0 6px 18px rgba(0,0,0,0.45),
                            0 0 22px rgba(var(--btn-glow, 229,57,53), 0.45); }
  .btn:active { transform: translateY(1px); filter: brightness(0.95); }
  .btn.red    { --btn-bg: #e53935; --btn-glow: 229,57,53; }
  .btn.amber  { --btn-bg: #f9a825; --btn-glow: 249,168,37; color: #1a1d24;
                text-shadow: 0 1px 0 rgba(255,255,255,0.15); }
  .btn.green  { --btn-bg: #43a047; --btn-glow: 67,160,71; }
  .btn.stop   { --btn-bg: #4a525c; --btn-glow: 90,90,100; }

  footer { position: fixed; bottom: 0; left: 0; right: 0;
           background: var(--panel); border-top: 1px solid var(--border);
           padding: 12px 28px; display: flex; align-items: center; gap: 14px;
           font-size: 13px; }
  .dot { width: 10px; height: 10px; border-radius: 50%;
         background: var(--text-dim); transition: background .2s ease; }
  .dot.live { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .status { flex: 1; color: var(--text-dim); }
  .status.live { color: var(--text); }
  footer .btn.stopall { background: var(--accent); }
</style>
</head>
<body>

<!-- ── Splash / connect screen ─────────────────────────── -->
<div id="splash">
  <!-- Layered background -->
  <div class="bg-mesh"></div>
  <div class="scanlight"></div>
  <div class="grid"></div>
  <div class="particles" id="particles"></div>
  <div class="pulse-wrap">
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
  </div>
  <div class="vignette"></div>

  <!-- Centered hero stack -->
  <div class="hero">
    <div class="kicker">Toyota Innovation Challenge · 2026</div>
    <div class="lockup" id="lockup">
      <img class="tmmc-logo" src="/static/tmmc_logo.png?v=__CACHEBUST__" alt="TMMC"
           onerror="document.getElementById('lockup').classList.add('broken')">
      <div class="fallback">TMMC</div>
      <div class="brand-text">
        <span class="line">Toyota Motor</span>
        <span class="line">Manufacturing</span>
        <span class="line">Canada Inc.</span>
      </div>
    </div>
    <button id="connectBtn">
      <span>CONNECT</span>
      <span class="arrow">→</span>
    </button>
  </div>

  <!-- Bottom status (centered across whole window) -->
  <div class="meta">
    <span class="status-dot"></span>
    System Online · Collaborative Robotics · 127.0.0.1
  </div>
</div>

<!-- ── Dashboard background (animated, same vibe as splash) ── -->
<div class="dashboard-bg">
  <div class="mesh"></div>
  <div class="scan"></div>
  <div class="grid"></div>
  <div class="vignette"></div>
</div>

<!-- Curtain blinds: close in over the splash, then open to reveal dashboard -->
<div id="curtainTop"></div>
<div id="curtainBottom"></div>

<!-- Dashboard content -->
<div class="dashboard" id="dashboard">
  <header>
    <div class="tmmc-banner">
      <img class="logo" src="/static/tmmc_logo.png?v=__CACHEBUST__" alt="TMMC"
           onerror="this.style.display='none'">
      <div class="title-block">
        <div class="kicker">Collaborative Robotics</div>
        <h1>Control Center</h1>
      </div>
      <div class="robot-status checking" id="robotStatus">
        <span class="rs-dot"></span>
        <span class="rs-text">Checking…</span>
      </div>
      <span class="meta">Innovation Challenge 2026</span>
    </div>
    <p>Click any card to launch a script. Click again to stop it.</p>
  </header>
  <main id="cards"></main>

  <footer>
    <span class="dot" id="dot"></span>
    <span class="status" id="status">Nothing running.</span>
    <button class="btn stopall" onclick="stopAll()">Stop All</button>
  </footer>
</div>
<script>
  const TOOLS = __TOOLS_JSON__;

  // Map file → {action, tool} for fast lookup
  const FILE_INDEX = {};
  TOOLS.forEach(t => t.actions.forEach(a => {
      FILE_INDEX[a.file] = { action: a, tool: t };
  }));

  function render() {
    const main = document.getElementById('cards');
    main.innerHTML = '';
    TOOLS.forEach((t, i) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.area = t.area;
      card.style.setProperty('--idx', i);

      const buttonsHtml = t.actions.map(a => `
          <button class="btn ${t.color}" data-file="${a.file}"
                  data-label="${a.label}">▶&nbsp; ${a.label}</button>
      `).join('');

      card.innerHTML = `
        <div class="accent-line"></div>
        <div class="card-body">
          <h2>${t.name}</h2>
          <p>${t.desc}</p>
          <div class="actions">${buttonsHtml}</div>
        </div>`;
      card.querySelectorAll('button[data-file]').forEach(btn => {
        btn.addEventListener('click', () => toggle(btn.dataset.file));
      });

      // Cursor-following spotlight + 3D tilt (pronounced)
      card.addEventListener('mousemove', (e) => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width;   // 0..1
        const py = (e.clientY - r.top)  / r.height;
        card.style.setProperty('--mx', (px * 100) + '%');
        card.style.setProperty('--my', (py * 100) + '%');
        // Tilt up to ±9deg (was ±3deg)
        card.style.setProperty('--rx', ((py - 0.5) * -18).toFixed(2) + 'deg');
        card.style.setProperty('--ry', ((px - 0.5) *  18).toFixed(2) + 'deg');
      });
      card.addEventListener('mouseleave', () => {
        card.style.setProperty('--rx', '0deg');
        card.style.setProperty('--ry', '0deg');
      });

      main.appendChild(card);
    });
  }

  async function getStatus() {
    const r = await fetch('/api/status');
    const j = await r.json();
    updateRobotStatus(j.robot);
    return j.running;
  }

  function updateRobotStatus(robot) {
    const el = document.getElementById('robotStatus');
    if (!el || !robot) return;
    el.classList.remove('ok', 'bad', 'checking');
    if (robot.connected) {
      el.classList.add('ok');
      el.querySelector('.rs-text').textContent = robot.message || 'Robot connected';
    } else {
      el.classList.add('bad');
      el.querySelector('.rs-text').textContent = robot.message || 'Robot offline';
    }
  }

  async function toggle(file) {
    const running = (await getStatus()).includes(file);
    const ep = running ? '/api/stop' : '/api/launch';
    await fetch(`${ep}?file=${encodeURIComponent(file)}`, {method: 'POST'});
    refresh();
  }

  async function stopAll() {
    await fetch('/api/stop_all', {method: 'POST'});
    refresh();
  }

  async function refresh() {
    const running = await getStatus();
    document.querySelectorAll('button[data-file]').forEach(btn => {
      const f = btn.dataset.file;
      const entry = FILE_INDEX[f];
      if (!entry) return;
      const isRunning = running.includes(f);
      btn.textContent = isRunning ? '■  Stop' : `▶  ${btn.dataset.label}`;
      btn.className = 'btn ' + (isRunning ? 'stop' : entry.tool.color);
    });
    const dot    = document.getElementById('dot');
    const status = document.getElementById('status');
    if (running.length) {
      dot.classList.add('live');
      status.classList.add('live');
      const names = running.map(f => FILE_INDEX[f]?.action.label || f).join(', ');
      status.textContent = `Running: ${names}`;
    } else {
      dot.classList.remove('live');
      status.classList.remove('live');
      status.textContent = 'Nothing running.';
    }
  }

  render();
  refresh();
  setInterval(refresh, 1000);

  // ── Splash: spawn floating particles ───────────────────
  (function spawnParticles() {
    const layer = document.getElementById('particles');
    if (!layer) return;
    const N = 28;
    for (let i = 0; i < N; i++) {
      const p = document.createElement('span');
      const dur = 8 + Math.random() * 10;        // 8-18s rise
      const delay = -Math.random() * dur;        // start in random phase
      const left = Math.random() * 100;          // % across screen
      const size = 1 + Math.random() * 3;        // 1-4 px
      const drift = (Math.random() - 0.5) * 60;  // horizontal drift
      p.style.left = left + '%';
      p.style.width = size + 'px';
      p.style.height = size + 'px';
      p.style.animationDuration = dur + 's';
      p.style.animationDelay = delay + 's';
      p.style.setProperty('--drift', drift + 'px');
      layer.appendChild(p);
    }
  })();

  // ── Splash → dashboard transition ─────────────────────────
  // Sequence:
  //   1) glitch the splash (~420ms)
  //   2) two curtains slide IN from top + bottom, meeting at the middle
  //   3) splash is fully hidden; curtains briefly form a red-edged seam
  //   4) curtains slide OUT to top + bottom edges, revealing the dashboard
  //   5) dashboard fades in + cards stagger in
  document.getElementById('connectBtn').addEventListener('click', () => {
    const splash       = document.getElementById('splash');
    const curtainTop   = document.getElementById('curtainTop');
    const curtainBot   = document.getElementById('curtainBottom');
    const dashboard    = document.getElementById('dashboard');

    splash.classList.add('glitching');

    // Curtains close in
    setTimeout(() => {
      curtainTop.classList.add('closing');
      curtainBot.classList.add('closing');
    }, 420);

    // After curtains meet: hide splash (no longer needed)
    setTimeout(() => {
      splash.classList.add('gone');
    }, 870);

    // Brief pause with the red seam, then open the blinds
    setTimeout(() => {
      dashboard.classList.add('show');
      document.body.classList.add('entered');     // cards start their stagger
      curtainTop.classList.add('opening');
      curtainBot.classList.add('opening');
    }, 1050);
  });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass    # silence default access logs

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            import time as _t
            # Strip any per-action `args` before sending to the browser
            tools_safe = []
            for t in TOOLS:
                safe = {k: v for k, v in t.items() if k != "actions"}
                safe["actions"] = [{k: v for k, v in a.items() if k != "args"}
                                    for a in t["actions"]]
                tools_safe.append(safe)
            html = HTML.replace("__TOOLS_JSON__", json.dumps(tools_safe))
            html = html.replace("__CACHEBUST__", str(int(_t.time())))
            return self._send_html(html)
        if url.path == "/api/status":
            return self._send_json(200, {
                "running": _running_files(),
                "robot":   _get_robot_status(),
            })
        if url.path.startswith("/static/"):
            return self._serve_static(url.path[len("/static/"):])
        self.send_response(404); self.end_headers()

    def _serve_static(self, name):
        # Only serve files from the script directory, no traversal
        if "/" in name or ".." in name:
            self.send_response(404); self.end_headers(); return
        path = SCRIPT_DIR / name
        if not path.exists() or not path.is_file():
            self.send_response(404); self.end_headers(); return
        # naive content-type
        ct = ("image/png"  if name.lower().endswith(".png")  else
              "image/jpeg" if name.lower().endswith((".jpg",".jpeg")) else
              "image/svg+xml" if name.lower().endswith(".svg") else
              "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        # No caching — so if the user replaces the file, it loads fresh.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        fname = q.get("file", [None])[0]

        if url.path == "/api/launch" and fname:
            ok, msg = _launch_file(fname)
            return self._send_json(200 if ok else 400, {"ok": ok, "msg": msg})

        if url.path == "/api/stop" and fname:
            ok, msg = _stop(fname)
            return self._send_json(200 if ok else 400, {"ok": ok, "msg": msg})

        if url.path == "/api/stop_all":
            _stop_all()
            return self._send_json(200, {"ok": True})

        self.send_response(404); self.end_headers()


def main():
    threading.Thread(target=_reaper, daemon=True).start()
    threading.Thread(target=_robot_probe_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Hana Control Center → {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down — stopping any running scripts...")
        _stop_all()
        server.shutdown()


if __name__ == "__main__":
    main()
