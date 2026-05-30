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
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = Path(__file__).parent.resolve()
PORT = 8765

# ── Tools catalog ─────────────────────────────────────────
TOOLS = [
    {
        "section": "RUN",
        "name":    "Pick & Place + Safety",
        "file":    "pickCVBlock.py",
        "desc":    "Full collaborative pick-and-place with hand safety, handoff, "
                   "and pick-retry (jidoka). This is the main demo script.",
        "color":   "red",
    },
    {
        "section": "RUN",
        "name":    "Hand Detection Demo",
        "file":    "hand_detection.py",
        "desc":    "Standalone hand detection with 5s auto-calibration and "
                   "click-to-redraw zone editing.",
        "color":   "red",
    },
    {
        "section": "OPERATOR",
        "name":    "Fatigue Monitor",
        "file":    "fatigue_monitor.py",
        "desc":    "Laptop webcam: detects drowsy operators via eye-closure "
                   "and yawn tracking. Shows alerts when attention drops.",
        "color":   "amber",
    },
    {
        "section": "CALIBRATION",
        "name":    "Camera Intrinsics",
        "file":    "calibrateCamera.py",
        "args":    ["--calibrate"],
        "desc":    "Re-calibrate camera lens distortion with an ArUco board. "
                   "Only run if camera_params.npz is wrong.",
        "color":   "amber",
    },
    {
        "section": "CALIBRATION",
        "name":    "Pixel → Robot Transform",
        "file":    "getTransformationMatrix.py",
        "desc":    "Camera-to-robot coordinate calibration (12 points). "
                   "Run once after the camera is mounted to its final position.",
        "color":   "amber",
    },
    {
        "section": "TOOLS",
        "name":    "Red Object Tracker",
        "file":    "red_object_tracker.py",
        "desc":    "Live HSV tuner + tight contour tracking. Prints pixel "
                   "and robot coordinates for any red object.",
        "color":   "green",
    },
    {
        "section": "TOOLS",
        "name":    "Manual Control",
        "file":    "manualControl.py",
        "desc":    "Drive the robot manually. Useful for finding handoff "
                   "coordinates or recovering from a stuck state.",
        "color":   "green",
    },
    {
        "section": "TOOLS",
        "name":    "Test Dobot",
        "file":    "testDobot.py",
        "desc":    "Sanity check: arm moves and gripper opens/closes. "
                   "Run first if something is broken.",
        "color":   "green",
    },
]

# ── Process tracking (thread-safe) ────────────────────────
_proc_lock = threading.Lock()
_processes: dict[str, subprocess.Popen] = {}     # file -> Popen

def _is_running(filename: str) -> bool:
    with _proc_lock:
        p = _processes.get(filename)
        return p is not None and p.poll() is None

def _launch(tool) -> tuple[bool, str]:
    path = SCRIPT_DIR / tool["file"]
    if not path.exists():
        return False, f"Missing file: {tool['file']}"
    if _is_running(tool["file"]):
        return False, "Already running"
    cmd = [sys.executable, str(path)] + tool.get("args", [])
    try:
        proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR))
        with _proc_lock:
            _processes[tool["file"]] = proc
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
  #splash { position: fixed; inset: 0; z-index: 100;
            background: radial-gradient(circle at 50% 40%, #1a1d24 0%, #0e1015 70%);
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; gap: 36px;
            transition: opacity .5s ease, visibility .5s ease; }
  #splash.hidden { opacity: 0; visibility: hidden; pointer-events: none; }
  #splash .label { color: var(--text-dim); font-size: 12px;
                   letter-spacing: .3em; text-transform: uppercase; }
  #splash h1 { margin: 0; font-size: 36px; font-weight: 700; color: var(--text); }
  #splash h1 span { color: var(--accent); }
  #splash .pulse-wrap { position: absolute; inset: 0;
                         display: flex; align-items: center; justify-content: center;
                         pointer-events: none; overflow: hidden; z-index: 0; }
  #splash .pulse-wrap .ring {
      position: absolute; top: 50%; left: 50%;
      width: 120px; height: 120px;
      margin-top: -60px; margin-left: -60px;
      border: 2px solid var(--accent);
      border-radius: 50%; opacity: 0;
      animation: radar 4s ease-out infinite;
  }
  #splash .pulse-wrap .ring:nth-child(2) { animation-delay: 0.8s; }
  #splash .pulse-wrap .ring:nth-child(3) { animation-delay: 1.6s; }
  #splash .pulse-wrap .ring:nth-child(4) { animation-delay: 2.4s; }
  #splash .pulse-wrap .ring:nth-child(5) { animation-delay: 3.2s; }
  #splash .pulse-wrap .core {
      width: 24px; height: 24px; border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 36px var(--accent);
      animation: corepulse 1.6s ease-in-out infinite;
      z-index: 1;
  }
  /* All splash content sits above the radar */
  #splash > *:not(.pulse-wrap) { position: relative; z-index: 2; }
  @keyframes radar {
      0%   { transform: scale(0.2);  opacity: 0.85; }
      80%  { opacity: 0.06; }
      100% { transform: scale(20);   opacity: 0; }
  }
  @keyframes corepulse {
      0%, 100% { transform: scale(1);   filter: brightness(1); }
      50%      { transform: scale(1.2); filter: brightness(1.4); }
  }
  #connectBtn { background: var(--accent); color: #fff; border: 0;
                padding: 16px 56px; font-size: 16px; font-weight: 700;
                letter-spacing: .08em; border-radius: 8px; cursor: pointer;
                box-shadow: 0 6px 20px rgba(229,57,53,0.35);
                transition: transform .08s ease, filter .15s ease,
                            box-shadow .15s ease; }
  #connectBtn:hover  { filter: brightness(1.1);
                       box-shadow: 0 8px 28px rgba(229,57,53,0.55); }
  #connectBtn:active { transform: translateY(1px); }
  #splash .meta { color: var(--text-dim); font-size: 12px; }

  /* ── Main dashboard ───────────────────────────────────── */
  header { padding: 28px 36px 16px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 28px; font-weight: 700;
              display: flex; align-items: baseline; gap: 12px; }
  header h1 .sub { color: var(--text); font-weight: 400; font-size: 18px; }
  header .meta { float: right; color: var(--text-dim); font-size: 12px;
                 margin-top: 12px; }
  header p { color: var(--text-dim); margin: 4px 0 0; font-size: 13px; }

  main { padding: 24px 28px 80px; display: grid;
         grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
         gap: 16px; max-width: 1400px; margin: 0 auto; }

  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 10px; overflow: hidden;
          transition: transform .12s ease, border-color .12s ease; }
  .card:hover { border-color: var(--text-dim); }

  .card-head { padding: 10px 16px; display: flex; align-items: center;
               justify-content: space-between; color: #fff; font-size: 11px;
               font-weight: 700; letter-spacing: .08em; }
  .card-head .file { font-family: "SF Mono", Menlo, monospace;
                     font-weight: 400; opacity: .9; font-size: 11px; }
  .card-head.red   { background: var(--red);   }
  .card-head.amber { background: var(--amber); color: #1a1d24; }
  .card-head.green { background: var(--green); }

  .card-body { padding: 16px 18px 18px; }
  .card-body h2 { margin: 0 0 6px; font-size: 16px; font-weight: 700; }
  .card-body p  { margin: 0 0 14px; font-size: 13px; line-height: 1.5;
                  color: var(--text-dim); }

  .btn { display: inline-flex; align-items: center; gap: 8px;
         padding: 9px 18px; border: 0; border-radius: 7px;
         font-size: 13px; font-weight: 700; cursor: pointer;
         color: #fff; transition: filter .12s ease, transform .05s ease; }
  .btn:hover { filter: brightness(1.15); }
  .btn:active { transform: translateY(1px); }
  .btn.red    { background: var(--red);   }
  .btn.amber  { background: var(--amber); color: #1a1d24; }
  .btn.green  { background: var(--green); }
  .btn.stop   { background: #555c66; }

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
  <div class="label">Toyota Innovation Challenge 2026</div>
  <h1>Collaborative <span>Robotics</span></h1>
  <div class="pulse-wrap">
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="core"></div>
  </div>
  <button id="connectBtn">CONNECT</button>
  <div class="meta">Local control center · 127.0.0.1</div>
</div>

<header>
  <span class="meta">Toyota Innovation Challenge 2026</span>
  <h1><span class="sub">Collaborative Robotics Control Center</span></h1>
  <p>Click any card to launch a script. Click again to stop it.</p>
</header>
<main id="cards"></main>

<footer>
  <span class="dot" id="dot"></span>
  <span class="status" id="status">Nothing running.</span>
  <button class="btn stopall" onclick="stopAll()">Stop All</button>
</footer>
<script>
  const TOOLS = __TOOLS_JSON__;

  function render() {
    const main = document.getElementById('cards');
    main.innerHTML = '';
    for (const t of TOOLS) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="card-head ${t.color}">
          <span>${t.section}</span>
          <span class="file">${t.file}</span>
        </div>
        <div class="card-body">
          <h2>${t.name}</h2>
          <p>${t.desc}</p>
          <button class="btn ${t.color}" data-file="${t.file}">▶  Launch</button>
        </div>`;
      card.querySelector('button').addEventListener('click', () => toggle(t));
      main.appendChild(card);
    }
  }

  async function getStatus() {
    const r = await fetch('/api/status');
    const j = await r.json();
    return j.running;
  }

  async function toggle(tool) {
    const running = (await getStatus()).includes(tool.file);
    const ep = running ? '/api/stop' : '/api/launch';
    await fetch(`${ep}?file=${encodeURIComponent(tool.file)}`, {method: 'POST'});
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
      const tool = TOOLS.find(t => t.file === f);
      const isRunning = running.includes(f);
      btn.textContent = isRunning ? '■  Stop' : '▶  Launch';
      btn.className = 'btn ' + (isRunning ? 'stop' : tool.color);
    });
    const dot = document.getElementById('dot');
    const status = document.getElementById('status');
    if (running.length) {
      dot.classList.add('live');
      status.classList.add('live');
      const names = running.map(f => TOOLS.find(t => t.file === f).name).join(', ');
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

  // Splash → dashboard transition
  document.getElementById('connectBtn').addEventListener('click', () => {
    document.getElementById('splash').classList.add('hidden');
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
            tools_safe = [{k: v for k, v in t.items() if k != "args"} for t in TOOLS]
            html = HTML.replace("__TOOLS_JSON__", json.dumps(tools_safe))
            return self._send_html(html)
        if url.path == "/api/status":
            return self._send_json(200, {"running": _running_files()})
        self.send_response(404); self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        fname = q.get("file", [None])[0]

        if url.path == "/api/launch" and fname:
            tool = next((t for t in TOOLS if t["file"] == fname), None)
            if not tool:
                return self._send_json(404, {"ok": False, "msg": "unknown tool"})
            ok, msg = _launch(tool)
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
