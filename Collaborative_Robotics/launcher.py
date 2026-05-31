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
        "name":    "Operator Monitor",
        "file":    "operator_monitor.py",
        "desc":    "Laptop webcam: identifies the operator from photos in "
                   "operators/ AND runs fatigue detection on the same feed. "
                   "Press T for the task dashboard.",
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
        "name":    "Manual Control",
        "file":    "manualControl.py",
        "desc":    "Drive the robot manually. Useful for finding handoff "
                   "coordinates or recovering from a stuck state.",
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
            background: radial-gradient(circle at 50% 50%, #1a1d24 0%, #0a0c10 80%);
            display: flex; flex-direction: column; align-items: center;
            justify-content: center;
            transition: opacity .5s ease, visibility .5s ease; }
  #splash.hidden { opacity: 0; visibility: hidden; pointer-events: none; }


  /* Radar (full-screen, behind everything) */
  #splash .pulse-wrap { position: absolute; inset: 0;
                         display: flex; align-items: center; justify-content: center;
                         pointer-events: none; overflow: hidden; z-index: 0; }
  #splash .pulse-wrap .ring {
      position: absolute; top: 50%; left: 50%;
      width: 180px; height: 180px;
      margin-top: -90px; margin-left: -90px;
      border: 2px solid var(--accent);
      border-radius: 50%; opacity: 0;
      animation: radar 4.5s ease-out infinite;
  }
  #splash .pulse-wrap .ring:nth-child(2) { animation-delay: 0.9s; }
  #splash .pulse-wrap .ring:nth-child(3) { animation-delay: 1.8s; }
  #splash .pulse-wrap .ring:nth-child(4) { animation-delay: 2.7s; }
  #splash .pulse-wrap .ring:nth-child(5) { animation-delay: 3.6s; }
  #splash > *:not(.pulse-wrap) { position: relative; z-index: 2; }
  @keyframes radar {
      0%   { transform: scale(0.15); opacity: 0.9; }
      80%  { opacity: 0.06; }
      100% { transform: scale(20);   opacity: 0; }
  }

  /* Hero: logo + brand text side-by-side, slightly left of center */
  #splash .hero { display: flex; flex-direction: column; align-items: center;
                  gap: 56px; }
  #splash .hero .lockup { display: flex; flex-direction: row;
                          align-items: center; justify-content: center;
                          gap: 14px;
                          /* shift left of dead-center a bit */
                          margin-left: -40px; }
  #splash .hero img.tmmc-logo { height: 160px; width: auto;
                                 filter: drop-shadow(0 4px 24px rgba(229,57,53,0.25)); }
  #splash .hero .brand-text { color: #fff; font-weight: 600;
                              font-size: 36px; line-height: 1.08;
                              letter-spacing: -0.01em; text-align: left; }
  /* Fallback text logo if image fails */
  #splash .hero .lockup.broken img.tmmc-logo { display: none; }
  #splash .hero .lockup .fallback { display: none; background: var(--accent);
                                     color: #fff; font-weight: 800;
                                     font-size: 30px; letter-spacing: .12em;
                                     padding: 18px 26px; border-radius: 10px;
                                     box-shadow: 0 8px 32px rgba(229,57,53,0.45); }
  #splash .hero .lockup.broken .fallback { display: block; }

  #connectBtn { background: var(--accent); color: #fff; border: 0;
                padding: 18px 72px; font-size: 15px; font-weight: 700;
                letter-spacing: .25em; border-radius: 8px; cursor: pointer;
                box-shadow: 0 6px 24px rgba(229,57,53,0.4);
                transition: transform .08s ease, filter .15s ease,
                            box-shadow .15s ease; }
  #connectBtn:hover  { filter: brightness(1.1);
                       box-shadow: 0 10px 36px rgba(229,57,53,0.6); }
  #connectBtn:active { transform: translateY(1px); }

  #splash .meta { position: absolute; bottom: 32px; left: 50%;
                  transform: translateX(-50%);
                  color: var(--text-dim); font-size: 11px;
                  letter-spacing: .15em; text-transform: uppercase; }

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
  <div class="pulse-wrap">
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
    <div class="ring"></div>
  </div>

  <div class="hero">
    <div class="lockup" id="lockup">
      <img class="tmmc-logo" src="/static/tmmc_logo.png?v=__CACHEBUST__" alt="TMMC"
           onerror="document.getElementById('lockup').classList.add('broken')">
      <div class="fallback">TMMC</div>
      <div class="brand-text">Toyota Motor<br>Manufacturing<br>Canada Inc.</div>
    </div>
    <button id="connectBtn">CONNECT</button>
  </div>

  <div class="meta">Collaborative Robotics · Innovation Challenge 2026 · 127.0.0.1</div>
</div>

<header>
  <div class="tmmc-banner">
    <img class="logo" src="/static/tmmc_logo.png?v=__CACHEBUST__" alt="TMMC"
         onerror="this.style.display='none'">
    <div class="title-block">
      <div class="kicker">Collaborative Robotics</div>
      <h1>Control Center</h1>
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
            import time as _t
            tools_safe = [{k: v for k, v in t.items() if k != "args"} for t in TOOLS]
            html = HTML.replace("__TOOLS_JSON__", json.dumps(tools_safe))
            html = html.replace("__CACHEBUST__", str(int(_t.time())))
            return self._send_html(html)
        if url.path == "/api/status":
            return self._send_json(200, {"running": _running_files()})
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
