#!/usr/bin/env python3
"""
Video frame trimmer with a browser-based GUI (for headless / remote servers).

Usage (on the server):
    python3 video_trimmer_web.py --input /path/to/input_video.mp4 --output /path/to/output_video.mp4
    # optionally: --host 127.0.0.1 --port 5000

Then, from your LOCAL machine, open an SSH tunnel to the server and browse
to it locally (this keeps the app off the open network):

    ssh -L 5000:localhost:5000 <user>@<gpu-server>

...and open http://localhost:5000 in your local browser.

Dependencies (on the server):
    pip install flask opencv-python

In the browser:
    - Drag the scrub slider (or click/drag the timeline bar) to preview any frame.
    - Click "Set to current" next to Start/End to mark the frame you're looking at,
      or type a frame number directly into the Start/End boxes.
    - The timeline bar highlights the selected range in blue.
    - "Preview selection" plays back just the selected frame range in the browser.
    - "Export clip" writes frames [start, end] (inclusive) to the output path on the server.
"""

import argparse
import os
import sys
import threading

try:
    import cv2
except ImportError:
    sys.exit("This tool requires opencv-python. Install it with: pip install opencv-python")

try:
    from flask import Flask, Response, jsonify, request
except ImportError:
    sys.exit("This tool requires Flask. Install it with: pip install flask")


app = Flask(__name__)

STATE = {
    "input_path": None,
    "output_path": None,
    "fps": 30.0,
    "total_frames": 0,
    "width": 0,
    "height": 0,
    "cap": None,
    "cap_lock": threading.Lock(),
    "preview_max_w": 960,
}

EXPORT_STATE = {
    "running": False,
    "count": 0,
    "total": 0,
    "done": False,
    "error": None,
    "output": None,
}
EXPORT_LOCK = threading.Lock()


INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Video Trimmer</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#1b1b1f; color:#eee;
         max-width: 820px; margin: 24px auto; padding: 0 16px; }
  h2 { font-weight: 600; margin-bottom: 4px; }
  #info { color:#9aa0a6; font-size: 13px; margin-bottom: 12px; }
  #frame { display:block; width:100%; max-width:100%; background:#000; border:1px solid #444; border-radius:4px; }
  .row { display:flex; align-items:center; gap:8px; margin-top:10px; flex-wrap: wrap; }
  input[type=range] { flex:1; }
  #timeline { position:relative; height:28px; background:#2a2a2e; border-radius:4px; margin-top:6px;
              cursor:pointer; overflow:hidden; }
  #selection { position:absolute; top:0; bottom:0; background:#3a7bd5; opacity:0.7; }
  #cursor { position:absolute; top:0; bottom:0; width:2px; background:#fff; }
  label { color:#c8c8cc; font-size: 13px; }
  input[type=number] { width:80px; background:#2a2a2e; color:#eee; border:1px solid #444; border-radius:4px; padding:4px; }
  input[type=text] { background:#2a2a2e; color:#eee; border:1px solid #444; border-radius:4px; padding:4px; flex:1; }
  button { background:#3a7bd5; color:#fff; border:none; border-radius:4px; padding:7px 14px; cursor:pointer; }
  button:hover { background:#2f66b3; }
  button:disabled { background:#555; cursor:not-allowed; }
  button.secondary { background:#444; }
  button.secondary:hover { background:#555; }
  #progressWrap { background:#2a2a2e; border-radius:4px; height:10px; margin-top:10px; overflow:hidden; display:none; }
  #progressBar { background:#3a7bd5; height:100%; width:0%; }
  #status { margin-top:8px; font-size: 13px; color:#9aa0a6; }
  #selLabel { margin-top:8px; font-size: 14px; }
</style>
</head>
<body>
  <h2>Video Trimmer</h2>
  <div id="info">loading…</div>

  <img id="frame" src="">

  <div class="row">
    <input id="scrub" type="range" min="0" max="0" value="0">
    <span id="posLabel" style="min-width:170px;">-</span>
  </div>

  <div id="timeline">
    <div id="selection"></div>
    <div id="cursor"></div>
  </div>

  <div class="row">
    <label>Start frame:</label>
    <input id="startInput" type="number" min="0" value="0">
    <button class="secondary" onclick="setStartToCurrent()">Set to current</button>
    <label style="margin-left:16px;">End frame:</label>
    <input id="endInput" type="number" min="0" value="0">
    <button class="secondary" onclick="setEndToCurrent()">Set to current</button>
  </div>

  <div id="selLabel"></div>

  <div class="row">
    <label>Output path:</label>
    <input id="outputInput" type="text" value="">
  </div>

  <div class="row">
    <button id="previewBtn" onclick="togglePreview()">Preview selection ▶</button>
    <button id="exportBtn" onclick="exportClip()">Export clip</button>
  </div>

  <div id="progressWrap"><div id="progressBar"></div></div>
  <div id="status"></div>

<script>
let totalFrames = 0, fps = 30, currentFrame = 0, startFrame = 0, endFrame = 0;
let playing = false, playTimer = null;
let fetchInFlight = false, pendingFrame = null;

async function init() {
  const res = await fetch('/api/info');
  const info = await res.json();
  totalFrames = info.total_frames;
  fps = info.fps;
  startFrame = 0;
  endFrame = totalFrames - 1;

  document.getElementById('info').textContent =
    `${info.name}  |  ${info.width}x${info.height}  |  ${fps.toFixed(2)} fps  |  ${totalFrames} frames total`;
  document.getElementById('scrub').max = totalFrames - 1;
  document.getElementById('startInput').max = totalFrames - 1;
  document.getElementById('endInput').max = totalFrames - 1;
  document.getElementById('startInput').value = startFrame;
  document.getElementById('endInput').value = endFrame;
  document.getElementById('outputInput').value = info.output;

  showFrame(0);
  updateSelectionUI();
}

function posText(idx) {
  const t = fps ? idx / fps : 0;
  return `Frame ${idx} / ${totalFrames - 1}  (${t.toFixed(2)}s)`;
}

function showFrame(idx) {
  idx = Math.max(0, Math.min(idx, totalFrames - 1));
  currentFrame = idx;
  document.getElementById('posLabel').textContent = posText(idx);
  document.getElementById('scrub').value = idx;
  updateCursor();

  if (fetchInFlight) { pendingFrame = idx; return; }
  fetchInFlight = true;
  const img = document.getElementById('frame');
  img.src = `/frame/${idx}?t=${Date.now()}`;
  img.onload = img.onerror = () => {
    fetchInFlight = false;
    if (pendingFrame !== null) {
      const next = pendingFrame;
      pendingFrame = null;
      showFrame(next);
    }
  };
}

function updateCursor() {
  const tl = document.getElementById('timeline');
  const w = tl.clientWidth;
  const frac = totalFrames > 1 ? currentFrame / (totalFrames - 1) : 0;
  document.getElementById('cursor').style.left = (frac * w) + 'px';
}

function updateSelectionUI() {
  const tl = document.getElementById('timeline');
  const w = tl.clientWidth;
  const f0 = totalFrames > 1 ? startFrame / (totalFrames - 1) : 0;
  const f1 = totalFrames > 1 ? endFrame / (totalFrames - 1) : 0;
  const sel = document.getElementById('selection');
  sel.style.left = (f0 * w) + 'px';
  sel.style.width = ((f1 - f0) * w) + 'px';
  updateCursor();

  const n = endFrame - startFrame + 1;
  const dur = fps ? n / fps : 0;
  document.getElementById('selLabel').textContent =
    `Selected: frames ${startFrame}–${endFrame}  (${n} frames, ${dur.toFixed(2)}s)`;
}

document.getElementById('scrub').addEventListener('input', (e) => {
  showFrame(parseInt(e.target.value, 10));
});

document.getElementById('timeline').addEventListener('click', (e) => {
  const tl = document.getElementById('timeline');
  const rect = tl.getBoundingClientRect();
  const frac = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
  showFrame(Math.round(frac * (totalFrames - 1)));
});

function setStartToCurrent() {
  startFrame = Math.min(currentFrame, endFrame);
  document.getElementById('startInput').value = startFrame;
  updateSelectionUI();
}
function setEndToCurrent() {
  endFrame = Math.max(currentFrame, startFrame);
  document.getElementById('endInput').value = endFrame;
  updateSelectionUI();
}
document.getElementById('startInput').addEventListener('change', (e) => {
  let v = Math.max(0, Math.min(parseInt(e.target.value || '0', 10), totalFrames - 1));
  if (v > endFrame) v = endFrame;
  startFrame = v;
  e.target.value = v;
  updateSelectionUI();
});
document.getElementById('endInput').addEventListener('change', (e) => {
  let v = Math.max(0, Math.min(parseInt(e.target.value || '0', 10), totalFrames - 1));
  if (v < startFrame) v = startFrame;
  endFrame = v;
  e.target.value = v;
  updateSelectionUI();
});

function togglePreview() {
  const btn = document.getElementById('previewBtn');
  if (playing) {
    playing = false;
    clearTimeout(playTimer);
    btn.textContent = 'Preview selection ▶';
    return;
  }
  playing = true;
  btn.textContent = 'Stop preview ■';
  let idx = startFrame;
  const delay = fps ? 1000 / fps : 33;
  const step = () => {
    if (!playing) return;
    if (idx > endFrame) {
      playing = false;
      btn.textContent = 'Preview selection ▶';
      return;
    }
    showFrame(idx);
    idx += 1;
    playTimer = setTimeout(step, delay);
  };
  step();
}

async function exportClip() {
  const exportBtn = document.getElementById('exportBtn');
  const output = document.getElementById('outputInput').value.trim();
  if (!output) { alert('Please enter an output path.'); return; }
  if (endFrame < startFrame) { alert('End frame must be >= start frame.'); return; }

  exportBtn.disabled = true;
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('status').textContent = 'Starting export…';

  const res = await fetch('/api/export', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({start: startFrame, end: endFrame, output: output})
  });
  const data = await res.json();
  if (data.status === 'error') {
    document.getElementById('status').textContent = 'Error: ' + data.message;
    exportBtn.disabled = false;
    return;
  }
  pollExport();
}

async function pollExport() {
  const res = await fetch('/api/export/status');
  const s = await res.json();
  const pct = s.total ? Math.round(100 * s.count / s.total) : 0;
  document.getElementById('progressBar').style.width = pct + '%';

  if (s.running) {
    document.getElementById('status').textContent = `Exporting… ${s.count}/${s.total} frames`;
    setTimeout(pollExport, 300);
  } else {
    document.getElementById('exportBtn').disabled = false;
    if (s.error) {
      document.getElementById('status').textContent = 'Error: ' + s.error;
    } else if (s.done) {
      document.getElementById('status').textContent = `Done — exported ${s.count} frames to ${s.output}`;
    }
  }
}

window.addEventListener('resize', () => { updateCursor(); updateSelectionUI(); });
init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/info")
def api_info():
    return jsonify({
        "name": os.path.basename(STATE["input_path"]),
        "fps": STATE["fps"],
        "total_frames": STATE["total_frames"],
        "width": STATE["width"],
        "height": STATE["height"],
        "output": STATE["output_path"],
    })


@app.route("/frame/<int:idx>")
def frame(idx):
    idx = max(0, min(idx, STATE["total_frames"] - 1))
    with STATE["cap_lock"]:
        cap = STATE["cap"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
    if not ret:
        return "", 404

    max_w = STATE["preview_max_w"]
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "", 500
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json(force=True)
    start = int(data.get("start", 0))
    end = int(data.get("end", 0))
    output_path = data.get("output") or STATE["output_path"]

    if end < start:
        return jsonify({"status": "error", "message": "End frame must be >= start frame."})

    with EXPORT_LOCK:
        if EXPORT_STATE["running"]:
            return jsonify({"status": "busy", "message": "An export is already running."})
        EXPORT_STATE.update({"running": True, "count": 0, "total": end - start + 1,
                              "done": False, "error": None, "output": output_path})

    threading.Thread(target=_run_export, args=(start, end, output_path), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/export/status")
def api_export_status():
    with EXPORT_LOCK:
        return jsonify(dict(EXPORT_STATE))


def _run_export(start, end, output_path):
    try:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        ext = os.path.splitext(output_path)[1].lower()
        fourcc_map = {".avi": "XVID", ".mov": "mp4v", ".mp4": "mp4v", ".mkv": "mp4v"}
        fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(ext, "mp4v"))
        writer = cv2.VideoWriter(output_path, fourcc, STATE["fps"], (STATE["width"], STATE["height"]))
        if not writer.isOpened():
            raise RuntimeError("Could not open the output file for writing.")

        cap = cv2.VideoCapture(STATE["input_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        count = 0
        for _ in range(start, end + 1):
            ret, img = cap.read()
            if not ret:
                break
            writer.write(img)
            count += 1
            with EXPORT_LOCK:
                EXPORT_STATE["count"] = count
        cap.release()
        writer.release()

        with EXPORT_LOCK:
            EXPORT_STATE["running"] = False
            EXPORT_STATE["done"] = True
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        with EXPORT_LOCK:
            EXPORT_STATE["running"] = False
            EXPORT_STATE["error"] = str(exc)


def parse_args():
    parser = argparse.ArgumentParser(description="Crop a video to a frame range via a browser GUI.")
    parser.add_argument("-i", "--input", required=True, help="Path to the raw input video file.")
    parser.add_argument("-o", "--output", required=True, help="Default path to write the cropped output video.")
    parser.add_argument("--host", default="127.0.0.1",
                         help="Interface to bind to (default: 127.0.0.1, i.e. only reachable via SSH tunnel).")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on (default: 5000).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        sys.exit(f"Input video not found: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.input}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        sys.exit("Could not determine the frame count for this video.")

    STATE.update({
        "input_path": args.input,
        "output_path": args.output,
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": total_frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "cap": cap,
    })

    print(f"Serving on http://{args.host}:{args.port}")
    print("If this is a remote server, tunnel to it from your local machine with:")
    print(f"    ssh -L {args.port}:localhost:{args.port} <user>@<server>")
    print(f"...then open http://localhost:{args.port} in your local browser.")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
