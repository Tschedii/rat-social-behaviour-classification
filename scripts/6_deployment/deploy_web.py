"""Browser version of gui.py: pick a checkpoint + 6 camera folders on the server, run
inference, save the per-frame label JSON and log the window predictions for review. Also
serves the review page (review_web.py) at /review, so one server and one tunnel cover both.

On the server (inside tmux is fine, a run keeps going if your SSH connection drops):

    python3 deploy_web.py [--port 8980]

On your Mac:

    ssh -L 8980:localhost:8980 <user>@<gpu-server>
    open http://localhost:8980            # deploy
    open http://localhost:8980/review     # review queue
    open http://localhost:8980/retrain    # retraining cycle (retrain_web.py)

Paths are browsed on the SERVER's filesystem. Settings persist to the same
~/.rat_deploy_gui_config.json gui.py uses, so both front ends share the last-used values.
Only one GPU job (inference or retraining) runs at a time; closing the browser tab does not
stop it.
"""
import argparse
import json
import os
import re
import sqlite3
import threading
import time
import traceback
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_web  # noqa: E402
import review_web  # noqa: E402
import web_common  # noqa: E402
from correction_log import DEFAULT_DB_PATH, init_db  # noqa: E402
from raw_frame_source import FRAME_RE  # noqa: E402

CONFIG_PATH = Path.home() / ".rat_deploy_gui_config.json"
DEFAULT_SUBDIRS = {"rgb": "cutout_all_objects", "mask1": "mask_object_1", "mask2": "mask_object_2"}
CAM_DIR_RE = re.compile(r"cam(?:era)?[_-]?([1-6])$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_config():
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    cfg = {k: v for k, v in cfg.items() if v is not None}
    cfg.setdefault("checkpoint", "")
    cams = list(cfg.get("cameras") or [])
    cfg["cameras"] = (cams + [""] * 6)[:6]
    cfg.setdefault("rat_choice", "both")
    cfg.setdefault("stride", 32)
    cfg["subdirs"] = {**DEFAULT_SUBDIRS, **{k: v for k, v in (cfg.get("subdirs") or {}).items() if v}}
    cfg.setdefault("out_path", str(Path.cwd() / "deploy_output.json"))
    cfg.setdefault("session_label", "")
    cfg.setdefault("log_predictions", True)
    cfg.setdefault("transform_meta_root", "")
    return cfg


def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# filesystem helpers (server side)
# ---------------------------------------------------------------------------

def list_dir(path, exts):
    p = Path(path or Path.home()).expanduser()
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    dirs, files = [], []
    try:
        entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
    except PermissionError:
        entries = []
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                dirs.append(e.name)
            elif exts and any(e.name.lower().endswith(x) for x in exts):
                files.append(e.name)
        except OSError:
            pass
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "files": files}


def count_frames(path):
    """Frame count + range of a frame_XXXXXX.<ext> folder, via scandir (fast on big folders)."""
    n, lo, hi = 0, None, None
    with os.scandir(path) as it:
        for e in it:
            m = FRAME_RE.search(e.name)
            if m:
                k = int(m.group(1))
                n += 1
                lo = k if lo is None else min(lo, k)
                hi = k if hi is None else max(hi, k)
    return n, lo, hi


def check_camera(cam_dir, subdirs):
    out = {"path": cam_dir, "ok": False, "parts": {}, "error": None}
    if not cam_dir:
        out["error"] = "not set"
        return out
    p = Path(cam_dir).expanduser()
    if not p.is_dir():
        out["error"] = "folder not found"
        return out
    ok = True
    for key in ("rgb", "mask1", "mask2"):
        sub = p / subdirs[key]
        if not sub.is_dir():
            out["parts"][key] = {"name": subdirs[key], "missing": True}
            ok = False
            continue
        n, lo, hi = count_frames(sub)
        out["parts"][key] = {"name": subdirs[key], "n": n, "lo": lo, "hi": hi}
        ok = ok and n > 0
    out["has_transform_meta"] = (p / "transform_meta.csv").exists()
    out["ok"] = ok
    return out


def auto_cameras(session_dir):
    """cam_1..cam_6 (or cam1, camera_1, ...) subfolders of a session folder, in order."""
    p = Path(session_dir).expanduser()
    found = {}
    for e in os.scandir(p):
        m = CAM_DIR_RE.search(e.name)
        if e.is_dir() and m:
            found.setdefault(int(m.group(1)), str(p / e.name))
    return {"cameras": [found.get(i, "") for i in range(1, 7)], "session_label": p.name}


def existing_session_rows(session_label):
    conn = init_db(review_web.DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE session_label = ?", (session_label,)).fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# inference job
# ---------------------------------------------------------------------------

class Cancelled(Exception):
    pass


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.status = "idle"   # idle | loading | running | done | error | cancelled
        self.done = 0
        self.total = 0
        self.log = []
        self.summary = None
        self.error = None
        self.started = None
        self.finished = None
        self.params = None
        self.cancel_requested = False

    def snapshot(self, since=0):
        with self.lock:
            return {
                "status": self.status, "done": self.done, "total": self.total,
                "log": self.log[since:], "log_len": len(self.log),
                "summary": self.summary, "error": self.error,
                "started": self.started, "finished": self.finished,
                "now": time.time(), "params": self.params,
            }

    def add_log(self, msg):
        with self.lock:
            for line in str(msg).splitlines() or [""]:
                self.log.append(line)

    def busy(self):
        return self.status in ("loading", "running")

    def start(self, params):
        with self.lock:
            if self.busy():
                raise RuntimeError("a run is already in progress")
            self.reset()
            self.status = "loading"
            self.started = time.time()
            self.params = params
        threading.Thread(target=self._run, args=(params,), daemon=True).start()

    def _run(self, p):
        try:
            self.add_log(f"loading checkpoint {p['checkpoint']} …")
            from infer_session import SessionInferer

            subdirs = {"rgb": p["subdirs"]["rgb"], 1: p["subdirs"]["mask1"], 2: p["subdirs"]["mask2"]}
            inferer = SessionInferer(p["checkpoint"], subdirs=subdirs, progress_log=self.add_log,
                                     transform_meta_root=p["transform_meta_root"] or None)
            with self.lock:
                self.status = "running"

            def progress_cb(done, total):
                with self.lock:
                    self.done, self.total = done, total
                    if self.cancel_requested:
                        raise Cancelled()

            result = inferer.run(p["cameras"], rats=p["rats"], stride=p["stride"], progress_cb=progress_cb)
            out_path = Path(p["out_path"]).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            self.add_log(f"wrote {out_path}")

            summary = {"out_path": str(out_path), "session_label": p["session_label"], "rats": {},
                       "logged": {}, "frame_range": result.get("session_frame_range"),
                       "scale_stats_ok": result.get("scale_stats_ok")}
            for rat_key, frames in result["rats"].items():
                counts = {}
                for f in frames:
                    counts[f["label"]] = counts.get(f["label"], 0) + 1
                summary["rats"][rat_key] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
                self.add_log(f"{rat_key}: {len(frames)} frames labeled — " +
                             ", ".join(f"{k}={v}" for k, v in summary["rats"][rat_key].items()))

            if p["log_predictions"]:
                from correction_log import log_window_predictions
                for rat in p["rats"]:
                    n = log_window_predictions(review_web.DB_PATH, p["session_label"], p["cameras"],
                                               p["checkpoint"], rat,
                                               result["window_predictions"][f"rat_{rat}"],
                                               subdirs=p["subdirs"])
                    summary["logged"][f"rat_{rat}"] = n
                    self.add_log(f"logged {n} rat_{rat} window predictions to the correction log "
                                 f"(session '{p['session_label']}')")
            with self.lock:
                self.summary = summary
                self.status = "done"
        except Cancelled:
            self.add_log("cancelled — nothing written, nothing logged")
            with self.lock:
                self.status = "cancelled"
        except Exception as e:
            self.add_log(f"ERROR: {type(e).__name__}: {e}")
            self.add_log(traceback.format_exc())
            with self.lock:
                self.status = "error"
                self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.finished = time.time()


JOB = Job()


def validate(body):
    """Returns (params, errors, warnings). Warnings can be overridden with force=true."""
    errors, warnings = [], []
    checkpoint = (body.get("checkpoint") or "").strip()
    cameras = [(c or "").strip() for c in body.get("cameras", [])]
    subdirs = {**DEFAULT_SUBDIRS, **(body.get("subdirs") or {})}
    out_path = (body.get("out_path") or "").strip()
    transform_meta_root = (body.get("transform_meta_root") or "").strip()

    if not checkpoint:
        errors.append("Select a checkpoint file.")
    elif not Path(checkpoint).expanduser().is_file():
        errors.append(f"Checkpoint not found: {checkpoint}")
    if len(cameras) != 6 or any(not c for c in cameras):
        errors.append("All 6 camera folders must be set.")
    else:
        for i, c in enumerate(cameras, start=1):
            chk = check_camera(c, subdirs)
            if not chk["ok"]:
                missing = [v["name"] for v in chk["parts"].values() if v.get("missing") or not v.get("n")]
                errors.append(f"Camera {i}: {chk['error'] or 'missing/empty ' + ', '.join(missing)}")
        if len(set(cameras)) != 6:
            errors.append("The same folder is set for more than one camera.")
    try:
        stride = int(body.get("stride", 32))
        if stride <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Stride must be a positive integer.")
        stride = 32
    if not out_path:
        errors.append("Choose an output JSON path.")
    if transform_meta_root and not Path(transform_meta_root).expanduser().is_dir():
        errors.append(f"Metadata folder not found: {transform_meta_root}")

    rat_choice = str(body.get("rat_choice", "both"))
    rats = (1, 2) if rat_choice == "both" else (int(rat_choice),)
    session_label = (body.get("session_label") or "").strip() or (Path(out_path).stem if out_path else "")
    log_predictions = bool(body.get("log_predictions", True))

    if not errors:
        if Path(out_path).expanduser().exists():
            warnings.append(f"{out_path} already exists and will be overwritten.")
        if log_predictions:
            n = existing_session_rows(session_label)
            if n:
                warnings.append(f"The correction log already has {n} rows for session '{session_label}'. "
                                f"Running again logs every window a second time (duplicates in the review "
                                f"queue and in the retraining export). Use a new session label unless "
                                f"that's intended.")
            if stride < 32:
                warnings.append("Stride < 32 logs overlapping windows, which become overlapping label "
                                "segments if exported for retraining. Use 32 for runs you plan to review.")

    params = {"checkpoint": checkpoint, "cameras": cameras, "subdirs": subdirs, "stride": stride,
              "rat_choice": rat_choice, "rats": rats, "out_path": out_path,
              "session_label": session_label, "log_predictions": log_predictions,
              "transform_meta_root": transform_meta_root}
    return params, errors, warnings


# ---------------------------------------------------------------------------
# HTTP: deploy routes here, everything else (review page + its /api/*) from review_web
# ---------------------------------------------------------------------------

class Handler(review_web.Handler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self._send(200, web_common.render(PAGE), "text/html; charset=utf-8")
            if url.path == "/review":
                return self._send(200, review_web.PAGE, "text/html; charset=utf-8")
            if url.path == "/dapi/config":
                return self._send(200, load_config())
            if url.path == "/dapi/ls":
                exts = [x for x in q.get("exts", "").split(",") if x]
                return self._send(200, list_dir(q.get("path"), exts))
            if url.path == "/dapi/check":
                subdirs = {k: q.get(k) or v for k, v in DEFAULT_SUBDIRS.items()}
                return self._send(200, check_camera(q.get("path", ""), subdirs))
            if url.path == "/dapi/autocams":
                return self._send(200, auto_cameras(q["path"]))
            if url.path == "/dapi/job":
                return self._send(200, JOB.snapshot(int(q.get("since", 0))))
            if url.path == "/retrain" or url.path.startswith("/rapi/"):
                if retrain_web.handle_get(self, url.path, q, review_web.DB_PATH, load_config()) is None:
                    return self._send(404, {"error": "not found"})
                return
        except FileNotFoundError as e:
            return self._send(404, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if not url.path.startswith(("/dapi/", "/rapi/")):
            return super().do_POST()
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if url.path.startswith("/rapi/"):
                if retrain_web.handle_post(self, url.path, body, review_web.DB_PATH, JOB.busy) is None:
                    return self._send(404, {"error": "not found"})
                return
            if url.path == "/dapi/run":
                if JOB.busy():
                    return self._send(409, {"error": "a run is already in progress"})
                if retrain_web.JOB.busy():
                    return self._send(409, {"error": "a retraining cycle is running on the GPU; wait for "
                                                    "it to finish or cancel it on the Retrain page"})
                params, errors, warnings = validate(body)
                cfg = {k: body.get(k) for k in ("checkpoint", "cameras", "rat_choice", "stride",
                                                 "subdirs", "out_path", "session_label", "log_predictions",
                                                 "transform_meta_root")}
                save_config(cfg)
                if errors:
                    return self._send(400, {"errors": errors})
                if warnings and not body.get("force"):
                    return self._send(409, {"warnings": warnings})
                JOB.start(params)
                return self._send(200, {"started": True, "session_label": params["session_label"]})
            if url.path == "/dapi/cancel":
                with JOB.lock:
                    JOB.cancel_requested = True
                return self._send(200, {"ok": True})
            if url.path == "/dapi/save-config":
                save_config(body)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rat Behaviour Deploy</title>
<style>
__CSS__
</style>
</head>
<body>
<header>
  <h1>Rat Behaviour</h1>
  <nav><a href="/" class="on">Deploy</a><a href="/review">Review</a><a href="/retrain">Retrain</a></nav>
</header>
<main>
  <section>
    <h2>Model</h2>
    <div class="row"><label class="k">Checkpoint</label><input type="text" id="checkpoint">
      <button data-browse="checkpoint" data-exts=".pt,.pth">Browse…</button></div>
  </section>

  <section>
    <h2>Camera folders (raw, native resolution)</h2>
    <p class="warnbox">Order matters: "Camera N" must be that physical camera's footage. Per-camera normalization is keyed by camera identity, not by input order.</p>
    <div class="row"><button id="autofill">Fill from session folder…</button>
      <span class="hint">pick a folder containing cam_1 … cam_6, fills all six and the session label</span></div>
    <div id="cams"></div>
  </section>

  <section>
    <h2>Options</h2>
    <div class="row"><label class="k">Analyze</label>
      <label><input type="radio" name="rat" value="1"> Rat 1 only</label>
      <label><input type="radio" name="rat" value="2"> Rat 2 only</label>
      <label><input type="radio" name="rat" value="both"> Both</label></div>
    <div class="row"><label class="k">Stride</label><input type="number" id="stride" min="1">
      <span class="hint">frames. 32 = non-overlapping, matches training; smaller = smoother boundaries, more compute</span></div>
    <div class="row"><label class="k">Subfolders</label>
      <span class="hint">cutout</span><input type="text" id="sub_rgb" style="flex:0 1 180px;min-width:120px">
      <span class="hint">mask 1</span><input type="text" id="sub_mask1" style="flex:0 1 160px;min-width:120px">
      <span class="hint">mask 2</span><input type="text" id="sub_mask2" style="flex:0 1 160px;min-width:120px"></div>
    <div class="row"><label class="k">Metadata</label><input type="text" id="transform_meta_root" placeholder="empty = folder recorded in the checkpoint">
      <button data-browse="transform_meta_root">Browse…</button></div>
    <div class="row"><label class="k"></label><span class="hint">training sessions' transform metadata, used to normalize the scale feature. Set it if the log warns that scale stats could not be computed</span></div>
  </section>

  <section>
    <h2>Output</h2>
    <div class="row"><label class="k">JSON</label><input type="text" id="out_path">
      <button data-browse="out_path" data-exts=".json" data-save="1">Browse…</button></div>
    <div class="row"><label class="k">Session</label><input type="text" id="session_label" placeholder="defaults to the output file name">
      <label><input type="checkbox" id="log_predictions"> Log predictions for later review/correction</label></div>
  </section>

  <section>
    <h2>Run</h2>
    <div class="row">
      <button class="primary" id="run">Run</button>
      <button class="danger" id="cancel" disabled>Cancel</button>
      <div class="progress"><i id="bar"></i></div>
      <span id="ptext" class="hint"></span>
    </div>
    <div id="summary"></div>
    <pre id="log"></pre>
  </section>
</main>

__MODAL_HTML__

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, body) {
  const r = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  const j = await r.json().catch(() => ({}));
  return {ok: r.ok, status: r.status, data: j};
}

// ---- form ----
$('#cams').innerHTML = [1,2,3,4,5,6].map(i => `
  <div class="row"><label class="k">Camera ${i}</label><input type="text" id="cam${i}">
    <button data-browse="cam${i}">Browse…</button><div class="st muted" id="st${i}"></div></div>`).join('');

function form() {
  return {
    checkpoint: $('#checkpoint').value.trim(),
    cameras: [1,2,3,4,5,6].map(i => $('#cam' + i).value.trim()),
    rat_choice: (document.querySelector('input[name=rat]:checked') || {}).value || 'both',
    stride: +$('#stride').value || 32,
    subdirs: {rgb: $('#sub_rgb').value.trim(), mask1: $('#sub_mask1').value.trim(), mask2: $('#sub_mask2').value.trim()},
    out_path: $('#out_path').value.trim(),
    session_label: $('#session_label').value.trim(),
    log_predictions: $('#log_predictions').checked,
    transform_meta_root: $('#transform_meta_root').value.trim(),
  };
}
function fill(c) {
  $('#checkpoint').value = c.checkpoint || '';
  (c.cameras || []).forEach((p, i) => { if (i < 6) $('#cam' + (i + 1)).value = p || ''; });
  const r = document.querySelector(`input[name=rat][value="${c.rat_choice || 'both'}"]`); if (r) r.checked = true;
  $('#stride').value = c.stride || 32;
  $('#sub_rgb').value = c.subdirs.rgb; $('#sub_mask1').value = c.subdirs.mask1; $('#sub_mask2').value = c.subdirs.mask2;
  $('#out_path').value = c.out_path || '';
  $('#session_label').value = c.session_label || '';
  $('#log_predictions').checked = c.log_predictions !== false;
  $('#transform_meta_root').value = c.transform_meta_root || '';
}

let saveT;
function saveSoon() { clearTimeout(saveT); saveT = setTimeout(() => api('/dapi/save-config', form()), 600); }

async function checkCam(i) {
  const f = form(), path = f.cameras[i - 1], el = $('#st' + i);
  if (!path) { el.className = 'st muted'; el.textContent = ''; return; }
  el.className = 'st muted'; el.textContent = 'checking…';
  const q = new URLSearchParams({path, ...f.subdirs});
  const {data} = await api('/dapi/check?' + q);
  if (data.error && !data.parts) { el.className = 'st bad'; el.textContent = '✗ ' + data.error; return; }
  if (data.error) { el.className = 'st bad'; el.textContent = '✗ ' + data.error; return; }
  const parts = Object.values(data.parts).map(p => p.missing ? `${p.name}: missing` :
    `${p.name}: ${p.n} frames${p.n ? ` (${p.lo}–${p.hi})` : ''}`);
  el.className = 'st ' + (data.ok ? 'ok' : 'bad');
  el.textContent = (data.ok ? '✓ ' : '✗ ') + parts.join(' · ') + (data.has_transform_meta ? ' · transform_meta.csv' : '');
}
function checkAll() { for (let i = 1; i <= 6; i++) checkCam(i); }

document.querySelectorAll('input').forEach(el => el.addEventListener('change', saveSoon));
for (let i = 1; i <= 6; i++) $('#cam' + i).addEventListener('change', () => checkCam(i));
['#sub_rgb', '#sub_mask1', '#sub_mask2'].forEach(s => $(s).addEventListener('change', checkAll));

__MODAL_JS__

document.querySelectorAll('[data-browse]').forEach(b => b.onclick = () => {
  const target = $('#' + b.dataset.browse);
  openBrowser({start: target.value, exts: b.dataset.exts, save: !!b.dataset.save, onpick: v => {
    target.value = v; saveSoon();
    const m = b.dataset.browse.match(/^cam(\d)$/); if (m) checkCam(+m[1]);
  }});
});
$('#autofill').onclick = () => openBrowser({start: $('#cam1').value.replace(/\/[^/]*$/, ''), onpick: async v => {
  const {ok, data} = await api('/dapi/autocams?' + new URLSearchParams({path: v}));
  if (!ok) return alert(data.error || 'failed');
  const missing = data.cameras.map((c, i) => c ? null : i + 1).filter(Boolean);
  data.cameras.forEach((c, i) => { if (c) $('#cam' + (i + 1)).value = c; });
  if (!$('#session_label').value || confirm(`Set session label to "${data.session_label}"?`)) $('#session_label').value = data.session_label;
  saveSoon(); checkAll();
  if (missing.length) alert(`No folder found for camera ${missing.join(', ')} in ${v}`);
}});

// ---- run ----
let logLen = 0, polling = null;
async function run(force) {
  const body = {...form(), force: !!force};
  const {ok, status, data} = await api('/dapi/run', body);
  if (ok) { logLen = 0; $('#log').textContent = ''; $('#summary').innerHTML = ''; poll(); return; }
  if (data.errors) return alert(data.errors.join('\n'));
  if (data.warnings) { if (confirm(data.warnings.join('\n\n') + '\n\nRun anyway?')) run(true); return; }
  alert(data.error || `failed (${status})`);
}
$('#run').onclick = () => run(false);
$('#cancel').onclick = async () => { if (confirm('Cancel the running inference? Nothing will be written or logged.')) await api('/dapi/cancel', {}); };

function fmtDur(s) { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; }

async function poll() {
  clearTimeout(polling);
  const {data: j} = await api('/dapi/job?since=' + logLen);
  if (j.log && j.log.length) { const pre = $('#log'); pre.textContent += j.log.join('\n') + '\n'; pre.scrollTop = pre.scrollHeight; }
  logLen = j.log_len;
  const busy = j.status === 'loading' || j.status === 'running';
  $('#run').disabled = busy; $('#cancel').disabled = !busy;
  const pct = j.total ? 100 * j.done / j.total : 0;
  $('#bar').style.width = (j.status === 'done' ? 100 : pct) + '%';
  let t = '';
  if (j.status === 'loading') t = 'loading model…';
  else if (j.status === 'running' && j.total) {
    const el = j.now - j.started, eta = j.done ? el / j.done * (j.total - j.done) : 0;
    t = `${j.done} / ${j.total} · ${fmtDur(el)} elapsed` + (j.done ? ` · ~${fmtDur(eta)} left` : '');
  } else if (j.status === 'done') t = `done in ${fmtDur(j.finished - j.started)}`;
  else if (j.status === 'error') t = 'error';
  else if (j.status === 'cancelled') t = 'cancelled';
  $('#ptext').textContent = t;
  if (j.status === 'done' && j.summary) renderSummary(j.summary);
  if (j.status === 'error') $('#summary').innerHTML = `<p style="color:var(--bad)">${esc(j.error)}</p>`;
  if (busy) polling = setTimeout(poll, 1000);
}

function renderSummary(s) {
  const rats = Object.keys(s.rats);
  const labels = [...new Set(rats.flatMap(r => Object.keys(s.rats[r])))];
  const logged = Object.values(s.logged).reduce((a, b) => a + b, 0);
  $('#summary').innerHTML = `
    ${s.scale_stats_ok === false ? `<p class="warnbox">The scale feature ran with neutral values because its normalization stats could not be computed (see the log). Set the Metadata folder in Options and run again before reviewing these predictions.</p>` : ''}
    <p>Wrote <code>${esc(s.out_path)}</code>${s.frame_range ? ` · frames ${s.frame_range[0]}–${s.frame_range[1]}` : ''}</p>
    <table><tr><th>frames per label</th>${rats.map(r => `<th>${esc(r)}</th>`).join('')}</tr>
    ${labels.map(l => `<tr><td>${esc(l)}</td>${rats.map(r => `<td class="num">${s.rats[r][l] || 0}</td>`).join('')}</tr>`).join('')}</table>
    ${logged ? `<p>Logged ${logged} window predictions as session <b>${esc(s.session_label)}</b>.
      <a href="/review?session=${encodeURIComponent(s.session_label)}">Review them →</a></p>` : ''}`;
}

api('/dapi/config').then(({data}) => { fill(data); checkAll(); });
poll();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--port", type=int, default=8980)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) = only reachable through an ssh -L tunnel")
    args = ap.parse_args()

    review_web.DB_PATH = args.db
    # the review page reads each row's own subfolder names; rows logged before those were
    # recorded fall back to the names last used for deployment
    sub = load_config()["subdirs"]
    review_web.SUBDIRS.update({"rgb": sub["rgb"], 1: sub["mask1"], 2: sub["mask2"]})
    init_db(args.db)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"correction log: {args.db}")
    print(f"serving on http://{args.host}:{args.port}  (deploy)  and  /review")
    print(f"on your Mac:  ssh -L {args.port}:localhost:{args.port} <you>@<server>   "
          f"then open http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
