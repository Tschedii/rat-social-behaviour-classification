"""Browser version of review_gui.py: the same least-confident-first review queue over
6_deployment/correction_log.py, served as a local web page so it works over plain SSH with
no X11 display on the server.

On the server:

    python3 review_web.py [--port 8980] [--db PATH]

On your Mac, tunnel the port and open the page:

    ssh -L 8980:localhost:8980 <user>@<gpu-server>
    open http://localhost:8980

Compared with review_gui.py it shows all 6 cameras (not just camera 1) and lets you scrub or
play through all 32 frames of the window, with the target rat's mask outlined on the cutout.
Verdicts go through the same correction_log functions, so export_corrections_for_retrain.py
and retrain_cycle.py pick them up unchanged.

Only standard library + cv2/numpy (already needed by raw_frame_source.py). torch is used only
to read each checkpoint's label list; without it, labels fall back to those already in the log.
"""
import argparse
import json
import sqlite3
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from correction_log import (DEFAULT_DB_PATH, confirm_correct, init_db, reset_to_pending,  # noqa: E402
                            row_subdirs, set_correction, stats)
from raw_frame_source import FrameDir  # noqa: E402

DB_PATH = str(DEFAULT_DB_PATH)
# only for rows logged before the correction log recorded each row's subfolder names
SUBDIRS = {"rgb": "cutout_all_objects", 1: "mask_object_1", 2: "mask_object_2"}
_label_lock = threading.Lock()
_label_cache = {}


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def query(sql, params=()):
    conn = init_db(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def get_row(row_id):
    rows = query("SELECT * FROM predictions WHERE id = ?", (row_id,))
    return rows[0] if rows else None


def review_queue(limit, session=None, rat=None):
    sql = "SELECT * FROM predictions WHERE status = 'pending'"
    params = []
    if session:
        sql += " AND session_label = ?"
        params.append(session)
    if rat:
        sql += " AND rat = ?"
        params.append(int(rat))
    sql += " ORDER BY confidence ASC LIMIT ?"
    params.append(int(limit))
    return query(sql, params)


def sessions():
    return query(
        "SELECT session_label, COUNT(*) AS total, "
        "SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending "
        "FROM predictions GROUP BY session_label ORDER BY session_label"
    )


def labels_for_checkpoint(checkpoint_path):
    with _label_lock:
        if checkpoint_path in _label_cache:
            return _label_cache[checkpoint_path]
        labels = None
        try:
            import torch
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:  # older torch without weights_only
                ckpt = torch.load(checkpoint_path, map_location="cpu")
            label_map = ckpt["label_map"]
            labels = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        except Exception as e:
            print(f"could not read label_map from {checkpoint_path}: {e} "
                  f"— falling back to labels already present in the log")
        if not labels:
            seen = query("SELECT DISTINCT predicted_label AS l FROM predictions "
                         "UNION SELECT DISTINCT corrected_label FROM predictions "
                         "WHERE corrected_label IS NOT NULL")
            labels = sorted(r["l"] for r in seen if r["l"])
        _label_cache[checkpoint_path] = labels
        return labels


@lru_cache(maxsize=64)
def frame_dir(path):
    return FrameDir(path)


def read_nearest(fd, fn, color):
    if fd.has(fn):
        return fd.read(fn, color=color)
    lo, hi = fd.available_range()
    keys = fd._index.keys()
    nearest = min(keys, key=lambda k: abs(k - fn)) if lo <= fn <= hi else min(max(fn, lo), hi)
    return fd.read(nearest, color=color)


def render_frame(row, cam, offset, view, width):
    cam_dir = Path(json.loads(row["camera_dirs"])[cam - 1])
    fn = row["window_start"] + max(0, min(offset, row["seq_len"] - 1))

    sub = row_subdirs(row, fallback=SUBDIRS)
    rgb = None
    mask = None
    if view in ("overlay", "rgb"):
        rgb = read_nearest(frame_dir(str(cam_dir / sub["rgb"])), fn, color=True)
    if view in ("overlay", "mask"):
        mask = read_nearest(frame_dir(str(cam_dir / sub[f"mask{row['rat']}"])), fn, color=False)

    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if view == "mask" else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if width and img.shape[1] > width:
        h = int(round(img.shape[0] * width / img.shape[1]))
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)

    if view == "overlay":
        # target rat: green tint + outline, drawn at display size so the line stays visible
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        on = mask > 0
        img[on] = (0.6 * img[on] + 0.4 * np.array((60, 220, 60))).astype(np.uint8)
        contours, _ = cv2.findContours(on.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (60, 255, 60), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise IOError("jpeg encode failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet, frames are requested a lot
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if ctype == "image/jpeg" else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if url.path == "/api/queue":
                return self._send(200, review_queue(q.get("limit", 50), q.get("session") or None,
                                                    q.get("rat") or None))
            if url.path == "/api/sessions":
                return self._send(200, sessions())
            if url.path == "/api/stats":
                return self._send(200, stats(DB_PATH))
            if url.path == "/api/labels":
                return self._send(200, labels_for_checkpoint(q["checkpoint"]))
            if url.path == "/api/frame":
                row = get_row(int(q["id"]))
                if row is None:
                    return self._send(404, {"error": "no such row"})
                jpg = render_frame(row, int(q.get("cam", 1)), int(q.get("offset", 0)),
                                   q.get("view", "overlay"), int(q.get("w", 480)))
                return self._send(200, jpg, "image/jpeg")
            return self._send(404, {"error": "not found"})
        except (FileNotFoundError, IndexError) as e:
            return self._send(404, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            if url.path != "/api/verdict":
                return self._send(404, {"error": "not found"})
            row_id = int(data["id"])
            note = (data.get("note") or "").strip() or None
            action = data["action"]
            if action == "confirm":
                confirm_correct(DB_PATH, row_id, note)
            elif action == "correct":
                if not data.get("label"):
                    return self._send(400, {"error": "missing label"})
                set_correction(DB_PATH, row_id, data["label"], note)
            elif action == "reset":
                if not reset_to_pending(DB_PATH, row_id):
                    return self._send(409, {"error": "row already exported, can't undo"})
            else:
                return self._send(400, {"error": f"unknown action {action}"})
            return self._send(200, get_row(row_id))
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Corrections</title>
<style>
:root {
  --bg: #f6f6f4; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #deddd8;
  --accent: #1f6feb; --good: #1a7f37; --warn: #b35900; --bad: #c62828; --chip: #efeee9;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #151515; --panel: #1f1f1f; --ink: #ecebe6; --muted: #9a9a94; --line: #333331;
          --accent: #58a6ff; --good: #3fb950; --warn: #e3a341; --bad: #f47067; --chip: #2a2a28; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 10px 16px;
         border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 2; }
header h1 { font-size: 15px; margin: 0 8px 0 0; }
header .stats { color: var(--muted); font-size: 13px; margin-left: auto; }
select, input, button { font: inherit; color: inherit; }
select, input[type=text] { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
button { background: var(--chip); border: 1px solid var(--line); border-radius: 6px; padding: 6px 12px; cursor: pointer; }
button:hover { border-color: var(--muted); }
button.primary { background: var(--good); border-color: var(--good); color: #fff; }
button.sel { outline: 2px solid var(--accent); outline-offset: 1px; }
main { padding: 16px; max-width: 1500px; margin: 0 auto; }
.info { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: baseline; margin-bottom: 12px; }
.info .big { font-size: 20px; font-weight: 600; }
.info .k { color: var(--muted); }
.conf { display: inline-block; width: 120px; height: 8px; background: var(--chip); border-radius: 4px;
        vertical-align: middle; overflow: hidden; margin-left: 6px; }
.conf > i { display: block; height: 100%; background: var(--warn); }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; max-width: 88vh; margin: 0 auto; }
.grid.single { max-width: 64vh; }
.grid.single { grid-template-columns: 1fr; }
.cam { position: relative; background: #000; border-radius: 6px; overflow: hidden; aspect-ratio: 1 / 1; cursor: zoom-in; }
.grid.single .cam { cursor: zoom-out; }
.cam img { width: 100%; height: 100%; object-fit: contain; display: block; }
.cam .tag { position: absolute; left: 6px; top: 6px; background: rgba(0,0,0,.6); color: #fff;
            font-size: 12px; padding: 1px 6px; border-radius: 4px; }
.cam .err { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            color: #aaa; font-size: 12px; padding: 12px; text-align: center; }
.player { display: flex; gap: 10px; align-items: center; margin: 10px 0 16px; flex-wrap: wrap; }
.player input[type=range] { flex: 1; min-width: 200px; }
.verdict { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.labels { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
.labels button .n { color: var(--muted); font-size: 11px; margin-right: 4px; }
.labels button.pred { border-color: var(--warn); }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.hint { color: var(--muted); font-size: 12px; margin-top: 10px; }
kbd { background: var(--chip); border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; font-size: 11px; }
.toast { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--bg);
         padding: 8px 14px; border-radius: 6px; opacity: 0; transition: opacity .2s; pointer-events: none; }
.toast.show { opacity: .92; }
.empty { padding: 60px 0; text-align: center; color: var(--muted); }
@media (max-width: 800px) { .grid { grid-template-columns: repeat(2, 1fr); max-width: none; } }
</style>
</head>
<body>
<header>
  <span id="navlinks" style="display:none"><a href="/" style="color:var(--muted);text-decoration:none">Deploy</a>
    · <a href="/retrain" style="color:var(--muted);text-decoration:none">Retrain</a></span>
  <h1>Review corrections</h1>
  <label>Session <select id="session"><option value="">all</option></select></label>
  <label>Rat <select id="rat"><option value="">both</option><option>1</option><option>2</option></select></label>
  <label>View <select id="view"><option value="overlay">cutout + mask</option><option value="rgb">cutout</option><option value="mask">mask</option></select></label>
  <button id="reload">Refresh queue</button>
  <span class="stats" id="stats"></span>
</header>
<main id="main"></main>
<div class="toast" id="toast"></div>

<script>
const $ = s => document.querySelector(s);
const state = { queue: [], pos: 0, offset: 16, playing: null, labels: [], chosen: null, history: [], single: null };

function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 1600); }

async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: {'Content-Type': 'application/json'},
                                       body: JSON.stringify(body) } : {});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function refreshStats() {
  const s = await api('/api/stats');
  const f = k => (s[k] ? s[k].count : 0);
  $('#stats').textContent = `pending ${f('pending')} · confirmed ${f('confirmed')} · corrected ${f('corrected')}`;
}

async function refreshSessions() {
  const sel = $('#session'), cur = sel.value;
  const list = await api('/api/sessions');
  sel.innerHTML = '<option value="">all</option>' + list.map(s =>
    `<option value="${esc(s.session_label)}">${esc(s.session_label)} (${s.pending} pending)</option>`).join('');
  sel.value = cur;
}

async function loadQueue() {
  stop();
  const p = new URLSearchParams({ limit: 200, session: $('#session').value, rat: $('#rat').value });
  state.queue = await api('/api/queue?' + p);
  state.pos = 0;
  await show();
  refreshStats();
}

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function cur() { return state.queue[state.pos]; }

async function show() {
  const row = cur();
  if (!row) {
    $('#main').innerHTML = `<div class="empty">Queue empty, nothing pending for this filter.<br>
      Run a deployment session with logging on, or change the filter.</div>`;
    return;
  }
  state.offset = Math.floor(row.seq_len / 2);
  try { state.labels = await api('/api/labels?checkpoint=' + encodeURIComponent(row.checkpoint)); }
  catch (e) { state.labels = []; toast('could not load labels: ' + e.message); }
  if (!state.labels.includes(row.predicted_label)) state.labels.push(row.predicted_label);
  state.chosen = row.predicted_label;

  const cams = [1,2,3,4,5,6].filter(c => state.single == null || c === state.single);
  $('#main').innerHTML = `
    <div class="info">
      <span class="big">${esc(row.predicted_label)}</span>
      <span><span class="k">confidence</span> ${row.confidence.toFixed(3)}<span class="conf"><i style="width:${(row.confidence*100).toFixed(1)}%"></i></span></span>
      <span><span class="k">session</span> ${esc(row.session_label)}</span>
      <span><span class="k">rat</span> ${row.rat}</span>
      <span><span class="k">frames</span> ${row.window_start}–${row.window_start + row.seq_len - 1}</span>
      <span class="k">${state.pos + 1} / ${state.queue.length} (least confident first)</span>
    </div>
    <div class="grid ${state.single != null ? 'single' : ''}" id="grid">
      ${cams.map(c => `<div class="cam" data-cam="${c}"><img alt="camera ${c}"><span class="tag">cam ${c}</span></div>`).join('')}
    </div>
    <div class="player">
      <button id="play">▶ Play</button>
      <input type="range" id="scrub" min="0" max="${row.seq_len - 1}" value="${state.offset}">
      <span id="fno" class="k"></span>
    </div>
    <div class="verdict">
      <div class="row">
        <button class="primary" id="confirm">✓ Prediction is correct</button>
        <button id="skip">Skip →</button>
        <button id="undo" ${state.history.length ? '' : 'disabled'}>↶ Undo last</button>
        <input type="text" id="note" placeholder="note (optional)" style="flex:1;min-width:160px">
      </div>
      <div class="labels" id="labels">
        ${state.labels.map((l, i) => `<button data-label="${esc(l)}" class="${l === row.predicted_label ? 'pred' : ''}">
          <span class="n">${i < 9 ? i + 1 : ''}</span>${esc(l)}</button>`).join('')}
      </div>
      <div class="row"><button id="save">Save correction</button><span class="k" id="chosen"></span></div>
      <div class="hint"><kbd>Enter</kbd> confirm · <kbd>1</kbd>–<kbd>9</kbd> pick label, then <kbd>S</kbd> save ·
        <kbd>→</kbd> skip · <kbd>U</kbd> undo · <kbd>Space</kbd> play/pause · <kbd>,</kbd> <kbd>.</kbd> step frame · click a camera to enlarge</div>
    </div>`;

  document.querySelectorAll('.cam').forEach(el => el.onclick = () => {
    state.single = state.single == null ? +el.dataset.cam : null; stop(); show(); });
  $('#scrub').oninput = e => { stop(); state.offset = +e.target.value; drawFrames(); };
  $('#play').onclick = togglePlay;
  $('#confirm').onclick = confirm;
  $('#skip').onclick = skip;
  $('#undo').onclick = undo;
  $('#save').onclick = save;
  document.querySelectorAll('#labels button').forEach(b => b.onclick = () => choose(b.dataset.label));
  choose(state.chosen);
  drawFrames();
}

function choose(label) {
  state.chosen = label;
  document.querySelectorAll('#labels button').forEach(b => b.classList.toggle('sel', b.dataset.label === label));
  const row = cur();
  $('#chosen').textContent = label === row.predicted_label ? '(same as prediction, use Confirm instead)' : `→ will save as "${label}"`;
}

function frameUrl(row, cam, offset) {
  const w = state.single != null ? 900 : 480;
  return `/api/frame?id=${row.id}&cam=${cam}&offset=${offset}&view=${$('#view').value}&w=${w}`;
}

function drawFrames() {
  const row = cur(); if (!row) return;
  $('#scrub').value = state.offset;
  $('#fno').textContent = `frame ${row.window_start + state.offset} (${state.offset + 1}/${row.seq_len})`;
  const loads = [];
  document.querySelectorAll('.cam').forEach(el => {
    const img = el.querySelector('img'), cam = +el.dataset.cam;
    loads.push(new Promise(res => {
      img.onload = () => { el.querySelector('.err')?.remove(); img.style.visibility = ''; res(); };
      img.onerror = () => { img.style.visibility = 'hidden';
        if (!el.querySelector('.err')) el.insertAdjacentHTML('beforeend', '<div class="err">no frame for this camera</div>'); res(); };
      img.src = frameUrl(row, cam, state.offset);
    }));
  });
  return Promise.all(loads);
}

function stop() { if (state.playing) { state.playing = false; const b = $('#play'); if (b) b.textContent = '▶ Play'; } }
async function togglePlay() {
  if (state.playing) return stop();
  state.playing = true; $('#play').textContent = '❚❚ Pause';
  const row = cur();
  while (state.playing && cur() === row) {
    state.offset = (state.offset + 1) % row.seq_len;
    const t = performance.now();
    await drawFrames();
    await new Promise(r => setTimeout(r, Math.max(0, 100 - (performance.now() - t))));
  }
}

async function verdict(action, label) {
  const row = cur(); if (!row) return;
  try {
    await api('/api/verdict', { id: row.id, action, label, note: $('#note').value });
    state.history.push(state.pos);
    toast(action === 'confirm' ? 'confirmed' : `corrected to "${label}"`);
    next();
  } catch (e) { toast('error: ' + e.message); }
}
function confirm() { verdict('confirm'); }
function save() {
  const row = cur(); if (!row) return;
  if (state.chosen === row.predicted_label) return verdict('confirm');
  verdict('correct', state.chosen);
}
function skip() { next(); }
function next() { stop(); state.pos++; show(); refreshStats(); }
async function undo() {
  if (!state.history.length) return;
  const pos = state.history.pop();
  try {
    await api('/api/verdict', { id: state.queue[pos].id, action: 'reset' });
    stop(); state.pos = pos; show(); refreshStats(); toast('undone, back to pending');
  } catch (e) { toast('error: ' + e.message); }
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || !cur()) return;
  if (e.key === 'Enter') { e.preventDefault(); confirm(); }
  else if (e.key === 'ArrowRight') skip();
  else if (e.key === 'u' || e.key === 'U') undo();
  else if (e.key === 's' || e.key === 'S') save();
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === ',' || e.key === '.') { stop(); const n = cur().seq_len;
    state.offset = (state.offset + (e.key === '.' ? 1 : n - 1)) % n; drawFrames(); }
  else if (/^[1-9]$/.test(e.key)) { const l = state.labels[+e.key - 1]; if (l) choose(l); }
});
$('#reload').onclick = () => { refreshSessions(); loadQueue(); };
$('#session').onchange = loadQueue;
$('#rat').onchange = loadQueue;
$('#view').onchange = () => drawFrames();

fetch('/dapi/config').then(r => { if (r.ok) $('#navlinks').style.display = ''; }).catch(() => {});
const initialSession = new URLSearchParams(location.search).get('session');
refreshSessions().then(() => { if (initialSession) $('#session').value = initialSession; return loadQueue(); });
</script>
</body>
</html>
"""


def main():
    global DB_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--port", type=int, default=8980)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) = only reachable through an ssh -L tunnel")
    # rows record their own subfolder names; these only apply to rows logged before that
    ap.add_argument("--rgb-subdir", default="cutout_all_objects")
    ap.add_argument("--mask1-subdir", default="mask_object_1")
    ap.add_argument("--mask2-subdir", default="mask_object_2")
    args = ap.parse_args()

    DB_PATH = args.db
    SUBDIRS.update({"rgb": args.rgb_subdir, 1: args.mask1_subdir, 2: args.mask2_subdir})
    init_db(DB_PATH)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"correction log: {DB_PATH}")
    print(f"serving on http://{args.host}:{args.port}")
    print(f"on your Mac:  ssh -L {args.port}:localhost:{args.port} <you>@<server>   "
          f"then open http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
