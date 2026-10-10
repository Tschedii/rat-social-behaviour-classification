"""Browser front end for retrain_cycle.py, served by deploy_web.py at /retrain.

It runs retrain_cycle.py unchanged as a subprocess (so the CLI and the page can never
disagree about what a cycle does) and streams its output to the page. On top it adds:
  - a preview of what the next export contains (sessions, rats, windows, labels)
  - a pre-flight check that no exported session is in the base split's EVAL set, which would
    leak reviewed footage into training and inflate the eval score
  - buttons for the decision a finished cycle waits for, calling retrain_cycle.promote() /
    reject() (the same code as `retrain_cycle.py --promote/--reject`), so the page and the
    CLI share one retraining state (~/.rat_retrain_state.json): a cycle started from either
    can be decided from either, also after a server restart
  - once a cycle has been promoted, the inputs come from that promoted base

Settings persist to ~/.rat_retrain_web_config.json.
"""
import argparse
import codecs
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY_DIR))
import retrain_cycle  # noqa: E402
from correction_log import exclude_sessions, stats  # noqa: E402
from export_corrections_for_retrain import build_export  # noqa: E402

RETRAIN_SCRIPT = DEPLOY_DIR / "retrain_cycle.py"
CONFIG_PATH = Path.home() / ".rat_retrain_web_config.json"
MAX_LOG_LINES = 20000

INPUT_FIELDS = retrain_cycle.BASE_KEYS

# (key, title, marker in retrain_cycle.py's output that means this stage has started)
STAGES = [
    ("export", "Export corrections", None),
    ("h5", "Build .h5", "2_build_h5_dataset.py"),
    ("index", "Window index", "build_index.py"),
    ("train", "Train", "train.py"),
    ("eval", "Evaluate", "evaluate.py"),
    ("compare", "Compare", "population_weighted_accuracy"),
]


# ---------------------------------------------------------------------------
# config + defaults
# ---------------------------------------------------------------------------

def load_config(deploy_checkpoint=""):
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = {k: v for k, v in json.loads(CONFIG_PATH.read_text()).items() if v not in (None, "")}
        except Exception:
            pass
    base = retrain_cycle.load_state()["base"]
    if base:  # a promoted cycle is the truth for the next one's inputs
        cfg.update({k: base[k] for k in INPUT_FIELDS})
        if base.get(retrain_cycle.META_KEY):
            cfg[retrain_cycle.META_KEY] = base[retrain_cycle.META_KEY]
    cfg.setdefault(retrain_cycle.META_KEY, "")
    cfg["promoted_from"] = base["promoted_from"] if base else None
    cfg.setdefault("checkpoint", deploy_checkpoint or "")
    cfg.setdefault("data_root", str(DEPLOY_DIR.parent.parent))
    cfg.setdefault("epochs", 5)
    for k in ("base_manifest", "base_split_csv", "base_label_dir"):
        cfg.setdefault(k, "")
    if not cfg.get("eval_report_current"):
        cfg["eval_report_current"] = guess_eval_report(cfg["checkpoint"]) or ""
    out = cfg.get("retrain_out_dir")
    if not out or (Path(out) / retrain_cycle.CYCLE_FILE).exists():  # used by an earlier cycle
        cfg["retrain_out_dir"] = suggest_out_dir(cfg["checkpoint"])
    return cfg


def pending():
    """The cycle awaiting a decision (from either the page or the CLI), or None."""
    state = retrain_cycle.load_state()
    cycle = retrain_cycle.pending_cycle(state)
    if not cycle:
        return None
    return {"retrain_out_dir": state["pending"], "new_checkpoint": cycle["new_checkpoint"],
            "comparison": cycle.get("comparison", ""), "n_windows": cycle.get("n_windows"),
            "sessions": cycle.get("sessions", []), "created_at": cycle.get("created_at")}


def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def guess_eval_report(checkpoint):
    """classification_report_eval.txt near the checkpoint, preferring an eval_* folder named
    after it (evaluate.py's usual layout: training_output/eval_<ckpt stem>/...)."""
    if not checkpoint:
        return None
    ck = Path(checkpoint).expanduser()
    candidates = []
    for base in (ck.parent, ck.parent.parent):
        if base.is_dir():
            candidates += list(base.glob("*/classification_report_eval.txt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (ck.stem not in p.parent.name, -p.stat().st_mtime))
    return str(candidates[0])


def suggest_out_dir(checkpoint):
    """retrain_YYYYMMDD next to the last promoted cycle, else next to the checkpoint."""
    promoted = (retrain_cycle.load_state()["base"] or {}).get("promoted_from")
    if promoted:
        base = Path(promoted).parent
    else:
        base = Path(checkpoint).expanduser().parent if checkpoint else DEPLOY_DIR / "training_output"
    stem = base / f"retrain_{time.strftime('%Y%m%d')}"
    p, i = stem, 2
    while p.exists():
        p = Path(f"{stem}_{i}")
        i += 1
    return str(p)


# ---------------------------------------------------------------------------
# preview + validation
# ---------------------------------------------------------------------------

def read_split(path):
    with open(path, newline="") as f:
        return {r["session_id"]: r["split"].strip().lower() for r in csv.DictReader(f)}


def preview(db_path, base_split_csv=None):
    out = {"stats": stats(db_path), "groups": [], "n_windows": 0, "error": None,
           "eval_leaks": [], "already_train": [], "split_error": None}
    try:
        manifest_rows, label_csvs, session_ids, exported_ids = build_export(db_path)
    except ValueError as e:
        out["error"] = str(e)
        return out
    for (session, rat), segments in sorted(label_csvs.items()):
        counts = Counter(s["behavior"] for s in segments)
        out["groups"].append({"session": session, "rat": rat, "windows": len(segments),
                              "labels": dict(counts.most_common())})
        out["n_windows"] += len(segments)
    out["n_rows"] = len(exported_ids)
    if base_split_csv:
        try:
            split = read_split(Path(base_split_csv).expanduser())
            out["eval_leaks"] = [s for s in session_ids if split.get(s) == "eval"]
            out["already_train"] = [s for s in session_ids if split.get(s) == "train"]
        except Exception as e:
            out["split_error"] = f"couldn't read {base_split_csv}: {e}"
    return out


def validate(body, db_path):
    """Returns (params, errors, warnings). Warnings can be overridden with force=true."""
    errors, warnings = [], []
    dry_run = bool(body.get("dry_run"))
    cyc = pending()
    if cyc and not dry_run:
        errors.append(f"The cycle in {cyc['retrain_out_dir']} is still awaiting a decision. "
                      f"Deploy or reject it (Result section) first.")

    fields = {k: (str(body.get(k) or "")).strip() or None for k in INPUT_FIELDS + (retrain_cycle.META_KEY,)}
    ns = lambda ignore: argparse.Namespace(**fields, ignore_promoted_base=ignore)  # noqa: E731
    state = retrain_cycle.load_state()
    inputs, hard = retrain_cycle.resolve_inputs(ns(True), state)
    _, all_errors = retrain_cycle.resolve_inputs(ns(False), state)
    errors += hard
    # differing from the promoted base is allowed, but only after an explicit confirmation
    chain_warnings = [e for e in all_errors if e not in hard]
    ignore_promoted_base = bool(chain_warnings)
    warnings += [e.replace("--ignore-promoted-base", "Run anyway") for e in chain_warnings]

    out_dir = (str(body.get("retrain_out_dir") or "")).strip()
    if not out_dir:
        errors.append("The output folder is not set.")
    try:
        epochs = int(body.get("epochs", 5))
        if epochs <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Epochs must be a positive integer.")
        epochs = 5

    pv = preview(db_path, inputs.get("base_split_csv") if not hard else None)
    if pv["error"]:
        errors.append(pv["error"])
    elif not pv["groups"]:
        errors.append("Nothing to retrain on: no confirmed/corrected rows waiting for export. "
                      "Review some predictions first.")
    if pv["eval_leaks"]:
        errors.append("These sessions are in the base split's EVAL set, so training on their corrections "
                      "would leak eval footage into training: " + ", ".join(pv["eval_leaks"]) +
                      ". Use \"Exclude from retraining\" in the correction log section first.")
    if pv["split_error"]:
        errors.append(pv["split_error"])
    if not errors:
        if pv["already_train"]:
            warnings.append("These sessions are already TRAIN sessions in the base split, so their "
                            "corrected windows are added on top of the existing labels for the same "
                            "footage: " + ", ".join(pv["already_train"]))
        out = Path(out_dir).expanduser()
        if out.exists() and any(out.iterdir()) and not dry_run:
            warnings.append(f"{out} already exists and isn't empty; files in it will be overwritten.")

    # retrain_cycle.py runs train.py with a different cwd, so every path must be absolute
    params = {**inputs, "retrain_out_dir": str(Path(out_dir).expanduser().resolve()) if out_dir else "",
              "epochs": epochs, "dry_run": dry_run, "ignore_promoted_base": ignore_promoted_base}
    return params, pv, errors, warnings


def build_cmd(p, db_path):
    cmd = [sys.executable, "-u", str(RETRAIN_SCRIPT),
           "--checkpoint", p["checkpoint"], "--db", str(db_path),
           "--base-manifest", p["base_manifest"], "--base-split-csv", p["base_split_csv"],
           "--base-label-dir", p["base_label_dir"], "--data-root", p["data_root"],
           "--retrain-out-dir", p["retrain_out_dir"],
           "--eval-report-current", p["eval_report_current"], "--epochs", str(p["epochs"])]
    if p.get(retrain_cycle.META_KEY):
        cmd += ["--transform-meta-root", p[retrain_cycle.META_KEY]]
    if p["dry_run"]:
        cmd.append("--dry-run")
    if p["ignore_promoted_base"]:
        cmd.append("--ignore-promoted-base")
    return cmd


def last_segment(text):
    """What a terminal would show for a line redrawn with \r (e.g. a tqdm bar)."""
    return next((seg for seg in reversed(text.split("\r")) if seg), "")


# ---------------------------------------------------------------------------
# job: one retrain_cycle.py subprocess, output streamed into memory
# ---------------------------------------------------------------------------

class RetrainJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.reset()

    def reset(self):
        self.status = "idle"   # idle | running | done | error | cancelled
        self.lines = []
        self.dropped = 0       # lines trimmed off the front (MAX_LOG_LINES)
        self.partial = ""      # current unfinished line as displayed (tqdm bars rewrite it with \r)
        self._raw = ""
        self.stage = None
        self.started = self.finished = None
        self.params = None
        self.returncode = None
        self.cmd = None

    def busy(self):
        return self.status == "running"

    def snapshot(self, since=0):
        with self.lock:
            start = max(0, since - self.dropped)
            return {"status": self.status, "log": self.lines[start:],
                    "log_len": self.dropped + len(self.lines), "partial": self.partial,
                    "stage": self.stage, "stages": [[k, t] for k, t, _ in STAGES],
                    "started": self.started, "finished": self.finished, "now": time.time(),
                    "params": self.params, "returncode": self.returncode, "cmd": self.cmd}

    def start(self, params, db_path):
        with self.lock:
            if self.busy():
                raise RuntimeError("a retraining cycle is already running")
            self.reset()
            self.status = "running"
            self.stage = "export"
            self.started = time.time()
            self.params = params
            cmd = build_cmd(params, db_path)
            self.cmd = " ".join(cmd)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            self.proc = subprocess.Popen(cmd, cwd=str(DEPLOY_DIR), stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, env=env, start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _line(self, line):
        # called with self.lock held
        self.lines.append(line)
        if len(self.lines) > MAX_LOG_LINES:
            cut = len(self.lines) - MAX_LOG_LINES
            del self.lines[:cut]
            self.dropped += cut
        if line.startswith("$ "):
            for key, _, marker in STAGES:
                if marker and marker in line:
                    self.stage = key

    def _pump(self):
        fd = self.proc.stdout.fileno()
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            with self.lock:
                parts = (self._raw + dec.decode(chunk)).split("\n")
                for part in parts[:-1]:
                    self._line(last_segment(part))
                # progress bars redraw with \r: show the latest redraw, keep only what follows it
                self._raw = parts[-1].split("\r")[-1]
                self.partial = last_segment(parts[-1])
        rc = self.proc.wait()
        with self.lock:
            if self.partial:
                self._line(self.partial)
            self.partial = self._raw = ""
            self.returncode = rc
            self.finished = time.time()
            if self.status == "cancelled":
                pass
            elif rc == 0:
                self.status = "done"
                self.stage = "finished"
            else:
                self.status = "error"

    def cancel(self):
        with self.lock:
            if not self.busy() or self.proc is None:
                return False
            self.status = "cancelled"
            pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)

        def hard_kill():
            time.sleep(10)
            if self.proc.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
        threading.Thread(target=hard_kill, daemon=True).start()
        return True


JOB = RetrainJob()


# ---------------------------------------------------------------------------
# HTTP routes, called from deploy_web.Handler
# ---------------------------------------------------------------------------

def handle_get(handler, path, q, db_path, deploy_cfg):
    if path == "/retrain":
        import web_common
        return handler._send(200, web_common.render(PAGE), "text/html; charset=utf-8")
    if path == "/rapi/config":
        return handler._send(200, load_config(deploy_cfg.get("checkpoint", "")))
    if path == "/rapi/preview":
        return handler._send(200, preview(db_path, q.get("base_split_csv") or None))
    if path == "/rapi/guess":
        ck = q.get("checkpoint", "")
        return handler._send(200, {"eval_report_current": guess_eval_report(ck),
                                   "retrain_out_dir": suggest_out_dir(ck)})
    if path == "/rapi/job":
        return handler._send(200, {**JOB.snapshot(int(q.get("since", 0))), "pending": pending()})
    return None


def handle_post(handler, path, body, db_path, other_busy):
    if path == "/rapi/run":
        if JOB.busy():
            return handler._send(409, {"error": "a retraining cycle is already running"})
        if other_busy():
            return handler._send(409, {"error": "a deployment inference run is in progress; wait for it "
                                                "to finish (both need the GPU)"})
        save_config({k: body.get(k) for k in INPUT_FIELDS + ("retrain_out_dir", "epochs", retrain_cycle.META_KEY)})
        params, pv, errors, warnings = validate(body, db_path)
        if errors:
            return handler._send(400, {"errors": errors})
        if warnings and not body.get("force"):
            return handler._send(409, {"warnings": warnings})
        JOB.start(params, db_path)
        return handler._send(200, {"started": True})
    if path == "/rapi/exclude":
        if JOB.busy():
            return handler._send(409, {"error": "wait for the running cycle to finish"})
        n = exclude_sessions(db_path, [str(s) for s in body.get("sessions", [])])
        return handler._send(200, {"ok": True, "rows": n})
    if path == "/rapi/cancel":
        return handler._send(200, {"ok": JOB.cancel()})
    if path == "/rapi/save-config":
        save_config(body)
        return handler._send(200, {"ok": True})
    if path in ("/rapi/deploy", "/rapi/reject"):
        if JOB.busy():
            return handler._send(409, {"error": "wait for the running cycle to finish"})
        cyc = pending()
        if not cyc or cyc["retrain_out_dir"] != body.get("retrain_out_dir"):
            return handler._send(409, {"error": "that cycle is no longer awaiting a decision"})
        try:
            if path == "/rapi/deploy":
                retrain_cycle.promote(cyc["retrain_out_dir"])
            else:
                retrain_cycle.reject(cyc["retrain_out_dir"])
        except (RuntimeError, FileNotFoundError) as e:
            return handler._send(409, {"error": str(e)})
        return handler._send(200, {"ok": True, "decision": "deployed" if path == "/rapi/deploy" else "rejected",
                                   "next": load_config()})
    return None


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rat Behaviour Retrain</title>
<style>
__CSS__
.row > label.k { width: 150px; }
.stages { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0 4px; }
.stages span { font-size: 12px; padding: 3px 10px; border-radius: 12px; background: var(--chip); color: var(--muted); border: 1px solid var(--line); }
.stages span.done { color: var(--good); border-color: var(--good); }
.stages span.on { color: #fff; background: var(--accent); border-color: var(--accent); }
.stages span.fail { color: #fff; background: var(--bad); border-color: var(--bad); }
.err { color: var(--bad); } .okc { color: var(--good); }
pre.compare { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-size: 12px; overflow: auto; }
#partial { color: var(--muted); }
.labels-cell { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Rat Behaviour</h1>
  <nav><a href="/">Deploy</a><a href="/review">Review</a><a href="/retrain" class="on">Retrain</a></nav>
</header>
<main>
  <section>
    <h2>Correction log: what the next cycle trains on</h2>
    <div id="preview" class="hint">loading…</div>
  </section>

  <section>
    <h2>Inputs</h2>
    <div class="row"><label class="k">Current checkpoint</label><input type="text" id="checkpoint">
      <button data-browse="checkpoint" data-exts=".pt,.pth">Browse…</button></div>
    <div class="row"><label class="k">Its eval report</label><input type="text" id="eval_report_current">
      <button data-browse="eval_report_current" data-exts=".txt">Browse…</button></div>
    <div class="row"><label class="k">Base manifest</label><input type="text" id="base_manifest">
      <button data-browse="base_manifest" data-exts=".csv">Browse…</button></div>
    <div class="row"><label class="k">Base split CSV</label><input type="text" id="base_split_csv">
      <button data-browse="base_split_csv" data-exts=".csv">Browse…</button></div>
    <div class="row"><label class="k">Base label dir</label><input type="text" id="base_label_dir">
      <button data-browse="base_label_dir">Browse…</button></div>
    <div class="row"><label class="k">Data root</label><input type="text" id="data_root">
      <button data-browse="data_root">Browse…</button></div>
    <div class="row"><label class="k">Metadata folder</label><input type="text" id="transform_meta_root" placeholder="empty = folder recorded in the checkpoint">
      <button data-browse="transform_meta_root">Browse…</button></div>
    <div class="row"><label class="k"></label><span class="hint">training sessions' transform metadata, for checkpoints trained with the scale feature. Set it if that folder has moved since training</span></div>
    <div class="row"><label class="k">Output folder</label><input type="text" id="retrain_out_dir">
      <button data-browse="retrain_out_dir">Browse…</button></div>
    <div class="row"><label class="k">Top-up epochs</label><input type="number" id="epochs" min="1">
      <span class="hint">resumes from the current checkpoint with its original hyperparameters; not a full retrain</span></div>
    <p class="warnbox" id="basenote"></p>
    <p class="hint">The current checkpoint and its eval report are the baseline the new checkpoint must beat. The base
      files are the ones the current <code>.h5</code> was built from; corrections are appended to them as TRAIN rows only.</p>
  </section>

  <section>
    <h2>Run</h2>
    <div class="row">
      <button id="dry">Dry run (show commands)</button>
      <button class="primary" id="run">Run retraining cycle</button>
      <button class="danger" id="cancel" disabled>Cancel</button>
      <span id="ptext" class="hint"></span>
    </div>
    <div class="stages" id="stages"></div>
    <pre id="log"></pre><div id="partial" class="hint"></div>
  </section>

  <section id="result" style="display:none">
    <h2>Result</h2>
    <div id="result_body"></div>
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
const FIELDS = ['checkpoint', 'eval_report_current', 'base_manifest', 'base_split_csv', 'base_label_dir', 'data_root', 'transform_meta_root', 'retrain_out_dir', 'epochs'];
function form() { const f = {}; FIELDS.forEach(k => f[k] = $('#' + k).value.trim()); f.epochs = +f.epochs || 5; return f; }
function fill(c) { FIELDS.forEach(k => { if (c[k] !== undefined && c[k] !== null) $('#' + k).value = c[k]; }); }
let saveT;
function saveSoon() { clearTimeout(saveT); saveT = setTimeout(() => api('/rapi/save-config', form()), 600); }
document.querySelectorAll('input').forEach(el => el.addEventListener('change', saveSoon));
$('#base_split_csv').addEventListener('change', loadPreview);
$('#checkpoint').addEventListener('change', guess);

async function guess() {
  const {data} = await api('/rapi/guess?' + new URLSearchParams({checkpoint: $('#checkpoint').value.trim()}));
  if (data.eval_report_current && confirm(`Use this eval report as the baseline?\n${data.eval_report_current}`))
    $('#eval_report_current').value = data.eval_report_current;
  $('#retrain_out_dir').value = data.retrain_out_dir; saveSoon();
}

async function loadPreview() {
  const {data: p} = await api('/rapi/preview?' + new URLSearchParams({base_split_csv: $('#base_split_csv').value.trim()}));
  const s = p.stats || {}, n = k => (s[k] ? s[k].count : 0), ex = k => (s[k] ? s[k].exported : 0);
  let h = `<p>pending ${n('pending')} · confirmed ${n('confirmed')} (${ex('confirmed')} exported) · corrected ${n('corrected')} (${ex('corrected')} exported)</p>`;
  if (p.error) h += `<p class="err">${esc(p.error)}</p>`;
  else if (!p.groups.length) h += `<p>Nothing waiting for export. <a href="/review">Review some predictions</a> first.</p>`;
  else {
    h += `<table><tr><th>session</th><th>rat</th><th>windows</th><th>labels</th></tr>` +
      p.groups.map(g => `<tr><td>${esc(g.session)}</td><td>${g.rat}</td><td class="num">${g.windows}</td>
        <td class="labels-cell">${Object.entries(g.labels).map(([l, c]) => `${esc(l)} ${c}`).join(' · ')}</td></tr>`).join('') +
      `</table><p>${p.n_windows} reviewed windows will be added as TRAIN rows.</p>`;
  }
  if (p.eval_leaks && p.eval_leaks.length) h += `<p class="err">✗ In the base split's EVAL set (would leak into training): ${p.eval_leaks.map(esc).join(', ')}
    <button id="exclude" style="margin-left:8px">Exclude from retraining</button></p>`;
  else if ($('#base_split_csv').value.trim() && !p.split_error && p.groups && p.groups.length) h += `<p class="okc">✓ none of these sessions are in the eval split</p>`;
  if (p.split_error) h += `<p class="err">${esc(p.split_error)}</p>`;
  if (p.already_train && p.already_train.length) h += `<p class="warnbox">Already TRAIN sessions in the base split: ${p.already_train.map(esc).join(', ')}</p>`;
  $('#preview').innerHTML = h;
  if ($('#exclude')) $('#exclude').onclick = async () => {
    if (!confirm(`Keep ${p.eval_leaks.join(', ')} out of all retraining exports (verdicts are kept)?`)) return;
    const {ok, data} = await api('/rapi/exclude', {sessions: p.eval_leaks});
    if (!ok) return alert(data.error);
    loadPreview();
  };
}

__MODAL_JS__

document.querySelectorAll('[data-browse]').forEach(b => b.onclick = () => {
  const target = $('#' + b.dataset.browse);
  openBrowser({start: target.value, exts: b.dataset.exts, save: false, onpick: v => {
    target.value = v; saveSoon();
    if (b.dataset.browse === 'base_split_csv') loadPreview();
    if (b.dataset.browse === 'checkpoint') guess();
  }});
});

let logLen = 0, polling = null;
async function run(dry, force) {
  const {ok, status, data} = await api('/rapi/run', {...form(), dry_run: dry, force: !!force});
  if (ok) { logLen = 0; lastDecision = null; $('#log').textContent = ''; poll(); return; }
  if (data.errors) return alert(data.errors.join('\n\n'));
  if (data.warnings) { if (confirm(data.warnings.join('\n\n') + '\n\nRun anyway?')) run(dry, true); return; }
  alert(data.error || `failed (${status})`);
}
$('#dry').onclick = () => run(true);
$('#run').onclick = () => { if (confirm('Start a retraining cycle? This builds a new .h5, trains on the GPU and can take hours.')) run(false); };
$('#cancel').onclick = async () => { if (confirm('Cancel the retraining cycle? Nothing is marked exported and the retraining state is unchanged.')) await api('/rapi/cancel', {}); };

function fmtDur(s) { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; }

function renderStages(j) {
  const keys = j.stages.map(s => s[0]);
  const cur = j.stage === 'finished' ? keys.length : keys.indexOf(j.stage);
  $('#stages').innerHTML = j.status === 'idle' ? '' : j.stages.map(([k, t], i) => {
    let cls = i < cur ? 'done' : i === cur ? (j.status === 'running' ? 'on' : j.status === 'error' || j.status === 'cancelled' ? 'fail' : 'done') : '';
    if (j.params && j.params.dry_run && j.status === 'done') cls = '';
    return `<span class="${cls}">${esc(t)}</span>`;
  }).join('');
}

async function poll() {
  clearTimeout(polling);
  const {data: j} = await api('/rapi/job?since=' + logLen);
  if (j.log && j.log.length) { const pre = $('#log'); pre.textContent += j.log.join('\n') + '\n'; pre.scrollTop = pre.scrollHeight; }
  logLen = j.log_len;
  $('#partial').textContent = j.partial || '';
  const busy = j.status === 'running';
  $('#run').disabled = busy; $('#dry').disabled = busy; $('#cancel').disabled = !busy;
  renderStages(j);
  let t = '';
  if (busy) t = `${j.params.dry_run ? 'dry run' : 'running'} · ${fmtDur(j.now - j.started)} elapsed`;
  else if (j.status === 'done') t = `${j.params.dry_run ? 'dry run finished' : 'finished'} in ${fmtDur(j.finished - j.started)}`;
  else if (j.status === 'error') t = `failed (exit ${j.returncode}), see the log`;
  else if (j.status === 'cancelled') t = 'cancelled';
  $('#ptext').textContent = t;
  renderResult(j);
  if (busy) polling = setTimeout(poll, 1500);
}

let lastDecision = null;
function renderResult(j) {
  const c = j.pending;
  if (!c && !lastDecision) { $('#result').style.display = 'none'; return; }
  $('#result').style.display = '';
  if (!c) { $('#result_body').innerHTML = lastDecision; return; }
  let h = `<p>Awaiting your decision: <code>${esc(c.retrain_out_dir)}</code><br>
    New checkpoint: <code>${esc(c.new_checkpoint)}</code> · trained on ${c.n_windows} reviewed windows from ${c.sessions.map(esc).join(', ')}</p>`;
  if (c.comparison) h += `<p class="hint">Population-weighted accuracy, current (first) vs new (second):</p><pre class="compare">${esc(c.comparison.replace(/\s+$/, ''))}</pre>`;
  h += `<p>Only deploy it if it beats the current checkpoint above. Nothing changes until you choose:</p>
    <div class="row"><button class="primary" id="deploy">Deploy this checkpoint</button>
    <button id="reject">Reject: keep the current one</button></div>
    <p class="hint">Deploy marks this cycle's reviewed rows as exported, points the deploy page at the new checkpoint and makes this
      cycle's merged manifest, split, labels and eval report the inputs of the next cycle. Reject leaves everything as it was;
      the reviewed rows are used again next cycle. Same as <code>retrain_cycle.py --promote/--reject</code>.</p>`;
  $('#result_body').innerHTML = h;
  const decide = async (action, question, done) => {
    if (!confirm(question)) return;
    const {ok, data} = await api('/rapi/' + action, {retrain_out_dir: c.retrain_out_dir});
    if (!ok) return alert(data.error);
    lastDecision = done; fill(data.next); showBase(data.next); loadPreview(); poll();
  };
  $('#deploy').onclick = () => decide('deploy', 'Deploy the new checkpoint?',
    `<p class="okc">✓ Deployed <code>${esc(c.new_checkpoint)}</code>. The deploy page uses it now, and the inputs above are this cycle's merged files.</p>`);
  $('#reject').onclick = () => decide('reject', 'Reject the new checkpoint and keep the current one?',
    `<p>Rejected. The current checkpoint stays deployed, and the reviewed rows will be used again in the next cycle.</p>`);
}

function showBase(c) {
  $('#basenote').innerHTML = c.promoted_from ? `Inputs come from the cycle promoted from <code>${esc(c.promoted_from)}</code>.
    Changing the base manifest, split or label dir would drop its corrections from training.` : '';
}

api('/rapi/config').then(({data}) => { fill(data); showBase(data); loadPreview(); });
poll();
</script>
</body>
</html>
"""
