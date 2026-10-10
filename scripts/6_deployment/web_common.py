"""Shared page pieces for the 6_deployment web front ends (deploy_web.py, retrain_web.py):
theme CSS and the server-side file browser modal. Pages embed them via the __CSS__,
__MODAL_HTML__ and __MODAL_JS__ placeholders (see render()).

The modal JS expects the page to define $, esc and api(path, body) -> {ok, status, data}, and
is opened with openBrowser({start, exts, save, onpick}).
"""

CSS = r""":root {
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
header { display: flex; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line);
         background: var(--panel); position: sticky; top: 0; z-index: 2; }
header h1 { font-size: 15px; margin: 0; }
header nav a { color: var(--muted); text-decoration: none; margin-right: 12px; }
header nav a.on { color: var(--ink); font-weight: 600; }
main { padding: 16px; max-width: 1100px; margin: 0 auto; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; }
section h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 10px; }
.row { display: flex; gap: 8px; align-items: center; margin: 6px 0; flex-wrap: wrap; }
.row > label.k { width: 90px; flex: none; color: var(--muted); }
input[type=text], input[type=number] { flex: 1; min-width: 200px; background: var(--bg); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--ink); }
input[type=number] { flex: none; width: 80px; min-width: 0; }
button { font: inherit; color: inherit; background: var(--chip); border: 1px solid var(--line); border-radius: 6px;
         padding: 5px 12px; cursor: pointer; }
button:hover:not(:disabled) { border-color: var(--muted); }
button:disabled { opacity: .5; cursor: default; }
button.primary { background: var(--good); border-color: var(--good); color: #fff; padding: 7px 20px; font-weight: 600; }
button.danger { color: var(--bad); }
.hint { color: var(--muted); font-size: 12px; }
.warnbox { color: var(--warn); font-size: 12px; margin: 0 0 8px; }
.st { font-size: 12px; min-width: 220px; flex: 1 0 100%; padding-left: 98px; margin-top: -3px; }
.st.ok { color: var(--good); } .st.bad { color: var(--bad); } .st.muted { color: var(--muted); }
.progress { height: 10px; background: var(--chip); border-radius: 5px; overflow: hidden; flex: 1; min-width: 200px; }
.progress > i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .3s; }
pre#log { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; max-height: 280px;
          overflow: auto; font-size: 12px; margin: 8px 0 0; white-space: pre-wrap; }
table { border-collapse: collapse; font-size: 13px; }
td, th { padding: 3px 12px 3px 0; text-align: left; }
th { color: var(--muted); font-weight: 500; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,.4); display: none; align-items: center; justify-content: center; z-index: 10; }
.modal-bg.show { display: flex; }
.modal { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; width: min(720px, 94vw);
         max-height: 80vh; display: flex; flex-direction: column; }
.modal .top { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; gap: 8px; }
.modal .list { overflow: auto; flex: 1; padding: 4px 0; }
.modal .list div { padding: 4px 14px; cursor: pointer; font: 13px ui-monospace, Menlo, monospace; }
.modal .list div:hover { background: var(--chip); }
.modal .list div.file { color: var(--accent); }
.modal .bottom { padding: 10px 12px; border-top: 1px solid var(--line); display: flex; gap: 8px; justify-content: flex-end; align-items: center; }
"""

MODAL_HTML = r"""<div class="modal-bg" id="modal"><div class="modal">
  <div class="top"><button id="m_up">↑ Up</button><input type="text" id="m_path"><button id="m_go">Go</button></div>
  <div class="list" id="m_list"></div>
  <div class="bottom"><span class="hint" id="m_hint" style="margin-right:auto"></span>
    <input type="text" id="m_name" placeholder="file name" style="display:none;flex:0 1 220px">
    <button id="m_cancel">Cancel</button><button id="m_ok" class="primary">Select this folder</button></div>
</div></div>
"""

MODAL_JS = r"""// ---- server-side file browser ----
let pick = null;
async function openBrowser(opts) {
  pick = opts;
  $('#m_name').style.display = opts.save ? '' : 'none';
  $('#m_ok').textContent = opts.save ? 'Save here' : (opts.exts ? 'Select' : 'Select this folder');
  $('#m_ok').style.display = opts.exts && !opts.save ? 'none' : '';
  $('#m_hint').textContent = opts.exts && !opts.save ? 'click a file to select it' : '';
  let start = opts.start || '';
  if (opts.save && start) { const parts = start.split('/'); $('#m_name').value = parts.pop(); start = parts.join('/'); }
  $('#modal').classList.add('show');
  await go(start);
}
async function go(path) {
  const {ok, data} = await api('/dapi/ls?' + new URLSearchParams({path, exts: (pick.exts || '')}));
  if (!ok) { if (path) return go(''); return; }
  pick.cur = data.path; $('#m_path').value = data.path; $('#m_up').disabled = !data.parent;
  $('#m_list').innerHTML = data.dirs.map(d => `<div data-d="${esc(d)}">📁 ${esc(d)}</div>`).join('') +
    data.files.map(f => `<div class="file" data-f="${esc(f)}">${esc(f)}</div>`).join('') ||
    '<div class="hint" style="cursor:default">(empty)</div>';
  $('#m_list').querySelectorAll('[data-d]').forEach(el => el.onclick = () => go(data.path + '/' + el.dataset.d));
  $('#m_list').querySelectorAll('[data-f]').forEach(el => el.onclick = () => {
    if (pick.save) $('#m_name').value = el.dataset.f; else done(data.path + '/' + el.dataset.f); });
  $('#m_up').onclick = () => go(data.parent);
}
function done(value) { $('#modal').classList.remove('show'); if (value != null) pick.onpick(value); }
$('#m_go').onclick = () => go($('#m_path').value);
$('#m_path').addEventListener('keydown', e => { if (e.key === 'Enter') go($('#m_path').value); });
$('#m_cancel').onclick = () => done(null);
$('#m_ok').onclick = () => {
  if (pick.save) { const n = $('#m_name').value.trim(); if (!n) return; done(pick.cur + '/' + (n.endsWith('.json') ? n : n + '.json')); }
  else done(pick.cur);
};
document.addEventListener('keydown', e => { if (e.key === 'Escape') done(null); });

"""


def render(page):
    return page.replace("__CSS__", CSS).replace("__MODAL_HTML__", MODAL_HTML).replace("__MODAL_JS__", MODAL_JS)
