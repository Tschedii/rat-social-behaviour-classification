"""Orchestrates one retraining cycle from the correction log: export reviewed corrections,
rebuild the .h5 + window_index.csv (reusing 2_build_h5_dataset.py / build_index.py exactly as
they already are), resume-train a few epochs from the currently-deployed checkpoint, evaluate
on the untouched eval split, and report whether the result actually beats what's deployed now
by population-weighted accuracy.

This script NEVER auto-promotes a new checkpoint into deployment. A finished cycle waits for
an explicit decision, taken after looking at the comparison:

    python3 retrain_cycle.py --promote <retrain-out-dir>   # deploy the new checkpoint
    python3 retrain_cycle.py --reject  <retrain-out-dir>   # keep the current one

This mirrors the reasoning used all through this project's checkpoint-selection work: a
checkpoint doesn't get trusted until it's shown to beat the alternative on the metric that
matters, on data that was never touched to produce it.

State between cycles (~/.rat_retrain_state.json) makes the decision stick:
  - Correction-log rows are only marked exported on --promote. A rejected (or never decided)
    cycle leaves them unexported, so they're included again in the next cycle instead of
    silently disappearing from all future training sets.
  - --promote makes that cycle's merged manifest / split / label dir, new checkpoint and new
    eval report the BASE of the next cycle, so `--base-*` / `--checkpoint` /
    `--eval-report-current` can be left out from then on. Passing base files that differ from
    the promoted ones is refused (it would drop the promoted corrections from training) unless
    you add --ignore-promoted-base. --promote also points the deploy GUIs at the new checkpoint.
  - A new cycle can't start while the previous one is still awaiting a decision.

First cycle, run on the GPU server from the 6_deployment/ directory:

    python3 retrain_cycle.py \\
        --checkpoint training_output/calibrated_epoch155_v2.pt \\
        --base-manifest /path/to/training_manifest.csv \\
        --base-split-csv /path/to/train_eval_split.csv \\
        --base-label-dir /path/to/label_clean \\
        --data-root /path/to/master_thesis_social_behaviour \\
        --retrain-out-dir training_output/retrain_$(date +%Y%m%d) \\
        --eval-report-current training_output/eval_calibrated_epoch155_v2/classification_report_eval.txt \\
        --epochs 5

Every cycle after a --promote:

    python3 retrain_cycle.py --retrain-out-dir training_output/retrain_$(date +%Y%m%d) --epochs 5

`--status` shows the current base and any cycle awaiting a decision. Add --dry-run to print
every command this would run (export, build_h5, build_index, train, evaluate, compare) without
executing any of them or touching the state; it's the only mode testable without a GPU/real data.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
# 6_deployment/ is a SIBLING of 5_rat_transformer_training/, not its parent — train.py,
# evaluate.py, helpers/ all live in that sibling directory.
PROJECT_ROOT = DEPLOY_DIR.parent / "5_rat_transformer_training"
STORE_TRUE_FLAGS = {"no_checkpoint", "use_flow", "use_scale_feature", "no_class_weights"}
# these describe THIS retraining run, not the resumed checkpoint's original training — never
# copy them forward from train_args
TRAIN_ARGS_OVERRIDE_KEYS = {"h5", "index_csv", "out_dir", "resume", "epochs", "normstats",
                            "label_map", "gpu_id"}

STATE_PATH = Path.home() / ".rat_retrain_state.json"
DEPLOY_CONFIG_PATH = Path.home() / ".rat_deploy_gui_config.json"  # gui.py / deploy_web.py settings
CYCLE_FILE = "cycle.json"
# carried from a promoted cycle into the next one; the three data files must match it
BASE_KEYS = ("checkpoint", "eval_report_current", "base_manifest", "base_split_csv", "base_label_dir",
             "data_root")
CHAINED_KEYS = ("base_manifest", "base_split_csv", "base_label_dir")
DIR_KEYS = ("base_label_dir", "data_root")
# optional, also carried from a promoted cycle: folder of the training sessions' transform
# metadata (scale feature). Unset = the path recorded in the checkpoint's train_args.
META_KEY = "transform_meta_root"

sys.path.insert(0, str(DEPLOY_DIR))
from export_corrections_for_retrain import build_export, merge_csv, write_export  # noqa: E402
from correction_log import DEFAULT_DB_PATH, mark_exported  # noqa: E402


def find_build_h5_script(override=None):
    if override:
        return Path(override)
    candidate = PROJECT_ROOT.parent / "4_building_training_data" / "2_build_h5_dataset.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"couldn't find 2_build_h5_dataset.py at {candidate} — pass --build-h5-script explicitly."
    )


def train_args_to_cli(train_args, overrides):
    """Reconstructs train.py's CLI flags from a checkpoint's saved train_args (vars(args) at
    the time it was trained), so a resume is an exact continuation of every original
    hyperparameter except the ones this retraining cycle deliberately overrides (new data
    paths, --resume itself, --epochs)."""
    args_out = list(overrides)  # overrides already includes --resume/--h5/--index-csv/etc.
    seen = {a.lstrip("-").replace("-", "_") for a in overrides if a.startswith("--")}
    for key, val in train_args.items():
        if key in TRAIN_ARGS_OVERRIDE_KEYS or key in seen:
            continue
        flag = "--" + key.replace("_", "-")
        if key in STORE_TRUE_FLAGS:
            if val:
                args_out.append(flag)
        elif val is not None:
            args_out += [flag, str(val)]
    return args_out


def scale_meta_root(train_args, override, dry_run):
    """The metadata folder this cycle trains and evaluates with, checked BEFORE the slow steps:
    a checkpoint trained with the scale feature needs the training sessions' transform metadata
    to normalize it, and the path recorded in the checkpoint goes stale when that folder moves.
    Returns the override to pass on (None = keep the recorded path)."""
    if not train_args.get("use_scale_feature"):
        if override:
            print(f"note: --transform-meta-root {override} ignored, the checkpoint doesn't use the scale feature")
        return None
    recorded = train_args.get(META_KEY)
    root = override or recorded
    if override:
        print(f"scale feature: using metadata folder {override} (overrides {recorded or 'none'} "
              f"recorded in the checkpoint)")
    if not root or not Path(root).is_dir():
        msg = (f"the checkpoint uses the scale feature, but its metadata folder "
               f"{'is not recorded' if not root else 'was not found: ' + root}.\n"
               f"Pass --transform-meta-root (Metadata folder on the Retrain page) with where the "
               f"training transform metadata lives now.")
        if dry_run:
            print(f"WARNING: {msg}")
        else:
            raise SystemExit(f"error: {msg}")
    elif not override:
        print(f"scale feature: using metadata folder {root} (recorded in the checkpoint)")
    return override


def script_accepts(script, flag):
    """Whether a script's argparse setup defines `flag` (read from its source, not executed)."""
    try:
        return f'"{flag}"' in Path(script).read_text() or f"'{flag}'" in Path(script).read_text()
    except OSError:
        return False


def run(cmd, dry_run, cwd=None, capture=False):
    """capture=True also keeps the command's output (printed as usual) in result.stdout."""
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return None
    if capture:
        result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout, end="", flush=True)
    else:
        result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {printable}")
    return result


# ---------------------------------------------------------------------------
# state between cycles
# ---------------------------------------------------------------------------

def load_state():
    state = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    state.setdefault("base", None)      # inputs for the next cycle, set by --promote
    state.setdefault("pending", None)   # retrain-out-dir of a cycle awaiting a decision
    state.setdefault("history", [])
    return state


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_cycle(out_dir):
    path = Path(out_dir).expanduser().resolve() / CYCLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"no {CYCLE_FILE} in {path.parent} — not a finished retraining cycle")
    return json.loads(path.read_text())


def save_cycle(out_dir, cycle):
    (Path(out_dir) / CYCLE_FILE).write_text(json.dumps(cycle, indent=2))


def pending_cycle(state):
    """The cycle awaiting a decision, or None. Clears a stale pointer (folder deleted, or the
    cycle was decided some other way)."""
    if not state["pending"]:
        return None
    try:
        cycle = load_cycle(state["pending"])
    except FileNotFoundError:
        cycle = None
    if cycle is None or cycle.get("status") != "awaiting_decision":
        state["pending"] = None
        save_state(state)
        return None
    return cycle


def resolve_inputs(args, state):
    """--checkpoint / --eval-report-current / --base-* / --data-root: explicit values win for the
    checkpoint, its eval report and the data root; the three base data files must equal the
    promoted base (if any) unless --ignore-promoted-base, because a different base silently
    drops every promoted cycle's corrections from training."""
    base = state["base"] or {}
    inputs, errors = {}, []
    for key in BASE_KEYS:
        given = getattr(args, key)
        value = str(Path(given).expanduser().resolve()) if given else base.get(key)
        if given and key in CHAINED_KEYS and base.get(key) and value != base[key] \
                and not args.ignore_promoted_base:
            errors.append(
                f"--{key.replace('_', '-')} {value}\n    differs from the base promoted from "
                f"{base.get('promoted_from')}:\n    {base[key]}\n    Using it would drop that "
                f"cycle's corrections from training. Leave it out to use the promoted base, or "
                f"add --ignore-promoted-base if this is intended."
            )
        if not value:
            errors.append(f"--{key.replace('_', '-')} is required (no promoted cycle to take it from yet)")
            continue
        exists = Path(value).is_dir() if key in DIR_KEYS else Path(value).is_file()
        if not exists:
            errors.append(f"--{key.replace('_', '-')}: not found: {value}")
        inputs[key] = value
    given = getattr(args, META_KEY, None)
    meta = str(Path(given).expanduser().resolve()) if given else base.get(META_KEY)
    if meta and not Path(meta).is_dir():
        errors.append(f"--transform-meta-root: not found: {meta}")
    inputs[META_KEY] = meta or None
    if args.checkpoint and base.get("checkpoint") and inputs.get("checkpoint") != base["checkpoint"] \
            and not args.eval_report_current:
        errors.append("--checkpoint differs from the promoted one, so also pass its "
                      "--eval-report-current (the promoted eval report belongs to the promoted checkpoint)")
    return inputs, errors


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------

def promote(out_dir):
    """Deploy a finished cycle's checkpoint: mark its correction-log rows exported, make its
    merged files + new checkpoint + eval report the next cycle's base, and point the deploy GUIs
    at the new checkpoint."""
    out_dir = str(Path(out_dir).expanduser().resolve())
    cycle = load_cycle(out_dir)
    if cycle["status"] != "awaiting_decision":
        raise RuntimeError(f"cycle {out_dir} was already {cycle['status']}")
    needed = {k: cycle[k] for k in ("new_checkpoint", "merged_manifest", "merged_split",
                                    "merged_label_dir", "eval_report_new")}
    missing = [f"{k}: {v}" for k, v in needed.items() if not Path(v).exists()]
    if missing:
        raise FileNotFoundError("can't promote, files from this cycle are missing:\n  " + "\n  ".join(missing))

    mark_exported(cycle["db"], cycle["row_ids"])
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    state = load_state()
    state["base"] = {
        "checkpoint": cycle["new_checkpoint"],
        "eval_report_current": cycle["eval_report_new"],
        "base_manifest": cycle["merged_manifest"],
        "base_split_csv": cycle["merged_split"],
        "base_label_dir": cycle["merged_label_dir"],
        "data_root": cycle["inputs"]["data_root"],
        META_KEY: cycle["inputs"].get(META_KEY),
        "promoted_from": out_dir,
        "promoted_at": now,
    }
    if state["pending"] == out_dir:
        state["pending"] = None
    state["history"].append({"retrain_out_dir": out_dir, "decision": "promoted", "at": now})
    save_state(state)

    deploy_cfg = {}
    if DEPLOY_CONFIG_PATH.exists():
        try:
            deploy_cfg = json.loads(DEPLOY_CONFIG_PATH.read_text())
        except Exception:
            pass
    deploy_cfg["checkpoint"] = cycle["new_checkpoint"]
    if cycle["inputs"].get(META_KEY):  # keep deploy's scale-feature metadata in step with training
        deploy_cfg[META_KEY] = cycle["inputs"][META_KEY]
    DEPLOY_CONFIG_PATH.write_text(json.dumps(deploy_cfg, indent=2))

    cycle.update(status="promoted", decided_at=now)
    save_cycle(out_dir, cycle)
    return cycle


def reject(out_dir):
    """Keep the current checkpoint. The cycle's rows were never marked exported, so they're
    included again in the next cycle; the base stays what it was."""
    out_dir = str(Path(out_dir).expanduser().resolve())
    cycle = load_cycle(out_dir)
    if cycle["status"] != "awaiting_decision":
        raise RuntimeError(f"cycle {out_dir} was already {cycle['status']}")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    state = load_state()
    if state["pending"] == out_dir:
        state["pending"] = None
    state["history"].append({"retrain_out_dir": out_dir, "decision": "rejected", "at": now})
    save_state(state)
    cycle.update(status="rejected", decided_at=now)
    save_cycle(out_dir, cycle)
    return cycle


def print_status():
    state = load_state()
    cycle = pending_cycle(state)
    print(f"state file: {STATE_PATH}")
    if state["base"]:
        print(f"\nbase for the next cycle (promoted from {state['base']['promoted_from']} "
              f"at {state['base']['promoted_at']}):")
        for k in BASE_KEYS:
            print(f"  {k:20s} {state['base'][k]}")
        print(f"  {META_KEY:20s} {state['base'].get(META_KEY) or '(the one recorded in the checkpoint)'}")
    else:
        print("\nno promoted cycle yet — the next cycle needs all --base-* / --checkpoint / "
              "--eval-report-current / --data-root arguments")
    if cycle:
        print(f"\nawaiting a decision: {state['pending']}")
        print(f"  new checkpoint: {cycle['new_checkpoint']}")
        if cycle.get("comparison"):
            print("  comparison:\n    " + cycle["comparison"].rstrip().replace("\n", "\n    "))
        print(f"  decide with --promote {state['pending']}  or  --reject {state['pending']}")
    else:
        print("\nno cycle awaiting a decision")


# ---------------------------------------------------------------------------
# one cycle
# ---------------------------------------------------------------------------

def run_cycle(args):
    state = load_state()
    cycle = pending_cycle(state)
    if cycle and not args.dry_run:
        raise SystemExit(
            f"the cycle in {state['pending']} is still awaiting a decision (new checkpoint "
            f"{cycle['new_checkpoint']}).\nDecide first:\n"
            f"  python3 retrain_cycle.py --promote {state['pending']}\n"
            f"  python3 retrain_cycle.py --reject  {state['pending']}"
        )
    inputs, errors = resolve_inputs(args, state)
    if errors:
        raise SystemExit("\n".join(errors))
    inputs["epochs"] = args.epochs
    if state["base"]:
        print(f"using the base promoted from {state['base']['promoted_from']}")

    import torch
    train_args = torch.load(inputs["checkpoint"], map_location="cpu")["args"]
    meta_override = scale_meta_root(train_args, inputs[META_KEY], args.dry_run)

    retrain_out = Path(args.retrain_out_dir).expanduser().resolve()
    export_dir = retrain_out / "corrections_export"
    retrain_out.mkdir(parents=True, exist_ok=True)

    # ---- 1. export reviewed corrections ----
    manifest_rows, label_csvs, session_ids, exported_ids = build_export(args.db)
    if not manifest_rows:
        print("nothing to retrain on — no confirmed/corrected rows pending export in the correction log. "
              "Review some predictions with 6_deployment/review_gui.py first.")
        return
    write_export(export_dir, manifest_rows, label_csvs, session_ids)
    n_windows = sum(len(v) for v in label_csvs.values())
    print(f"exported {len(manifest_rows)} manifest rows ({n_windows} windows, {len(session_ids)} session(s))")

    merged_manifest = export_dir / "merged_manifest.csv"
    merged_split = export_dir / "merged_split.csv"
    merge_csv(inputs["base_manifest"], export_dir / "corrections_manifest.csv", merged_manifest)
    merge_csv(inputs["base_split_csv"], export_dir / "corrections_split.csv", merged_split)

    merged_label_dir = retrain_out / "merged_label_clean"
    if not args.dry_run:
        if merged_label_dir.exists():
            shutil.rmtree(merged_label_dir)
        shutil.copytree(inputs["base_label_dir"], merged_label_dir)
        for f in (export_dir / "label_clean").glob("*.csv"):
            shutil.copy(f, merged_label_dir / f.name)
    print(f"merged label dir -> {merged_label_dir} (base + {n_windows} corrected/confirmed segments)")

    # ---- 2. rebuild the .h5 ----
    build_h5_script = find_build_h5_script(args.build_h5_script)
    h5_out = retrain_out / "rat_behaviour_retrain.h5"
    normstats_path = PROJECT_ROOT / "rgb_normstats.json"
    cmd = [
        sys.executable, str(build_h5_script),
        "--manifest", str(merged_manifest),
        "--data-root", inputs["data_root"],
        "--label-dir", str(merged_label_dir),
        "--split-csv", str(merged_split),
        "--cams", "1,2,3,4,5,6",
        "--frames-per-segment", "32",
        "--normstats", str(normstats_path),
        "--out", str(h5_out),
        "--report", str(retrain_out / "conversion_report.json"),
    ]
    run(cmd, args.dry_run)

    # ---- 3. rebuild window_index.csv ----
    index_out_dir = retrain_out
    cmd = [
        sys.executable, args.build_index_script,
        "--h5", str(h5_out),
        "--split-csv", str(merged_split),
        "--out-dir", str(index_out_dir),
    ]
    run(cmd, args.dry_run)
    index_csv = index_out_dir / "window_index.csv"

    # ---- 4. resume-train a few epochs ----
    train_out_dir = retrain_out / "training_output"
    overrides = [
        "--h5", str(h5_out),
        "--index-csv", str(index_csv),
        "--normstats", str(normstats_path),
        "--out-dir", str(train_out_dir),
        "--resume", inputs["checkpoint"],
        "--epochs", str(args.epochs),
    ]
    if meta_override:
        overrides += ["--transform-meta-root", meta_override]
    cmd = [sys.executable, args.train_script] + train_args_to_cli(train_args, overrides)
    run(cmd, args.dry_run, cwd=str(PROJECT_ROOT))

    # ---- 5. evaluate the new checkpoint on the untouched eval split ----
    new_checkpoint = train_out_dir / "best.pt"
    eval_out_dir = retrain_out / "eval_new"
    cmd = [
        sys.executable, args.evaluate_script,
        "--h5", str(h5_out), "--index-csv", str(index_csv), "--normstats", str(normstats_path),
        "--checkpoint", str(new_checkpoint), "--split", "eval", "--batch-size", "2",
        "--out-dir", str(eval_out_dir),
    ]
    if meta_override:
        if script_accepts(args.evaluate_script, "--transform-meta-root"):
            cmd += ["--transform-meta-root", meta_override]
        else:
            # evaluate.py rebuilds the model and its inputs from the new checkpoint's saved args
            new_meta = None if args.dry_run else \
                torch.load(new_checkpoint, map_location="cpu")["args"].get(META_KEY)
            if args.dry_run:
                print("note: evaluate.py has no --transform-meta-root, so it uses the metadata folder "
                      "saved in the new checkpoint (train.py saves the one passed to it)")
            elif new_meta != meta_override:
                print(f"WARNING: the new checkpoint recorded metadata folder {new_meta}, not "
                      f"{meta_override}, and evaluate.py can't be given it; its scale feature may run "
                      f"with neutral values")
    run(cmd, args.dry_run)

    # ---- 6. compare against the currently-deployed checkpoint ----
    new_report = eval_out_dir / "classification_report_eval.txt"
    cmd = [sys.executable, args.pop_weighted_script, inputs["eval_report_current"], str(new_report)]
    result = run(cmd, args.dry_run, capture=True)

    if args.dry_run:
        print("\n(dry run complete — nothing was executed, and neither the correction log nor the "
              "retraining state was changed)")
        return

    # ---- 7. wait for a decision; nothing is marked exported until --promote ----
    save_cycle(retrain_out, {
        "status": "awaiting_decision",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "db": str(Path(args.db).expanduser().resolve()),
        "inputs": inputs,
        "row_ids": exported_ids,
        "sessions": session_ids,
        "n_windows": n_windows,
        "new_checkpoint": str(new_checkpoint),
        "merged_manifest": str(merged_manifest),
        "merged_split": str(merged_split),
        "merged_label_dir": str(merged_label_dir),
        "eval_report_new": str(new_report),
        "comparison": result.stdout if result is not None else "",
    })
    state["pending"] = str(retrain_out)
    save_state(state)
    print(f"\nnew checkpoint: {new_checkpoint}")
    print("Compare the population-weighted accuracy table above, then decide:\n"
          f"  python3 retrain_cycle.py --promote {retrain_out}   # deploy it; this cycle becomes the next base\n"
          f"  python3 retrain_cycle.py --reject  {retrain_out}   # keep the current checkpoint\n"
          f"The {len(exported_ids)} reviewed rows used here stay unexported until you promote.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = ap.add_mutually_exclusive_group()
    actions.add_argument("--promote", metavar="RETRAIN_OUT_DIR", help="deploy a finished cycle's checkpoint")
    actions.add_argument("--reject", metavar="RETRAIN_OUT_DIR", help="discard a finished cycle's checkpoint")
    actions.add_argument("--status", action="store_true", help="show the current base and any undecided cycle")

    ap.add_argument("--checkpoint", default=None,
                    help="currently-deployed checkpoint to resume from (default: the promoted one)")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--base-manifest", default=None, help="default: the promoted cycle's merged manifest")
    ap.add_argument("--base-split-csv", default=None, help="default: the promoted cycle's merged split")
    ap.add_argument("--base-label-dir", default=None, help="default: the promoted cycle's merged label dir")
    ap.add_argument("--data-root", default=None, help="default: the one the promoted cycle used")
    ap.add_argument("--ignore-promoted-base", action="store_true",
                    help="allow --base-* files that differ from the promoted cycle's")
    ap.add_argument("--retrain-out-dir", default=None)
    ap.add_argument("--eval-report-current", default=None,
                    help="classification_report_eval.txt for --checkpoint, to compare against "
                         "(default: the promoted cycle's)")
    ap.add_argument("--transform-meta-root", default=None,
                    help="folder of the training sessions' transform metadata, for checkpoints trained "
                         "with the scale feature (default: the promoted cycle's, else the one recorded "
                         "in the checkpoint). Set it if that folder has moved since training")
    ap.add_argument("--epochs", type=int, default=5, help="top-up epochs for this cycle, not a full retrain")
    ap.add_argument("--build-h5-script", default=None)
    ap.add_argument("--build-index-script", default=str(PROJECT_ROOT / "helpers" / "build_index.py"))
    ap.add_argument("--train-script", default=str(PROJECT_ROOT / "train.py"))
    ap.add_argument("--evaluate-script", default=str(PROJECT_ROOT / "evaluate.py"))
    ap.add_argument("--pop-weighted-script", default=str(PROJECT_ROOT / "helpers" / "population_weighted_accuracy.py"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.status:
        return print_status()
    if not (args.promote or args.reject or args.retrain_out_dir):
        ap.error("--retrain-out-dir is required to run a cycle")
    try:
        if args.promote:
            cycle = promote(args.promote)
            print(f"promoted {cycle['new_checkpoint']}\n"
                  f"  - marked {len(cycle['row_ids'])} correction-log rows as exported\n"
                  f"  - the deploy GUIs now use it ({DEPLOY_CONFIG_PATH})\n"
                  f"  - the next cycle's base is this cycle's merged files; run the next cycle with just "
                  f"--retrain-out-dir (and --epochs)")
        elif args.reject:
            cycle = reject(args.reject)
            print(f"rejected {cycle['new_checkpoint']}; the current checkpoint and base stay as they are, "
                  f"and the {len(cycle['row_ids'])} reviewed rows will be included in the next cycle")
        else:
            run_cycle(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
