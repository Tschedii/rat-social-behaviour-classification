"""Turns reviewed rows from the correction log (6_deployment/correction_log.py) into the exact
manifest-row / label-CSV / split-CSV format 4_building_training_data/2_build_h5_dataset.py
and helpers/build_index.py already expect — so retraining reuses those proven scripts
unchanged instead of a second, parallel H5-writing implementation that could drift from them.

Only rows a human has reviewed (status 'confirmed' or 'corrected') are ever exported — an
unreviewed prediction, however confident, is never treated as ground truth. Every exported
row becomes a TRAIN-split row; this script never writes to eval, so the fixed eval split
correction-log entries can never leak into (and inflate) the number the retrained checkpoint
gets judged on.

Output (under --out-dir, default 6_deployment/corrections_export/):
    corrections_manifest.csv     -- rows matching 2_build_h5_dataset.py's --manifest schema
    corrections_split.csv        -- session_id,split rows (all "train") for --split-csv
    label_clean/<session>_rat_<N>.csv   -- one per (session, rat), behavior/start_frame/end_frame

With a base (the manifest / split / label dir the current .h5 was built from) it also writes
merged_manifest.csv, merged_split.csv and merged_label_clean/ = base + corrections, ready to
hand straight to 2_build_h5_dataset.py. Without one, only the correction-only fragments are
written, for you to merge by hand.

The base works like retrain_cycle.py's: once a cycle has been promoted (retrain_cycle.py
--promote, or "Deploy" on the Retrain page), --base-* default to that cycle's merged files,
and passing different ones is refused unless --ignore-promoted-base, because an older base
silently drops the promoted corrections from training. Before the first promotion, pass them:

    python3 export_corrections_for_retrain.py \\
        --base-manifest /path/to/training_manifest.csv \\
        --base-split-csv /path/to/train_eval_split.csv \\
        --base-label-dir /path/to/label_clean \\
        --out-dir 6_deployment/corrections_export

By default this only writes files and leaves the correction log alone, so the rows stay
available to retrain_cycle.py (which marks them exported only when its checkpoint is
promoted). Pass --mark-exported only if you build and train on this export by hand and keep
the resulting checkpoint; it's refused while a retrain_cycle.py cycle awaits a decision.
"""
import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correction_log import DEFAULT_DB_PATH, get_exportable, mark_exported, row_subdirs

MANIFEST_FIELDS = ["rat_id", "object_id", "experiment_name", "label_csv_path"] + [
    f"cam{c}_{k}_path" for c in range(1, 7) for k in ("rgb", "mask")
]


def build_export(db_path):
    rows = get_exportable(db_path)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["session_label"], r["rat"])].append(r)

    manifest_rows = []
    label_csvs = {}   # (session, rat) -> list of {"behavior","start_frame","end_frame"}
    session_ids = set()
    exported_ids = []

    for (session_label, rat), group_rows in groups.items():
        cam_lists = {json.dumps(json.loads(r["camera_dirs"])) for r in group_rows}
        if len(cam_lists) > 1:
            raise ValueError(
                f"session '{session_label}' rat {rat} has predictions logged against different "
                f"camera folder sets — refusing to merge them under one session_id, since that "
                f"would silently mix footage from different recordings. Re-run with distinct "
                f"session labels per recording."
            )
        camera_dirs = json.loads(group_rows[0]["camera_dirs"])
        if len(camera_dirs) != 6:
            raise ValueError(f"session '{session_label}' rat {rat}: expected 6 camera dirs, got {len(camera_dirs)}")
        sub_sets = {json.dumps(row_subdirs(r), sort_keys=True) for r in group_rows}
        if len(sub_sets) > 1:
            raise ValueError(
                f"session '{session_label}' rat {rat} has predictions logged with different camera "
                f"subfolder names ({', '.join(sorted(sub_sets))}) — refusing to merge them under one "
                f"session_id. Re-run with distinct session labels per subfolder layout."
            )
        subdirs = row_subdirs(group_rows[0])
        folders = {}
        for cam_idx, cam_dir in enumerate(camera_dirs, start=1):
            folders[f"cam{cam_idx}_rgb_path"] = Path(cam_dir) / subdirs["rgb"]
            folders[f"cam{cam_idx}_mask_path"] = Path(cam_dir) / subdirs[f"mask{rat}"]
        missing = [str(f) for f in folders.values() if not f.is_dir()]
        if missing:
            hint = ("" if group_rows[0].get("subdirs") else
                    f" These rows were logged before subfolder names were recorded, so the default names "
                    f"were assumed; if the session used others, record them with: python3 correction_log.py "
                    f"--set-subdirs '{session_label}' RGB MASK1 MASK2")
            raise ValueError(f"session '{session_label}' rat {rat}: camera folders not found: "
                             f"{', '.join(missing[:3])}{' …' if len(missing) > 3 else ''}.{hint}")

        segments = []
        for r in group_rows:
            behavior = r["corrected_label"] if r["status"] == "corrected" else r["predicted_label"]
            segments.append({
                "behavior": behavior,
                "start_frame": r["window_start"],
                "end_frame": r["window_start"] + r["seq_len"],
            })
            exported_ids.append(r["id"])
        label_csvs[(session_label, rat)] = segments
        session_ids.add(session_label)

        label_csv_name = f"{session_label}_rat_{rat}.csv"
        manifest_row = {
            "rat_id": f"rat_{rat}",
            "object_id": rat,
            "experiment_name": session_label,
            "label_csv_path": label_csv_name,
        }
        manifest_row.update({k: str(v) for k, v in folders.items()})
        manifest_rows.append(manifest_row)

    return manifest_rows, label_csvs, sorted(session_ids), exported_ids


def write_export(out_dir, manifest_rows, label_csvs, session_ids):
    out_dir = Path(out_dir)
    label_dir = out_dir / "label_clean"
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "corrections_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest_rows)

    with open(out_dir / "corrections_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "split"])
        for sid in session_ids:
            w.writerow([sid, "train"])

    for (session_label, rat), segments in label_csvs.items():
        path = label_dir / f"{session_label}_rat_{rat}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["behavior", "start_frame", "end_frame"])
            w.writeheader()
            w.writerows(segments)


def merge_csv(base_path, extra_path, out_path):
    """Row-concatenate two CSVs that share the same header (base's header wins)."""
    with open(base_path, newline="") as f:
        base_rows = list(csv.DictReader(f))
        fieldnames = list(base_rows[0].keys()) if base_rows else list(csv.DictReader(open(base_path)).fieldnames)
    with open(extra_path, newline="") as f:
        extra_rows = list(csv.DictReader(f))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)
        w.writerows(extra_rows)


def resolve_base(args):
    """--base-* explicit or from the promoted cycle; the same chaining rule as retrain_cycle.py."""
    import retrain_cycle  # imported here: retrain_cycle imports this module at load time
    state = retrain_cycle.load_state()
    promoted = state["base"] or {}
    base, errors = {}, []
    for key in retrain_cycle.CHAINED_KEYS:
        given = getattr(args, key)
        value = str(Path(given).expanduser().resolve()) if given else promoted.get(key)
        if given and promoted.get(key) and value != promoted[key] and not args.ignore_promoted_base:
            errors.append(
                f"--{key.replace('_', '-')} {value}\n    differs from the base promoted from "
                f"{promoted.get('promoted_from')}:\n    {promoted[key]}\n    Using it would drop that "
                f"cycle's corrections. Leave it out to use the promoted base, or add "
                f"--ignore-promoted-base if this is intended."
            )
        if value and not (Path(value).is_dir() if key == "base_label_dir" else Path(value).is_file()):
            errors.append(f"--{key.replace('_', '-')}: not found: {value}")
        base[key] = value
    if bool(base["base_manifest"]) != bool(base["base_split_csv"]):
        errors.append("--base-manifest and --base-split-csv go together")
    return base, promoted, errors, retrain_cycle.pending_cycle(state), state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "corrections_export"))
    ap.add_argument("--base-manifest", default=None, help="default: the promoted cycle's merged manifest")
    ap.add_argument("--base-split-csv", default=None, help="default: the promoted cycle's merged split")
    ap.add_argument("--base-label-dir", default=None, help="default: the promoted cycle's merged label dir")
    ap.add_argument("--ignore-promoted-base", action="store_true",
                    help="allow --base-* files that differ from the promoted cycle's")
    ap.add_argument("--mark-exported", action="store_true",
                    help="mark the exported rows in the correction log, so no later export or "
                         "retraining cycle includes them again (only for hand-built retraining)")
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default
    args = ap.parse_args()

    base, promoted, errors, pending, state = resolve_base(args)
    if args.mark_exported and pending:
        errors.append(f"--mark-exported is refused while the retraining cycle in {state['pending']} awaits "
                      f"a decision (its rows would be marked before you promote or reject it). Decide it "
                      f"first with retrain_cycle.py --promote/--reject.")
    if errors:
        raise SystemExit("\n".join(errors))
    if args.dry_run:
        print("note: --dry-run is the default now; nothing is marked exported unless you pass --mark-exported")
    if promoted and not (args.base_manifest or args.base_split_csv or args.base_label_dir):
        print(f"using the base promoted from {promoted['promoted_from']}")

    try:
        manifest_rows, label_csvs, session_ids, exported_ids = build_export(args.db)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    if not manifest_rows:
        print("nothing to export — no confirmed/corrected rows pending export in the correction log.")
        return

    out_dir = Path(args.out_dir)
    write_export(out_dir, manifest_rows, label_csvs, session_ids)
    n_windows = sum(len(v) for v in label_csvs.values())
    print(f"wrote {len(manifest_rows)} manifest rows ({n_windows} corrected/confirmed windows, "
          f"{len(session_ids)} session(s)) to {out_dir}")

    if base["base_manifest"]:
        merge_csv(base["base_manifest"], out_dir / "corrections_manifest.csv", out_dir / "merged_manifest.csv")
        merge_csv(base["base_split_csv"], out_dir / "corrections_split.csv", out_dir / "merged_split.csv")
        print(f"merged with the base -> {out_dir / 'merged_manifest.csv'}, {out_dir / 'merged_split.csv'}")
    if base["base_label_dir"]:
        merged_labels = out_dir / "merged_label_clean"
        if merged_labels.exists():
            shutil.rmtree(merged_labels)
        shutil.copytree(base["base_label_dir"], merged_labels)
        for f in (out_dir / "label_clean").glob("*.csv"):
            shutil.copy(f, merged_labels / f.name)
        print(f"merged label dir -> {merged_labels} (pass it as 2_build_h5_dataset.py's --label-dir)")
    elif base["base_manifest"]:
        print(f"NOTE: label_csv_path values in merged_manifest.csv for the new rows are bare filenames "
              f"(e.g. '{session_ids[0]}_rat_1.csv') resolved against --label-dir at build time — pass "
              f"--base-label-dir to get a merged label dir, or merge the original label_clean/ with "
              f"{out_dir / 'label_clean'} yourself.")

    if args.mark_exported:
        mark_exported(args.db, exported_ids)
        print(f"marked {len(exported_ids)} correction-log rows as exported")
    else:
        print("correction log unchanged: these rows stay available to retrain_cycle.py "
              "(pass --mark-exported only for hand-built retraining you keep)")


if __name__ == "__main__":
    main()
