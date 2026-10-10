"""
qc_experiment_frames_v2.py
===========================
Frame-wise quality control over one or many experiments' raw camera folders.

MULTI-EXPERIMENT INPUT
-----------------------
--experiment_dir can now point at either:
  (a) a single experiment root, e.g. .../experimental_data/grooming_1
      (cam*/cutouts/... directly underneath) — same as before, or
  (b) a parent folder containing many experiments, e.g.
      .../experimental_data  (grooming_1/, RAT_familiar_.../, ... underneath)

Each immediate subdirectory of the given path that contains at least one
qualifying cutouts stream is treated as its own experiment: discovered,
scanned, and analyzed independently, with its own output subfolder
(<out_dir>/<experiment_name>/frames.csv etc.) and its own cross-camera
outlier check. Cameras are NEVER pooled across experiments — "cam1" in one
session and "cam1" in another are different recordings and must not be
compared to each other as if they were the same camera over time. If no
subdirectory qualifies, the given path itself is treated as a single
experiment (the original behaviour, e.g. if you point it straight at
grooming_1). After all experiments are processed, a consolidated rollup
(all_experiments_streams_summary.csv / all_experiments_suspicious_runs.csv)
is written at the top of --out_dir, with an "experiment" column added
throughout so it's always unambiguous which experiment a row came from.

WHY THE GREEN CHECK CHANGED
---------------------------
The original green check required (a) fg_fraction > 0.5 — i.e. rat pixels
covering over half the *entire frame* — before it even looked at colour, and
(b) the *mean* green channel across all foreground pixels to exceed red and
blue by 1.15x. That combination was built to catch a total green-screen
failure (a whole frame reading as solid green), and it does that. It cannot
catch a partial reflection halo around the rat's silhouette (e.g. off nearby
plexiglass): a real rat crop is typically only a few percent of the frame
(so the fg_fraction gate never opens), and even when it does, a halo
affecting ~7-10% of the rat's own pixels barely moves the *mean* colour of
all foreground pixels combined — nowhere near a 1.15x shift. Checked against
real cutouts, this is exactly what happened: a visibly present green fringe,
green_fraction near 0 reported.

THE FIX
-------
Green contamination is now measured per rat pixel, not per frame:
  green_fraction = (fraction of the rat's OWN foreground pixels that are
                     individually green-dominant, i.e. G channel exceeds
                     both R and B by --green_delta)
This is computed regardless of how much of the frame the rat covers, so a
thin edge halo on a small crop is visible in the number instead of being
diluted away by averaging.

Because some baseline level of this (e.g. a consistent plexiglass
reflection) may be a normal, expected feature of a given camera/session
rather than a one-off defect, green_fraction is NOT flagged against a fixed
absolute cutoff. It's tracked the same way fg_fraction already is: a
per-stream robust (median/MAD) baseline is established, and only frames
whose green_fraction deviates abnormally FROM THAT STREAM'S OWN BASELINE are
flagged (flag_green). The baseline itself (median_green_fraction, in
streams_summary.csv) is always reported, so you can see how much halo is
"normal" for a given camera even on frames that aren't flagged.

STREAM SCOPING: CUTOUTS ONLY, NEVER CROPPED
--------------------------------------------
Discovery only descends into folders whose path contains "cutouts", and
explicitly skips anything with "cropped" anywhere in the path — including a
"cropped" folder nested *inside* cutouts/ (e.g. cutouts/object_1_cropped,
which is exactly where 2_extract_crops_v2.py's default output lands). This
matches the standing rule for this pipeline: mask/data quality is controlled
at the cutouts level, not the crops level, so this script should never see
crop-script output, intentionally or by accident. Both are configurable via
--path_must_contain / --path_must_not_contain if your layout differs.

Everything else — missing-frame detection, suspicious-run collapsing, and
the cross-camera outlier check (now run per-experiment) — is unchanged from
the original script.

Usage
-----
    # one experiment
    python qc_experiment_frames_v2.py \\
        --experiment_dir ~/projects/data_master_thesis/experimental_data/grooming_1 \\
        --out_dir qc_out/grooming_1

    # every experiment under a parent folder, auto-detected
    python qc_experiment_frames_v2.py \\
        --experiment_dir ~/projects/data_master_thesis/experimental_data \\
        --out_dir qc_out

Outputs (written to --out_dir)
-------------------------------
  <out_dir>/<experiment_name>/frames.csv            one row per frame
  <out_dir>/<experiment_name>/streams_summary.csv    one row per (camera, stream)
  <out_dir>/<experiment_name>/cameras_summary.csv    one row per camera (this experiment only)
  <out_dir>/<experiment_name>/suspicious_runs.csv    consecutive flagged/missing runs
  <out_dir>/all_experiments_streams_summary.csv      rollup across every experiment scanned
  <out_dir>/all_experiments_suspicious_runs.csv       rollup across every experiment scanned
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
CAM_RE = re.compile(r"cam_?(\d+)", re.IGNORECASE)
DIGITS_RE = re.compile(r"\d+")


# ── Discovery ──────────────────────────────────────────────────────────────────

def _path_qualifies(rel_lower: str, path_must_contain, path_must_not_contain):
    if any(tok.lower() not in rel_lower for tok in path_must_contain):
        return False
    if any(tok.lower() in rel_lower for tok in path_must_not_contain):
        return False
    return True


def discover_streams(experiment_dir: Path, path_must_contain=("cutouts",),
                      path_must_not_contain=("cropped",)):
    """
    Recursively find every directory that directly contains >=1 supported
    image file, restricted to paths containing all of path_must_contain and
    none of path_must_not_contain (case-insensitive substring match against
    the path relative to experiment_dir). Defaults: must contain "cutouts",
    must not contain "cropped" — so cutouts/object_1 qualifies, but
    cutouts/object_1_cropped or a sibling .../cropped/object_1_cropped does
    not, even though both may exist side by side on disk.
    Each qualifying folder is treated as an independent "stream" to QC.
    Returns list of dicts: {camera, stream_path (rel to experiment_dir), files}
    """
    streams = []
    skipped_cropped = []
    for root, _dirs, files in os.walk(experiment_dir):
        imgs = sorted(f for f in files if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS)
        if not imgs:
            continue
        root_path = Path(root)
        rel = root_path.relative_to(experiment_dir)
        rel_lower = str(rel).lower()

        if any(tok.lower() not in rel_lower for tok in path_must_contain):
            continue
        if any(tok.lower() in rel_lower for tok in path_must_not_contain):
            skipped_cropped.append(str(rel))
            continue

        m = CAM_RE.search(str(rel))
        camera = f"cam{int(m.group(1))}" if m else "UNKNOWN"
        streams.append({
            "camera": camera,
            "stream_path": str(rel) if str(rel) != "." else "(root)",
            "dir": root_path,
            "files": imgs,
        })
    streams.sort(key=lambda s: (s["camera"], s["stream_path"]))
    if skipped_cropped:
        print(f"    Skipped {len(skipped_cropped)} folder(s) matching "
              f"{path_must_not_contain} (crop-script output, not in scope):")
        for p in skipped_cropped:
            print(f"        {p}")
    return streams


def _has_any_stream(d: Path, path_must_contain, path_must_not_contain):
    for root, _dirs, files in os.walk(d):
        if not any(Path(f).suffix.lower() in SUPPORTED_EXTENSIONS for f in files):
            continue
        rel_lower = str(Path(root).relative_to(d)).lower()
        if _path_qualifies(rel_lower, path_must_contain, path_must_not_contain):
            return True
    return False


def discover_experiments(root: Path, path_must_contain, path_must_not_contain):
    """
    Distinguishes "root is one experiment" from "root is a parent of many
    experiments" using the camera-folder naming convention (cam1, cam_2, ...):

      - If root's own immediate subdirectories look like camera folders
        (match CAM_RE), root itself IS a single experiment — its children
        are cameras, not sibling experiments. This is what keeps pointing
        straight at e.g. .../grooming_1 (whose children are cam1, cam2, ...)
        working unchanged: without this check, each camera folder would
        wrongly be treated as its own "experiment".
      - Otherwise, each immediate subdirectory that contains at least one
        qualifying cutouts stream is treated as its own experiment (the
        .../experimental_data case: children are grooming_1, RAT_..., etc.).
      - If neither applies (e.g. root has no subdirectories at all, or none
        qualify), root itself is treated as a single experiment as a
        last-resort fallback.
    Returns list of (experiment_name, experiment_path) tuples, sorted by name.
    """
    if not root.is_dir():
        return []
    subdirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)

    if any(CAM_RE.search(d.name) for d in subdirs):
        return [(root.name, root)]

    experiments = [(d.name, d) for d in subdirs
                   if _has_any_stream(d, path_must_contain, path_must_not_contain)]
    if experiments:
        return experiments
    return [(root.name, root)]


def parse_frame_number(stem: str):
    matches = DIGITS_RE.findall(stem)
    if not matches:
        return None
    return int(matches[-1])


# ── Per-frame metrics ───────────────────────────────────────────────────────────

def _err(error_type: str, error_msg: str):
    return {"error": True, "error_type": error_type, "error_msg": error_msg}


def compute_frame_metrics(path_str: str, bg_threshold: int, green_delta: int):
    """
    Reads one frame and computes its metrics. On any failure, returns a
    specific error_type + error_msg rather than a single generic label, so a
    zero-byte file, a permission problem, and a genuinely corrupt/truncated
    image are all distinguishable from frames.csv without opening the file
    by hand:

      missing_file      - path did not exist at read time
      permission_denied  - OS denied read access
      empty_file         - file exists but is 0 bytes
      unreadable_header   - file has bytes but couldn't even be opened for
                            reading (rare; OS-level I/O error)
      decode_failed       - file opened fine but cv2.imread could not decode
                            it as an image (corrupt/truncated encoding,
                            unsupported/wrong format, non-image content) —
                            includes the file size and first bytes (hex) of
                            the file to help identify which
      unexpected_shape     - decoded to something other than a 3-channel
                            image (e.g. grayscale, RGBA) — processing below
                            assumes BGR
      <ExceptionClassName> - any other Python/OpenCV/NumPy exception raised
                            while reading or processing the frame, tagged
                            with its exception class and message
    """
    p = Path(path_str)
    try:
        if not p.exists():
            return _err("missing_file", f"path does not exist: {path_str}")

        try:
            size = p.stat().st_size
        except PermissionError as e:
            return _err("permission_denied", f"stat() denied: {e}")
        except OSError as e:
            return _err("os_error", f"stat() failed: {e}")

        if size == 0:
            return _err("empty_file", "file exists but is 0 bytes")

        try:
            with open(path_str, "rb") as fh:
                header = fh.read(16)
        except PermissionError as e:
            return _err("permission_denied", f"open() denied: {e}")
        except OSError as e:
            return _err("unreadable_header", f"open()/read() failed: {e}")

        img = cv2.imread(path_str)
        if img is None:
            return _err(
                "decode_failed",
                f"cv2.imread returned None — file has {size} bytes, "
                f"header={header.hex()} (corrupt/truncated encoding, "
                f"unsupported format, or non-image content)",
            )

        if img.ndim != 3 or img.shape[2] != 3:
            return _err("unexpected_shape", f"decoded array has shape {img.shape}, expected (H, W, 3)")

        h, w = img.shape[:2]
        total_px = h * w
        mask = img.max(axis=2) > bg_threshold
        fg_px = int(mask.sum())
        fg_frac = fg_px / total_px if total_px else 0.0

        if fg_px > 0:
            fg_pixels = img[mask]  # BGR
            b, g, r = fg_pixels[:, 0].astype(np.int16), fg_pixels[:, 1].astype(np.int16), fg_pixels[:, 2].astype(np.int16)
            mean_b, mean_g, mean_r = float(b.mean()), float(g.mean()), float(r.mean())
            # per-pixel green dominance, measured only against the rat's own
            # pixels — not diluted by averaging, not gated on frame coverage.
            green_dominant = (g > r + green_delta) & (g > b + green_delta)
            green_fraction = float(green_dominant.sum()) / fg_px
        else:
            mean_b = mean_g = mean_r = float("nan")
            green_fraction = float("nan")

        return {
            "error": False,
            "error_type": "",
            "error_msg": "",
            "height": h,
            "width": w,
            "fg_pixels": fg_px,
            "bg_pixels": total_px - fg_px,
            "fg_fraction": fg_frac,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "green_fraction": green_fraction,
        }
    except Exception as e:
        # Catch-all so one bad file can never crash a worker process or take
        # down the whole scan — still reported with its real exception type
        # and message rather than silently skipped.
        return _err(type(e).__name__, str(e))


def _worker(args):
    path_str, bg_threshold, green_delta = args
    return path_str, compute_frame_metrics(path_str, bg_threshold, green_delta)


# ── Analysis ────────────────────────────────────────────────────────────────────

def robust_z(values: np.ndarray):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    robust_std = mad * 1.4826
    if robust_std < 1e-9:
        std = values.std()
        robust_std = std if std > 1e-9 else 1e-9
    return (values - median) / robust_std, median, robust_std


def find_runs(flagged_frame_numbers, step):
    """Collapse a sorted list of frame numbers into consecutive (step-spaced) runs."""
    if not flagged_frame_numbers:
        return []
    runs = []
    start = prev = flagged_frame_numbers[0]
    for fn in flagged_frame_numbers[1:]:
        if fn - prev == step:
            prev = fn
            continue
        runs.append((start, prev, (prev - start) // step + 1))
        start = prev = fn
    runs.append((start, prev, (prev - start) // step + 1))
    return sorted(runs, key=lambda r: r[2], reverse=True)


def infer_step(sorted_frame_numbers):
    diffs = [b - a for a, b in zip(sorted_frame_numbers, sorted_frame_numbers[1:]) if b > a]
    if not diffs:
        return 1
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(counts)])


def analyze_stream(records, z_thresh, min_fg, max_fg, green_z_thresh, min_fg_for_green):
    """
    records: list of dicts, one per frame, already containing compute_frame_metrics
              output plus filename/frame_number. Mutates in place adding flags,
              returns (stream_summary dict, list of suspicious_run tuples,
              list of missing_run tuples).
    """
    ok = [r for r in records if not r["error"]]
    fracs = np.array([r["fg_fraction"] for r in ok]) if ok else np.array([])

    if len(fracs) >= 2:
        z, median, robust_std = robust_z(fracs)
        for r, zi in zip(ok, z):
            r["deviation_z"] = float(zi)
    else:
        median = float(fracs[0]) if len(fracs) else float("nan")
        robust_std = float("nan")
        for r in ok:
            r["deviation_z"] = 0.0

    # green_fraction gets the same robust per-stream baseline treatment as
    # fg_fraction, instead of a fixed absolute cutoff. Frames with too few
    # foreground pixels to trust a ratio are excluded from the baseline
    # computation (but still get a green_deviation_z of 0 / not flagged,
    # rather than a noisy ratio driving a false positive).
    green_ok = [r for r in ok if r["fg_pixels"] >= min_fg_for_green and not np.isnan(r["green_fraction"])]
    green_vals = np.array([r["green_fraction"] for r in green_ok])
    if len(green_vals) >= 2:
        gz, g_median, g_robust_std = robust_z(green_vals)
        for r, zi in zip(green_ok, gz):
            r["green_deviation_z"] = float(zi)
    else:
        g_median = float(green_vals[0]) if len(green_vals) else float("nan")
        g_robust_std = float("nan")
        for r in green_ok:
            r["green_deviation_z"] = 0.0

    for r in records:
        if r["error"]:
            # Error frames are tracked as their own "read_error" run category
            # (below), not folded into flag_* / suspicious_content — so a
            # run of unreadable files is never confused with a run of
            # anomalous-but-readable ones.
            r["flag_empty"] = r["flag_overfull"] = r["flag_deviant"] = r["flag_green"] = False
            r["suspicious"] = True
            r.setdefault("green_deviation_z", float("nan"))
            continue
        r["flag_empty"] = r["fg_fraction"] < min_fg
        r["flag_overfull"] = r["fg_fraction"] > max_fg
        r["flag_deviant"] = abs(r["deviation_z"]) > z_thresh
        r.setdefault("green_deviation_z", float("nan"))
        gz = r["green_deviation_z"]
        r["flag_green"] = bool(not np.isnan(gz) and gz > green_z_thresh)
        r["suspicious"] = r["flag_empty"] or r["flag_overfull"] or r["flag_deviant"] or r["flag_green"]

    # frame-number-based gap / run analysis
    numbered = sorted([r for r in records if r["frame_number"] is not None],
                       key=lambda r: r["frame_number"])
    n_unparsed = len(records) - len(numbered)

    missing_runs = []
    duplicate_numbers = []
    step = 1
    if len(numbered) >= 2:
        seq = [r["frame_number"] for r in numbered]
        step = infer_step(seq)
        gaps = []
        for a, b in zip(seq, seq[1:]):
            if b == a:
                duplicate_numbers.append(a)
            elif b - a > step:
                missing_lo, missing_hi = a + step, b - step
                gaps.append((missing_lo, missing_hi, (missing_hi - missing_lo) // step + 1))
        missing_runs = sorted(gaps, key=lambda g: g[2], reverse=True)

    # "suspicious_content" runs = flagged, readable frames only. Read errors
    # get their own run category (error_runs) so the two are never conflated
    # in suspicious_runs.csv.
    content_flagged_numbers = sorted(
        r["frame_number"] for r in numbered
        if not r["error"] and (r["flag_empty"] or r["flag_overfull"] or r["flag_deviant"] or r["flag_green"])
    )
    suspicious_runs = find_runs(content_flagged_numbers, step) if content_flagged_numbers else []

    error_frame_numbers = sorted(r["frame_number"] for r in numbered if r["error"])
    error_runs = find_runs(error_frame_numbers, step) if error_frame_numbers else []

    error_type_counts = defaultdict(int)
    for r in records:
        if r["error"]:
            error_type_counts[r.get("error_type") or "unknown"] += 1
    error_type_breakdown = "; ".join(f"{k}:{v}" for k, v in sorted(error_type_counts.items(), key=lambda kv: -kv[1]))

    n_errors = sum(1 for r in records if r["error"])
    summary = {
        "n_frames": len(records),
        "n_errors": n_errors,
        "error_type_breakdown": error_type_breakdown,
        "n_unparsed_frame_numbers": n_unparsed,
        "n_suspicious": len(content_flagged_numbers),
        "n_flag_empty": sum(1 for r in records if r.get("flag_empty")),
        "n_flag_overfull": sum(1 for r in records if r.get("flag_overfull")),
        "n_flag_deviant": sum(1 for r in records if r.get("flag_deviant")),
        "n_flag_green": sum(1 for r in records if r.get("flag_green")),
        "median_fg_fraction": median,
        "robust_std_fg_fraction": robust_std,
        "median_green_fraction": g_median,
        "robust_std_green_fraction": g_robust_std,
        "inferred_frame_step": step,
        "n_duplicate_frame_numbers": len(duplicate_numbers),
        "n_missing_frames": sum(g[2] for g in missing_runs),
        "n_missing_runs": len(missing_runs),
        "longest_missing_run": missing_runs[0] if missing_runs else None,
        "n_suspicious_runs": len(suspicious_runs),
        "longest_suspicious_run": suspicious_runs[0] if suspicious_runs else None,
    }
    return summary, suspicious_runs, missing_runs, error_runs


# ── Per-experiment pipeline ──────────────────────────────────────────────────────

FRAME_FIELDS = [
    "experiment", "camera", "stream_path", "filename", "frame_number",
    "error", "error_type", "error_msg", "height", "width",
    "fg_pixels", "bg_pixels", "fg_fraction",
    "mean_r", "mean_g", "mean_b", "deviation_z",
    "green_fraction", "green_deviation_z",
    "flag_empty", "flag_overfull", "flag_deviant", "flag_green", "suspicious",
]
STREAM_FIELDS = [
    "experiment", "camera", "stream_path", "n_frames", "n_errors", "error_type_breakdown",
    "n_unparsed_frame_numbers",
    "n_suspicious", "n_flag_empty", "n_flag_overfull", "n_flag_deviant", "n_flag_green",
    "median_fg_fraction", "robust_std_fg_fraction",
    "median_green_fraction", "robust_std_green_fraction",
    "inferred_frame_step",
    "n_duplicate_frame_numbers", "n_missing_frames", "n_missing_runs", "longest_missing_run",
    "n_suspicious_runs", "longest_suspicious_run",
]
RUNS_FIELDS = ["experiment", "camera", "stream_path", "kind", "start_frame", "end_frame", "n_frames"]
CAMERA_FIELDS = ["experiment", "camera", "overall_median_fg_fraction", "z_vs_other_cameras", "outlier_vs_other_cameras"]


def run_qc_on_experiment(experiment_name: str, experiment_dir: Path, out_dir: Path, args,
                          must_contain, must_not_contain):
    """Runs the full scan/analysis/write pipeline for one experiment. Returns
    (stream_summaries, suspicious_runs_rows, camera_rows) for the rollup."""
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {experiment_name}")
    print(f"{'#'*70}")
    print(f"Path         : {experiment_dir}")
    print(f"Output       : {out_dir}")
    print("Discovering image streams (folders that directly contain images)...")
    streams = discover_streams(experiment_dir, must_contain, must_not_contain)
    if not streams:
        print("  No image folders found matching the path filter.")
        return [], [], []

    for s in streams:
        print(f"    {s['camera']:<8} {s['stream_path']:<40} {len(s['files']):>6} files")

    total_files = sum(len(s["files"]) for s in streams)
    print(f"\n  Total frames to process: {total_files:,}  (workers={args.workers})\n")

    jobs = []
    for s in streams:
        for f in s["files"]:
            jobs.append((str(s["dir"] / f), args.bg_threshold, args.green_delta))

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            path_str, metrics = fut.result()
            results[path_str] = metrics
            done += 1
            if done % 2000 == 0 or done == total_files:
                print(f"    processed {done:>7,}/{total_files:,}")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_writer_rows = []
    stream_summaries = []
    suspicious_runs_rows = []
    camera_fg_medians = defaultdict(list)

    for s in streams:
        records = []
        for f in s["files"]:
            path_str = str(s["dir"] / f)
            metrics = results[path_str]
            rec = {
                "experiment": experiment_name,
                "camera": s["camera"],
                "stream_path": s["stream_path"],
                "filename": f,
                "frame_number": parse_frame_number(Path(f).stem),
            }
            rec.update(metrics)
            records.append(rec)

        summary, suspicious_runs, missing_runs, error_runs = analyze_stream(
            records, args.z_thresh, args.min_fg, args.max_fg,
            args.green_z_thresh, args.min_fg_for_green)
        summary["experiment"] = experiment_name
        summary["camera"] = s["camera"]
        summary["stream_path"] = s["stream_path"]
        stream_summaries.append(summary)

        if not np.isnan(summary["median_fg_fraction"]):
            camera_fg_medians[s["camera"]].append(summary["median_fg_fraction"])

        for lo, hi, length in suspicious_runs:
            suspicious_runs_rows.append({
                "experiment": experiment_name, "camera": s["camera"], "stream_path": s["stream_path"],
                "kind": "suspicious_content", "start_frame": lo, "end_frame": hi, "n_frames": length,
            })
        for lo, hi, length in missing_runs:
            suspicious_runs_rows.append({
                "experiment": experiment_name, "camera": s["camera"], "stream_path": s["stream_path"],
                "kind": "missing_frames", "start_frame": lo, "end_frame": hi, "n_frames": length,
            })
        for lo, hi, length in error_runs:
            suspicious_runs_rows.append({
                "experiment": experiment_name, "camera": s["camera"], "stream_path": s["stream_path"],
                "kind": "read_error", "start_frame": lo, "end_frame": hi, "n_frames": length,
            })

        for rec in records:
            frames_writer_rows.append({k: rec.get(k, "") for k in FRAME_FIELDS})

    with open(out_dir / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        w.writeheader()
        w.writerows(frames_writer_rows)

    with open(out_dir / "streams_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STREAM_FIELDS)
        w.writeheader()
        w.writerows(stream_summaries)

    suspicious_runs_rows.sort(key=lambda r: r["n_frames"], reverse=True)
    with open(out_dir / "suspicious_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        w.writeheader()
        w.writerows(suspicious_runs_rows)

    # cross-camera outlier check — scoped to THIS experiment only, cameras
    # are never pooled across experiments.
    cams = sorted(camera_fg_medians)
    cam_overall_median = {c: float(np.mean(camera_fg_medians[c])) for c in cams}
    camera_rows = []
    if len(cams) >= 2:
        vals = np.array([cam_overall_median[c] for c in cams])
        z, med, rstd = robust_z(vals)
        for c, zi in zip(cams, z):
            camera_rows.append({
                "experiment": experiment_name, "camera": c,
                "overall_median_fg_fraction": cam_overall_median[c],
                "z_vs_other_cameras": float(zi),
                "outlier_vs_other_cameras": bool(abs(zi) > args.camera_outlier_z),
            })
    else:
        for c in cams:
            camera_rows.append({
                "experiment": experiment_name, "camera": c,
                "overall_median_fg_fraction": cam_overall_median[c],
                "z_vs_other_cameras": 0.0, "outlier_vs_other_cameras": False,
            })
    with open(out_dir / "cameras_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMERA_FIELDS)
        w.writeheader()
        w.writerows(camera_rows)

    print(f"\n  {'-'*66}")
    print("  SUMMARY")
    print(f"  {'-'*66}")
    for s in stream_summaries:
        flag_bits = []
        if s["n_errors"]: flag_bits.append(f"errors={s['n_errors']} ({s['error_type_breakdown']})")
        if s["n_flag_empty"]: flag_bits.append(f"empty={s['n_flag_empty']}")
        if s["n_flag_overfull"]: flag_bits.append(f"overfull={s['n_flag_overfull']}")
        if s["n_flag_deviant"]: flag_bits.append(f"deviant={s['n_flag_deviant']}")
        if s["n_flag_green"]: flag_bits.append(f"green={s['n_flag_green']}")
        if s["n_missing_frames"]: flag_bits.append(f"missing={s['n_missing_frames']}")
        flags_str = ", ".join(flag_bits) if flag_bits else "clean"
        g_med = s.get("median_green_fraction")
        g_med_str = f"  (baseline green_fraction={g_med:.3f})" if g_med is not None and not np.isnan(g_med) else ""
        print(f"    {s['camera']:<8} {s['stream_path']:<40} n={s['n_frames']:>6}  {flags_str}{g_med_str}")

    outliers = [r for r in camera_rows if r["outlier_vs_other_cameras"]]
    if outliers:
        print(f"\n  [ALERT] camera(s) statistically inconsistent with the rest of THIS experiment:")
        for r in outliers:
            print(f"      {r['camera']}: median fg-fraction {r['overall_median_fg_fraction']:.3f} "
                  f"(z={r['z_vs_other_cameras']:.1f} vs other cameras)")

    top_runs = suspicious_runs_rows[:10]
    if top_runs:
        print(f"\n  Longest suspicious/missing runs in this experiment (see suspicious_runs.csv for all):")
        for r in top_runs:
            print(f"      {r['camera']:<8} {r['stream_path']:<30} {r['kind']:<18} "
                  f"frames {r['start_frame']}-{r['end_frame']} ({r['n_frames']} frames)")

    print(f"\n  Wrote:")
    print(f"    {out_dir / 'frames.csv'}")
    print(f"    {out_dir / 'streams_summary.csv'}")
    print(f"    {out_dir / 'cameras_summary.csv'}")
    print(f"    {out_dir / 'suspicious_runs.csv'}")

    return stream_summaries, suspicious_runs_rows, camera_rows


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment_dir", required=True,
                    help="A single experiment's root, OR a parent folder containing many experiment "
                         "folders — auto-detected (see module docstring).")
    ap.add_argument("--out_dir", default=None, help="Default: qc_out/<experiment_dir folder name>")
    ap.add_argument("--bg_threshold", type=int, default=8,
                    help="Max channel value treated as background (matches 2_extract_crops.py). Default 8.")
    ap.add_argument("--z_thresh", type=float, default=4.0,
                    help="Robust z-score above which a frame's fg-fraction is 'deviant' vs its own stream. Default 4.0.")
    ap.add_argument("--min_fg", type=float, default=0.002,
                    help="fg_fraction below this = essentially empty (no rat detected). Default 0.002.")
    ap.add_argument("--max_fg", type=float, default=0.95,
                    help="fg_fraction above this = suspiciously full frame (background-subtraction failure candidate). Default 0.95.")
    ap.add_argument("--green_delta", type=int, default=20,
                    help="A rat pixel counts as green-dominant if its G channel exceeds both R and B "
                         "by at least this much. Default 20.")
    ap.add_argument("--green_z_thresh", type=float, default=4.0,
                    help="Robust z-score above which a frame's green_fraction (share of the rat's OWN "
                         "pixels that are green-dominant) is flagged vs its own stream's baseline. "
                         "This baseline can be nonzero (e.g. a consistent reflection) without being "
                         "flagged — only abnormal deviation from it is. Default 4.0.")
    ap.add_argument("--min_fg_for_green", type=int, default=50,
                    help="Minimum foreground pixel count before green_fraction is trusted enough to "
                         "contribute to the stream's baseline or be flagged. Default 50.")
    ap.add_argument("--camera_outlier_z", type=float, default=3.0,
                    help="Robust z-score above which a camera's overall median fg_fraction is flagged vs sibling cameras in the same experiment. Default 3.0.")
    ap.add_argument("--path_must_contain", type=str, default="cutouts",
                    help="Comma-separated substring(s) a folder's path must ALL contain to be scanned "
                         "(case-insensitive). Default 'cutouts' — cropped/model-input folders are never "
                         "cutouts-level data and are out of scope for this script.")
    ap.add_argument("--path_must_not_contain", type=str, default="cropped",
                    help="Comma-separated substring(s) that exclude a folder if ANY are present in its "
                         "path (case-insensitive), even if --path_must_contain also matches — this is what "
                         "keeps a nested cutouts/object_1_cropped folder out of the scan. Default 'cropped'.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    must_contain = tuple(t.strip() for t in args.path_must_contain.split(",") if t.strip())
    must_not_contain = tuple(t.strip() for t in args.path_must_not_contain.split(",") if t.strip())

    input_dir = Path(args.experiment_dir).resolve()
    if not input_dir.is_dir():
        print(f"ERROR: not a directory: {input_dir}")
        sys.exit(1)

    out_root = Path(args.out_dir).resolve() if args.out_dir else Path("qc_out") / input_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Input        : {input_dir}")
    print(f"Output root  : {out_root}")
    print(f"Path filter  : must contain {must_contain}, must not contain {must_not_contain}")
    print("Detecting experiments (immediate subfolders containing qualifying cutouts streams)...")
    experiments = discover_experiments(input_dir, must_contain, must_not_contain)

    if len(experiments) == 1 and experiments[0][1] == input_dir:
        print(f"  No qualifying sub-experiments found under the input path — "
              f"treating the input path itself as a single experiment.")
    else:
        print(f"  Found {len(experiments)} experiment(s): {', '.join(n for n, _ in experiments)}")

    all_stream_summaries = []
    all_runs = []
    all_camera_rows = []

    for name, path in experiments:
        summaries, runs, cam_rows = run_qc_on_experiment(
            name, path, out_root / name, args, must_contain, must_not_contain)
        all_stream_summaries.extend(summaries)
        all_runs.extend(runs)
        all_camera_rows.extend(cam_rows)

    if len(experiments) > 1:
        with open(out_root / "all_experiments_streams_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STREAM_FIELDS)
            w.writeheader()
            w.writerows(all_stream_summaries)
        all_runs_sorted = sorted(all_runs, key=lambda r: r["n_frames"], reverse=True)
        with open(out_root / "all_experiments_suspicious_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
            w.writeheader()
            w.writerows(all_runs_sorted)
        with open(out_root / "all_experiments_cameras_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CAMERA_FIELDS)
            w.writeheader()
            w.writerows(all_camera_rows)

        print(f"\n{'='*70}")
        print(f"Scanned {len(experiments)} experiments. Rollups written:")
        print(f"  {out_root / 'all_experiments_streams_summary.csv'}")
        print(f"  {out_root / 'all_experiments_suspicious_runs.csv'}")
        print(f"  {out_root / 'all_experiments_cameras_summary.csv'}")


if __name__ == "__main__":
    main()