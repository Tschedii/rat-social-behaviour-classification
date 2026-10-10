#!/usr/bin/env python3
"""
Convert the remote training_data/ tree (per manifest CSV) + local label_clean/
segment CSVs into a single HDF5 dataset for training a video transformer.

Must be run somewhere that can see the actual frame files -- i.e. on
<user>@<gpu-server> itself (after copying label_clean/ + this script there),
or against training_data/ mounted locally (e.g. via sshfs) with label_clean/
available on the same machine.

Layout of the produced .h5:

    /                                   (root attrs: fps, classes, cams, storage, created_with_args)
    /classes                            string dataset, sorted unique behavior names
    /index/segment_id                   string dataset, one row per segment
    /index/experiment                   string dataset
    /index/rat_id                       string dataset ("rat_1"/"rat_2")
    /index/object_id                    int dataset (1 or 2)
    /index/behavior                     string dataset
    /index/behavior_id                  int dataset (index into /classes)
    /index/start_frame                  int dataset (original bout start, from label_clean)
    /index/end_frame                    int dataset (original bout end, from label_clean)
    /index/num_frames                   int dataset -- TARGET length (== --frames-per-segment when
                                         capped). This is NOT necessarily the true stored length --
                                         see index/min_true_frames and /data/.../frame_id below.
    /index/min_true_frames              int dataset -- min, across this segment's cams, of the ACTUAL
                                         number of frames written (len of frame_id). Equals
                                         num_frames except for the handful of bouts that start/end
                                         right at the edge of what's actually on disk (e.g. sessions
                                         trimmed at the front -- see conversion_report.json
                                         clipped_segments). Use this to filter/flag those cheaply
                                         without opening every cam group.
    /index/split                        string dataset ("train"/"eval")
    /index/is_padded                    bool dataset (True if the bout was shorter than
                                         --frames-per-segment, i.e. contains repeated/held frames)
    /data/<segment_id>/cam_<n>/rgb      see --storage below, length == len(frame_id) for this cam
    /data/<segment_id>/cam_<n>/mask     see --storage below, same length as rgb
    /data/<segment_id>/cam_<n>/frame_id int64 (T,) -- the EXACT original frame number (matching the
                                         source frame_XXXXXX.png filename, and the `key` column of
                                         training_metadata/<experiment>/cam_<n>/transform_meta.csv)
                                         for every stored frame, in order. This is the join key back
                                         to transform_meta.csv or any other per-frame metadata.
                                         Repeated values indicate a held/padded frame (short bout).
                                         T can differ from index/num_frames (see above) and can
                                         differ ACROSS CAMS for the same segment when cameras don't
                                         start/stop in perfect sync -- always trust len(frame_id)
                                         over index/num_frames for a specific cam's array shape.

<segment_id> = "{experiment}__{rat_id}__{row_index_in_label_csv:05d}"

--storage png (default): each frame's ORIGINAL PNG BYTES are stored verbatim
in a variable-length uint8 dataset of shape (T,) -- i.e. no decode, no
re-encode, no resize. This is lossless and keeps the exact same
resolution/quality as the source, and is what keeps the .h5 close to the
size of the source PNGs on disk (source PNG compression is far better than
generic gzip on raw decoded arrays -- ~10x smaller in practice). Decode with
`cv2.imdecode(np.frombuffer(h5file['data/.../rgb'][t], np.uint8), cv2.IMREAD_COLOR)`
at training/dataloader time (see decode_example() below). --resize is not
compatible with this mode (resizing requires decoding first) -- use
--storage array if you need resizing.

--storage array: frames are decoded to raw uint8 arrays (T,H,W,3)/(T,H,W),
optionally resized via --resize, and gzip-compressed by h5py. Much larger on
disk than --storage png for the same resolution, but gives you plain numpy
arrays with no decode step needed at load time.

Frame ranges are [start_frame, end_frame) exactly like the label CSVs
(end_frame = round(end_seconds * fps)). If a segment extends beyond the
frames actually present on disk (some exports are trimmed, e.g. starting at
frame 105 rather than 0), the clip is silently clamped to the available
range and the clipping is logged in the conversion report -- never
zero-padded, since that would fabricate frames. The exact frame numbers that
survive clipping are what gets written to frame_id, so this is always
faithfully recorded even for those edge-case segments.

"point"-type label rows (single timestamp, no end_frame) are skipped by
default -- there is no clip to extract. Use --point-window N to instead
export a fixed N-frame window centered on the point event.

Usage:
    python build_h5_dataset.py \
        --manifest training_manifest.csv \
        --data-root /path/to/master_thesis_social_behaviour \
        --label-dir label_clean \
        --split-csv label_clean/train_eval_split.csv \
        --cams 1,2,3,4,5,6 \
        --frames-per-segment 32 \
        --normstats rgb_normstats.json \
        --out rat_behaviour.h5

    # decoding a stored frame later, and joining it to transform_meta.csv:
    import h5py, cv2, numpy as np, csv
    f = h5py.File('rat_behaviour.h5')
    seg = 'grooming_1__rat_1__00000'
    raw = f[f'data/{seg}/cam_1/rgb'][3]                 # raw PNG bytes, clip-local index 3
    frame_no = int(f[f'data/{seg}/cam_1/frame_id'][3])  # original frame number, e.g. 281
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # frame_no now matches the `key` column (as "frame_%06d" % frame_no) in
    # training_metadata/<experiment>/cam_1/transform_meta.csv for a bbox/scale/presence lookup.
"""
import argparse
import csv
import json
import os
import re
import sys

import h5py
import numpy as np

from frame_io import FrameDir, fixed_length_indices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True,
                   help="Root the manifest's relative training_data/... paths are joined against")
    p.add_argument("--label-dir", required=True,
                   help="Local folder of cleaned per-segment label CSVs (label_clean/)")
    p.add_argument("--split-csv", required=True,
                   help="train_eval_split.csv (used only to build session_id -> split lookup)")
    p.add_argument("--cams", default="1", help="Comma list of camera indices to include, e.g. 1,2,3,4,5,6")
    p.add_argument("--storage", default="png", choices=["png", "array"],
                   help="'png' (default): store original PNG bytes verbatim, lossless, full "
                        "resolution, ~10x smaller than 'array' on disk. 'array': decode to raw "
                        "uint8 arrays (required if you also want --resize). See module docstring.")
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"),
                   help="Resize frames to WxH. Only valid with --storage array (resizing requires "
                        "decoding). Not needed/used with --storage png.")
    p.add_argument("--frames-per-segment", type=int, default=16,
                   help="Resample every bout to exactly this many frames (standard fixed-T input "
                        "for video transformers, and required for plain tensor stacking/batching). "
                        "Bouts longer than this are uniformly sparse-sampled; bouts shorter than "
                        "this (common here -- ~28%% of bouts are already <=16 frames) have frames "
                        "repeated/held to fill the length, never zero-padded. See index/is_padded "
                        "in the output to identify which segments were repeat-padded, and "
                        "index/min_true_frames + /data/.../frame_id for the actual per-cam length. "
                        "Pass 0 to keep every real frame with no resampling (variable length per "
                        "segment -- you'll need a custom collate_fn to batch these).")
    p.add_argument("--point-window", type=int, default=0,
                   help="If >0, export point-type (instantaneous) label rows too, as a window of "
                        "this many frames centered on the event. 0 = skip point rows.")
    p.add_argument("--normstats", default=None,
                   help="Optional rgb_normstats.json from compute_rgb_normstats.py -- stored as "
                        "h5 root attrs for the dataloader to use, frames themselves stay raw uint8.")
    p.add_argument("--out", default="rat_behaviour.h5")
    p.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    p.add_argument("--report", default="conversion_report.json")
    return p.parse_args()


def session_of(label_filename):
    base = re.sub(r"\.csv$", "", label_filename)
    base = re.sub(r"_precise$", "", base)
    base = re.sub(r"_rat_[12]$", "", base)
    return base


def load_split_map(split_csv):
    m = {}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            m[row["session_id"]] = row["split"]
    return m


def load_segments(label_path, point_window):
    """Returns list of (row_idx, start_frame, end_frame, behavior)."""
    segs = []
    with open(label_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            behavior = (row.get("behavior") or "").strip()
            if not behavior:
                continue
            start = row.get("start_frame", "").strip()
            end = row.get("end_frame", "").strip()
            if start == "":
                continue
            start = int(start)
            if end == "":
                if point_window <= 0:
                    continue
                half = point_window // 2
                segs.append((i, start - half, start + (point_window - half), behavior))
            else:
                segs.append((i, start, int(end), behavior))
    return segs


def sample_avg_frame_bytes(plan, cams, data_root, n_dirs=6, n_frames=15):
    """Empirically sample real on-disk file sizes to estimate --storage png output size
    (measuring file size only, no decode -- cheap even for many dirs)."""
    import random
    seen_pairs = {}
    for seg in plan:
        row = seg["row"]
        for cam in cams:
            rgb_dir = os.path.join(data_root, row[f"cam{cam}_rgb_path"])
            mask_dir = os.path.join(data_root, row[f"cam{cam}_mask_path"])
            seen_pairs.setdefault((rgb_dir, mask_dir), None)
        if len(seen_pairs) >= n_dirs:
            break

    rgb_sizes, mask_sizes = [], []
    for rgb_dir, mask_dir in seen_pairs:
        try:
            rgb_fd = FrameDir(rgb_dir)
            mask_fd = FrameDir(mask_dir)
        except FileNotFoundError:
            continue
        lo, hi = rgb_fd.available_range()
        candidates = [f for f in range(lo, hi + 1) if rgb_fd.has(f) and mask_fd.has(f)]
        if not candidates:
            continue
        sample = random.sample(candidates, min(n_frames, len(candidates)))
        for fn in sample:
            rgb_sizes.append(rgb_fd.file_size(fn))
            mask_sizes.append(mask_fd.file_size(fn))

    if not rgb_sizes:
        return None  # couldn't reach any real files (e.g. dry-run without data-root access)
    return sum(rgb_sizes) / len(rgb_sizes), sum(mask_sizes) / len(mask_sizes)


def estimate_bytes_array(n_segments, avg_frames, n_cams, h, w):
    per_frame = h * w * 3 + h * w  # rgb + mask, uint8, decoded raw
    return n_segments * avg_frames * n_cams * per_frame


def main():
    args = parse_args()
    cams = [int(c) for c in args.cams.split(",")]
    compression = None if args.compression == "none" else args.compression

    split_map = load_split_map(args.split_csv)

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    print(f"Manifest: {len(rows)} (rat, experiment) entries.")

    # Pass 1: gather every segment + resolve its clip length, so we can print
    # a size estimate before touching the (potentially huge) frame files.
    plan = []  # list of dicts, one per segment
    behaviors = set()
    missing_label_files = []

    for row in rows:
        rat_id = row["rat_id"].strip().replace(" ", "_").lower()  # "Rat 1" -> "rat_1"
        object_id = int(row["object_id"])
        experiment = row["experiment_name"].strip()
        label_fname = os.path.basename(row["label_csv_path"])
        label_path = os.path.join(args.label_dir, label_fname)

        if not os.path.exists(label_path):
            missing_label_files.append(label_path)
            continue

        session_id = session_of(label_fname)
        split = split_map.get(session_id, "unassigned")

        segs = load_segments(label_path, args.point_window)
        for row_idx, start, end, behavior in segs:
            behaviors.add(behavior)
            plan.append({
                "row": row, "rat_id": rat_id, "object_id": object_id,
                "experiment": experiment, "session_id": session_id, "split": split,
                "row_idx": row_idx, "start": start, "end": end, "behavior": behavior,
            })

    if missing_label_files:
        print(f"WARNING: {len(missing_label_files)} label CSVs referenced by the manifest were not "
              f"found under --label-dir (first few: {missing_label_files[:3]})")

    classes = sorted(behaviors)
    class_to_id = {b: i for i, b in enumerate(classes)}
    print(f"{len(plan)} segments to export, {len(classes)} classes: {classes}")

    if args.storage == "png" and args.resize:
        print("ERROR: --resize requires --storage array (resizing needs a decode step; "
              "'png' mode stores original bytes verbatim).", file=sys.stderr)
        sys.exit(1)

    if args.frames_per_segment > 0:
        avg_frames = args.frames_per_segment
        total_frame_slots = len(plan) * args.frames_per_segment
    else:
        per_seg_frames = [min(s["end"] - s["start"], 10_000) for s in plan]
        avg_frames = sum(per_seg_frames) / max(len(plan), 1)
        total_frame_slots = sum(per_seg_frames)

    if args.storage == "png":
        sampled = sample_avg_frame_bytes(plan, cams, args.data_root)
        if sampled is None:
            print("Could not sample real files under --data-root to estimate size "
                  "(check the path is reachable) -- skipping size estimate.")
        else:
            avg_rgb_b, avg_mask_b = sampled
            est_bytes = total_frame_slots * len(cams) * (avg_rgb_b + avg_mask_b)
            print(f"Measured avg on-disk size: rgb={avg_rgb_b:,.0f}B mask={avg_mask_b:,.0f}B per frame")
            print(f"Estimated .h5 size (lossless PNG passthrough): {est_bytes / 1e9:.2f} GB "
                  f"({len(plan)} segments x ~{avg_frames:.0f} frames x {len(cams)} cams)")
    else:
        h, w = (args.resize[1], args.resize[0]) if args.resize else (1024, 1024)
        est_bytes = estimate_bytes_array(len(plan), avg_frames, len(cams), h, w)
        print(f"Rough uncompressed size estimate (--storage array, before gzip): {est_bytes / 1e9:.2f} GB "
              f"({len(plan)} segments x ~{avg_frames:.0f} frames x {len(cams)} cams x {h}x{w})")
        if est_bytes > 200e9:
            print("This is very large. Consider --resize, a smaller --frames-per-segment, "
                  "fewer --cams, or --storage png instead.")

    normstats_attr = None
    if args.normstats:
        with open(args.normstats) as f:
            normstats_attr = json.dumps(json.load(f))

    report = {"clipped_segments": [], "empty_segments": [], "read_errors": []}
    frame_dir_cache = {}

    def get_frame_dir(path):
        if path not in frame_dir_cache:
            frame_dir_cache[path] = FrameDir(path)
        return frame_dir_cache[path]

    resize_wh = tuple(args.resize) if args.resize else None
    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(args.out, "w") as h5:
        h5.attrs["fps"] = 30
        h5.attrs["classes"] = classes
        h5.attrs["cams"] = cams
        h5.attrs["storage"] = args.storage
        h5.attrs["created_with_args"] = json.dumps(vars(args))
        if normstats_attr:
            h5.attrs["rgb_normstats"] = normstats_attr

        idx_segment_id, idx_experiment, idx_rat, idx_object = [], [], [], []
        idx_behavior, idx_behavior_id, idx_start, idx_end, idx_numframes, idx_split = [], [], [], [], [], []
        idx_padded, idx_min_true_frames = [], []

        data_grp = h5.create_group("data")

        for n, seg in enumerate(plan):
            row = seg["row"]
            seg_id = f"{seg['experiment']}__{seg['rat_id']}__{seg['row_idx']:05d}"

            true_length = seg["end"] - seg["start"]
            if args.frames_per_segment > 0:
                wanted_frames = fixed_length_indices(seg["start"], seg["end"], args.frames_per_segment)
                is_padded = true_length < args.frames_per_segment
            else:
                wanted_frames = list(range(seg["start"], seg["end"]))
                is_padded = False

            if not wanted_frames:
                report["empty_segments"].append(seg_id)
                continue

            seg_grp = data_grp.create_group(seg_id)
            wrote_any_cam = False
            true_lengths_this_seg = []

            for cam in cams:
                rgb_path = os.path.join(args.data_root, row[f"cam{cam}_rgb_path"])
                mask_path = os.path.join(args.data_root, row[f"cam{cam}_mask_path"])
                try:
                    rgb_fd = get_frame_dir(rgb_path)
                    mask_fd = get_frame_dir(mask_path)
                except FileNotFoundError as e:
                    report["read_errors"].append({"segment": seg_id, "cam": cam, "error": str(e)})
                    continue

                lo, hi = rgb_fd.available_range()
                clipped = [f for f in wanted_frames if lo <= f <= hi]
                if len(clipped) < len(wanted_frames):
                    report["clipped_segments"].append({
                        "segment": seg_id, "cam": cam,
                        "requested": [wanted_frames[0], wanted_frames[-1]],
                        "available": [lo, hi],
                        "kept": len(clipped), "dropped": len(wanted_frames) - len(clipped),
                    })
                if not clipped:
                    continue

                rgb_frames, mask_frames, frame_ids = [], [], []
                read_cache = {}  # frame_no -> (rgb_payload, mask_payload); repeat-padded short
                                 # bouts re-request the same fn, so cache the read within this cam/segment
                for fn in clipped:
                    if fn not in read_cache:
                        if not rgb_fd.has(fn) or not mask_fd.has(fn):
                            continue
                        if args.storage == "png":
                            rgb_payload = np.frombuffer(rgb_fd.read_bytes(fn), dtype=np.uint8)
                            mask_payload = np.frombuffer(mask_fd.read_bytes(fn), dtype=np.uint8)
                        else:
                            rgb_payload = rgb_fd.read(fn, color=True)
                            mask_payload = mask_fd.read(fn, color=False)
                            if resize_wh:
                                import cv2
                                rgb_payload = cv2.resize(rgb_payload, resize_wh, interpolation=cv2.INTER_AREA)
                                mask_payload = cv2.resize(mask_payload, resize_wh, interpolation=cv2.INTER_NEAREST)
                        read_cache[fn] = (rgb_payload, mask_payload)
                    if fn not in read_cache:
                        continue
                    rgb_payload, mask_payload = read_cache[fn]
                    rgb_frames.append(rgb_payload)
                    mask_frames.append(mask_payload)
                    frame_ids.append(fn)

                if not rgb_frames:
                    continue

                cam_grp = seg_grp.create_group(f"cam_{cam}")
                if args.storage == "png":
                    # vlen byte arrays: original PNG bytes, verbatim -- no compression filter
                    # (they're already compressed; gzip-on-top wastes CPU for ~0 extra savings).
                    rgb_ds = cam_grp.create_dataset("rgb", (len(rgb_frames),), dtype=vlen_u8)
                    mask_ds = cam_grp.create_dataset("mask", (len(mask_frames),), dtype=vlen_u8)
                    for t, payload in enumerate(rgb_frames):
                        rgb_ds[t] = payload
                    for t, payload in enumerate(mask_frames):
                        mask_ds[t] = payload
                else:
                    cam_grp.create_dataset("rgb", data=np.stack(rgb_frames), compression=compression)
                    cam_grp.create_dataset("mask", data=np.stack(mask_frames), compression=compression)
                cam_grp.create_dataset("frame_id", data=np.array(frame_ids, dtype=np.int64))
                true_lengths_this_seg.append(len(frame_ids))
                wrote_any_cam = True

            if not wrote_any_cam:
                del data_grp[seg_id]
                report["empty_segments"].append(seg_id)
                continue

            idx_segment_id.append(seg_id)
            idx_experiment.append(seg["experiment"])
            idx_rat.append(seg["rat_id"])
            idx_object.append(seg["object_id"])
            idx_behavior.append(seg["behavior"])
            idx_behavior_id.append(class_to_id[seg["behavior"]])
            idx_start.append(seg["start"])
            idx_end.append(seg["end"])
            idx_numframes.append(len(wanted_frames))
            idx_split.append(seg["split"])
            idx_padded.append(is_padded)
            idx_min_true_frames.append(min(true_lengths_this_seg))

            if (n + 1) % 50 == 0 or n + 1 == len(plan):
                print(f"  [{n + 1}/{len(plan)}] {seg_id}")

        str_dt = h5py.string_dtype(encoding="utf-8")
        idx_grp = h5.create_group("index")
        idx_grp.create_dataset("segment_id", data=np.array(idx_segment_id, dtype=object), dtype=str_dt)
        idx_grp.create_dataset("experiment", data=np.array(idx_experiment, dtype=object), dtype=str_dt)
        idx_grp.create_dataset("rat_id", data=np.array(idx_rat, dtype=object), dtype=str_dt)
        idx_grp.create_dataset("object_id", data=np.array(idx_object, dtype=np.int32))
        idx_grp.create_dataset("behavior", data=np.array(idx_behavior, dtype=object), dtype=str_dt)
        idx_grp.create_dataset("behavior_id", data=np.array(idx_behavior_id, dtype=np.int32))
        idx_grp.create_dataset("start_frame", data=np.array(idx_start, dtype=np.int64))
        idx_grp.create_dataset("end_frame", data=np.array(idx_end, dtype=np.int64))
        idx_grp.create_dataset("num_frames", data=np.array(idx_numframes, dtype=np.int32))
        idx_grp.create_dataset("split", data=np.array(idx_split, dtype=object), dtype=str_dt)
        idx_grp.create_dataset("is_padded", data=np.array(idx_padded, dtype=bool))
        idx_grp.create_dataset("min_true_frames", data=np.array(idx_min_true_frames, dtype=np.int32))
        h5.create_dataset("classes", data=np.array(classes, dtype=object), dtype=str_dt)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {args.out}")
    print(f"{len(idx_segment_id)} segments written, "
          f"{len(report['empty_segments'])} empty/skipped, "
          f"{len(report['clipped_segments'])} clipped, "
          f"{len(report['read_errors'])} read errors.")
    print(f"Full report: {args.report}")


if __name__ == "__main__":
    main()