#!/usr/bin/env python3
"""
Compute per-channel RGB normalization stats (mean/std in [0,1] range) over
the camera views listed in training_manifest.csv.

Must be run somewhere that can see the actual frame files -- i.e. on
<user>@<gpu-server> itself, or against training_data/ mounted locally
(e.g. via sshfs). It cannot be run from a machine that only has the
manifest CSV.

Only computes over unique rgb directories: rat_1 and rat_2 rows for the same
experiment/camera point at the identical rgb folder (both rats are visible
in the same frame), so we dedupe by path first to avoid double-counting and
skewing the stats.

Usage:
    python compute_rgb_normstats.py \
        --manifest training_manifest.csv \
        --data-root /path/to/master_thesis_social_behaviour \
        --cams 1,2,3,4,5,6 \
        --samples-per-video 40 \
        --out rgb_normstats.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

from frame_io import FrameDir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="Path to training_manifest.csv")
    p.add_argument("--data-root", required=True,
                   help="Root dir the manifest's relative paths are joined against "
                        "(e.g. /path/to/master_thesis_social_behaviour)")
    p.add_argument("--cams", default="1,2,3,4,5,6", help="Comma list of camera indices to include")
    p.add_argument("--samples-per-video", type=int, default=40,
                   help="Frames sampled per unique rgb folder (uniform random, seeded)")
    p.add_argument("--out", default="rgb_normstats.json")
    p.add_argument("--force", action="store_true", help="Recompute even if --out already exists")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collect_unique_rgb_dirs(manifest_path, data_root, cams):
    seen = {}  # abs_path -> cam_index (first one wins, just for reporting)
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            for cam in cams:
                col = f"cam{cam}_rgb_path"
                rel = row.get(col, "").strip()
                if not rel:
                    continue
                abs_path = os.path.join(data_root, rel)
                seen.setdefault(abs_path, cam)
    return seen


def main():
    args = parse_args()
    cams = [int(c) for c in args.cams.split(",")]

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} already exists -- skipping (pass --force to recompute).")
        return

    rgb_dirs = collect_unique_rgb_dirs(args.manifest, args.data_root, cams)
    print(f"Found {len(rgb_dirs)} unique rgb folders across cams {cams}.")

    rng = np.random.default_rng(args.seed)

    # running sums in float64, per-channel, in [0,1] scale
    global_sum = np.zeros(3, dtype=np.float64)
    global_sumsq = np.zeros(3, dtype=np.float64)
    global_pixels = 0
    global_frames = 0

    per_cam = {c: {"sum": np.zeros(3), "sumsq": np.zeros(3), "pixels": 0, "frames": 0} for c in cams}

    failed = []
    for i, (path, cam) in enumerate(rgb_dirs.items(), 1):
        try:
            fd = FrameDir(path)
        except FileNotFoundError as e:
            failed.append((path, str(e)))
            continue

        frame_nos = fd.sample_frames(args.samples_per_video, rng)
        for fn in frame_nos:
            img = fd.read(fn, color=True).astype(np.float64) / 255.0  # (H,W,3) RGB in [0,1]
            px = img.shape[0] * img.shape[1]
            s = img.sum(axis=(0, 1))
            sq = (img ** 2).sum(axis=(0, 1))

            global_sum += s
            global_sumsq += sq
            global_pixels += px
            global_frames += 1

            per_cam[cam]["sum"] += s
            per_cam[cam]["sumsq"] += sq
            per_cam[cam]["pixels"] += px
            per_cam[cam]["frames"] += 1

        if i % 10 == 0 or i == len(rgb_dirs):
            print(f"  [{i}/{len(rgb_dirs)}] {path} -> sampled {len(frame_nos)} frames")

    if global_pixels == 0:
        print("No frames were read successfully -- check --data-root and manifest paths.", file=sys.stderr)
        sys.exit(1)

    mean = global_sum / global_pixels
    var = global_sumsq / global_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    result = {
        "scale": "[0,1] (divide raw uint8 pixel by 255 before applying mean/std)",
        "channel_order": "RGB",
        "global": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n_frames_sampled": global_frames,
            "n_pixels": int(global_pixels),
        },
        "per_camera": {},
        "failed_dirs": failed,
    }

    for c in cams:
        pc = per_cam[c]
        if pc["pixels"] == 0:
            continue
        m = pc["sum"] / pc["pixels"]
        v = pc["sumsq"] / pc["pixels"] - m ** 2
        s = np.sqrt(np.maximum(v, 0))
        result["per_camera"][f"cam_{c}"] = {
            "mean": m.tolist(), "std": s.tolist(),
            "n_frames_sampled": pc["frames"], "n_pixels": int(pc["pixels"]),
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nGlobal RGB mean (0-1 scale): {mean}")
    print(f"Global RGB std  (0-1 scale): {std}")
    if failed:
        print(f"\n{len(failed)} rgb dirs could not be read (see '{args.out}' -> failed_dirs).")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()