#!/usr/bin/env python3
"""
prepare_dataset.py

Turn the reviewed SAM3 archive into a training-ready dataset for a promptless,
full-resolution, class-agnostic rat segmenter.

INPUT LAYOUT (as verified on the real archive)
----------------------------------------------
Raw videos (model INPUTS - the original, un-masked footage), decoded with decord:
    <raw_root>/<experiment>/<experiment>_cam{1..6}.mp4
    <raw_root>/empty_arena/cam{1..6}.mp4          (reference only, no masks)

Masks (model TARGETS - grayscale/palette PNG, values 0=bg, 1=rat1, 2=rat2):
    <mask_root>/<experiment>/cam_{1..6}/masks_raw/frame_000000.png ...

Note the two naming conventions that MUST be mapped:
    mask side : cam_K   (underscore)   video side : <exp>_camK.mp4  (no underscore)
    empty arena video   : camK.mp4     (no experiment prefix)

Video frame index i corresponds to mask frame_00000i (confirmed: matching frame
counts and resolution per camera). Mask indices are NOT assumed contiguous - we
key on the actual mask files present.

(The cutouts/ folder is NOT used for training - the background is already
blacked out there, so it carries no shadow/occlusion signal. Training inputs
MUST be the original video frames.)

WHAT THIS PRODUCES
------------------
Per (experiment, camera):
  - one median background image  -> <out_root>/backgrounds/<exp>__cam<K>.png
Global:
  - manifest.jsonl  : one row per labelled frame, with paths, camera, session,
                      split, and QA flags
  - dataset_stats.json
  - qa_report.jsonl : frames flagged as suspicious targets

The foreground prior and the flow-warped previous-mask channel are computed
on-the-fly in the dataloader from (frame, background); precomputing them for
~5x10^5 frames would waste disk. compute_foreground_prior() lives here so the
dataloader and this script share one implementation.

SPLIT
-----
By SESSION (experiment), never by frame, and all six cameras of a session go to
the same split (they are simultaneous views -> frame-level or camera-level
splits leak). Long sessions may additionally be chunked temporally with a gap
buffer so they contribute to both splits without leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from decord import VideoReader, cpu

FRAME_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
CAM_DIR_RE = re.compile(r"cam_?(\d+)$", re.IGNORECASE)   # matches 'cam_1' and 'cam1'
EMPTY_ARENA_NAME = "empty_arena"
MASK_SUBDIR = "masks_raw"


# ---------------------------------------------------------------------------
# Palette mask handling
# ---------------------------------------------------------------------------

def load_label_mask(path: str) -> np.ndarray:
    """Read a palette label PNG back as raw ids (0/1/2), NOT RGB colours."""
    return np.array(cv2.imread(path, cv2.IMREAD_UNCHANGED) if False else _pil_label(path))


def _pil_label(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path))


def split_palette_mask(label: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split a 0/1/2 label map into boolean masks.
      obj1  : label == 1
      obj2  : label == 2
      anyrat: label >= 1   (class-agnostic 'rat' target)
    """
    label = np.asarray(label)
    if label.ndim == 3:  # safety: a palette file read as RGB
        label = label[..., 0]
    return {
        "obj1": label == 1,
        "obj2": label == 2,
        "anyrat": label >= 1,
    }


# ---------------------------------------------------------------------------
# Video access (decord) + resolving the video path for a mask camera
# ---------------------------------------------------------------------------

def video_path_for(raw_root: Path, experiment: str, cam: int) -> Optional[Path]:
    """
    Map a mask-side (experiment, cam) to its source video.
    Experiments: <raw_root>/<exp>/<exp>_camK.mp4
    Empty arena: <raw_root>/empty_arena/camK.mp4
    Tolerates a couple of alternative spellings just in case.
    """
    exp_dir = raw_root / experiment
    candidates = [
        exp_dir / f"{experiment}_cam{cam}.mp4",
        exp_dir / f"cam{cam}.mp4",
        exp_dir / f"cam_{cam}.mp4",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def open_video(path: str) -> VideoReader:
    return VideoReader(str(path), ctx=cpu(0))


def read_frame_rgb(vr: VideoReader, idx: int) -> Optional[np.ndarray]:
    """decord is frame-exact. Returns HxWx3 uint8 RGB, or None if out of range."""
    if idx < 0 or idx >= len(vr):
        return None
    return vr[idx].asnumpy()


def compute_background_from_video(
    video_path: str,
    max_samples: int = 150,
    band_rows: int = 300,
) -> np.ndarray:
    """
    Per-pixel temporal median over an evenly-strided sample of decoded frames.
    Median (not mean) so a lingering rat does not smear in. Banded to cap
    memory at band_rows x W x 3 x n_samples. Returns BGR uint8 (cv2 convention)
    so it round-trips through cv2.imwrite/imread cleanly.
    """
    vr = open_video(video_path)
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    sample_idx = (
        np.linspace(0, n - 1, min(max_samples, n)).round().astype(int)
    )
    sample_idx = sorted(set(int(i) for i in sample_idx))

    # decord batch-decodes efficiently; fetch the sample once as RGB
    frames_rgb = vr.get_batch(sample_idx).asnumpy()  # (S, H, W, 3) RGB
    S, H, W, _ = frames_rgb.shape

    bg_rgb = np.empty((H, W, 3), dtype=np.uint8)
    for y0 in range(0, H, band_rows):
        y1 = min(y0 + band_rows, H)
        band = frames_rgb[:, y0:y1]                     # (S, bh, W, 3)
        bg_rgb[y0:y1] = np.median(band, axis=0).astype(np.uint8)

    return cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Shadow-suppressed foreground prior  (shared with the dataloader)
# ---------------------------------------------------------------------------

def compute_foreground_prior(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray,
    diff_thresh: float = 0.10,
    shadow_v_lo: float = 0.35,
    shadow_v_hi: float = 0.98,
    shadow_sat_diff: float = 0.12,
    shadow_hue_diff: float = 0.10,
) -> np.ndarray:
    """
    Appearance-independent foreground probability in [0,1], with cast-shadow
    suppression. Returns float32 HxW.

    Idea: a cast shadow keeps the background's HUE and (roughly) its
    SATURATION but drops VALUE. A real rat changes hue/saturation too. So a
    pixel that only got darker (value ratio in a shadow band) while hue and
    saturation stayed close to the background is classified as shadow, not
    foreground.
    """
    f = frame_bgr.astype(np.float32) / 255.0
    b = background_bgr.astype(np.float32) / 255.0

    # magnitude of colour change
    diff = np.linalg.norm(f - b, axis=2) / np.sqrt(3.0)
    fg = diff > diff_thresh

    fhsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    bhsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    fh, fs, fv = fhsv[..., 0] / 179.0, fhsv[..., 1] / 255.0, fhsv[..., 2] / 255.0
    bh, bs, bv = bhsv[..., 0] / 179.0, bhsv[..., 1] / 255.0, bhsv[..., 2] / 255.0

    v_ratio = np.divide(fv, np.maximum(bv, 1e-3))
    hue_dist = np.minimum(np.abs(fh - bh), 1.0 - np.abs(fh - bh))  # circular
    sat_dist = np.abs(fs - bs)

    is_shadow = (
        (v_ratio > shadow_v_lo)
        & (v_ratio < shadow_v_hi)
        & (hue_dist < shadow_hue_diff)
        & (sat_dist < shadow_sat_diff)
    )

    prob = diff.copy()
    prob[is_shadow] = 0.0
    prob[~fg] = 0.0
    # squash to [0,1] with a soft gain around the threshold
    prob = np.clip((prob - diff_thresh) / (1.0 - diff_thresh), 0.0, 1.0)
    return prob.astype(np.float32)


# ---------------------------------------------------------------------------
# QA on target masks
# ---------------------------------------------------------------------------

@dataclass
class QAThresholds:
    min_area: int = 150            # a rat smaller than this is suspicious
    max_area_frac: float = 0.30    # a rat larger than 30% of frame is suspicious
    max_components: int = 4        # many disconnected blobs -> suspicious
    max_area_jump: float = 3.0     # >3x area change vs neighbour -> suspicious


def _num_components(mask: np.ndarray) -> int:
    m = mask.astype(np.uint8)
    n, _ = cv2.connectedComponents(m, connectivity=8)
    return max(0, n - 1)  # minus background


def qa_mask_frame(
    label: np.ndarray,
    thr: QAThresholds,
    prev_areas: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    parts = split_palette_mask(label)
    h, w = label.shape[:2]
    frame_px = h * w
    flags: List[str] = []
    areas: Dict[str, int] = {}

    for key in ("obj1", "obj2"):
        m = parts[key]
        area = int(m.sum())
        areas[key] = area
        if area == 0:
            flags.append(f"{key}_empty")
            continue
        if area < thr.min_area:
            flags.append(f"{key}_tiny")
        if area > thr.max_area_frac * frame_px:
            flags.append(f"{key}_huge")
        if _num_components(m) > thr.max_components:
            flags.append(f"{key}_fragmented")
        if prev_areas and prev_areas.get(key, 0) > 0 and area > 0:
            r = max(area, prev_areas[key]) / max(1, min(area, prev_areas[key]))
            if r > thr.max_area_jump:
                flags.append(f"{key}_area_jump")

    return {"areas": areas, "flags": flags}


# ---------------------------------------------------------------------------
# Scanning the archive
# ---------------------------------------------------------------------------

@dataclass
class FrameRow:
    experiment: str
    cam: int
    frame_idx: int          # index into the source video AND the mask name
    video_path: str         # source .mp4; dataloader reads frame_idx via decord
    mask_path: str
    background_path: str
    split: str = "train"
    qa_flags: List[str] = field(default_factory=list)
    obj1_area: int = 0
    obj2_area: int = 0


def _list_mask_frames(folder: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        m = FRAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def scan_archive(raw_root: str, mask_root: str) -> Tuple[Dict[Tuple[str, int], Dict], List[str]]:
    """
    Walk the MASK tree (it defines what is labelled) and pair each
    (experiment, cam) with its source video in the RAW tree.

    Returns:
        recordings: { (experiment, cam): {"masks": {idx: path}, "video": Path} }
        warnings:   human-readable skip notices
    """
    raw_root = Path(raw_root).expanduser().resolve()
    mask_root = Path(mask_root).expanduser().resolve()

    recordings: Dict[Tuple[str, int], Dict] = {}
    warnings: List[str] = []

    for exp_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        exp = exp_dir.name
        if exp == EMPTY_ARENA_NAME:
            continue  # reference only; handled separately

        for cam_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            cm = CAM_DIR_RE.search(cam_dir.name)
            if not cm:
                continue
            cam = int(cm.group(1))

            mask_dir = cam_dir / MASK_SUBDIR
            masks = _list_mask_frames(mask_dir)
            if not masks:
                warnings.append(f"SKIP {exp}/{cam_dir.name}: no '{MASK_SUBDIR}/' with frame_*.png")
                continue

            video = video_path_for(raw_root, exp, cam)
            if video is None:
                warnings.append(f"SKIP {exp}/{cam_dir.name}: no source video found under {raw_root/exp}")
                continue

            recordings[(exp, cam)] = {"masks": masks, "video": video}

    return recordings, warnings


# ---------------------------------------------------------------------------
# Split: by session (experiment), long sessions chunked with a gap buffer
# ---------------------------------------------------------------------------

def assign_splits(
    per_exp_frame_indices: Dict[str, List[int]],
    val_experiments: List[str],
    long_session_min_frames: int,
    long_val_tail_frac: float,
    gap_frames: int,
) -> Dict[Tuple[str, int], str]:
    """
    Returns {(experiment, frame_idx) -> 'train'|'val'} applied identically
    across all cameras of a session.

    - Experiments named in val_experiments go entirely to val.
    - Any experiment with >= long_session_min_frames also contributes a
      temporal TAIL (last long_val_tail_frac of its frames) to val, separated
      from the train head by gap_frames dropped frames to prevent leakage.
    """
    split: Dict[Tuple[str, int], str] = {}

    for exp, idxs in per_exp_frame_indices.items():
        idxs = sorted(idxs)
        if exp in val_experiments:
            for i in idxs:
                split[(exp, i)] = "val"
            continue

        n = len(idxs)
        if n >= long_session_min_frames and long_val_tail_frac > 0:
            n_val = int(round(n * long_val_tail_frac))
            val_start = n - n_val
            gap_start = max(0, val_start - gap_frames)
            for j, i in enumerate(idxs):
                if j >= val_start:
                    split[(exp, i)] = "val"
                elif j >= gap_start:
                    split[(exp, i)] = "gap"   # dropped: buffer between splits
                else:
                    split[(exp, i)] = "train"
        else:
            for i in idxs:
                split[(exp, i)] = "train"

    return split


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_dataset(
    raw_root: str,
    mask_root: str,
    out_root: str,
    val_experiments: Optional[List[str]] = None,
    test_experiments: Optional[List[str]] = None,
    exclude_experiments: Optional[List[str]] = None,
    long_session_min_frames: int = 5000,
    long_val_tail_frac: float = 0.15,
    gap_frames: int = 300,
    bg_max_samples: int = 150,
    run_qa: bool = True,
    drop_qa_flagged_from_train: bool = False,
    n_align_overlays: int = 2,
) -> Dict:
    out_root = Path(out_root).expanduser().resolve()
    bg_dir = out_root / "backgrounds"
    align_dir = out_root / "alignment_overlays"
    bg_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    raw_root_p = Path(raw_root).expanduser().resolve()
    val_experiments = list(val_experiments or [])
    test_experiments = list(test_experiments or [])
    exclude_experiments = set(exclude_experiments or [])

    print(f"Scanning archive...\n  raw:  {raw_root}\n  mask: {mask_root}")
    recordings, warnings = scan_archive(raw_root, mask_root)
    for w in warnings:
        print("  " + w)
    if not recordings:
        raise RuntimeError("No (experiment, cam) recordings with masks found.")

    # drop excluded experiments entirely
    if exclude_experiments:
        recordings = {k: v for k, v in recordings.items() if k[0] not in exclude_experiments}

    experiments = sorted({exp for (exp, _cam) in recordings})
    cams = sorted({cam for (_exp, cam) in recordings})
    print(f"Found {len(experiments)} experiments, cameras {cams}, {len(recordings)} recordings")

    # frame indices present (labelled) per experiment, unioned across cameras
    per_exp_idx: Dict[str, set] = {e: set() for e in experiments}
    for (exp, cam), rec in recordings.items():
        per_exp_idx[exp].update(rec["masks"].keys())
    per_exp_idx_list = {e: sorted(v) for e, v in per_exp_idx.items()}

    # test experiments are held out wholesale (never train/val); implement by
    # marking every one of their frames 'test' after the normal split.
    split_map = assign_splits(
        per_exp_idx_list,
        val_experiments=val_experiments,
        long_session_min_frames=long_session_min_frames,
        long_val_tail_frac=long_val_tail_frac,
        gap_frames=gap_frames,
    )
    for exp in test_experiments:
        for i in per_exp_idx_list.get(exp, []):
            split_map[(exp, i)] = "test"

    # ---- empty-arena per-camera reference backgrounds (optional) ----
    empty_refs: Dict[int, Path] = {}
    empty_dir = raw_root_p / EMPTY_ARENA_NAME
    if empty_dir.is_dir():
        print("Building empty-arena reference backgrounds...")
        for cam in cams:
            vp = video_path_for(raw_root_p, EMPTY_ARENA_NAME, cam)
            if vp is None:
                continue
            ref_path = bg_dir / f"_empty_arena__cam{cam}.png"
            if not ref_path.exists():
                cv2.imwrite(str(ref_path), compute_background_from_video(str(vp), bg_max_samples))
            empty_refs[cam] = ref_path
            print(f"  empty_arena cam{cam} -> {ref_path.name}")

    thr = QAThresholds()
    rows: List[FrameRow] = []
    qa_report: List[Dict] = []
    bg_divergence: Dict[str, float] = {}

    for (exp, cam), rec in sorted(recordings.items()):
        masks = rec["masks"]
        video = str(rec["video"])

        # per-session background straight from the source video
        bg_path = bg_dir / f"{exp}__cam{cam}.png"
        if not bg_path.exists():
            bg = compute_background_from_video(video, max_samples=bg_max_samples)
            cv2.imwrite(str(bg_path), bg)
        else:
            bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)

        # divergence vs empty-arena reference (a broken/contaminated bg flag)
        if cam in empty_refs:
            ref = cv2.imread(str(empty_refs[cam]), cv2.IMREAD_COLOR)
            if ref is not None and ref.shape == bg.shape:
                div = float(np.abs(bg.astype(np.int16) - ref.astype(np.int16)).mean())
                bg_divergence[f"{exp}__cam{cam}"] = round(div, 2)
                if div > 25.0:
                    qa_report.append({"experiment": exp, "cam": cam, "frame_idx": -1,
                                      "flags": [f"bg_diverges_from_empty_arena_{div:.1f}"]})

        # alignment overlays: sample a few labelled frames, overlay mask on the
        # decoded video frame, write to disk for the user to eyeball
        vr = open_video(video)
        sorted_idx = sorted(masks)
        pick = np.linspace(0, len(sorted_idx) - 1, min(n_align_overlays, len(sorted_idx)))
        for j in {int(round(x)) for x in pick}:
            idx = sorted_idx[j]
            fr = read_frame_rgb(vr, idx)
            if fr is None:
                qa_report.append({"experiment": exp, "cam": cam, "frame_idx": idx,
                                  "flags": ["video_frame_out_of_range"]})
                continue
            lab = _pil_label(str(masks[idx]))
            ov = fr.copy()
            if lab.shape[:2] != ov.shape[:2]:
                qa_report.append({"experiment": exp, "cam": cam, "frame_idx": idx,
                                  "flags": [f"resolution_mismatch_mask{lab.shape[:2]}_vs_video{ov.shape[:2]}"]})
                continue
            ov[lab == 1] = [255, 0, 0]
            ov[lab == 2] = [0, 0, 255]
            cv2.imwrite(str(align_dir / f"{exp}__cam{cam}__f{idx:06d}.png"),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

        print(f"  {exp} cam{cam}: {len(masks)} labelled frames  video_frames={len(vr)}"
              f"  bg -> {bg_path.name}")

        prev_areas: Optional[Dict[str, int]] = None
        n_video = len(vr)
        for idx in sorted_idx:
            if idx >= n_video:
                qa_report.append({"experiment": exp, "cam": cam, "frame_idx": idx,
                                  "flags": ["mask_index_beyond_video"]})
                continue

            split = split_map.get((exp, idx), "train")
            if split == "gap":
                continue

            row = FrameRow(
                experiment=exp, cam=cam, frame_idx=idx,
                video_path=video,
                mask_path=str(masks[idx]),
                background_path=str(bg_path),
                split=split,
            )

            if run_qa:
                label = _pil_label(str(masks[idx]))
                qa = qa_mask_frame(label, thr, prev_areas=prev_areas)
                row.qa_flags = qa["flags"]
                row.obj1_area = qa["areas"].get("obj1", 0)
                row.obj2_area = qa["areas"].get("obj2", 0)
                prev_areas = qa["areas"]
                if qa["flags"]:
                    qa_report.append({"experiment": exp, "cam": cam,
                                      "frame_idx": idx, "flags": qa["flags"]})
                    if drop_qa_flagged_from_train and split == "train":
                        continue

            rows.append(row)

    # ---- write manifest + stats ----
    manifest_path = out_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    qa_path = out_root / "qa_report.jsonl"
    with qa_path.open("w", encoding="utf-8") as f:
        for q in qa_report:
            f.write(json.dumps(q) + "\n")

    n_train = sum(1 for r in rows if r.split == "train")
    n_val = sum(1 for r in rows if r.split == "val")
    n_test = sum(1 for r in rows if r.split == "test")
    stats = {
        "experiments": experiments,
        "cameras": cams,
        "recordings": len(recordings),
        "labelled_frames_total": len(rows),
        "train_frames": n_train,
        "val_frames": n_val,
        "test_frames": n_test,
        "val_experiments_heldout": val_experiments,
        "test_experiments_heldout": test_experiments,
        "excluded_experiments": sorted(exclude_experiments),
        "skipped_recordings": warnings,
        "qa_flagged": len(qa_report),
        "bg_divergence_vs_empty_arena": bg_divergence,
        "per_experiment_frame_counts": {e: len(v) for e, v in per_exp_idx_list.items()},
        "manifest": str(manifest_path),
        "backgrounds_dir": str(bg_dir),
        "alignment_overlays_dir": str(align_dir),
        "qa_report": str(qa_path),
    }
    with (out_root / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== DATASET SUMMARY ===")
    print(f"Labelled frames: {len(rows)}  (train={n_train}, val={n_val}, test={n_test})")
    print(f"QA-flagged:      {len(qa_report)}")
    print(f"Manifest:        {manifest_path}")
    print(f"Backgrounds:     {bg_dir}")
    print(f"Align overlays:  {align_dir}   <-- eyeball a few to confirm mask sits on the rats")
    print(f"Stats:           {out_root / 'dataset_stats.json'}")
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a training dataset from the reviewed SAM3 archive.")
    ap.add_argument("--raw_root", required=True,
                    help="root of source videos, e.g. ~/projects/data_master_thesis/raw_data")
    ap.add_argument("--mask_root", required=True,
                    help="root of masks, e.g. ~/projects/data_master_thesis/experimental_data")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--val_experiments", nargs="*", default=[],
                    help="experiment names held out entirely for validation")
    ap.add_argument("--test_experiments", nargs="*", default=[],
                    help="experiment names held out entirely for test (never train/val)")
    ap.add_argument("--exclude_experiments", nargs="*", default=[],
                    help="experiment names to drop from the dataset completely")
    ap.add_argument("--long_session_min_frames", type=int, default=5000)
    ap.add_argument("--long_val_tail_frac", type=float, default=0.15)
    ap.add_argument("--gap_frames", type=int, default=300)
    ap.add_argument("--bg_max_samples", type=int, default=150)
    ap.add_argument("--n_align_overlays", type=int, default=2,
                    help="per camera, how many mask-on-video overlays to write for eyeballing")
    ap.add_argument("--no_qa", action="store_true")
    ap.add_argument("--drop_qa_flagged_from_train", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        raw_root=args.raw_root,
        mask_root=args.mask_root,
        out_root=args.out_root,
        val_experiments=args.val_experiments,
        test_experiments=args.test_experiments,
        exclude_experiments=args.exclude_experiments,
        long_session_min_frames=args.long_session_min_frames,
        long_val_tail_frac=args.long_val_tail_frac,
        gap_frames=args.gap_frames,
        bg_max_samples=args.bg_max_samples,
        run_qa=not args.no_qa,
        drop_qa_flagged_from_train=args.drop_qa_flagged_from_train,
        n_align_overlays=args.n_align_overlays,
    )


if __name__ == "__main__":
    main()