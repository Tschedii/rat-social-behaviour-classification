#!/usr/bin/env python3
"""
fragment_video.py

Frame-accurate video fragmentation.
Splits a video into fixed-length segments with an exact frame overlap.

This avoids time-based rounding issues from MoviePy and guarantees that:
    total_segment_frames - overlaps == original_frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Separation scorer: uses the standalone promptless segmenter (best.pt) to
# decide, for a candidate frame, whether the two rats are SEPARATED (two
# distinct foreground blobs) or TOUCHING (one merged blob). Cutting segment
# boundaries where the rats are separated means every re-seed/handoff starts
# from an unambiguous two-rat state, which is where the touching-rats merge
# was coming from. Class-agnostic model => it can't tell rat1 from rat2, but
# it can tell "one blob" from "two blobs", which is exactly the cut signal.
# ---------------------------------------------------------------------------
class SeparationScorer:
    def __init__(
        self,
        ckpt_path: str,
        background_bgr: np.ndarray,
        device: str = "cuda:0",
        train_h: int = 640,
        train_w: int = 1024,
        thresh: float = 0.5,
        min_blob_area: int = 400,
    ):
        import os as _os
        import sys as _sys
        import torch

        _ckpt_dir = _os.path.dirname(_os.path.abspath(ckpt_path))
        if _ckpt_dir and _ckpt_dir not in _sys.path:
            _sys.path.insert(0, _ckpt_dir)

        from train_segmenter import ResNet34UNet, IMAGENET_MEAN, IMAGENET_STD
        from prepare_dataset import compute_foreground_prior

        self._torch = torch
        self._compute_prior = compute_foreground_prior
        self._mean = IMAGENET_MEAN
        self._std = IMAGENET_STD
        self.device = device if (torch.cuda.is_available() or "cpu" in device) else "cpu"
        self.train_h = train_h
        self.train_w = train_w
        self.thresh = thresh
        self.min_blob_area = min_blob_area
        self.background_bgr = background_bgr

        self.model = ResNet34UNet(in_ch=4, pretrained=False).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        print(f"[separation] loaded {ckpt_path} on {self.device}")

    def _union_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        H0, W0 = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        bg = self.background_bgr
        if bg.shape[:2] != (H0, W0):
            bg = cv2.resize(bg, (W0, H0), interpolation=cv2.INTER_AREA)
        prior_full = self._compute_prior(frame_bgr, bg)

        rgb = cv2.resize(frame_rgb, (self.train_w, self.train_h),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        pr = cv2.resize(prior_full, (self.train_w, self.train_h),
                        interpolation=cv2.INTER_AREA)
        x = np.concatenate([rgb, pr[..., None]], axis=2).transpose(2, 0, 1)[None]
        xt = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            logit = self.model(xt)
            prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()
        # keep detection at train res (separation doesn't need full res -> fast)
        return (prob >= self.thresh)

    def score(self, frame_rgb: np.ndarray) -> Tuple[bool, float, int]:
        """
        Returns (separated, gap, n_blobs):
          separated : True if >=2 distinct foreground blobs each >= min_blob_area
          gap       : distance between the two largest blobs' centroids (px at
                      detection res); larger = more clearly separated. 0 if <2 blobs.
          n_blobs   : number of qualifying blobs.
        """
        union = self._union_mask(frame_rgb).astype(np.uint8)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(union, 8)
        blobs = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.min_blob_area]
        blobs.sort(key=lambda k: stats[k, cv2.CC_STAT_AREA], reverse=True)
        if len(blobs) < 2:
            return False, 0.0, len(blobs)
        c0 = cent[blobs[0]]
        c1 = cent[blobs[1]]
        gap = float(np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
        # also require the two blobs not to be bridged: gap must exceed a small
        # margin relative to detection width so "barely touching" doesn't count
        return True, gap, len(blobs)


def get_video_info(video_path: str) -> tuple[int, float, int, int]:
    """
    Return total_frames, fps, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0:
        raise RuntimeError("Could not determine total frame count.")
    if fps <= 0:
        raise RuntimeError("Could not determine FPS.")

    return total_frames, fps, width, height


def count_video_frames(video_path: str) -> int:
    """
    Count frames by reading the written file metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open segment: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def _load_config(config_path: str) -> Dict[str, Any]:
    config_path = str(Path(config_path).expanduser().resolve())
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a JSON object")
    return cfg


def _plan_segments_separation_aware(
    video_path: str,
    total_frames: int,
    fps: float,
    width: int,
    overlap_frames: int,
    min_frames: int,
    max_frames: int,
    sep_ckpt: Optional[str],
    sep_device: str,
    sep_stride: int,
    sep_min_gap_px: float,
) -> List[Tuple[int, int]]:
    """
    Plan (start, end_exclusive) segment spans so that each cut lands where the
    two rats are SEPARATED. For each segment, scan candidate cut frames in
    [start+min_frames, start+max_frames] (strided) and pick the frame with the
    best separation (two distinct blobs, largest centroid gap). If no candidate
    in the window shows the rats separated, fall back to the most-separated
    frame seen; if none had two blobs at all, cut at max_frames (best effort).
    """
    from decord import VideoReader, cpu
    from model_mask_refiner import compute_video_background_bgr

    if sep_ckpt is None:
        raise ValueError("separation_aware requires sep_ckpt (path to best.pt)")

    print("[separation] computing video background (median)...")
    bg = compute_video_background_bgr(video_path)
    scorer = SeparationScorer(sep_ckpt, background_bgr=bg, device=sep_device)

    vr = VideoReader(str(video_path), ctx=cpu(0))

    segments: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        # last segment: take whatever remains
        if start + min_frames >= total_frames:
            segments.append((start, total_frames))
            break

        win_lo = start + min_frames
        win_hi = min(start + max_frames, total_frames - 1)

        best_cut = None
        best_gap = -1.0
        any_two_blob = False

        cand = list(range(win_lo, win_hi + 1, sep_stride))
        if not cand or cand[-1] != win_hi:
            cand.append(win_hi)

        for f in cand:
            frame_rgb = vr[f].asnumpy()
            separated, gap, n_blobs = scorer.score(frame_rgb)
            if n_blobs >= 2:
                any_two_blob = True
            if separated and gap >= sep_min_gap_px and gap > best_gap:
                best_gap = gap
                best_cut = f

        if best_cut is None:
            # no confidently-separated frame; pick the most-separated two-blob
            # frame if we saw one, else fall back to the window's end (max len).
            if any_two_blob:
                # rescan to grab the max-gap frame even if below the margin
                for f in cand:
                    frame_rgb = vr[f].asnumpy()
                    separated, gap, n_blobs = scorer.score(frame_rgb)
                    if n_blobs >= 2 and gap > best_gap:
                        best_gap = gap
                        best_cut = f
            if best_cut is None:
                best_cut = win_hi
                print(f"  [separation] frames {win_lo}-{win_hi}: rats never "
                      f"clearly separated; cutting at {best_cut} (best effort)")
            else:
                print(f"  [separation] frames {win_lo}-{win_hi}: weak separation, "
                      f"cutting at {best_cut} (gap={best_gap:.0f}px)")
        else:
            print(f"  [separation] cut at frame {best_cut} "
                  f"(gap={best_gap:.0f}px, window {win_lo}-{win_hi})")

        end_excl = best_cut + 1  # cut frame is the last frame of this segment
        segments.append((start, end_excl))
        if end_excl >= total_frames:
            break
        start = end_excl - overlap_frames

    return segments


def run_fragment_video(
    input_video: str,
    output_dir: str,
    segment_seconds: float,
    overlap_frames: int = 1,
    clear_output_dir: bool = False,
    *,
    separation_aware: bool = False,
    min_segment_seconds: float = 15.0,
    max_segment_seconds: float = 23.0,
    sep_ckpt: Optional[str] = None,
    sep_device: str = "cuda:0",
    sep_stride: int = 5,
    sep_min_gap_frac: float = 0.10,
) -> dict:
    input_video = str(Path(input_video).expanduser().resolve())
    output_dir = str(Path(output_dir).expanduser().resolve())

    input_path = Path(input_video)
    output_path = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0")
    if overlap_frames < 0:
        raise ValueError("overlap_frames must be >= 0")

    output_path.mkdir(parents=True, exist_ok=True)

    if clear_output_dir:
        for p in output_path.glob("*.mp4"):
            p.unlink()

    total_frames, fps, width, height = get_video_info(str(input_path))
    segment_frames_nominal = int(round(float(segment_seconds) * fps))

    if segment_frames_nominal <= 0:
        raise RuntimeError("Segment length in frames is zero. Increase segment_seconds.")

    print("\n=== ORIGINAL VIDEO INFO ===")
    print(f"Input video:      {input_video}")
    print(f"Output directory: {output_dir}")
    print(f"FPS:              {fps}")
    print(f"Resolution:       {width} x {height}")
    print(f"Original frames:  {total_frames}")
    print(f"Segment seconds:  {segment_seconds}")
    print(f"Segment frames:   {segment_frames_nominal}")
    print(f"Overlap:          {overlap_frames} frame(s)")
    if separation_aware:
        print(f"Separation-aware: ON (window {min_segment_seconds}-{max_segment_seconds}s)")
    print("===========================\n")

    # Build the cut plan. Fixed by default; separation-aware if requested and
    # the scorer loads.
    if separation_aware:
        segments = _plan_segments_separation_aware(
            video_path=str(input_path),
            total_frames=total_frames,
            fps=fps,
            width=width,
            overlap_frames=overlap_frames,
            min_frames=int(round(min_segment_seconds * fps)),
            max_frames=int(round(max_segment_seconds * fps)),
            sep_ckpt=sep_ckpt,
            sep_device=sep_device,
            sep_stride=max(1, int(sep_stride)),
            sep_min_gap_px=sep_min_gap_frac * 1024.0,  # gap measured at train_w=1024
        )
    else:
        segments = []
        start_frame = 0
        while start_frame < total_frames:
            end_frame_exclusive = min(start_frame + segment_frames_nominal, total_frames)
            segments.append((start_frame, end_frame_exclusive))
            if end_frame_exclusive >= total_frames:
                break
            start_frame = end_frame_exclusive - overlap_frames

    created = 0
    segment_paths = []
    total_segment_frames = 0
    segment_frames_map: Dict[str, int] = {}

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    try:
        for i, (start_f, end_f_excl) in enumerate(segments, start=1):
            out_file = output_path / f"segment_{i:03d}.mp4"

            writer = cv2.VideoWriter(
                str(out_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {out_file}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

            frames_written = 0
            for _ in range(start_f, end_f_excl):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                frames_written += 1

            writer.release()

            actual_segment_frames = count_video_frames(str(out_file))
            total_segment_frames += actual_segment_frames
            segment_frames_map[out_file.name] = int(actual_segment_frames)

            print(f"Segment {i:03d}:")
            print(f"  Start frame: {start_f}")
            print(f"  End frame:   {end_f_excl - 1}")
            print(f"  Frames:      {actual_segment_frames}")
            print(f"  File:        {out_file}\n")

            if out_file.is_file():
                created += 1
                segment_paths.append(str(out_file.resolve()))

    finally:
        cap.release()

    # Manifest with exact per-segment frame counts. The segmentation chain
    # uses this to locate the start segment / restart frame without opening
    # a VideoReader for every segment file.
    manifest_path = output_path / "segments_manifest.json"
    manifest = {
        "input_video": input_video,
        "fps": fps,
        "width": width,
        "height": height,
        "segment_seconds": float(segment_seconds),
        "overlap_frames": int(overlap_frames),
        "original_frames": int(total_frames),
        "separation_aware": bool(separation_aware),
        "segment_frames": segment_frames_map,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote manifest: {manifest_path}")

    expected_stitched_frames = total_segment_frames - max(0, created - 1) * overlap_frames

    print("=========== SUMMARY ===========")
    print(f"Original frames:           {total_frames}")
    print(f"Total segment frames:      {total_segment_frames}")
    print(f"Segments created:          {created}")
    print(f"Expected stitched frames:  {expected_stitched_frames}")
    print("================================\n")

    return {
        "input_video": input_video,
        "output_dir": output_dir,
        "segment_seconds": float(segment_seconds),
        "overlap_frames": overlap_frames,
        "original_frames": total_frames,
        "segment_frames_total": total_segment_frames,
        "expected_stitched_frames": expected_stitched_frames,
        "created_segments": created,
        "segment_paths": segment_paths,
    }


def run_fragment_video_from_config(
    config_path: str,
    *,
    segment_seconds: float | None = None,
    clear_output_dir: bool = False,
) -> dict:
    cfg = _load_config(config_path)

    server = cfg.get("server", {}) or {}
    fragment = cfg.get("fragment", {}) or {}

    input_video = server.get("input_video")
    output_dir = server.get("segments_dir")

    if not input_video:
        raise KeyError("Missing required key 'server.input_video'")
    if not output_dir:
        raise KeyError("Missing required key 'server.segments_dir'")

    if segment_seconds is None:
        segment_seconds = float(fragment.get("segment_seconds", 30))

    overlap_frames = int(fragment.get("overlap_frames", 1))

    # Separation-aware fragmentation: cut where the two rats are separated so
    # segment boundaries never fall on a touching-rats frame. Reads an optional
    # `fragment.separation` block; falls back to the refiner ckpt if present.
    sep_cfg = fragment.get("separation", {}) or {}
    separation_aware = bool(sep_cfg.get("enabled", False))
    refiner_cfg = cfg.get("refiner", {}) or {}
    sep_ckpt = sep_cfg.get("ckpt") or refiner_cfg.get("ckpt")
    sep_device = sep_cfg.get("device") or refiner_cfg.get("device", "cuda:0")

    return run_fragment_video(
        input_video=str(input_video),
        output_dir=str(output_dir),
        segment_seconds=float(segment_seconds),
        overlap_frames=overlap_frames,
        clear_output_dir=clear_output_dir,
        separation_aware=separation_aware,
        min_segment_seconds=float(sep_cfg.get("min_segment_seconds", 15.0)),
        max_segment_seconds=float(sep_cfg.get("max_segment_seconds", 23.0)),
        sep_ckpt=sep_ckpt,
        sep_device=sep_device,
        sep_stride=int(sep_cfg.get("stride", 5)),
        sep_min_gap_frac=float(sep_cfg.get("min_gap_frac", 0.10)),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Split a video into fixed-length overlapping segments using exact frame indices."
    )
    ap.add_argument("--input_video", required=False)
    ap.add_argument("--output_dir", required=False)
    ap.add_argument("--segment_seconds", type=float, required=False)
    ap.add_argument("--overlap_frames", type=int, default=1)
    ap.add_argument("--config", required=False, help="Optional config JSON with server.input_video and server.segments_dir")
    ap.add_argument("--clear_output_dir", action="store_true")
    ap.add_argument("--separation_aware", action="store_true",
                    help="Cut segments where the two rats are separated (needs --sep_ckpt)")
    ap.add_argument("--min_segment_seconds", type=float, default=15.0)
    ap.add_argument("--max_segment_seconds", type=float, default=23.0)
    ap.add_argument("--sep_ckpt", required=False, help="Path to best.pt for separation detection")
    ap.add_argument("--sep_device", default="cuda:0")
    ap.add_argument("--sep_stride", type=int, default=5,
                    help="Check every Nth frame in the cut window (speed vs precision)")
    ap.add_argument("--sep_min_gap_frac", type=float, default=0.10,
                    help="Min centroid gap (fraction of 1024px detection width) to count as separated")

    args = ap.parse_args()

    if args.config:
        result = run_fragment_video_from_config(
            config_path=args.config,
            segment_seconds=args.segment_seconds,
            clear_output_dir=args.clear_output_dir,
        )
    else:
        if not args.input_video:
            raise ValueError("--input_video is required when --config is not used")
        if not args.output_dir:
            raise ValueError("--output_dir is required when --config is not used")
        if args.segment_seconds is None:
            raise ValueError("--segment_seconds is required when --config is not used")

        result = run_fragment_video(
            input_video=args.input_video,
            output_dir=args.output_dir,
            segment_seconds=args.segment_seconds,
            overlap_frames=args.overlap_frames,
            clear_output_dir=args.clear_output_dir,
            separation_aware=args.separation_aware,
            min_segment_seconds=args.min_segment_seconds,
            max_segment_seconds=args.max_segment_seconds,
            sep_ckpt=args.sep_ckpt,
            sep_device=args.sep_device,
            sep_stride=args.sep_stride,
            sep_min_gap_frac=args.sep_min_gap_frac,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()