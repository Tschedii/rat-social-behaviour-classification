"""
Demo: reproduces the background computation used by
model_mask_refiner.compute_video_background_bgr and shows what it removes.

Usage:
    python /path/to/master_thesis_social_behaviour/scripts/3_qualitiy_control/background_demo.py --video /path/to/master_thesis_social_behaviour/raw_data/Day_10_rPEG007vsNovel_1_24_06_2026/cam2.mp4 --out_dir ~/projects/data_master_thesis/experimental_data/qc_out

Produces:
    out/background.png        the per-pixel temporal median (the "empty arena")
    out/sample_frame.png      one raw frame for comparison
    out/foreground_prior.png  |frame - background| heatmap (illustrative -
                               the pipeline's actual prior fn (compute_foreground_prior)
                               lives in prepare_dataset.py, not included here;
                               this is a simple stand-in with the same idea)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def compute_video_background_bgr(video_path: str, max_samples: int = 150,
                                  band_rows: int = 300) -> np.ndarray:
    """Per-pixel temporal median over a strided sample of the full video.

    Same approach as model_mask_refiner.compute_video_background_bgr, but
    using cv2.VideoCapture instead of decord so it runs with no extra deps.
    Processed in row bands to bound peak memory on large videos.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise RuntimeError(f"Could not read frame count from: {video_path}")

    idx = np.linspace(0, n - 1, min(max_samples, n)).round().astype(int)
    idx = sorted(set(int(i) for i in idx))

    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame_bgr = cap.read()
        if ok:
            frames.append(frame_bgr)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be sampled from: {video_path}")

    stack = np.stack(frames, axis=0)  # S, H, W, 3 (BGR)
    S, H, W, _ = stack.shape

    bg_bgr = np.empty((H, W, 3), dtype=np.uint8)
    for y0 in range(0, H, band_rows):
        y1 = min(y0 + band_rows, H)
        bg_bgr[y0:y1] = np.median(stack[:, y0:y1], axis=0).astype(np.uint8)

    return bg_bgr, stack[len(stack) // 2]  # background + one sample frame


def foreground_prior_demo(frame_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """
    Illustrative stand-in for prepare_dataset.compute_foreground_prior.
    Absolute per-channel difference from the background, normalized to [0,1].
    The real pipeline's version may add blur/normalization/morphology on top
    of this same core idea (frame vs. static background).
    """
    diff = cv2.absdiff(frame_bgr, bg_bgr).astype(np.float32)
    diff_gray = diff.max(axis=2)  # strongest channel deviation per pixel
    diff_gray = diff_gray / max(diff_gray.max(), 1e-6)
    return diff_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", default="bg_demo_out")
    ap.add_argument("--max_samples", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_bgr, sample_frame_bgr = compute_video_background_bgr(
        args.video, max_samples=args.max_samples
    )

    prior = foreground_prior_demo(sample_frame_bgr, bg_bgr)
    prior_heat = cv2.applyColorMap((prior * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    cv2.imwrite(str(out_dir / "background.png"), bg_bgr)
    cv2.imwrite(str(out_dir / "sample_frame.png"), sample_frame_bgr)
    cv2.imwrite(str(out_dir / "foreground_prior.png"), prior_heat)

    # side-by-side composite for a quick look
    h, w = bg_bgr.shape[:2]
    composite = np.hstack([sample_frame_bgr, bg_bgr, prior_heat])
    cv2.imwrite(str(out_dir / "composite.png"), composite)

    print(f"Saved: {out_dir/'background.png'}, {out_dir/'sample_frame.png'}, "
          f"{out_dir/'foreground_prior.png'}, {out_dir/'composite.png'}")


if __name__ == "__main__":
    main()
