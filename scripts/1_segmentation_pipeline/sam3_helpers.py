#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import queue
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil
from pathlib import Path


import cv2
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
SEGMENT_RE = re.compile(r"segment_\d{3}\.mp4$", re.IGNORECASE)


# ============================================================
# Data containers
# ============================================================

@dataclass
class SegmentChainConfig:
    video_id: str
    prompts_json: str
    segments_root: str = "data/segments"
    out_root: str = "data/sam3_outputs"
    auto_pos: int = 8
    auto_neg: int = 8
    seed: int = 123


@dataclass
class OutputPaths:
    out_dir: str
    masks_raw_dir: str
    overlay_avi_path: str  # only used by the cv2 fallback writer
    overlay_mp4_path: str


@dataclass
class SegmentContext:
    predictor: Any
    writer: Any
    obj_ids_sorted: List[int]
    paths: OutputPaths


@dataclass
class ObjectMemory:
    obj_id: int
    # Each entry: (mask_bool, weight, (cx, cy)). Mask stored ONCE, weighted in fusion.
    mask_history: deque = field(default_factory=lambda: deque(maxlen=10))
    area_history: deque = field(default_factory=lambda: deque(maxlen=20))
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=20))
    appearance_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_fused_mask: Optional[np.ndarray] = None
    # Recompute the fused shape prior at most every N updates (it is only
    # consumed as a soft IoU reference and as the next-segment prompt mask,
    # so a few frames of staleness is harmless and saves ~30 full-frame
    # passes per object per frame).
    fuse_every: int = 8
    _updates_since_fuse: int = field(default=0, repr=False)

    def update(self, frame_rgb: np.ndarray, mask: np.ndarray, weight: int = 1) -> None:
        m = np.asarray(mask).astype(bool)
        if not np.any(m):
            return

        hist = compute_mask_histogram(frame_rgb, m)
        ys, xs = np.where(m)
        area = float(len(xs))
        cx = float(xs.mean())
        cy = float(ys.mean())

        w_int = max(1, int(weight))

        # Mask stored once with its weight (no repeated full-frame copies).
        self.mask_history.append((m, w_int, (cx, cy)))

        # Scalar/histogram histories are cheap; keep repeated appends so
        # medians/means stay weighted exactly like before.
        for _ in range(w_int):
            self.area_history.append(area)
            self.centroid_history.append((cx, cy))
            self.appearance_history.append(hist)

        self._updates_since_fuse += 1
        if self.last_fused_mask is None or self._updates_since_fuse >= self.fuse_every:
            self.last_fused_mask = build_fused_shape_prior(list(self.mask_history))
            self._updates_since_fuse = 0

    def refresh_fused_mask(self) -> Optional[np.ndarray]:
        """Force an up-to-date fused prior (used at segment boundaries)."""
        if self.mask_history:
            self.last_fused_mask = build_fused_shape_prior(list(self.mask_history))
            self._updates_since_fuse = 0
        return self.last_fused_mask


# ============================================================
# Basic IO helpers
# ============================================================

def list_segments(folder: str) -> List[str]:
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        return []

    segment_paths = sorted(
        str(p)
        for p in folder_path.iterdir()
        if p.is_file() and SEGMENT_RE.match(p.name)
    )
    if segment_paths:
        return segment_paths

    return sorted(
        str(p)
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def load_prompts_json(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "objects" not in data or not isinstance(data["objects"], dict):
        raise ValueError(f"Invalid prompts JSON: {path}")

    prompts: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for obj_id_str, obj_data in data["objects"].items():
        points = np.asarray(obj_data["points_xy"], dtype=np.float32)
        labels = np.asarray(obj_data["labels"], dtype=np.int32)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Object {obj_id_str}: points_xy must be Nx2")

        if labels.ndim != 1 or len(labels) != len(points):
            raise ValueError(f"Object {obj_id_str}: labels must match points_xy length")

        prompts[int(obj_id_str)] = (points, labels)

    if not prompts:
        raise ValueError("No objects found in prompts JSON")

    return prompts


def abs_to_rel(points_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    return pts


def extract_masks(outputs: dict) -> np.ndarray:
    for key in (
        "out_binary_masks",
        "masks",
        "out_masks",
        "mask",
        "mask_logits",
        "out_mask_logits",
    ):
        if key in outputs and outputs[key] is not None:
            return outputs[key]

    raise KeyError(f"No masks found in outputs. Keys={list(outputs.keys())}")


def build_label_mask(last_masks: Dict[int, np.ndarray], h: int, w: int) -> np.ndarray:
    label = np.zeros((h, w), dtype=np.uint8)
    for oid, mask in last_masks.items():
        label[np.asarray(mask).astype(bool)] = int(oid)
    return label


def build_visible_mask(label: np.ndarray) -> np.ndarray:
    viz = np.zeros_like(label, dtype=np.uint8)
    viz[label == 1] = 127
    viz[label == 2] = 255
    return viz


def apply_overlay(frame_rgb: np.ndarray, last_masks: Dict[int, np.ndarray]) -> np.ndarray:
    overlay = frame_rgb.copy()
    color_map = {
        1: np.array([0, 0, 255], dtype=np.uint8),
        2: np.array([255, 0, 0], dtype=np.uint8),
    }

    for oid, mask in last_masks.items():
        color = color_map.get(oid)
        if color is None:
            continue
        m = np.asarray(mask).astype(bool)
        overlay[m] = (0.5 * overlay[m] + 0.5 * color).astype(np.uint8)

    return overlay


def parse_candidate_masks_from_outputs(
    outputs: dict,
    obj_ids_sorted: List[int],
) -> Dict[int, np.ndarray]:
    masks = np.asarray(extract_masks(outputs))
    candidates: Dict[int, np.ndarray] = {}

    if masks.ndim == 2:
        if obj_ids_sorted:
            candidates[obj_ids_sorted[0]] = masks.astype(bool)
        return candidates

    out_obj_ids = outputs.get("out_obj_ids")
    if out_obj_ids is not None:
        for i, oid in enumerate(out_obj_ids):
            if i < masks.shape[0]:
                candidates[int(oid)] = masks[i].astype(bool)
        return candidates

    for i, oid in enumerate(obj_ids_sorted):
        if i < masks.shape[0]:
            candidates[int(oid)] = masks[i].astype(bool)

    return candidates


def update_last_masks_from_outputs(
    outputs: dict,
    obj_ids_sorted: List[int],
    last_masks: Dict[int, np.ndarray],
) -> None:
    parsed = parse_candidate_masks_from_outputs(outputs, obj_ids_sorted)
    for oid, mask in parsed.items():
        last_masks[int(oid)] = np.asarray(mask).astype(bool)


# Palette for label PNGs: index 0 = black background, 1 = red (rat 1),
# 2 = blue (rat 2), 3+ = green fallback. Pixel VALUES stay 0/1/2 on disk;
# the palette only controls how image viewers render them.
_LABEL_PALETTE = [0, 0, 0, 255, 70, 70, 80, 140, 255, 90, 220, 90]
_LABEL_PALETTE += [0] * (256 * 3 - len(_LABEL_PALETTE))


def save_label_mask(frame_idx: int, label: np.ndarray, masks_raw_dir: str) -> None:
    """
    Save the raw label mask as an indexed-color PNG.

    - Values on disk remain the object ids (0 background, 1, 2, ...), so
      any code that reads labels keeps working IF it reads via PIL:
          label = np.array(Image.open(path))          # -> 0/1/2
      or via load_label_mask() below.
      NOTE: cv2.imread() would expand the palette to BGR colors; do not use
      it for reading these files (or use load_label_mask()).
    - The embedded palette makes both rats clearly visible (red / blue) in
      any normal image viewer, replacing the old masks_viz output.
    - compress_level=1 makes saving several times faster than the PIL
      default (level 6) with only slightly larger files.
    """
    raw_path = os.path.join(masks_raw_dir, f"frame_{frame_idx:06d}.png")
    img = Image.fromarray(np.ascontiguousarray(label), mode="P")
    img.putpalette(_LABEL_PALETTE)
    img.save(raw_path, optimize=False, compress_level=1)


def load_label_mask(path: str) -> np.ndarray:
    """Read a label PNG back as its raw id values (0/1/2), not RGB colors."""
    return np.array(Image.open(path))


def save_masks(frame_idx: int, label: np.ndarray, masks_raw_dir: str, masks_viz_dir: str = None) -> None:
    # Backward-compatible wrapper; masks_viz output has been removed.
    save_label_mask(frame_idx, label, masks_raw_dir)


# ============================================================
# Overlay video sinks + async frame writer
# ============================================================

class FfmpegVideoSink:
    """
    Pipe raw BGR frames straight into ffmpeg/libx264 producing the final MP4.
    One encode instead of MJPG-AVI + re-encode, and no multi-GB intermediate.
    """

    @staticmethod
    def available() -> bool:
        return shutil.which("ffmpeg") is not None

    def __init__(self, out_path: str, w: int, h: int, fps: float,
                 crf: int = 20, preset: str = "veryfast"):
        self.out_path = str(Path(out_path).expanduser().resolve())
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        self._frame_shape = (h, w, 3)

        cmd = [
            shutil.which("ffmpeg"), "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{fps}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self.out_path,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape != self._frame_shape:
            raise ValueError(
                f"Frame shape {frame_bgr.shape} does not match sink {self._frame_shape}"
            )
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError as e:
            stderr = b""
            try:
                stderr = self.proc.stderr.read() or b""
            except Exception:
                pass
            raise RuntimeError(
                f"ffmpeg overlay encoder died: {stderr.decode(errors='replace')}"
            ) from e

    def release(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception as e:
            print(f"[FfmpegVideoSink] Warning while closing: {e}")
        finally:
            self.proc = None


class AsyncFrameWriter:
    """
    Background thread that handles ALL per-frame disk work (label PNG,
    overlay blend, video encode, throttled restart checkpoint) so that the
    SAM3 propagate loop never blocks on I/O.

    The video sink is only ever touched from the writer thread, so both
    cv2.VideoWriter and the ffmpeg pipe are safe here.
    """

    def __init__(
        self,
        masks_raw_dir: str,
        video_sink,
        checkpoint_fn=None,
        checkpoint_every: int = 25,
        queue_size: int = 32,
    ):
        self.masks_raw_dir = masks_raw_dir
        self.video_sink = video_sink
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_every = max(1, int(checkpoint_every))

        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._error: Optional[BaseException] = None
        self._frames_written = 0
        self._last_checkpoint: Optional[dict] = None
        self._closed = False

        self._thread = threading.Thread(
            target=self._loop, name="frame-writer", daemon=True
        )
        self._thread.start()

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def raise_if_failed(self) -> None:
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError(f"Async frame writer failed: {err}") from err

    def submit(
        self,
        frame_rgb: np.ndarray,
        masks: Dict[int, np.ndarray],
        frame_idx_global: int,
        checkpoint: Optional[dict] = None,
    ) -> None:
        self.raise_if_failed()
        self._q.put((frame_rgb, masks, int(frame_idx_global), checkpoint))

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            try:
                if self._error is None:
                    frame_rgb, masks, frame_idx_global, checkpoint = item
                    h, w = frame_rgb.shape[:2]
                    label = build_label_mask(masks, h, w)
                    save_label_mask(frame_idx_global, label, self.masks_raw_dir)

                    overlay_rgb = apply_overlay(frame_rgb, masks)
                    self.video_sink.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

                    self._frames_written += 1
                    if checkpoint is not None:
                        self._last_checkpoint = checkpoint
                        if (
                            self.checkpoint_fn is not None
                            and self._frames_written % self.checkpoint_every == 0
                        ):
                            self.checkpoint_fn(checkpoint)
            except BaseException as e:  # keep the thread alive, surface later
                self._error = e
            finally:
                self._q.task_done()

    def flush(self) -> None:
        """Block until everything queued so far is on disk, then persist the
        most recent checkpoint (so restart points at the true last frame)."""
        self._q.join()
        if self.checkpoint_fn is not None and self._last_checkpoint is not None:
            try:
                self.checkpoint_fn(self._last_checkpoint)
            except Exception as e:
                print(f"[AsyncFrameWriter] Warning while saving checkpoint: {e}")
        self.raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            self._q.put(None)
            self._thread.join(timeout=60)
            try:
                self.video_sink.release()
            except Exception as e:
                print(f"[AsyncFrameWriter] Warning while releasing sink: {e}")
        self.raise_if_failed()


# ============================================================
# Profiling
# ============================================================

class SegmentProfiler:
    """
    Splits the propagate loop into three buckets so you can see where time
    actually goes:

      - predictor: wall time the SAM3 stream generator spends producing each
        response (i.e. GPU/model work between yields).
      - decode: decord frame fetch (vr[frame_idx].asnumpy()).
      - post: everything else per frame (candidate parse, identity swap,
        review handling, and the non-blocking writer.submit call).

    Interpretation:
      - predictor >> post  -> GPU-bound; the async writer has hidden I/O.
        Python-side micro-opts won't move wall-clock.
      - post large / growing -> the writer queue is backing up (disk can't
        keep pace); raise queue_size or write less.

    All timers are pure wall clock (perf_counter); overhead is a few
    microseconds per frame, so it is safe to leave enabled.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self.t_predictor = 0.0
        self.t_decode = 0.0
        self.t_post = 0.0
        self.n_frames = 0
        # totals across all segments, for the final summary
        self._tot_predictor = 0.0
        self._tot_decode = 0.0
        self._tot_post = 0.0
        self._tot_frames = 0

    # --- context managers for the three buckets ---
    def predictor(self):
        return _Stopwatch(self, "t_predictor")

    def decode(self):
        return _Stopwatch(self, "t_decode")

    def post(self):
        return _Stopwatch(self, "t_post")

    def tick_frame(self) -> None:
        self.n_frames += 1

    def segment_summary(self, seg_name: str = "") -> None:
        if not self.enabled:
            return
        pred, dec, post = self.t_predictor, self.t_decode, self.t_post
        total = pred + dec + post
        n = max(self.n_frames, 1)
        overhead = (dec + post) / max(pred, 1e-9) * 100.0
        tag = f" [{seg_name}]" if seg_name else ""
        print(
            f"  profile{tag}: frames={self.n_frames}  "
            f"predictor={pred:.1f}s ({pred/max(total,1e-9)*100:.0f}%)  "
            f"decode={dec:.1f}s ({dec/max(total,1e-9)*100:.0f}%)  "
            f"post={post:.1f}s ({post/max(total,1e-9)*100:.0f}%)  "
            f"| {1000*pred/n:.1f}/{1000*dec/n:.1f}/{1000*post/n:.1f} ms per frame  "
            f"| non-predictor overhead {overhead:.0f}%"
        )
        # accumulate then reset per-segment counters
        self._tot_predictor += pred
        self._tot_decode += dec
        self._tot_post += post
        self._tot_frames += self.n_frames
        self.t_predictor = self.t_decode = self.t_post = 0.0
        self.n_frames = 0

    def final_summary(self) -> None:
        if not self.enabled or self._tot_frames == 0:
            return
        pred, dec, post = self._tot_predictor, self._tot_decode, self._tot_post
        total = pred + dec + post
        n = self._tot_frames
        overhead = (dec + post) / max(pred, 1e-9) * 100.0
        print("\n=== PROFILE SUMMARY (all segments) ===")
        print(f"Frames processed: {n}")
        print(f"Predictor (GPU):  {pred:8.1f}s  ({pred/max(total,1e-9)*100:5.1f}%)  {1000*pred/n:6.2f} ms/frame")
        print(f"Decode (decord):  {dec:8.1f}s  ({dec/max(total,1e-9)*100:5.1f}%)  {1000*dec/n:6.2f} ms/frame")
        print(f"Post-processing:  {post:8.1f}s  ({post/max(total,1e-9)*100:5.1f}%)  {1000*post/n:6.2f} ms/frame")
        print(f"Non-predictor overhead vs GPU time: {overhead:.0f}%")
        if overhead < 15:
            print("Verdict: GPU-bound. Async writer is hiding I/O; Python-side opts won't help wall-clock.")
        elif post > dec:
            print("Verdict: post-processing/writer-bound. Writer queue may be backing up -> raise queue_size or write less.")
        else:
            print("Verdict: decode-bound. Consider decoding at lower resolution.")
        print("======================================")


class _Stopwatch:
    """Accumulates elapsed wall time into a named attribute of a profiler."""

    __slots__ = ("_prof", "_attr", "_t0")

    def __init__(self, prof: "SegmentProfiler", attr: str):
        self._prof = prof
        self._attr = attr
        self._t0 = 0.0

    def __enter__(self):
        if self._prof.enabled:
            import time as _time
            self._t0 = _time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._prof.enabled:
            import time as _time
            setattr(
                self._prof,
                self._attr,
                getattr(self._prof, self._attr) + (_time.perf_counter() - self._t0),
            )
        return False


def maybe_convert_avi_to_mp4(avi_path: str, mp4_path: str) -> bool:
    avi_path = str(Path(avi_path).expanduser().resolve())
    mp4_path = str(Path(mp4_path).expanduser().resolve())

    avi_file = Path(avi_path)
    mp4_file = Path(mp4_path)

    if not avi_file.exists():
        print(f"[ffmpeg] AVI input does not exist: {avi_file}")
        return False

    if avi_file.stat().st_size == 0:
        print(f"[ffmpeg] AVI input is empty: {avi_file}")
        return False

    mp4_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("[ffmpeg] ffmpeg executable not found in PATH")
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(avi_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(mp4_file),
    ]

    print(f"[ffmpeg] Converting AVI -> MP4")
    print(f"[ffmpeg] input : {avi_file}")
    print(f"[ffmpeg] output: {mp4_file}")

    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        print(f"[ffmpeg] Failed to launch ffmpeg: {e}")
        return False

    if result.returncode != 0:
        print("[ffmpeg] Conversion failed")
        print(result.stderr)
        return False

    if not mp4_file.exists():
        print(f"[ffmpeg] Conversion reported success but output file is missing: {mp4_file}")
        return False

    if mp4_file.stat().st_size == 0:
        print(f"[ffmpeg] Output MP4 is empty: {mp4_file}")
        return False

    print(f"[ffmpeg] Conversion successful: {mp4_file}")
    return True


# ============================================================
# Mask / appearance utilities
# ============================================================

def compute_mask_histogram(frame_rgb: np.ndarray, mask: np.ndarray, bins: int = 16) -> np.ndarray:
    pixels = frame_rgb[np.asarray(mask).astype(bool)]
    if len(pixels) == 0:
        return np.zeros((bins * 3,), dtype=np.float32)

    # bins must be a power of two <= 256 for the shift trick (default 16 -> shift 4)
    shift = 8 - int(np.log2(bins))
    idx = pixels.astype(np.uint8) >> shift

    feats = []
    for c in range(3):
        hist = np.bincount(idx[:, c], minlength=bins).astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        feats.append(hist)

    return np.concatenate(feats, axis=0)


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1 is None or h2 is None or len(h1) == 0 or len(h2) == 0:
        return 0.0
    return float(np.minimum(h1, h2).sum())


def build_fused_shape_prior(mask_history: List) -> Optional[np.ndarray]:
    """
    Fuse a history of masks into a soft shape prior.

    Accepts entries of the form (mask, weight, (cx, cy)) as stored by
    ObjectMemory. Bare masks are still accepted for backward compatibility
    (weight 1, centroid computed on the fly).
    """
    if not mask_history:
        return None

    entries = []
    for item in mask_history:
        if isinstance(item, tuple) and len(item) == 3:
            m, w_int, (cx, cy) = item
            m = np.asarray(m).astype(bool)
            if not np.any(m):
                continue
            entries.append((m, float(max(1, int(w_int))), (float(cx), float(cy))))
        else:
            m = np.asarray(item).astype(bool)
            if not np.any(m):
                continue
            ys, xs = np.where(m)
            entries.append((m, 1.0, (float(xs.mean()), float(ys.mean()))))

    if not entries:
        return None

    h, w = entries[0][0].shape
    ref_cx, ref_cy = entries[-1][2]
    acc = np.zeros((h, w), dtype=np.float32)
    total_weight = 0.0

    for m, weight, (cx, cy) in entries:
        dx = int(round(ref_cx - cx))
        dy = int(round(ref_cy - cy))

        shifted = np.roll(m.astype(np.uint8), shift=(dy, dx), axis=(0, 1))

        if dy > 0:
            shifted[:dy, :] = 0
        elif dy < 0:
            shifted[dy:, :] = 0

        if dx > 0:
            shifted[:, :dx] = 0
        elif dx < 0:
            shifted[:, dx:] = 0

        acc += weight * shifted.astype(np.float32)
        total_weight += weight

    avg = acc / max(total_weight, 1.0)
    return avg >= 0.4


def compute_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    return float(xs.mean()), float(ys.mean())


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(float(union), 1.0)


def _extract_candidate_features(
    frame_rgb: np.ndarray,
    candidate_mask: np.ndarray,
) -> Optional[dict]:
    m = np.asarray(candidate_mask).astype(bool)
    if not np.any(m):
        return None

    ys, xs = np.where(m)
    return {
        "mask": m,
        "area": float(len(xs)),
        "centroid": (float(xs.mean()), float(ys.mean())),
        # histogram computed lazily by caller (skipped when the swap check
        # is gated out by distance)
        "hist": None,
        "_frame_rgb": frame_rgb,
    }


def _feature_hist(feats: dict) -> np.ndarray:
    if feats["hist"] is None:
        feats["hist"] = compute_mask_histogram(feats["_frame_rgb"], feats["mask"])
    return feats["hist"]


def _score_features_against_memory(feats: Optional[dict], memory: ObjectMemory) -> float:
    if feats is None:
        return -1e9

    score = 0.0
    area = feats["area"]
    cx, cy = feats["centroid"]

    if memory.area_history:
        ref_area = float(np.median(memory.area_history))
        area_ratio = min(area, ref_area) / max(area, ref_area)
        score += 1.5 * area_ratio

    if memory.centroid_history:
        ref_cx, ref_cy = memory.centroid_history[-1]
        jump = np.hypot(cx - ref_cx, cy - ref_cy)
        score += 1.0 * np.exp(-jump / 80.0)

    if memory.appearance_history:
        hist = _feature_hist(feats)
        sims = [histogram_similarity(hist, h) for h in memory.appearance_history]
        score += 2.0 * float(np.mean(sims))

    if memory.last_fused_mask is not None:
        score += 2.0 * compute_iou(feats["mask"], memory.last_fused_mask)

    return score


def compute_identity_score(
    frame_rgb: np.ndarray,
    candidate_mask: np.ndarray,
    memory: ObjectMemory,
) -> float:
    feats = _extract_candidate_features(frame_rgb, candidate_mask)
    return _score_features_against_memory(feats, memory)


def maybe_resolve_identity_swap(
    frame_rgb: np.ndarray,
    candidates: Dict[int, np.ndarray],
    obj_ids_sorted: List[int],
    object_memories: Dict[int, ObjectMemory],
    swap_check_radius: float = 200.0,
) -> Dict[int, np.ndarray]:
    if len(obj_ids_sorted) != 2:
        return candidates

    a, b = obj_ids_sorted
    if a not in candidates or b not in candidates:
        return candidates

    feat_a = _extract_candidate_features(frame_rgb, candidates[a])
    feat_b = _extract_candidate_features(frame_rgb, candidates[b])

    if feat_a is None or feat_b is None:
        return candidates

    # A swap is only plausible when the two animals are close. When they are
    # far apart, skip the (histogram-heavy) scoring entirely.
    ax, ay = feat_a["centroid"]
    bx, by = feat_b["centroid"]
    if np.hypot(ax - bx, ay - by) > swap_check_radius:
        return candidates

    keep_score = (
        _score_features_against_memory(feat_a, object_memories[a])
        + _score_features_against_memory(feat_b, object_memories[b])
    )
    swap_score = (
        _score_features_against_memory(feat_a, object_memories[b])
        + _score_features_against_memory(feat_b, object_memories[a])
    )

    if swap_score > keep_score + 0.25:
        return {a: candidates[b], b: candidates[a]}

    return candidates


# ============================================================
# Prompt generation from masks
# ============================================================

def _erode_bool(mask_bool: np.ndarray, iters: int = 3) -> np.ndarray:
    """Erode a boolean mask by `iters` iterations of a 3x3 kernel."""
    k = np.ones((3, 3), np.uint8)
    return cv2.erode(mask_bool.astype(np.uint8), k, iterations=iters).astype(bool)


def sample_points_from_mask(
    mask: np.ndarray,
    num_pos: int,
    num_neg: int,
    seed: int = 0,
    spread_whole_mask: bool = False,
    max_total_points: Optional[int] = None,
    other_masks: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample point prompts from a mask.

    spread_whole_mask: when True (used to RE-DESCRIBE a whole object at a
    segment boundary), positive points are spread uniformly across the ENTIRE
    mask area (body AND tail/extremities) by farthest-point sampling, instead
    of being concentrated in the distance-transform core. Core-weighting is
    right for a single correction click mid-tracking, but at a re-seed it makes
    SAM3 reconstruct only the compact centre - the object collapses to a blob
    and the tail is lost. Spreading the points makes SAM3 reproduce the full
    shape.

    max_total_points: hard cap on pos+neg points. SAM3 silently keeps only the
    first 8 and last 8 of any over-cap prompt (the 'Too many points (60)'
    warning), so sending 60 wastes ~44 of them and, worse, the kept 16 are the
    arbitrary first/last rather than a good spread. Capping here (e.g. 16) means
    every point we send is actually used.

    other_masks: list of the OTHER objects' masks (same frame). Used at re-seed
    to keep two touching/crossing objects from merging: (1) positive points are
    never placed inside another object's (dilated) region - critical when the
    rats' tails cross, since a positive point that lands on the other rat's tail
    makes SAM3 grab that rat too and the two identities melt into one; (2)
    negative points are preferentially placed ON the other objects, explicitly
    telling SAM3 'that neighbouring rat is NOT you'. This is the fix for the
    boundary-only merge regression introduced by whole-mask tail spreading.
    """
    rng = np.random.default_rng(seed)
    m = np.asarray(mask).astype(bool)

    if not np.any(m):
        raise RuntimeError("Mask is empty; cannot initialize next segment from it.")

    # Build a combined "other objects" mask and a dilated exclusion zone.
    other_union = None
    other_excl = None
    if other_masks:
        other_union = np.zeros_like(m, dtype=bool)
        for om in other_masks:
            if om is None:
                continue
            om = np.asarray(om).astype(bool)
            if om.shape == m.shape:
                other_union |= om
        if other_union.any():
            # Exclude a dilated band around the other object so positive points
            # keep clear of the contested crossing/overlap region.
            _k = np.ones((5, 5), np.uint8)
            other_excl = cv2.dilate(other_union.astype(np.uint8), _k, iterations=4).astype(bool)
            # Spare only our OWN deep core that is also far from the other object
            # (so a compact isolated body always keeps some positives), but do
            # NOT spare interior pixels that overlap the other object - those
            # are exactly the crossing-tail pixels we must avoid seeding on.
            own_core_far = _erode_bool(m, 5) & (~other_union)
            other_excl = other_excl & (~own_core_far)
        else:
            other_union = None

    # Apply the usable-budget cap up front so pos/neg are chosen within it.
    if max_total_points is not None and max_total_points > 0:
        # Split budget ~2:1 positive:negative, at least 1 neg.
        cap_pos = max(1, int(round(max_total_points * 2 / 3)))
        cap_neg = max(1, max_total_points - cap_pos)
        num_pos = min(num_pos, cap_pos)
        num_neg = min(num_neg, cap_neg)

    def _select_spread(
        coords_xy: np.ndarray,
        quality: np.ndarray,
        k: int,
        target_spacing: float,
    ) -> np.ndarray:
        n = len(coords_xy)
        if n == 0 or k <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if k >= n:
            return coords_xy.astype(np.float32)

        coords_xy = coords_xy.astype(np.float32)
        quality = quality.astype(np.float32)

        qmin = float(quality.min())
        qmax = float(quality.max())
        if qmax > qmin:
            q = (quality - qmin) / (qmax - qmin)
        else:
            q = np.ones(n, dtype=np.float32)

        first_idx = int(np.argmax(q + rng.normal(0, 1e-6, size=n)))
        selected = [first_idx]

        min_d2 = np.sum((coords_xy - coords_xy[first_idx]) ** 2, axis=1)
        min_d2[first_idx] = 0.0

        for _ in range(1, k):
            diversity = np.clip(np.sqrt(min_d2) / max(target_spacing, 1.0), 0.0, 1.0)
            score = 0.75 * q + 0.25 * diversity
            score[selected] = -np.inf
            next_idx = int(np.argmax(score))
            selected.append(next_idx)

            d2 = np.sum((coords_xy - coords_xy[next_idx]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        return coords_xy[selected]

    m_u8 = (m.astype(np.uint8) * 255)
    dist_in = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)

    ys, xs = np.where(m)
    pos_coords_all = np.stack([xs, ys], axis=1).astype(np.float32)
    pos_dist = dist_in[ys, xs]

    # Cross-object avoidance: drop positive candidates that fall in the dilated
    # other-object zone (e.g. where the two rats' tails cross). Keep at least
    # the object's own core so we never end up with zero positives.
    if other_excl is not None and other_excl.any():
        keep = ~other_excl[ys, xs]
        n_keep = int(np.count_nonzero(keep))
        # Apply exclusion as long as a minimal number of our own pixels remain
        # outside the contested zone. Only skip entirely if we'd be left with
        # almost nothing (then rely on negatives to disambiguate).
        if n_keep >= max(1, min(num_pos, 4)):
            pos_coords_all = pos_coords_all[keep]
            pos_dist = pos_dist[keep]

    maxd = float(pos_dist.max()) if len(pos_dist) > 0 else 0.0
    if maxd <= 0:
        maxd = 1.0

    obj_area = float(np.count_nonzero(m))
    spacing = max(8.0, np.sqrt(obj_area / max(num_pos, 1)) * 0.7)

    # Whole-mask spread: farthest-point sample across ALL interior pixels so the
    # positive points cover the entire object (body + tail), not just the core.
    # A small distance-transform floor keeps points off the antialiased edge but
    # does NOT concentrate them in the centre.
    if spread_whole_mask:
        # Include thin structures (tail): all mask pixels minus the 1px rim.
        interior = pos_dist >= 1.0
        cand = pos_coords_all[interior] if np.any(interior) else pos_coords_all
        cand = cand.astype(np.float32)

        # TRUE farthest-point sampling: repeatedly pick the candidate whose
        # nearest-already-selected distance is largest. This guarantees coverage
        # of thin extremities (the tail) regardless of how few pixels they have,
        # because once the body is covered the farthest remaining pixel is out on
        # the tail. The blended-score _select_spread saturates its diversity term
        # and lets the denser body dominate, which is why the tail was missed.
        k = int(min(num_pos, len(cand)))
        if k <= 0:
            pos_pts = np.zeros((0, 2), dtype=np.float32)
        else:
            # start from the mask centroid's farthest pixel for determinism
            c0 = cand.mean(axis=0)
            start = int(np.argmax(np.sum((cand - c0) ** 2, axis=1)))
            selected = [start]
            min_d2 = np.sum((cand - cand[start]) ** 2, axis=1)
            for _ in range(1, k):
                nxt = int(np.argmax(min_d2))
                if min_d2[nxt] <= 0:
                    break
                selected.append(nxt)
                d2 = np.sum((cand - cand[nxt]) ** 2, axis=1)
                min_d2 = np.minimum(min_d2, d2)
            pos_pts = cand[selected]
        if len(pos_pts) > 0:
            pos_pts = np.unique(pos_pts.astype(np.int32), axis=0).astype(np.float32)
            if len(pos_pts) > num_pos:
                pos_pts = pos_pts[:num_pos]
    else:
        core_keep = pos_dist >= 0.55 * maxd
        mid_keep = (pos_dist >= 0.20 * maxd) & (pos_dist < 0.55 * maxd)
        edge_keep = (pos_dist > 0.0) & (pos_dist < 0.20 * maxd)

        core_coords = pos_coords_all[core_keep]
        core_q = pos_dist[core_keep]

        mid_coords = pos_coords_all[mid_keep]
        mid_q = pos_dist[mid_keep]

        edge_coords = pos_coords_all[edge_keep]
        edge_q = 1.0 / (pos_dist[edge_keep] + 1.0)

        k_core = min(max(3, num_pos // 2), num_pos)
        k_mid = min(2, max(0, num_pos - k_core))
        k_edge = max(0, num_pos - k_core - k_mid)

        pos_parts = []
        if len(core_coords) > 0 and k_core > 0:
            pos_parts.append(_select_spread(core_coords, core_q, min(k_core, len(core_coords)), spacing))
        if len(mid_coords) > 0 and k_mid > 0:
            pos_parts.append(_select_spread(mid_coords, mid_q, min(k_mid, len(mid_coords)), spacing * 0.9))
        if len(edge_coords) > 0 and k_edge > 0:
            pos_parts.append(_select_spread(edge_coords, edge_q, min(k_edge, len(edge_coords)), spacing * 0.8))

        pos_pts = np.concatenate(pos_parts, axis=0) if pos_parts else np.zeros((0, 2), dtype=np.float32)

        if len(pos_pts) < num_pos:
            need = num_pos - len(pos_pts)
            extra = _select_spread(pos_coords_all, pos_dist, min(need, len(pos_coords_all)), spacing)
            pos_pts = np.concatenate([pos_pts, extra], axis=0) if len(pos_pts) > 0 else extra

        if len(pos_pts) > 0:
            pos_pts = np.unique(pos_pts.astype(np.int32), axis=0).astype(np.float32)
            if len(pos_pts) > num_pos:
                pos_pts = pos_pts[:num_pos]

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(m.astype(np.uint8), kernel, iterations=5).astype(bool)
    neg_ring = dilated & (~m)

    if np.count_nonzero(neg_ring) < max(num_neg, 4):
        dilated = cv2.dilate(m.astype(np.uint8), kernel, iterations=10).astype(bool)
        neg_ring = dilated & (~m)

    ys_neg, xs_neg = np.where(neg_ring)
    if len(xs_neg) == 0:
        ys_neg, xs_neg = np.where(~m)

    neg_coords = np.stack([xs_neg, ys_neg], axis=1).astype(np.float32)

    inv_u8 = ((~m).astype(np.uint8) * 255)
    dist_out = cv2.distanceTransform(inv_u8, cv2.DIST_L2, 5)
    neg_q = -dist_out[ys_neg, xs_neg]

    # Reserve part of the negative budget for points placed ON the other object,
    # explicitly telling SAM3 that neighbouring rat is NOT this object. This is
    # what stops the two masks merging when their tails cross at a re-seed.
    other_neg_pts = np.zeros((0, 2), dtype=np.float32)
    n_other = 0
    if other_union is not None and other_union.any():
        n_other = min(max(2, num_neg // 2), num_neg)
        oy, ox = np.where(_erode_bool(other_union, 2))
        if len(ox) == 0:
            oy, ox = np.where(other_union)
        if len(ox) > 0:
            other_coords = np.stack([ox, oy], axis=1).astype(np.float32)
            # spread the other-object negatives across that object
            other_neg_pts = _select_spread(
                other_coords,
                np.ones(len(other_coords), dtype=np.float32),
                min(n_other, len(other_coords)),
                spacing,
            )

    n_ring = max(0, num_neg - len(other_neg_pts))
    ring_neg_pts = _select_spread(
        neg_coords,
        neg_q,
        min(n_ring, len(neg_coords)),
        spacing * 0.8,
    ) if n_ring > 0 else np.zeros((0, 2), dtype=np.float32)

    if len(other_neg_pts) > 0 and len(ring_neg_pts) > 0:
        neg_pts = np.concatenate([other_neg_pts, ring_neg_pts], axis=0)
    elif len(other_neg_pts) > 0:
        neg_pts = other_neg_pts
    else:
        neg_pts = ring_neg_pts

    points = np.concatenate([pos_pts, neg_pts], axis=0).astype(np.float32)
    labels = np.concatenate(
        [
            np.ones(len(pos_pts), dtype=np.int32),
            np.zeros(len(neg_pts), dtype=np.int32),
        ]
    )

    return points, labels


# ============================================================
# Manual review / relabel helpers
# ============================================================

def save_review_overlay(
    frame_rgb: np.ndarray,
    candidate_mask: Optional[np.ndarray],
    review_dir: str,
    global_frame_idx: int,
    obj_id: int,
) -> Tuple[str, Optional[str]]:
    os.makedirs(review_dir, exist_ok=True)

    frame_path = os.path.join(
        review_dir,
        f"frame_{global_frame_idx:06d}_obj_{obj_id}_frame.png",
    )
    Image.fromarray(frame_rgb).save(frame_path)

    overlay_path = None
    if candidate_mask is not None:
        overlay = frame_rgb.copy()
        m = np.asarray(candidate_mask).astype(bool)
        overlay[m] = (
            0.5 * overlay[m] + 0.5 * np.array([255, 255, 0], dtype=np.uint8)
        ).astype(np.uint8)

        overlay_path = os.path.join(
            review_dir,
            f"frame_{global_frame_idx:06d}_obj_{obj_id}_candidate_overlay.png",
        )
        Image.fromarray(overlay).save(overlay_path)

    return frame_path, overlay_path


class _ReviewTally:
    """Lightweight, process-global tally of manual-review events, so an A/B
    run (refiner on vs off) is two summary numbers instead of hand-counting.
    Observes only; never alters review behaviour."""
    def __init__(self):
        self.events = []          # list of dicts
        self.segment_frames = None  # nominal frames per segment, for boundary test
        self._pending_reason = None
        self._append_log_path = None

    def set_segment_frames(self, n):
        self.segment_frames = int(n) if n else None

    def note_reason(self, reason):
        self._pending_reason = reason

    def record(self, obj_id, global_frame_idx, decision):
        reason = self._pending_reason or "unknown"
        # classify reason into a coarse bucket
        r = reason.lower()
        if "area change" in r:
            bucket = "area_change"
        elif "no predicted mask" in r or "disappeared" in r:
            bucket = "absent_no_mask"
        elif "reappear" in r or "previously absent" in r:
            bucket = "reappeared"
        elif "centroid" in r or "jump" in r:
            bucket = "centroid_jump"
        else:
            bucket = "other"
        # segment boundary: global frame lands on a segment start
        at_boundary = False
        if self.segment_frames:
            # first frame of any segment (accounting for 1-frame overlap chains)
            at_boundary = (global_frame_idx % self.segment_frames) in (0, )
        event = {
            "obj_id": int(obj_id),
            "frame": int(global_frame_idx),
            "decision": decision,
            "reason_bucket": bucket,
            "reason": reason,
            "segment_boundary": bool(at_boundary),
        }
        self.events.append(event)
        self._pending_reason = None
        # Write-through: append to a JSONL log the instant the review happens,
        # so an interrupted run still leaves every trigger on disk (the summary
        # JSON is only written at chain end). This is what makes trigger reasons
        # recoverable even when the console log has scrolled away.
        if self._append_log_path is not None:
            try:
                import json as _json
                with open(self._append_log_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(event) + "\n")
            except Exception:
                pass  # never let logging break the run

    def set_append_log(self, path: str) -> None:
        self._append_log_path = path

    def summary_str(self) -> str:
        n = len(self.events)
        if n == 0:
            return "\n=== REVIEW SUMMARY ===\ntotal reviews: 0\n"
        from collections import Counter
        by_reason = Counter(e["reason_bucket"] for e in self.events)
        by_decision = Counter(e["decision"] for e in self.events)
        boundary = sum(1 for e in self.events if e["segment_boundary"])
        mid = n - boundary
        lines = ["\n=== REVIEW SUMMARY ==="]
        lines.append(f"total reviews: {n}")
        lines.append("  by reason:   " + ", ".join(f"{k}={v}" for k, v in by_reason.most_common()))
        lines.append("  by decision: " + ", ".join(f"{k}={v}" for k, v in by_decision.most_common()))
        lines.append(f"  by location: segment_boundary={boundary}, mid_segment={mid}")
        return "\n".join(lines) + "\n"

    def write_json(self, path: str) -> None:
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump({"total": len(self.events), "events": self.events}, f, indent=2)


REVIEW_TALLY = _ReviewTally()


def ask_user_review_decision(
    frame_path: str,
    overlay_path: Optional[str],
    obj_id: int,
    global_frame_idx: int,
) -> str:
    print("\n=== MANUAL REVIEW REQUIRED ===")
    print(f"Object: {obj_id}")
    print(f"Global frame: {global_frame_idx}")
    print(f"Frame image: {frame_path}")
    if overlay_path is not None:
        print(f"Candidate overlay: {overlay_path}")

    print("Decision options:")
    print("  y = accept current candidate mask")
    print("  r = relabel this object on this frame")
    print("  s = keep object absent / skip")

    while True:
        choice = input("Your choice [y/r/s]: ").strip().lower()
        if choice in {"y", "yes"}:
            REVIEW_TALLY.record(obj_id, global_frame_idx, "accept")
            return "accept"
        if choice in {"r", "relabel"}:
            REVIEW_TALLY.record(obj_id, global_frame_idx, "relabel")
            return "relabel"
        if choice in {"s", "skip"}:
            REVIEW_TALLY.record(obj_id, global_frame_idx, "skip")
            return "skip"
        print("Please enter y, r, or s.")


def write_single_frame_video(frame_rgb: np.ndarray, out_path: str, fps: float = 1.0) -> None:
    h, w = frame_rgb.shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for temp frame video: {out_path}")

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)
    writer.release()


def wait_for_file(path: str, timeout_sec: float = 3.0, poll_sec: float = 0.1) -> bool:
    import time

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.isfile(path):
            return True
        time.sleep(poll_sec)

    return os.path.isfile(path)


def cleanup_cuda_after_preview(device: str) -> None:
    if not str(device).startswith("cuda"):
        return

    try:
        import torch
    except Exception:
        return

    try:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def relabel_object_on_frame(
    frame_rgb: np.ndarray,
    candidate_mask: Optional[np.ndarray],
    obj_id: int,
    predictor,
    session_id: str,
    frame_index_in_segment: int,
    image_shape: Tuple[int, int, int],
    review_cfg: dict,
    global_frame_idx: Optional[int] = None,
    sam3_device: str = "auto",
) -> Optional[np.ndarray]:
    from absent_relabel import run_absent_relabel

    return run_absent_relabel(
        frame_rgb=frame_rgb,
        candidate_mask=candidate_mask,
        obj_id=obj_id,
        predictor=predictor,
        session_id=session_id,
        frame_index_in_segment=frame_index_in_segment,
        image_shape=image_shape,
        review_cfg=review_cfg,
        global_frame_idx=global_frame_idx,
        sam3_device=sam3_device,
    )


# ============================================================
# Segment config / input helpers
# ============================================================

def normalize_config(cfg: SegmentChainConfig) -> SegmentChainConfig:
    return SegmentChainConfig(
        video_id=cfg.video_id,
        prompts_json=str(Path(cfg.prompts_json).expanduser()),
        segments_root=str(Path(cfg.segments_root).expanduser()),
        out_root=str(Path(cfg.out_root).expanduser()),
        auto_pos=cfg.auto_pos,
        auto_neg=cfg.auto_neg,
        seed=cfg.seed,
    )


def resolve_inputs(
    cfg: SegmentChainConfig,
) -> Tuple[Path, List[str], Dict[int, Tuple[np.ndarray, np.ndarray]], List[int]]:
    segments_dir = Path(cfg.segments_root).expanduser().resolve()

    if not segments_dir.exists():
        raise FileNotFoundError(f"Segments directory does not exist: {segments_dir}")
    if not segments_dir.is_dir():
        raise NotADirectoryError(f"Segments path is not a directory: {segments_dir}")

    seg_paths = list_segments(str(segments_dir))
    if not seg_paths:
        raise FileNotFoundError(f"No segments found in {segments_dir}")

    prompts_path = Path(cfg.prompts_json).expanduser().resolve()
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts JSON not found: {prompts_path}")

    prompts_seg1 = load_prompts_json(str(prompts_path))
    obj_ids_sorted = sorted(int(k) for k in prompts_seg1.keys())

    print(f"[resolve_inputs] segments_dir: {segments_dir}")
    print(f"[resolve_inputs] found {len(seg_paths)} segment(s):")
    for i, path in enumerate(seg_paths, start=1):
        print(f"  [{i}] {path}")

    return segments_dir, seg_paths, prompts_seg1, obj_ids_sorted


def infer_video_properties(first_segment_path: str) -> Tuple[int, int, float]:
    vr0 = VideoReader(str(first_segment_path), ctx=cpu(0))
    h, w = vr0[0].shape[:2]
    fps = float(vr0.get_avg_fps())
    return h, w, fps


def load_segments_manifest(segments_dir: str) -> Optional[Dict[str, int]]:
    """
    Read segments_manifest.json (written by fragment_video.py) mapping
    segment filename -> frame count. Returns None if absent/invalid.
    """
    manifest_path = Path(segments_dir).expanduser().resolve() / "segments_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("segment_frames")
        if isinstance(frames, dict) and frames:
            return {str(k): int(v) for k, v in frames.items()}
    except Exception as e:
        print(f"[load_segments_manifest] Ignoring invalid manifest: {e}")
    return None


# ============================================================
# Candidate processing / review integration
# ============================================================

def process_candidates_with_review(
    candidates: Dict[int, np.ndarray],
    obj_ids_sorted: List[int],
    track_states,
    object_memories: Dict[int, ObjectMemory],
    absent_handler,
    image_shape: Tuple[int, ...],
    frame_idx_global: int,
    frame_rgb: np.ndarray,
    predictor,
    session_id: str,
    frame_index_in_segment: int,
    review_cfg: dict,
    sam3_device: str = "auto",
    relabel_previous_frame: bool = False,
    previous_frame_rgb: Optional[np.ndarray] = None,
    previous_frame_idx_global: Optional[int] = None,
    previous_frame_index_in_segment: Optional[int] = None,
    previous_visible_masks: Optional[Dict[int, np.ndarray]] = None,
    segment_boundary: bool = False,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[str, Any]]]:
    visible_masks: Dict[int, np.ndarray] = {}
    review_dir = review_cfg.get("review_dir", "data/manual_review")
    rewrite_previous: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # Cross-object merge detection.
    # Two visually identical rats in contact can cause SAM3 to collapse both
    # into ONE identity (obj_1's mask covers both rats; obj_2 vanishes or shrinks
    # to a sliver). This is a fundamental limit of appearance+proximity tracking
    # and cannot be prevented from the pipeline - but it CAN be detected and
    # flagged, so corrupted two-rat masks are never silently written into the
    # data used for behaviour analysis.
    #
    # Signature: two objects' candidate masks overlap heavily (high IoU or one
    # largely contained in the other), OR one object collapsed to near-zero
    # while another is abnormally large. When detected, mark the involved
    # objects for review instead of writing the merged output.
    merged_flag_ids = set()
    try:
        merge_iou_thresh = float(review_cfg.get("merge_iou_thresh", 0.5))
        merge_contain_thresh = float(review_cfg.get("merge_contain_thresh", 0.8))
        present = {
            oid: np.asarray(candidates[oid]).astype(bool)
            for oid in obj_ids_sorted
            if candidates.get(oid) is not None and int(np.asarray(candidates[oid]).sum()) > 0
        }
        oids_present = list(present.keys())
        for i in range(len(oids_present)):
            for j in range(i + 1, len(oids_present)):
                a_id, b_id = oids_present[i], oids_present[j]
                a, b = present[a_id], present[b_id]
                inter = int(np.logical_and(a, b).sum())
                if inter == 0:
                    continue
                union = int(np.logical_or(a, b).sum())
                iou = inter / union if union > 0 else 0.0
                a_area = int(a.sum())
                b_area = int(b.sum())
                # containment: how much of the smaller sits inside the larger
                smaller = min(a_area, b_area)
                contain = inter / smaller if smaller > 0 else 0.0
                if iou >= merge_iou_thresh or contain >= merge_contain_thresh:
                    # the two masks are occupying substantially the same region
                    merged_flag_ids.add(a_id)
                    merged_flag_ids.add(b_id)
        if merged_flag_ids:
            print(
                f"[frame={frame_idx_global}] MERGE DETECTED: objects "
                f"{sorted(merged_flag_ids)} masks overlap heavily (identical "
                f"rats in contact). Flagging for review; not writing merged masks."
            )
    except Exception as _e:
        print(f"[frame={frame_idx_global}] merge-detection skipped ({_e})")

    for oid in obj_ids_sorted:
        candidate_mask = candidates.get(oid)

        accepted_mask, new_state, decision = absent_handler.update(
            state=track_states[oid],
            predicted_mask=candidate_mask,
            image_shape=image_shape,
            frame_idx=frame_idx_global,
            reinit_mask=None,
            segment_boundary=segment_boundary,
        )
        track_states[oid] = new_state

        # Force a review if this object was implicated in a cross-object merge,
        # even if the tracker's own per-object checks would have accepted it.
        if oid in merged_flag_ids and not decision.needs_user_relabel:
            decision.needs_user_relabel = True
            decision.reason = (
                "cross-object merge detected (identical rats in contact); "
                "masks occupy the same region - manual review required"
            )

        if decision.needs_user_relabel:
            REVIEW_TALLY.note_reason(getattr(decision, "reason", "unknown"))
            use_previous = (
                relabel_previous_frame
                and previous_frame_rgb is not None
                and previous_frame_index_in_segment is not None
                and frame_index_in_segment > 0
            )

            target_frame_rgb = previous_frame_rgb if use_previous else frame_rgb
            target_frame_idx_global = (
                previous_frame_idx_global if use_previous else frame_idx_global
            )
            target_frame_index_in_segment = (
                previous_frame_index_in_segment if use_previous else frame_index_in_segment
            )
            target_candidate_mask = None
            if use_previous and previous_visible_masks is not None:
                target_candidate_mask = previous_visible_masks.get(oid)
            if target_candidate_mask is None:
                target_candidate_mask = candidate_mask

            frame_path, overlay_path = save_review_overlay(
                frame_rgb=target_frame_rgb,
                candidate_mask=target_candidate_mask,
                review_dir=review_dir,
                global_frame_idx=target_frame_idx_global if target_frame_idx_global is not None else frame_idx_global,
                obj_id=oid,
            )

            action = ask_user_review_decision(
                frame_path=frame_path,
                overlay_path=overlay_path,
                obj_id=oid,
                global_frame_idx=target_frame_idx_global if target_frame_idx_global is not None else frame_idx_global,
            )

            if action == "accept":
                accepted_for_reinit = candidate_mask
                if accepted_for_reinit is not None:
                    accepted_mask, new_state, _ = absent_handler.update(
                        state=track_states[oid],
                        predicted_mask=None,
                        image_shape=image_shape,
                        frame_idx=frame_idx_global,
                        reinit_mask=accepted_for_reinit,
                    )
                    track_states[oid] = new_state
                else:
                    accepted_mask = None

            elif action == "relabel":
                manual_mask = relabel_object_on_frame(
                    frame_rgb=target_frame_rgb,
                    candidate_mask=target_candidate_mask,
                    obj_id=oid,
                    predictor=predictor,
                    session_id=session_id,
                    frame_index_in_segment=target_frame_index_in_segment,
                    image_shape=image_shape,
                    review_cfg=review_cfg,
                    global_frame_idx=target_frame_idx_global,
                    sam3_device=sam3_device,
                )

                if manual_mask is not None:
                    accepted_mask, new_state, _ = absent_handler.update(
                        state=track_states[oid],
                        predicted_mask=None,
                        image_shape=image_shape,
                        frame_idx=frame_idx_global,
                        reinit_mask=manual_mask,
                    )
                    track_states[oid] = new_state

                    memory_frame_rgb = target_frame_rgb if use_previous else frame_rgb
                    object_memories[oid].update(
                        frame_rgb=memory_frame_rgb,
                        mask=np.asarray(manual_mask).astype(bool),
                        weight=3,
                    )

                    if use_previous and previous_frame_idx_global is not None:
                        updated_previous_masks = {
                            int(k): np.asarray(v).astype(bool).copy()
                            for k, v in (previous_visible_masks or {}).items()
                        }
                        updated_previous_masks[oid] = np.asarray(manual_mask).astype(bool)
                        rewrite_previous = {
                            "frame_rgb": previous_frame_rgb,
                            "frame_idx_global": previous_frame_idx_global,
                            "masks": updated_previous_masks,
                        }
                else:
                    accepted_mask = None

            elif action == "skip":
                accepted_mask = None

        if accepted_mask is not None:
            visible_masks[oid] = np.asarray(accepted_mask).astype(bool)
            object_memories[oid].update(
                frame_rgb=frame_rgb,
                mask=np.asarray(accepted_mask).astype(bool),
                weight=1,
            )

    return visible_masks, rewrite_previous


# ============================================================
# Live relabel preview
# ============================================================

def make_live_relabel_preview_updater(
    predictor,
    session_id: str,
    frame_index_in_segment: int,
    obj_id: int,
    image_shape: Tuple[int, int, int],
):
    h, w = image_shape[:2]

    def _update_preview(state) -> None:
        obj = state.objects[str(obj_id)]
        points_xy = obj.get("points_xy", [])
        labels = obj.get("labels", [])

        if not points_xy:
            state.predicted_masks[str(obj_id)] = None
            state.mask_version += 1
            return

        points_abs = np.asarray(points_xy, dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.int32)
        points_rel = abs_to_rel(points_abs, w, h)

        resp = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index_in_segment,
                "points": points_rel.tolist(),
                "point_labels": labels_np.astype(int).tolist(),
                "obj_id": int(obj_id),
                "rel_coordinates": True,
            }
        )

        outputs = resp.get("outputs")
        if outputs is None:
            state.predicted_masks[str(obj_id)] = None
            state.mask_version += 1
            return

        candidates = parse_candidate_masks_from_outputs(outputs, [obj_id])
        state.predicted_masks[str(obj_id)] = candidates.get(obj_id)
        state.mask_version += 1

    return _update_preview


# ============================================================
# Prompt handoff between segments
# ============================================================

def build_prompts_for_segment(
    segment_index: int,
    prompts_seg1: Dict[int, Tuple[np.ndarray, np.ndarray]],
    last_masks: Optional[Dict[int, np.ndarray]],
    obj_ids_sorted: List[int],
    auto_pos: int,
    auto_neg: int,
    seed: int,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], bool]:
    if segment_index == 0:
        return prompts_seg1, False

    if last_masks is None:
        raise RuntimeError("Previous segment did not produce last_masks.")

    prompts: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for oid in obj_ids_sorted:
        mask = last_masks.get(oid)
        if mask is None:
            continue

        pts_abs, labs = sample_points_from_mask(
            mask,
            num_pos=auto_pos,
            num_neg=auto_neg,
            seed=seed + 1000 * segment_index + oid,
        )
        prompts[oid] = (pts_abs, labs)

    return prompts, True
