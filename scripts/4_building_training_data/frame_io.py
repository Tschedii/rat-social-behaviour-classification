"""
Shared frame I/O helpers for the rat-behaviour H5 pipeline.

Assumes each rgb/mask path from training_manifest.csv is a directory of
per-frame images named like `frame_000105.png` (confirmed against the local
Day_14_..._locomotion_frames export: files are `frame_%06d.png`, frame number
== the same frame index used in the label CSVs' start_frame/end_frame, and
also == the `key` column in training_metadata/*/cam_*/transform_meta.csv).

Frame numbering is NOT assumed to start at 0 or be contiguous -- some
exports only contain a trimmed range (e.g. the local Day_14 export starts at
frame 000105, not 000000, exactly matching that experiment's
transform_meta.csv). read_frame_range() clips requested ranges to the
frames that actually exist on disk and reports what was clipped, rather than
silently zero-padding or crashing.
"""

import os
import re
import glob
import numpy as np
import cv2

FRAME_RE = re.compile(r"(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


class FrameDir:
    """Indexes a directory of frame_XXXXXX.png files by integer frame number."""

    def __init__(self, path):
        self.path = path
        self._index = {}  # frame_number -> filepath
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Not a directory: {path}")
        for fp in glob.glob(os.path.join(path, "*")):
            m = FRAME_RE.search(os.path.basename(fp))
            if m:
                self._index[int(m.group(1))] = fp
        if not self._index:
            raise FileNotFoundError(f"No frame_XXXXXX.<ext> files found in {path}")

    def available_range(self):
        keys = self._index.keys()
        return min(keys), max(keys)

    def available_count(self):
        return len(self._index)

    def has(self, frame_no):
        return frame_no in self._index

    def read(self, frame_no, color=True):
        fp = self._index[frame_no]
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(fp, flag)
        if img is None:
            raise IOError(f"Failed to read {fp}")
        if color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def read_bytes(self, frame_no):
        """Raw on-disk file bytes for this frame, undecoded (for lossless PNG passthrough)."""
        fp = self._index[frame_no]
        with open(fp, "rb") as f:
            return f.read()

    def file_size(self, frame_no):
        return os.path.getsize(self._index[frame_no])

    def read_range(self, start, end, color=True, resize=None, stride=1):
        """
        Read frames in [start, end) (end-exclusive, matching how the label
        CSVs were generated: end_frame = round(end_seconds * fps)).

        Clips [start, end) to the frames actually present. Returns
        (frames_array, actual_start, actual_end, missing_frame_count).
        frames_array has shape (T, H, W, C) for rgb or (T, H, W) for masks.
        """
        lo, hi = self.available_range()
        clipped_start = max(start, lo)
        clipped_end = min(end, hi + 1)

        if clipped_start >= clipped_end:
            # requested range doesn't overlap what's on disk at all
            return None, clipped_start, clipped_start, max(end - start, 0)

        frame_nos = list(range(clipped_start, clipped_end, stride))
        frames = []
        missing = 0
        for fn in frame_nos:
            if not self.has(fn):
                missing += 1
                continue
            img = self.read(fn, color=color)
            if resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            frames.append(img)

        if not frames:
            return None, clipped_start, clipped_end, missing

        return np.stack(frames, axis=0), clipped_start, clipped_end, missing

    def sample_frames(self, n, rng):
        """Uniformly sample n frame numbers (without replacement) for stats."""
        keys = sorted(self._index.keys())
        if n >= len(keys):
            return keys
        idx = rng.choice(len(keys), size=n, replace=False)
        return [keys[i] for i in sorted(idx)]


def uniform_subsample_range(start, end, n_frames):
    """Return n_frames evenly spaced integer frame numbers in [start, end)."""
    total = end - start
    if total <= 0:
        return []
    if n_frames >= total:
        return list(range(start, end))
    return sorted(set(int(round(start + i * (total - 1) / (n_frames - 1))) for i in range(n_frames)))


def fixed_length_indices(start, end, n_frames):
    """
    Return EXACTLY n_frames integer frame numbers spanning [start, end),
    for use when the model needs a fixed clip length T for batching.

    - If the bout has >= n_frames real frames, this is a uniform sparse
      sample (same spirit as uniform_subsample_range, but guaranteed to
      return exactly n_frames even when rounding would otherwise collapse
      duplicates near the boundary).
    - If the bout is SHORTER than n_frames (common here -- ~28% of bouts
      are already <=16 frames, some as short as 1), frames are repeated
      (held) to fill the length, not zero-padded. A held frame is a
      truthful representation of "nothing changed here"; a zero frame is
      not, and would train the model on a fabricated black frame.

    Returns a list of n_frames frame numbers, monotonically non-decreasing,
    with `is_padded = True` (see build_h5_dataset.py) whenever end-start < n_frames.

    NOTE: this is a pure function of (start, end, n_frames) -- it does NOT
    know which frames actually exist on disk. build_h5_dataset.py clips the
    result against each camera's real available range and records the exact
    surviving frame numbers in /data/<seg>/cam_N/frame_id, so downstream code
    (e.g. joining against training_metadata/*/cam_*/transform_meta.csv) never
    has to re-derive this list by hand.
    """
    total = end - start
    if total <= 0 or n_frames <= 0:
        return []
    if n_frames == 1:
        return [start]
    idx = np.linspace(start, end - 1, num=n_frames)
    return np.round(idx).astype(int).tolist()