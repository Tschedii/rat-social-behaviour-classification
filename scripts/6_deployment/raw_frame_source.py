"""Reads raw per-camera frame folders for deployment inference — the same
frame_XXXXXX.<ext> convention as 4_building_training_data/frame_io.py's
FrameDir, deliberately kept in sync with it (vendored here rather than
imported across the numbered-folder boundary, since 6_deployment/ should be
scp-able as a self-contained unit).

Also replicates two small pieces of existing pipeline logic exactly, so
deployment-time features are computed identically to how they were at
training time:
  - camera-visibility scoring: helpers/build_index.py's decode_mask_coverage
    (96x96 nearest-downsize, fraction of non-zero pixels)
  - the scale (registration zoom factor) lookup: helpers/rat_h5_dataset.py's
    _scale_table (transform_meta.csv, keyed by "frame_NNNNNN" -> scale)
"""
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image

FRAME_RE = re.compile(r"(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


class FrameDir:
    """Indexes a directory of frame_XXXXXX.<ext> files by integer frame number."""

    def __init__(self, path):
        self.path = str(path)
        self._index = {}
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"Not a directory: {self.path}")
        for fp in glob.glob(os.path.join(self.path, "*")):
            m = FRAME_RE.search(os.path.basename(fp))
            if m:
                self._index[int(m.group(1))] = fp
        if not self._index:
            raise FileNotFoundError(f"No frame_XXXXXX.<ext> files found in {self.path}")

    def available_range(self):
        keys = self._index.keys()
        return min(keys), max(keys)

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


def read_or_hold(frame_dir, fn, window_frame_nos, color):
    """Reads frame `fn` from `frame_dir`; if that exact frame is missing (a small gap in an
    otherwise-continuous recording), holds the nearest available frame within the current
    window instead of crashing — the same "repeat rather than fabricate" philosophy
    rat_h5_dataset.py's _resample_indices already uses for short/gappy bouts."""
    if frame_dir.has(fn):
        return frame_dir.read(fn, color=color)
    for candidate in sorted(window_frame_nos, key=lambda x: abs(x - fn)):
        if frame_dir.has(candidate):
            return frame_dir.read(candidate, color=color)
    lo, hi = frame_dir.available_range()
    nearest = min(max(fn, lo), hi)
    return frame_dir.read(nearest, color=color)


def compute_visibility(mask_frames, thumb=96):
    """mask_frames: list of grayscale uint8 arrays. Returns mean fraction of non-zero pixels
    after downsizing each to thumb x thumb (nearest) — exactly helpers/build_index.py's
    decode_mask_coverage, so deployment-time camera ranking matches training-time semantics."""
    if not mask_frames:
        return 0.0
    fracs = []
    for arr in mask_frames:
        im = Image.fromarray(arr).resize((thumb, thumb), Image.NEAREST)
        fracs.append(float((np.asarray(im) > 0).mean()))
    return float(np.mean(fracs))


def read_transform_meta(path):
    """path: a transform_meta.csv (columns "key", "scale"; key = "frame_NNNNNN"). Returns
    {frame_number: scale}. Mirrors rat_h5_dataset.py's _scale_table exactly."""
    df = pd.read_csv(path, usecols=["key", "scale"])
    frame_no = df["key"].str.replace("frame_", "", regex=False).astype(int)
    return dict(zip(frame_no, df["scale"]))
