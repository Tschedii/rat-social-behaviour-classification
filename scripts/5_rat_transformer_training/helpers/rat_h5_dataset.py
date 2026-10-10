"""PyTorch Dataset reading bout windows directly out of rat_behaviour_32f.h5,
using the manifest produced by helpers/build_index.py.

Each sample is one bout, temporally fixed at T=32 frames (by construction of
the .h5), with `num_cams` cameras selected per bout (top-K by precomputed
mask visibility for eval, weighted-random for train so the model doesn't
overfit to always seeing the single best camera).
"""
import io
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CAMS = [f"cam_{i}" for i in range(1, 7)]
SEQ_LEN = 32  # canonical window length; the .h5's own name promises this, but ~2% of bouts
              # have one or more cameras stored with a different frame count (seen: 7-31
              # instead of 32, and one bout with all 6 cams at 10) — resample defensively.


def _decode_png(raw_bytes, size, resample):
    im = Image.open(io.BytesIO(raw_bytes.tobytes()))
    if im.size != (size, size):
        im = im.resize((size, size), resample)
    return np.asarray(im)


def _decode_gray(raw_bytes, size):
    """Decode straight to grayscale at `size` — used only for optical flow, which is
    inherently low-frequency/smooth (unlike RGB texture), so computing it at a much smaller
    resolution than the native 1024px RGB is a deliberate, justified exception: there's no
    fine detail in a flow field to lose by downsampling, unlike in the RGB stream."""
    im = Image.open(io.BytesIO(raw_bytes.tobytes())).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def _tokenize_session(s):
    return tuple(sorted(s.lower().split("_")))


def _resolve_session_dir(transform_meta_root, session_id, _cache={}):
    """session_id spelling can differ between window_index.csv (inherited from
    train_eval_split.csv) and the actual folder names under transform_meta_root — the same
    word-order-swap quirk build_index.py already handles for the .h5 join (observed: e.g.
    "Unfamiliar_rat_for_calibration_..." vs the real folder "RAT_unfamiliar_for_calibration_...").
    Resolve by exact match first, falling back to the same token-set match build_index.py
    uses, so a scale lookup for one of these sessions doesn't silently fall through to the
    neutral default instead of its real value. Returns session_id unchanged if nothing
    resolves (the caller's existing "file not found" handling still applies)."""
    root = Path(transform_meta_root)
    if (root / session_id).is_dir():
        return session_id
    key = str(root)
    if key not in _cache:
        _cache[key] = {
            _tokenize_session(d.name): d.name for d in root.iterdir() if d.is_dir()
        }
    return _cache[key].get(_tokenize_session(session_id), session_id)


def compute_log_scale_stats(transform_meta_root, session_ids):
    """Global mean/std of log(scale) across every transform_meta.csv for the given sessions,
    read directly from the CSVs (not from bouts) since that's cheap and gives every frame equal
    weight regardless of how bouts happen to be split. `scale` = 1024 / max(bbox_w, bbox_h), the
    per-frame zoom factor the upstream tracker used to register each frame to a common size —
    feeding it (as a normalized scalar) lets the model know how zoomed-in a given frame was,
    since that's real information the pipeline doesn't otherwise preserve (see README/model.py:
    the crop-then-uniform-scale keeps shape/aspect ratio intact, but not absolute apparent size).
    Call once, before constructing datasets, and pass the same (mean, std) to both splits."""
    root = Path(transform_meta_root)
    vals = []
    missing = []
    for session_id in sorted(set(session_ids)):
        resolved = _resolve_session_dir(root, session_id)
        for cam in CAMS:
            p = root / resolved / cam / "transform_meta.csv"
            if not p.exists():
                missing.append(str(p))
                continue
            vals.append(np.log(pd.read_csv(p, usecols=["scale"])["scale"].to_numpy(dtype=np.float64)))
    if missing:
        print(f"compute_log_scale_stats: {len(missing)} transform_meta.csv files not found "
              f"(first few: {missing[:3]}) — proceeding with what's available.")
    if not vals:
        raise ValueError(f"no transform_meta.csv files found under {transform_meta_root} "
                          f"for sessions={sorted(set(session_ids))[:5]}...")
    all_vals = np.concatenate(vals)
    return float(all_vals.mean()), float(all_vals.std() + 1e-6)


def _resample_indices(n, target=SEQ_LEN):
    """Map `target` output positions onto `n` available source frames via nearest-neighbor
    stretching — the same style of resampling the .h5 already uses elsewhere for bouts
    shorter than 32 frames (confirmed empirically: a 13-frame bout's frame_id reads
    [0,0,1,1,2,2,...]). Identity when n == target, so this is a no-op for the ~98% of
    (bout, camera) pairs that already have exactly 32 frames."""
    if n == target:
        return np.arange(n)
    return np.minimum((np.arange(target) * n) // target, n - 1)


class RatBoutDataset(Dataset):
    def __init__(
        self,
        index_csv,
        h5_path,
        normstats_path,
        split,
        num_cams=3,
        frame_size=1024,  # native .h5 resolution; _decode_png only resizes if this differs
        cam_selection="topk",  # "topk" (deterministic) or "weighted_random" (train-time aug)
        augment=False,
        duration_stats=None,  # (mean, std) of log1p(duration_frames); computed from train if None
        compute_flow=False,
        flow_size=256,
        flow_scale=20.0,  # pixels; flow is divided by this and clipped to [-1,1]
        exclude_label_ids=None,  # original label_ids to drop entirely (e.g. a class being descoped)
        label_id_remap=None,  # {original_label_id: new_label_id}, applied after excluding rows
        transform_meta_root=None,  # dir of <session_id>/cam_N/transform_meta.csv (see build_h5 docstring)
        compute_scale=False,
        log_scale_stats=None,  # (mean, std) of log(scale); computed from train if None
    ):
        df = pd.read_csv(index_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no rows with split={split!r} in {index_csv}")

        if exclude_label_ids:
            self.df = self.df[~self.df["label_id"].isin(exclude_label_ids)].reset_index(drop=True)
            if len(self.df) == 0:
                raise ValueError(f"excluding label_ids={exclude_label_ids} left no rows for split={split!r}")
        if label_id_remap:
            self.df["label_id"] = self.df["label_id"].map(label_id_remap)
            if self.df["label_id"].isna().any():
                raise ValueError("label_id_remap is missing an entry for a label_id present in the data")
            self.df["label_id"] = self.df["label_id"].astype(int)

        self.h5_path = str(h5_path)
        self.h5 = None  # opened lazily per worker process
        self.num_cams = num_cams
        self.frame_size = frame_size
        self.cam_selection = cam_selection
        self.augment = augment
        self.compute_flow = compute_flow
        self.flow_size = flow_size
        self.flow_scale = flow_scale
        self._flow_calc = None  # lazy, created once per worker process (see _ensure_open)
        self.transform_meta_root = transform_meta_root
        self.compute_scale = compute_scale
        self._scale_table_cache = {}  # (session_id, cam) -> {frame_no: scale}, built on demand
        if compute_scale:
            if log_scale_stats is None:
                raise ValueError("compute_scale=True requires log_scale_stats — compute once via "
                                  "compute_log_scale_stats() and pass the same value to both splits")
            self.log_scale_mean, self.log_scale_std = log_scale_stats

        normstats = json.loads(Path(normstats_path).read_text())
        self.per_cam_mean = {
            f"cam_{i}": np.array(normstats["per_camera"][f"cam_{i}"]["mean"], dtype=np.float32)
            for i in range(1, 7)
        }
        self.per_cam_std = {
            f"cam_{i}": np.array(normstats["per_camera"][f"cam_{i}"]["std"], dtype=np.float32)
            for i in range(1, 7)
        }

        log_dur = np.log1p(self.df["duration_frames"].to_numpy(dtype=np.float32))
        if duration_stats is None:
            self.duration_mean, self.duration_std = float(log_dur.mean()), float(log_dur.std() + 1e-6)
        else:
            self.duration_mean, self.duration_std = duration_stats
        self._log_dur = (log_dur - self.duration_mean) / self.duration_std

    def duration_stats(self):
        """Call on the train split, pass result to eval split ctor so both use the same stats."""
        return (self.duration_mean, self.duration_std)

    def __len__(self):
        return len(self.df)

    def _ensure_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r", swmr=True)
        if self.compute_flow and self._flow_calc is None:
            import cv2
            self._flow_calc = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def _flow_sequence(self, raw_bytes_arr, src_idx, flip):
        """Returns (SEQ_LEN, 2, flow_size, flow_size) float32: flow[0] is zeros (no prior
        frame to compare against), flow[t] is the motion from frame t-1 to frame t for
        t=1..SEQ_LEN-1, using the same resampled frame indices as the RGB/mask stream so the
        two streams stay aligned."""
        size = self.flow_size
        gray = [_decode_gray(raw_bytes_arr[src_t], size) for src_t in src_idx]
        if flip:
            gray = [g[:, ::-1].copy() for g in gray]
        flow_seq = np.zeros((SEQ_LEN, 2, size, size), dtype=np.float32)
        for t in range(1, SEQ_LEN):
            flow = self._flow_calc.calc(gray[t - 1], gray[t], None)  # (H, W, 2)
            flow_seq[t] = flow.transpose(2, 0, 1)
        flow_seq = np.clip(flow_seq / self.flow_scale, -1.0, 1.0)
        return flow_seq

    def _scale_table(self, session_id, cam):
        key = (session_id, cam)
        if key not in self._scale_table_cache:
            resolved = _resolve_session_dir(self.transform_meta_root, session_id)
            p = Path(self.transform_meta_root) / resolved / cam / "transform_meta.csv"
            if p.exists():
                df = pd.read_csv(p, usecols=["key", "scale"])
                frame_no = df["key"].str.replace("frame_", "", regex=False).astype(int)
                self._scale_table_cache[key] = dict(zip(frame_no, df["scale"]))
            else:
                self._scale_table_cache[key] = {}  # missing file -> every lookup falls back below
        return self._scale_table_cache[key]

    def _scale_sequence(self, session_id, cam, frame_ids):
        """Returns (SEQ_LEN,) float32: standardized log(scale) for each sampled frame, looked
        up by original frame number in transform_meta.csv. Missing frames (file absent, or that
        exact frame_id not tracked) fall back to scale=1.0 (log=0, i.e. the dataset mean after
        standardization is close to but not exactly 0 — an untracked-but-plausible neutral)."""
        table = self._scale_table(session_id, cam)
        scale = np.array([table.get(int(fn), 1.0) for fn in frame_ids], dtype=np.float64)
        log_scale = (np.log(scale) - self.log_scale_mean) / self.log_scale_std
        return log_scale.astype(np.float32)

    def _select_cams(self, row, rng):
        vis = np.array([row[f"{c}_vis"] for c in CAMS], dtype=np.float32)
        if self.cam_selection == "topk" or not self.augment:
            order = np.argsort(-vis)
            return [CAMS[i] for i in order[: self.num_cams]], vis[order[: self.num_cams]]
        # weighted random selection without replacement, biased towards visible cams
        weights = vis + 0.05
        weights = weights / weights.sum()
        chosen = rng.choice(len(CAMS), size=self.num_cams, replace=False, p=weights)
        return [CAMS[i] for i in chosen], vis[chosen]

    def __getitem__(self, idx):
        self._ensure_open()
        row = self.df.iloc[idx]
        rng = np.random.default_rng()
        cams, cam_vis = self._select_cams(row, rng)

        flip = self.augment and rng.random() < 0.5
        brightness = 1.0 + (rng.uniform(-0.1, 0.1) if self.augment else 0.0)

        grp = self.h5["data"][row["h5_key"]]
        cam_tensors = []
        flow_tensors = []
        scale_tensors = []
        for cam in cams:
            cam_grp = grp[cam]
            rgb_raw = cam_grp["rgb"][:]
            mask_raw = cam_grp["mask"][:]
            src_idx = _resample_indices(len(rgb_raw))  # handles the ~2% of (bout,cam) pairs
            frames = np.empty((SEQ_LEN, 4, self.frame_size, self.frame_size), dtype=np.float32)
            mean, std = self.per_cam_mean[cam], self.per_cam_std[cam]
            for t, src_t in enumerate(src_idx):
                rgb = _decode_png(rgb_raw[src_t], self.frame_size, Image.BILINEAR).astype(np.float32) / 255.0
                mask = _decode_png(mask_raw[src_t], self.frame_size, Image.NEAREST).astype(np.float32) / 255.0
                if flip:
                    rgb = rgb[:, ::-1, :].copy()
                    mask = mask[:, ::-1].copy()
                rgb = np.clip(rgb * brightness, 0.0, 1.0)
                rgb = (rgb - mean) / std
                frames[t, :3] = rgb.transpose(2, 0, 1)
                frames[t, 3] = mask
            cam_tensors.append(frames)
            if self.compute_flow:
                flow_tensors.append(self._flow_sequence(rgb_raw, src_idx, flip))
            if self.compute_scale:
                frame_ids = cam_grp["frame_id"][:][src_idx]  # original frame numbers, same order as rgb/mask
                scale_tensors.append(self._scale_sequence(row["session_id"], cam, frame_ids))

        images = torch.from_numpy(np.stack(cam_tensors, axis=0))  # (num_cams, T, 4, H, W)
        cam_conf = torch.from_numpy(cam_vis.astype(np.float32))
        duration = torch.tensor([self._log_dur[idx]], dtype=torch.float32)
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        sample = {
            "images": images,
            "cam_conf": cam_conf,
            "duration": duration,
            "label": label,
        }
        if self.compute_flow:
            sample["flow"] = torch.from_numpy(np.stack(flow_tensors, axis=0))  # (num_cams, T, 2, Hf, Wf)
        if self.compute_scale:
            sample["scale"] = torch.from_numpy(np.stack(scale_tensors, axis=0))  # (num_cams, T)
        return sample
