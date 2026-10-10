"""Deployment inference: turns 6 cameras' worth of raw frame folders into a per-frame
behavior-label JSON, using a trained checkpoint.

Deliberately reuses helpers/checkpoint_utils.py's build_model() and
helpers/rat_h5_dataset.py's compute_log_scale_stats() rather than reconstructing the model
or its normalization from scratch — those are the exact functions evaluate.py and
calibrate.py already use to rebuild a checkpoint correctly, so this stays in lockstep with
the training pipeline instead of drifting from it.

Per-camera raw layout expected (configurable via `subdirs`):
    <camera_dir>/cutout_all_objects/frame_NNNNNN.png   -- RGB, all objects visible
    <camera_dir>/mask_object_1/frame_NNNNNN.png         -- binary mask, rat 1
    <camera_dir>/mask_object_2/frame_NNNNNN.png         -- binary mask, rat 2
    <camera_dir>/transform_meta.csv                     -- optional; only read if the loaded
                                                             checkpoint uses the scale feature

Windowing: the model only ever saw fixed 32-frame bouts, so a continuous recording is cut
into consecutive SEQ_LEN=32-frame windows starting every `stride` frames (default: stride ==
SEQ_LEN, i.e. non-overlapping -- this matches the training data's own bout structure most
closely). A smaller stride gives smoother frame-level label boundaries at proportionally
higher inference cost; where windows overlap, each frame keeps whichever covering window's
prediction had the highest softmax confidence.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# This script lives in 6_deployment/, a SIBLING of 5_rat_transformer_training/ (not a
# subfolder of it) — model.py/helpers/train.py are one level up, then across into that
# sibling directory, not straight up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent / "5_rat_transformer_training"
sys.path.insert(0, str(PROJECT_ROOT))
from helpers.checkpoint_utils import build_model, load_checkpoint  # noqa: E402
from helpers.rat_h5_dataset import compute_log_scale_stats  # noqa: E402
from train import pick_device  # noqa: E402

from raw_frame_source import FrameDir, compute_visibility, read_or_hold, read_transform_meta  # noqa: E402

SEQ_LEN = 32
NUM_VIS_SAMPLES = 8

DEFAULT_SUBDIRS = {
    "rgb": "cutout_all_objects",
    1: "mask_object_1",
    2: "mask_object_2",
}
TRANSFORM_META_NAME = "transform_meta.csv"


class SessionInferer:
    def __init__(self, checkpoint_path, window_index_csv=None, normstats_path=None,
                 device=None, subdirs=None, progress_log=print, transform_meta_root=None):
        """transform_meta_root: folder of the TRAINING sessions' transform metadata, used to
        compute the scale feature's normalization stats. Overrides the path recorded in the
        checkpoint (train_args["transform_meta_root"]), for when that folder has moved since
        training. Ignored for checkpoints trained without the scale feature."""
        self.checkpoint_path = str(checkpoint_path)
        self.subdirs = subdirs or DEFAULT_SUBDIRS
        self.log = progress_log
        self.device = device or pick_device()

        self.ckpt, self.train_args, self.label_map, self.idx_to_label = load_checkpoint(
            self.checkpoint_path, self.device
        )
        if self.train_args["num_cams"] != 6:
            raise ValueError(
                f"checkpoint was trained with num_cams={self.train_args['num_cams']}, but this "
                f"deployment tool assumes 6 cameras (matching the project's current setup)."
            )
        self.model = build_model(self.train_args, self.label_map, self.device)
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()

        self.frame_size = self.train_args["frame_size"]
        self.use_flow = self.train_args.get("use_flow", False)
        self.use_scale = self.train_args.get("use_scale_feature", False)
        self.flow_size = self.train_args.get("flow_size", 256)
        self.flow_scale = self.train_args.get("flow_scale", 20.0)
        self._flow_calc = None
        if self.use_flow:
            import cv2
            self._flow_calc = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        normstats_path = normstats_path or (PROJECT_ROOT / "rgb_normstats.json")
        normstats = json.loads(Path(normstats_path).read_text())
        self.per_cam_mean = {
            f"cam_{i}": np.array(normstats["per_camera"][f"cam_{i}"]["mean"], dtype=np.float32)
            for i in range(1, 7)
        }
        self.per_cam_std = {
            f"cam_{i}": np.array(normstats["per_camera"][f"cam_{i}"]["std"], dtype=np.float32)
            for i in range(1, 7)
        }

        window_index_csv = window_index_csv or (PROJECT_ROOT / "window_index.csv")
        df = pd.read_csv(window_index_csv)
        train_df = df[df["split"] == "train"]
        if len(train_df) == 0:
            raise ValueError(f"no split=='train' rows in {window_index_csv}")
        log_dur = np.log1p(train_df["duration_frames"].to_numpy(dtype=np.float32))
        self.duration_mean, self.duration_std = float(log_dur.mean()), float(log_dur.std() + 1e-6)

        self.log_scale_stats = None
        self.transform_meta_root = None
        if self.use_scale:
            recorded = self.train_args.get("transform_meta_root")
            root = str(Path(transform_meta_root).expanduser()) if transform_meta_root else recorded
            self.transform_meta_root = root
            if transform_meta_root:
                self.log(f"scale feature: using metadata folder {root} "
                         f"(overrides {recorded or 'none'} recorded in the checkpoint)")
            if root:
                try:
                    train_sessions = train_df["session_id"].unique()
                    self.log_scale_stats = compute_log_scale_stats(root, train_sessions)
                    self.log(f"scale feature: normalization stats computed from {len(train_sessions)} "
                             f"train sessions under {root}")
                except Exception as e:
                    self.log(f"WARNING: could not compute scale normalization stats ({e}) — "
                             f"falling back to neutral scale=1.0 for every frame. Set the "
                             f"metadata folder (--transform-meta-root, or Metadata on the Deploy "
                             f"page) to where the training transform metadata lives now.")
            else:
                self.log("WARNING: checkpoint uses the scale feature but has no "
                         "transform_meta_root recorded — falling back to neutral scale=1.0. "
                         "Set the metadata folder (--transform-meta-root, or Metadata on the "
                         "Deploy page).")

        self.log(f"loaded checkpoint {self.checkpoint_path} "
                 f"(epoch={self.ckpt.get('epoch')}, classes={self.idx_to_label}, "
                 f"use_flow={self.use_flow}, use_scale={self.use_scale})")

    # ---- per-camera raw access -------------------------------------------------

    def _open_camera(self, camera_dir):
        camera_dir = Path(camera_dir)
        rgb_fd = FrameDir(camera_dir / self.subdirs["rgb"])
        mask_fds = {
            1: FrameDir(camera_dir / self.subdirs[1]),
            2: FrameDir(camera_dir / self.subdirs[2]),
        }
        scale_table = None
        tm_path = camera_dir / TRANSFORM_META_NAME
        if self.use_scale and tm_path.exists():
            scale_table = read_transform_meta(tm_path)
        return {"dir": camera_dir, "rgb": rgb_fd, "masks": mask_fds, "scale_table": scale_table}

    # ---- per-window tensor construction ----------------------------------------

    def _camera_window(self, cam, cam_index, frame_nos, vis_sample_frames, rat):
        """Returns (frames_4ch[T,4,H,W] float32, cam_conf: float, scale_seq[T] or None,
        flow_seq[T,2,Hf,Wf] or None) for one camera, one window, one target rat."""
        mean, std = self.per_cam_mean[f"cam_{cam_index + 1}"], self.per_cam_std[f"cam_{cam_index + 1}"]
        mask_fd = cam["masks"][rat]

        frames = np.empty((SEQ_LEN, 4, self.frame_size, self.frame_size), dtype=np.float32)
        raw_rgb_u8 = [] if self.use_flow else None
        for t, fn in enumerate(frame_nos):
            rgb = read_or_hold(cam["rgb"], fn, frame_nos, color=True)
            mask = read_or_hold(mask_fd, fn, frame_nos, color=False)
            if rgb.shape[:2] != (self.frame_size, self.frame_size):
                import cv2
                rgb = cv2.resize(rgb, (self.frame_size, self.frame_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (self.frame_size, self.frame_size), interpolation=cv2.INTER_NEAREST)
            if raw_rgb_u8 is not None:
                raw_rgb_u8.append(rgb)
            rgb_f = rgb.astype(np.float32) / 255.0
            rgb_f = (rgb_f - mean) / std
            frames[t, :3] = rgb_f.transpose(2, 0, 1)
            frames[t, 3] = mask.astype(np.float32) / 255.0

        vis_frames = [read_or_hold(mask_fd, fn, frame_nos, color=False) for fn in vis_sample_frames]
        cam_conf = compute_visibility(vis_frames)

        scale_seq = None
        if self.use_scale:
            table = cam["scale_table"] or {}
            if table:
                scale_vals = np.array([table.get(fn, 1.0) for fn in frame_nos], dtype=np.float64)
            else:
                scale_vals = np.ones(SEQ_LEN, dtype=np.float64)
            if self.log_scale_stats is not None:
                log_scale = (np.log(scale_vals) - self.log_scale_stats[0]) / self.log_scale_stats[1]
            else:
                log_scale = np.zeros(SEQ_LEN, dtype=np.float64)
            scale_seq = log_scale.astype(np.float32)

        flow_seq = None
        if self.use_flow:
            import cv2
            gray = [cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) for a in raw_rgb_u8]
            if gray[0].shape[:2] != (self.flow_size, self.flow_size):
                gray = [cv2.resize(g, (self.flow_size, self.flow_size), interpolation=cv2.INTER_LINEAR)
                        for g in gray]
            flow_seq = np.zeros((SEQ_LEN, 2, self.flow_size, self.flow_size), dtype=np.float32)
            for t in range(1, SEQ_LEN):
                fl = self._flow_calc.calc(gray[t - 1], gray[t], None)
                flow_seq[t] = fl.transpose(2, 0, 1)
            flow_seq = np.clip(flow_seq / self.flow_scale, -1.0, 1.0)

        return frames, cam_conf, scale_seq, flow_seq

    def _predict_window(self, cams, frame_nos, rat):
        vis_idx = np.linspace(0, SEQ_LEN - 1, NUM_VIS_SAMPLES).astype(int)
        vis_sample_frames = [frame_nos[i] for i in vis_idx]

        per_cam = [self._camera_window(cam, ci, frame_nos, vis_sample_frames, rat)
                   for ci, cam in enumerate(cams)]
        images = np.stack([p[0] for p in per_cam], axis=0)          # (6, T, 4, H, W)
        cam_conf = np.array([p[1] for p in per_cam], dtype=np.float32)
        scale = np.stack([p[2] for p in per_cam], axis=0) if self.use_scale else None
        flow = np.stack([p[3] for p in per_cam], axis=0) if self.use_flow else None

        # topk-by-visibility camera ordering, exactly RatBoutDataset._select_cams with
        # cam_selection="topk" — with num_cams==6 this reorders all 6 cams, drops none.
        order = np.argsort(-cam_conf)
        images, cam_conf = images[order], cam_conf[order]
        if scale is not None:
            scale = scale[order]
        if flow is not None:
            flow = flow[order]

        images_t = torch.from_numpy(images).unsqueeze(0).to(self.device)
        cam_conf_t = torch.from_numpy(cam_conf).unsqueeze(0).to(self.device)
        dur = (np.log1p(SEQ_LEN) - self.duration_mean) / self.duration_std
        duration_t = torch.tensor([[dur]], dtype=torch.float32, device=self.device)
        scale_t = torch.from_numpy(scale).unsqueeze(0).to(self.device) if scale is not None else None
        flow_t = torch.from_numpy(flow).unsqueeze(0).to(self.device) if flow is not None else None

        with torch.no_grad():
            logits = self.model(images_t, cam_conf_t, duration_t, flow=flow_t, scale=scale_t)
            probs = F.softmax(logits, dim=-1)[0]
            conf, pred = probs.max(dim=0)
        return self.idx_to_label[int(pred)], float(conf)

    # ---- public entry point ----------------------------------------------------

    def run(self, camera_dirs, rats=(1, 2), stride=SEQ_LEN, progress_cb=None):
        if len(camera_dirs) != 6:
            raise ValueError(f"expected exactly 6 camera directories, got {len(camera_dirs)}")
        cams = [self._open_camera(d) for d in camera_dirs]

        starts, ends = zip(*(c["rgb"].available_range() for c in cams))
        session_start, session_end = max(starts), min(ends)
        if session_end - session_start + 1 < SEQ_LEN:
            raise ValueError(
                f"cameras only overlap over frames [{session_start}, {session_end}] "
                f"({session_end - session_start + 1} frames) — need at least {SEQ_LEN}."
            )

        window_starts = list(range(session_start, session_end - SEQ_LEN + 2, stride))
        n_total = len(window_starts) * len(rats)
        self.log(f"session frames [{session_start}, {session_end}], "
                 f"{len(window_starts)} windows x {len(rats)} rat(s) = {n_total} inference passes")

        results = {f"rat_{r}": {} for r in rats}
        window_predictions = {f"rat_{r}": [] for r in rats}  # per-window, not frame-flattened —
                                                               # this is what the correction log wants
        done = 0
        for w_start in window_starts:
            frame_nos = list(range(w_start, w_start + SEQ_LEN))
            for rat in rats:
                label, conf = self._predict_window(cams, frame_nos, rat)
                entry = {"label": label, "confidence": conf, "window_start": w_start}
                window_predictions[f"rat_{rat}"].append(
                    {"window_start": w_start, "seq_len": SEQ_LEN, "label": label, "confidence": conf}
                )
                bucket = results[f"rat_{rat}"]
                for fn in frame_nos:
                    prev = bucket.get(fn)
                    if prev is None or entry["confidence"] > prev["confidence"]:
                        bucket[fn] = entry
                done += 1
                if progress_cb:
                    progress_cb(done, n_total)

        return {
            "checkpoint": str(Path(self.checkpoint_path).resolve()),
            "checkpoint_epoch": self.ckpt.get("epoch"),
            "cameras": [str(Path(d).resolve()) for d in camera_dirs],
            "seq_len": SEQ_LEN,
            "stride": stride,
            "session_frame_range": [session_start, session_end],
            # False = checkpoint uses the scale feature but its normalization stats couldn't be
            # computed, so every frame got the neutral value; None = checkpoint has no scale feature
            "scale_stats_ok": (self.log_scale_stats is not None) if self.use_scale else None,
            "transform_meta_root": self.transform_meta_root,
            "window_predictions": window_predictions,  # per (rat, window) — feed to correction_log
            "rats": {
                rk: [
                    {"frame": fn, "label": v["label"], "confidence": round(v["confidence"], 4),
                     "window_start": v["window_start"]}
                    for fn, v in sorted(frames.items())
                ]
                for rk, frames in results.items()
            },
        }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cameras", nargs=6, required=True, metavar=("CAM1", "CAM2", "CAM3", "CAM4", "CAM5", "CAM6"))
    ap.add_argument("--rats", default="1,2", help="which object(s) to classify, e.g. '1', '2', or '1,2'")
    ap.add_argument("--stride", type=int, default=SEQ_LEN)
    ap.add_argument("--window-index-csv", default=None)
    ap.add_argument("--normstats", default=None)
    ap.add_argument("--transform-meta-root", default=None,
                     help="folder of the training sessions' transform metadata, for the scale "
                          "feature's normalization stats. Defaults to the path recorded in the "
                          "checkpoint; set it if that folder has moved since training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-label", default=None,
                     help="name for this deployment session in the correction log (defaults to "
                          "the output filename's stem)")
    ap.add_argument("--no-log", action="store_true",
                     help="skip writing predictions to the correction log (6_deployment/correction_log.py)")
    ap.add_argument("--correction-db", default=None, help="defaults to ~/.rat_deploy_corrections.db")
    ap.add_argument("--rgb-subdir", default=DEFAULT_SUBDIRS["rgb"])
    ap.add_argument("--mask1-subdir", default=DEFAULT_SUBDIRS[1])
    ap.add_argument("--mask2-subdir", default=DEFAULT_SUBDIRS[2])
    args = ap.parse_args()

    rats = tuple(int(r) for r in args.rats.split(","))
    subdirs = {"rgb": args.rgb_subdir, 1: args.mask1_subdir, 2: args.mask2_subdir}
    inferer = SessionInferer(args.checkpoint, window_index_csv=args.window_index_csv,
                              normstats_path=args.normstats, subdirs=subdirs,
                              transform_meta_root=args.transform_meta_root)
    result = inferer.run(args.cameras, rats=rats, stride=args.stride)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    if not args.no_log:
        from correction_log import log_window_predictions
        session_label = args.session_label or Path(args.out).stem
        for rat in rats:
            n = log_window_predictions(
                args.correction_db, session_label, args.cameras, args.checkpoint, rat,
                result["window_predictions"][f"rat_{rat}"], subdirs=subdirs,
            )
            print(f"logged {n} rat_{rat} window predictions to correction log")


if __name__ == "__main__":
    main()
