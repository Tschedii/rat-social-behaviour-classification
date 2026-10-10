"""Evaluate one or more checkpoints on a labelled session that was never packed into the
training .h5, straight from the raw segmentation output:

    <session_dir>/cam_N/{all_objects, object_1, object_2}/frame_XXXXXX.png   (1920x1200 cutouts)

For every labelled bout it rebuilds what 2_build_training_data.py + 2_build_h5_dataset.py +
RatBoutDataset would have produced (32 sampled frames per camera, the joint two-rat crop
letterboxed to 1024x1024, the focal rat's mask, DIS optical flow, the registration scale and the
camera visibility) and runs the same computation as CNNTransformer.forward, one camera at a
time, so a bout never needs the full (6, 32, 4, 1024, 1024) float tensor in memory (about
3.2 GB, too much for an 8 GB laptop). All checkpoints share one pass over the data, since the
inputs only differ in the duration and scale standardisation, which each checkpoint takes from
its own training index / logs.

    python evaluate_raw_session.py \
        --session-dir /path/to/Day_7_rPEG008_rPEG007_25-08-2026 \
        --labels ../../label_clean/test/Day_7_rPEG008_rPEG007_25-08-2026_evaluation.csv \
        --object-id 1 \
        --model v2_avg5 training_output_v2/averaged_top5.pt index_v2/window_index.csv 0.3132 0.3340 \
        --model v1_avg5 training_output/averaged_top5.pt window_index.csv 0.2633 0.3391 \
        --out-dir eval_day7

The two numbers after each index are the log(scale) mean/std that train.py logged for that run
("scale feature: log-scale stats over N sessions"); they are not recomputable here because the
transform_meta.csv files only exist on the server.

Per-bout logits are appended to <out-dir>/logits_<name>.csv as they are computed, so an
interrupted run resumes where it stopped.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helpers.checkpoint_utils import build_model, load_checkpoint, resolve_exclude_remap
from helpers.confusion_matrix import save_confusion_matrix
from train import pick_device

CAMS = [f"cam_{i}" for i in range(1, 7)]
SEQ_LEN = 32
CROP_SIZE = 1024      # 2_build_training_data.py --size
CROP_MARGIN = 0.08    # 2_build_training_data.py --margin
CROP_MARGIN_MIN = 8   # 2_build_training_data.py --margin_min
BG_THRESHOLD = 8      # 2_build_training_data.py --bg_threshold
VIS_THUMB = 96        # helpers/build_index.py
NUM_VIS_SAMPLES = 8   # helpers/build_index.py default


# ── Crop (verbatim from 2_build_training_data.py) ─────────────────────────────

def extract_mask(img, bg_threshold=BG_THRESHOLD):
    mask = (img.max(axis=2) > bg_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def compute_crop_rect(union_mask, margin_frac, margin_min):
    ys, xs = np.where(union_mask > 0)
    if xs.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    w, h = x2 - x1, y2 - y1
    mx = max(margin_min, int(round(w * margin_frac)))
    my = max(margin_min, int(round(h * margin_frac)))
    H, W = union_mask.shape[:2]
    return (max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my))


def square_letterbox(src, rect, target_size, interp):
    x1, y1, x2, y2 = rect
    crop = src[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    side = max(h, w)
    if crop.ndim == 3:
        canvas = np.zeros((side, side, crop.shape[2]), dtype=crop.dtype)
    else:
        canvas = np.zeros((side, side), dtype=crop.dtype)
    yo, xo = (side - h) // 2, (side - w) // 2
    canvas[yo:yo + h, xo:xo + w] = crop
    scale = target_size / side
    out = cv2.resize(canvas, (target_size, target_size), interpolation=interp)
    return out, scale


def build_frame(all_rgb, obj1_rgb, obj2_rgb):
    """(rgb BGR, mask_1, mask_2, scale) for one frame, or None if no rat is present."""
    m1 = extract_mask(obj1_rgb) if obj1_rgb is not None else np.zeros(all_rgb.shape[:2], np.uint8)
    m2 = extract_mask(obj2_rgb) if obj2_rgb is not None else np.zeros(all_rgb.shape[:2], np.uint8)
    rect = compute_crop_rect(cv2.bitwise_or(m1, m2), CROP_MARGIN, CROP_MARGIN_MIN)
    if rect is None:
        return None
    rgb_out, scale = square_letterbox(all_rgb, rect, CROP_SIZE, cv2.INTER_LINEAR)
    m1_out, _ = square_letterbox(m1, rect, CROP_SIZE, cv2.INTER_NEAREST)
    m2_out, _ = square_letterbox(m2, rect, CROP_SIZE, cv2.INTER_NEAREST)
    keep = (m1_out > 0) | (m2_out > 0)
    rgb_out[~keep] = 0
    m1_out = np.where(m1_out > 127, 255, 0).astype(np.uint8)
    m2_out = np.where(m2_out > 127, 255, 0).astype(np.uint8)
    return rgb_out, m1_out, m2_out, round(scale, 4)  # transform_meta.csv stores 4 decimals


# ── Model: CNNTransformer.forward, split into per-camera and per-bout halves ──

@torch.no_grad()
def camera_embeddings(model, frames, flow):
    """frames: (T, 4, H, W), flow: (T, 2, Hf, Wf) or None -> (T, d_model). Same ops as the
    per-frame part of CNNTransformer.forward (backbone, flow backbone, fusion_proj)."""
    feats = model._chunked_forward(model.backbone, frames)
    if model.use_flow:
        flow_feats = model._chunked_forward(model.flow_backbone, flow)
        feats = model.fusion_proj(torch.cat([feats, flow_feats], dim=-1))
    return feats


@torch.no_grad()
def bout_logits(model, feats, cam_conf, scale, duration):
    """feats: (T, C, d_model) in the dataset's camera order, cam_conf: (C,), scale: (T, C) or
    None, duration: scalar tensor -> (num_classes,). Same ops as the rest of
    CNNTransformer.forward with B=1."""
    T, C, _ = feats.shape
    conf = cam_conf.unsqueeze(0).expand(T, C)
    fused = model.camera_fusion(feats, conf, scale=scale)       # (T, d_model)
    bout_embed = model.temporal(fused.unsqueeze(0))             # (1, d_model)
    return model.head(torch.cat([bout_embed, duration.reshape(1, 1)], dim=-1))[0]


def check_split_forward(model, device):
    """The split forward above must match CNNTransformer.forward exactly; check on a small
    random input (resolution only matters to the CNNs, which are fully convolutional)."""
    g = torch.Generator().manual_seed(0)
    C, H = 3, 128
    images = torch.randn(1, C, SEQ_LEN, 4, H, H, generator=g).to(device)
    flow = torch.randn(1, C, SEQ_LEN, 2, 64, 64, generator=g).to(device) if model.use_flow else None
    cam_conf = torch.rand(1, C, generator=g).to(device)
    scale = torch.randn(1, C, SEQ_LEN, generator=g).to(device)
    duration = torch.randn(1, 1, generator=g).to(device)
    with torch.no_grad():
        ref = model(images, cam_conf, duration, flow=flow, scale=scale)[0]
        feats = torch.stack([
            camera_embeddings(model, images[0, c], flow[0, c] if flow is not None else None)
            for c in range(C)], dim=1)
        out = bout_logits(model, feats, cam_conf[0], scale[0].T, duration[0, 0])
    err = (ref - out).abs().max().item()
    if err > 1e-3:
        raise RuntimeError(f"split forward differs from CNNTransformer.forward by {err:.2e}")
    return err


# ── Bout sampling (2_build_h5_dataset.py + frame_io.py + rat_h5_dataset.py) ──

def fixed_length_indices(start, end, n_frames=SEQ_LEN):
    """frame_io.fixed_length_indices: n_frames frame numbers spanning [start, end)."""
    if end - start <= 0:
        return []
    return np.round(np.linspace(start, end - 1, num=n_frames)).astype(int).tolist()


def resample_indices(n, target=SEQ_LEN):
    """rat_h5_dataset._resample_indices: nearest-neighbour stretch of n stored frames to 32."""
    if n == target:
        return np.arange(n)
    return np.minimum((np.arange(target) * n) // target, n - 1)


def index_frames(folder):
    return {int(p.stem.split("_")[-1]): p for p in Path(folder).glob("frame_*.png")}


class RawBoutCameraDataset(Dataset):
    """One item per (bout, camera): the stored frames 2_build_h5_dataset.py would have written
    for that camera, already stretched to 32 by the dataset's resampling, plus flow, scale and
    visibility. RGB stays uint8 here; normalisation happens on the device."""

    def __init__(self, session_dir, bouts, object_id, flow_size, flow_scale):
        self.session_dir = Path(session_dir)
        self.bouts = bouts
        self.object_id = object_id
        self.flow_size = flow_size
        self.flow_scale = flow_scale
        self.frames = {
            cam: {tag: index_frames(self.session_dir / cam / tag)
                  for tag in ("all_objects", "object_1", "object_2")}
            for cam in CAMS
        }
        self._flow_calc = None

    def __len__(self):
        return len(self.bouts) * len(CAMS)

    def _crop(self, cam, fn):
        idx = self.frames[cam]
        if fn not in idx["all_objects"]:
            return None
        all_rgb = cv2.imread(str(idx["all_objects"][fn]))
        obj1 = cv2.imread(str(idx["object_1"][fn])) if fn in idx["object_1"] else None
        obj2 = cv2.imread(str(idx["object_2"][fn])) if fn in idx["object_2"] else None
        return build_frame(all_rgb, obj1, obj2)

    def __getitem__(self, i):
        if self._flow_calc is None:
            cv2.setNumThreads(1)
            self._flow_calc = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        bout = self.bouts[i // len(CAMS)]
        cam = CAMS[i % len(CAMS)]

        # The crop folder's available range. The builder only clips against it; it matters
        # only for bouts at the very start/end of a recording.
        lo, hi = min(self.frames[cam]["all_objects"]), max(self.frames[cam]["all_objects"])
        wanted = [fn for fn in fixed_length_indices(bout["start_frame"], bout["end_frame"]) if lo <= fn <= hi]

        stored, cache = [], {}
        for fn in wanted:
            if fn not in cache:
                cache[fn] = self._crop(cam, fn)
            if cache[fn] is not None:  # no crop written for this frame (no rat) -> builder skips it
                stored.append((fn, cache[fn]))

        if not stored:
            # The builder writes no cam group in this case; training never hit it (the dataset
            # would raise a KeyError), so mark it and let the caller decide.
            return {"bout_idx": i // len(CAMS), "cam": cam, "n_stored": 0}

        # build_index.py visibility: 8 of the STORED frames, focal mask at 96x96 nearest
        n = len(stored)
        vis_idx = np.linspace(0, n - 1, min(NUM_VIS_SAMPLES, n)).astype(int)
        vis = float(np.mean([
            float((np.asarray(Image.fromarray(stored[j][1][self.object_id]).resize(
                (VIS_THUMB, VIS_THUMB), Image.NEAREST)) > 0).mean())
            for j in vis_idx]))

        src_idx = resample_indices(n)
        rgb = np.stack([stored[j][1][0][..., ::-1] for j in src_idx])         # (T, H, W, 3) RGB uint8
        mask = np.stack([stored[j][1][self.object_id] for j in src_idx])      # (T, H, W) 0/255
        scale = np.array([stored[j][1][3] for j in src_idx], dtype=np.float64)
        frame_ids = np.array([stored[j][0] for j in src_idx])

        # _decode_gray + _flow_sequence: PIL luma, PIL bilinear resize, DIS fast
        gray = [np.asarray(Image.fromarray(np.ascontiguousarray(f)).convert("L").resize(
            (self.flow_size, self.flow_size), Image.BILINEAR)) for f in rgb]
        flow = np.zeros((SEQ_LEN, 2, self.flow_size, self.flow_size), dtype=np.float32)
        for t in range(1, SEQ_LEN):
            flow[t] = self._flow_calc.calc(gray[t - 1], gray[t], None).transpose(2, 0, 1)
        flow = np.clip(flow / self.flow_scale, -1.0, 1.0)

        return {"bout_idx": i // len(CAMS), "cam": cam, "n_stored": n,
                "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
                "mask": torch.from_numpy(mask), "flow": torch.from_numpy(flow),
                "scale": scale, "frame_ids": frame_ids, "vis": vis}


# ── Labels and checkpoints ────────────────────────────────────────────────────

def load_bouts(labels_csv, label_map):
    df = pd.read_csv(labels_csv)
    if "annotation_type" in df.columns:
        df = df[df["annotation_type"] == "segment"]
    df["behavior"] = df["behavior"].str.replace(r"\s*/\s*", "/", regex=True)  # "Approaching / Following"
    unknown = sorted(set(df["behavior"]) - set(label_map))
    if unknown:
        print(f"dropping {int(df['behavior'].isin(unknown).sum())} bouts with classes the model "
              f"does not predict: {unknown}")
        df = df[~df["behavior"].isin(unknown)]
    df = df.reset_index(drop=True)
    df["duration_frames"] = df["end_frame"] - df["start_frame"]
    df["label_id"] = df["behavior"].map(label_map).astype(int)
    return df


def load_model_spec(name, ckpt_path, index_csv, log_scale_mean, log_scale_std, device, micro_batch_size):
    ckpt, train_args, label_map, idx_to_label = load_checkpoint(ckpt_path, "cpu")
    exclude_ids, _ = resolve_exclude_remap(train_args, index_csv, label_map,
                                           Path(index_csv).parent / "label_map.json")
    # duration stats exactly as checkpoint_utils.load_model_and_datasets: train split, after
    # dropping excluded classes
    idx = pd.read_csv(index_csv)
    train = idx[(idx["split"] == "train") & ~idx["label_id"].isin(exclude_ids)]
    log_dur = np.log1p(train["duration_frames"].to_numpy(dtype=np.float32))
    model = build_model(train_args, label_map, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.micro_batch_size = min(model.micro_batch_size, micro_batch_size)  # memory only, same result
    return {
        "name": name, "path": str(ckpt_path), "model": model, "label_map": label_map,
        "idx_to_label": idx_to_label, "train_args": train_args,
        "duration_stats": (float(log_dur.mean()), float(log_dur.std() + 1e-6)),
        "log_scale_stats": (float(log_scale_mean), float(log_scale_std)),
        "averaged_from": ckpt.get("averaged_from"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--labels", required=True, help="label CSV with start_frame, end_frame (exclusive), behavior")
    ap.add_argument("--object-id", type=int, required=True, choices=[1, 2], help="which object the labels belong to")
    ap.add_argument("--model", nargs=5, action="append", required=True,
                    metavar=("NAME", "CHECKPOINT", "INDEX_CSV", "LOG_SCALE_MEAN", "LOG_SCALE_STD"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--micro-batch-size", type=int, default=4, help="frames per backbone call (memory only)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N bouts (smoke test)")
    args = ap.parse_args()

    device = torch.device(pick_device())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [load_model_spec(n, c, i, m, s, device, args.micro_batch_size) for n, c, i, m, s in args.model]
    label_map = specs[0]["label_map"]
    for spec in specs:
        if spec["label_map"] != label_map:
            raise ValueError(f"{spec['name']} has a different label_map than {specs[0]['name']}")
        if not spec["train_args"].get("use_flow") or spec["train_args"]["num_cams"] != len(CAMS):
            raise ValueError(f"{spec['name']}: this script assumes use_flow=True and num_cams=6")
        err = check_split_forward(spec["model"], device)
        print(f"{spec['name']}: split forward matches CNNTransformer.forward (max |diff| {err:.1e}); "
              f"duration stats {spec['duration_stats']}, log-scale stats {spec['log_scale_stats']}")
    idx_to_label = specs[0]["idx_to_label"]
    flow_size, flow_scale = specs[0]["train_args"]["flow_size"], specs[0]["train_args"]["flow_scale"]
    norm = json.loads((Path(__file__).parent / "rgb_normstats.json").read_text())["per_camera"]
    cam_mean = {c: torch.tensor(norm[c]["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1) for c in CAMS}
    cam_std = {c: torch.tensor(norm[c]["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1) for c in CAMS}

    bouts = load_bouts(args.labels, label_map)
    if args.limit:
        bouts = bouts.iloc[: args.limit]
    bouts.to_csv(out_dir / "bouts.csv", index=False)
    print(f"{len(bouts)} bouts, object_{args.object_id}: {bouts['behavior'].value_counts().to_dict()}")

    # resume per model: a bout is loaded again only if some model still lacks it, and only those
    # models run on it (so a checkpoint can be added to an existing out-dir later)
    logit_cols = [f"logit_{c}" for c in idx_to_label]
    dropped_csv = out_dir / "dropped_bouts.csv"
    dropped = set(pd.read_csv(dropped_csv)["segment_id"]) if dropped_csv.exists() else set()
    needs = {sid: [] for sid in bouts["segment_id"] if sid not in dropped}
    for spec in specs:
        p = out_dir / f"logits_{spec['name']}.csv"
        done = set(pd.read_csv(p)["segment_id"]) if p.exists() else set()
        for sid in needs:
            if sid not in done:
                needs[sid].append(spec)
        print(f"{spec['name']}: {sum(spec in v for v in needs.values())} bouts to go")
    todo = bouts[bouts["segment_id"].map(lambda sid: bool(needs.get(sid)))].reset_index(drop=True)
    print(f"{len(todo)} of {len(bouts)} bouts to load ({len(dropped)} dropped earlier)")

    ds = RawBoutCameraDataset(args.session_dir, todo.to_dict("records"), args.object_id, flow_size, flow_scale)
    loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.num_workers,
                        prefetch_factor=2 if args.num_workers else None,
                        persistent_workers=False)

    per_bout = {}
    for item in tqdm(loader, total=len(ds), desc="bout x camera"):
        b, cam = item["bout_idx"], item["cam"]
        active = needs[todo.loc[b, "segment_id"]]
        entry = per_bout.setdefault(b, {"feats": {s["name"]: {} for s in specs}, "vis": {}, "scale": {},
                                        "empty": [], "log": []})
        if item["n_stored"] == 0:
            entry["empty"].append(cam)
        else:
            scale = np.asarray(item["scale"], dtype=np.float64)  # the loader turns numpy into tensors
            frame_ids = np.asarray(item["frame_ids"])
            entry["log"].append({"segment_id": todo.loc[b, "segment_id"], "cam": cam,
                                 "n_stored": item["n_stored"], "vis": item["vis"],
                                 "scale_min": float(scale.min()), "scale_max": float(scale.max()),
                                 "first_frame": int(frame_ids[0]), "last_frame": int(frame_ids[-1])})
            # Everything up to fusion_proj is per frame (GroupNorm, attention pooling, linear
            # layers), so building and embedding the frames a chunk at a time is exact and keeps
            # the (32, 4, 1024, 1024) float tensor out of memory.
            chunks = {spec["name"]: [] for spec in active}
            with torch.no_grad():
                for t0 in range(0, SEQ_LEN, args.micro_batch_size):
                    sl = slice(t0, t0 + args.micro_batch_size)
                    rgb = item["rgb"][sl].to(device).permute(0, 3, 1, 2).float().div_(255.0)
                    frames = torch.cat([(rgb - cam_mean[cam]) / cam_std[cam],
                                        item["mask"][sl].to(device).float().div_(255.0).unsqueeze(1)], dim=1)
                    flow = item["flow"][sl].to(device)
                    for spec in active:
                        chunks[spec["name"]].append(camera_embeddings(spec["model"], frames, flow))
                    del rgb, frames, flow
            for spec in active:
                entry["feats"][spec["name"]][cam] = torch.cat(chunks[spec["name"]], dim=0)
            if device.type == "mps":
                # the MPS allocator keeps freed activation buffers wired; on an 8 GB machine that
                # pushes everything else into swap within an hour
                torch.mps.empty_cache()
            entry["vis"][cam] = item["vis"]
            entry["scale"][cam] = scale

        if len(entry["vis"]) + len(entry["empty"]) < len(CAMS):
            continue
        row = todo.loc[b]
        del per_bout[b]
        if entry["empty"]:
            if len(entry["empty"]) < len(CAMS):
                # the builder would write this bout without those cams, and RatBoutDataset
                # (num_cams=6) would then fail on it, so training never saw such a bout
                raise RuntimeError(f"bout {row['segment_id']} has no crop in {entry['empty']} "
                                   f"but has one in the other cameras")
            # no crop in any camera: 2_build_h5_dataset.py drops the bout (empty_segments)
            print(f"\ndropping bout {row['segment_id']} ({row['behavior']}, frames "
                  f"{row['start_frame']}-{row['end_frame']}): no segmented frame in any camera")
            pd.DataFrame([{"segment_id": row["segment_id"], "behavior": row["behavior"],
                           "start_frame": row["start_frame"], "end_frame": row["end_frame"],
                           "reason": "no segmented frame in any camera"}]).to_csv(
                dropped_csv, mode="a", index=False, header=not dropped_csv.exists())
            continue

        # bout complete: camera order = RatBoutDataset._select_cams(topk) on build_index.py's vis
        vis = np.array([entry["vis"][c] for c in CAMS], dtype=np.float32)
        order = np.argsort(-vis)
        cams_sorted = [CAMS[k] for k in order]
        cam_conf = torch.from_numpy(vis[order]).to(device)
        for spec in active:
            feats = torch.stack([entry["feats"][spec["name"]][c] for c in cams_sorted], dim=1)  # (T, C, d)
            m, s = spec["log_scale_stats"]
            scale = np.stack([(np.log(entry["scale"][c]) - m) / s for c in cams_sorted], axis=1)  # (T, C)
            dm, dsd = spec["duration_stats"]
            dur = (np.log1p(np.float32(row["duration_frames"])) - dm) / dsd
            logits = bout_logits(spec["model"], feats, cam_conf,
                                 torch.from_numpy(scale.astype(np.float32)).to(device),
                                 torch.tensor(dur, dtype=torch.float32, device=device))
            logits = logits.cpu().numpy()
            rec = {"segment_id": row["segment_id"], "start_frame": int(row["start_frame"]),
                   "end_frame": int(row["end_frame"]), "behavior": row["behavior"],
                   "true_label_id": int(row["label_id"]), "pred_label_id": int(logits.argmax()),
                   "cam_order": ",".join(cams_sorted), **dict(zip(logit_cols, logits.tolist()))}
            p = out_dir / f"logits_{spec['name']}.csv"
            pd.DataFrame([rec]).to_csv(p, mode="a", header=not p.exists(), index=False)
        p = out_dir / "camera_inputs.csv"
        pd.DataFrame(entry["log"]).to_csv(p, mode="a", header=not p.exists(), index=False)

    p = out_dir / "camera_inputs.csv"
    if p.exists():  # a bout redone for an added model logs its cameras again
        pd.read_csv(p).drop_duplicates(["segment_id", "cam"], keep="last").to_csv(p, index=False)

    # ── reports ──
    summary = {}
    for spec in specs:
        res = pd.read_csv(out_dir / f"logits_{spec['name']}.csv").drop_duplicates("segment_id", keep="last")
        res = res[res["segment_id"].isin(bouts["segment_id"])]
        y, yhat = res["true_label_id"].to_numpy(), res["pred_label_id"].to_numpy()
        labels = list(range(len(idx_to_label)))
        report = classification_report(y, yhat, labels=labels, target_names=idx_to_label, digits=3, zero_division=0)
        present = sorted(set(y))
        summary[spec["name"]] = {
            "checkpoint": spec["path"], "averaged_from": spec["averaged_from"], "n_bouts": int(len(res)),
            "n_bouts_dropped": int(len(pd.read_csv(dropped_csv))) if dropped_csv.exists() else 0,
            "accuracy": float(accuracy_score(y, yhat)),
            "macro_f1_all_9_classes": float(f1_score(y, yhat, labels=labels, average="macro", zero_division=0)),
            "macro_f1_classes_present": float(f1_score(y, yhat, labels=present, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, yhat, labels=present, average="weighted", zero_division=0)),
        }
        sub = out_dir / spec["name"]
        sub.mkdir(exist_ok=True)
        (sub / "classification_report.txt").write_text(report)
        save_confusion_matrix(y, yhat, idx_to_label, sub / "confusion_matrix.png",
                              title=f"{spec['name']} on {Path(args.session_dir).name}")
        res["pred_behavior"] = res["pred_label_id"].map(dict(enumerate(idx_to_label)))
        res["correct"] = res["true_label_id"] == res["pred_label_id"]
        res.to_csv(sub / "predictions.csv", index=False)
        print(f"\n=== {spec['name']} ({len(res)} bouts) ===\n{report}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
