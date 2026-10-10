"""Train the CNN+Transformer rat behavior classifier.

    python train.py \
        --h5 /path/to/rat_behaviour_32f.h5 \
        --index-csv /path/to/rat_transformer_training/window_index.csv \
        --normstats /path/to/rgb_normstats.json \
        --out-dir /path/to/rat_transformer_training/checkpoints

Run helpers/build_index.py first to produce window_index.csv / label_map.json.
"""
import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from helpers.confusion_matrix import save_confusion_matrix
from helpers.rat_h5_dataset import RatBoutDataset, compute_log_scale_stats
from model import CNNTransformer


def pick_device(gpu_id=None):
    if torch.cuda.is_available():
        return f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_logging(out_dir):
    """Writes to both stdout (for an attached terminal/tmux pane) and
    {out_dir}/train.log (so `tail -f train.log` works no matter how the job
    was launched — nohup, tmux, a batch scheduler, etc). tqdm's progress bar
    goes to stderr directly and isn't duplicated here, so the log file stays
    readable (one line per epoch/event) instead of full of bar-redraw spam."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(Path(out_dir) / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def class_weights_inv_sqrt_freq(labels, num_classes, min_weight=0.0, multipliers=None):
    """Inverse-sqrt class frequency, with an optional floor and optional manual multipliers.

    Diagnosed via evaluate.py: Exploring stayed at exactly 0.000 F1 through two different
    sampler strategies (full inverse-freq. and inverse-sqrt), while classes with a similarly
    low sample count recovered. Exploring is the *most common* class, so its inverse-sqrt loss
    weight (0.53, the lowest of all 10) barely penalises getting it wrong — a problem the
    sampler can't fix, since sampling controls exposure frequency, not gradient magnitude once
    a sample is seen. min_weight floors every class's weight (as a fraction of the mean) so no
    class, common or not, is under-penalised purely for being frequent.

    multipliers (optional array of length num_classes) applies on top of the frequency-derived,
    floored weight — a way to redirect gradient budget based on current per-class *results*
    (a weak class despite ample support, or a class already strong enough to ease off), which
    frequency alone can't express. Renormalised to mean 1 afterwards so it stays on the same
    scale as min_weight.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    if min_weight > 0:
        w = np.clip(w, min_weight, None)
    if multipliers is not None:
        w = w * np.asarray(multipliers, dtype=np.float64)
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_balanced_sampler(labels, num_classes):
    """WeightedRandomSampler using inverse-*square-root* class frequency (matches the loss's
    class weighting below), so rare classes are seen more often per epoch without fully
    equalizing exposure to common ones.

    This is a different lever from the loss's class weighting: loss weighting scales *how
    much a mistake costs* once a sample is seen, but doesn't change how *often* a rare class's
    gradient signal shows up in a given epoch. Originally used full inverse frequency (1/count)
    here, which fixed the intended target — evaluate.py had shown Grooming/Rearing/Body touch
    as learned-but-under-triggered (decent precision, near-zero recall), and full-strength
    oversampling raised their recall substantially (Rearing 0.02->0.51 F1 0.04->0.49, Grooming
    F1 0.11->0.34, Body touch F1 0.12->0.39). But it over-corrected: Investigation's recall
    collapsed (0.46->0.16, F1 0.26->0.19), Approaching/Following softened (F1 0.42->0.38), and
    Exploring — already weak — was pushed to a full 0/0/0 collapse, plausibly because shrinking
    a common class's relative sampling frequency that much left it too rarely seen. Softening to
    sqrt aims to keep most of the Grooming/Rearing/Body touch gains while giving the previously-
    strong common classes enough exposure to hold their ground.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.05):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


_TOP_CKPT_RE = re.compile(r"^epoch(\d+)_f1([0-9.]+)\.pt$")


def scan_top_checkpoints(top_dir):
    """Reconstruct the current top-k list from filenames alone (epoch{N}_f1{score}.pt) —
    no need to load any checkpoint just to read its score, and resume-safe: a resumed run
    picks up exactly where the top-k list left off, no matter how it got there. Returns a
    list of (f1, epoch, path) sorted ascending (index 0 = worst of the kept ones)."""
    found = []
    for p in Path(top_dir).glob("*.pt"):
        m = _TOP_CKPT_RE.match(p.name)
        if m:
            found.append((float(m.group(2)), int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def update_top_checkpoints(top_dir, current, epoch, f1, ckpt, k, log):
    """Saves `ckpt` into top_dir if it earns a spot among the k best eval macro-F1 scores seen
    so far, deleting the worst-ranked file if that pushes the count over k. `current` is the
    list scan_top_checkpoints() returns; call site owns re-assigning its return value."""
    if k <= 0:
        return current
    if len(current) >= k and f1 <= current[0][0]:
        return current
    path = Path(top_dir) / f"epoch{epoch:03d}_f1{f1:.4f}.pt"
    torch.save(ckpt, path)
    current.append((f1, epoch, path))
    current.sort(key=lambda t: t[0])
    log.info(f"  -> saved to top_checkpoints/ (rank-worthy: {[round(c[0], 4) for c in current]})")
    if len(current) > k:
        worst_f1, worst_epoch, worst_path = current.pop(0)
        worst_path.unlink(missing_ok=True)
        log.info(f"  -> pruned top_checkpoints/{worst_path.name} (epoch {worst_epoch}, f1 {worst_f1:.4f}) to stay at top-{k}")
    return current


def run_epoch(model, loader, device, criterion, optimizer, scaler, lr_fn, step_offset, total_steps,
              warmup_steps, base_lrs, train, grad_accum_steps=1):
    """Note on grad_accum_steps: at native (1024x1024) resolution the physical batch size
    has to stay small (the raw input tensor itself, before it ever reaches the model, is
    batch*num_cams*T*4*H*W floats — see README), so a large *effective* batch size is
    reached by accumulating gradients over several small physical batches before each
    optimizer step, rather than by making the physical batch bigger."""
    model.train(train)
    all_preds, all_labels = [], []
    total_loss, total_n = 0.0, 0
    step = step_offset
    n_batches = len(loader)
    if train:
        optimizer.zero_grad(set_to_none=True)
    accum_count = 0

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
    for i, batch in enumerate(pbar):
        images = batch["images"].to(device, non_blocking=True)
        cam_conf = batch["cam_conf"].to(device, non_blocking=True)
        duration = batch["duration"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True) if "flow" in batch else None
        scale = batch["scale"].to(device, non_blocking=True) if "scale" in batch else None

        with torch.set_grad_enabled(train):
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, cam_conf, duration, flow=flow, scale=scale)
                    loss = criterion(logits, labels)
            else:
                logits = model(images, cam_conf, duration, flow=flow, scale=scale)
                loss = criterion(logits, labels)

        if train:
            (loss / grad_accum_steps if scaler is None else scaler.scale(loss / grad_accum_steps)).backward()
            accum_count += 1
            if accum_count == grad_accum_steps or i == n_batches - 1:
                for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                    pg["lr"] = lr_fn(step, total_steps, warmup_steps, base_lr)
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                step += 1

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        all_preds.append(logits.detach().argmax(-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    metrics = {
        "loss": total_loss / total_n,
        "acc": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro"),
    }
    return metrics, step, labels_np, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--normstats", required=True)
    ap.add_argument("--label-map", default=None, help="defaults to label_map.json next to index-csv")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--num-cams", type=int, default=3)
    ap.add_argument(
        "--frame-size", type=int, default=1024,
        help="1024 = native resolution as stored in the .h5, no resizing at all. Only lower this "
        "if you deliberately want to downsample (not recommended — see README).",
    )
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument(
        "--gpu-id", type=int, default=None,
        help="select a specific GPU by index on a multi-GPU machine (e.g. 0 or 1). "
        "Alternatively set CUDA_VISIBLE_DEVICES before launching — check `nvidia-smi` first "
        "on a shared server to see which GPU is actually free.",
    )
    ap.add_argument(
        "--micro-batch-size", type=int, default=8,
        help="max frames processed through the CNN backbone at once (chunked + gradient-"
        "checkpointed). Controls GPU memory only, not results — lower this first if you hit "
        "CUDA OOM at native resolution (raise it if a 20GB+ GPU has headroom to spare).",
    )
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="disable gradient checkpointing in the backbone (faster, needs much more memory).")
    ap.add_argument(
        "--backbone-norm", default="groupnorm", choices=["groupnorm", "batchnorm"],
        help="groupnorm (default) is required for correctness at batch-size=1 (native-res "
        "memory forces batch-size=1, and BatchNorm's statistics from a single bout's "
        "correlated frames are unreliable — see README/model.py). Only switch to batchnorm "
        "if you also raise --batch-size well above 1 (e.g. by using --frame-size < 1024).",
    )
    ap.add_argument(
        "--backbone-pooling", default="attention", choices=["attention", "avg"],
        help="'attention' (default) replaces global average pooling with a learned spatial "
        "attention pool, so the CNN can localise fine body-part detail (tail, paws) instead "
        "of always averaging over the whole crop — targets Immobile, which stayed at 0.000 "
        "recall through every class-imbalance fix, suggesting the problem is upstream of "
        "balancing. 'avg' reproduces the original pooling exactly (bit-for-bit — see "
        "model.py's tests) for comparison. Resuming from a pre-attention-pooling checkpoint "
        "with --backbone-pooling attention warm-starts every conv/norm weight; only the new "
        "attention-scoring layer starts fresh (see the 'loaded backbone' log line).",
    )
    ap.add_argument(
        "--use-flow", action="store_true",
        help="add a second-stream optical flow branch (dense motion between adjacent frames), "
        "fused into the per-frame embedding before camera_fusion/temporal — those stages are "
        "unaffected, so resuming a non-flow checkpoint with --use-flow warm-starts everything "
        "except the new flow_backbone/fusion_proj weights (see the 'loaded backbone' log line). "
        "Targets motion-defined classes (Locomotion, Grooming, Approaching/Following) that a "
        "single static frame can't disambiguate from a similar-looking but still pose.",
    )
    ap.add_argument("--flow-size", type=int, default=256,
                     help="resolution flow is computed/consumed at. Flow fields are inherently "
                     "low-frequency/smooth (unlike RGB texture), so downsampling here is a "
                     "deliberate, justified exception to training the RGB stream at native res.")
    ap.add_argument("--flow-scale", type=float, default=20.0,
                     help="pixels; raw flow is divided by this and clipped to [-1,1] before the "
                     "flow backbone sees it, so its input scale doesn't depend on --flow-size.")
    ap.add_argument("--flow-dim", type=int, default=64,
                     help="embedding size of the flow branch before it's concatenated with the "
                     "RGB embedding (d_model) and projected back down to d_model.")
    ap.add_argument(
        "--use-scale-feature", action="store_true",
        help="feed each frame's registration zoom factor (scale = 1024/max(bbox_w,bbox_h), "
        "from transform_meta.csv) into camera fusion as an extra per-frame signal. The upstream "
        "tracker crops each frame to the tracked bounding box and uniformly rescales it to a "
        "fixed output size — that preserves shape/aspect ratio but destroys absolute apparent "
        "size, so the model otherwise has no way to know how zoomed-in a given frame was. "
        "Requires --transform-meta-root. Always warm-startable (camera_fusion.scale_proj is a "
        "tiny Linear(1,d_model), present in every model — this flag only controls whether the "
        "dataset computes it and forward() is given a real value instead of None).",
    )
    ap.add_argument(
        "--transform-meta-root", default=None,
        help="dir of <session_id>/cam_N/transform_meta.csv, one per camera per recording "
        "session (see 2_build_h5_dataset.py's docstring for the join-key convention). Required "
        "when --use-scale-feature is set.",
    )
    ap.add_argument(
        "--dropout", type=float, default=0.1,
        help="dropout inside camera_fusion and the temporal transformer (CNNTransformer's "
        "constructor default, previously never exposed here — every run so far has silently "
        "used 0.1). Doesn't change any parameter shapes, so raising it is always an "
        "exact-match resume (no missing/unexpected keys, Adam momentum carries over).",
    )

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument(
        "--batch-size", type=int, default=1,
        help="physical batch size. At native 1024 resolution the raw input tensor itself "
        "(batch*num_cams*32*4*1024*1024 floats) is the memory bottleneck, not just the model, "
        "so this has to stay small — use --grad-accum-steps for a larger effective batch.",
    )
    ap.add_argument(
        "--grad-accum-steps", type=int, default=16,
        help="accumulate gradients over this many batches per optimizer step. "
        "effective batch size = batch-size * grad-accum-steps.",
    )
    ap.add_argument(
        "--num-workers", type=int, default=12,
        help="PNG decode of native-resolution frames is CPU-bound; with many cores available "
        "this can go fairly high, but each worker's prefetch buffer holds a full-resolution "
        "sample (~1-2GB at 1024px/3 cams) — watch host RAM (`free -h`) and lower this if it's "
        "climbing unboundedly rather than plateauing.",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=2)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument(
        "--min-class-weight", type=float, default=0.75,
        help="floor on any class's inverse-sqrt loss weight, as a fraction of the mean weight "
        "(0 = no floor, the original behaviour). Exploring's natural weight is 0.53 — the "
        "lowest of all 10 classes, since it's the most common — and it stayed at 0.000 F1 "
        "through two sampler strategies, pointing at the loss weight (not sampling) as its "
        "limiter. Default 0.75 lifts it and any other common class out of that zone.",
    )
    ap.add_argument(
        "--exclude-classes", default=None,
        help="comma-separated class names to drop entirely from train+eval (e.g. 'Immobile'). "
        "Remaining classes keep their relative order and get renumbered contiguously starting "
        "at 0. Resuming a checkpoint trained on the full class set warm-starts every kept "
        "class's classifier row by name (not position) — only the dropped class's row is lost.",
    )
    ap.add_argument(
        "--class-weight-multiplier", default=None,
        help="manual per-class multiplier on top of the inverse-sqrt+floor loss weight, as "
        "'ClassName:mult,ClassName:mult' (e.g. 'Tail investigation:1.3,Rearing:0.85'). Weights "
        "are renormalised to mean 1 after multipliers are applied, same as the floor. Use this "
        "to redirect gradient budget based on current per-class results rather than only "
        "training-set frequency — e.g. boosting classes still weak despite ample support, or "
        "easing off a class that's already strong.",
    )
    ap.add_argument(
        "--sampler", default="balanced", choices=["balanced", "none"],
        help="'balanced' (default) uses a WeightedRandomSampler (inverse-square-root class "
        "frequency, matching the loss weighting) on the train split, so rare classes are seen "
        "more often each epoch without fully equalizing exposure to common ones — targets "
        "classes that evaluate.py shows as high-precision/near-zero-recall (learned but "
        "under-triggered), which loss weighting alone doesn't fix. 'none' uses plain shuffling. "
        "Only affects train; eval always reflects the true class distribution.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--confusion-matrix-every", type=int, default=2,
        help="save an eval-split confusion matrix to {out-dir}/confusion_matrices/ every N "
        "epochs (and always on the final epoch). Uses eval predictions already computed for "
        "the epoch's metrics, so this adds no extra forward passes.",
    )
    ap.add_argument(
        "--keep-top-k", type=int, default=5,
        help="keep this many checkpoints in {out-dir}/top_checkpoints/, ranked by eval "
        "macro-F1 (not just the single best.pt, which gets silently overwritten the moment a "
        "later epoch beats it — there was previously no way to recover, say, epoch 134's exact "
        "weights once epoch 149 beat its score). Files are named "
        "epoch{N:03d}_f1{score:.4f}.pt; the worst-ranked one is deleted whenever a new epoch "
        "would push the count over k. 0 disables this (last.pt/best.pt are unaffected either way).",
    )
    ap.add_argument(
        "--resume", default=None,
        help="path to a checkpoint (e.g. {out-dir}/last.pt) to resume from — restores model "
        "and optimizer state, and continues from the next epoch. Lets you change speed-only "
        "knobs (--micro-batch-size, --no-checkpoint, --batch-size, --grad-accum-steps, "
        "--num-workers, --sampler) without losing progress. Architecture args (--num-cams, "
        "--d-model, --backbone, --backbone-norm) must match the checkpoint's — validated on load.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(exist_ok=True)
    (out_dir / "top_checkpoints").mkdir(exist_ok=True)
    log = setup_logging(out_dir)

    label_map_path = args.label_map or (Path(args.index_csv).parent / "label_map.json")
    full_label_map = json.loads(Path(label_map_path).read_text())

    excluded_names = set()
    if args.exclude_classes:
        excluded_names = {n.strip() for n in args.exclude_classes.split(",") if n.strip()}
        unknown = excluded_names - set(full_label_map.keys())
        if unknown:
            raise ValueError(f"--exclude-classes names not in label_map.json: {unknown}")

    exclude_label_ids = {full_label_map[n] for n in excluded_names}
    # kept classes renumbered contiguously, preserving their original relative order
    kept_old_ids = sorted(v for v in full_label_map.values() if v not in exclude_label_ids)
    old_id_to_new_id = {old: new for new, old in enumerate(kept_old_ids)}
    label_id_remap = old_id_to_new_id if excluded_names else None
    label_map = {
        name: old_id_to_new_id[old_id]
        for name, old_id in full_label_map.items()
        if old_id not in exclude_label_ids
    }
    if excluded_names:
        log.info(f"excluding classes: {sorted(excluded_names)} — {len(full_label_map)} -> {len(label_map)} classes")

    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(num_classes)]

    device = pick_device(args.gpu_id)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")
    log.info(f"device={device}, classes={num_classes}")

    log_scale_stats = None
    if args.use_scale_feature:
        if not args.transform_meta_root:
            raise ValueError("--use-scale-feature requires --transform-meta-root")
        all_sessions = pd.read_csv(args.index_csv)["session_id"].unique()
        log_scale_stats = compute_log_scale_stats(args.transform_meta_root, all_sessions)
        log.info(f"scale feature: log-scale stats over {len(all_sessions)} sessions — "
                 f"mean={log_scale_stats[0]:.4f} std={log_scale_stats[1]:.4f}")

    train_ds = RatBoutDataset(
        args.index_csv, args.h5, args.normstats, split="train",
        num_cams=args.num_cams, frame_size=args.frame_size,
        cam_selection="weighted_random", augment=True,
        compute_flow=args.use_flow, flow_size=args.flow_size, flow_scale=args.flow_scale,
        exclude_label_ids=exclude_label_ids, label_id_remap=label_id_remap,
        transform_meta_root=args.transform_meta_root, compute_scale=args.use_scale_feature,
        log_scale_stats=log_scale_stats,
    )
    eval_ds = RatBoutDataset(
        args.index_csv, args.h5, args.normstats, split="eval",
        num_cams=args.num_cams, frame_size=args.frame_size,
        cam_selection="topk", augment=False,
        duration_stats=train_ds.duration_stats(),
        compute_flow=args.use_flow, flow_size=args.flow_size, flow_scale=args.flow_scale,
        exclude_label_ids=exclude_label_ids, label_id_remap=label_id_remap,
        transform_meta_root=args.transform_meta_root, compute_scale=args.use_scale_feature,
        log_scale_stats=log_scale_stats,
    )
    log.info(f"train={len(train_ds)} eval={len(eval_ds)}")

    if args.sampler == "balanced":
        sampler = make_balanced_sampler(train_ds.df["label_id"].to_numpy(), num_classes)
        log.info("train sampler: balanced (WeightedRandomSampler, inverse-sqrt class frequency)")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
        )
    else:
        log.info("train sampler: none (plain shuffle)")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
        )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        prev_args = resume_ckpt["args"]
        # num_cams is deliberately NOT checked here: only CameraFusion.cam_embed depends on it
        # (an nn.Embedding(num_cams, d_model)) — every other module (backbone, flow_backbone,
        # the rest of camera_fusion, temporal, head) is num_cams-independent, so a camera-count
        # change is warm-startable the same way pooling/flow were, just with cam_embed missing.
        arch_keys = ["d_model", "backbone", "backbone_norm"]
        mismatches = {
            k: (prev_args.get(k), getattr(args, k))
            for k in arch_keys if prev_args.get(k) != getattr(args, k)
        }
        if mismatches:
            raise ValueError(
                f"--resume checkpoint was trained with different architecture args, "
                f"can't load its weights: {mismatches} (checkpoint value, current value)"
            )
        log.info(f"resuming from {args.resume} (checkpoint epoch {resume_ckpt['epoch']}, "
                 f"eval macro-F1 {resume_ckpt['eval_macro_f1']:.3f})")

    model = CNNTransformer(
        num_classes=num_classes, num_cams=args.num_cams, seq_len=32,
        d_model=args.d_model, backbone_arch=args.backbone, pretrained=(resume_ckpt is None),
        micro_batch_size=args.micro_batch_size, use_checkpoint=not args.no_checkpoint,
        backbone_norm=args.backbone_norm, backbone_pooling=args.backbone_pooling,
        use_flow=args.use_flow, flow_dim=args.flow_dim, dropout=args.dropout,
    ).to(device)
    if resume_ckpt is not None:
        prev_pooling = prev_args.get("backbone_pooling", "avg")  # older checkpoints predate this arg
        if prev_pooling != args.backbone_pooling:
            log.info(f"backbone pooling changed on resume: {prev_pooling} -> {args.backbone_pooling}")
        ckpt_sd = dict(resume_ckpt["model"])
        prev_num_cams = prev_args.get("num_cams")
        if prev_num_cams != args.num_cams:
            # cam_embed is an nn.Embedding(num_cams, d_model) — same key name, different shape,
            # so a plain strict=False load would crash on a size mismatch (unlike the pooling/
            # flow additions, which are entirely new key names). Pop it so it's reported as
            # "missing" and initializes fresh, instead of crashing.
            ckpt_sd.pop("camera_fusion.cam_embed.weight", None)
            log.info(f"num_cams changed on resume: {prev_num_cams} -> {args.num_cams} — "
                     f"camera_fusion.cam_embed restarts fresh, everything else warm-starts")

        prev_label_map = resume_ckpt.get("label_map", label_map)
        remapped_classes = []
        if set(prev_label_map.keys()) != set(label_map.keys()):
            # class set changed (e.g. --exclude-classes): the final classifier row for each
            # class means something different by *name*, not by position, so a plain
            # strict=False load (which matches by tensor name+shape) can't handle this — it
            # would either crash on the shape mismatch or silently misalign rows. Remap by hand:
            # copy each kept class's row from its old position to its new one, drop the rest.
            old_w, old_b = ckpt_sd.pop("head.1.weight"), ckpt_sd.pop("head.1.bias")
            new_w, new_b = model.head[1].weight.data.clone(), model.head[1].bias.data.clone()
            for name, new_id in label_map.items():
                if name in prev_label_map:
                    new_w[new_id] = old_w[prev_label_map[name]]
                    new_b[new_id] = old_b[prev_label_map[name]]
                    remapped_classes.append(name)
            dropped = sorted(set(prev_label_map.keys()) - set(label_map.keys()))
            fresh = sorted(set(label_map.keys()) - set(prev_label_map.keys()))
            log.info(f"class set changed on resume — {len(remapped_classes)} classifier rows "
                     f"warm-started by name, dropped: {dropped or 'none'}, fresh (untrained): {fresh or 'none'}")

        # strict=False on purpose: --backbone-pooling, --use-flow, and --num-cams are all
        # allowed to differ from the checkpoint (that's the warm-start path — every matching
        # module loads by name, only the new/resized module's params are missing). Log what's
        # missing/unexpected so a *real* mismatch elsewhere doesn't silently pass. Also expect
        # "camera_fusion.scale_proj.*" missing on any checkpoint saved before the scale-feature
        # change — that submodule is now always part of CameraFusion (see model.py), so it's a
        # one-time warm-start gap regardless of whether --use-scale-feature is even set here.
        missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
        if remapped_classes:
            model.head[1].weight.data.copy_(new_w)
            model.head[1].bias.data.copy_(new_b)
        log.info(f"loaded checkpoint weights — missing: {missing or 'none'}, unexpected: {unexpected or 'none'}")

    if args.no_class_weights:
        class_weights = None
    else:
        multipliers = None
        if args.class_weight_multiplier:
            mult_by_name = {}
            for item in args.class_weight_multiplier.split(","):
                name, _, val = item.rpartition(":")
                name = name.strip()
                if name not in label_map:
                    raise ValueError(f"--class-weight-multiplier references unknown class {name!r}")
                mult_by_name[name] = float(val)
            multipliers = [mult_by_name.get(n, 1.0) for n in class_names]
            log.info(f"class weight multipliers: {dict(zip(class_names, multipliers))}")
        class_weights = class_weights_inv_sqrt_freq(
            train_ds.df["label_id"].to_numpy(), num_classes,
            min_weight=args.min_class_weight, multipliers=multipliers,
        ).to(device)
        log.info(f"class weights: {dict(zip(class_names, [round(w, 2) for w in class_weights.tolist()]))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    backbone_params = list(model.backbone_parameters())
    head_params = list(model.head_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [args.lr * args.backbone_lr_mult, args.lr]

    epoch_start = 0
    best_f1 = -1.0
    top_ckpt_dir = out_dir / "top_checkpoints"
    top_ckpts = scan_top_checkpoints(top_ckpt_dir)
    if top_ckpts:
        log.info(f"found {len(top_ckpts)} existing top_checkpoints/: "
                 f"{[(e, round(f, 4)) for f, e, _ in top_ckpts]}")
    if resume_ckpt is not None:
        # Adam's saved momentum/variance buffers are keyed to the checkpoint's exact parameter
        # set. If the pooling module differs (missing/unexpected non-empty above), the new
        # model has a different parameter count and loading the old optimizer state would
        # crash — and would be meaningless anyway, since there's no momentum history for a
        # brand-new parameter. Only restore it on an exact match; the model *weights* just
        # loaded are unaffected either way, only Adam's momentum resets (re-accumulates within
        # a few steps — far cheaper than losing the warm-started weights would be).
        exact_match = not missing and not unexpected
        if "optimizer" in resume_ckpt and exact_match:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        elif "optimizer" in resume_ckpt:
            log.info("skipping optimizer-state resume (architecture changed) — starting Adam "
                     "momentum fresh; model weights above are still warm-started.")
        epoch_start = resume_ckpt["epoch"] + 1
        best_f1 = resume_ckpt["eval_macro_f1"]

    for p in backbone_params:
        p.requires_grad = epoch_start >= args.freeze_backbone_epochs

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    # older checkpoints (pre-resume-support) don't carry global_step; approximate it from
    # epoch_start so the cosine LR schedule doesn't restart its warmup on resume.
    global_step = resume_ckpt.get("global_step", epoch_start * steps_per_epoch) if resume_ckpt else 0
    log.info(
        f"physical batch={args.batch_size}, grad_accum={args.grad_accum_steps} "
        f"-> effective batch={args.batch_size * args.grad_accum_steps}, "
        f"{steps_per_epoch} optimizer steps/epoch"
    )
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    history_path = out_dir / "metrics.csv"
    if resume_ckpt is None or not history_path.exists():
        with open(history_path, "w") as fh:
            fh.write("epoch,train_loss,train_acc,train_f1,eval_loss,eval_acc,eval_f1,seconds\n")

    for epoch in range(epoch_start, args.epochs):
        if epoch == args.freeze_backbone_epochs:
            log.info(f"unfreezing backbone at epoch {epoch}")
            for p in backbone_params:
                p.requires_grad = True

        t0 = time.time()
        train_metrics, global_step, _, _ = run_epoch(
            model, train_loader, device, criterion, optimizer, scaler,
            cosine_warmup_lr, global_step, total_steps, warmup_steps, base_lrs, train=True,
            grad_accum_steps=args.grad_accum_steps,
        )
        with torch.no_grad():
            eval_metrics, _, eval_labels, eval_preds = run_epoch(
                model, eval_loader, device, criterion, optimizer, scaler,
                cosine_warmup_lr, global_step, total_steps, warmup_steps, base_lrs, train=False,
                grad_accum_steps=1,
            )
        dt = time.time() - t0

        log.info(
            f"epoch {epoch:03d} | train loss {train_metrics['loss']:.3f} acc {train_metrics['acc']:.3f} "
            f"f1 {train_metrics['macro_f1']:.3f} | eval loss {eval_metrics['loss']:.3f} "
            f"acc {eval_metrics['acc']:.3f} f1 {eval_metrics['macro_f1']:.3f} | {dt:.0f}s"
        )
        with open(history_path, "a") as fh:
            fh.write(
                f"{epoch},{train_metrics['loss']:.4f},{train_metrics['acc']:.4f},{train_metrics['macro_f1']:.4f},"
                f"{eval_metrics['loss']:.4f},{eval_metrics['acc']:.4f},{eval_metrics['macro_f1']:.4f},{dt:.1f}\n"
            )

        is_last_epoch = epoch == args.epochs - 1
        if (epoch + 1) % args.confusion_matrix_every == 0 or is_last_epoch:
            cm_path = out_dir / "confusion_matrices" / f"epoch_{epoch:03d}.png"
            save_confusion_matrix(eval_labels, eval_preds, class_names, cm_path,
                                   title=f"eval — epoch {epoch}")
            log.info(f"saved confusion matrix -> {cm_path}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "args": vars(args),
            "label_map": label_map,
            "epoch": epoch,
            "eval_macro_f1": eval_metrics["macro_f1"],
        }
        torch.save(ckpt, out_dir / "last.pt")
        if eval_metrics["macro_f1"] > best_f1:
            best_f1 = eval_metrics["macro_f1"]
            torch.save(ckpt, out_dir / "best.pt")
            log.info(f"  -> new best eval macro-F1 {best_f1:.3f}, saved best.pt")
        top_ckpts = update_top_checkpoints(
            top_ckpt_dir, top_ckpts, epoch, eval_metrics["macro_f1"], ckpt, args.keep_top_k, log
        )

    log.info(f"done. best eval macro-F1={best_f1:.3f}")


if __name__ == "__main__":
    main()
