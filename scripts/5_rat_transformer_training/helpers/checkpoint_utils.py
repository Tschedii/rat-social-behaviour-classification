"""Shared model/dataset reconstruction from a saved checkpoint. Factored out of evaluate.py
so that evaluate.py, average_checkpoints.py, and calibrate.py all agree exactly on how to
rebuild a runnable model + matching dataset from a .pt file — one definition of "how do you
load this checkpoint correctly," not three copies that can silently drift apart.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from helpers.rat_h5_dataset import RatBoutDataset, compute_log_scale_stats
from model import CNNTransformer


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt["args"]
    label_map = ckpt["label_map"]
    idx_to_label = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return ckpt, train_args, label_map, idx_to_label


def resolve_exclude_remap(train_args, index_csv, label_map, label_map_path=None):
    """--exclude-classes means the checkpoint's label_map is already reduced/renumbered, but
    window_index.csv's label_id column still uses the ORIGINAL full numbering — this
    reconstructs the same exclude+remap the training run used, and cross-checks it against the
    checkpoint's own saved label_map so a mismatched --label-map path fails loudly, not
    silently."""
    exclude_label_ids, label_id_remap = set(), None
    if train_args.get("exclude_classes"):
        label_map_path = label_map_path or (Path(index_csv).parent / "label_map.json")
        full_label_map = json.loads(Path(label_map_path).read_text())
        excluded_names = {n.strip() for n in train_args["exclude_classes"].split(",") if n.strip()}
        exclude_label_ids = {full_label_map[n] for n in excluded_names}
        kept_old_ids = sorted(v for v in full_label_map.values() if v not in exclude_label_ids)
        label_id_remap = {old: new for new, old in enumerate(kept_old_ids)}
        rebuilt = {n: label_id_remap[old_id] for n, old_id in full_label_map.items() if old_id not in exclude_label_ids}
        if rebuilt != label_map:
            raise ValueError(
                f"reconstructed label_map from --exclude-classes doesn't match the checkpoint's "
                f"saved label_map — {rebuilt} vs {label_map}. Check --label-map points at the "
                f"same label_map.json the checkpoint was trained against."
            )
    return exclude_label_ids, label_id_remap


def build_model(train_args, label_map, device):
    use_flow = train_args.get("use_flow", False)
    return CNNTransformer(
        num_classes=len(label_map), num_cams=train_args["num_cams"], seq_len=32,
        d_model=train_args["d_model"], backbone_arch=train_args["backbone"], pretrained=False,
        micro_batch_size=train_args.get("micro_batch_size", 4),
        use_checkpoint=not train_args.get("no_checkpoint", False),
        backbone_norm=train_args.get("backbone_norm", "groupnorm"),
        backbone_pooling=train_args.get("backbone_pooling", "avg"),
        use_flow=use_flow, flow_dim=train_args.get("flow_dim", 64),
    ).to(device)


def build_dataset(h5, index_csv, normstats, split, train_args, exclude_label_ids, label_id_remap,
                   duration_stats=None):
    use_flow = train_args.get("use_flow", False)
    use_scale = train_args.get("use_scale_feature", False)
    log_scale_stats = None
    if use_scale:
        all_sessions = pd.read_csv(index_csv)["session_id"].unique()
        log_scale_stats = compute_log_scale_stats(train_args["transform_meta_root"], all_sessions)
    return RatBoutDataset(
        index_csv, h5, normstats, split=split,
        num_cams=train_args["num_cams"], frame_size=train_args["frame_size"],
        cam_selection="topk", augment=False,
        duration_stats=duration_stats,
        compute_flow=use_flow, flow_size=train_args.get("flow_size", 256),
        flow_scale=train_args.get("flow_scale", 20.0),
        exclude_label_ids=exclude_label_ids, label_id_remap=label_id_remap,
        transform_meta_root=train_args.get("transform_meta_root"), compute_scale=use_scale,
        log_scale_stats=log_scale_stats,
    )


def load_model_and_datasets(checkpoint_path, h5, index_csv, normstats, split, device, label_map_path=None):
    """One-call convenience: everything evaluate.py needs to go from a checkpoint path to a
    loaded model + ready DataLoader-able dataset for `split`, duration-standardized against
    the checkpoint's own train split (matching what it was trained/evaluated with)."""
    ckpt, train_args, label_map, idx_to_label = load_checkpoint(checkpoint_path, device)
    exclude_label_ids, label_id_remap = resolve_exclude_remap(train_args, index_csv, label_map, label_map_path)

    train_ds_for_stats = build_dataset(h5, index_csv, normstats, "train", train_args,
                                        exclude_label_ids, label_id_remap)
    ds = build_dataset(h5, index_csv, normstats, split, train_args, exclude_label_ids, label_id_remap,
                        duration_stats=train_ds_for_stats.duration_stats())

    model = build_model(train_args, label_map, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ds, ckpt, train_args, label_map, idx_to_label
