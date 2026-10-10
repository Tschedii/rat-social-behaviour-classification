"""Standalone evaluation of a trained checkpoint: classification report +
confusion matrix, on a given split (default: eval).

    python evaluate.py \
        --h5 /path/to/rat_behaviour_32f.h5 \
        --index-csv /path/to/rat_transformer_training/window_index.csv \
        --normstats /path/to/rgb_normstats.json \
        --checkpoint /path/to/rat_transformer_training/checkpoints/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from helpers.checkpoint_utils import load_model_and_datasets
from helpers.confusion_matrix import save_confusion_matrix
from train import pick_device


def run_inference(model, loader, device):
    """Returns (labels, preds, logits) as numpy arrays, in the dataset's row order
    (loader must use shuffle=False for that order to match ds.df)."""
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            images = batch["images"].to(device)
            cam_conf = batch["cam_conf"].to(device)
            duration = batch["duration"].to(device)
            flow = batch["flow"].to(device) if "flow" in batch else None
            scale = batch["scale"].to(device) if "scale" in batch else None
            logits = model(images, cam_conf, duration, flow=flow, scale=scale)
            all_logits.append(logits.cpu().numpy())
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(batch["label"].numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--normstats", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default=None, help="defaults to label_map.json next to index-csv")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default=None, help="defaults to checkpoint's directory")
    args = ap.parse_args()

    device = pick_device()
    model, ds, ckpt, train_args, label_map, idx_to_label = load_model_and_datasets(
        args.checkpoint, args.h5, args.index_csv, args.normstats, args.split, device, args.label_map,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    labels, preds, _logits = run_inference(model, loader, device)

    report = classification_report(
        labels, preds, labels=list(range(len(idx_to_label))),
        target_names=idx_to_label, digits=3, zero_division=0,
    )
    print(report)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"classification_report_{args.split}.txt").write_text(report)

    fig_path = out_dir / f"confusion_matrix_{args.split}.png"
    save_confusion_matrix(labels, preds, idx_to_label, fig_path, title=f"{args.split} split")
    print(f"wrote {fig_path}")

    # Per-sample predictions, keyed by h5_key (loader uses shuffle=False, so ds.df's row
    # order matches preds/labels exactly) — feeds helpers/occlusion_diagnostic.py, which
    # checks whether errors correlate with camera visibility without needing a GPU.
    per_sample = ds.df[["h5_key", "behavior"]].copy()
    per_sample["true_label_id"] = labels
    per_sample["pred_label_id"] = preds
    per_sample["correct"] = (labels == preds)
    per_sample_path = out_dir / f"predictions_{args.split}.csv"
    per_sample.to_csv(per_sample_path, index=False)
    print(f"wrote {per_sample_path}")


if __name__ == "__main__":
    main()
