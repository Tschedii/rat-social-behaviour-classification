"""Per-class logit-bias calibration, fit on a held-out slice of the TRAIN split (never eval —
fitting calibration against the same split you report the final number on would be tuning
directly against the metric, not a legitimate improvement).

A per-class additive bias on the logits is exactly equivalent to shifting the final
classifier layer's bias term (head[1].bias in model.py — logits = Linear(features), so
logits + b == Linear with bias' = bias + b). That means calibration bakes directly into a
real, standalone checkpoint: no separate bias file to keep track of, no special-cased
inference path — evaluate.py loads a calibrated checkpoint exactly like any other.

    python helpers/calibrate.py \
        --h5 /path/to/rat_behaviour_32f.h5 \
        --index-csv window_index.csv \
        --normstats rgb_normstats.json \
        --checkpoint training_output/averaged_top5.pt \
        --out training_output/calibrated_top5avg.pt

Usage after this: run evaluate.py against --out on the eval split for the real, reportable
number, then helpers/population_weighted_accuracy.py on the resulting report — the score this
script prints is measured on the calibration slice, which exists purely to fit the bias and
should not be quoted as a result.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Run as `python helpers/calibrate.py` (matching every other script in this project), so
# sys.path[0] is this file's own directory (helpers/), not the project root — evaluate.py,
# model.py, and train.py all live one level up. Insert that explicitly rather than relying on
# cwd, since sys.path[0] is set from __file__'s location regardless of what directory the
# command was run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import run_inference  # noqa: E402
from helpers.checkpoint_utils import (  # noqa: E402
    build_dataset, build_model, load_checkpoint, resolve_exclude_remap,
)
from helpers.population_weighted_accuracy import POPULATION_COUNTS  # noqa: E402
from train import pick_device  # noqa: E402


def carve_calib_split(train_ds, frac, seed):
    """Stratified sample of `frac` of the train split's rows, by behavior — held out purely
    for fitting calibration, never used for gradient updates or reported as an eval number.
    Mutates train_ds.df in place to the sampled subset and returns it; call this on a
    dataset object you don't need the full train split from afterward."""
    calib_df = (
        train_ds.df.groupby("behavior", group_keys=False)[train_ds.df.columns]
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )
    train_ds.df = calib_df
    return train_ds


def macro_f1_scorer(num_classes):
    """Unweighted macro-F1 — treats every class as equally important regardless of how common
    it actually is. Available for comparison, but NOT the default: on this project it turned
    out to actively mislead the fit (see population_weighted_accuracy_scorer's docstring)."""
    def score(logits, labels, bias):
        preds = (logits + bias).argmax(axis=1)
        return f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0)
    return score


def population_weighted_accuracy_scorer(class_names, population_counts):
    """Weights each class's calibration-slice recall by its TRUE population share (from
    population_weighted_accuracy.py's reference table), not by macro-F1's implicit "every
    class counts equally" or the eval split's accidental proportions. This is the default for
    a concrete, measured reason: calibrating epoch 155 against macro-F1 pushed Exploring's
    recall 0.300->0.650 by sacrificing Approaching/Following, Body touch, and Grooming
    (0.457->0.272, 0.411->0.329, 0.469->0.306) — a trade macro-F1 rates as a big win because it
    weights all 9 classes equally, but which only barely improved real population-weighted
    accuracy (46.8%->48.3%) because it overshot how much to correct for a class that's common,
    not infinitely important. Optimizing population-weighted accuracy directly avoids that
    overshoot by construction."""
    total_w = sum(population_counts[c] for c in class_names)

    def score(logits, labels, bias):
        preds = (logits + bias).argmax(axis=1)
        acc = 0.0
        for idx, name in enumerate(class_names):
            mask = labels == idx
            if mask.sum() == 0:
                continue
            recall_c = (preds[mask] == idx).mean()
            acc += population_counts[name] * recall_c
        return acc / total_w
    return score


def fit_per_class_bias(logits, labels, num_classes, score_fn, n_passes=4, lo=-6.0, hi=6.0, step=0.1):
    """Coordinate ascent: repeatedly sweep each class's bias over a grid, holding the others
    fixed, picking whichever value maximizes score_fn(logits, labels, bias) on the calibration
    slice. Cheap — everything here is numpy on already-computed logits, no forward passes — so
    a fine grid and several passes cost nothing. Converges within a couple of passes in
    practice since each class's optimum doesn't depend strongly on the others once they're
    roughly sane."""
    bias = np.zeros(num_classes, dtype=np.float64)
    candidates = np.arange(lo, hi + step / 2, step)

    best_score = score_fn(logits, labels, bias)
    for _ in range(n_passes):
        improved = False
        for c in range(num_classes):
            best_c, best_c_score = bias[c], best_score
            for cand in candidates:
                trial = bias.copy()
                trial[c] = cand
                s = score_fn(logits, labels, trial)
                if s > best_c_score:
                    best_c, best_c_score = cand, s
            if best_c != bias[c]:
                bias[c] = best_c
                best_score = best_c_score
                improved = True
        if not improved:
            break
    return bias, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--normstats", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label-map", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-frac", type=float, default=0.15,
                     help="fraction of the TRAIN split (stratified by class) held out to fit "
                          "calibration. Never touches eval.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--objective", default="population_weighted_accuracy",
        choices=["population_weighted_accuracy", "macro_f1"],
        help="what the coordinate-ascent search maximizes on the calibration slice. Default "
        "weights each class's recall by its true share of the full dataset (see "
        "population_weighted_accuracy.py) — 'macro_f1' weights every class equally regardless "
        "of prevalence, which on this project overshot: it pushed Exploring's recall a long "
        "way by sacrificing several more-common classes, for only a small real-world gain.",
    )
    args = ap.parse_args()

    device = pick_device()
    ckpt, train_args, label_map, idx_to_label = load_checkpoint(args.checkpoint, device)
    num_classes = len(label_map)
    exclude_label_ids, label_id_remap = resolve_exclude_remap(train_args, args.index_csv, label_map, args.label_map)

    train_ds_for_stats = build_dataset(args.h5, args.index_csv, args.normstats, "train", train_args,
                                        exclude_label_ids, label_id_remap)
    duration_stats = train_ds_for_stats.duration_stats()

    calib_ds = build_dataset(args.h5, args.index_csv, args.normstats, "train", train_args,
                              exclude_label_ids, label_id_remap, duration_stats=duration_stats)
    calib_ds = carve_calib_split(calib_ds, args.calib_frac, args.seed)
    print(f"calibration slice: {len(calib_ds)} bouts held out from train "
          f"({args.calib_frac:.0%}, stratified by class, seed={args.seed})")

    model = build_model(train_args, label_map, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.objective == "population_weighted_accuracy":
        missing = set(idx_to_label) - set(POPULATION_COUNTS)
        if missing:
            raise ValueError(f"POPULATION_COUNTS (in population_weighted_accuracy.py) doesn't "
                              f"cover {missing} — update it to match the current label set, or "
                              f"pass --objective macro_f1 instead.")
        score_fn = population_weighted_accuracy_scorer(idx_to_label, POPULATION_COUNTS)
    else:
        score_fn = macro_f1_scorer(num_classes)

    loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    labels, preds, logits = run_inference(model, loader, device)
    zero_bias = np.zeros(num_classes, dtype=np.float64)
    uncalibrated_score = score_fn(logits, labels, zero_bias)
    print(f"calib-slice {args.objective} before calibration: {uncalibrated_score:.4f}")

    bias, calibrated_score = fit_per_class_bias(logits, labels, num_classes, score_fn)
    print(f"calib-slice {args.objective} after calibration:  {calibrated_score:.4f}")
    print("fitted per-class bias:", dict(zip(idx_to_label, np.round(bias, 3).tolist())))

    with torch.no_grad():
        model.head[1].bias += torch.tensor(bias, dtype=model.head[1].bias.dtype, device=model.head[1].bias.device)

    out_ckpt = dict(ckpt)
    out_ckpt["model"] = model.state_dict()
    out_ckpt["calibrated_from"] = {
        "path": str(Path(args.checkpoint).resolve()),
        "source_epoch": ckpt.get("epoch"),
        "source_eval_macro_f1": ckpt.get("eval_macro_f1"),
        "calib_frac": args.calib_frac,
        "calib_seed": args.seed,
        "calib_slice_size": len(calib_ds),
        "objective": args.objective,
        "calib_score_before": float(uncalibrated_score),
        "calib_score_after": float(calibrated_score),
        "bias": dict(zip(idx_to_label, bias.tolist())),
    }
    out_ckpt.pop("optimizer", None)  # bias-shifted weights have no matching momentum history
    out_ckpt["eval_macro_f1"] = None  # unknown on the real eval split until evaluate.py is run
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, args.out)
    print(f"\nwrote {args.out}")
    print("run evaluate.py against it on --split eval next, then population_weighted_accuracy.py "
          "on the report, for the real, reportable numbers — the score above is on the "
          "calibration slice, not eval.")


if __name__ == "__main__":
    main()
