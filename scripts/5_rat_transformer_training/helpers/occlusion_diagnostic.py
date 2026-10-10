"""Cheap, no-GPU diagnostic: does occlusion (low camera visibility) actually explain the
model's errors? Answers this *before* committing a GPU to the expensive 6-camera experiment.

Correlates each eval bout's correctness (from evaluate.py's predictions_eval.csv) with the
visibility of whichever cameras were actually used at eval time — RatBoutDataset's "topk"
selection deterministically picks the top --num-cams by cam_N_vis, so that's recomputed here
directly from window_index.csv rather than needing evaluate.py to export camera IDs.

Usage (run evaluate.py first — it now also writes predictions_<split>.csv):
    python helpers/occlusion_diagnostic.py \
        --index-csv window_index.csv \
        --predictions training_output/predictions_eval.csv \
        --num-cams 3
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

CAMS = [f"cam_{i}" for i in range(1, 7)]


def top_k_mean_visibility(row, k):
    vis = sorted((row[f"{c}_vis"] for c in CAMS), reverse=True)
    return float(np.mean(vis[:k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--predictions", required=True, help="predictions_<split>.csv from evaluate.py")
    ap.add_argument("--num-cams", type=int, default=3, help="must match the run's --num-cams")
    args = ap.parse_args()

    index_df = pd.read_csv(args.index_csv)
    preds_df = pd.read_csv(args.predictions)
    df = preds_df.merge(index_df, on="h5_key", how="left", suffixes=("", "_idx"))
    missing = df[f"cam_1_vis"].isna().sum()
    if missing:
        print(f"WARNING: {missing}/{len(df)} predicted rows had no matching window_index.csv "
              f"row (h5_key mismatch) — dropping them from the analysis.")
        df = df.dropna(subset=["cam_1_vis"])

    df["used_cam_vis"] = df.apply(lambda r: top_k_mean_visibility(r, args.num_cams), axis=1)

    correct = df[df["correct"]]["used_cam_vis"]
    incorrect = df[~df["correct"]]["used_cam_vis"]

    print(f"n={len(df)} (correct={len(correct)}, incorrect={len(incorrect)})")
    print(f"mean visibility | correct: {correct.mean():.3f}   incorrect: {incorrect.mean():.3f}")
    print(f"median visibility | correct: {correct.median():.3f}   incorrect: {incorrect.median():.3f}")

    if len(correct) > 1 and len(incorrect) > 1:
        u_stat, p_value = stats.mannwhitneyu(correct, incorrect, alternative="greater")
        print(f"\nMann-Whitney U test (H1: correct predictions have HIGHER visibility): "
              f"p = {p_value:.4f}")
        if p_value < 0.05:
            print("=> SIGNIFICANT: misclassifications are associated with lower camera "
                  "visibility. Occlusion looks like a real contributor — the 6-camera "
                  "experiment is plausibly worth the GPU time.")
        else:
            print("=> NOT significant at p<0.05: no clear evidence that low visibility drives "
                  "the current errors. The 6-camera experiment is a lower-priority bet than "
                  "the architecture/loss-weight changes.")
    else:
        print("not enough samples in one group to run a statistical test.")

    print("\nPer-class breakdown (mean used-camera visibility, correct vs incorrect):")
    for behavior, g in df.groupby("behavior"):
        c, i = g[g["correct"]]["used_cam_vis"], g[~g["correct"]]["used_cam_vis"]
        c_str = f"{c.mean():.3f} (n={len(c)})" if len(c) else "n/a"
        i_str = f"{i.mean():.3f} (n={len(i)})" if len(i) else "n/a"
        print(f"  {behavior:24s} correct: {c_str:16s} incorrect: {i_str}")


if __name__ == "__main__":
    main()
