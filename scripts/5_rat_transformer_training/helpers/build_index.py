"""
Build the training manifest that joins rat_behaviour_32f.h5 (bout windows) to
train_eval_split.csv (behavior labels).

The .h5 has no explicit label field. Each top-level group is named
    {session_id}__{rat}__{ordinal:05d}
where `ordinal` is reconstructed here as the row's position within
train_eval_split.csv for that (session_id, rat) pair, in file order. This
script performs that join, reports how well it matched (so you can sanity
check before trusting it), and computes per-camera mask-visibility (fraction
of non-zero mask pixels, averaged over a subsample of the bout's 32 frames)
so the dataset/model can pick or weight cameras without decoding all 6 for
every bout during training.

Usage:
    python helpers/build_index.py \
        --h5 /path/to/rat_behaviour_32f.h5 \
        --split-csv /path/to/train_eval_split.csv \
        --out-dir /path/to/rat_transformer_training
"""
import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CAMS = [f"cam_{i}" for i in range(1, 7)]
VIS_THUMB = 96  # downsize masks to this before counting nonzero pixels (speed)


def decode_mask_coverage(raw_bytes_arr):
    """raw_bytes_arr: object array of per-frame PNG-encoded mask bytes.
    Returns mean fraction of non-zero pixels across the given frames."""
    fracs = []
    for b in raw_bytes_arr:
        im = Image.open(io.BytesIO(b.tobytes())).convert("L")
        im = im.resize((VIS_THUMB, VIS_THUMB), Image.NEAREST)
        arr = np.asarray(im)
        fracs.append(float((arr > 0).mean()))
    return float(np.mean(fracs)) if fracs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--num-vis-samples",
        type=int,
        default=8,
        help="how many of the 32 frames per (bout,cam) to decode when estimating "
        "camera visibility (subsample for speed).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(args.h5, "r")
    h5_keys = set(f["data"].keys())
    classes = [c.decode() if isinstance(c, bytes) else c for c in f["classes"][:]]
    label_map = {name: i for i, name in enumerate(classes)}
    print(f"[build_index] {len(h5_keys)} bout groups in .h5, {len(classes)} classes: {classes}")

    split_df = pd.read_csv(args.split_csv)
    print(f"[build_index] {len(split_df)} rows in split csv")

    # ordinal = row position within (session_id, rat), preserving file order. Counted BEFORE
    # dropping 'point' rows: 2_build_h5_dataset.py names each bout by its row index in the label
    # CSV and skips point rows without renumbering, so counting after the drop would join every
    # later bout of that file to the wrong .h5 group (and label it with another bout's behavior).
    split_df["ordinal"] = split_df.groupby(["session_id", "rat"]).cumcount()

    # 'point'-type rows are instantaneous markers with no corresponding bout window;
    # 'segment' and blank/NaN type rows are both real bouts (segment_count + blank_count
    # == the total bout count reported elsewhere for this dataset).
    n_before = len(split_df)
    split_df = split_df[split_df["type"] != "point"].copy()
    print(f"[build_index] dropped {n_before - len(split_df)} 'point'-type rows (not bout windows)")

    # session_id spelling can differ between the split csv and the h5 (observed: word-order
    # swap, e.g. "Unfamiliar_rat_for_calibration_..." vs "RAT_unfamiliar_for_calibration_...").
    # Resolve by exact match first, falling back to a token-set match (underscore-split,
    # lowercased, order-independent) so such renames don't silently break the join.
    h5_sessions = set(k.rsplit("__", 2)[0] for k in h5_keys)

    def tokenize(s):
        return tuple(sorted(s.lower().split("_")))

    h5_by_token = {}
    for s in h5_sessions:
        h5_by_token.setdefault(tokenize(s), s)
    alias = {}
    for s in split_df["session_id"].unique():
        alias[s] = s if s in h5_sessions else h5_by_token.get(tokenize(s))
    unresolved = [s for s, v in alias.items() if v is None]
    if unresolved:
        print(f"[build_index] WARNING: {len(unresolved)} session_id(s) have no h5 counterpart "
              f"even after token-set matching: {unresolved}", file=sys.stderr)
    split_df["session_h5"] = split_df["session_id"].map(alias)

    split_df["h5_key"] = (
        split_df["session_h5"].astype(str) + "__" + split_df["rat"] + "__"
        + split_df["ordinal"].apply(lambda x: f"{int(x):05d}")
    )

    matched = split_df[split_df["h5_key"].isin(h5_keys)].copy()
    unmatched_rows = split_df[~split_df["h5_key"].isin(h5_keys)].copy()
    orphan_keys = sorted(h5_keys - set(split_df["h5_key"]))

    match_rate = len(matched) / len(split_df) if len(split_df) else 0.0
    print(
        f"[build_index] matched {len(matched)}/{len(split_df)} split rows "
        f"({match_rate:.1%}); {len(orphan_keys)} h5 groups had no matching split row"
    )
    if match_rate < 0.9:
        print(
            "[build_index] WARNING: match rate below 90% — the ordinal-join "
            "assumption may not hold for this export. Inspect unmatched_report.csv "
            "before trusting window_index.csv.",
            file=sys.stderr,
        )

    unknown_behavior = matched[~matched["behavior"].isin(label_map)]
    if len(unknown_behavior):
        print(
            f"[build_index] dropping {len(unknown_behavior)} matched rows with a "
            f"behavior not in classes list: {sorted(unknown_behavior['behavior'].unique())}"
        )
    matched = matched[matched["behavior"].isin(label_map)].copy()
    matched["label_id"] = matched["behavior"].map(label_map)

    # per-camera visibility
    vis_cols = {c: [] for c in CAMS}
    for h5_key in tqdm(matched["h5_key"], desc="computing camera visibility"):
        grp = f["data"][h5_key]
        for cam in CAMS:
            if cam not in grp:
                vis_cols[cam].append(0.0)
                continue
            mask_ds = grp[cam]["mask"]
            n = mask_ds.shape[0]
            idx = np.linspace(0, n - 1, min(args.num_vis_samples, n)).astype(int)
            vis_cols[cam].append(decode_mask_coverage(mask_ds[idx]))

    for cam in CAMS:
        matched[f"{cam}_vis"] = vis_cols[cam]

    def top_cams(row, k=3):
        ranked = sorted(CAMS, key=lambda c: row[f"{c}_vis"], reverse=True)
        return ",".join(ranked[:k])

    matched["best_cams"] = matched.apply(top_cams, axis=1)

    keep_cols = [
        "h5_key", "session_id", "rat", "behavior", "label_id", "split",
        "start_frame", "end_frame", "duration_frames",
    ] + [f"{c}_vis" for c in CAMS] + ["best_cams"]
    manifest = matched[keep_cols].reset_index(drop=True)

    manifest_path = out_dir / "window_index.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[build_index] wrote {manifest_path} ({len(manifest)} rows)")

    label_map_path = out_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2))
    print(f"[build_index] wrote {label_map_path}")

    report_path = out_dir / "unmatched_report.csv"
    pd.DataFrame(
        {
            "unmatched_split_row_h5_key": list(unmatched_rows["h5_key"]) + [None] * len(orphan_keys),
            "orphan_h5_key": [None] * len(unmatched_rows) + orphan_keys,
        }
    ).to_csv(report_path, index=False)
    print(f"[build_index] wrote {report_path} for manual inspection")

    print()
    print("split counts:\n", manifest["split"].value_counts())
    print()
    print("class counts:\n", manifest["behavior"].value_counts())


if __name__ == "__main__":
    main()
