"""
3_build_training_data.py
========================
Builds the model-ready training tensors from the per-experiment `cutouts`
folders produced upstream (all_objects / object_1 / object_2).


python /path/to/master_thesis_social_behaviour/scripts/2_preprocessing_pgn/2_build_training_data.py --input ~/projects/data_master_thesis/experimental_data/Day_14_rPEG007vsNovel_1_29_06_2026_locomotion --output training_data --size 1024 --metadata



For every frame it emits THREE spatially-registered outputs:

  rgb/<key>.png            512x512x3  — both rats together, background zeroed
  mask_object_1/<key>.png  512x512    — uniform binary mask (0 / 255) of object 1
  mask_object_2/<key>.png  512x512    — uniform binary mask (0 / 255) of object 2

WHY THIS SCRIPT EXISTS (and why you can't just run 2_extract_crops_v2.py
three times, once per folder):

  * ALIGNMENT. The two masks are meant to become extra input channels next
    to the RGB (a 5-channel [R,G,B,mask_focal,mask_partner] tensor). For that
    to be valid, pixel (i,j) in the RGB must be the same physical point as
    (i,j) in each mask. This requires the *identical* crop window, letterbox
    offset and scale factor to be applied to all three images. Running the
    old script per-folder gives each image its own tight crop and its own
    scale -> the channels would not register. Here the transform is computed
    ONCE per frame (from the union of both rats) and applied to all three.

  * NO DISTORTION, HIGH RESOLUTION ON THE RATS. process_both() in the v2
    script keeps the whole 1920-wide frame, so at 512px the animals occupy a
    small fraction of the tensor. This script crops to the union bounding box
    of the two rats (plus a relative margin), then pads that crop to a SQUARE
    canvas (black letterbox on the short side) before an isotropic resize.
    Padding-to-square, never stretching, guarantees zero aspect-ratio
    distortion while giving the rats the full resolution budget.

  * FOCAL-AGNOSTIC ON DISK. The masks are stored as object_1 / object_2, not
    focal / partner. Your per-object labels decide the roles at load time:
    when the focal is object_1 feed [RGB, mask_1, mask_2]; when it is
    object_2 feed [RGB, mask_2, mask_1] (just swap the two mask channels).
    This halves storage and keeps the data reusable for either focal.

Relative SCALE between frames is deliberately not preserved (a frame with the
rats far apart is zoomed out, a frame with them close is zoomed in). That
information belongs in your geometry_motion features / 3D lift, not in the
pixels — and the per-frame scale factor is logged to the metadata CSV so it
stays auditable, exactly as in the v2 script.

The mask extraction (max-channel threshold + morphological close) and the
mask-first, colour-blind background re-zeroing are taken verbatim from
2_extract_crops_v2.py so behaviour matches the rest of the pipeline.

Layout auto-detection: a `cutouts` folder may either hold sub-folders
(all_objects/, object_1/, object_2/) or hold the files flat with
`_all_objects` / `_object_1` / `_object_2` suffixes. Both are handled.
Point --input at a single cutouts folder, at an experiment folder, or at a
parent containing many experiments — every `cutouts` folder found is
processed and mirrored under --output (default: ./training_data).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
OBJECT_TAGS = ("all_objects", "object_1", "object_2")


# ── Mask (verbatim from 2_extract_crops_v2.py) ────────────────────────────────

def extract_mask(img: np.ndarray, bg_threshold: int = 8) -> np.ndarray:
    """Binary mask: pixels with max channel value > bg_threshold are rat.
    Morphological closing fills small holes inside the rat body."""
    mask = (img.max(axis=2) > bg_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ── Aligned square transform ──────────────────────────────────────────────────

def compute_crop_rect(union_mask: np.ndarray, margin_frac: float,
                      margin_min: int):
    """Tight bbox of the non-zero region, expanded by a relative margin and
    clamped to the image. Returns (x1, y1, x2, y2) or None if empty."""
    ys, xs = np.where(union_mask > 0)
    if xs.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    w, h = x2 - x1, y2 - y1
    mx = max(margin_min, int(round(w * margin_frac)))
    my = max(margin_min, int(round(h * margin_frac)))
    H, W = union_mask.shape[:2]
    return (max(0, x1 - mx), max(0, y1 - my),
            min(W, x2 + mx), min(H, y2 + my))


def square_letterbox(src: np.ndarray, rect, target_size: int, interp: int):
    """Crop `rect` from `src`, pad the crop to a square canvas (centred, black
    on the short side), then isotropically resize to target_size.

    The square side and centring offsets depend ONLY on the crop dimensions,
    which are identical across the three images sharing one `rect` — so the
    RGB and both masks come out pixel-aligned. No axis is ever scaled
    independently, so there is no aspect-ratio distortion.
    """
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


def build_frame(all_rgb, obj1_rgb, obj2_rgb, bg_threshold, target_size,
                margin_frac, margin_min):
    """Produce (rgb_out, mask1_out, mask2_out, meta) for one frame, or
    (None, None, None, meta) if no rat is present.

    obj1_rgb / obj2_rgb may be None (object absent in this frame) -> that
    object's mask is all zeros and it does not contribute to the crop.
    """
    m1 = extract_mask(obj1_rgb, bg_threshold) if obj1_rgb is not None \
        else np.zeros(all_rgb.shape[:2], np.uint8)
    m2 = extract_mask(obj2_rgb, bg_threshold) if obj2_rgb is not None \
        else np.zeros(all_rgb.shape[:2], np.uint8)

    union = cv2.bitwise_or(m1, m2)
    rect = compute_crop_rect(union, margin_frac, margin_min)
    if rect is None:
        return None, None, None, dict(present_1=m1.any(), present_2=m2.any(),
                                      bbox=None, scale=None)

    rgb_out, scale = square_letterbox(all_rgb, rect, target_size, cv2.INTER_LINEAR)
    m1_out, _ = square_letterbox(m1, rect, target_size, cv2.INTER_NEAREST)
    m2_out, _ = square_letterbox(m2, rect, target_size, cv2.INTER_NEAREST)

    # Colour-blind halo removal: keep only RGB pixels attributed to a rat, so
    # every visible pixel in the tensor belongs to exactly one mask channel.
    keep = (m1_out > 0) | (m2_out > 0)
    rgb_out[~keep] = 0
    # Enforce strictly uniform {0,255} masks after nearest-neighbour resize.
    m1_out = np.where(m1_out > 127, 255, 0).astype(np.uint8)
    m2_out = np.where(m2_out > 127, 255, 0).astype(np.uint8)

    x1, y1, x2, y2 = rect
    meta = dict(present_1=bool(m1.any()), present_2=bool(m2.any()),
                bbox=(x1, y1, x2 - x1, y2 - y1), scale=round(scale, 4))
    return rgb_out, m1_out, m2_out, meta


# ── Frame discovery & pairing ─────────────────────────────────────────────────

def frame_key(path: Path) -> str:
    """`frame_000123_object_1.png` -> `frame_000123` (strips the object tag)."""
    stem = path.stem
    for tag in OBJECT_TAGS:
        if stem.endswith("_" + tag):
            return stem[: -(len(tag) + 1)]
    return stem


def index_tag_dir(cutouts: Path, tag: str):
    """Return {frame_key: path} for one object tag, handling both the
    sub-folder layout (cutouts/<tag>/*.png) and the flat layout
    (cutouts/*_<tag>.png)."""
    out = {}
    sub = cutouts / tag
    if sub.is_dir():
        for p in sub.iterdir():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                out[frame_key(p)] = p
    else:
        for p in cutouts.iterdir():
            if (p.suffix.lower() in SUPPORTED_EXTENSIONS
                    and p.stem.endswith("_" + tag)):
                out[frame_key(p)] = p
    return out


def looks_like_cutouts(d: Path) -> bool:
    """A cutouts folder is anchored on `all_objects` AND at least one single
    object. This deliberately rejects a lone `all_objects/` (or `object_1/`)
    sub-folder, so we never mistake the tag sub-folders for experiments."""
    # Sub-folder layout: an all_objects sub-dir plus an object_N sub-dir.
    if (d / "all_objects").is_dir() and (
            (d / "object_1").is_dir() or (d / "object_2").is_dir()):
        return True
    # Flat layout: files carrying an all_objects tag and an object_N tag.
    has_all = has_obj = False
    for p in d.iterdir():
        if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        if p.stem.endswith("_all_objects"):
            has_all = True
        elif p.stem.endswith("_object_1") or p.stem.endswith("_object_2"):
            has_obj = True
        if has_all and has_obj:
            return True
    return False


def find_cutouts_dirs(root: Path):
    if looks_like_cutouts(root):
        return [root]
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if looks_like_cutouts(d):
            found.append(d)
            dirnames[:] = []  # a cutouts folder has no cutouts folders inside it
    return sorted(set(found))


def output_subpath(cutouts: Path, root: Path) -> Path:
    """Map a discovered cutouts folder to <experiment>/<camera> under the
    output root.

      experiment = the name of the --input path itself (e.g. `grooming_1`)
      camera     = the cutouts folder's location beneath --input, with any
                   `cutouts` component stripped (e.g. `grooming_1/camera_3/
                   cutouts` -> camera `camera_3`; a camera folder that *is*
                   the cutouts folder keeps its own name).

    So `--input .../grooming_1` containing six camera folders produces
    training_data/grooming_1/<camera_1..6>/...
    """
    experiment = root.name
    try:
        rel = cutouts.relative_to(root)
    except ValueError:
        rel = Path(cutouts.name)
    cam_parts = [p for p in rel.parts if p != "cutouts"]
    if cam_parts:
        return Path(experiment, *cam_parts)
    return Path(experiment)


# ── Processing ────────────────────────────────────────────────────────────────

def process_cutouts(cutouts: Path, root: Path, out_root: Path, args):
    idx_all = index_tag_dir(cutouts, "all_objects")
    idx_1 = index_tag_dir(cutouts, "object_1")
    idx_2 = index_tag_dir(cutouts, "object_2")

    keys = sorted(set(idx_all) | set(idx_1) | set(idx_2))
    if not keys:
        print(f"  [skip] no frames found in {cutouts}")
        return dict(written=0, empty=0, error=0, no_all=0)

    sub = output_subpath(cutouts, root)
    exp = str(sub)
    exp_out = out_root / sub
    dir_rgb = exp_out / "rgb"
    dir_m1 = exp_out / "mask_object_1"
    dir_m2 = exp_out / "mask_object_2"
    dir_stack = exp_out / "stack"
    if not args.dry_run:
        for d in (dir_rgb, dir_m1, dir_m2):
            d.mkdir(parents=True, exist_ok=True)
        if args.save_stack:
            dir_stack.mkdir(parents=True, exist_ok=True)

    meta_writer, meta_file = None, None
    if args.metadata and not args.dry_run:
        meta_file = open(exp_out / "transform_meta.csv", "w", newline="")
        meta_writer = csv.writer(meta_file)
        meta_writer.writerow(["key", "present_1", "present_2",
                              "bbox_x", "bbox_y", "bbox_w", "bbox_h", "scale"])

    print(f"\n[{exp}]  {len(keys)} frame(s)  ->  {exp_out}")
    stats = dict(written=0, empty=0, error=0, no_all=0, downsampled=0)

    for i, key in enumerate(keys):
        p_all = idx_all.get(key)
        if p_all is None:
            stats["no_all"] += 1
            continue
        if args.dry_run:
            stats["written"] += 1
            continue

        all_rgb = cv2.imread(str(p_all))
        if all_rgb is None:
            print(f"  [ERROR] cannot read {p_all.name}")
            stats["error"] += 1
            continue
        obj1 = cv2.imread(str(idx_1[key])) if key in idx_1 else None
        obj2 = cv2.imread(str(idx_2[key])) if key in idx_2 else None

        try:
            rgb, m1, m2, meta = build_frame(
                all_rgb, obj1, obj2, args.bg_threshold, args.size,
                args.margin, args.margin_min)
        except Exception as e:
            print(f"  [ERROR] {key}: {e}")
            stats["error"] += 1
            continue

        if rgb is None:
            stats["empty"] += 1
            if meta_writer:
                meta_writer.writerow([key, meta["present_1"], meta["present_2"],
                                      None, None, None, None, None])
            continue

        cv2.imwrite(str(dir_rgb / f"{key}.png"), rgb)
        cv2.imwrite(str(dir_m1 / f"{key}.png"), m1)
        cv2.imwrite(str(dir_m2 / f"{key}.png"), m2)
        if args.save_stack:
            # [R,G,B,mask1,mask2] uint8; masks 0/255. cv2 loads BGR -> reorder.
            stack = np.dstack([rgb[..., ::-1], m1, m2]).astype(np.uint8)
            np.save(dir_stack / f"{key}.npy", stack)

        if meta_writer:
            bx = meta["bbox"] or (None, None, None, None)
            meta_writer.writerow([key, meta["present_1"], meta["present_2"],
                                  *bx, meta["scale"]])
        stats["written"] += 1
        if meta["scale"] is not None and meta["scale"] < 1.0:
            stats["downsampled"] += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(keys):
            print(f"  [{i+1:>6}/{len(keys)}]  written={stats['written']}  "
                  f"empty={stats['empty']}  no_all={stats['no_all']}  "
                  f"errors={stats['error']}")

    if meta_file:
        meta_file.close()
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=("Build aligned square RGB + object_1/object_2 masks for "
                     "transformer training from cutouts folders."),
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--input", required=True,
                    help="A cutouts folder, an experiment folder, or a parent "
                         "of many experiments. All cutouts folders found are "
                         "processed.")
    ap.add_argument("--output", default="training_data",
                    help="Output root. Default: ./training_data")
    ap.add_argument("--size", type=int, default=1024,
                    help="Square output side in px. Default 1024: the native "
                         "two-rat union crop is ~850px, so 1024 captures full "
                         "native detail (no downsampling) and keeps a rat "
                         "foot at ~2 ViT-16 patches. NOTE: this is the STORAGE "
                         "size; whether the transformer resolves feet also "
                         "depends on the model INPUT size you resize to at "
                         "train time. Going above ~1024 only upsamples beyond "
                         "native and adds no real detail.")
    ap.add_argument("--margin", type=float, default=0.08,
                    help="Margin around the two-rat union bbox, as a fraction "
                         "of its size, before letterboxing. Default 0.08.")
    ap.add_argument("--margin_min", type=int, default=8,
                    help="Minimum margin in px. Default 8.")
    ap.add_argument("--bg_threshold", type=int, default=8,
                    help="Max channel value treated as background "
                         "(matches 2_extract_crops_v2.py). Default 8.")
    ap.add_argument("--metadata", action="store_true",
                    help="Write transform_meta.csv per experiment "
                         "(bbox + per-frame scale factor). Recommended.")
    ap.add_argument("--save_stack", action="store_true",
                    help="Also write a 5-channel [R,G,B,m1,m2] .npy per frame.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Count frames without writing anything.")
    args = ap.parse_args()

    root = Path(args.input).resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        sys.exit(1)
    out_root = Path(args.output).resolve()

    cutouts_dirs = find_cutouts_dirs(root)
    if not cutouts_dirs:
        print(f"No cutouts folders found under {root}")
        sys.exit(0)

    print(f"Input        : {root}")
    print(f"Output       : {out_root}")
    print(f"Output size  : {args.size}x{args.size} (square, no distortion)")
    print(f"Union margin : {args.margin*100:.0f}% (min {args.margin_min}px)")
    print(f"BG threshold : {args.bg_threshold}")
    print(f"Experiments  : {len(cutouts_dirs)}")
    if args.dry_run:
        print("DRY RUN")

    total = dict(written=0, empty=0, error=0, no_all=0, downsampled=0)
    for c in cutouts_dirs:
        s = process_cutouts(c, root, out_root, args)
        for k in total:
            total[k] += s.get(k, 0)

    print(f"\n{'='*55}\nDone.")
    print(f"  written     : {total['written']}")
    print(f"  empty       : {total['empty']}  (no rat in frame)")
    print(f"  no_all      : {total['no_all']}  (missing all_objects image)")
    print(f"  errors      : {total['error']}")
    print(f"  downsampled : {total['downsampled']}  (scale<1: native detail "
          f"reduced at --size {args.size})")
    print(f"  output      : {out_root}")
    if total["downsampled"]:
        frac = 100 * total["downsampled"] / max(1, total["written"])
        print(f"\n  [NOTE] {frac:.1f}% of frames were downsampled below native "
              f"resolution.\n         These are wide-separation frames (large "
              f"union bbox); grooming\n         frames are typically not among "
              f"them. Raise --size only if\n         the downsampled frames "
              f"matter for your fine-detail classes.")


if __name__ == "__main__":
    main()