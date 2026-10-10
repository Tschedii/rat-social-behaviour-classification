"""
cutout_objects.py — Extract per-object cutouts from a video using label-map masks.

QUICK START (no config file needed):
    python cutout_objects.py --video myvideo.mp4 --masks ./masks_dir --output ./cutouts

That will auto-discover all object IDs from the mask files and produce:
    cutouts/
        object_1/frame_000001.png  ...
        object_2/frame_000001.png  ...
        ...
        all_objects/frame_000001.png  ...   (all objects combined)

FULL OPTIONS:
    python cutout_objects.py \\
        --video   myvideo.mp4 \\
        --masks   ./masks_dir \\
        --output  ./cutouts \\
        --objects 1 2          # limit to specific IDs (default: all discovered)
        --format  png          # png (default) or jpg
        --transparent          # PNG alpha channel instead of black background
        --overwrite            # re-write files that already exist
        --config  cfg.json     # optional: read video/masks/output paths from JSON

JSON CONFIG FORMAT (optional):
    {
        "server": {
            "input_video": "/path/to/video.mp4",
            "segments_dir": "/path/to/masks",
            "outputs_root": "/path/to/output"
        },
        "label": { "object_ids": [1, 2] },
        "video_id": "myexp"
    }
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = IMAGE_EXTENSIONS | {".npy", ".npz"}

FRAME_REGEXES = [
    re.compile(r"(?:frame|frm|f)[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|[_-])(\d{3,})(?:$|[_-])"),
]

# Only match object IDs when there is an EXPLICIT keyword — never a bare number,
# because bare numbers in filenames like frame_000001 are almost always frame indices.
OBJECT_KEYWORD_REGEX = re.compile(
    r"(?:object|obj|id|rat|animal|track)[_-]?(\d+)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    input_video: Path
    segments_dir: Path
    outputs_root: Path
    object_ids: Tuple[int, ...]        # empty = auto-discover
    video_id: str = "video"


def parse_config_file(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    server = raw.get("server", {})
    label = raw.get("label", {})
    object_ids = tuple(int(x) for x in label.get("object_ids", []))
    return AppConfig(
        input_video=Path(server["input_video"]).expanduser(),
        segments_dir=Path(server["segments_dir"]).expanduser(),
        outputs_root=Path(server["outputs_root"]).expanduser(),
        object_ids=object_ids,
        video_id=str(raw.get("video_id", "video")),
    )


# ---------------------------------------------------------------------------
# Mask index
# ---------------------------------------------------------------------------

class SegmentIndex:
    def __init__(self, segments_dir: Path, object_ids: Sequence[int]) -> None:
        self.segments_dir = segments_dir
        self.object_ids = tuple(object_ids)
        self._index: Dict[Tuple[int, int], Path] = {}
        self._shared_index: Dict[int, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.segments_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.segments_dir}")

        for path in self.segments_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MASK_EXTENSIONS:
                continue

            frame_idx = self._extract_frame_idx(path)
            object_id = self._extract_object_id(path)
            if frame_idx is None:
                continue

            if object_id is None:
                # Shared label map (pixel value = object ID)
                existing = self._shared_index.get(frame_idx)
                if existing is None or len(path.parts) < len(existing.parts):
                    self._shared_index[frame_idx] = path
                continue

            if self.object_ids and object_id not in self.object_ids:
                continue

            key = (frame_idx, object_id)
            existing = self._index.get(key)
            if existing is None or len(path.parts) < len(existing.parts):
                self._index[key] = path

    def _extract_frame_idx(self, path: Path) -> Optional[int]:
        for text in [path.stem, *path.parts]:
            for regex in FRAME_REGEXES:
                m = regex.search(text)
                if m:
                    return int(m.group(1))
        return None

    def _extract_object_id(self, path: Path) -> Optional[int]:
        """
        Return an object ID only when an explicit keyword (rat, obj, object, id,
        animal, track) is present in the filename stem or a parent directory name.

        We never fall back to matching a bare number, because filenames like
        frame_000001.png contain a frame index that looks identical to object ID 1.
        """
        candidates = [path.stem, *reversed(path.parts[:-1])]
        for text in candidates:
            m = OBJECT_KEYWORD_REGEX.search(text)
            if m:
                value = int(m.group(1))
                if not self.object_ids or value in self.object_ids:
                    return value
        return None

    def get(self, frame_idx: int, object_id: int) -> Optional[Path]:
        return self._index.get((frame_idx, object_id))

    def get_shared(self, frame_idx: int) -> Optional[Path]:
        return self._shared_index.get(frame_idx)

    def known_frames(self) -> List[int]:
        frames = {f for f, _ in self._index}
        frames.update(self._shared_index)
        return sorted(frames)

    def all_object_ids(self) -> List[int]:
        return sorted({obj for _, obj in self._index})

    def discover_object_ids_from_shared(self) -> List[int]:
        """Scan a sample of shared label maps to find all non-zero pixel values."""
        discovered: set[int] = set()
        paths = list(self._shared_index.values())[:20]  # sample up to 20 frames
        for p in paths:
            try:
                arr = load_label_map(p)
                for v in np.unique(arr):
                    if v != 0:
                        discovered.add(int(v))
            except Exception:
                pass
        return sorted(discovered)

    def stats(self) -> Dict[int, int]:
        counts: Dict[int, int] = {obj_id: 0 for obj_id in self.object_ids}
        for _, object_id in self._index:
            counts[object_id] = counts.get(object_id, 0) + 1
        if self._shared_index:
            counts[-1] = len(self._shared_index)
        return counts


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_label_map(mask_path: Path) -> np.ndarray:
    """Load mask as 2D uint8, preserving raw label values (0, 1, 2, ...)."""
    suffix = mask_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    elif suffix == ".npy":
        mask = np.load(mask_path)
    elif suffix == ".npz":
        data = np.load(mask_path)
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"NPZ has no arrays: {mask_path}")
        mask = data[keys[0]]
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}")

    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2:
        raise ValueError(f"Mask not 2D after squeeze: {mask_path} shape={mask.shape}")
    return mask.astype(np.uint8)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a per-object mask file (binary: 0 or 255)."""
    mask = load_label_map(mask_path)
    return np.where(mask > 0, np.uint8(255), np.uint8(0))


def extract_object_mask(label_map: np.ndarray, object_id: int) -> np.ndarray:
    """Extract a binary mask for one object ID from a label map."""
    return np.where(label_map == object_id, np.uint8(255), np.uint8(0))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def ensure_same_size(mask: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_mask(frame_bgr: np.ndarray, mask: np.ndarray, transparent: bool = False) -> np.ndarray:
    """Apply binary mask (0/255). Outside-mask pixels become black (or transparent)."""
    if transparent:
        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        bgra[mask == 0, :3] = 0
        return bgra
    result = np.zeros_like(frame_bgr)
    result[mask > 0] = frame_bgr[mask > 0]
    return result


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write: {path}")


def combine_masks(masks: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    masks = list(masks)
    if not masks:
        return None
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        combined = np.maximum(combined, m)
    return combined


def derive_cutouts_root(segments_dir: Path, explicit_output: Optional[Path]) -> Path:
    """
    Place the cutouts folder as a sibling of masks_raw (i.e. next to it).

    Example:
        segments_dir = .../video_001/full/masks_raw
        → cutouts_root = .../video_001/full/cutouts

    An explicit --output argument always wins.
    """
    if explicit_output is not None:
        return explicit_output.expanduser().resolve()
    return segments_dir.resolve().parent / "cutouts"


def build_output_dirs(outputs_root: Path, object_ids: Sequence[int]) -> Dict[str, Path]:
    dirs: Dict[str, Path] = {}
    for oid in object_ids:
        dirs[f"object_{oid}"] = outputs_root / f"object_{oid}"
    dirs["all_objects"] = outputs_root / "all_objects"
    return dirs


# ---------------------------------------------------------------------------
# Frame reader — sequential only, no cap.set() drift
# ---------------------------------------------------------------------------

def iter_target_frames(cap: cv2.VideoCapture, target_frames: List[int]):
    """Yield (frame_idx, frame_bgr) for each index in target_frames (sorted asc)."""
    current = 0
    for target in target_frames:
        while current < target:
            ok, _ = cap.read()
            if not ok:
                return
            current += 1
        ok, frame = cap.read()
        if not ok:
            return
        current += 1
        yield target, frame


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_video(
    config: AppConfig,
    output_format: str = "png",
    transparent: bool = False,
    overwrite: bool = False,
) -> None:
    if output_format not in {"png", "jpg"}:
        raise ValueError("output_format must be 'png' or 'jpg'")
    if transparent and output_format != "png":
        raise ValueError("Transparent output requires PNG format")

    # Build index (no object filter yet — need to discover IDs from shared maps)
    index = SegmentIndex(config.segments_dir, object_ids=())

    # Resolve object IDs
    object_ids = list(config.object_ids)
    if not object_ids:
        object_ids = index.discover_object_ids_from_shared()
        if not object_ids:
            object_ids = index.all_object_ids()
        if not object_ids:
            raise RuntimeError(
                "Could not auto-discover any object IDs from the mask files.\n"
                "Use --objects 1 2 3 to specify them explicitly."
            )
        print(f"Auto-discovered object IDs: {object_ids}")

    known = index.known_frames()
    if not known:
        raise RuntimeError("No masks indexed. Check --masks path and file naming.")

    print(f"Masks found for {len(known)} frames  (first={known[0]}, last={known[-1]})")
    print(f"Object IDs to extract: {object_ids}")

    cap = cv2.VideoCapture(str(config.input_video))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {config.input_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.2f} fps")

    output_dirs = build_output_dirs(config.outputs_root, object_ids)
    ext = ".png" if output_format == "png" else ".jpg"

    processed = 0
    written = 0
    missing_by_object: Dict[int, int] = {oid: 0 for oid in object_ids}

    try:
        for frame_idx, frame in iter_target_frames(cap, known):
            processed += 1
            frame_masks: Dict[int, np.ndarray] = {}

            shared_label_map: Optional[np.ndarray] = None
            shared_path = index.get_shared(frame_idx)
            if shared_path is not None:
                raw = load_label_map(shared_path)
                shared_label_map = ensure_same_size(raw, frame.shape)

            for oid in object_ids:
                # 1. Dedicated per-object file
                mask_path = index.get(frame_idx, oid)
                if mask_path is not None:
                    mask = load_binary_mask(mask_path)
                    frame_masks[oid] = ensure_same_size(mask, frame.shape)
                    continue

                # 2. Extract from shared label map
                if shared_label_map is not None:
                    mask = extract_object_mask(shared_label_map, oid)
                    if np.any(mask):
                        frame_masks[oid] = mask
                        continue

                missing_by_object[oid] += 1

            if not frame_masks:
                continue

            frame_name = f"frame_{frame_idx:06d}{ext}"

            # Individual object cutouts (respects --objects filter)
            for oid, mask in frame_masks.items():
                out_path = output_dirs[f"object_{oid}"] / frame_name
                if overwrite or not out_path.exists():
                    save_image(out_path, apply_mask(frame, mask, transparent))
                    written += 1

            # Combined: always use ALL non-background IDs from the shared label map,
            # regardless of the --objects filter, so all_objects is never just one rat.
            if shared_label_map is not None:
                all_masks = [
                    extract_object_mask(shared_label_map, int(v))
                    for v in np.unique(shared_label_map) if v != 0
                ]
                combined = combine_masks(all_masks)
            else:
                # No shared map — fall back to whatever per-object masks we have
                combined = combine_masks(frame_masks.values())

            if combined is not None:
                out_path = output_dirs["all_objects"] / frame_name
                if overwrite or not out_path.exists():
                    save_image(out_path, apply_mask(frame, combined, transparent))
                    written += 1

    finally:
        cap.release()

    print("\n--- Done ---")
    print(f"Video            : {config.input_video}")
    print(f"Masks dir        : {config.segments_dir}")
    print(f"Output root      : {config.outputs_root}")
    print(f"Frames processed : {processed}")
    print(f"Files written    : {written}")
    print(f"Missing per obj  : {missing_by_object}")
    print("Output dirs:")
    for key, path in output_dirs.items():
        print(f"  {key:20s}: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional config (original interface: script.py config.json)
    p.add_argument("config_positional", nargs="?", type=Path, metavar="CONFIG",
                   help="JSON config file (positional, for backwards compatibility)")

    # Primary inputs (can be set via --config or directly)
    p.add_argument("--video",   "-v", type=Path, help="Path to input video file")
    p.add_argument("--masks",   "-m", type=Path, help="Directory containing label-map mask files")
    p.add_argument("--output",  "-o", type=Path, help="Output root directory")

    # Optional config file (overridden by CLI flags)
    p.add_argument("--config",  "-c", type=Path, help="JSON config file (flag form)")

    # Object selection
    p.add_argument(
        "--objects", nargs="+", metavar="ID",
        help=(
            "Object IDs to extract. Use --objects all to auto-discover every object "
            "and generate individual folders + all_objects in one run. "
            "Or pass specific IDs: --objects 1 2"
        ),
    )

    # Output options
    p.add_argument("--format",      choices=["png", "jpg"], default="png", dest="output_format")
    p.add_argument("--transparent", "-t", action="store_true",
                   help="PNG with alpha channel (transparent background)")
    p.add_argument("--overwrite",   "-O", action="store_true",
                   help="Overwrite existing output files")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # --- Build config ---
    config_path = args.config or args.config_positional
    if config_path:
        config = parse_config_file(config_path)
    else:
        if not args.video or not args.masks:
            parser.error(
                "Provide either --config path/to/cfg.json\n"
                "   or both --video path/to/video.mp4 --masks path/to/masks_dir"
            )
        masks_dir = args.masks.expanduser().resolve()
        config = AppConfig(
            input_video=args.video.expanduser().resolve(),
            segments_dir=masks_dir,
            outputs_root=derive_cutouts_root(masks_dir, args.output),
            object_ids=(),
        )

    # CLI overrides
    if args.video:
        config = replace(config, input_video=args.video.expanduser().resolve())
    if args.masks:
        config = replace(config, segments_dir=args.masks.expanduser().resolve())
    # Always re-derive outputs_root from segments_dir so it lands in video_XXX_cutouts,
    # unless the user explicitly passed --output.
    config = replace(
        config,
        outputs_root=derive_cutouts_root(config.segments_dir, args.output),
    )
    if args.objects:
        if args.objects == ["all"]:
            # Empty tuple triggers auto-discovery of all IDs in process_video
            config = replace(config, object_ids=())
        else:
            config = replace(config, object_ids=tuple(int(x) for x in args.objects))

    process_video(
        config=config,
        output_format=args.output_format,
        transparent=args.transparent,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())