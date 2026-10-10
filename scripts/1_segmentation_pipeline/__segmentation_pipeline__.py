#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fragment_video import run_fragment_video
from label_first_frame import run_label_first_frame
from sam3_segment import run_sam3_segment_chain
from first_label_ui import run_fragment_setup_ui


def _load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a JSON object")
    return cfg


def _require(d: Dict[str, Any], key: str, parent: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{parent}.{key}'")
    return d[key]


def _server_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _require(cfg, "server", "config")


def _fragment_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("fragment", {}) or {}


def _sam3_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("sam3", {}) or {}


def _label_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("label", {}) or {}


def _tracking_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("tracking", {}) or {}


def _refiner_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("refiner", {}) or {}


def _segment_seconds_from_config(cfg: Dict[str, Any]) -> float:
    fragment = _fragment_cfg(cfg)
    value = float(fragment.get("segment_seconds", 30))
    if value <= 0:
        raise ValueError("fragment.segment_seconds must be > 0")
    return value


def _video_run_dir(cfg: Dict[str, Any]) -> Path:
    server = _server_cfg(cfg)
    video_id = str(_require(cfg, "video_id", "config"))
    out_root = Path(_require(server, "outputs_root", "server")).expanduser().resolve()
    run_dir = out_root / video_id / "full"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _restart_state_path(cfg: Dict[str, Any]) -> Path:
    return _video_run_dir(cfg) / "restart_state.json"


def _load_restart_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = _restart_state_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No restart checkpoint found at: {path}\n"
            "You need to save restart_state.json during segmentation/review."
        )
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"Restart state must be a JSON object: {path}")
    return state


def _ensure_segments(
    cfg: Dict[str, Any],
    config_path: str,
    *,
    force_recreate: bool = False,
) -> List[str]:
    server = _server_cfg(cfg)
    fragment = _fragment_cfg(cfg)

    segments_dir = Path(_require(server, "segments_dir", "server")).expanduser().resolve()
    input_video = Path(_require(server, "input_video", "server")).expanduser().resolve()

    segments_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(str(p.resolve()) for p in segments_dir.glob("*.mp4"))
    if existing and not force_recreate:
        print(f"Using existing segments from: {segments_dir}")
        return existing

    if force_recreate and existing:
        print(f"Removing existing segments from: {segments_dir}")
        for p in segments_dir.glob("*.mp4"):
            p.unlink()

    segment_seconds = _segment_seconds_from_config(cfg)
    overlap_frames = int(fragment.get("overlap_frames", 1))

    # Separation-aware fragmentation: cut where the two animals are separated so
    # segment boundaries never land on a touching frame. Read from the
    # fragment.separation block (written by the first-label UI); falls back to
    # the refiner ckpt for the detector model.
    sep_cfg = fragment.get("separation", {}) or {}
    separation_aware = bool(sep_cfg.get("enabled", False))
    refiner_cfg = cfg.get("refiner", {}) or {}
    sep_ckpt = sep_cfg.get("ckpt") or refiner_cfg.get("ckpt")
    sep_device = sep_cfg.get("device") or refiner_cfg.get("device", "cuda:0")

    print("No segment files found or recreation requested. Creating segments now...")
    if separation_aware:
        print(
            f"Separation-aware fragmentation ON "
            f"(window {sep_cfg.get('min_segment_seconds', 15.0)}-"
            f"{sep_cfg.get('max_segment_seconds', 23.0)}s)"
        )
        if not sep_ckpt:
            raise ValueError(
                "fragment.separation.enabled is true but no ckpt is set "
                "(neither fragment.separation.ckpt nor refiner.ckpt)."
            )
    else:
        print(f"Using fixed segment length from config: {segment_seconds} sec")

    result = run_fragment_video(
        input_video=str(input_video),
        output_dir=str(segments_dir),
        segment_seconds=segment_seconds,
        overlap_frames=overlap_frames,
        separation_aware=separation_aware,
        min_segment_seconds=float(sep_cfg.get("min_segment_seconds", 15.0)),
        max_segment_seconds=float(sep_cfg.get("max_segment_seconds", 23.0)),
        sep_ckpt=sep_ckpt,
        sep_device=sep_device,
        sep_stride=int(sep_cfg.get("stride", 5)),
        sep_min_gap_frac=float(sep_cfg.get("min_gap_frac", 0.10)),
    )

    segment_paths = result.get("segment_paths", [])
    if not segment_paths:
        raise RuntimeError(f"No segments were created in: {segments_dir}")

    return [str(Path(p).expanduser().resolve()) for p in segment_paths]


def _label_segment_frame(
    cfg: Dict[str, Any],
    config_path: str,
    *,
    segment_index: int,
    frame_index: int,
) -> Dict[str, Any]:
    video_id = str(cfg.get("video_id", "video"))
    server = _server_cfg(cfg)
    label = _label_cfg(cfg)

    seg_paths = _ensure_segments(cfg, config_path)
    if segment_index < 0 or segment_index >= len(seg_paths):
        raise IndexError(
            f"segment_index {segment_index} out of range [0, {len(seg_paths)-1}]"
        )

    segment_path = seg_paths[segment_index]
    prompts_json = str(Path(_require(server, "prompts_json", "server")).expanduser().resolve())
    frame0_image = str(Path(_require(server, "frame0_image", "server")).expanduser().resolve())

    Path(prompts_json).parent.mkdir(parents=True, exist_ok=True)
    Path(frame0_image).parent.mkdir(parents=True, exist_ok=True)

    object_ids = tuple(label.get("object_ids", [1, 2]))
    if not object_ids:
        raise ValueError("label.object_ids must contain at least one object id")

    print("\n=== LABEL SEGMENT FRAME ===")
    print(f"Video ID:        {video_id}")
    print(f"Segment index:   {segment_index}")
    print(f"Segment video:   {segment_path}")
    print(f"Prompts JSON:    {prompts_json}")
    print(f"Frame image out: {frame0_image}")
    print(f"Frame index:     {frame_index}")
    print(f"Object IDs:      {object_ids}")
    print("Open the labeling UI, place the points, and click Save.")

    result = run_label_first_frame(
        video_path=segment_path,
        output_json=prompts_json,
        frame_index=frame_index,
        output_frame_image=frame0_image,
        object_ids=object_ids,
        host=str(label.get("host", "127.0.0.1")),
        port=int(label.get("port", 5000)),
        open_browser=bool(label.get("open_browser", False)),
        enable_sam_preview=bool(label.get("enable_sam_preview", True)),
        sam3_device=str(label.get("sam3_device", _sam3_cfg(cfg).get("device", "auto"))),
    )

    print("Label save completed.")
    return result


def _label_first_segment(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    fragment = _fragment_cfg(cfg)
    frame_index = int(fragment.get("first_label_frame", 0))
    return _label_segment_frame(
        cfg,
        config_path,
        segment_index=0,
        frame_index=frame_index,
    )


def _segment_from_config(
    cfg: Dict[str, Any],
    config_path: str,
    *,
    restart: bool = False,
) -> Dict[str, Any]:
    video_id = str(_require(cfg, "video_id", "config"))
    server = _server_cfg(cfg)
    sam3 = _sam3_cfg(cfg)
    label = _label_cfg(cfg)
    tracking = _tracking_cfg(cfg)
    fragment = _fragment_cfg(cfg)
    refiner = dict(_refiner_cfg(cfg))
    # inject the FULL input video so the refiner computes a clean per-video
    # median background (same as the training dataset builder)
    if refiner.get("ckpt"):
        refiner.setdefault(
            "input_video",
            str(Path(_require(server, "input_video", "server")).expanduser().resolve()),
        )

    seg_paths = _ensure_segments(cfg, config_path)
    segments_dir = str(Path(_require(server, "segments_dir", "server")).expanduser().resolve())
    prompts_json = str(Path(_require(server, "prompts_json", "server")).expanduser().resolve())
    out_root = str(Path(_require(server, "outputs_root", "server")).expanduser().resolve())
    first_label_frame = int(fragment.get("first_label_frame", 0))

    Path(out_root).mkdir(parents=True, exist_ok=True)

    print("\n=== RUN SEGMENTATION ===")
    print(f"Video ID:          {video_id}")
    print(f"Segments dir:      {segments_dir}")
    print(f"First segment:     {seg_paths[0]}")
    print(f"Num segments:      {len(seg_paths)}")
    print(f"Prompts JSON:      {prompts_json}")
    print(f"Outputs root:      {out_root}")
    print(f"First label frame: {first_label_frame}")
    print(f"Restart:           {restart}")

    return run_sam3_segment_chain(
        video_id=video_id,
        prompts_json=prompts_json,
        segments_root=segments_dir,
        out_root=out_root,
        auto_pos=int(sam3.get("auto_pos", 8)),
        auto_neg=int(sam3.get("auto_neg", 8)),
        seed=int(sam3.get("seed", 123)),
        tracking_cfg=tracking,
        review_cfg={
            "host": str(label.get("host", "127.0.0.1")),
            "port": int(label.get("port", 5000)),
            "open_browser": bool(label.get("open_browser", False)),
        },
        sam3_device=str(sam3.get("device", "auto")),
        first_label_frame=first_label_frame,
        restart=restart,
        profile=bool(sam3.get("profile", False)),
        refiner_cfg=refiner,
    )


def _run_fragment_with_ui(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    print("\n=== FRAGMENT SETUP UI ===")
    print("Opening UI for fragment setup...")

    selection_result = run_fragment_setup_ui(config_path=config_path)

    if not isinstance(selection_result, dict):
        raise RuntimeError("Fragment setup UI did not return a valid selection dict")

    print("UI saved successfully.")
    print(json.dumps(selection_result, indent=2))

    cfg = _load_config(config_path)
    segment_paths = _ensure_segments(cfg, config_path, force_recreate=True)

    return {
        "status": "fragmented",
        "config_path": str(Path(config_path).expanduser().resolve()),
        "selection": selection_result,
        "segment_paths": segment_paths,
        "segments_dir": str(Path(_server_cfg(cfg)["segments_dir"]).expanduser().resolve()),
    }


def run_pipeline_label_then_segment(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    cfg = _load_config(config_path)

    label_result = _label_first_segment(cfg, config_path)

    print("\nSave detected. Starting segmentation automatically...")

    cfg = _load_config(config_path)
    segment_result = _segment_from_config(cfg, config_path, restart=False)

    return {
        "status": "done",
        "label": label_result,
        "segment": segment_result,
    }


def run_pipeline_restart(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    state = _load_restart_state(cfg)

    segment_index = int(_require(state, "segment_index", "restart_state"))
    frame_index = int(_require(state, "frame_idx_in_segment", "restart_state"))

    print("\n=== RESTART MODE ===")
    print(f"Recovered checkpoint: segment_index={segment_index}, frame_index={frame_index}")

    label_result = _label_segment_frame(
        cfg,
        config_path,
        segment_index=segment_index,
        frame_index=frame_index,
    )

    print("\nSave detected. Restarting segmentation from saved checkpoint...")

    cfg = _load_config(config_path)
    segment_result = _segment_from_config(
        cfg,
        config_path,
        restart=True,
    )

    return {
        "status": "restarted",
        "restart_state": state,
        "label": label_result,
        "segment": segment_result,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Segmentation pipeline driven by a JSON config.")
    ap.add_argument("--config", "-config", required=True, help="Path to JSON config file")
    ap.add_argument(
        "--restart",
        "-restart",
        action="store_true",
        help=(
            "Resume from the last saved labeled/reviewed frame. "
            "This reloads the last checkpoint, relabels that frame, "
            "then continues segmentation from there."
        ),
    )
    ap.add_argument(
        "command",
        nargs="?",
        default="label_then_segment",
        choices=["fragment", "label", "segment", "label_then_segment"],
        help=(
            "fragment: open setup UI, update existing config, then create segments; "
            "label: label first frame of first segment and then run segmentation; "
            "segment: run segmentation only; "
            "label_then_segment: explicit alias for label"
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    if args.restart:
        result = run_pipeline_restart(cfg, args.config)
    elif args.command == "fragment":
        result = _run_fragment_with_ui(cfg, args.config)
    elif args.command in {"label", "label_then_segment"}:
        cfg = _load_config(args.config)
        result = run_pipeline_label_then_segment(cfg, args.config)
    elif args.command == "segment":
        cfg = _load_config(args.config)
        result = _segment_from_config(cfg, args.config, restart=False)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()