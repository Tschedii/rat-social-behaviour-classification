#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import nullcontext as _nullctx
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from decord import VideoReader, cpu

from sam3.model_builder import build_sam3_video_predictor

from sam3_helpers import (
    SegmentChainConfig,
    OutputPaths,
    SegmentContext,
    ObjectMemory,
    AsyncFrameWriter,
    FfmpegVideoSink,
    SegmentProfiler,
    abs_to_rel,
    maybe_convert_avi_to_mp4,
    normalize_config,
    resolve_inputs,
    infer_video_properties,
    load_segments_manifest,
    parse_candidate_masks_from_outputs,
    maybe_resolve_identity_swap,
    process_candidates_with_review,
    sample_points_from_mask,
    REVIEW_TALLY,
)

from absent_tracker import (
    AbsentHandler,
    AbsentHandlingConfig,
    ObjectTrackState,
)

from label_first_frame import run_label_first_frame

try:
    from model_mask_refiner import ModelMaskRefiner, compute_video_background_bgr
except Exception as _refiner_import_err:  # refiner is optional
    ModelMaskRefiner = None
    compute_video_background_bgr = None
    _REFINER_IMPORT_ERR = _refiner_import_err
else:
    _REFINER_IMPORT_ERR = None


def _reseed_masks_batch(
    predictor,
    session_id: str,
    frame_idx: int,
    masks_by_obj: Dict[int, np.ndarray],
) -> bool:
    """
    Lossless seam handoff: condition segment N+1 on segment N's CLEAN masks for
    the overlap frame, by feeding each object's mask directly to SAM3 via
    tracker.add_new_mask - instead of sampling points from those masks. This is
    the fix for the boundary merge where two touching rats collapse into one
    identity on the frame right after the seam: point re-seeding builds a thin,
    ambiguous conditioning that the first propagation step resolves by merging,
    whereas mask-conditioning gives SAM3 the exact separated two-mask state.

    All objects MUST be added before tracking starts (SAM3 only allows
    registering a new object id while tracking_has_started is False), so this
    adds them in one batch on the fresh session.

    Returns True if all masks were conditioned, False if the API path was
    unavailable (caller then falls back to whole-mask point prompting).
    """
    import torch

    # Resolve inference_state from the session id (same lookup the dispatcher does)
    try:
        session = predictor._get_session(session_id)
        inference_state = session["state"] if isinstance(session, dict) else session
    except Exception as e:
        print(f"  [reseed] could not resolve inference_state ({e}); falling back to points")
        return False

    # Resolve the tracker that owns add_new_mask
    tracker = None
    model = getattr(predictor, "model", None)
    if model is not None:
        tracker = getattr(model, "tracker", None)
    if tracker is None or not hasattr(tracker, "add_new_mask"):
        print("  [reseed] tracker.add_new_mask unavailable; falling back to points")
        return False

    # add_new_mask -> _obj_id_to_idx only allows registering a NEW object id
    # while `tracking_has_started` is False. So ALL objects must be conditioned
    # with their masks up front, before any propagation. Registering one object
    # via a point (which starts tracking) then adding another as a new mask is
    # what previously threw. Here we add every object's mask in one batch on the
    # fresh session, before tracking starts.
    try:
        for oid, mask_np in masks_by_obj.items():
            m = np.asarray(mask_np)
            if m.ndim >= 3:
                m = m[..., 0] if m.shape[-1] <= 4 else m[0]
            m = m > 0
            if int(m.sum()) == 0:
                continue
            mask_t = torch.as_tensor(m).to(torch.float32)
            tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=int(oid),
                mask=mask_t,
            )
        return True
    except Exception as e:
        print(f"  [reseed] batch add_new_mask failed ({e}); falling back to points")
        return False


@dataclass
class RelabelRequest:
    segment_index: int
    segment_path: str
    frame_idx_in_segment: int
    frame_idx_global: int
    obj_id: int
    reason: str
    relabel_output_json: str


def _video_out_dir(out_root: str, video_id: str) -> Path:
    return Path(out_root).expanduser().resolve() / video_id / "full"


def _restart_state_path(out_root: str, video_id: str) -> Path:
    return _video_out_dir(out_root, video_id) / "restart_state.json"


def save_restart_state(
    out_root: str,
    video_id: str,
    *,
    segment_index: int,
    segment_path: str,
    frame_idx_in_segment: int,
    frame_idx_global: int,
    status: str = "checkpoint",
) -> str:
    path = _restart_state_path(out_root, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "segment_index": int(segment_index),
        "segment_path": str(Path(segment_path).expanduser().resolve()),
        "frame_idx_in_segment": int(frame_idx_in_segment),
        "frame_idx_global": int(frame_idx_global),
        "status": status,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def load_restart_state(out_root: str, video_id: str) -> Dict[str, Any]:
    path = _restart_state_path(out_root, video_id)
    if not path.exists():
        raise FileNotFoundError(f"No restart checkpoint found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid restart checkpoint JSON: {path}")
    return data


def prepare_output_paths(
    cfg: SegmentChainConfig,
    w: int,
    h: int,
    fps: float,
) -> Tuple[OutputPaths, Any, bool]:
    """
    Returns (paths, video_sink, sink_writes_mp4_directly).

    Preferred sink: ffmpeg pipe encoding straight to the final MP4 (one
    encode, no intermediate multi-GB MJPG AVI). Falls back to the old
    cv2 MJPG AVI + post-hoc conversion when ffmpeg is unavailable.
    masks_viz output has been removed; the raw label PNGs now carry a color
    palette (rat 1 = red, rat 2 = blue) so they are directly viewable.
    """
    out_dir = Path(cfg.out_root).expanduser().resolve() / cfg.video_id / "full"
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_raw_dir = out_dir / "masks_raw"
    masks_raw_dir.mkdir(parents=True, exist_ok=True)

    overlay_avi_path = out_dir / "overlay.avi"
    overlay_mp4_path = out_dir / "overlay.mp4"

    paths = OutputPaths(
        out_dir=str(out_dir),
        masks_raw_dir=str(masks_raw_dir),
        overlay_avi_path=str(overlay_avi_path),
        overlay_mp4_path=str(overlay_mp4_path),
    )

    if FfmpegVideoSink.available():
        sink = FfmpegVideoSink(str(overlay_mp4_path), w, h, fps)
        if not sink.isOpened():
            raise RuntimeError(f"Could not start ffmpeg sink: {overlay_mp4_path}")
        print(f"Overlay video: encoding directly to MP4 via ffmpeg ({overlay_mp4_path})")
        return paths, sink, True

    print("Overlay video: ffmpeg not found, falling back to cv2 MJPG AVI + conversion")
    writer = cv2.VideoWriter(
        str(overlay_avi_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {overlay_avi_path}")

    return paths, writer, False


def build_context(
    cfg: SegmentChainConfig,
    obj_ids_sorted,
    paths: OutputPaths,
    writer: Any,
) -> SegmentContext:
    print("Building predictor...")
    predictor = build_sam3_video_predictor()
    print("Predictor ready.")
    return SegmentContext(
        predictor=predictor,
        writer=writer,
        obj_ids_sorted=obj_ids_sorted,
        paths=paths,
    )


def release_predictor_resources(context: Optional[SegmentContext]) -> None:
    try:
        if context is not None and hasattr(context, "predictor"):
            predictor = context.predictor
            context.predictor = None
            try:
                if predictor is not None and hasattr(predictor, "close"):
                    predictor.close()
            except Exception as e:
                print(f"Warning while closing predictor: {e}")
            try:
                del predictor
            except Exception:
                pass
    except Exception as e:
        print(f"Warning while releasing predictor: {e}")

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        print(f"Warning while clearing CUDA cache: {e}")


def finalize_outputs(paths: OutputPaths, sink_writes_mp4_directly: bool) -> Optional[str]:
    mp4_path = str(Path(paths.overlay_mp4_path).expanduser().resolve())

    print("\n=== FINALIZING OUTPUTS ===")
    print(f"Overlay MP4 path: {mp4_path}")

    if sink_writes_mp4_directly:
        mp4_file = Path(mp4_path)
        if not mp4_file.exists() or mp4_file.stat().st_size == 0:
            print(f"[finalize_outputs] MP4 missing or empty: {mp4_file}")
            return None
        return mp4_path

    avi_path = str(Path(paths.overlay_avi_path).expanduser().resolve())
    avi_file = Path(avi_path)
    if not avi_file.exists():
        print(f"[finalize_outputs] AVI file does not exist: {avi_file}")
        return None

    if avi_file.stat().st_size == 0:
        print(f"[finalize_outputs] AVI file is empty: {avi_file}")
        return None

    ok = maybe_convert_avi_to_mp4(avi_path, mp4_path)
    if not ok:
        print("[finalize_outputs] MP4 conversion failed; keeping AVI only.")
        return None

    return mp4_path


def find_segment_index_and_local_frame(
    seg_paths: list[str],
    first_label_frame: int,
) -> Tuple[int, int, int]:
    """
    Returns:
        start_segment_index,
        local_frame_index_within_that_segment,
        global_frame_offset_of_that_segment

    Uses segments_manifest.json (written at fragmentation time) when present,
    so we don't have to open a VideoReader per segment just to get len().
    """
    if first_label_frame < 0:
        raise ValueError("first_label_frame must be >= 0")

    manifest = None
    if seg_paths:
        manifest = load_segments_manifest(str(Path(seg_paths[0]).parent))

    frames_seen = 0
    for si, seg_path in enumerate(seg_paths):
        seg_name = Path(seg_path).name
        if manifest is not None and seg_name in manifest:
            n_frames = int(manifest[seg_name])
        else:
            vr = VideoReader(str(seg_path), ctx=cpu(0))
            n_frames = len(vr)

        if first_label_frame < frames_seen + n_frames:
            local_idx = first_label_frame - frames_seen
            return si, local_idx, frames_seen

        frames_seen += n_frames

    raise ValueError(
        f"first_label_frame={first_label_frame} is outside total segmented video length"
    )


def run_one_segment(
    context: SegmentContext,
    video_path: str,
    obj_prompts: Dict[int, Tuple[np.ndarray, np.ndarray]],
    global_frame_start: int,
    skip_prompt_frame_output: bool,
    track_states: Dict[int, ObjectTrackState],
    object_memories: Dict[int, ObjectMemory],
    absent_handler: AbsentHandler,
    review_cfg: dict,
    sam3_device: str = "auto",
    start_frame_in_segment: int = 0,
    reseed_masks: Optional[Dict[int, np.ndarray]] = None,
    *,
    video_id: str,
    out_root: str,
    segment_index: int,
    profiler: Optional[SegmentProfiler] = None,
) -> Tuple[Dict[int, np.ndarray], int]:
    predictor = context.predictor
    writer = context.writer
    obj_ids_sorted = context.obj_ids_sorted
    paths = context.paths

    video_path = str(Path(video_path).expanduser().resolve())

    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)

    if start_frame_in_segment < 0 or start_frame_in_segment >= num_frames:
        raise ValueError(
            f"start_frame_in_segment={start_frame_in_segment} is out of range "
            f"for segment with {num_frames} frames: {video_path}"
        )

    start_frame = vr[start_frame_in_segment].asnumpy()

    # Structural segment-boundary fix: at the start of every segment SAM3 is
    # re-prompted/re-seeded, so each object's prev_mask (from the previous
    # segment's last frame) is stale and NOT temporally continuous with this
    # segment's first frame. Comparing against it makes the tracker's
    # area-change / IoU continuity check fire spuriously at every transition
    # (and on every restart). Clear it so the first frame of this segment hits
    # the existing "no previous mask; accepted" path, whichever review call
    # handles it. This is the robust fix - independent of call site, restart
    # vs organic transition, and overlap handling.
    for _oid in obj_ids_sorted:
        if _oid in track_states:
            track_states[_oid].prev_mask = None

    session_id = None
    frames_written = 0
    last_visible_masks: Dict[int, np.ndarray] = {}

    pending_frame_rgb: Optional[np.ndarray] = None
    pending_masks: Optional[Dict[int, np.ndarray]] = None
    pending_frame_idx_global: Optional[int] = None
    pending_frame_idx_in_segment: Optional[int] = None
    _seen_first_new_frame = False

    try:
        resp = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": video_path,
            }
        )
        session_id = resp["session_id"]

        h, w = start_frame.shape[:2]
        start_visible: Dict[int, np.ndarray] = {}
        candidates: Dict[int, np.ndarray] = {}

        # Preferred seam handoff: condition N+1 directly on the clean carried
        # masks (segment N's masks for this overlap frame) via add_new_mask,
        # before any tracking starts. This gives SAM3 the exact separated
        # two-mask state, preventing the touching-rats merge on frame T+1 that
        # point re-seeding causes. Only on continuation segments (reseed_masks
        # present). Falls back to point prompts if unavailable.
        used_mask_reseed = False
        if reseed_masks:
            nonempty = {o: m for o, m in reseed_masks.items()
                        if m is not None and int(np.asarray(m).sum()) > 0}
            if nonempty:
                used_mask_reseed = _reseed_masks_batch(
                    predictor=predictor,
                    session_id=session_id,
                    frame_idx=start_frame_in_segment,
                    masks_by_obj=nonempty,
                )
                if used_mask_reseed:
                    # The conditioned masks ARE the frame-T masks; use them as
                    # this frame's candidates (no point prompting needed).
                    for _oid, _m in nonempty.items():
                        candidates[_oid] = np.asarray(_m).astype(bool)
                    print(
                        f"  [reseed] conditioned {len(nonempty)} object(s) via "
                        f"add_new_mask (lossless seam handoff)"
                    )

        # Point-prompt path: initial segment, or fallback if mask re-seed failed.
        if not used_mask_reseed:
            for obj_id, (points_abs, labels) in sorted(obj_prompts.items()):
                points_rel = abs_to_rel(points_abs, w, h)
                resp = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": start_frame_in_segment,
                        "points": points_rel.tolist(),
                        "point_labels": labels.astype(int).tolist(),
                        "obj_id": int(obj_id),
                        "rel_coordinates": True,
                    }
                )
                outputs = resp.get("outputs")
                if outputs is None:
                    continue
                # SAM3 returns masks for all registered objects; keep fullest set
                parsed = parse_candidate_masks_from_outputs(outputs, obj_ids_sorted)
                for _oid, _m in parsed.items():
                    candidates[_oid] = _m

        if candidates:
            if getattr(context, "refiner", None) is not None:
                candidates = context.refiner.refine_candidates(
                    frame_rgb=start_frame,
                    candidates=candidates,
                    obj_ids_sorted=obj_ids_sorted,
                )
            candidates = maybe_resolve_identity_swap(
                frame_rgb=start_frame,
                candidates=candidates,
                obj_ids_sorted=obj_ids_sorted,
                object_memories=object_memories,
            )

            # Seam-frame mask carry: on a continuation boundary, if SAM3's
            # re-seed produced no / a negligible mask for an object, fall back
            # to the mask carried across the seam (its last-good mask from the
            # previous segment's final frame) - the seam frame is the SAME frame
            # the previous segment already segmented well (overlap_frames=1).
            if skip_prompt_frame_output and reseed_masks:
                for _oid in obj_ids_sorted:
                    carried = reseed_masks.get(_oid)
                    if carried is None:
                        continue
                    cand = candidates.get(_oid)
                    cand_area = 0 if cand is None else int(np.asarray(cand).sum())
                    carried_area = int(np.asarray(carried).sum())
                    if carried_area > 0 and cand_area < 0.25 * carried_area:
                        candidates[_oid] = np.asarray(carried).astype(bool)
                        print(
                            f"  [seam-carry] obj {_oid}: SAM3 re-seed weak "
                            f"(area {cand_area} < 25% of {carried_area}); "
                            f"carried last-good mask onto seam frame"
                        )

            start_visible, _ = process_candidates_with_review(
                candidates=candidates,
                obj_ids_sorted=obj_ids_sorted,
                track_states=track_states,
                object_memories=object_memories,
                absent_handler=absent_handler,
                image_shape=start_frame.shape,
                frame_idx_global=global_frame_start,
                frame_rgb=start_frame,
                predictor=predictor,
                session_id=session_id,
                frame_index_in_segment=start_frame_in_segment,
                review_cfg=review_cfg,
                sam3_device=sam3_device,
                relabel_previous_frame=True,
                previous_frame_rgb=None,
                previous_frame_idx_global=None,
                previous_frame_index_in_segment=None,
                previous_visible_masks=None,
                segment_boundary=skip_prompt_frame_output,
            )
            if start_visible:
                last_visible_masks = start_visible

        if start_visible and not skip_prompt_frame_output:
            pending_frame_rgb = start_frame
            pending_masks = {
                int(k): np.asarray(v).astype(bool)
                for k, v in start_visible.items()
            }
            pending_frame_idx_global = global_frame_start
            pending_frame_idx_in_segment = start_frame_in_segment

        import time as _time
        _prof_on = profiler is not None and profiler.enabled
        _t_pred_start = _time.perf_counter() if _prof_on else 0.0

        for resp in predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            # Time from the previous yield to now is the model's work to
            # produce this response (the generator is lazy).
            if _prof_on:
                profiler.t_predictor += _time.perf_counter() - _t_pred_start

            frame_idx = resp.get("frame_index")
            outputs = resp.get("outputs")

            if frame_idx is None or outputs is None:
                if _prof_on:
                    _t_pred_start = _time.perf_counter()
                continue

            if frame_idx <= start_frame_in_segment:
                if _prof_on:
                    _t_pred_start = _time.perf_counter()
                continue

            _post_cm = profiler.post() if profiler is not None else _nullctx()
            with _post_cm:
                if profiler is not None:
                    with profiler.decode():
                        frame_rgb = vr[frame_idx].asnumpy()
                else:
                    frame_rgb = vr[frame_idx].asnumpy()

                global_idx = global_frame_start + (frame_idx - start_frame_in_segment)

                candidates = parse_candidate_masks_from_outputs(outputs, obj_ids_sorted)
                if getattr(context, "refiner", None) is not None:
                    candidates = context.refiner.refine_candidates(
                        frame_rgb=frame_rgb,
                        candidates=candidates,
                        obj_ids_sorted=obj_ids_sorted,
                    )
                candidates = maybe_resolve_identity_swap(
                    frame_rgb=frame_rgb,
                    candidates=candidates,
                    obj_ids_sorted=obj_ids_sorted,
                    object_memories=object_memories,
                )

                # The first genuinely-new frame of a CONTINUATION segment is a
                # re-seed seam: SAM3 was re-prompted, so its mask is not
                # continuous with the previous segment's last frame. Flag it so
                # the tracker skips continuity checks there (prevents a review
                # trigger at every segment transition).
                is_segment_boundary = (not _seen_first_new_frame) and (global_frame_start > 0)
                _seen_first_new_frame = True

                visible_masks, rewrite_previous = process_candidates_with_review(
                    candidates=candidates,
                    obj_ids_sorted=obj_ids_sorted,
                    track_states=track_states,
                    object_memories=object_memories,
                    absent_handler=absent_handler,
                    image_shape=frame_rgb.shape,
                    frame_idx_global=global_idx,
                    frame_rgb=frame_rgb,
                    predictor=predictor,
                    session_id=session_id,
                    frame_index_in_segment=frame_idx,
                    review_cfg=review_cfg,
                    sam3_device=sam3_device,
                    relabel_previous_frame=True,
                    previous_frame_rgb=pending_frame_rgb,
                    previous_frame_idx_global=pending_frame_idx_global,
                    previous_frame_index_in_segment=pending_frame_idx_in_segment,
                    previous_visible_masks=pending_masks,
                    segment_boundary=is_segment_boundary,
                )

                if (
                    rewrite_previous is not None
                    and pending_frame_idx_global == rewrite_previous.get("frame_idx_global")
                ):
                    pending_frame_rgb = rewrite_previous["frame_rgb"]
                    pending_masks = rewrite_previous["masks"]

                if (
                    pending_frame_rgb is not None
                    and pending_masks is not None
                    and pending_frame_idx_global is not None
                    and pending_frame_idx_in_segment is not None
                ):
                    # All disk work (label PNG, overlay, video encode, throttled
                    # restart checkpoint) happens on the writer thread; the GPU
                    # propagate loop never blocks on I/O.
                    writer.submit(
                        frame_rgb=pending_frame_rgb,
                        masks=pending_masks,
                        frame_idx_global=pending_frame_idx_global,
                        checkpoint=dict(
                            out_root=out_root,
                            video_id=video_id,
                            segment_index=segment_index,
                            segment_path=video_path,
                            frame_idx_in_segment=pending_frame_idx_in_segment,
                            frame_idx_global=pending_frame_idx_global,
                            status="written",
                        ),
                    )
                    frames_written += 1
                    if frames_written % 25 == 0:
                        print(f"  queued {frames_written} frames in current segment")

                pending_frame_rgb = frame_rgb
                # astype(bool) already returns a fresh array; no extra .copy() needed
                pending_masks = {
                    int(k): np.asarray(v).astype(bool)
                    for k, v in visible_masks.items()
                }
                pending_frame_idx_global = global_idx
                pending_frame_idx_in_segment = frame_idx
                last_visible_masks = visible_masks

            if _prof_on:
                profiler.tick_frame()
                _t_pred_start = _time.perf_counter()

        if (
            pending_frame_rgb is not None
            and pending_masks is not None
            and pending_frame_idx_global is not None
            and pending_frame_idx_in_segment is not None
        ):
            writer.submit(
                frame_rgb=pending_frame_rgb,
                masks=pending_masks,
                frame_idx_global=pending_frame_idx_global,
                checkpoint=dict(
                    out_root=out_root,
                    video_id=video_id,
                    segment_index=segment_index,
                    segment_path=video_path,
                    frame_idx_in_segment=pending_frame_idx_in_segment,
                    frame_idx_global=pending_frame_idx_global,
                    status="written",
                ),
            )
            frames_written += 1

        # Make sure everything for this segment is on disk (and the restart
        # checkpoint points at the true last written frame) before moving on.
        writer.flush()

        return last_visible_masks, frames_written

    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
            except Exception as e:
                print(f"Warning while closing session: {e}")


def run_sam3_segment_chain(
    video_id: str,
    prompts_json: str,
    first_label_frame: int,
    segments_root: str = "data/segments",
    out_root: str = "data/sam3_outputs",
    auto_pos: int = 8,
    auto_neg: int = 8,
    seed: int = 123,
    tracking_cfg: Optional[dict] = None,
    review_cfg: Optional[dict] = None,
    sam3_device: str = "auto",
    restart: bool = False,
    profile: bool = False,
    refiner_cfg: Optional[dict] = None,
):
    tracking_cfg = tracking_cfg or {}
    review_cfg = review_cfg or {}
    refiner_cfg = refiner_cfg or {}

    profiler = SegmentProfiler(enabled=profile)

    cfg = normalize_config(
        SegmentChainConfig(
            video_id=video_id,
            prompts_json=prompts_json,
            segments_root=segments_root,
            out_root=out_root,
            auto_pos=auto_pos,
            auto_neg=auto_neg,
            seed=seed,
        )
    )

    segments_dir, seg_paths, prompts_seg1, obj_ids_sorted = resolve_inputs(cfg)
    h, w, fps = infer_video_properties(seg_paths[0])
    paths, video_sink, sink_writes_mp4 = prepare_output_paths(cfg, w, h, fps)

    # Background writer: label PNGs, overlay encode and restart checkpoints
    # all happen off the GPU loop. Checkpoint every 25 written frames (plus
    # once per segment flush) instead of every single frame.
    writer = AsyncFrameWriter(
        masks_raw_dir=paths.masks_raw_dir,
        video_sink=video_sink,
        checkpoint_fn=lambda ckpt: save_restart_state(**ckpt),
        checkpoint_every=25,
    )

    review_cfg = dict(review_cfg)
    if "review_dir" not in review_cfg:
        review_cfg["review_dir"] = os.path.join(paths.out_dir, "manual_review")
    os.makedirs(review_cfg["review_dir"], exist_ok=True)

    if restart:
        state = load_restart_state(cfg.out_root, cfg.video_id)
        start_segment_index = int(state["segment_index"])
        start_frame_in_segment = int(state["frame_idx_in_segment"])
        first_label_frame = int(state["frame_idx_global"])

        print(
            f"RESTART: segment={start_segment_index}, "
            f"local_frame={start_frame_in_segment}, global_frame={first_label_frame}"
        )

        restart_segment_path = seg_paths[start_segment_index]

        print("\n=== RELABEL RESTART FRAME ===")
        run_label_first_frame(
            video_path=restart_segment_path,
            output_json=cfg.prompts_json,
            frame_index=start_frame_in_segment,
            output_frame_image=None,
            object_ids=tuple(int(x) for x in obj_ids_sorted),
            host=review_cfg.get("host", "127.0.0.1"),
            port=int(review_cfg.get("port", 5000)),
            open_browser=bool(review_cfg.get("open_browser", False)),
            enable_sam_preview=True,
            sam3_device=sam3_device,
        )

        _, _, prompts_seg1, _ = resolve_inputs(cfg)
    else:
        start_segment_index, start_frame_in_segment, _ = find_segment_index_and_local_frame(
            seg_paths, first_label_frame
        )

    print(
        f"Starting SAM3 from global frame {first_label_frame} "
        f"(segment {start_segment_index}, local frame {start_frame_in_segment})"
    )

    context = build_context(cfg, obj_ids_sorted, paths, writer)

    # Register nominal segment length so the review tally can classify each
    # trigger as segment-boundary vs mid-segment.
    try:
        _manifest = load_segments_manifest(segments_dir)
        if _manifest:
            _seg_len = int(next(iter(_manifest.values())))
            REVIEW_TALLY.set_segment_frames(_seg_len)
    except Exception:
        pass
    try:
        REVIEW_TALLY.set_append_log(
            os.path.join(paths.out_dir, "review_tally_live.jsonl")
        )
    except Exception:
        pass

    # Optional promptless-model mask refiner: fuses the trained segmenter's
    # pixels into SAM3's per-object masks (SAM3 keeps identity; the model
    # sharpens/recovers pixels). Enabled only if refiner_cfg has a checkpoint.
    context.refiner = None
    ckpt = refiner_cfg.get("ckpt")
    if ckpt:
        if ModelMaskRefiner is None:
            print(f"[refiner] requested but import failed: {_REFINER_IMPORT_ERR}. "
                  f"Continuing with pure SAM3.")
        else:
            input_video = refiner_cfg.get("input_video")
            bg_source = input_video or seg_paths[0]  # prefer full video for a clean bg
            print(f"[refiner] computing median background from: {bg_source}")
            background_bgr = compute_video_background_bgr(
                bg_source, max_samples=int(refiner_cfg.get("bg_max_samples", 150))
            )
            context.refiner = ModelMaskRefiner(
                ckpt_path=ckpt,
                background_bgr=background_bgr,
                device=refiner_cfg.get("device", sam3_device if sam3_device != "auto" else "cuda:0"),
                train_h=int(refiner_cfg.get("train_h", 640)),
                train_w=int(refiner_cfg.get("train_w", 1024)),
                thresh=float(refiner_cfg.get("thresh", 0.5)),
                dilate_radius=int(refiner_cfg.get("dilate_radius", 25)),
                mode=refiner_cfg.get("mode", "dilated_replace"),
                min_iou_vs_sam=float(refiner_cfg.get("min_iou_vs_sam", 0.5)),
            )

    absent_handler = AbsentHandler(
        AbsentHandlingConfig(
            min_area=tracking_cfg.get("min_mask_area", 100),
            max_area_ratio=tracking_cfg.get("max_area_ratio", 0.60),
            max_area_change_ratio=tracking_cfg.get("max_area_change_ratio", 0.10),
            min_iou_with_prev=tracking_cfg.get("min_iou_with_prev", 0.02),
            max_centroid_jump=tracking_cfg.get("max_centroid_jump", 120.0),
            max_bad_frames_before_absent=tracking_cfg.get(
                "max_bad_frames_before_absent", 1
            ),
            require_manual_relabel_on_reappearance=tracking_cfg.get(
                "require_manual_relabel_on_reappearance", True
            ),
            enable_reid=tracking_cfg.get("enable_reid", True),
            reid_min_area=tracking_cfg.get("reid_min_area", 100),
            reid_min_iou_with_last_good=tracking_cfg.get(
                "reid_min_iou_with_last_good", 0.05
            ),
            reid_max_centroid_jump=tracking_cfg.get(
                "reid_max_centroid_jump", 160.0
            ),
            verbose=tracking_cfg.get("verbose", True),
        )
    )

    track_states: Dict[int, ObjectTrackState] = {
        oid: ObjectTrackState(obj_id=oid) for oid in obj_ids_sorted
    }

    object_memories: Dict[int, ObjectMemory] = {
        oid: ObjectMemory(obj_id=oid) for oid in obj_ids_sorted
    }

    global_frame = first_label_frame
    last_visible_masks: Dict[int, np.ndarray] = {}

    try:
        for local_si, seg_path in enumerate(seg_paths[start_segment_index:]):
            si = start_segment_index + local_si
            seg_name = os.path.basename(seg_path)
            print(f"\n[{si + 1}/{len(seg_paths)}] Processing {seg_name}")

            if si == start_segment_index:
                prompts = prompts_seg1
                skip_frame0_output = False
                segment_start_frame = start_frame_in_segment
                reseed_masks = None
            else:
                prompts = {}
                reseed_masks = {}
                # Pass 1: collect every object's carry mask FIRST, so that when
                # we sample points for one object we can see all the others (the
                # cross-object avoidance needs the full set, not just objects
                # processed so far).
                for oid in obj_ids_sorted:
                    state = track_states[oid]
                    mem = object_memories[oid]

                    fused = mem.refresh_fused_mask()
                    prompt_mask = fused if fused is not None else state.last_good_mask

                    if prompt_mask is None:
                        print(
                            f"  warning: object {oid} has no prompt mask memory; "
                            f"skipping prompt for this segment"
                        )
                        continue

                    if state.is_absent:
                        print(
                            f"  object {oid} is currently absent; "
                            f"using fused memory / last_good_mask to try recovery in new segment"
                        )

                    # Fix 1: carry the actual mask across the seam.
                    reseed_masks[oid] = np.asarray(prompt_mask).astype(bool)

                # Pass 2: sample re-seed points, each object avoiding the others.
                for oid in obj_ids_sorted:
                    prompt_mask = reseed_masks.get(oid)
                    if prompt_mask is None:
                        continue

                    # Whole-mask spread + 16-point cap + cross-object avoidance:
                    # positive points avoid the other rat's (dilated) region and
                    # negatives land on the other rat, so crossing tails don't
                    # cause the two masks to merge at the re-seed.
                    _other = [
                        m2 for o2, m2 in reseed_masks.items()
                        if o2 != oid and m2 is not None
                    ]
                    pts_abs, labs = sample_points_from_mask(
                        prompt_mask,
                        num_pos=auto_pos,
                        num_neg=auto_neg,
                        seed=seed + 1000 * si + oid,
                        spread_whole_mask=True,
                        max_total_points=16,
                        other_masks=_other if _other else None,
                    )
                    prompts[oid] = (pts_abs, labs)

                skip_frame0_output = True
                segment_start_frame = 0

            if not prompts:
                print("  warning: no prompts available for this segment; outputs may be empty")

            last_visible_masks, written = run_one_segment(
                context=context,
                video_path=seg_path,
                obj_prompts=prompts,
                global_frame_start=global_frame,
                skip_prompt_frame_output=skip_frame0_output,
                track_states=track_states,
                object_memories=object_memories,
                absent_handler=absent_handler,
                review_cfg=review_cfg,
                sam3_device=sam3_device,
                start_frame_in_segment=segment_start_frame,
                reseed_masks=reseed_masks,
                video_id=cfg.video_id,
                out_root=cfg.out_root,
                segment_index=si,
                profiler=profiler,
            )

            global_frame += written
            print(f"  segment done, wrote {written} stitched frames, total now {global_frame}")
            profiler.segment_summary(seg_name)

            for oid in obj_ids_sorted:
                st = track_states[oid]
                mem = object_memories[oid]
                print(
                    f"    object {oid}: absent={st.is_absent}, "
                    f"absent_count={st.absent_count}, "
                    f"has_last_good_mask={st.last_good_mask is not None}, "
                    f"has_fused_mask={mem.last_fused_mask is not None}, "
                    f"waiting_for_manual_reinit={getattr(st, 'waiting_for_manual_reinit', False)}"
                )

        print("\nDone.")
        print(f"Overlay video: {paths.overlay_mp4_path if sink_writes_mp4 else paths.overlay_avi_path}")
        print(f"Raw masks:     {paths.masks_raw_dir} (color-palette PNGs: rat 1 = red, rat 2 = blue)")
        print(f"Manual review: {review_cfg['review_dir']}")
        print(f"Frames saved:  {global_frame - first_label_frame}")
        profiler.final_summary()

        # Review tally: total prompts + breakdown, so an A/B run (refiner on vs
        # off) is two comparable numbers. Also written as JSON next to outputs.
        print(REVIEW_TALLY.summary_str())
        try:
            tally_path = os.path.join(paths.out_dir, "review_tally.json")
            REVIEW_TALLY.write_json(tally_path)
            print(f"Review tally:  {tally_path}")
        except Exception as _e:
            print(f"(could not write review tally json: {_e})")

    finally:
        try:
            writer.close()
        except Exception as e:
            print(f"Warning while closing frame writer: {e}")

        release_predictor_resources(context)

    overlay_mp4_path = finalize_outputs(paths, sink_writes_mp4)

    return {
        "status": "done",
        "video_id": cfg.video_id,
        "segments_dir": str(segments_dir),
        "segment_paths": [str(p) for p in seg_paths],
        "prompts_json": cfg.prompts_json,
        "out_dir": paths.out_dir,
        "masks_raw_dir": paths.masks_raw_dir,
        "overlay_avi_path": paths.overlay_avi_path if not sink_writes_mp4 else None,
        "overlay_mp4_path": overlay_mp4_path,
        "manual_review_dir": review_cfg["review_dir"],
        "frames_saved": global_frame - first_label_frame,
        "num_segments": len(seg_paths),
        "tracking_cfg": tracking_cfg,
        "review_cfg": review_cfg,
        "sam3_device": sam3_device,
        "first_label_frame": first_label_frame,
        "start_segment_index": start_segment_index,
        "start_frame_in_segment": start_frame_in_segment,
        "restart": restart,
        "restart_state_path": str(_restart_state_path(cfg.out_root, cfg.video_id)),
    }


def find_first_segment_path(segments_dir: str) -> str:
    segments_dir = str(Path(segments_dir).expanduser().resolve())
    candidates = sorted(
        os.path.join(segments_dir, name)
        for name in os.listdir(segments_dir)
        if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not candidates:
        raise FileNotFoundError(f"No segment video found in: {segments_dir}")
    return candidates[0]


def run_label_first_segment_then_segment(
    server_cfg: dict,
    object_ids: Tuple[int, ...] = (1, 2),
    label_host: str = "127.0.0.1",
    label_port: int = 5000,
    label_open_browser: bool = False,
    tracking_cfg: Optional[dict] = None,
    review_cfg: Optional[dict] = None,
    sam3_device: str = "auto",
    auto_pos: int = 8,
    auto_neg: int = 8,
    seed: int = 123,
    first_label_frame: int = 0,
    restart: bool = False,
    profile: bool = False,
):
    tracking_cfg = tracking_cfg or {}
    review_cfg = dict(review_cfg or {})

    if "video_id" not in server_cfg:
        raise KeyError("server_cfg['video_id'] is required")

    server_paths = server_cfg.get("server", server_cfg)
    segments_dir = server_paths.get("segments_dir")
    prompts_json = server_paths.get("prompts_json")
    frame0_image = server_paths.get("frame0_image")
    outputs_root = server_paths.get("outputs_root")

    if not segments_dir or not prompts_json or not outputs_root:
        raise KeyError(
            "server_cfg must provide server.segments_dir, server.prompts_json, and server.outputs_root"
        )

    segments_dir = str(Path(segments_dir).expanduser().resolve())
    seg_paths = sorted(
        os.path.join(segments_dir, name)
        for name in os.listdir(segments_dir)
        if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not seg_paths:
        raise FileNotFoundError(f"No segment video found in: {segments_dir}")

    if restart:
        state = load_restart_state(outputs_root, server_cfg["video_id"])
        start_segment_index = int(state["segment_index"])
        local_frame_index = int(state["frame_idx_in_segment"])
        first_label_frame = int(state["frame_idx_global"])
        first_segment_path = seg_paths[start_segment_index]
    else:
        start_segment_index, local_frame_index, _ = find_segment_index_and_local_frame(
            seg_paths, first_label_frame
        )
        first_segment_path = seg_paths[start_segment_index]

    os.makedirs(os.path.dirname(prompts_json) or ".", exist_ok=True)
    if frame0_image:
        os.makedirs(os.path.dirname(frame0_image) or ".", exist_ok=True)

    print("\n=== STEP 1/2: LABEL FIRST SEGMENT FRAME ===")
    print(f"First segment: {first_segment_path}")
    print(f"Label frame:   global={first_label_frame}, local={local_frame_index}")
    print(f"Prompts JSON:  {prompts_json}")

    run_label_first_frame(
        video_path=first_segment_path,
        output_json=prompts_json,
        frame_index=local_frame_index,
        output_frame_image=frame0_image if not restart else None,
        object_ids=tuple(int(x) for x in object_ids),
        host=label_host,
        port=label_port,
        open_browser=label_open_browser,
        enable_sam_preview=True,
        sam3_device=sam3_device,
    )

    review_cfg.setdefault("host", label_host)
    review_cfg.setdefault("port", label_port)
    review_cfg.setdefault("open_browser", label_open_browser)

    print("\n=== STEP 2/2: START SEGMENTED SAM3 TRACKING ===")
    return run_sam3_segment_chain(
        video_id=server_cfg["video_id"],
        prompts_json=prompts_json,
        segments_root=segments_dir,
        out_root=outputs_root,
        auto_pos=auto_pos,
        auto_neg=auto_neg,
        seed=seed,
        tracking_cfg=tracking_cfg,
        review_cfg=review_cfg,
        sam3_device=sam3_device,
        first_label_frame=first_label_frame,
        restart=restart,
        profile=profile,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run SAM3 sequentially on video segments with manual review, memory, and anti-swap logic."
    )

    ap.add_argument("--video_id", required=True, help="e.g. video_001")
    ap.add_argument("--prompts_json", required=True, help="Prompt JSON from first-frame labeling")
    ap.add_argument(
        "--segments_root",
        default="data/segments",
        help="Folder containing the segment video files",
    )
    ap.add_argument("--out_root", default="data/sam3_outputs", help="Output root")

    ap.add_argument("--auto_pos", type=int, default=8, help="Positive points sampled from memory mask")
    ap.add_argument("--auto_neg", type=int, default=8, help="Negative points sampled from memory mask")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for point sampling")
    ap.add_argument(
        "--first_label_frame",
        type=int,
        default=0,
        help="Global frame index where SAM3 initialization/tracking should start",
    )

    ap.add_argument("--min_mask_area", type=int, default=100)
    ap.add_argument("--max_area_ratio", type=float, default=0.60)
    ap.add_argument("--max_area_change_ratio", type=float, default=0.10)
    ap.add_argument("--min_iou_with_prev", type=float, default=0.02)
    ap.add_argument("--max_centroid_jump", type=float, default=120.0)
    ap.add_argument("--max_bad_frames_before_absent", type=int, default=2)
    ap.add_argument("--disable_manual_relabel_on_reappearance", action="store_true")

    ap.add_argument("--enable_reid", action="store_true")
    ap.add_argument("--reid_min_area", type=int, default=100)
    ap.add_argument("--reid_min_iou_with_last_good", type=float, default=0.05)
    ap.add_argument("--reid_max_centroid_jump", type=float, default=160.0)

    ap.add_argument("--review_host", default="127.0.0.1")
    ap.add_argument("--review_port", type=int, default=5000)
    ap.add_argument("--review_open_browser", action="store_true")
    ap.add_argument("--review_dir", default=None)

    ap.add_argument(
        "--sam3_device",
        default="auto",
        help="SAM3 device for relabel preview and related UI flows",
    )

    ap.add_argument(
        "--restart",
        action="store_true",
        help="Reload last checkpoint, relabel that frame, and continue from there",
    )

    ap.add_argument("--quiet", action="store_true", help="Disable absent/re-id log messages")

    ap.add_argument(
        "--profile",
        action="store_true",
        help="Time the propagate loop (predictor vs decode vs post-processing) per segment and overall",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tracking_cfg = {
        "min_mask_area": args.min_mask_area,
        "max_area_ratio": args.max_area_ratio,
        "max_area_change_ratio": args.max_area_change_ratio,
        "min_iou_with_prev": args.min_iou_with_prev,
        "max_centroid_jump": args.max_centroid_jump,
        "max_bad_frames_before_absent": args.max_bad_frames_before_absent,
        "require_manual_relabel_on_reappearance": not args.disable_manual_relabel_on_reappearance,
        "enable_reid": args.enable_reid,
        "reid_min_area": args.reid_min_area,
        "reid_min_iou_with_last_good": args.reid_min_iou_with_last_good,
        "reid_max_centroid_jump": args.reid_max_centroid_jump,
        "verbose": not args.quiet,
    }

    review_cfg = {
        "host": args.review_host,
        "port": args.review_port,
        "open_browser": args.review_open_browser,
    }
    if args.review_dir:
        review_cfg["review_dir"] = args.review_dir

    result = run_sam3_segment_chain(
        video_id=args.video_id,
        prompts_json=args.prompts_json,
        segments_root=args.segments_root,
        out_root=args.out_root,
        auto_pos=args.auto_pos,
        auto_neg=args.auto_neg,
        seed=args.seed,
        tracking_cfg=tracking_cfg,
        review_cfg=review_cfg,
        sam3_device=args.sam3_device,
        first_label_frame=args.first_label_frame,
        restart=args.restart,
        profile=args.profile,
    )

    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()