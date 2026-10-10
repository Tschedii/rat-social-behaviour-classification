#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from sam3_helpers import (
    load_prompts_json,
    wait_for_file,
    write_single_frame_video,
)
from label_preview_backends import ActiveSessionPreviewBackend


def run_absent_relabel(
    frame_rgb: np.ndarray,
    candidate_mask: Optional[np.ndarray],
    obj_id: int,
    predictor,
    session_id: str,
    frame_index_in_segment: int,
    image_shape: Tuple[int, int, int],
    review_cfg: dict,
    global_frame_idx: Optional[int] = None,
    sam3_device: str = "auto",
) -> Optional[np.ndarray]:
    """
    Manual relabel for an absent object on the exact frame passed in.

    The caller decides which frame should be relabeled. This function does not
    shift frame indices internally.
    """
    from label_first_frame import run_label_first_frame

    relabel_frame_index_in_segment = frame_index_in_segment
    relabel_global_frame_idx = global_frame_idx

    h, w = image_shape[:2]
    review_dir = review_cfg.get("review_dir", "data/manual_review")
    host = review_cfg.get("host", "127.0.0.1")
    port = review_cfg.get("port", 5000)
    open_browser = review_cfg.get("open_browser", False)

    os.makedirs(review_dir, exist_ok=True)

    frame_tag = (
        f"frame_{relabel_global_frame_idx:06d}_obj_{obj_id}"
        if relabel_global_frame_idx is not None
        else f"obj_{obj_id}"
    )

    relabel_dir = os.path.join(review_dir, f"{frame_tag}_relabel")
    os.makedirs(relabel_dir, exist_ok=True)

    tmp_video = os.path.join(relabel_dir, f"{frame_tag}.mp4")
    tmp_json = os.path.join(relabel_dir, f"{frame_tag}_points.json")
    tmp_frame = os.path.join(relabel_dir, f"{frame_tag}.png")
    tmp_overlay = os.path.join(relabel_dir, f"{frame_tag}_overlay.png")

    # Save the plain frame image
    Image.fromarray(frame_rgb).save(tmp_frame)

    # Save overlay image if a candidate mask exists
    display_frame = frame_rgb.copy()
    if candidate_mask is not None:
        m = np.asarray(candidate_mask).astype(bool)
        display_frame[m] = (
            0.5 * display_frame[m] + 0.5 * np.array([255, 255, 0], dtype=np.uint8)
        ).astype(np.uint8)
        Image.fromarray(display_frame).save(tmp_overlay)

    # Use the plain frame for the temporary single-frame video
    write_single_frame_video(frame_rgb, tmp_video, fps=1.0)

    print("\n=== ABSENT OBJECT RELABEL ===")
    print(f"Object:                    {obj_id}")
    if global_frame_idx is not None:
        print(f"Triggered on frame:        {global_frame_idx}")
        print(f"Relabel frame:            {relabel_global_frame_idx}")
    print(f"Segment frame used:        {relabel_frame_index_in_segment}")
    print(f"Relabel directory:         {relabel_dir}")
    print(f"Relabel video:             {tmp_video}")
    print(f"Relabel image:             {tmp_frame}")
    if candidate_mask is not None:
        print(f"Overlay image:             {tmp_overlay}")
    print(f"Output JSON:               {tmp_json}")
    print("Open the relabel UI in your browser, add points, and click Save.")
    print("Live preview reuses the active SAM3 session. No second model is created.")

    if os.path.isfile(tmp_json):
        os.remove(tmp_json)

    preview_backend = ActiveSessionPreviewBackend(
        predictor=predictor,
        session_id=session_id,
        frame_index_in_segment=relabel_frame_index_in_segment,
        image_width=w,
        image_height=h,
    )

    run_label_first_frame(
        video_path=tmp_video,
        output_json=tmp_json,
        output_frame_image=tmp_frame,
        object_ids=(obj_id,),
        host=host,
        port=port,
        open_browser=open_browser,
        enable_sam_preview=True,
        sam3_device=sam3_device,
        preview_backend=preview_backend,
    )

    if not wait_for_file(tmp_json, timeout_sec=3.0, poll_sec=0.1):
        raise RuntimeError(
            "\nRelabel UI closed, but no JSON was saved.\n"
            f"Expected JSON: {tmp_json}"
        )

    prompts = load_prompts_json(tmp_json)
    if obj_id not in prompts:
        raise RuntimeError(
            f"Relabel JSON exists but does not contain object {obj_id}: {tmp_json}"
        )

    points_abs, labels = prompts[obj_id]
    manual_mask = preview_backend.predict(points_abs, labels, obj_id)

    if manual_mask is None:
        raise RuntimeError(
            f"SAM3 returned no mask for object {obj_id} after manual relabel."
        )

    return manual_mask
