#!/usr/bin/env python3
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from sam3_helpers import abs_to_rel, parse_candidate_masks_from_outputs


class PreviewBackend(Protocol):
    def predict(
        self,
        points_abs,
        labels,
        obj_id: int,
    ) -> Optional[np.ndarray]:
        ...

    def run_request(self, payload: dict) -> dict:
        ...


def _as_numpy_bool_mask(mask_like) -> Optional[np.ndarray]:
    if mask_like is None:
        return None

    mask = mask_like
    try:
        if hasattr(mask, "detach"):
            mask = mask.detach()
        if hasattr(mask, "cpu"):
            mask = mask.cpu()
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
    except Exception:
        pass

    mask = np.asarray(mask)
    if mask.ndim >= 3:
        mask = mask[0]
    return mask > 0.0


def _run_handle_request(target: object, payload: dict) -> dict:
    if not hasattr(target, "handle_request"):
        raise RuntimeError(
            f"Preview target of type {type(target).__name__} does not expose handle_request()."
        )

    if torch is None:
        return target.handle_request(payload)

    device_type = None
    if torch.cuda.is_available():
        device_type = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device_type = "mps"

    def _call_with_ctx(autocast_ctx):
        with torch.inference_mode():
            with autocast_ctx:
                return target.handle_request(payload)

    if device_type != "cuda":
        return _call_with_ctx(nullcontext())

    retry_contexts = [
        nullcontext(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        torch.autocast(device_type="cuda", dtype=torch.float16),
        torch.autocast(device_type="cuda", enabled=False),
    ]

    last_exc = None
    for ctx in retry_contexts:
        try:
            return _call_with_ctx(ctx)
        except RuntimeError as exc:
            msg = str(exc)
            dtype_mismatch = (
                "Input type (c10::BFloat16) and bias type (float) should be the same" in msg
                or "Input type (c10::Half) and bias type (float) should be the same" in msg
                or ("bias type" in msg and "should be the same" in msg)
            )
            if not dtype_mismatch:
                raise
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Preview request failed before a handle_request call could complete.")


@dataclass
class ActiveSessionPreviewBackend:
    predictor: object
    session_id: str
    frame_index_in_segment: int
    image_width: int
    image_height: int

    def run_request(self, payload: dict) -> dict:
        return _run_handle_request(self.predictor, payload)

    def predict(
        self,
        points_abs,
        labels,
        obj_id: int,
    ) -> Optional[np.ndarray]:
        if points_abs is None or len(points_abs) == 0:
            return None

        points_abs = np.asarray(points_abs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        points_rel = abs_to_rel(points_abs, self.image_width, self.image_height)

        payload = {
            "type": "add_prompt",
            "session_id": self.session_id,
            "frame_index": self.frame_index_in_segment,
            "points": points_rel.tolist(),
            "point_labels": labels.astype(int).tolist(),
            "obj_id": int(obj_id),
            "rel_coordinates": True,
        }

        resp = self.run_request(payload)
        outputs = resp.get("outputs")
        if outputs is None:
            return None

        candidates = parse_candidate_masks_from_outputs(outputs, [obj_id])
        return candidates.get(obj_id)


@dataclass
class LocalSessionPreviewBackend:
    request_target: object
    session_id: str
    frame_index: int
    image_width: int
    image_height: int

    def run_request(self, payload: dict) -> dict:
        return _run_handle_request(self.request_target, payload)

    def predict(
        self,
        points_abs,
        labels,
        obj_id: int,
    ) -> Optional[np.ndarray]:
        if points_abs is None or len(points_abs) == 0:
            return None

        points_abs = np.asarray(points_abs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        points_rel = abs_to_rel(points_abs, self.image_width, self.image_height)

        payload = {
            "type": "add_prompt",
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "points": points_rel.tolist(),
            "point_labels": labels.astype(int).tolist(),
            "obj_id": int(obj_id),
            "rel_coordinates": True,
        }

        resp = self.run_request(payload)
        outputs = resp.get("outputs")
        if outputs is None:
            return None

        candidates = parse_candidate_masks_from_outputs(outputs, [obj_id])
        return candidates.get(obj_id)


@dataclass
class DirectPredictorPreviewBackend:
    predictor: object
    inference_state: object
    image_width: int
    image_height: int
    frame_index: int = 0
    torch_mod: object = None
    device_obj: object = None

    def run_request(self, payload: dict) -> dict:
        raise RuntimeError(
            "DirectPredictorPreviewBackend does not expose handle_request(); use predict() instead."
        )

    def predict(
        self,
        points_abs,
        labels,
        obj_id: int,
    ) -> Optional[np.ndarray]:
        if points_abs is None or len(points_abs) == 0:
            return None
        if self.predictor is None or self.inference_state is None:
            return None

        points_abs = np.asarray(points_abs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        points_rel = abs_to_rel(points_abs, self.image_width, self.image_height)

        if self.torch_mod is not None and self.device_obj is not None:
            points_tensor = self.torch_mod.tensor(
                points_rel, dtype=self.torch_mod.float32, device=self.device_obj
            )
            labels_tensor = self.torch_mod.tensor(
                labels, dtype=self.torch_mod.int32, device=self.device_obj
            )
        else:
            points_tensor = points_rel
            labels_tensor = labels

        result = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=int(self.frame_index),
            obj_id=int(obj_id),
            points=points_tensor,
            labels=labels_tensor,
        )

        if isinstance(result, tuple) and len(result) >= 4:
            _, out_obj_ids, _, video_res_masks = result
            for i, out_obj_id in enumerate(out_obj_ids):
                if int(out_obj_id) == int(obj_id):
                    return _as_numpy_bool_mask(video_res_masks[i])

        return None


def _pick_attr(obj: object, names: list[str]):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def try_build_local_preview_backend(
    sam_session: object,
    image_width: int,
    image_height: int,
    frame_index: int = 0,
) -> Optional[PreviewBackend]:
    """
    Best-effort adapter for locally initialized SAM preview sessions.

    Supports both request-style predictors and the direct
    predictor.add_new_points_or_box(...) API used by the local preview session.
    Returns None instead of crashing when the session object does not expose
    the expected predictor/session structure.
    """
    if sam_session is None:
        return None

    direct_request_target = sam_session if hasattr(sam_session, "handle_request") else None
    direct_session_id = _pick_attr(
        sam_session,
        ["session_id", "active_session_id", "preview_session_id", "sam_session_id"],
    )
    if direct_request_target is not None and direct_session_id is not None:
        return LocalSessionPreviewBackend(
            request_target=direct_request_target,
            session_id=str(direct_session_id),
            frame_index=frame_index,
            image_width=image_width,
            image_height=image_height,
        )

    predictor = _pick_attr(
        sam_session,
        ["predictor", "sam_predictor", "video_predictor", "preview_predictor", "_predictor"],
    )
    session_id = _pick_attr(
        sam_session,
        ["session_id", "active_session_id", "preview_session_id", "sam_session_id"],
    )
    if predictor is not None and session_id is not None and hasattr(predictor, "handle_request"):
        return LocalSessionPreviewBackend(
            request_target=predictor,
            session_id=str(session_id),
            frame_index=frame_index,
            image_width=image_width,
            image_height=image_height,
        )

    inference_state = _pick_attr(
        sam_session,
        ["inference_state", "state", "preview_state", "session_state"],
    )
    if predictor is not None and inference_state is not None and hasattr(predictor, "add_new_points_or_box"):
        return DirectPredictorPreviewBackend(
            predictor=predictor,
            inference_state=inference_state,
            image_width=image_width,
            image_height=image_height,
            frame_index=frame_index,
            torch_mod=_pick_attr(sam_session, ["torch_mod"]),
            device_obj=_pick_attr(sam_session, ["device_obj", "device"]),
        )

    nested = _pick_attr(
        sam_session,
        ["state", "preview_state", "session", "_session"],
    )
    if nested is not None:
        nested_target = nested if hasattr(nested, "handle_request") else _pick_attr(
            nested,
            ["predictor", "sam_predictor", "video_predictor", "_predictor"],
        )
        nested_session_id = _pick_attr(
            nested,
            ["session_id", "active_session_id", "preview_session_id", "sam_session_id"],
        )
        if nested_target is not None and nested_session_id is not None and hasattr(nested_target, "handle_request"):
            return LocalSessionPreviewBackend(
                request_target=nested_target,
                session_id=str(nested_session_id),
                frame_index=frame_index,
                image_width=image_width,
                image_height=image_height,
            )

    return None
