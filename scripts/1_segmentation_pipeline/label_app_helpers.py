#!/usr/bin/env python3
# label frist frame helpers
import os
import io
import sys
import gc
import json
import shutil
import socket
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import nullcontext
from typing import Tuple, Optional, Any

# Avoid shadowing the installed sam3 package with the local repo parent directory.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
sys.path = [p for p in sys.path if Path(p).resolve() != _PROJECT_ROOT]

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as e:
    raise SystemExit(
        "opencv-python is required.\n"
        "Install in your environment:\n"
        "  pip install opencv-python flask imageio numpy pillow"
    ) from e

try:
    from flask import Flask
    from werkzeug.serving import make_server
except ImportError as e:
    raise SystemExit(
        "flask is required.\n"
        "Install in your environment:\n"
        "  pip install flask"
    ) from e


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def build_default_palette() -> bytes:
    """
    Build a deterministic 256-color palette.
    Good enough for indexed mask PNGs without carrying a huge literal.
    """
    pal = []
    for i in range(256):
        r = ((i >> 0) & 1) * 128 + ((i >> 3) & 1) * 64 + ((i >> 6) & 1) * 32
        g = ((i >> 1) & 1) * 128 + ((i >> 4) & 1) * 64 + ((i >> 7) & 1) * 32
        b = ((i >> 2) & 1) * 128 + ((i >> 5) & 1) * 64
        pal.extend([r, g, b])
    return bytes(pal[:768])


DAVIS_PALETTE = build_default_palette()


@dataclass
class LabelAppConfig:
    video_path: str
    output_json: str
    frame_index: int = 0
    output_frame_image: Optional[str] = None
    object_ids: Tuple[int, ...] = (1, 2)
    host: str = "127.0.0.1"
    port: int = 5000
    open_browser: bool = False
    wait_timeout: Optional[float] = None
    enable_sam_preview: bool = True
    sam3_device: str = "auto"
    run_video_tracking: bool = False
    output_mask_dir: Optional[str] = None
    keep_extracted_frames: bool = False
    extracted_frames_dir: Optional[str] = None


@dataclass
class LabelState:
    video_path: str
    output_json: str
    frame_index: int
    frame_shape: tuple[int, int]
    object_ids: list[int]
    objects: dict[str, dict]
    current_index: int = 0
    saved: bool = False
    predicted_masks: dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    mask_version: int = 0

    @classmethod
    def create(
        cls,
        video_path: str,
        output_json: str,
        frame_index: int,
        h: int,
        w: int,
        object_ids: Tuple[int, ...],
    ) -> "LabelState":
        return cls(
            video_path=video_path,
            output_json=output_json,
            frame_index=frame_index,
            frame_shape=(h, w),
            object_ids=list(object_ids),
            objects={str(obj_id): {"points_xy": [], "labels": []} for obj_id in object_ids},
            predicted_masks={str(obj_id): None for obj_id in object_ids},
        )


@dataclass
class SamPreviewSession:
    torch_mod: Any = None
    sam_model: Any = None
    predictor: Any = None
    inference_state: Any = None
    device_obj: Any = None
    session_frames_dir: Optional[str] = None
    cleanup_session_frames: bool = False
    preview_available: bool = False


class ServerThread(threading.Thread):
    def __init__(self, app_: Flask, host_: str, port_: int):
        super().__init__(daemon=True)
        self.server = make_server(host_, port_, app_, threaded=True)
        self.ctx = app_.app_context()
        self.ctx.push()

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass


def normalize_config(cfg: LabelAppConfig) -> LabelAppConfig:
    return LabelAppConfig(
        video_path=str(Path(cfg.video_path).expanduser().resolve()),
        output_json=str(Path(cfg.output_json).expanduser().resolve()),
        frame_index=int(cfg.frame_index),
        output_frame_image=str(Path(cfg.output_frame_image).expanduser().resolve())
        if cfg.output_frame_image
        else None,
        object_ids=tuple(int(x) for x in cfg.object_ids),
        host=cfg.host,
        port=int(cfg.port),
        open_browser=bool(cfg.open_browser),
        wait_timeout=cfg.wait_timeout,
        enable_sam_preview=bool(cfg.enable_sam_preview),
        sam3_device=str(cfg.sam3_device),
        run_video_tracking=bool(cfg.run_video_tracking),
        output_mask_dir=str(Path(cfg.output_mask_dir).expanduser().resolve())
        if cfg.output_mask_dir
        else None,
        keep_extracted_frames=bool(cfg.keep_extracted_frames),
        extracted_frames_dir=str(Path(cfg.extracted_frames_dir).expanduser().resolve())
        if cfg.extracted_frames_dir
        else None,
    )


def cleanup_cuda(torch_mod, *objs, move_to_cpu_first: bool = False) -> None:
    if torch_mod is None:
        return

    if move_to_cpu_first:
        for obj in objs:
            if obj is None:
                continue
            try:
                if hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
            try:
                if hasattr(obj, "model") and obj.model is not None and hasattr(obj.model, "to"):
                    obj.model.to("cpu")
            except Exception:
                pass
            try:
                if hasattr(obj, "detector") and obj.detector is not None and hasattr(obj.detector, "to"):
                    obj.detector.to("cpu")
            except Exception:
                pass
            try:
                if hasattr(obj, "tracker") and obj.tracker is not None and hasattr(obj.tracker, "to"):
                    obj.tracker.to("cpu")
            except Exception:
                pass

    gc.collect()

    if torch_mod.cuda.is_available():
        try:
            torch_mod.cuda.synchronize()
        except Exception:
            pass
        try:
            torch_mod.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch_mod.cuda.ipc_collect()
        except Exception:
            pass


def log_cuda_mem(torch_mod, tag: str) -> None:
    if torch_mod is None or not torch_mod.cuda.is_available():
        return
    print(
        f"[CUDA] {tag}: "
        f"allocated={torch_mod.cuda.memory_allocated()/1024**2:.1f} MiB, "
        f"reserved={torch_mod.cuda.memory_reserved()/1024**2:.1f} MiB, "
        f"max_allocated={torch_mod.cuda.max_memory_allocated()/1024**2:.1f} MiB"
    )


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def get_video_num_frames(video_path: str) -> int:
    cap = open_video_capture(video_path)
    try:
        num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(num, 0)
    finally:
        cap.release()


def read_video_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = open_video_capture(video_path)
    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")
        if num_frames > 0 and frame_index >= num_frames:
            raise ValueError(
                f"frame_index {frame_index} is out of range for video with {num_frames} frames"
            )

        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        if not ok:
            # some backends ignore return value, so continue anyway
            pass

        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"Could not read frame {frame_index} from: {video_path}"
            )

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_first_frame(video_path: str) -> np.ndarray:
    return read_video_frame(video_path, 0)


def extract_video_frames(video_path: str, frames_dir: str) -> list[str]:
    os.makedirs(frames_dir, exist_ok=True)

    cap = open_video_capture(video_path)
    frame_paths: list[str] = []
    idx = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            out_path = os.path.join(frames_dir, f"{idx:05d}.jpg")
            if not cv2.imwrite(out_path, frame_bgr):
                raise RuntimeError(f"Failed to write frame to: {out_path}")

            frame_paths.append(out_path)
            idx += 1
    finally:
        cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from: {video_path}")

    return frame_paths


def mask_to_png_bytes(mask: np.ndarray) -> io.BytesIO:
    mask_u8 = (mask.astype(bool).astype(np.uint8) * 255)
    pil_img = Image.fromarray(mask_u8, mode="L")
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    return bio


def save_frame_image(frame_rgb: np.ndarray, output_frame_image: str) -> None:
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(output_frame_image) or ".", exist_ok=True)
    imageio.imwrite(output_frame_image, frame_rgb)


def save_ann_png(path: str, mask: np.ndarray, palette: bytes | None = DAVIS_PALETTE) -> None:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = Image.fromarray(mask, mode="P")
    if palette is not None:
        pal = list(palette)
        if len(pal) < 768:
            pal += [0] * (768 - len(pal))
        else:
            pal = pal[:768]
        img.putpalette(pal)
    img.save(path)


def put_per_obj_mask(per_obj_mask: dict[int, np.ndarray], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for object_id in sorted(per_obj_mask.keys())[::-1]:
        object_mask = np.asarray(per_obj_mask[object_id]).astype(bool).reshape(height, width)
        mask[object_mask] = np.uint8(object_id)
    return mask


def init_sam3_state_for_single_frame_only(predictor, frame_rgb: np.ndarray):
    import imageio.v2 as imageio

    tmp_dir = tempfile.mkdtemp(prefix="sam3_single_frame_only_")
    frame_path = os.path.join(tmp_dir, "00000.jpg")
    imageio.imwrite(frame_path, frame_rgb)

    state = predictor.init_state(video_path=tmp_dir)
    return state, tmp_dir


def resolve_device(torch_mod, device_str: str):
    if device_str == "auto":
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
        return torch_mod.device("cpu")

    if device_str.startswith("cuda"):
        if not torch_mod.cuda.is_available():
            raise RuntimeError("sam3_device=cuda requested, but CUDA is not available.")
        return torch_mod.device(device_str)

    if device_str == "mps":
        if not hasattr(torch_mod.backends, "mps") or not torch_mod.backends.mps.is_available():
            raise RuntimeError("sam3_device=mps requested, but MPS is not available.")
        return torch_mod.device("mps")

    return torch_mod.device("cpu")


def sam3_autocast_ctx(torch_mod, device):
    return nullcontext()


def build_sam3_tracker_bundle(torch_mod, sam3_device: str):
    from sam3.model_builder import build_sam3_video_model

    device = resolve_device(torch_mod, sam3_device)

    if device.type == "cuda":
        if torch_mod.cuda.get_device_properties(0).major >= 8:
            torch_mod.backends.cuda.matmul.allow_tf32 = True
            torch_mod.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "Support for MPS is preliminary. SAM3 is trained with CUDA and may "
            "produce different or degraded results on MPS."
        )

    print("Initializing SAM3 video model...")
    sam3_model = build_sam3_video_model(device=device)
    predictor = sam3_model.tracker

    if device.type == "cuda":
        predictor.backbone = sam3_model.detector.backbone

    print(f"SAM3 initialized on device: {device}")
    return sam3_model, predictor, device


def sam3_points_tensors(
    points_abs: np.ndarray,
    labels: np.ndarray,
    height: int,
    width: int,
    torch_mod,
    device,
):
    rel_points = [[float(x) / width, float(y) / height] for x, y in points_abs]
    points_tensor = torch_mod.tensor(rel_points, dtype=torch_mod.float32, device=device)
    labels_tensor = torch_mod.tensor(labels, dtype=torch_mod.int32, device=device)
    return points_tensor, labels_tensor


def init_sam3_state_from_video(
    predictor,
    video_path: str,
    frames_dir: str | None = None,
    keep_extracted_frames: bool = False,
):
    if frames_dir is not None and keep_extracted_frames:
        os.makedirs(frames_dir, exist_ok=True)

    extracted_dir = None
    cleanup_dir = False

    try:
        try:
            state = predictor.init_state(video_path=video_path)
            return state, None, False
        except Exception as first_err:
            print(
                f"SAM3 init_state(video_path=video_path) failed, "
                f"falling back to extracted frames: {first_err}"
            )

            if frames_dir is not None:
                extracted_dir = frames_dir
                os.makedirs(extracted_dir, exist_ok=True)
            else:
                extracted_dir = tempfile.mkdtemp(prefix="sam3_video_frames_")
                cleanup_dir = True

            extract_video_frames(video_path, extracted_dir)
            state = predictor.init_state(video_path=extracted_dir)
            return state, extracted_dir, cleanup_dir
    except Exception:
        if cleanup_dir and extracted_dir is not None:
            shutil.rmtree(extracted_dir, ignore_errors=True)
        raise


def run_sam3_video_propagation(
    video_path: str,
    prompts_by_object: dict[int, dict],
    output_mask_dir: str,
    sam3_device: str = "auto",
    keep_extracted_frames: bool = False,
    extracted_frames_dir: str | None = None,
) -> None:
    import torch

    video_path = str(Path(video_path).expanduser().resolve())
    output_mask_dir = str(Path(output_mask_dir).expanduser().resolve())
    if extracted_frames_dir is not None:
        extracted_frames_dir = str(Path(extracted_frames_dir).expanduser().resolve())

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    os.makedirs(output_mask_dir, exist_ok=True)

    first_frame = read_first_frame(video_path)
    height, width = first_frame.shape[:2]

    sam3_model = None
    predictor = None
    inference_state = None
    temp_frames_dir = None
    cleanup_frames = False
    device = None

    try:
        sam3_model, predictor, device = build_sam3_tracker_bundle(torch, sam3_device)

        inference_state, temp_frames_dir, cleanup_frames = init_sam3_state_from_video(
            predictor=predictor,
            video_path=video_path,
            frames_dir=extracted_frames_dir,
            keep_extracted_frames=keep_extracted_frames,
        )

        if isinstance(inference_state, dict):
            if "device" in inference_state:
                inference_state["device"] = device
            if "storage_device" in inference_state:
                inference_state["storage_device"] = device

        with torch.inference_mode(), sam3_autocast_ctx(torch, device):
            for obj_id, obj in prompts_by_object.items():
                points_xy = obj.get("points_xy", [])
                labels = obj.get("labels", [])

                if not points_xy:
                    continue

                points_abs = np.asarray(points_xy, dtype=np.float32)
                labels_np = np.asarray(labels, dtype=np.int32)

                points_tensor, labels_tensor = sam3_points_tensors(
                    points_abs=points_abs,
                    labels=labels_np,
                    height=height,
                    width=width,
                    torch_mod=torch,
                    device=device,
                )

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=int(obj_id),
                    points=points_tensor,
                    labels=labels_tensor,
                )

            print("Running SAM3 video propagation...")
            frame_count = 0

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                per_obj_output_mask = {}

                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_logits = out_mask_logits[i]
                    mask_np = (mask_logits[0] > 0.0).detach().cpu().numpy().astype(bool)
                    per_obj_output_mask[int(out_obj_id)] = mask_np

                packed = put_per_obj_mask(per_obj_output_mask, height, width)
                out_path = os.path.join(output_mask_dir, f"{int(out_frame_idx):05d}.png")
                save_ann_png(out_path, packed, palette=DAVIS_PALETTE)
                frame_count += 1

        print(f"Saved {frame_count} propagated masks to: {output_mask_dir}")

    finally:
        cleanup_cuda(
            torch,
            predictor,
            sam3_model,
            move_to_cpu_first=(device is not None and getattr(device, "type", "") == "cuda"),
        )
        if cleanup_frames and temp_frames_dir is not None:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)


def validate_config(cfg: LabelAppConfig) -> None:
    if not os.path.isfile(cfg.video_path):
        raise FileNotFoundError(cfg.video_path)

    num_frames = get_video_num_frames(cfg.video_path)
    if num_frames > 0 and not (0 <= cfg.frame_index < num_frames):
        raise ValueError(
            f"frame_index {cfg.frame_index} is out of range for {cfg.video_path} "
            f"(num_frames={num_frames})"
        )

    if cfg.run_video_tracking and not cfg.output_mask_dir:
        raise RuntimeError("output_mask_dir is required when run_video_tracking=True.")
    if cfg.keep_extracted_frames and not cfg.extracted_frames_dir:
        raise RuntimeError("extracted_frames_dir is required when keep_extracted_frames=True.")


def prepare_frame_assets(
    cfg: LabelAppConfig,
) -> tuple[np.ndarray, int, int, str, Optional[str]]:
    import imageio.v2 as imageio

    frame_rgb = read_video_frame(cfg.video_path, cfg.frame_index)
    h, w = frame_rgb.shape[:2]

    print(f"Loaded frame {cfg.frame_index}: {w} x {h}")
    print(f"Video: {cfg.video_path}")

    if cfg.output_frame_image:
        save_frame_image(frame_rgb, cfg.output_frame_image)
        print(f"Saved frame image to: {cfg.output_frame_image}")

    tmp_dir = None
    if cfg.output_frame_image and os.path.isfile(cfg.output_frame_image):
        frame_path_to_serve = str(Path(cfg.output_frame_image).resolve())
    else:
        tmp_dir = tempfile.mkdtemp(prefix="label_frame_")
        filename = f"frame_{cfg.frame_index:06d}.jpg"
        frame_path_to_serve = str(Path(tmp_dir, filename).resolve())
        imageio.imwrite(frame_path_to_serve, frame_rgb)

    os.makedirs(os.path.dirname(cfg.output_json) or ".", exist_ok=True)
    return frame_rgb, h, w, frame_path_to_serve, tmp_dir


def init_sam_preview(cfg: LabelAppConfig, frame_rgb: np.ndarray) -> SamPreviewSession:
    session = SamPreviewSession()

    if not (cfg.enable_sam_preview or cfg.run_video_tracking):
        return session

    import torch

    session.torch_mod = torch
    session.sam_model, session.predictor, session.device_obj = build_sam3_tracker_bundle(
        torch, cfg.sam3_device
    )

    try:
        if cfg.enable_sam_preview and not cfg.run_video_tracking:
            print("[preview] initializing SAM3 from single frame only...")
            session.inference_state, session.session_frames_dir = init_sam3_state_for_single_frame_only(
                predictor=session.predictor,
                frame_rgb=frame_rgb,
            )
            session.cleanup_session_frames = True
        else:
            print("[preview] initializing SAM3 from full video...")
            session.inference_state, session.session_frames_dir, session.cleanup_session_frames = init_sam3_state_from_video(
                predictor=session.predictor,
                video_path=cfg.video_path,
                frames_dir=cfg.extracted_frames_dir,
                keep_extracted_frames=cfg.keep_extracted_frames,
            )

        if isinstance(session.inference_state, dict) and session.device_obj is not None:
            if "device" in session.inference_state:
                session.inference_state["device"] = session.device_obj
            if "storage_device" in session.inference_state:
                session.inference_state["storage_device"] = session.device_obj

        if session.device_obj is not None and session.device_obj.type != "cuda":
            print(
                "[preview] CPU/MPS preview disabled for this SAM3 build; "
                "continuing in points-only mode."
            )
            session.predictor = None
            session.inference_state = None
            return session

        session.preview_available = True
        log_cuda_mem(torch, "after SAM3 init")
        return session

    except Exception as e:
        print(f"[preview] disabled due to init failure: {e}")
        session.predictor = None
        session.inference_state = None
        return session



def rebuild_preview_state_for_current_frame(
    state: LabelState,
    sam: SamPreviewSession,
) -> None:
    if sam.predictor is None:
        raise RuntimeError('SAM3 preview predictor is not initialized.')

    frame_rgb = read_video_frame(state.video_path, state.frame_index)

    if sam.cleanup_session_frames and sam.session_frames_dir is not None:
        shutil.rmtree(sam.session_frames_dir, ignore_errors=True)
        sam.session_frames_dir = None
        sam.cleanup_session_frames = False

    sam.inference_state, sam.session_frames_dir = init_sam3_state_for_single_frame_only(
        predictor=sam.predictor,
        frame_rgb=frame_rgb,
    )
    sam.cleanup_session_frames = True

    if isinstance(sam.inference_state, dict):
        if 'device' in sam.inference_state:
            sam.inference_state['device'] = sam.device_obj
        if 'storage_device' in sam.inference_state:
            sam.inference_state['storage_device'] = sam.device_obj


def clear_predicted_masks(state: LabelState) -> None:
    for obj_id in state.object_ids:
        state.predicted_masks[str(obj_id)] = None


def replay_all_points_into_sam3(
    state: LabelState,
    sam: SamPreviewSession,
) -> None:
    if (
        not sam.preview_available
        or sam.predictor is None
        or sam.inference_state is None
        or sam.device_obj is None
        or sam.torch_mod is None
    ):
        clear_predicted_masks(state)
        state.mask_version += 1
        return

    h, w = state.frame_shape

    try:
        sam.predictor.clear_all_points_in_video(sam.inference_state)
    except Exception:
        rebuild_preview_state_for_current_frame(state, sam)

    clear_predicted_masks(state)

    if isinstance(sam.inference_state, dict):
        if "device" in sam.inference_state:
            sam.inference_state["device"] = sam.device_obj
        if "storage_device" in sam.inference_state:
            sam.inference_state["storage_device"] = sam.device_obj

    with sam.torch_mod.inference_mode(), sam3_autocast_ctx(sam.torch_mod, sam.device_obj):
        for obj_id in state.object_ids:
            obj = state.objects[str(obj_id)]
            points_xy = obj["points_xy"]
            labels = obj["labels"]

            if not points_xy:
                continue

            points_abs = np.asarray(points_xy, dtype=np.float32)
            labels_np = np.asarray(labels, dtype=np.int32)

            points_tensor, labels_tensor = sam3_points_tensors(
                points_abs=points_abs,
                labels=labels_np,
                height=h,
                width=w,
                torch_mod=sam.torch_mod,
                device=sam.device_obj,
            )

            result = sam.predictor.add_new_points_or_box(
                inference_state=sam.inference_state,
                frame_idx=0,
                obj_id=int(obj_id),
                points=points_tensor,
                labels=labels_tensor,
            )

            if not isinstance(result, tuple) or len(result) < 4:
                raise RuntimeError(
                    "Unexpected return value from predictor.add_new_points_or_box(...)."
                )

            _, out_obj_ids, _, video_res_masks = result

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (video_res_masks[i][0] > 0.0).detach().cpu().numpy().astype(bool)
                state.predicted_masks[str(int(out_obj_id))] = mask

    state.mask_version += 1


def write_prompts_json(output_json: str, state: LabelState) -> None:
    for obj_id in state.object_ids:
        obj = state.objects[str(obj_id)]
        if len(obj["points_xy"]) == 0:
            raise RuntimeError(f"Object {obj_id} has no points.")

    output = {
        "frame_index": int(state.frame_index),
        "objects": state.objects,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved prompts JSON to: {output_json}")


def load_prompts_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_run_video_tracking(cfg: LabelAppConfig, state: LabelState) -> None:
    if not cfg.run_video_tracking:
        return

    prompts_by_object = {
        int(obj_id): state.objects[str(obj_id)]
        for obj_id in state.object_ids
    }

    run_sam3_video_propagation(
        video_path=cfg.video_path,
        prompts_by_object=prompts_by_object,
        output_mask_dir=cfg.output_mask_dir,
        sam3_device=cfg.sam3_device,
        keep_extracted_frames=cfg.keep_extracted_frames,
        extracted_frames_dir=cfg.extracted_frames_dir,
    )


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def cleanup_label_session(
    sam: Optional[SamPreviewSession],
    state: LabelState,
    tmp_dir: Optional[str],
) -> None:
    if sam is None:
        sam = SamPreviewSession()

    try:
        clear_predicted_masks(state)
    except Exception:
        pass

    log_cuda_mem(sam.torch_mod, "before SAM3 cleanup")

    cleanup_cuda(
        sam.torch_mod,
        sam.predictor,
        sam.sam_model,
        move_to_cpu_first=(
            sam.device_obj is not None and getattr(sam.device_obj, "type", "") == "cuda"
        ),
    )

    if sam.cleanup_session_frames and sam.session_frames_dir is not None:
        shutil.rmtree(sam.session_frames_dir, ignore_errors=True)

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    log_cuda_mem(sam.torch_mod, "after SAM3 cleanup")
    print("Label server stopped.")
