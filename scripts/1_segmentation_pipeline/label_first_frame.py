#!/usr/bin/env python3
# label first frame

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from label_app_helpers import (
    LabelAppConfig,
    LabelState,
    SamPreviewSession,
    ServerThread,
    cleanup_label_session,
    init_sam_preview,
    normalize_config,
    port_in_use,
    prepare_frame_assets,
    validate_config,
)
from label_preview_backends import (
    PreviewBackend,
    try_build_local_preview_backend,
)
from label_webapp import create_label_app


def wait_for_server_ready(url: str) -> None:
    import requests

    health_url = f"{url}/health"
    for _ in range(50):
        try:
            r = requests.get(health_url, timeout=0.5)
            if r.ok:
                return
        except Exception:
            time.sleep(0.1)

    raise RuntimeError(f"Label server failed to start on {url}")


def run_label_first_frame(
    video_path: str,
    output_json: str,
    frame_index: int = 0,
    output_frame_image: str | None = None,
    object_ids: tuple[int, ...] = (1, 2),
    host: str = "127.0.0.1",
    port: int = 5000,
    open_browser: bool = False,
    wait_timeout: float | None = None,
    enable_sam_preview: bool = True,
    sam3_device: str = "auto",
    run_video_tracking: bool = False,
    output_mask_dir: str | None = None,
    keep_extracted_frames: bool = False,
    extracted_frames_dir: str | None = None,
    preview_update_fn=None,
    preview_backend: Optional[PreviewBackend] = None,
) -> dict:
    cfg = normalize_config(
        LabelAppConfig(
            video_path=video_path,
            output_json=output_json,
            frame_index=frame_index,
            output_frame_image=output_frame_image,
            object_ids=object_ids,
            host=host,
            port=port,
            open_browser=open_browser,
            wait_timeout=wait_timeout,
            enable_sam_preview=enable_sam_preview,
            sam3_device=sam3_device,
            run_video_tracking=run_video_tracking,
            output_mask_dir=output_mask_dir,
            keep_extracted_frames=keep_extracted_frames,
            extracted_frames_dir=extracted_frames_dir,
        )
    )
    validate_config(cfg)

    frame_rgb, h, w, frame_path_to_serve, tmp_dir = prepare_frame_assets(cfg)
    state = LabelState.create(
        cfg.video_path,
        cfg.output_json,
        cfg.frame_index,
        h,
        w,
        cfg.object_ids,
    )

    sam = SamPreviewSession()
    effective_preview_backend: Optional[PreviewBackend] = preview_backend

    if effective_preview_backend is not None:
        cfg.enable_sam_preview = True
        print("Using provided active SAM3 preview backend. No second model is created.")
    elif cfg.enable_sam_preview or cfg.run_video_tracking:
        if cfg.enable_sam_preview:
            print("Initializing SAM3 preview model...")
        else:
            print("Initializing SAM3 for tracking-only mode...")

        sam = init_sam_preview(cfg, frame_rgb)

        if cfg.enable_sam_preview:
            effective_preview_backend = try_build_local_preview_backend(
                sam_session=sam,
                image_width=w,
                image_height=h,
                frame_index=0,
            )

            if effective_preview_backend is not None:
                print("Local SAM3 preview backend ready.")
            else:
                print(
                    "Warning: local SAM3 session was initialized, but could not be adapted "
                    "to the preview backend API. Continuing without live preview."
                )
    else:
        print("SAM3 preview disabled. Starting manual labeling UI without local model preview.")

    save_event = threading.Event()
    error_holder = {"error": None}

    app = create_label_app(
        cfg=cfg,
        state=state,
        frame_path_to_serve=frame_path_to_serve,
        save_event=save_event,
        error_holder=error_holder,
        preview_backend=effective_preview_backend,
        preview_update_fn=preview_update_fn,
    )

    if port_in_use(cfg.host, cfg.port):
        raise RuntimeError(
            f"Port {cfg.port} on host {cfg.host} is already in use.\n"
            f"Choose another port, or stop the existing process:\n"
            f"  lsof -i :{cfg.port}"
        )

    server_thread = ServerThread(app, cfg.host, cfg.port)
    server_thread.start()

    url = f"http://{cfg.host}:{cfg.port}"
    print("\nNotebook-style labeling app is starting...")
    print(f"URL: {url}")
    print(f"Debug preview URL: {url}/debug_preview")
    print(f"Label frame index: {cfg.frame_index}")

    if cfg.host in ("127.0.0.1", "localhost"):
        print("\nIf you are connecting remotely, use SSH port forwarding, for example:")
        print(f"ssh -L {cfg.port}:localhost:{cfg.port} user@your-server")
        print(f"Then open: http://localhost:{cfg.port}")

    try:
        wait_for_server_ready(url)
        print("Notebook-style labeling app is ready.")
        print(f"Open this URL in your browser: {url}")

        if cfg.open_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()

        if cfg.wait_timeout is None:
            save_event.wait()
        else:
            ok = save_event.wait(timeout=cfg.wait_timeout)
            if not ok:
                raise TimeoutError(
                    f"Labeling UI timed out after {cfg.wait_timeout} seconds without Save."
                )

        if error_holder["error"] is not None:
            raise RuntimeError(f"Labeling failed: {error_holder['error']}")

        if not Path(cfg.output_json).is_file():
            raise RuntimeError(
                "Save was signaled, but output JSON was not found.\n"
                f"Expected: {cfg.output_json}"
            )

        return {
            "frame_index": cfg.frame_index,
            "objects": state.objects,
        }

    finally:
        server_thread.shutdown()
        server_thread.join(timeout=2.0)
        cleanup_label_session(sam, state, tmp_dir)


def run_label_frame(*args, **kwargs):
    return run_label_first_frame(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Notebook-style browser labeling of any frame of a video "
            "with optional SAM3 preview/tracking"
        )
    )
    ap.add_argument("--video_path", required=True, help="Path to the input video")
    ap.add_argument("--output_json", required=True, help="Path to save prompts JSON")
    ap.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Frame index in the video to label",
    )
    ap.add_argument(
        "--output_frame_image",
        default=None,
        help="Optional path to save the labeled frame image",
    )
    ap.add_argument(
        "--object_ids",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Object IDs to label, e.g. --object_ids 1 2 3",
    )
    ap.add_argument("--host", default="127.0.0.1", help="Host for web app")
    ap.add_argument("--port", type=int, default=5000, help="Port for web app")
    ap.add_argument("--open_browser", action="store_true", help="Open browser automatically")
    ap.add_argument(
        "--wait_timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for the labeling session",
    )
    ap.add_argument("--disable_sam_preview", action="store_true", help="Disable live mask preview")
    ap.add_argument("--sam3_device", default="auto", help="auto, cuda, cpu, or mps")
    ap.add_argument(
        "--run_video_tracking",
        action="store_true",
        help="After labeling, run full video propagation",
    )
    ap.add_argument(
        "--output_mask_dir",
        default=None,
        help="Directory to save propagated per-frame masks",
    )
    ap.add_argument(
        "--keep_extracted_frames",
        action="store_true",
        help="Keep extracted JPEG frames if video->frames fallback is needed",
    )
    ap.add_argument(
        "--extracted_frames_dir",
        default=None,
        help="Directory to store extracted frames when fallback extraction is used",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_video_tracking and not args.output_mask_dir:
        raise SystemExit("--output_mask_dir is required when --run_video_tracking is set")

    if args.keep_extracted_frames and not args.extracted_frames_dir:
        raise SystemExit("--extracted_frames_dir is required when --keep_extracted_frames is set")

    run_label_first_frame(
        video_path=args.video_path,
        output_json=args.output_json,
        frame_index=args.frame_index,
        output_frame_image=args.output_frame_image,
        object_ids=tuple(args.object_ids),
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        wait_timeout=args.wait_timeout,
        enable_sam_preview=not args.disable_sam_preview,
        sam3_device=args.sam3_device,
        run_video_tracking=args.run_video_tracking,
        output_mask_dir=args.output_mask_dir,
        keep_extracted_frames=args.keep_extracted_frames,
        extracted_frames_dir=args.extracted_frames_dir,
    )


if __name__ == "__main__":
    main()
