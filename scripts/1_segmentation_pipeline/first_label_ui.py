#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from pathlib import Path
from threading import Thread

import cv2
import requests
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

STATE = {
    "config_path": None,
    "mode": None,
    "video_path": None,
    "fps": 0.0,
    "frame_count": 0,
    "duration_sec": 0.0,
    "preview_duration_sec": 30.0,
    "preview_max_frame": 0,
    "selected_frame": 0,
    "num_objects": 1,
    "fragment_length_sec": 30.0,
    "overlap_frames": 1,
    # Separation-aware fragmentation: cut where the two rats are separated so
    # segment boundaries never fall on a touching-rats frame. When enabled, the
    # fixed fragment length is ignored and min/max bound the cut search window.
    "separation_enabled": False,
    "min_segment_seconds": 15.0,
    "max_segment_seconds": 23.0,
    "output_path": None,
    "saved_output": None,
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level config JSON must be an object")
    return data


def save_json(path: str, data: dict) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def probe_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0:
        fps = 25.0

    duration_sec = frame_count / fps if frame_count > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def extract_frame_jpeg(video_path: str, frame_idx: int) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames")

    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx}")

    out_dir = Path("/tmp/first_label_ui")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_{frame_idx}.jpg"

    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        raise RuntimeError(f"Could not save preview frame to {out_path}")

    return str(out_path)


def _get_fragment_cfg(cfg: dict) -> dict:
    fragment = cfg.get("fragment")
    if not isinstance(fragment, dict):
        fragment = {}
        cfg["fragment"] = fragment
    return fragment


def _get_label_cfg(cfg: dict) -> dict:
    label = cfg.get("label")
    if not isinstance(label, dict):
        label = {}
        cfg["label"] = label
    return label


def _get_first_label_ui_cfg(cfg: dict) -> dict:
    ui_cfg = cfg.get("first_label_ui")
    if not isinstance(ui_cfg, dict):
        ui_cfg = {}
        cfg["first_label_ui"] = ui_cfg
    return ui_cfg


def _build_saved_payload() -> dict:
    first_label_time_sec = 0.0
    if STATE["fps"] > 0:
        first_label_time_sec = round(STATE["selected_frame"] / STATE["fps"], 4)

    return {
        "mode": STATE["mode"],
        "config_path": STATE["config_path"],
        "video_path": STATE["video_path"],
        "num_objects": STATE["num_objects"],
        "fragment_length_sec": STATE["fragment_length_sec"],
        "overlap_frames": STATE["overlap_frames"],
        "separation_enabled": STATE["separation_enabled"],
        "min_segment_seconds": STATE["min_segment_seconds"],
        "max_segment_seconds": STATE["max_segment_seconds"],
        "first_label_frame": STATE["selected_frame"],
        "first_label_time_sec": first_label_time_sec,
        "preview_duration_sec": STATE["preview_duration_sec"],
    }


@app.route("/")
def index():
    return render_template("first_label_ui.html")


@app.route("/api/config")
def api_config():
    return jsonify(
        {
            "video_path": STATE["video_path"],
            "fps": STATE["fps"],
            "frame_count": STATE["frame_count"],
            "duration_sec": STATE["duration_sec"],
            "preview_duration_sec": STATE["preview_duration_sec"],
            "preview_max_frame": STATE["preview_max_frame"],
            "selected_frame": STATE["selected_frame"],
            "num_objects": STATE["num_objects"],
            "fragment_length_sec": STATE["fragment_length_sec"],
            "overlap_frames": STATE["overlap_frames"],
            "separation_enabled": STATE["separation_enabled"],
            "min_segment_seconds": STATE["min_segment_seconds"],
            "max_segment_seconds": STATE["max_segment_seconds"],
            "mode": STATE["mode"],
            "output_path": STATE["output_path"],
            "config_path": STATE["config_path"],
            "first_label_frame": STATE["selected_frame"],
            "first_label_time_sec": (
                round(STATE["selected_frame"] / STATE["fps"], 4)
                if STATE["fps"] > 0 else 0.0
            ),
        }
    )


@app.route("/video")
def serve_video():
    video_path = STATE["video_path"]
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    return send_file(video_path, conditional=True)


@app.route("/api/frame_preview")
def api_frame_preview():
    try:
        frame_idx = int(request.args.get("frame", STATE["selected_frame"]))
        jpg_path = extract_frame_jpeg(STATE["video_path"], frame_idx)
        return send_file(jpg_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/select_frame", methods=["POST"])
def api_select_frame():
    data = request.get_json(force=True)
    frame_idx = int(data.get("frame", 0))
    frame_idx = max(0, min(frame_idx, STATE["preview_max_frame"]))
    STATE["selected_frame"] = frame_idx
    return jsonify(
        {
            "ok": True,
            "selected_frame": STATE["selected_frame"],
            "first_label_frame": STATE["selected_frame"],
            "first_label_time_sec": (
                round(STATE["selected_frame"] / STATE["fps"], 4)
                if STATE["fps"] > 0 else 0.0
            ),
        }
    )


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)

    num_objects = int(data.get("num_objects", STATE["num_objects"]))
    overlap_frames = int(data.get("overlap_frames", STATE["overlap_frames"]))
    separation_enabled = bool(data.get("separation_enabled", STATE["separation_enabled"]))
    min_seg = float(data.get("min_segment_seconds", STATE["min_segment_seconds"]))
    max_seg = float(data.get("max_segment_seconds", STATE["max_segment_seconds"]))
    # segment_seconds is still accepted (used as the fallback when separation is
    # off) but is no longer the primary control.
    fragment_length_sec = float(
        data.get("fragment_length_sec", STATE["fragment_length_sec"])
    )

    if num_objects < 1:
        return jsonify({"error": "num_objects must be >= 1"}), 400
    if overlap_frames < 1:
        return jsonify({"error": "overlap_frames must be >= 1"}), 400
    if separation_enabled:
        if min_seg <= 0 or max_seg <= 0:
            return jsonify({"error": "min/max segment seconds must be > 0"}), 400
        if min_seg > max_seg:
            return jsonify({"error": "min_segment_seconds must be <= max_segment_seconds"}), 400
    elif fragment_length_sec <= 0:
        return jsonify({"error": "fragment_length_sec must be > 0"}), 400

    STATE["num_objects"] = num_objects
    STATE["overlap_frames"] = overlap_frames
    STATE["separation_enabled"] = separation_enabled
    STATE["min_segment_seconds"] = min_seg
    STATE["max_segment_seconds"] = max_seg
    STATE["fragment_length_sec"] = fragment_length_sec

    return jsonify(
        {
            "ok": True,
            "num_objects": STATE["num_objects"],
            "overlap_frames": STATE["overlap_frames"],
            "separation_enabled": STATE["separation_enabled"],
            "min_segment_seconds": STATE["min_segment_seconds"],
            "max_segment_seconds": STATE["max_segment_seconds"],
            "fragment_length_sec": STATE["fragment_length_sec"],
        }
    )


@app.route("/api/save", methods=["POST"])
def api_save():
    try:
        config_path = STATE["config_path"]
        cfg = load_json(config_path)

        fragment = _get_fragment_cfg(cfg)
        label = _get_label_cfg(cfg)
        ui_cfg = _get_first_label_ui_cfg(cfg)

        first_label_time_sec = (
            round(STATE["selected_frame"] / STATE["fps"], 4)
            if STATE["fps"] > 0 else 0.0
        )

        # write directly into existing config
        # segment_seconds is retained as the fallback used only when
        # separation-aware fragmentation is OFF.
        fragment["segment_seconds"] = STATE["fragment_length_sec"]
        fragment["first_label_frame"] = STATE["selected_frame"]
        fragment["overlap_frames"] = STATE["overlap_frames"]

        # Separation-aware fragmentation block. Preserve any existing ckpt/
        # device the user set (e.g. copied from the refiner); only update the
        # enabled flag and the min/max window here.
        sep = fragment.get("separation", {}) or {}
        sep["enabled"] = STATE["separation_enabled"]
        sep["min_segment_seconds"] = STATE["min_segment_seconds"]
        sep["max_segment_seconds"] = STATE["max_segment_seconds"]
        # if no ckpt set for separation, fall back to the refiner's ckpt so the
        # detector has a model to load
        refiner_cfg = cfg.get("refiner", {}) or {}
        if not sep.get("ckpt") and refiner_cfg.get("ckpt"):
            sep["ckpt"] = refiner_cfg["ckpt"]
        if not sep.get("device") and refiner_cfg.get("device"):
            sep["device"] = refiner_cfg["device"]
        fragment["separation"] = sep

        label["object_ids"] = list(range(1, STATE["num_objects"] + 1))

        # keep UI metadata inside config as well
        ui_cfg["mode"] = STATE["mode"]
        ui_cfg["config_path"] = STATE["config_path"]
        ui_cfg["video_path"] = STATE["video_path"]
        ui_cfg["num_objects"] = STATE["num_objects"]
        ui_cfg["fragment_length_sec"] = STATE["fragment_length_sec"]
        ui_cfg["first_label_frame"] = STATE["selected_frame"]
        ui_cfg["first_label_time_sec"] = first_label_time_sec
        ui_cfg["preview_duration_sec"] = STATE["preview_duration_sec"]

        save_json(config_path, cfg)

        saved_output = _build_saved_payload()
        STATE["saved_output"] = saved_output

        return jsonify(
            {
                "ok": True,
                "output_path": str(Path(config_path).expanduser().resolve()),
                "config_path": str(Path(config_path).expanduser().resolve()),
                "message": f"JSON config updated and saved: {Path(config_path).expanduser().resolve()}",
                "data": saved_output,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        return jsonify({"error": "Not running with Werkzeug server"}), 500
    func()
    return jsonify({"ok": True})


def initialize_state(config_path: str, mode: str):
    cfg = load_json(config_path)

    server_cfg = cfg.get("server", {}) or {}
    fragment_cfg = cfg.get("fragment", {}) or {}
    label_cfg = cfg.get("label", {}) or {}
    ui_cfg = cfg.get("first_label_ui", {}) or {}

    video_path = server_cfg.get("input_video")
    if not video_path:
        raise ValueError("Missing server.input_video in config JSON")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    meta = probe_video(video_path)

    preview_duration_sec = min(30.0, meta["duration_sec"])
    preview_max_frame = 0
    if meta["fps"] > 0:
        preview_max_frame = min(
            meta["frame_count"] - 1,
            max(0, int(math.floor(preview_duration_sec * meta["fps"])) - 1),
        )

    object_ids = label_cfg.get("object_ids", [1])
    if not isinstance(object_ids, list) or len(object_ids) == 0:
        object_ids = [1]

    selected_frame = fragment_cfg.get(
        "first_label_frame",
        ui_cfg.get("first_label_frame", 0),
    )
    try:
        selected_frame = int(selected_frame)
    except Exception:
        selected_frame = 0
    selected_frame = max(0, min(selected_frame, preview_max_frame))

    fragment_length_sec = fragment_cfg.get(
        "segment_seconds",
        ui_cfg.get("fragment_length_sec", 30.0),
    )
    try:
        fragment_length_sec = float(fragment_length_sec)
    except Exception:
        fragment_length_sec = 30.0
    if fragment_length_sec <= 0:
        fragment_length_sec = 30.0

    STATE["config_path"] = str(Path(config_path).expanduser().resolve())
    STATE["mode"] = mode
    STATE["video_path"] = video_path
    STATE["fps"] = meta["fps"]
    STATE["frame_count"] = meta["frame_count"]
    STATE["duration_sec"] = meta["duration_sec"]
    STATE["preview_duration_sec"] = preview_duration_sec
    STATE["preview_max_frame"] = preview_max_frame
    STATE["selected_frame"] = selected_frame
    sep_cfg = fragment_cfg.get("separation", {}) or {}

    def _f(v, d):
        try:
            return float(v)
        except Exception:
            return d

    STATE["num_objects"] = len(object_ids)
    STATE["fragment_length_sec"] = fragment_length_sec
    STATE["overlap_frames"] = int(fragment_cfg.get("overlap_frames", 1) or 1)
    STATE["separation_enabled"] = bool(sep_cfg.get("enabled", False))
    STATE["min_segment_seconds"] = _f(sep_cfg.get("min_segment_seconds", 15.0), 15.0)
    STATE["max_segment_seconds"] = _f(sep_cfg.get("max_segment_seconds", 23.0), 23.0)
    STATE["output_path"] = str(Path(config_path).expanduser().resolve())
    STATE["saved_output"] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "mode",
        nargs="?",
        default="fragment",
        help="Pipeline mode, e.g. fragment",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def run_fragment_setup_ui(
    config_path: str,
    mode: str = "fragment",
    host: str = "127.0.0.1",
    port: int = 5000,
) -> dict:
    initialize_state(config_path, mode)

    print(f"Loaded config: {STATE['config_path']}")
    print(f"Video: {STATE['video_path']}")
    print(f"Preview duration: {STATE['preview_duration_sec']} sec")
    print(f"Open UI at: http://{host}:{port}")

    server_thread = Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    try:
        while STATE["saved_output"] is None:
            time.sleep(0.2)
    finally:
        try:
            requests.post(f"http://{host}:{port}/api/shutdown", timeout=1)
        except Exception:
            pass

    return STATE["saved_output"]


def main():
    args = parse_args()
    result = run_fragment_setup_ui(
        config_path=args.config,
        mode=args.mode,
        host=args.host,
        port=args.port,
    )

    print("\n=== SAVED RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()