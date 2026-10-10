#!/usr/bin/env python3
from __future__ import annotations

import random
import os

from flask import Flask, jsonify, request, send_file, render_template

from label_app_helpers import (
    clear_predicted_masks,
    mask_to_png_bytes,
    maybe_run_video_tracking,
    write_prompts_json,
)


def _predicted_available_map(state) -> dict[str, bool]:
    return {
        str(obj_id): state.predicted_masks.get(str(obj_id)) is not None
        for obj_id in state.object_ids
    }


def _preview_available(preview_backend) -> bool:
    return preview_backend is not None


def _get_current_obj_id(state) -> int:
    return state.object_ids[state.current_index]


def _refresh_preview_for_object(state, preview_backend, obj_id: int) -> None:
    if preview_backend is None:
        clear_predicted_masks(state)
        state.mask_version += 1
        return

    obj_key = str(obj_id)
    obj = state.objects[obj_key]
    points_xy = obj.get("points_xy", [])
    labels = obj.get("labels", [])

    if not points_xy:
        state.predicted_masks[obj_key] = None
        state.mask_version += 1
        return

    mask = preview_backend.predict(
        points_abs=points_xy,
        labels=labels,
        obj_id=obj_id,
    )

    state.predicted_masks[obj_key] = mask
    state.mask_version += 1


def build_overlay_colors(object_ids: list[int]) -> dict[str, str]:
    # default for <=2 objects
    default_colors = {
        "1": "#ff4d4f",
        "2": "#40a9ff",
    }

    if len(object_ids) <= 2:
        return {
            str(obj_id): default_colors.get(str(obj_id), "#73d13d")
            for obj_id in object_ids
        }

    # random colors for >2 objects
    rng = random.Random(12345)  # deterministic
    colors = {}
    used = set()

    for obj_id in object_ids:
        while True:
            color = "#{:02x}{:02x}{:02x}".format(
                rng.randint(40, 235),
                rng.randint(40, 235),
                rng.randint(40, 235),
            )
            if color not in used:
                used.add(color)
                colors[str(obj_id)] = color
                break

    return colors


def create_label_app(
    cfg,
    state,
    frame_path_to_serve,
    save_event,
    error_holder,
    preview_backend=None,
    preview_update_fn=None,
):
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )

    overlay_colors = build_overlay_colors(state.object_ids)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/")
    def index():
        return render_template("label_first_frame.html")

    @app.get("/frame")
    def frame():
        return send_file(frame_path_to_serve)

    @app.get("/debug_preview")
    def debug_preview():
        current_obj_id = _get_current_obj_id(state)
        return jsonify(
            {
                "ok": True,
                "current_object_id": current_obj_id,
                "mask_version": state.mask_version,
                "preview_available": _preview_available(preview_backend),
                "predicted_available": _predicted_available_map(state),
            }
        )

    @app.get("/state")
    def get_state():
        current_obj_id = _get_current_obj_id(state)
        return jsonify(
            {
                "frame_index": state.frame_index,
                "frame_shape": state.frame_shape,
                "object_ids": state.object_ids,
                "current_object_id": current_obj_id,
                "current_object_index": state.current_index,
                "current_index": state.current_index,
                "objects": state.objects,
                "saved": state.saved,
                "mask_version": state.mask_version,
                "preview_available": _preview_available(preview_backend),
                "predicted_available": _predicted_available_map(state),
                "overlay_colors": overlay_colors,
            }
        )

    @app.post("/set_current_object")
    def set_current_object():
        payload = request.get_json(silent=True) or {}
        obj_id = int(payload["obj_id"])

        if obj_id not in state.object_ids:
            return jsonify({"error": f"Unknown object id {obj_id}"}), 400

        state.current_index = state.object_ids.index(obj_id)
        return jsonify(
            {
                "ok": True,
                "current_object_id": obj_id,
                "current_object_index": state.current_index,
            }
        )

    @app.post("/next_object")
    def next_object():
        state.current_index = (state.current_index + 1) % len(state.object_ids)
        return jsonify(
            {
                "ok": True,
                "current_object_id": _get_current_obj_id(state),
                "current_object_index": state.current_index,
            }
        )

    @app.post("/add_point")
    def add_point():
        payload = request.get_json(force=True)

        x = float(payload["x"])
        y = float(payload["y"])
        label = int(payload["label"])
        obj_id = int(payload.get("obj_id", _get_current_obj_id(state)))

        if obj_id not in state.object_ids:
            return jsonify({"error": f"Unknown object id {obj_id}"}), 400

        if label not in (0, 1):
            return jsonify({"error": "label must be 0 or 1"}), 400

        obj = state.objects[str(obj_id)]
        obj["points_xy"].append([x, y])
        obj["labels"].append(label)

        try:
            _refresh_preview_for_object(state, preview_backend, obj_id)
            if preview_update_fn is not None:
                preview_update_fn(state=state, obj_id=obj_id)
        except Exception as e:
            error_holder["error"] = str(e)
            return jsonify({"error": str(e)}), 500

        return jsonify(
            {
                "ok": True,
                "mask_version": state.mask_version,
                "predicted_available": _predicted_available_map(state),
            }
        )

    @app.post("/undo")
    @app.post("/undo_point")
    def undo_point():
        payload = request.get_json(silent=True) or {}
        obj_id = int(payload.get("obj_id", _get_current_obj_id(state)))

        if obj_id not in state.object_ids:
            return jsonify({"error": f"Unknown object id {obj_id}"}), 400

        obj = state.objects[str(obj_id)]
        if obj["points_xy"]:
            obj["points_xy"].pop()
            obj["labels"].pop()

        try:
            _refresh_preview_for_object(state, preview_backend, obj_id)
            if preview_update_fn is not None:
                preview_update_fn(state=state, obj_id=obj_id)
        except Exception as e:
            error_holder["error"] = str(e)
            return jsonify({"error": str(e)}), 500

        return jsonify(
            {
                "ok": True,
                "mask_version": state.mask_version,
                "predicted_available": _predicted_available_map(state),
            }
        )

    @app.post("/clear_object")
    def clear_object():
        payload = request.get_json(silent=True) or {}
        obj_id = int(payload.get("obj_id", _get_current_obj_id(state)))

        if obj_id not in state.object_ids:
            return jsonify({"error": f"Unknown object id {obj_id}"}), 400

        state.objects[str(obj_id)] = {"points_xy": [], "labels": []}
        state.predicted_masks[str(obj_id)] = None

        try:
            _refresh_preview_for_object(state, preview_backend, obj_id)
            if preview_update_fn is not None:
                preview_update_fn(state=state, obj_id=obj_id)
        except Exception as e:
            error_holder["error"] = str(e)
            return jsonify({"error": str(e)}), 500

        return jsonify(
            {
                "ok": True,
                "mask_version": state.mask_version,
                "predicted_available": _predicted_available_map(state),
            }
        )

    @app.get("/mask/<int:obj_id>")
    def get_mask(obj_id: int):
        if obj_id not in state.object_ids:
            return jsonify({"error": f"Unknown object id {obj_id}"}), 400

        mask = state.predicted_masks.get(str(obj_id))
        if mask is None:
            return ("", 204)

        return send_file(mask_to_png_bytes(mask), mimetype="image/png")

    @app.post("/save")
    def save():
        try:
            write_prompts_json(cfg.output_json, state)
            maybe_run_video_tracking(cfg, state)
            state.saved = True
            save_event.set()
            return jsonify({"ok": True, "output_json": cfg.output_json})
        except Exception as e:
            error_holder["error"] = str(e)
            return jsonify({"error": str(e)}), 500

    return app