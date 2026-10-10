"""Review queue for 6_deployment/correction_log.py: shows the model's least-confident predictions
first (active-learning triage — correcting the model's most uncertain calls is far higher
value per minute than reviewing everything uniformly), lets you confirm or correct each one,
and writes the verdict back to the correction log for later export into a retraining cycle.

    python3 review_gui.py [--db PATH] [--limit 50]

For each queued prediction, shows the window's middle frame (RGB cutout + the target rat's
mask, from camera 1 only — a sanity-check view, not meant to replace looking at the original
footage for a genuinely ambiguous case) alongside the predicted label and confidence.
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correction_log import DEFAULT_DB_PATH, confirm_correct, get_review_queue, row_subdirs, set_correction
from raw_frame_source import FrameDir

_label_map_cache = {}


def labels_for_checkpoint(checkpoint_path):
    if checkpoint_path not in _label_map_cache:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        label_map = ckpt["label_map"]
        _label_map_cache[checkpoint_path] = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return _label_map_cache[checkpoint_path]


class ReviewGUI:
    def __init__(self, root, db_path, limit, subdirs):
        self.root = root
        self.db_path = db_path
        self.limit = limit
        self.subdirs = subdirs
        root.title("Rat Behaviour — Review Corrections")

        self.queue = []
        self.pos = 0
        self._photo_refs = []  # keep PhotoImage refs alive

        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        self.status_label = ttk.Label(top, text="", font=("", 11, "bold"))
        self.status_label.pack(side="left")
        ttk.Button(top, text="Refresh queue", command=self.reload_queue).pack(side="right")

        info_frame = ttk.Frame(root)
        info_frame.pack(fill="x", padx=8, pady=4)
        self.info_label = ttk.Label(info_frame, text="", justify="left")
        self.info_label.pack(anchor="w")

        img_frame = ttk.Frame(root)
        img_frame.pack(padx=8, pady=8)
        self.rgb_canvas = tk.Label(img_frame)
        self.rgb_canvas.pack(side="left", padx=4)
        self.mask_canvas = tk.Label(img_frame)
        self.mask_canvas.pack(side="left", padx=4)

        action_frame = ttk.LabelFrame(root, text="Verdict")
        action_frame.pack(fill="x", padx=8, pady=6)
        ttk.Button(action_frame, text="✓ Prediction is correct", command=self.confirm_current).pack(
            side="left", padx=6, pady=6
        )
        ttk.Label(action_frame, text="Correct label:").pack(side="left", padx=(20, 4))
        self.correct_var = tk.StringVar()
        self.correct_combo = ttk.Combobox(action_frame, textvariable=self.correct_var, state="readonly", width=25)
        self.correct_combo.pack(side="left", padx=4)
        ttk.Button(action_frame, text="Save correction", command=self.save_correction).pack(side="left", padx=6)
        ttk.Button(action_frame, text="Skip →", command=self.skip_current).pack(side="left", padx=(20, 6))

        self.reload_queue()

    def reload_queue(self):
        self.queue = get_review_queue(self.db_path, limit=self.limit, status="pending")
        self.pos = 0
        self.show_current()

    def show_current(self):
        self._photo_refs.clear()
        if self.pos >= len(self.queue):
            self.status_label.config(text="Queue empty — nothing pending review.")
            self.info_label.config(text="Run \"Refresh queue\" after logging more predictions, "
                                         "or run a deployment session to generate more.")
            self.rgb_canvas.config(image="")
            self.mask_canvas.config(image="")
            self.correct_combo["values"] = []
            return

        row = self.queue[self.pos]
        self.status_label.config(text=f"Reviewing {self.pos + 1} / {len(self.queue)} "
                                       f"(least confident first)")
        self.info_label.config(text=(
            f"session: {row['session_label']}   rat: {row['rat']}   "
            f"window_start: {row['window_start']}   "
            f"predicted: {row['predicted_label']}   confidence: {row['confidence']:.3f}\n"
            f"checkpoint: {row['checkpoint']}"
        ))

        try:
            labels = labels_for_checkpoint(row["checkpoint"])
        except Exception as e:
            labels = []
            messagebox.showwarning("Could not load label list", f"{e}\nType the correction manually not available "
                                                                  f"— falling back to an empty list.")
        self.correct_combo["values"] = labels
        if row["predicted_label"] in labels:
            self.correct_var.set(row["predicted_label"])
        elif labels:
            self.correct_var.set(labels[0])

        self._load_images(row)

    def _load_images(self, row):
        import json
        from PIL import Image, ImageTk

        try:
            camera_dirs = json.loads(row["camera_dirs"])
            cam1_dir = Path(camera_dirs[0])
            mid_frame = row["window_start"] + row["seq_len"] // 2

            sub = row_subdirs(row, fallback=self.subdirs)  # --*-subdir only for rows logged without names
            rgb_fd = FrameDir(cam1_dir / sub["rgb"])
            mask_fd = FrameDir(cam1_dir / sub[f"mask{row['rat']}"])

            rgb_arr = rgb_fd.read(mid_frame if rgb_fd.has(mid_frame) else min(rgb_fd._index), color=True)
            mask_arr = mask_fd.read(mid_frame if mask_fd.has(mid_frame) else min(mask_fd._index), color=False)

            rgb_im = Image.fromarray(rgb_arr).resize((320, 320))
            mask_im = Image.fromarray(mask_arr).resize((320, 320))

            rgb_photo = ImageTk.PhotoImage(rgb_im)
            mask_photo = ImageTk.PhotoImage(mask_im)
            self._photo_refs.extend([rgb_photo, mask_photo])
            self.rgb_canvas.config(image=rgb_photo)
            self.mask_canvas.config(image=mask_photo)
        except Exception as e:
            self.rgb_canvas.config(image="", text=f"(couldn't load frame: {e})")
            self.mask_canvas.config(image="", text="")

    def confirm_current(self):
        row = self.queue[self.pos]
        confirm_correct(self.db_path, row["id"])
        self._advance()

    def save_correction(self):
        row = self.queue[self.pos]
        label = self.correct_var.get()
        if not label:
            messagebox.showerror("No label", "Choose a correct label first.")
            return
        set_correction(self.db_path, row["id"], label)
        self._advance()

    def skip_current(self):
        self._advance()

    def _advance(self):
        self.pos += 1
        self.show_current()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--checkpoint", default=None, help="unused directly — kept for symmetry with "
                                                         "gui.py's launcher; label lists are read "
                                                         "per-row from each row's own checkpoint")
    ap.add_argument("--rgb-subdir", default="cutout_all_objects")
    ap.add_argument("--mask1-subdir", default="mask_object_1")
    ap.add_argument("--mask2-subdir", default="mask_object_2")
    args = ap.parse_args()

    subdirs = {"rgb": args.rgb_subdir, 1: args.mask1_subdir, 2: args.mask2_subdir}
    root = tk.Tk()
    ReviewGUI(root, args.db, args.limit, subdirs)
    root.mainloop()


if __name__ == "__main__":
    main()
