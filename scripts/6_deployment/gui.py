"""Small Tkinter GUI for deployment: pick a checkpoint + 6 raw camera folders, run
inference, save a per-frame label JSON.

Run on the GPU server (needs a display — X11 forwarding from your Mac via `ssh -X`
with XQuartz installed, or a VNC session):

    python3 gui.py

Tkinter ships with Python's standard library, but a headless server install sometimes
omits the OS-level Tk package. If `import tkinter` fails, install it (Debian/Ubuntu:
`sudo apt install python3-tk`; conda: `conda install tk`) and retry.

Each of the 6 camera folders is expected to contain (see infer_session.py's docstring
for the exact layout, and 6_deployment/README-ish note in that file's header — subfolder
names are editable in the Settings section below if your raw layout uses different
names):
    cutout_all_objects/frame_NNNNNN.png
    mask_object_1/frame_NNNNNN.png
    mask_object_2/frame_NNNNNN.png
    transform_meta.csv   (optional)

Settings (checkpoint path, subfolder names, last-used camera folders) persist to
~/.rat_deploy_gui_config.json between runs so you don't have to re-browse every time.
"""
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

CONFIG_PATH = Path.home() / ".rat_deploy_gui_config.json"
DEFAULT_SUBDIRS = {"rgb": "cutout_all_objects", "mask1": "mask_object_1", "mask2": "mask_object_2"}


def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


class DeployGUI:
    def __init__(self, root):
        self.root = root
        root.title("Rat Behaviour — Deploy")
        self.cfg = load_config()
        self.msg_queue = queue.Queue()
        self.worker = None

        pad = {"padx": 6, "pady": 4}

        # ---- checkpoint ----
        ckpt_frame = ttk.LabelFrame(root, text="Model")
        ckpt_frame.pack(fill="x", **pad)
        self.ckpt_var = tk.StringVar(value=self.cfg.get("checkpoint", ""))
        ttk.Entry(ckpt_frame, textvariable=self.ckpt_var, width=70).pack(side="left", padx=4, pady=4, fill="x", expand=True)
        ttk.Button(ckpt_frame, text="Browse…", command=self._browse_checkpoint).pack(side="left", padx=4)

        # ---- 6 camera paths ----
        cams_frame = ttk.LabelFrame(root, text="Camera folders (raw, native resolution)")
        cams_frame.pack(fill="x", **pad)
        ttk.Label(
            cams_frame,
            text="Order matters: \"Camera N\" must be that physical camera's footage — "
                 "per-camera normalization is keyed by camera identity, not by input order.",
            foreground="#8A5A00",
        ).pack(anchor="w", padx=4, pady=(4, 0))
        self.cam_vars = []
        saved_cams = self.cfg.get("cameras", [""] * 6)
        for i in range(6):
            row = ttk.Frame(cams_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=f"Camera {i + 1}", width=10).pack(side="left")
            var = tk.StringVar(value=saved_cams[i] if i < len(saved_cams) else "")
            self.cam_vars.append(var)
            ttk.Entry(row, textvariable=var, width=60).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Browse…", command=lambda v=var: self._browse_folder(v)).pack(side="left", padx=4)

        # ---- options ----
        opts_frame = ttk.LabelFrame(root, text="Options")
        opts_frame.pack(fill="x", **pad)

        ttk.Label(opts_frame, text="Analyze:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.rat_choice = tk.StringVar(value=self.cfg.get("rat_choice", "both"))
        for i, (label, val) in enumerate([("Rat 1 only", "1"), ("Rat 2 only", "2"), ("Both", "both")]):
            ttk.Radiobutton(opts_frame, text=label, variable=self.rat_choice, value=val).grid(
                row=0, column=1 + i, sticky="w", padx=4
            )

        ttk.Label(opts_frame, text="Window stride (frames):").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.stride_var = tk.StringVar(value=str(self.cfg.get("stride", 32)))
        ttk.Entry(opts_frame, textvariable=self.stride_var, width=8).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(opts_frame, text="(32 = non-overlapping, matches training; smaller = smoother "
                                    "frame boundaries, more compute)").grid(row=1, column=2, columnspan=3, sticky="w")

        ttk.Label(opts_frame, text="Metadata folder:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        meta_row = ttk.Frame(opts_frame)
        meta_row.grid(row=2, column=1, columnspan=4, sticky="we", padx=4)
        self.meta_var = tk.StringVar(value=self.cfg.get("transform_meta_root", "") or "")
        ttk.Entry(meta_row, textvariable=self.meta_var, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(meta_row, text="Browse…", command=self._browse_meta).pack(side="left", padx=4)
        ttk.Label(opts_frame, text="(training sessions' transform metadata, for the scale feature; empty = the "
                                    "folder recorded in the checkpoint. Set it if that folder has moved)"
                  ).grid(row=3, column=1, columnspan=4, sticky="w", padx=4)
        opts_frame.columnconfigure(4, weight=1)

        # ---- subfolder name settings (collapsible-ish, just shown plainly) ----
        settings_frame = ttk.LabelFrame(root, text="Raw subfolder names (edit if your layout differs)")
        settings_frame.pack(fill="x", **pad)
        saved_subdirs = self.cfg.get("subdirs", DEFAULT_SUBDIRS)
        self.subdir_vars = {}
        for i, (key, label) in enumerate([("rgb", "All-objects cutout"), ("mask1", "Mask object 1"), ("mask2", "Mask object 2")]):
            ttk.Label(settings_frame, text=label + ":").grid(row=0, column=2 * i, sticky="w", padx=4, pady=4)
            var = tk.StringVar(value=saved_subdirs.get(key, DEFAULT_SUBDIRS[key]))
            self.subdir_vars[key] = var
            ttk.Entry(settings_frame, textvariable=var, width=20).grid(row=0, column=2 * i + 1, padx=4)

        # ---- output ----
        out_frame = ttk.LabelFrame(root, text="Output")
        out_frame.pack(fill="x", **pad)
        out_row1 = ttk.Frame(out_frame)
        out_row1.pack(fill="x")
        self.out_var = tk.StringVar(value=self.cfg.get("out_path", str(Path.cwd() / "deploy_output.json")))
        ttk.Entry(out_row1, textvariable=self.out_var, width=70).pack(side="left", padx=4, pady=4, fill="x", expand=True)
        ttk.Button(out_row1, text="Browse…", command=self._browse_output).pack(side="left", padx=4)
        out_row2 = ttk.Frame(out_frame)
        out_row2.pack(fill="x")
        ttk.Label(out_row2, text="Session label (for the correction log):").pack(side="left", padx=4, pady=(0, 4))
        self.session_label_var = tk.StringVar(value=self.cfg.get("session_label", ""))
        ttk.Entry(out_row2, textvariable=self.session_label_var, width=40).pack(side="left", padx=4)
        self.log_predictions_var = tk.BooleanVar(value=self.cfg.get("log_predictions", True))
        ttk.Checkbutton(out_row2, text="Log predictions for later review/correction",
                         variable=self.log_predictions_var).pack(side="left", padx=12)

        # ---- run controls ----
        run_frame = ttk.Frame(root)
        run_frame.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_frame, text="Run", command=self._start_run)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(run_frame, text="Review corrections…", command=self._launch_review).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(run_frame, mode="determinate", length=300)
        self.progress.pack(side="left", padx=8, fill="x", expand=True)
        self.progress_label = ttk.Label(run_frame, text="")
        self.progress_label.pack(side="left", padx=4)

        # ---- log ----
        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, height=14, width=100, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._poll_queue)

    # ---- helpers ----

    def _log(self, msg):
        self.msg_queue.put(("log", msg))

    def _browse_checkpoint(self):
        p = filedialog.askopenfilename(title="Select checkpoint (.pt)", filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")])
        if p:
            self.ckpt_var.set(p)

    def _browse_folder(self, var):
        p = filedialog.askdirectory(title="Select camera folder")
        if p:
            var.set(p)

    def _browse_meta(self):
        p = filedialog.askdirectory(title="Select the training transform metadata folder")
        if p:
            self.meta_var.set(p)

    def _browse_output(self):
        p = filedialog.asksaveasfilename(title="Save output JSON as", defaultextension=".json",
                                          filetypes=[("JSON", "*.json")])
        if p:
            self.out_var.set(p)

    def _current_config(self):
        return {
            "checkpoint": self.ckpt_var.get(),
            "cameras": [v.get() for v in self.cam_vars],
            "rat_choice": self.rat_choice.get(),
            "stride": self.stride_var.get(),
            "subdirs": {k: v.get() for k, v in self.subdir_vars.items()},
            "out_path": self.out_var.get(),
            "session_label": self.session_label_var.get(),
            "log_predictions": self.log_predictions_var.get(),
            "transform_meta_root": self.meta_var.get().strip(),
        }

    def _launch_review(self):
        import subprocess
        import sys
        script = str(Path(__file__).resolve().parent / "review_gui.py")
        checkpoint = self.ckpt_var.get().strip()
        cmd = [sys.executable, script]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        subprocess.Popen(cmd)

    def _on_close(self):
        save_config(self._current_config())
        self.root.destroy()

    # ---- run ----

    def _start_run(self):
        checkpoint = self.ckpt_var.get().strip()
        cameras = [v.get().strip() for v in self.cam_vars]
        out_path = self.out_var.get().strip()

        if not checkpoint:
            messagebox.showerror("Missing input", "Select a checkpoint file.")
            return
        if any(not c for c in cameras):
            messagebox.showerror("Missing input", "All 6 camera folders must be set.")
            return
        try:
            stride = int(self.stride_var.get())
            if stride <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Stride must be a positive integer.")
            return
        if not out_path:
            messagebox.showerror("Missing input", "Choose an output JSON path.")
            return
        meta_root = self.meta_var.get().strip()
        if meta_root and not Path(meta_root).expanduser().is_dir():
            messagebox.showerror("Invalid input", f"Metadata folder not found: {meta_root}")
            return

        rat_choice = self.rat_choice.get()
        rats = (1, 2) if rat_choice == "both" else (int(rat_choice),)

        save_config(self._current_config())

        self.run_btn.config(state="disabled")
        self.progress["value"] = 0
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        subdirs = {
            "rgb": self.subdir_vars["rgb"].get(),
            1: self.subdir_vars["mask1"].get(),
            2: self.subdir_vars["mask2"].get(),
        }

        session_label = self.session_label_var.get().strip() or Path(out_path).stem
        do_log = self.log_predictions_var.get()

        self.worker = threading.Thread(
            target=self._run_worker,
            args=(checkpoint, cameras, rats, stride, subdirs, out_path, session_label, do_log,
                  meta_root or None),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, checkpoint, cameras, rats, stride, subdirs, out_path, session_label, do_log,
                    meta_root=None):
        try:
            from infer_session import SessionInferer

            inferer = SessionInferer(checkpoint, subdirs=subdirs, progress_log=self._log,
                                     transform_meta_root=meta_root)

            def progress_cb(done, total):
                self.msg_queue.put(("progress", (done, total)))

            result = inferer.run(cameras, rats=rats, stride=stride, progress_cb=progress_cb)
            Path(out_path).write_text(json.dumps(result, indent=2))

            summary_lines = [f"\nWrote {out_path}"]
            for rat_key, frames in result["rats"].items():
                counts = {}
                for f in frames:
                    counts[f["label"]] = counts.get(f["label"], 0) + 1
                summary_lines.append(f"{rat_key}: {len(frames)} frames labeled — " +
                                      ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))

            if do_log:
                from correction_log import log_window_predictions
                for rat in rats:
                    n = log_window_predictions(
                        None, session_label, cameras, checkpoint, rat,
                        result["window_predictions"][f"rat_{rat}"], subdirs=subdirs,
                    )
                    summary_lines.append(f"logged {n} rat_{rat} window predictions to the correction log "
                                          f"(session '{session_label}') — use \"Review corrections…\" to triage them")

            self._log("\n".join(summary_lines))
            self.msg_queue.put(("done", result.get("scale_stats_ok")))
        except Exception as e:
            self._log(f"\nERROR: {e}")
            self.msg_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log_text.config(state="normal")
                    self.log_text.insert("end", payload + "\n")
                    self.log_text.see("end")
                    self.log_text.config(state="disabled")
                elif kind == "progress":
                    done, total = payload
                    self.progress["maximum"] = total
                    self.progress["value"] = done
                    self.progress_label.config(text=f"{done}/{total}")
                elif kind == "done":
                    self.run_btn.config(state="normal")
                    if payload is False:  # scale_stats_ok: the scale feature ran with neutral values
                        messagebox.showwarning(
                            "Done, with a warning",
                            "Inference complete, but the scale feature ran with neutral values because its "
                            "normalization stats could not be computed (see the log). Set the Metadata "
                            "folder in Options and run again before reviewing these predictions.")
                    else:
                        messagebox.showinfo("Done", "Inference complete.")
                elif kind == "error":
                    self.run_btn.config(state="normal")
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)


def main():
    root = tk.Tk()
    DeployGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
