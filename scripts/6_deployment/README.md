# 6_deployment

Runs the trained behaviour model on new recordings, lets you check its predictions, and feeds the checked predictions back into training. The loop has three steps:

```
  ┌────────────┐     ┌────────────┐     ┌──────────────┐
  │ 1. Deploy  │ ──▶ │ 2. Review  │ ──▶ │ 3. Retrain   │
  │ label a    │     │ confirm or │     │ train on the │
  │ recording  │     │ correct    │     │ reviewed     │
  └────────────┘     └────────────┘     │ windows      │
        ▲                               └──────┬───────┘
        │         promote (you decide)         │
        └──────────────────────────────────────┘
```

1. **Deploy**: the model labels every 32-frame window of a recording and writes a per-frame label JSON. Every window prediction is also logged to the correction log.
2. **Review**: you go through the logged predictions, least confident first, and confirm or correct them.
3. **Retrain**: the reviewed windows are added to the training set as train rows, the model trains a few more epochs, and the result is compared with the current model on the untouched eval split. You then **promote** the new checkpoint (it becomes the deployed model and the base for the next cycle) or **reject** it.

Everything runs on the GPU server. The easiest way to use it is the web interface, opened in your Mac's browser through an SSH tunnel.

## Quick start

On the server, inside `tmux` so long runs survive a dropped connection:

```bash
conda activate sam3
cd ~/projects/master_thesis_social_behaviour/scripts/6_deployment
python deploy_web.py
```

On your Mac, in a separate terminal that stays open:

```bash
ssh -L 8980:localhost:8980 <user>@<gpu-server>
```

Then open **http://localhost:8980**. The three pages are linked in the header:

| Page | URL | Does |
|---|---|---|
| Deploy | `/` | run the model on a recording |
| Review | `/review` | confirm or correct predictions |
| Retrain | `/retrain` | run a retraining cycle and decide on the result |

The server only listens on `localhost`, so it can only be reached through your tunnel. Use `--port` to change the port and `--db` to use a correction log other than `~/.rat_deploy_corrections.db`.

## Input data

Each recording needs **6 camera folders**. Each camera folder contains:

```
cam_N/
  cutout_all_objects/frame_000000.png ...   RGB cutout, all animals visible
  mask_object_1/frame_000000.png ...        binary mask of rat 1
  mask_object_2/frame_000000.png ...        binary mask of rat 2
  transform_meta.csv                        optional, only used by checkpoints trained with the scale feature
```

The three subfolder names can be changed on the Deploy page (or with `--rgb-subdir`, `--mask1-subdir`, `--mask2-subdir` on the command line). The names you use are stored with every logged prediction, so review and retraining read the same folders.

**Camera order matters.** "Camera N" must be that physical camera's footage, because normalization is tied to the camera, not to the input position.

## 1. Deploy

On the **Deploy** page:

1. **Checkpoint**: the `.pt` file of the model to run. After a promote, this is filled in automatically.
2. **Camera folders**: **Fill from session folder…** takes a recording folder containing `cam_1` … `cam_6` and fills all six, plus the session label. Under each camera, a check shows whether the three subfolders exist and how many frames they hold.
3. **Options**: which rat(s) to analyze, and the window stride.
   - **32** (default) means non-overlapping windows, which matches how the model was trained.
   - A smaller stride gives smoother label boundaries but costs more compute. Only use 32 for runs you plan to review, since overlapping windows become overlapping training segments.
   - **Metadata**: the folder with the training sessions' transform metadata. Checkpoints trained with the scale feature need it to normalize that feature. Leave it empty to use the folder recorded in the checkpoint, and set it if that folder has moved since training.
4. **Output**: the path of the label JSON, the **session label** (the recording's name in the correction log), and whether to log predictions for review.
5. **Run**. Progress, time remaining and the log are shown live. **Cancel** stops the run without writing or logging anything. Closing the tab does not stop a run.

When it finishes you get a table of frames per label and a link to review the new predictions.

**Use one session label per recording.** Running the same session twice logs every window twice; the page warns you before that happens.

Without the web page, the same run from the command line:

```bash
python infer_session.py --checkpoint /path/to/model.pt --cameras /path/cam_1 /path/cam_2 /path/cam_3 /path/cam_4 /path/cam_5 /path/cam_6 --rats 1,2 --stride 32 --out /path/to/Day_11.json --session-label Day_11_rPEG007vsNovel_2
```

Add `--transform-meta-root /path/to/training_metadata` if the metadata folder recorded in the checkpoint has moved.

### Output JSON

- `rats.rat_1` / `rats.rat_2`: one entry per frame with `label`, `confidence` and the `window_start` it came from. Where windows overlap, a frame keeps the most confident window's label.
- `window_predictions`: one entry per window, which is what gets logged for review.
- `checkpoint`, `cameras`, `stride`, `session_frame_range`, `transform_meta_root`: what the run used.
- `scale_stats_ok`: `false` if the checkpoint uses the scale feature but its normalization stats could not be computed, so every frame got a neutral scale value. `null` if the checkpoint has no scale feature.

## 2. Review

The **Review** page shows the pending predictions, **least confident first**, since those are the ones most worth correcting. Filter by session or rat at the top.

For each window you see all 6 cameras with the rat being judged outlined in green. Scrub or play through the 32 frames, and click a camera to enlarge it.

| Key | Action |
|---|---|
| `Enter` | prediction is correct |
| `1` to `9`, then `S` | pick the correct label, then save the correction |
| `→` | skip (stays pending) |
| `U` | undo the last verdict |
| `Space` | play or pause |
| `,` `.` | step one frame back or forward |

A note field is optional. Undo is not possible once a row has been used in a promoted retraining cycle, or its session was excluded from retraining.

To see how many rows are pending, confirmed, corrected and exported:

```bash
python correction_log.py --stats
```

Only rows you confirmed or corrected are ever used for training. A pending prediction is never treated as ground truth, however confident it is.

## 3. Retrain

The **Retrain** page runs `retrain_cycle.py` and shows its output live.

### Before the first cycle

Fill in the inputs once:

| Field | What it is |
|---|---|
| Current checkpoint | the deployed model; training resumes from it with its original hyperparameters |
| Its eval report | that model's `classification_report_eval.txt`; the new model has to beat it |
| Base manifest, base split CSV, base label dir | the files the current `.h5` was built from |
| Data root | passed to `2_build_h5_dataset.py` as `--data-root` (normally the project root) |
| Metadata folder | the training sessions' transform metadata, for checkpoints trained with the scale feature. Leave it empty to use the folder recorded in the checkpoint; set it if that folder has moved |
| Output folder | where this cycle writes everything (suggested as `retrain_YYYYMMDD`) |
| Top-up epochs | how many epochs to train on top of the checkpoint (default 5) |

After the first promote, all inputs except the output folder and epochs come from the promoted cycle automatically. Promoting a cycle that used a metadata folder also sets it on the Deploy page.

A checkpoint trained with the scale feature is checked before anything runs: if its metadata folder doesn't exist, the cycle stops right away and asks for the Metadata folder, before it trains with a broken path. A dry run only warns.

### Running a cycle

The **Correction log** section at the top shows what the cycle will train on: sessions, rats, windows and labels.

- If a reviewed session is in the base split's **eval** set, the run is blocked, because training on it would leak eval footage and inflate the score. **Exclude from retraining** keeps that session out of all future exports (your verdicts are kept).
- **Dry run** prints every command without running anything.
- **Run retraining cycle** goes through six stages: export corrections → build `.h5` → window index → train → evaluate → compare. **Cancel** stops the whole process; nothing is marked and the state is unchanged.

Only one GPU job runs at a time: inference is blocked while a cycle trains, and the other way round.

### Deciding

When a cycle finishes, the **Result** section shows the population-weighted accuracy of the current model next to the new one. Nothing changes until you choose:

- **Deploy this checkpoint** (promote):
  - the cycle's reviewed rows are marked exported
  - the Deploy page and `gui.py` switch to the new checkpoint
  - the cycle's merged manifest, split, label folder and eval report become the base of the next cycle, so its corrections stay in every later training set
- **Reject**: the current model stays deployed, and the reviewed rows are used again in the next cycle.

A new cycle can't start until the previous one is decided. The decision can be made after a server restart, and from either the page or the command line.

### From the command line

The first cycle needs all inputs:

```bash
python retrain_cycle.py --checkpoint /path/to/current.pt --eval-report-current /path/to/classification_report_eval.txt --base-manifest /path/to/training_manifest.csv --base-split-csv /path/to/train_eval_split.csv --base-label-dir /path/to/label_clean --data-root /path/to/master_thesis_social_behaviour --retrain-out-dir training_output/retrain_20260925 --epochs 5
```

Decide:

```bash
python retrain_cycle.py --promote training_output/retrain_20260925
```

```bash
python retrain_cycle.py --reject training_output/retrain_20260925
```

If the metadata folder recorded in the checkpoint has moved, add `--transform-meta-root /path/to/training_metadata`. It is passed to `train.py`, and to `evaluate.py` if that script accepts it. Otherwise evaluation uses the folder saved in the new checkpoint.

Every cycle after a promote:

```bash
python retrain_cycle.py --retrain-out-dir training_output/retrain_20261002 --epochs 5
```

Show the current base and any cycle waiting for a decision:

```bash
python retrain_cycle.py --status
```

Passing base files that differ from the promoted ones is refused, because it would drop the promoted corrections from training. Add `--ignore-promoted-base` if that is really what you want. `--dry-run` prints the plan without running or changing anything.

### What a cycle writes

Inside the output folder:

```
corrections_export/     corrections_manifest.csv, corrections_split.csv, label_clean/,
                        merged_manifest.csv, merged_split.csv (base + corrections)
merged_label_clean/     base label CSVs + the new ones
rat_behaviour_retrain.h5, window_index.csv, conversion_report.json
training_output/best.pt the new checkpoint
eval_new/               its eval report
cycle.json              rows used, paths, comparison, decision status
```

### Exporting by hand

To build the dataset yourself instead of running a full cycle:

```bash
python export_corrections_for_retrain.py --out-dir corrections_export
```

It writes the same corrections and merged files (using the promoted base, or the `--base-*` files you pass) and **does not mark anything exported**, so the rows stay available to `retrain_cycle.py`. Add `--mark-exported` only if you train on this export by hand and keep the result; it is refused while a cycle waits for a decision.

## Files

| File | Role |
|---|---|
| `deploy_web.py` | web server: Deploy page, and hosts the Review and Retrain pages |
| `review_web.py` | Review page (can also run on its own) |
| `retrain_web.py` | Retrain page |
| `web_common.py` | shared page styles and the server-side file browser |
| `infer_session.py` | runs the model over a recording; also a command-line tool |
| `raw_frame_source.py` | reads frame folders the same way the training pipeline does |
| `correction_log.py` | the SQLite log of predictions and verdicts |
| `export_corrections_for_retrain.py` | turns reviewed rows into manifest, split and label files |
| `retrain_cycle.py` | one retraining cycle, plus `--promote`, `--reject`, `--status` |
| `gui.py`, `review_gui.py` | older Tkinter versions of Deploy and Review (need a display) |

`infer_session.py` and `retrain_cycle.py` rely on the sibling folders `../5_rat_transformer_training/` (model, `train.py`, `evaluate.py`, helpers) and `../4_building_training_data/` (`2_build_h5_dataset.py`).

### State kept between runs

All in your home folder on the server:

| File | Holds |
|---|---|
| `~/.rat_deploy_corrections.db` | every logged prediction and your verdicts |
| `~/.rat_retrain_state.json` | the promoted base and the cycle waiting for a decision |
| `~/.rat_deploy_gui_config.json` | last-used Deploy settings, including the deployed checkpoint |
| `~/.rat_retrain_web_config.json` | last-used Retrain page settings |

## Troubleshooting

**`no display name and no $DISPLAY environment variable`**
You started `gui.py` or `review_gui.py` over plain SSH. Use the web interface instead (`python deploy_web.py`), which needs no display.

**`could not compute scale normalization stats` in the Deploy log**
The metadata folder recorded in the checkpoint no longer exists, so the scale feature ran with neutral values and predictions may be worse. Set **Metadata** in Options (or `--transform-meta-root`) to where that folder lives now and run again before reviewing.

**The Deploy page shows ✗ under a camera**
A subfolder is missing or empty. Check the camera path and the subfolder names in Options.

**Export or Retrain says "camera folders not found" for an old session**
That session was logged before subfolder names were recorded, so the default names were assumed. If it used other names, record them:

```bash
python correction_log.py --set-subdirs SESSION_LABEL RGB_SUBDIR MASK1_SUBDIR MASK2_SUBDIR
```

**"still awaiting a decision"**
The previous retraining cycle hasn't been promoted or rejected. Decide it on the Retrain page or with `retrain_cycle.py --promote/--reject`.

**"differs from the base promoted from …"**
You passed older base files than the last promoted cycle's. Leave them out to use the promoted base.

**"a run is already in progress" / "a retraining cycle is running on the GPU"**
Only one GPU job runs at a time. Wait for it, or cancel it on its page.
