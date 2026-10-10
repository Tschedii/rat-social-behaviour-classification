# Rat behaviour transformer — design notes

## What the data actually is (reverse-engineered from the files)

`rat_behaviour_32f.h5` is **already a fully preprocessed, bout-level dataset**, not raw video:

- Top level: 2295 groups, one per **bout**, named `{session_id}__{rat}__{ordinal:05d}`
  (e.g. `Day_10_rPEG007vsNovel_1_24_06_2026__rat_1__00000`). `ordinal` is the bout's row
  position within `train_eval_split.csv` for that `(session_id, rat)` pair.
- Each bout group has `cam_1`..`cam_6` subgroups. Each camera has `rgb`, `mask`, `frame_id`,
  all length **32** — every bout, regardless of its original duration (1 frame to 3538
  frames), has been temporally resampled to exactly **32 steps** (confirmed: a 275-frame
  bout gets `frame_id = [279, 288, 297, ...]`, evenly spaced; a 13-frame bout gets
  `[0,0,1,1,2,2,...]`, nearest-neighbor upsampled).
- `rgb`/`mask` entries are raw **PNG-encoded bytes** (1024×1024, RGB / single-channel binary
  0-255 mask) — already **cropped and scaled to the target rat** using the bounding boxes in
  `training_metadata/<session>/cam_N/transform_meta.csv`. So per-rat cropping is already done;
  we do not need to re-crop.
- `classes` dataset gives the canonical 10-class label order.

This has two big consequences for training:

1. **The training window is fixed at 32 frames** (~1.07 s at 30 fps) by construction of the
   `.h5` file itself — not a choice we make in the dataloader. Short and long behaviors are
   all warped onto the same 32-step temporal grid. The tradeoff: absolute duration is
   destroyed by the resampling, so we re-inject it as an auxiliary scalar
   (`log(duration_frames)`, standardized) concatenated into the classifier head, since
   duration is a real discriminative cue (e.g. Immobile/Grooming bouts run much longer than
   Approaching/Investigation).
2. Six synchronized cameras exist per bout, but the target rat is not always visible/
   unoccluded in all of them. Rather than trust the `present_1/present_2` flags in
   `transform_meta.csv` (indexed by raw frame number, not the resampled `frame_id` grid, so
   joining them back is fiddly), `helpers/build_index.py` computes **mask coverage directly
   from the decoded mask PNGs already in the `.h5`** — cheap, always correct by construction,
   and used both to (a) pick the best camera(s) per bout and (b) weight camera fusion at
   train time.

## Label join

`train_eval_split.csv` doesn't reference the `.h5` file directly. The join key is derived:
sort each `(session_id, rat)`'s rows in file order, treat row position as `ordinal`, and look
up `{session_id}__{rat}__{ordinal:05d}` in the `.h5`. `helpers/build_index.py` does this,
**reports the match rate**, and drops anything it can't match rather than guessing — check its
printed summary before trusting the resulting manifest. Session-level train/eval split was
verified to have zero leakage (no session appears in both splits).

## Model: CNN + Transformer hybrid (yes — and it's the right call here)

With ~80 minutes of labeled video total across 10 classes, training a ViT-style model from
raw pixels is data-starved. A hybrid is more sample-efficient and is what's implemented:

```
per-frame RGB+mask (4×1024×1024, native — never resized)
        │  shared, ImageNet-pretrained CNN backbone (resnet18 by default)
        ▼
per-frame, per-camera embedding (D=512)
        │  + learned camera-id embedding
        │  cross-camera self-attention, attention-pooled over cameras
        ▼               (handles variable / occluded cameras per frame)
per-frame fused embedding (D)
        │  + learned positional embedding (32 positions) + CLS token
        │  Transformer encoder (temporal self-attention across the 32 steps)
        ▼
bout-level embedding (CLS output)
        │  concat with standardized log(duration_frames)
        ▼
Linear classifier → 10 classes
```

- **CNN** does the heavy lifting of learning "what does this rat's body/pose/fur pattern
  look like in this crop" — this is what pretrained conv nets are good at with limited data
  (transfer learning from ImageNet textures/edges).
- **Transformer (temporal)** captures how that appearance evolves over the 32-frame window —
  attention lets it learn things like "posture holds nearly static → Immobile" vs. "cyclic
  limb motion → Locomotion" without hand-crafted motion features.
- **Transformer (cross-camera)** replaces a fixed "pick the best camera" heuristic with a
  learned, occlusion-aware fusion — the coverage-based visibility weighting is only used to
  reduce I/O (pick a subset of cameras to decode/train on, e.g. top-3 of 6), not to hard-code
  which camera matters.
- The mask is fed in as a 4th input channel (RGBM) rather than used to hard-mask pixels, so
  the network can still see the second rat / context for social behaviors (Approaching,
  Body touch) while having an explicit signal for "this is the labeled rat".

This is *not* a from-scratch video transformer (e.g. TimeSformer/ViViT) — those need far more
data than 2.3k clips to train attention over raw patches. Reusing a pretrained CNN for spatial
features and only training attention over the *much shorter* sequence (32 steps × ≤6 cameras)
is the standard way to make transformers work in a low-data video regime. Only the CNN's input
resolution is at stake in the discussion below — the transformer stages always operate on
fixed-size `(d_model,)` embeddings, so they're completely unaffected by input resolution.

### Training at native resolution (no downsampling), and why that's more than a config flag

By requirement, the CNN sees the `.h5`'s images exactly as stored — `--frame-size 1024`,
which the dataset code treats as "skip resizing entirely" rather than "resize to 1024".
That has two real consequences, both handled in `model.py`/`train.py`, not just documented:

1. **GPU memory.** A resnet18 forward/backward over a batch of frames at 1024×1024 uses
   roughly the pixel-count ratio (~21x vs. a typical 224×224 setup) in activation memory —
   enough to exceed most single-GPU budgets if done naively. Fix: `CNNTransformer` splits
   the flattened `(batch × num_cams × 32)` frames into small `--micro-batch-size` chunks and
   runs each through `torch.utils.checkpoint` — activations for a chunk are discarded after
   its forward pass and recomputed on demand during backward, so peak memory is bounded by
   one chunk, not the whole batch. This changes *when* things are computed, not the math.
   Separately, the *physical* `--batch-size` still has to stay small (default 1), because the
   raw input tensor itself — `batch × num_cams × 32 × 4 × 1024 × 1024` floats, before it ever
   reaches the model — is already multiple GB at batch-size 1. `--grad-accum-steps` (default
   16) recovers a larger *effective* batch size by accumulating gradients over several small
   physical batches before each optimizer step.

2. **BatchNorm becomes wrong, not just slow.** This was caught by an actual correctness
   test, not by inspection: chunking the backbone forward *should* be a pure no-op
   mathematically (same frames in, same result out, just computed in pieces) — but with the
   original ResNet's BatchNorm layers, chunked output did not match unchunked output. Reason:
   BatchNorm normalizes using statistics computed from whatever's in the current
   batch/chunk, and at batch-size 1 (forced by point 1) every training step's "batch" is just
   the 32×(≤6 cam) frames of a *single* bout — a small, highly correlated sample, and chunking
   it further only makes those per-chunk statistics noisier. Fix: `model.py`'s
   `convert_bn_to_gn()` replaces every `BatchNorm2d` in the backbone with `GroupNorm`
   (per-sample, channel-group statistics — no batch-dependence at all), which is now the
   default (`--backbone-norm groupnorm`). Verified empirically after the fix: chunked and
   unchunked forward/backward now agree to float32 rounding precision, and gradients are
   identical whether checkpointing is on or off.

Practical fallout: expect this to be noticeably slower per step than a 224px setup — verified
on this machine's Apple Silicon (MPS backend, not a real GPU) at 1024×1024, batch=1, 2
cameras: ~94s/training-step, ~24s/eval-step. A real CUDA GPU (Colab T4/A100) will be
substantially faster, but still much slower than 224px would be — budget accordingly, and
tune `--micro-batch-size` down first if you hit CUDA OOM (it only trades speed for memory,
never changes results), then `--batch-size`/`--num-cams` if that's not enough.

## Files

- `helpers/build_index.py` — joins the `.h5` to `train_eval_split.csv`, computes per-camera
  mask-visibility, writes `window_index.csv` + `label_map.json`. **Run this first.**
- `helpers/rat_h5_dataset.py` — `torch.utils.data.Dataset` reading directly from the `.h5`.
- `model.py` — `CNNTransformer` (backbone + camera fusion + temporal transformer + head).
- `train.py` — training loop (class-weighted loss, AMP, backbone warmup/unfreeze, checkpointing).
- `evaluate.py` — standalone evaluation / confusion matrix / per-class report from a checkpoint.

## Running it

```bash
python helpers/build_index.py \
  --h5 /path/to/rat_behaviour_32f.h5 \
  --split-csv /path/to/train_eval_split.csv \
  --out-dir /path/to/rat_transformer_training

python train.py \
  --h5 /path/to/rat_behaviour_32f.h5 \
  --index-csv /path/to/rat_transformer_training/window_index.csv \
  --normstats /path/to/rat_transformer_training/rgb_normstats.json \
  --out-dir /path/to/rat_transformer_training/checkpoints
  # defaults already assume native 1024px resolution, batch-size=1, grad-accum=16,
  # micro-batch-size=4, groupnorm backbone — tune micro-batch-size/batch-size/num-cams
  # down first if you hit GPU OOM (see "Training at native resolution" above)
```

## Running on a shared multi-GPU server

`run_on_server.sh` launches training detached from the SSH session (`nohup ... &`), so it
survives disconnects. Before running it:

- `nvidia-smi` to see which of the server's two GPUs (RTX 4500 Ada / RTX A4500) is actually
  free — it's a shared box (`<user>@<gpu-server>`) — and set `GPU_ID` in the script accordingly, or pass
  `--gpu-id` directly to `train.py`. `CUDA_VISIBLE_DEVICES=<id>` before the command works too.
- Copy this whole folder to the server path referenced in the script (`window_index.csv`,
  `label_map.json`, `rgb_normstats.json` are all already in here and small — no need to rebuild
  the index unless the underlying `.h5`/label data changed).

Everything lands under one `--out-dir` (`training_output/` by convention here):
- `train.log` — clean, timestamped, one line per epoch/event (safe to `tail -f`; tqdm's
  progress bars go to `stdout.log` instead so the log file doesn't fill with bar-redraw spam).
- `metrics.csv` — per-epoch loss/accuracy/F1, for plotting later.
- `confusion_matrices/epoch_NNN.png` — saved every `--confusion-matrix-every` epochs (default
  2) plus always on the final epoch, computed from predictions already gathered for that
  epoch's eval metrics (no extra forward pass).
- `last.pt` / `best.pt` — checkpoints (latest, and best eval macro-F1 so far).

With 20-24GB-class GPUs (RTX 4500 Ada / A4500) and 192 CPU cores available, `run_on_server.sh`
raises the Colab defaults somewhat: `--batch-size 2`, `--micro-batch-size 8`,
`--num-workers 16`. These are still starting points, not measured optima for this specific
hardware (I don't have direct access to run on it) — watch `nvidia-smi` and `free -h` during
the first epoch and adjust `--micro-batch-size` first if you hit CUDA OOM (memory only, never
changes results), then `--batch-size`/`--num-cams` if that's not enough headroom.

## Known caveats / next steps worth knowing about

- The bout→ordinal join is a reconstruction, not something read from an explicit manifest —
  `build_index.py` prints its match rate; if it's not close to 100%, inspect
  `unmatched_report.csv` it writes before trusting results.
- Training at native 1024×1024 resolution is I/O- and compute-heavy per the section above;
  the tradeoffs (micro-batching, gradient checkpointing, GroupNorm, small physical batch +
  gradient accumulation) are all in service of not losing any pixel detail, by requirement.
  If training time ever becomes prohibitive, the fallback most worth considering is a
  precomputed cache re-encoded at a smaller resolution — not built here since it directly
  trades away the detail this design was built to preserve.
- Class imbalance (Locomotion/Grooming/Rearing ≈3-4x rarer than Exploring/Air sniffing) is
  handled with inverse-sqrt-frequency class weights in the loss; consider also trying focal
  loss if macro-F1 on rare classes lags after the baseline run.
