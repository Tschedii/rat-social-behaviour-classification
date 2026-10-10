#!/usr/bin/env bash
# Launch (or resume) training on the lab server, detached from the SSH session (survives
# disconnects). Safe to re-run any time: it auto-detects an existing checkpoint and resumes
# from it rather than starting over, so if training is stopped (kill, crash, or you want to
# change a speed setting like --micro-batch-size) just re-running this script picks up where
# it left off.
#
# Usage:
#   1. Check GPU 0 (RTX 4500 Ada, 24GB) is free before launching — it's a shared server:
#        nvidia-smi
#   2. Adjust paths/GPU_ID below if your layout differs from what's assumed here.
#   3. Run:  bash run_on_server.sh
#   4. Monitor from another shell:  tail -f training_output/train.log
#      or watch progress numerically:  column -s, -t training_output/metrics.csv | less -S
#   5. Confusion matrices land in training_output/confusion_matrices/epoch_XXX.png as training
#      proceeds (every 2nd epoch by default, plus the final one) — pull them to view, e.g.:
#        scp <user>@<gpu-server>:.../training_output/confusion_matrices/epoch_003.png .
#   6. To stop: find the main process (not the --num-workers subprocesses) via
#        nvidia-smi   # bottom "Processes" table gives the real PID
#      then `kill <PID>`. Checkpoints only save at epoch boundaries, so this is always safe —
#      re-running this script afterwards resumes from the last completed epoch.
#
# --epochs is 330: sized against the actual thesis hand-in date, 2026-10-04 (38 days from
# 2026-08-27) — Sept 13 was only a self-imposed "stop actively training" checkpoint, not the
# real deadline. With a 10-day writeup/analysis buffer before Oct 4, that's ~28 usable days.
# 6-camera epochs are ESTIMATED at ~2x the measured 3-camera time (~1.55h -> ~3.1h — chunking
# keeps peak memory flat regardless of camera count, but total frames per bout doubles) — that
# estimate is UNVERIFIED until the first few 6-camera epochs actually run. Check real per-epoch
# time after ~3 epochs and recompute this number if it's meaningfully off
# (epochs_affordable = buffered_days_left * 24 / real_epoch_hours). 330 is generous on purpose —
# it's cheap to stop early once results plateau; it's not cheap to run out of schedule room.
#
# As of epoch 115 (eval macro-F1 0.342, 9-class-equivalent once Immobile is dropped below),
# this resume adds four more warm-startable changes on top of the attention-pooling/
# weight-floor/flow resume from before:
#   --num-cams 6                  all six cameras instead of three. Only camera_fusion's
#                                  cam_embed (an nn.Embedding(num_cams, d_model)) depends on
#                                  camera count — it restarts fresh; the backbone, flow branch,
#                                  camera-fusion transformer layer, temporal transformer, and
#                                  head all warm-start untouched. WARNING: epoch time roughly
#                                  DOUBLES (more frames per bout to run through the backbone),
#                                  ~1.55h -> ~3.1h/epoch — size --epochs to the days you can
#                                  actually spend: epochs_affordable ~= days_left * 24 / 3.1.
#   --exclude-classes Immobile    drops Immobile entirely (train+eval), per-choice: it's stayed
#                                  the weakest class (F1 0.24 at epoch 115) despite attention
#                                  pooling helping a little. Every other class's classifier row
#                                  warm-starts by NAME (not position) — see train.py's resume
#                                  logic — so dropping one class doesn't perturb the others'
#                                  learned weights at all.
#   --class-weight-multiplier     manual boost/ease-off on top of the inverse-sqrt+floor
#                                  weight, based on the epoch-115 per-class picture: Tail
#                                  investigation (F1 0.13, weakest remaining) and Exploring/Air
#                                  sniffing (0.21/0.23) get boosted; Rearing (F1 0.70, far
#                                  ahead of everything else) eases off to free up gradient
#                                  budget. Adjust the values below as new results come in.
#   --use-scale-feature           feeds each frame's registration zoom factor (scale =
#                                  1024/max(bbox_w,bbox_h), from transform_meta.csv) into camera
#                                  fusion. The upstream tracker crops each frame to the tracked
#                                  bounding box and rescales it to a fixed 1024px output — that
#                                  keeps shape/aspect ratio intact but destroys absolute apparent
#                                  size, so without this the model has no way to know how
#                                  zoomed-in a given frame was. camera_fusion.scale_proj is a
#                                  tiny Linear(1,d_model) that's now unconditionally part of the
#                                  model (present even when this flag is off, just unused) — one
#                                  more warm-start gap on this resume, nothing after it.
# Metrics.csv will show a real discontinuity at this resume: macro-F1/accuracy are no longer
# comparable to earlier rows one-for-one, since they're now computed over 9 classes instead of
# 10 and Immobile's near-zero contribution is gone from the denominator.
#
# --weight-decay 0.1 (was 0.05) and --label-smoothing 0.1 (was 0.05): added at epoch 142, once
# overfitting became the clearer signal than "which class needs more/less weight." Evidence: the
# 5-epoch-window mean eval macro-F1 went 0.408 -> 0.391 -> 0.438 -> 0.463 -> 0.456 (the LAST
# window is the first one to drop, not just noise) while train macro-F1 kept climbing hard
# (0.594 -> 0.638) — the train/eval gap jumped from 0.132 to 0.182 in one window, the sharpest
# single-window increase of the whole run, with 8 epochs since the last new-best checkpoint
# (epoch 134). Unlike every earlier change here, this reins in capacity rather than adding it —
# no architecture change, so this is an EXACT-MATCH resume (no missing/unexpected keys) and,
# unlike every prior resume, Adam's momentum carries over instead of resetting — no disruptive
# post-resume dip expected this time.
#
# Round 2, at epoch 150: the 0.1/0.1 bump WASN'T enough — the gap kept widening, not narrowing
# (0.182 -> 0.190 -> 0.209, the widest of the whole run), while eval macro-F1 stayed flat in a
# 0.44-0.49 band for 15+ epochs. Pushing harder rather than incrementally, and adding a second
# lever that was never tunable before:
#   --weight-decay 0.1 -> 0.2
#   --dropout 0.1 -> 0.2      NEW flag — dropout inside camera_fusion/temporal was always
#                             hardcoded to 0.1 and never exposed. Doesn't change any parameter
#                             shape, so still an exact-match resume, Adam momentum still carries
#                             over.
# --class-weight-multiplier: REVISED against a fresh evaluate.py report on epoch 149's best.pt
# (macro-F1 0.494) — the epoch-134 picture above turned out to be stale in three ways the
# 15-epoch plateau didn't predict: Approaching/Following self-corrected (49.4%->58.0% recall,
# no boost needed after all), Locomotion crashed (70.0%->37.5%, confusion matrix shows 18/40
# of its bouts now going to Investigation specifically), and Tail investigation's 1.4x boost
# succeeded outright (47.2%->66.0%). Exploring is confirmed stuck, not just noisy: exactly
# 15.0% recall at both checkpoints despite its existing 1.2x boost.
#   Exploring 15.0% (unchanged)     -> confirmed stuck -> 1.2x -> 1.5x
#   Air sniffing 48.5%->30.3%       -> worsening despite boost -> 1.15x -> 1.35x
#   Approaching/Following 49.4%->58.0% -> self-corrected -> no boost needed, back to 1.0x
#   Tail investigation 47.2%->66.0% -> succeeded -> ease back 1.4x -> 1.2x, free up budget
#   Investigation 43.9%->50.9%      -> recall gain came AT Locomotion's expense (still the
#                                       catch-all) -> ease further 0.85x -> 0.8x
#   Locomotion 70.0%->37.5%         -> new problem, being cannibalized by Investigation ->
#                                       new boost 1.0x -> 1.3x to compete for prediction mass
#   Rearing 69.4%->83.7%            -> even more dominant -> keep easing, 0.75x unchanged
#   Grooming/Body touch             -> stable -> left at 1.0x
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/master_thesis_social_behaviour}"
SCRIPT_DIR="$PROJECT_ROOT/scripts/5_rat_transformer_training"
H5="$PROJECT_ROOT/rat_behaviour_32f.h5"
OUT_DIR="$SCRIPT_DIR/training_output"

GPU_ID=0   # RTX 4500 Ada, 24GB — confirmed free via nvidia-smi

cd "$SCRIPT_DIR"
mkdir -p "$OUT_DIR"

PID_FILE="$OUT_DIR/train.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "a training process (pid $(cat "$PID_FILE")) is already running against $OUT_DIR — refusing to start a second one."
  echo "(this is what caused the earlier CUDA OOM: two instances briefly competed for the same GPU memory.)"
  echo "if that process is actually dead, remove $PID_FILE and re-run."
  exit 1
fi

RESUME_ARGS=()
if [ -f "$OUT_DIR/last.pt" ]; then
  echo "found existing checkpoint at $OUT_DIR/last.pt -> resuming"
  RESUME_ARGS=(--resume "$OUT_DIR/last.pt")
else
  echo "no existing checkpoint -> starting fresh"
fi

nohup python3 train.py \
  --h5 "$H5" \
  --index-csv "$SCRIPT_DIR/window_index.csv" \
  --normstats "$SCRIPT_DIR/rgb_normstats.json" \
  --out-dir "$OUT_DIR" \
  --gpu-id "$GPU_ID" \
  --epochs 330 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --num-cams 6 \
  --micro-batch-size 16 \
  --num-workers 16 \
  --confusion-matrix-every 2 \
  --sampler balanced \
  --backbone-pooling attention \
  --min-class-weight 0.75 \
  --use-flow \
  --flow-size 256 \
  --flow-scale 20.0 \
  --flow-dim 64 \
  --exclude-classes "Immobile" \
  --class-weight-multiplier "Exploring:1.5,Air sniffing:1.35,Tail investigation:1.2,Investigation:0.8,Locomotion:1.3,Rearing:0.75" \
  --use-scale-feature \
  --transform-meta-root "$PROJECT_ROOT/training_metadata" \
  --weight-decay 0.2 \
  --label-smoothing 0.1 \
  --dropout 0.2 \
  "${RESUME_ARGS[@]}" \
  > "$OUT_DIR/stdout.log" 2>&1 &

echo $! > "$PID_FILE"
echo "training started in background, pid $!"
echo "tail -f $OUT_DIR/train.log     # clean, timestamped progress log"
echo "tail -f $OUT_DIR/stdout.log    # raw stdout incl. tqdm progress bars"
