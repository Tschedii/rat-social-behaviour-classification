#!/usr/bin/env bash
# Training on the v2 dataset (2026-09-28): new Day 11 rat_1 labels, Day 11 rat_2 shifted +101
# frames, new Day 13 session, rebalanced train/eval split, and the build_index.py ordinal fix
# (point rows used to shift every later Unfamiliar bout onto the wrong .h5 group).
#
# Same hyperparameters as run_on_server.sh, so the data is the only thing that changes. Writes to
# its own H5 / index / output folder, so the v1 run and its checkpoints stay untouched.
#
# First launch warm-starts from INIT_CKPT (the v1 run's best.pt); any later launch resumes from
# training_output_v2/last.pt, exactly like run_on_server.sh. The epoch counter and the best
# eval macro-F1 carry over from INIT_CKPT, so best.pt only appears once the NEW eval split beats
# the v1 score; pick checkpoints from training_output_v2/top_checkpoints/ instead.
#
# Before running: build $H5 with 2_build_h5_dataset.py and $INDEX_DIR with helpers/build_index.py
# (see the commands in the chat / notes), and check the GPU is free with nvidia-smi.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/master_thesis_social_behaviour}"
SCRIPT_DIR="$PROJECT_ROOT/scripts/5_rat_transformer_training"
H5="$PROJECT_ROOT/training_data/rat_behaviour_32f_v2.h5"
META_ROOT="$PROJECT_ROOT/training_data/training_metadata"
INDEX_DIR="$SCRIPT_DIR/index_v2"
OUT_DIR="$SCRIPT_DIR/training_output_v2"
INIT_CKPT="$SCRIPT_DIR/training_output/best.pt"   # warm start for the first launch only

GPU_ID=0   # RTX 4500 Ada, 24GB — check nvidia-smi first

cd "$SCRIPT_DIR"
mkdir -p "$OUT_DIR"

for f in "$H5" "$INDEX_DIR/window_index.csv" "$INDEX_DIR/label_map.json"; do
  [ -f "$f" ] || { echo "missing $f — build the .h5 and the index first"; exit 1; }
done
[ -d "$META_ROOT" ] || { echo "missing transform metadata folder $META_ROOT"; exit 1; }

PID_FILE="$OUT_DIR/train.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "a training process (pid $(cat "$PID_FILE")) is already running against $OUT_DIR — refusing to start a second one."
  echo "if that process is actually dead, remove $PID_FILE and re-run."
  exit 1
fi
if [ -f "$SCRIPT_DIR/training_output/train.pid" ] && kill -0 "$(cat "$SCRIPT_DIR/training_output/train.pid")" 2>/dev/null; then
  echo "the v1 run (training_output/) is still running — stop it first, both won't fit on one GPU."
  exit 1
fi

if [ -f "$OUT_DIR/last.pt" ]; then
  echo "found $OUT_DIR/last.pt -> resuming the v2 run"
  RESUME="$OUT_DIR/last.pt"
else
  [ -f "$INIT_CKPT" ] || { echo "missing warm-start checkpoint $INIT_CKPT"; exit 1; }
  echo "no v2 checkpoint yet -> warm-starting from $INIT_CKPT"
  RESUME="$INIT_CKPT"
fi

nohup python3 train.py \
  --h5 "$H5" \
  --index-csv "$INDEX_DIR/window_index.csv" \
  --normstats "$SCRIPT_DIR/rgb_normstats.json" \
  --out-dir "$OUT_DIR" \
  --gpu-id "$GPU_ID" \
  --epochs 177 \
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
  --class-weight-multiplier "Exploring:1.2,Air sniffing:1.15,Tail investigation:1.2,Investigation:0.8,Locomotion:1.3,Rearing:1.0" \
  --use-scale-feature \
  --transform-meta-root "$META_ROOT" \
  --weight-decay 0.2 \
  --label-smoothing 0.1 \
  --dropout 0.2 \
  --resume "$RESUME" \
  > "$OUT_DIR/stdout.log" 2>&1 &

echo $! > "$PID_FILE"
echo "training started in background, pid $!"
echo "tail -f $OUT_DIR/train.log     # clean, timestamped progress log"
echo "tail -f $OUT_DIR/stdout.log    # raw stdout incl. tqdm progress bars"
