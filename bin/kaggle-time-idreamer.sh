#!/usr/bin/env bash
# Times IDreamer training on a Kaggle GPU (T4 or P100), PLAN.md item 6.
#
# Use: in a Kaggle notebook with GPU and internet on, run
#   !bash bin/kaggle-time-idreamer.sh        (after cloning), or paste this file
#   into a cell with %%bash at the top.
# Settings through environment variables:
#   STEPS   environment steps per timing run (default 1.5e4)
#   BRANCH  git branch to clone (default idreamer-vmas-fixes, which has IDreamer)
# Output: /kaggle/working/timing/timing.jsonl, one JSON line per run with GPU
# name, compile setting, wall seconds, last steps per second and peak GPU MiB.
# Divide 1e5 by the steps per second to get hours per 100,000-step run.
#
# It runs twice: model.compile=False, then True, to see whether torch.compile
# pays for its compile time on Kaggle.
#
# To undo: delete this file. Nothing else uses it.
set -euo pipefail
STEPS=${STEPS:-1.5e4}
BRANCH=${BRANCH:-idreamer-vmas-fixes}
ROOT=/kaggle/working/co-imaginer
OUT=/kaggle/working/timing

# Same pinned uv as bin/uv, then the exact locked packages (AGENTS.md, Kaggle).
pip install -q uv==0.13.7
[ -d "$ROOT" ] || git clone -q --branch "$BRANCH" https://github.com/aswinmylraj/co-imaginer.git "$ROOT"
cd "$ROOT"
uv sync --locked
mkdir -p "$OUT"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
for COMPILE in False True; do
  RUN="$OUT/compile_$COMPILE"
  # Sample GPU memory every 5 s in the background, to find the peak.
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$RUN.gpu.csv" &
  SAMPLER=$!
  START=$(date +%s)
  # IDreamer on Discovery with 4 robots, offline W&B, no evaluation episodes so
  # only training is timed.
  (cd third_party/r2dreamer && WANDB_MODE=offline PYTHONPATH="$ROOT:." "$ROOT/.venv/bin/python" train.py \
    env=vmas_discovery logdir="$RUN" seed=0 env.steps="$STEPS" env.eval_episode_num=0 \
    model.compile="$COMPILE" > "$RUN.log" 2>&1)
  END=$(date +%s)
  kill "$SAMPLER"
  # Last logged training speed and the highest memory sample.
  FPS=$(grep '"fps/fps"' "$RUN/metrics.jsonl" | tail -1 | sed 's/.*"fps\/fps": \([0-9.]*\).*/\1/')
  PEAK=$(sort -n "$RUN.gpu.csv" | tail -1)
  echo "{\"gpu\": \"$GPU\", \"compile\": \"$COMPILE\", \"steps\": \"$STEPS\", \"wall_seconds\": $((END - START)), \"last_fps\": $FPS, \"peak_gpu_mib\": $PEAK}" | tee -a "$OUT/timing.jsonl"
done
