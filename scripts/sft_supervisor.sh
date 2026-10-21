#!/bin/bash
# Supervises SFT training: restarts from the latest HF Trainer checkpoint on
# crash (same environment instability as scripts/grpo_supervisor.sh — sporadic
# process kills with no traceback). Unlike GRPO's custom step-counting loop,
# HF's Trainer keeps one continuous checkpoint-N sequence inside a single
# output_dir and understands resume_from_checkpoint=True natively, so this
# script is a much simpler retry loop around that.
set -u

OUTPUT_DIR="checkpoints/sft-1.7b-3ep"
MAX_RETRIES=30
LOG="$OUTPUT_DIR/train.log"
mkdir -p "$OUTPUT_DIR"

# Mirrors grpo_supervisor.sh's is_valid_checkpoint: a save killed mid-write
# leaves a checkpoint-N dir that exists but is missing its actual weights.
# HF's own get_last_checkpoint() only checks the directory name pattern, not
# completeness, so an interrupted checkpoint must be removed here or Trainer
# will try (and fail) to resume from it.
is_valid_checkpoint() {
  [ -f "$1/trainer_state.json" ] && { [ -f "$1/model.safetensors" ] || [ -f "$1/adapter_model.safetensors" ]; }
}

for d in "$OUTPUT_DIR"/checkpoint-*; do
  [ -d "$d" ] || continue
  if ! is_valid_checkpoint "$d"; then
    echo "[supervisor] Removing incomplete checkpoint: $d" >> "$LOG"
    rm -rf "$d"
  fi
done

retry=0
while [ "$retry" -lt "$MAX_RETRIES" ]; do
  if [ -d "$OUTPUT_DIR/final" ]; then
    echo "[supervisor] Final model already exists at $OUTPUT_DIR/final. Done." >> "$LOG"
    break
  fi

  has_checkpoint=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
  if [ -n "$has_checkpoint" ]; then
    resume_arg="auto"
  else
    resume_arg=""
  fi

  echo "[supervisor] Attempt $((retry+1))/$MAX_RETRIES (resume=${resume_arg:-none}) ..." >> "$LOG"

  if [ -n "$resume_arg" ]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m qwen_chess_rl.training.sft.train \
      --resume-from-checkpoint "$resume_arg" \
      >> "$LOG" 2>&1
  else
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m qwen_chess_rl.training.sft.train \
      >> "$LOG" 2>&1
  fi
  exit_code=$?

  if [ -d "$OUTPUT_DIR/final" ]; then
    echo "[supervisor] Training process exited cleanly (code $exit_code). Final model saved." >> "$LOG"
    break
  fi

  for d in "$OUTPUT_DIR"/checkpoint-*; do
    [ -d "$d" ] || continue
    is_valid_checkpoint "$d" || { echo "[supervisor] Removing incomplete checkpoint: $d" >> "$LOG"; rm -rf "$d"; }
  done

  echo "[supervisor] Attempt exited (code $exit_code). Retrying in 20s..." >> "$LOG"
  retry=$((retry+1))
  sleep 20
done

if [ "$retry" -ge "$MAX_RETRIES" ]; then
  echo "[supervisor] Hit max retries ($MAX_RETRIES). Stopping." >> "$LOG"
fi
