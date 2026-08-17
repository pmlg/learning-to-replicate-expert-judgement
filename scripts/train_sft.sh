#!/bin/bash
# Train the SFT model on chess position data
# Usage: ./scripts/train_sft.sh [options]
#
# By default uses 50k train / 2k val examples (full 2M is not required for
# a useful LoRA run). Pass --max-train-samples -1 for the full set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Default values
CONFIG=""
MODEL_NAME="Qwen/Qwen3-4B"
DATA_PATH="data/processed"
OUTPUT_DIR="checkpoints/sft"
BATCH_SIZE=24
LR=2e-4
MAX_STEPS=-1
MAX_TRAIN_SAMPLES=50000
MAX_EVAL_SAMPLES=2000
NO_WANDB=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --data) DATA_PATH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
        --max-eval-samples) MAX_EVAL_SAMPLES="$2"; shift 2 ;;
        --no-wandb) NO_WANDB="--no-wandb"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== SFT Training ==="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Max train samples: $MAX_TRAIN_SAMPLES (-1 = full)"
echo "Max eval samples: $MAX_EVAL_SAMPLES (-1 = full)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

uv run python -m qwen_chess_rl.training.sft.train \
    ${CONFIG:+--config "$CONFIG"} \
    --model-name "$MODEL_NAME" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --max-steps "$MAX_STEPS" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --max-eval-samples "$MAX_EVAL_SAMPLES" \
    $NO_WANDB

echo "SFT training complete. Checkpoint saved to $OUTPUT_DIR"
