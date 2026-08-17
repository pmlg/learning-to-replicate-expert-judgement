#!/bin/bash
# Train the GRPO RL model starting from an SFT checkpoint
# Usage: ./scripts/train_grpo.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Default values
CONFIG=""
SFT_CHECKPOINT="checkpoints/sft/final"
DATA_PATH=""
OUTPUT_DIR="checkpoints/rl-luffy"
STEPS=1000
LR=5e-6
SFT_ALPHA="0.85"
MAX_TRAIN_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --sft-checkpoint) SFT_CHECKPOINT="$2"; shift 2 ;;
        --data) DATA_PATH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --sft-alpha) SFT_ALPHA="$2"; shift 2 ;;
        --max-train-samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== GRPO RL Training ==="
echo "SFT checkpoint: $SFT_CHECKPOINT"
echo "Data: ${DATA_PATH:-<config default>}"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS"
echo "Learning rate: $LR"
echo "sft_alpha: ${SFT_ALPHA:-<config default>}"
echo "Max train samples: ${MAX_TRAIN_SAMPLES:-<config default>}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export STOCKFISH_PATH=${STOCKFISH_PATH:-stockfish}

# Verify Stockfish is available
if ! command -v "$STOCKFISH_PATH" &> /dev/null; then
    echo "WARNING: Stockfish not found at $STOCKFISH_PATH. Reward computation will fail."
    echo "Install Stockfish: sudo apt-get install stockfish"
fi

uv run python -m qwen_chess_rl.training.rl.grpo \
    ${CONFIG:+--config "$CONFIG"} \
    --sft-checkpoint "$SFT_CHECKPOINT" \
    ${DATA_PATH:+--data-path "$DATA_PATH"} \
    --steps "$STEPS" \
    --lr "$LR" \
    ${SFT_ALPHA:+--sft-alpha "$SFT_ALPHA"} \
    ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"}

echo "GRPO training complete. Model saved to $OUTPUT_DIR"
