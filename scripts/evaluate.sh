#!/bin/bash
# Evaluate a trained model checkpoint
# Usage: ./scripts/evaluate.sh <checkpoint_path> [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [options]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR    Results directory (default: evaluation_results)"
    echo "  --stockfish-games N Games vs Stockfish (default: 20)"
    echo "  --random-games N    Games vs random (default: 10)"
    echo "  --depth N           Stockfish depth (default: 8)"
    echo "  --temperature T     Sampling temperature (default: 0.1)"
    echo "  --compare CKPT ...  Additional checkpoints to compare"
    exit 1
fi

CHECKPOINT="$1"
shift

echo "=== Model Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export STOCKFISH_PATH=${STOCKFISH_PATH:-stockfish}

python -m qwen_chess_rl.evaluation.benchmark \
    "$CHECKPOINT" \
    "$@"

echo "Evaluation complete. Results saved to evaluation_results/"
