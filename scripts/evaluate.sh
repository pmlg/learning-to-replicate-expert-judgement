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
    echo "  --output-dir DIR         Results directory (default: evaluation_results)"
    echo "  --stockfish-games N      Games vs Stockfish (default: 20)"
    echo "  --random-games N         Games vs random (default: 10)"
    echo "  --stockfish-depth N      Stockfish depth (default: 8)"
    echo "  --stockfish-skill-level N  Stockfish Skill Level 0-20 (default: full strength)"
    echo "  --temperature T          Sampling temperature (default: 0.1)"
    echo "  --engine {llm,search,hybrid}  Policy: raw LLM, alpha-beta search, or LLM+search (default: llm)"
    echo "  --search-depth N         Alpha-beta depth for search/hybrid (default: 4)"
    echo "  --search-time-limit T    Per-move time budget (s) for search/hybrid (default: 1.0)"
    echo "  --hybrid-margin CP       Centipawn margin for hybrid engine (default: 50)"
    echo "  --compare CKPT ...       Additional checkpoints to compare"
    exit 1
fi

CHECKPOINT="$1"
shift

echo "=== Model Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export STOCKFISH_PATH=${STOCKFISH_PATH:-stockfish}

uv run python -m qwen_chess_rl.evaluation.benchmark \
    "$CHECKPOINT" \
    "$@"

echo "Evaluation complete. Results saved to evaluation_results/"
