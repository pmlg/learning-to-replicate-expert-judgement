# Evaluation

## Overview

The evaluation suite measures the model's chess playing strength through multiple metrics and benchmarks.

## Metrics

1. **Centipawn Loss**: Average difference between model moves and Stockfish top moves
2. **Legal Move %**: Rate of legal moves generated
3. **Win Rate**: Performance vs Stockfish (depth 8) and random player
4. **Estimated Elo**: Calculated from match results using standard Elo formula
5. **Puzzle Accuracy**: % correct on tactical puzzle datasets

## Usage

```bash
# Benchmark a checkpoint
bash scripts/evaluate.sh checkpoints/sft/final

# Compare multiple checkpoints
bash scripts/evaluate.sh checkpoints/sft/final --compare checkpoints/rl/final

# Custom options
bash scripts/evaluate.sh checkpoints/sft/final \
    --stockfish-games 50 \
    --depth 12 \
    --temperature 0.1
```
