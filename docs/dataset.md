# Dataset Preparation

## Overview

The data pipeline downloads the [Lichess chess position evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) dataset, filters for quality, samples ~2 million positions, and splits into train/validation/test sets.

## Pipeline Steps

1. **Download**: Downloads the full dataset from HuggingFace Datasets
2. **Filter**: Removes low-quality positions (centipawn loss > 200, invalid FEN)
3. **Sample**: Deterministically samples ~2M positions
4. **Split**: 98% train / 1% validation / 1% test
5. **Save**: Arrow format for fast loading

## Usage

```bash
# Full pipeline
python -m qwen_chess_rl.data.download

# Custom options
python -c "
from qwen_chess_rl.data.download import run_pipeline
paths = run_pipeline(
    cache_dir='data/raw',
    output_dir='data/processed',
    target_size=2_000_000,
    num_proc=8,
)
"
```

## Data Format

Each example contains:
- `fen`: FEN string of the position
- `cp` or `mate`: Stockfish evaluation
- `bestmove`: Best move in UCI notation (optional)
- `pv`: Principal variation (optional)
