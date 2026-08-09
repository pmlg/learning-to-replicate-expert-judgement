# Dataset Preparation

## Overview

The data pipeline streams the [Lichess chess position evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) dataset, filters for quality, samples ~2 million positions, and splits into train/validation/test sets.

The full dataset is ~40GB of Parquet across 20 shards on the Hub. By default the pipeline streams rows and stops as soon as it has collected enough positions to fill the sample, so it only pulls the shards it actually needs instead of downloading everything up front. Pass `streaming=False` to `run_pipeline` if you want to materialize the entire dataset locally instead.

## Pipeline Steps

1. **Download**: Streams rows from the dataset on HuggingFace Datasets (or downloads the full dataset if `streaming=False`)
2. **Filter**: Removes low-quality positions (centipawn loss > 200, invalid FEN)
3. **Sample**: Deterministically samples ~2M positions (shuffled with a buffer while streaming)
4. **Split**: 98% train / 1% validation / 1% test
5. **Save**: Arrow format for fast loading

## Usage

```bash
# Full pipeline (streams by default)
python -m qwen_chess_rl.data.download

# Custom options
python -c "
from qwen_chess_rl.data.download import run_pipeline
paths = run_pipeline(
    cache_dir='data/raw',
    output_dir='data/processed',
    target_size=2_000_000,
    num_proc=8,
    streaming=True,  # set False to download the full dataset up front
)
"
```

## Data Format

Each example contains:
- `fen`: FEN string of the position
- `cp` or `mate`: Stockfish evaluation
- `bestmove`: Best move in UCI notation (optional)
- `pv`: Principal variation (optional)
