"""Download, filter, and sample the Lichess chess position evaluation dataset."""

import logging
import os
import random
from pathlib import Path
from typing import Optional

os.environ.setdefault("USE_TORCH", "0")  # this pipeline is CPU-only; skip datasets' torch probing

import datasets
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "Lichess/chess-position-evaluations"
DEFAULT_CACHE_DIR = Path("data/raw")
DEFAULT_SAMPLE_DIR = Path("data/processed")
SEED = 42
TARGET_SAMPLE_SIZE = 2_000_000

# Stockfish evaluation quality thresholds
# Positions with "mate" score or centipawn loss near zero are highest quality
MAX_CP_LOSS = 200  # Filter out blunder positions
MIN_CP_LOSS = -200  # Lower bound for reasonable positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_fen(fen: str) -> bool:
    """Quick validation that a FEN string looks plausible."""
    import chess
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False


def _quality_filter(row: dict) -> bool:
    """Return True if the position is high-enough quality for training."""
    cp = row.get("cp")
    # Keep positions with valid centipawn evaluations
    if cp is not None and (cp < MIN_CP_LOSS or cp > MAX_CP_LOSS):
        return False
    # Keep positions with mate evaluations
    mate = row.get("mate")
    if mate is not None and abs(mate) > 10:
        # Very long forced mates are less useful for training
        return False
    # Keep if cp is None but mate is set (mate is valuable signal)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dataset(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    split: str = "train",
    num_proc: int = 4,
) -> datasets.Dataset:
    """Download the full Lichess chess position evaluations dataset.

    Args:
        cache_dir: Directory to cache the raw dataset.
        split: Dataset split to download (default "train", also "validation").
        num_proc: Number of processes for data loading.

    Returns:
        The downloaded HuggingFace dataset object.

    Raises:
        ConnectionError: If the dataset cannot be reached.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s (split=%s) ...", DATASET_NAME, split)
    try:
        ds = datasets.load_dataset(
            DATASET_NAME,
            split=split,
            cache_dir=str(cache_dir),
            num_proc=num_proc,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise ConnectionError(
            f"Failed to download dataset {DATASET_NAME}: {exc}"
        ) from exc

    logger.info("Downloaded %s: %s rows", split, len(ds))
    return ds


def filter_positions(
    ds: datasets.Dataset,
    max_cp_loss: int = MAX_CP_LOSS,
    min_cp_loss: int = MIN_CP_LOSS,
    validate_fen: bool = False,
    num_proc: int = 8,
) -> datasets.Dataset:
    """Apply quality filters to the raw dataset.

    Args:
        ds: Raw HuggingFace dataset.
        max_cp_loss: Maximum centipawn loss to keep.
        min_cp_loss: Minimum centipawn loss to keep.
        validate_fen: Whether to validate FEN strings via python-chess. Off by
            default: on this dataset it rejects ~0 rows beyond the cp/mate
            filter, yet costs far more time (full board parsing per row).
        num_proc: Number of parallel processes for filtering.

    Returns:
        Filtered dataset.
    """
    logger.info("Applying quality filters (cp=[%s, %s]) ...", min_cp_loss, max_cp_loss)

    initial_count = len(ds)
    ds = ds.filter(
        _quality_filter,
        num_proc=num_proc,
        desc="Filtering positions",
    )

    if validate_fen:
        logger.info("Validating FEN strings ...")
        ds = ds.filter(
            lambda row: _is_valid_fen(str(row.get("fen", ""))),
            num_proc=max(1, num_proc // 2),
            desc="Validating FEN",
        )

    logger.info(
        "Filtered: %s → %s rows (%.1f%% kept)",
        initial_count,
        len(ds),
        100.0 * len(ds) / initial_count if initial_count else 0,
    )
    return ds


def sample_positions(
    ds: datasets.Dataset,
    n: int = TARGET_SAMPLE_SIZE,
    seed: int = SEED,
) -> datasets.Dataset:
    """Sample *n* positions deterministically.

    Args:
        ds: Filtered dataset.
        n: Desired sample size (~2 million).
        seed: Random seed for reproducibility.

    Returns:
        Shuffled and truncated dataset.
    """
    if len(ds) <= n:
        logger.warning(
            "Dataset has only %s rows (requested %s); using all.",
            len(ds),
            n,
        )
        return ds.shuffle(seed=seed)

    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(n))
    logger.info("Sampled %s positions (seed=%s).", len(ds), seed)
    return ds


def _list_shard_files() -> list[str]:
    """List the dataset's Parquet shard filenames on the Hub."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET_NAME, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def sample_by_shards(
    n: int = TARGET_SAMPLE_SIZE,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    seed: int = SEED,
    max_cp_loss: int = MAX_CP_LOSS,
    min_cp_loss: int = MIN_CP_LOSS,
    validate_fen: bool = False,
    num_proc: int = 8,
) -> datasets.Dataset:
    """Download and filter Parquet shards one at a time until *n* positions are collected.

    The dataset is split into ~20 Parquet shards (~2GB / ~48M rows each) on
    the Hub. Downloading whole shards and filtering them locally — instead
    of the full dataset or row-by-row remote streaming — keeps the download
    small (usually just 1-2 shards for a multi-million row sample) while
    filtering at full local multiprocessing speed.

    Args:
        n: Desired sample size (~2 million).
        cache_dir: Directory to cache downloaded shards.
        seed: Random seed controlling shard order.
        max_cp_loss: Maximum centipawn loss to keep.
        min_cp_loss: Minimum centipawn loss to keep.
        validate_fen: Whether to validate FEN strings via python-chess. Off by
            default; see :func:`filter_positions`.
        num_proc: Number of parallel processes for filtering.

    Returns:
        A filtered dataset with at least *n* rows, or all available rows if
        the whole dataset is exhausted first.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    shard_files = _list_shard_files()
    order = list(range(len(shard_files)))
    random.Random(seed).shuffle(order)

    collected = []
    total = 0
    for i in order:
        shard = shard_files[i]
        logger.info("Downloading shard %s ...", shard)
        local_path = hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=shard,
            cache_dir=str(cache_dir),
        )
        shard_ds = datasets.load_dataset("parquet", data_files=local_path, split="train")
        shard_ds = filter_positions(
            shard_ds,
            max_cp_loss=max_cp_loss,
            min_cp_loss=min_cp_loss,
            validate_fen=validate_fen,
            num_proc=num_proc,
        )
        collected.append(shard_ds)
        total += len(shard_ds)
        logger.info(
            "Collected %s/%s positions so far (%s shard(s) read).", total, n, len(collected)
        )
        if total >= n:
            break
    else:
        logger.warning(
            "Exhausted all %s shards; only %s positions available (requested %s).",
            len(shard_files),
            total,
            n,
        )

    return datasets.concatenate_datasets(collected)


def split_train_val_test(
    ds: datasets.Dataset,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    seed: int = SEED,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """Split into train / validation / test sets.

    Args:
        ds: Shuffled dataset.
        train_ratio: Fraction for training (default 0.98).
        val_ratio: Fraction for validation (default 0.01).
        seed: Random seed.

    Returns:
        (train_dataset, val_dataset, test_dataset).
    """
    total = len(ds)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)
    test_ds = ds.select(test_idx)

    logger.info("Split: train=%s, val=%s, test=%s", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds


def save_processed(
    ds: datasets.Dataset,
    output_dir: str | Path = DEFAULT_SAMPLE_DIR,
    name: str = "train",
) -> Path:
    """Save a processed split to disk as Arrow (Parquet) format.

    Args:
        ds: Processed dataset split.
        output_dir: Output directory.
        name: Split name (train/val/test).

    Returns:
        Path to the saved data directory.
    """
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    logger.info("Saved %s split to %s", name, output_dir)
    return output_dir


def load_processed(
    input_dir: str | Path,
) -> datasets.Dataset:
    """Load a previously saved processed dataset split.

    Args:
        input_dir: Path to saved split directory.

    Returns:
        Loaded dataset.
    """
    return datasets.load_from_disk(str(input_dir))


def run_pipeline(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_SAMPLE_DIR,
    target_size: int = TARGET_SAMPLE_SIZE,
    num_proc: int = 8,
    seed: int = SEED,
    streaming: bool = True,
    validate_fen: bool = False,
) -> dict[str, Path]:
    """Run the full download → filter → sample → split → save pipeline.

    Args:
        cache_dir: Cache directory for raw data (shards, when streaming).
        output_dir: Output directory for processed data.
        target_size: Number of positions to sample.
        num_proc: Parallel processes.
        seed: Random seed.
        streaming: If True (default), download and filter Parquet shards one
            at a time, stopping once `target_size` positions have been
            collected — usually just 1-2 of the ~20 shards — instead of
            downloading the full ~40GB dataset up front.
        validate_fen: Whether to validate FEN strings via python-chess. Off by
            default; see :func:`filter_positions`.

    Returns:
        Dictionary mapping split names to saved paths.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if streaming:
        filtered = sample_by_shards(
            n=target_size,
            cache_dir=cache_dir,
            seed=seed,
            validate_fen=validate_fen,
            num_proc=num_proc,
        )
    else:
        raw = download_dataset(cache_dir=cache_dir, num_proc=num_proc)
        filtered = filter_positions(raw, num_proc=num_proc, validate_fen=validate_fen)

    sampled = sample_positions(filtered, n=target_size, seed=seed)
    train, val, test = split_train_val_test(sampled, seed=seed)

    paths = {}
    for name, split_ds in [("train", train), ("val", val), ("test", test)]:
        paths[name] = save_processed(split_ds, output_dir=output_dir, name=name)

    return paths


if __name__ == "__main__":
    run_pipeline()
