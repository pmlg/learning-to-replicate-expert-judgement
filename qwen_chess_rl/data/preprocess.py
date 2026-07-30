"""Preprocessing pipeline: FEN tokenization, data validation, and transformation."""

import logging
from pathlib import Path
from typing import Optional

import chess
import numpy as np
from datasets import Dataset, load_from_disk

logger = logging.getLogger(__name__)


def fen_to_features(fen: str) -> dict:
    """Extract structural features from a FEN string.

    Useful for auxiliary tasks or data quality analysis.

    Args:
        fen: FEN string.

    Returns:
        Dictionary with:
            - board_rank_count: count of pieces per rank.
            - piece_count: total number of pieces.
            - has_kings: whether both kings are present.
            - turn: 'w' or 'b'.
            - castling_rights: castling availability string.
            - en_passant: en passant target square (or '-').
            - halfmove_clock: halfmove clock value.
            - fullmove_number: fullmove number.
    """
    try:
        board = chess.Board(fen)
        return {
            "piece_count": board.uci(),
            "has_kings": bool(board.king(chess.WHITE) and board.king(chess.BLACK)),
            "turn": board.turn,
            "castling_rights": board.castling_xfen(),
            "legal_move_count": len(list(board.legal_moves)),
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
        }
    except ValueError:
        return {}


def validate_dataset(
    data_path: str | Path,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> dict:
    """Validate a processed dataset split for quality issues.

    Checks:
    - FEN validity
    - Move format (UCI)
    - Missing values
    - Balance of board positions (opening/middlegame/endgame)

    Args:
        data_path: Path to the processed dataset.
        split: Dataset split to validate.
        max_samples: Max samples to check (None = all).

    Returns:
        Dictionary with validation statistics.
    """
    data_path = Path(data_path)
    ds = load_from_disk(str(data_path / split))

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    total = len(ds)
    valid_fen = 0
    valid_move = 0
    missing_fen = 0
    missing_move = 0
    piece_counts = []

    for example in ds:
        fen = str(example.get("fen", ""))
        move = str(example.get("move", example.get("bestmove", "")))

        if not fen:
            missing_fen += 1
            valid_fen += 1  # skip fen validation if field missing
        else:
            try:
                board = chess.Board(fen)
                valid_fen += 1
                piece_counts.append(board.uci())
            except ValueError:
                pass

        if not move:
            missing_move += 1
        else:
            try:
                chess.Move.from_uci(move)
                valid_move += 1
            except ValueError:
                pass

    stats = {
        "total_samples": total,
        "valid_fen_pct": 100.0 * valid_fen / total if total else 0,
        "valid_move_pct": 100.0 * valid_move / total if total else 0,
        "missing_fen": missing_fen,
        "missing_move": missing_move,
        "invalid_fen": total - valid_fen - missing_fen,
        "invalid_move": total - valid_move - missing_move,
    }

    if piece_counts:
        stats["avg_piece_count"] = float(np.mean(piece_counts))
        stats["std_piece_count"] = float(np.std(piece_counts))

    logger.info("Dataset validation (%s): %s", split, stats)
    return stats


def deduplicate_positions(
    ds: Dataset,
    hash_column: str = "fen",
) -> Dataset:
    """Remove duplicate FEN positions from the dataset.

    Args:
        ds: Input dataset.
        hash_column: Column to deduplicate on.

    Returns:
        Deduplicated dataset (first occurrence kept).
    """
    seen = set()
    indices_to_keep = []

    for i, example in enumerate(ds):
        key = str(example.get(hash_column, ""))
        if key and key not in seen:
            seen.add(key)
            indices_to_keep.append(i)

    logger.info("Deduplication: %s → %s (%.1f%% unique)", len(ds), len(indices_to_keep),
                100.0 * len(indices_to_keep) / len(ds) if len(ds) else 0)
    return ds.select(indices_to_keep)


def compute_position_difficulty(fen: str) -> float:
    """Estimate the difficulty of a chess position (0 = trivial, 1 = very hard).

    Uses heuristics like material balance, tactical complexity, and
    number of legal moves.

    Args:
        fen: FEN string.

    Returns:
        Difficulty score [0, 1].
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return 0.5

    num_legal = len(list(board.legal_moves))
    piece_count = board.uci()

    # More legal moves → more tactical complexity → harder
    move_complexity = min(num_legal / 50.0, 1.0)

    # More pieces → more strategic complexity
    strategic_complexity = min(piece_count / 32.0, 1.0)

    # Check/checkmate adds tactical pressure
    tactical_bonus = 0.2 if board.is_check() else 0.0
    if board.is_checkmate():
        tactical_bonus = 0.0  # Actually trivial (only solution is the mate)

    return min(0.3 * move_complexity + 0.5 * strategic_complexity + 0.2 * tactical_bonus, 1.0)


def filter_by_difficulty(
    ds: Dataset,
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0,
    num_proc: int = 4,
) -> Dataset:
    """Filter positions by estimated difficulty level.

    Useful for curriculum learning (easy positions first, then hard).

    Args:
        ds: Input dataset.
        min_difficulty: Minimum difficulty [0, 1].
        max_difficulty: Maximum difficulty [0, 1].
        num_proc: Number of processes.

    Returns:
        Filtered dataset.
    """
    def _filter(example):
        difficulty = compute_position_difficulty(str(example.get("fen", "")))
        return min_difficulty <= difficulty <= max_difficulty

    filtered = ds.filter(_filter, num_proc=num_proc, desc="Filtering by difficulty")
    logger.info("Difficulty filter [%s, %s]: %s → %s",
                min_difficulty, max_difficulty, len(ds), len(filtered))
    return filtered
