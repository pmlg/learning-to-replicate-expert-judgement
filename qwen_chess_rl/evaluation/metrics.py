"""Evaluation metrics for chess model playing strength."""

import logging
from pathlib import Path
from typing import Optional

import chess
import numpy as np

from qwen_chess_rl.training.rl.reward import evaluate_position_stockfish

logger = logging.getLogger(__name__)


def compute_centipawn_loss(
    fen: str,
    proposed_move: str,
    stockfish_path: Optional[str] = None,
    depth: int = 20,
) -> float:
    """Compute centipawn loss for a move in a given position.

    Args:
        fen: FEN position.
        proposed_move: Move in UCI notation.
        stockfish_path: Path to Stockfish binary.
        depth: Analysis depth.

    Returns:
        Centipawn loss (absolute difference between best move score and
        proposed move score). Returns -1 if evaluation fails.
    """
    best_eval = evaluate_position_stockfish(fen, stockfish_path, depth=depth)
    best_score = best_eval.get("score")

    if best_score is None:
        return -1.0

    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(proposed_move)
        if move not in board.legal_moves:
            return 1000.0  # High penalty for illegal moves
        board.push(move)
        proposed_eval = evaluate_position_stockfish(board.fen(), stockfish_path, depth=depth)
        proposed_score = proposed_eval.get("score", 0)
        # proposed_score is relative to the opponent (side to move after the
        # move); negate to compare in the same reference frame as best_score.
        return max(0.0, best_score + proposed_score) if proposed_score is not None else -1.0
    except (ValueError, chess.IllegalMoveError):
        return 1000.0


def legal_move_percentage(
    positions: list[tuple[str, str]],
) -> float:
    """Calculate the percentage of moves that are legal.

    Args:
        positions: List of (fen, move) pairs.

    Returns:
        Fraction of legal moves (0.0 to 1.0).
    """
    if not positions:
        return 0.0

    legal_count = 0
    for fen, move_uci in positions:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                legal_count += 1
        except (ValueError, chess.IllegalMoveError):
            continue

    return legal_count / len(positions)


def compute_win_rate(
    results: list[dict],
) -> dict[str, float]:
    """Compute win/draw/loss rates from match results.

    Args:
        results: List of match result dicts with keys 'result'
                 ('win', 'draw', 'loss', or numeric where 1=win, 0.5=draw, 0=loss).

    Returns:
        Dictionary with 'win_rate', 'draw_rate', 'loss_rate'.
    """
    n = len(results)
    if n == 0:
        return {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

    wins = draws = losses = 0
    for r in results:
        result = r.get("result", 0)
        if isinstance(result, str):
            if result.lower() == "win":
                wins += 1
            elif result.lower() == "draw":
                draws += 1
            else:
                losses += 1
        else:
            if result >= 0.9:
                wins += 1
            elif result >= 0.4:
                draws += 1
            else:
                losses += 1

    return {
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
    }


def estimate_elo(
    wins: int,
    draws: int,
    losses: int,
    opponent_elo: float,
) -> float:
    """Estimate Elo rating from match results against a known opponent.

    Uses standard Elo formula: expected score = 1 / (1 + 10^((opp_elo - our_elo)/400)).

    Args:
        wins: Number of wins.
        draws: Number of draws.
        losses: Number of losses.
        opponent_elo: Known Elo of the opponent.

    Returns:
        Estimated Elo rating.
    """
    total = wins + draws + losses
    if total == 0:
        return 0.0

    score = (wins + 0.5 * draws) / total
    # Solve: score = 1 / (1 + 10^((opp_elo - our_elo)/400))
    # => our_elo = opp_elo - 400 * log10(1/score - 1)
    if score <= 0.0:
        return opponent_elo - 400
    if score >= 1.0:
        return opponent_elo + 400

    import math
    return opponent_elo - 400 * math.log10(1.0 / score - 1.0)


def puzzle_accuracy(
    puzzles: list[dict],
    model_predictions: list[str],
) -> float:
    """Evaluate model accuracy on a chess puzzle dataset.

    Each puzzle contains a FEN position and the correct move.
    Returns the fraction of puzzles where the model's move matches.

    Args:
        puzzles: List of dicts with 'fen' and 'solution' (correct UCI move).
        model_predictions: List of model-predicted UCI moves (same order).

    Returns:
        Accuracy fraction (0.0 to 1.0).
    """
    if not puzzles:
        return 0.0

    correct = sum(
        1 for p, pred in zip(puzzles, model_predictions)
        if pred and p.get("solution", "").startswith(pred)
    )
    return correct / len(puzzles)


def compute_all_metrics(
    positions: list[tuple[str, str]],
) -> dict:
    """Compute all standard metrics for a set of (FEN, move) pairs.

    Args:
        positions: List of (fen, move) pairs.

    Returns:
        Dictionary with all computed metrics.
    """
    if not positions:
        return {
            "avg_centipawn_loss": 0.0,
            "legal_move_pct": 0.0,
            "sample_count": 0,
        }

    cp_losses = []
    for fen, move in positions:
        cp = compute_centipawn_loss(fen, move)
        if cp >= 0:
            cp_losses.append(cp)

    return {
        "avg_centipawn_loss": float(np.mean(cp_losses)) if cp_losses else 0.0,
        "median_centipawn_loss": float(np.median(cp_losses)) if cp_losses else 0.0,
        "std_centipawn_loss": float(np.std(cp_losses)) if len(cp_losses) > 1 else 0.0,
        "legal_move_pct": legal_move_percentage(positions),
        "sample_count": len(positions),
    }
