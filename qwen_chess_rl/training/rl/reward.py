"""Stockfish-based reward function for chess move evaluation in RL training."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import chess
import chess.engine

logger = logging.getLogger(__name__)

# Reward scaling constants
LEGAL_MOVE_BONUS = 0.0  # no reward just for legality — only the outcome matters
CHECKMATE_BONUS = 10.0
BLUNDER_PENALTY = -5.0
DRAW_AVOIDANCE_BONUS = 3.0
REPETITION_AVOIDANCE_BONUS = 2.0
MAX_CP_PENALTY = -10.0

# Episode-outcome reward constants (multi-ply rollouts, see rollout.py).
# Winning is the only thing that scores positively; a normal (undecided)
# move carries no reward at all; losing is a heavy penalty; a draw is at
# worst a small penalty (see DRAW_REWARD) unless it was salvaged from a
# clearly lost position, in which case it's rewarded (DRAW_REWARD_WHEN_LOSING).
WIN_REWARD = 20.0
LOSS_REWARD = -20.0
DRAW_REWARD = 0.0  # a draw from a balanced/winning start is neutral, not punished
# Capped (non-terminated) rollouts didn't reach a decisive outcome — no
# natural conclusion means no reward, regardless of the position's eval.

# A draw's value depends on context: failing to convert a winning/equal
# position into a win is still discouraged (DRAW_REWARD), but salvaging a
# draw from a position that was already clearly lost is a genuinely good
# outcome — winning isn't realistic there, so the objective becomes "don't
# lose." LOSING_EVAL_THRESHOLD (centipawns, policy's perspective) is the
# rough cutoff for "clearly worse" used to tell the two cases apart.
LOSING_EVAL_THRESHOLD = -150.0
DRAW_REWARD_WHEN_LOSING = 10.0


def _find_stockfish() -> str | None:
    """Locate the Stockfish engine binary.

    Checks common installation paths and PATH.

    Returns:
        Path to Stockfish binary, or None if not found.
    """
    candidates = [
        "stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def evaluate_position_stockfish(
    fen: str,
    stockfish_path: Optional[str] = None,
    depth: int = 15,
    timeout: float = 5.0,
) -> dict:
    """Evaluate a chess position using Stockfish.

    Args:
        fen: FEN string of the position.
        stockfish_path: Path to Stockfish binary (auto-detect if None).
        depth: Analysis depth for Stockfish.
        timeout: Maximum time in seconds for analysis.

    Returns:
        Dictionary with keys:
            - 'score': Centipawn score (positive = white advantage).
            - 'mate': Mate in N moves (None if no forced mate).
            - 'bestmove': Best move in UCI notation.
            - 'pv': Principal variation (list of UCI moves).
    """
    stockfish_path = stockfish_path or _find_stockfish()
    if stockfish_path is None:
        logger.warning("Stockfish not found; returning dummy evaluation.")
        return {"score": 0, "mate": None, "bestmove": None, "pv": []}

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        board = chess.Board(fen)
        result = engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=timeout),
        )
        engine.quit()

        score_obj = result.get("score")
        if score_obj is None:
            return {"score": 0, "mate": None, "bestmove": None, "pv": []}

        pv = result.get("pv", [])
        bestmove = str(pv[0]) if pv else None

        # PovScore itself has no .score()/.mate() — those live on the
        # underlying Score, reached via .relative (relative to side to move).
        relative_score = score_obj.relative

        if relative_score.is_mate():
            mate_in = relative_score.mate()
            return {
                "score": 0 if mate_in is None else (10000 if mate_in > 0 else -10000),
                "mate": mate_in,
                "bestmove": bestmove,
                "pv": [str(m) for m in pv],
            }
        else:
            centipawns = relative_score.score()  # relative to side to move
            return {
                "score": centipawns,
                "mate": None,
                "bestmove": bestmove,
                "pv": [str(m) for m in pv],
            }
    except Exception as exc:
        logger.error("Stockfish evaluation failed: %s", exc)
        return {"score": 0, "mate": None, "bestmove": None, "pv": []}


def compute_reward(
    fen: str,
    proposed_move: str,
    stockfish_path: Optional[str] = None,
    eval_depth: int = 15,
) -> dict:
    """Compute reward for a proposed move given a FEN position.

    The reward considers multiple chess-quality factors:

    1. **Legal move** (+5): The move must be legal in the given position.
    2. **Centipawn loss** (0 to -10): Scaled penalty proportional to how much
       Stockfish judges the move to be worse than the best move.
    3. **Checkmate** (+10): If the proposed move is checkmate.
    4. **Blunder** (-5): Additional penalty if centipawn loss > 100.
    5. **Draw avoidance** (+3): If the position would otherwise lead to a draw.
    6. **Repetition avoidance** (+2): If the move prevents threefold repetition.

    Args:
        fen: FEN string of the current board position.
        proposed_move: Proposed move in UCI notation (e.g., 'e2e4').
        stockfish_path: Path to Stockfish binary (auto-detect if None).
        eval_depth: Stockfish analysis depth.

    Returns:
        Dictionary with keys:
            - 'total_reward': Combined reward score.
            - 'breakdown': Component scores for interpretability.
            - 'legal': Whether the move is legal.
            - 'best_move': Stockfish's best move in this position.
            - 'cp_loss': Centipawn loss of the proposed move.
    """
    reward_breakdown = {
        "legal_move": 0.0,
        "centipawn_score": 0.0,
        "checkmate": 0.0,
        "blunder_penalty": 0.0,
        "draw_avoidance": 0.0,
        "repetition_avoidance": 0.0,
    }
    total_reward = 0.0

    try:
        board = chess.Board(fen)
    except ValueError:
        logger.warning("Invalid FEN: %s", fen)
        return {
            "total_reward": -10.0,
            "breakdown": reward_breakdown,
            "legal": False,
            "best_move": None,
            "cp_loss": None,
        }

    # 1. Legal move check
    try:
        move = chess.Move.from_uci(proposed_move)
        is_legal = move in board.legal_moves
    except (ValueError, AttributeError):
        move = None
        is_legal = False

    if not is_legal or move is None:
        logger.debug("Illegal move: %s in position %s", proposed_move, fen)
        return {
            "total_reward": -10.0,
            "breakdown": {**reward_breakdown, "legal_move": -10.0},
            "legal": False,
            "best_move": None,
            "cp_loss": None,
        }

    reward_breakdown["legal_move"] = LEGAL_MOVE_BONUS
    total_reward += LEGAL_MOVE_BONUS

    # 2. Evaluate position with Stockfish
    eval_result = evaluate_position_stockfish(fen, stockfish_path, depth=eval_depth)
    best_move_uci = eval_result.get("bestmove")
    sf_score = eval_result.get("score", 0)

    if sf_score is not None:
        # Centipawn loss: difference between best move score and proposed move score
        board.push(move)
        proposed_eval = evaluate_position_stockfish(
            board.fen(), stockfish_path, depth=eval_depth
        )
        board.pop()

        proposed_score = proposed_eval.get("score", 0)
        # proposed_score is relative to the opponent (side to move after our
        # move was pushed) — negate to compare in the same reference frame
        # as sf_score before summing.
        cp_loss = max(0.0, sf_score + proposed_score)
        cp_reward = -min(cp_loss / 100.0, abs(MAX_CP_PENALTY))
        reward_breakdown["centipawn_score"] = cp_reward
        total_reward += cp_reward

        # 4. Blunder penalty
        if cp_loss > 100:
            reward_breakdown["blunder_penalty"] = BLUNDER_PENALTY
            total_reward += BLUNDER_PENALTY
    else:
        cp_loss = None

    # 3-6. Evaluate move consequences
    ev_board = board.copy()
    ev_board.push(move)

    # 3. Checkmate detection
    if ev_board.is_checkmate():
        reward_breakdown["checkmate"] = CHECKMATE_BONUS
        total_reward += CHECKMATE_BONUS

    # 5. Draw avoidance
    if ev_board.is_stalemate() or ev_board.is_insufficient_material():
        if len(list(board.legal_moves)) > 1:
            reward_breakdown["draw_avoidance"] = -DRAW_AVOIDANCE_BONUS
            total_reward -= DRAW_AVOIDANCE_BONUS

    # 6. Repetition avoidance
    if ev_board.is_repetition(count=2):
        reward_breakdown["repetition_avoidance"] = -REPETITION_AVOIDANCE_BONUS
        total_reward -= REPETITION_AVOIDANCE_BONUS

    return {
        "total_reward": total_reward,
        "breakdown": reward_breakdown,
        "legal": True,
        "best_move": best_move_uci,
        "cp_loss": cp_loss,
    }


def compute_episode_reward(
    outcome: Optional[str],
    starting_eval: Optional[float] = None,
) -> float:
    """Convert a rollout's terminal outcome into a scalar reward.

    Only a decisive outcome carries reward: winning is the sole positive
    signal, losing is a heavy penalty, and a capped rollout — one that
    didn't reach a natural conclusion, i.e. "just a normal move" — gets
    no reward at all, regardless of the position's eval at the cutoff.

    Args:
        outcome: 'win', 'loss', 'draw' if the game reached a natural
            conclusion during the rollout, or None if it was truncated
            (capped rollout) before ending.
        starting_eval: Centipawn evaluation of the position *before* the
            rollout began, from the policy's perspective. Only used when
            ``outcome`` is 'draw' — if the starting position was already
            clearly lost (below LOSING_EVAL_THRESHOLD), winning wasn't a
            realistic objective, so salvaging a draw is rewarded well
            instead of just "less bad than losing".

    Returns:
        Scalar reward on the same scale as WIN_REWARD/LOSS_REWARD.
    """
    if outcome == "win":
        return WIN_REWARD
    if outcome == "loss":
        return LOSS_REWARD
    if outcome == "draw":
        if starting_eval is not None and starting_eval < LOSING_EVAL_THRESHOLD:
            return DRAW_REWARD_WHEN_LOSING
        return DRAW_REWARD
    # Capped (no natural conclusion) — a "normal move" gets no reward.
    return 0.0


def batch_compute_rewards(
    positions: list[tuple[str, str]],
    stockfish_path: Optional[str] = None,
    eval_depth: int = 15,
    num_workers: int = 4,
) -> list[dict]:
    """Compute rewards for a batch of (FEN, move) pairs.

    Args:
        positions: List of (fen, move) tuples.
        stockfish_path: Path to Stockfish binary.
        eval_depth: Stockfish analysis depth.
        num_workers: Number of parallel workers (not yet implemented).

    Returns:
        List of reward dictionaries.
    """
    results = []
    for fen, move in positions:
        reward = compute_reward(fen, move, stockfish_path, eval_depth)
        results.append(reward)
    return results
