"""Stockfish-based reward function for chess move evaluation in RL training.

Reward philosophy (per user requirement: *moves should be legal and the reward
should reward winning*):

    r = Q(s, a) = P(mover wins | position after the proposed move)

This is the dense win-probability reward that Chess-R1 (arXiv:2507.00726)
proved outperforms sparse binary rewards for LLM chess RL. It directly
measures "does this move win the game", is bounded in [0, 1], and contains
no arbitrary hand-tuned constants. (The old code's delta variant compared
win-prob before vs after from DIFFERENT perspectives — the post-move score
is relative to the opponent — which systematically mis-scored moves; the
perspective conversion `1 - winprob_after` is applied here.)

Design decisions:
- Illegal moves: hard floor of -1.0 (a gate, not a bonus). Legal moves get
  a *graded* signal from the win-probability delta, not a flat +5.
- Win probability: sigmoid(score) with Lichess-style scale
  p = 1 / (1 + 10^(-cp/400)). Checkmate => p=0 or p=1 exactly.
- Sign handling: python-chess scores are relative to the side to move. After
  pushing a move the turn flips, so the post-move score MUST be negated
  before comparing to the pre-move score. The old code compared them raw,
  which made a *good* move look like a ~100cp blunder.
- Engine lifecycle: one persistent engine process per batch call, reused for
  all evals. The old code spawned a fresh `popen_uci` per evaluation, which
  dominates runtime in RL rollouts.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import chess
import chess.engine

logger = logging.getLogger(__name__)

# Win-probability conversion scale (Lichess / Stockfish WDL convention).
CP_SCALE = 400.0

# Reward floors
ILLEGAL_REWARD = -1.0
INVALID_FEN_REWARD = -1.0
# Kept for back-compat with main's test_rollout.py — legality alone earns
# nothing (the new reward's illegal floor is ILLEGAL_REWARD).
LEGAL_MOVE_BONUS = 0.0


# ---------------------------------------------------------------------------
# Episode-level reward (used by GRPO rollout loop: main's grpo.py imports
# compute_episode_reward and these constants from this module).
# ---------------------------------------------------------------------------
WIN_REWARD: float = 20.0
LOSS_REWARD: float = -20.0
DRAW_REWARD: float = 0.0
DRAW_REWARD_WHEN_LOSING: float = 10.0
LOSING_EVAL_THRESHOLD: float = -150.0  # centipawns


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


def _find_stockfish() -> Optional[str]:
    """Locate the Stockfish engine binary.

    Checks common installation paths, PATH, and ~/bin (user-installed).
    """
    candidates = [
        "stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        str(Path.home() / "bin" / "stockfish"),
    ]
    for candidate in candidates:
        found = shutil.which(candidate) if "/" not in candidate else candidate
        try:
            result = subprocess.run(
                [found or candidate, "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return found or candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _score_to_win_prob(score_obj: Optional[chess.engine.Score]) -> float:
    """Convert a python-chess Score (relative to side to move) to win prob.

    Returns:
        win probability in [0, 1] for the side to move.
    """
    if score_obj is None:
        return 0.5
    if score_obj.is_mate():
        mate_in = score_obj.mate()
        # mate() > 0: side to move delivers mate; < 0: side to move gets mated.
        return 1.0 if mate_in and mate_in > 0 else 0.0
    cp = score_obj.score()
    if cp is None:
        return 0.5
    return 1.0 / (1.0 + 10.0 ** (-cp / CP_SCALE))


class RewardEngine:
    """Persistent Stockfish engine wrapper for fast batch reward computation.

    Spawning a UCI engine per evaluation is the dominant cost in RL reward
    loops; this class keeps one engine alive for a whole batch.
    """

    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 12):
        self._engine = None
        self.stockfish_path = stockfish_path or _find_stockfish()
        self.depth = depth
        if self.stockfish_path is None:
            logger.warning("Stockfish not found; reward computation disabled.")

    def __enter__(self):
        if self.stockfish_path is not None:
            self._engine = self._spawn()
        return self

    def __exit__(self, *exc):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001 - engine may already be dead
                pass
            self._engine = None

    def _spawn(self):
        if self.stockfish_path is None:
            raise RuntimeError("Stockfish binary not found")
        return chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def _ensure_alive(self):
        """Restart the engine if it died (crash, parse error, OOM)."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001
                pass
            self._engine = None
        self._engine = self._spawn()

    def eval_win_prob(self, board: chess.Board) -> float:
        """Evaluate a position and return win prob for the side to move.

        Robust to engine death: on any failure the engine is fully closed
        and a fresh process retries the SAME analysis once. (Resuming a
        restarted process is unsafe — python-chess replays pending futures
        against a positionless engine, which yields corrupt PVs.)
        """
        if self._engine is None:
            return 0.5
        for attempt in range(2):
            try:
                info = self._engine.analyse(
                    board, chess.engine.Limit(depth=self.depth)
                )
                score = info.get("score")
                # PovScore.relative is a PROPERTY -> Score from side-to-move
                # perspective (verified against python-chess runtime; the
                # type-checker stubs mislabel it as a method).
                return _score_to_win_prob(
                    score.relative if score is not None else None
                )
            except Exception as exc:  # noqa: BLE001 - engine errors are non-fatal
                if attempt == 0:
                    logger.warning(
                        "Stockfish eval failed (attempt %d), restarting: %s",
                        attempt + 1, exc,
                    )
                    try:
                        self._ensure_alive()
                    except Exception as spawn_exc:  # noqa: BLE001
                        logger.error("Stockfish respawn failed: %s", spawn_exc)
                        self._engine = None
                        return 0.5
                else:
                    logger.error("Stockfish eval failed on retry: %s", exc)
                    try:
                        self._ensure_alive()
                    except Exception:  # noqa: BLE001
                        self._engine = None
                    return 0.5
        return 0.5


def evaluate_position_stockfish(
    fen: str,
    stockfish_path: Optional[str] = None,
    depth: int = 15,
    timeout: float = 5.0,
) -> dict:
    """[back-compat] Evaluate a position, return raw Stockfish info dict.

    Kept for existing callers (evaluation/metrics.py, LUFFY trace
    generation). Returns keys: 'score' (centipawns relative to side to
    move), 'mate', 'bestmove', 'pv'. For RL rewards use compute_reward /
    RewardEngine instead (win-probability based, engine-reuse, crash-safe).
    """
    stockfish_path = stockfish_path or _find_stockfish()
    if stockfish_path is None:
        return {"score": 0, "mate": None, "bestmove": None, "pv": []}
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            board = chess.Board(fen)
            result = engine.analyse(
                board, chess.engine.Limit(depth=depth, time=timeout)
            )
        finally:
            try:
                engine.quit()
            except Exception:  # noqa: BLE001
                pass
        score_obj = result.get("score")
        if score_obj is None:
            return {"score": 0, "mate": None, "bestmove": None, "pv": []}
        # PovScore.relative is a property -> side-to-move perspective.
        rel = score_obj.relative if score_obj is not None else None
        pv = result.get("pv", [])
        bestmove = str(pv[0]) if pv else None
        if rel is not None and rel.is_mate():
            mate_in = rel.mate()
            return {
                "score": 0 if mate_in is None else (10000 if mate_in > 0 else -10000),
                "mate": mate_in,
                "bestmove": bestmove,
                "pv": [str(m) for m in pv],
            }
        cp = rel.score() if rel is not None else 0
        return {
            "score": cp if cp is not None else 0,
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
    eval_depth: int = 12,
    engine: Optional[RewardEngine] = None,
) -> dict:
    """Compute the win-probability reward Q(s,a) for a proposed move.

    r = P(mover wins | position after the move), in [0, 1].

    - Illegal / invalid: -1.0 (hard floor).
    - Checkmate by the mover: 1.0 (win prob after = 1).
    - Best move: high (approaches 1.0 in winning positions).
    - Blunder: much lower than the alternatives in the same group.

    Args:
        fen: FEN of the current position (side to move = the mover).
        proposed_move: Move in UCI notation (e.g., 'e2e4').
        stockfish_path: Path to Stockfish binary (auto-detect if None).
        eval_depth: Search depth for win-prob estimation.
        engine: Reusable RewardEngine (avoids process spawn per move).

    Returns:
        dict with total_reward (float), breakdown, legal, best info.
    """
    breakdown = {
        "legal_move": False,
        "winprob_before": None,
        "winprob_after": None,
        "reward_delta": 0.0,
        "checkmate": False,
        "cp_before": None,
        "cp_after": None,
    }
    try:
        board = chess.Board(fen)
    except ValueError:
        logger.warning("Invalid FEN: %s", fen)
        return {
            "total_reward": INVALID_FEN_REWARD,
            "breakdown": breakdown,
            "legal": False,
            "best_move": None,
            "cp_loss": None,
        }

    try:
        move = chess.Move.from_uci(proposed_move)
        is_legal = move in board.legal_moves
    except (ValueError, AttributeError):
        move = None
        is_legal = False

    if not is_legal or move is None:
        return {
            "total_reward": ILLEGAL_REWARD,
            "breakdown": {**breakdown, "legal_move": False},
            "legal": False,
            "best_move": None,
            "cp_loss": None,
        }

    breakdown["legal_move"] = True
    board.push(move)
    is_checkmate = board.is_checkmate()
    breakdown["checkmate"] = is_checkmate

    if engine is not None:
        winprob_before = engine.eval_win_prob(chess.Board(fen))
        winprob_after = engine.eval_win_prob(board)
    else:
        with RewardEngine(stockfish_path, eval_depth) as eng:
            winprob_before = eng.eval_win_prob(chess.Board(fen))
            winprob_after = eng.eval_win_prob(board)

    breakdown["winprob_before"] = winprob_before
    breakdown["winprob_after"] = winprob_after

    # Perspective conversion: winprob_after is relative to the NEW side to
    # move (the opponent). Mover's win prob after the move = 1 - that.
    mover_winprob_after = 1.0 - winprob_after
    breakdown["mover_winprob_after"] = mover_winprob_after

    # Chess-R1 style dense reward: Q(s,a) = mover's win probability after
    # the move, in [0,1]. Directly rewards winning; checkmate -> 1.0.
    reward = mover_winprob_after
    breakdown["reward_delta"] = mover_winprob_after - winprob_before

    return {
        "total_reward": reward,
        "breakdown": breakdown,
        "legal": True,
        "best_move": None,  # not computed; see RewardEngine note
        "cp_loss": None,
    }


def batch_compute_rewards(
    positions: list[tuple[str, str]],
    stockfish_path: Optional[str] = None,
    eval_depth: int = 12,
    num_workers: int = 4,
) -> list[dict]:
    """Compute rewards for a batch of (FEN, move) pairs.

    Uses a single persistent engine process for the whole batch
    (process-spawn overhead was the old bottleneck).
    `num_workers` is accepted for API compatibility; the persistent
    engine already removes the spawn cost, so parallelism is unnecessary.
    """
    stockfish_path = stockfish_path or _find_stockfish()
    if stockfish_path is None:
        logger.warning("Stockfish not found; returning neutral rewards.")
        return [
            {
                "total_reward": 0.0,
                "breakdown": {},
                "legal": True,
                "best_move": None,
                "cp_loss": None,
            }
            for _ in positions
        ]

    results = []
    with RewardEngine(stockfish_path, eval_depth) as engine:
        for fen, move in positions:
            results.append(
                compute_reward(fen, move, engine=engine)
            )
    return results
