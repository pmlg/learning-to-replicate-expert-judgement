"""Multi-ply rollouts against a reduced-strength Stockfish opponent.

The reward for a GRPO candidate is tied to the eventual game outcome
(win/loss/draw), not just the quality of one isolated move — see
:func:`qwen_chess_rl.training.rl.reward.compute_episode_reward`. A rollout
either plays to a natural conclusion (checkmate, stalemate, insufficient
material, repetition, 50-move rule) or is truncated at a ply cap, in which
case Stockfish's eval at the cutoff stands in for the outcome.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import chess
import chess.engine

from qwen_chess_rl.data.dataset import format_fen_prompt
from qwen_chess_rl.engine.classical import choose_move, evaluate_move
from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.reward import _find_stockfish

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """A played-out trajectory starting from the model's first move.

    ``own_prompts``/``own_moves`` are aligned by index: ``own_prompts[i]``
    is the position the model saw, ``own_moves[i]`` is the move it played
    there. Both include the very first (fen, first_move) pair the caller
    supplied — the model's own moves are exactly what the GRPO loss should
    credit with the trajectory's outcome.
    """

    own_prompts: list[str] = field(default_factory=list)
    own_moves: list[str] = field(default_factory=list)
    outcome: str | None = None  # 'win' | 'loss' | 'draw', or None if capped
    cutoff_eval: float | None = None  # centipawns, policy's perspective; only set if capped
    starting_eval: float | None = None  # centipawns, policy's perspective, before any moves
    num_plies: int = 0


def _score_relative_to_policy(
    score_obj,
    board: chess.Board,
    policy_color: bool,
) -> float:
    """Convert a python-chess PovScore into centipawns from the policy's view."""
    relative = score_obj.relative
    if relative.is_mate():
        mate_in = relative.mate()
        raw = 0.0 if mate_in is None else (10000.0 if mate_in > 0 else -10000.0)
    else:
        raw = float(relative.score() or 0)
    # `relative` is from the perspective of whoever is to move right now.
    return raw if board.turn == policy_color else -raw


def _vet_move(
    board: chess.Board,
    llm_move: chess.Move,
    cfg: GRPOConfig,
) -> chess.Move:
    """Let a shallow classical search veto a legal-but-losing LLM move.

    Keeps the LLM's move only when a *searched* evaluation (not a bare
    static one — see :func:`qwen_chess_rl.engine.classical.evaluate_move`)
    puts it within ``rollout_vet_margin`` centipawns of the search's own
    best move; otherwise substitutes the search's move so the trajectory's
    reward can teach the policy to prefer it. Comparing static evals right
    after each candidate (as HybridPlayer's margin check does) misses any
    tactic that only shows up in the opponent's reply — e.g. a move that
    hangs mate next ply looks fine until searched one ply deeper.
    """
    search_best = choose_move(
        board, max_depth=cfg.rollout_vet_depth, time_limit=cfg.rollout_vet_time_limit,
    )
    if search_best is None or search_best == llm_move:
        return llm_move

    llm_score = evaluate_move(board, llm_move, cfg.rollout_vet_depth, cfg.rollout_vet_time_limit)
    best_score = evaluate_move(board, search_best, cfg.rollout_vet_depth, cfg.rollout_vet_time_limit)

    if best_score - llm_score <= cfg.rollout_vet_margin:
        return llm_move
    return search_best


def _select_opponent_move(
    board: chess.Board,
    engine: chess.engine.SimpleEngine | None,
    depth: int,
) -> chess.Move:
    """Pick the opponent's move via a shallow, deliberately weak Stockfish search.

    Falls back to a random legal move if the engine is unavailable or errors —
    keeps rollouts non-fatal rather than crashing training on an engine hiccup.
    """
    if engine is not None:
        try:
            # time= is required, not just depth= — python-chess's own hang
            # safety-net (SimpleEngine.timeout, default 10s) only activates
            # when Limit.time is set. A depth-only Limit on a pathological
            # position can otherwise block forever with no exception ever
            # raised, silently freezing the whole training run.
            play_result = engine.play(board, chess.engine.Limit(depth=depth, time=5.0))
            if play_result.move is not None:
                return play_result.move
        except Exception as exc:
            logger.error("Opponent move selection failed, using random move: %s", exc)
    return random.choice(list(board.legal_moves))


def _cutoff_eval(
    board: chess.Board,
    engine: chess.engine.SimpleEngine | None,
    policy_color: bool,
    depth: int,
) -> float:
    """Evaluate the truncation position, from the policy's perspective."""
    if engine is None:
        return 0.0
    try:
        # See _select_opponent_move — time= activates python-chess's hang
        # safety-net; depth-only Limit can otherwise block forever.
        info = engine.analyse(board, chess.engine.Limit(depth=depth, time=5.0))
        score_obj = info.get("score")
        if score_obj is None:
            return 0.0
        return _score_relative_to_policy(score_obj, board, policy_color)
    except Exception as exc:
        logger.error("Cutoff evaluation failed: %s", exc)
        return 0.0


def play_rollout(
    model,
    tokenizer,
    fen: str,
    first_move: str,
    cfg: GRPOConfig,
    stockfish_path: str | None = None,
    max_plies: int | None = None,
) -> RolloutResult | None:
    """Play out a game starting with ``first_move`` from ``fen``.

    The model plays whichever side is to move at ``fen`` throughout the
    rollout; a shallow Stockfish (``cfg.opponent_depth``) plays the other
    side. An illegal move by the model at any point — including the
    caller-supplied ``first_move`` — ends the rollout as a loss, except for
    the very first move: an illegal first move returns None so the caller
    can apply its existing illegal-move penalty instead.

    Args:
        model: Current policy model, used to generate the model's own
            moves after the first.
        tokenizer: HuggingFace tokenizer.
        fen: Starting position.
        first_move: The model's first move, already sampled by the caller
            (e.g. via generate_moves) — not re-sampled here.
        cfg: GRPO configuration.
        stockfish_path: Path to Stockfish (auto-detected if None).
        max_plies: Ply cap (both sides combined), or None to play to a
            natural conclusion.

    Returns:
        RolloutResult, or None if ``first_move`` is illegal.
    """
    stockfish_path = stockfish_path or _find_stockfish()

    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(first_move)
    except ValueError:
        move = None
    if move is None or move not in board.legal_moves:
        return None

    policy_color = board.turn

    engine = None
    if stockfish_path is not None:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as exc:
            logger.error("Failed to start Stockfish for rollout: %s", exc)

    # Eval the position *before* any move, to tell "salvaged a draw from a
    # losing spot" apart from "settled for a draw from an equal/winning one".
    starting_eval = _cutoff_eval(board, engine, policy_color, cfg.eval_depth)

    result = RolloutResult(
        own_prompts=[format_fen_prompt(fen)],
        own_moves=[first_move],
        starting_eval=starting_eval,
    )
    board.push(move)
    result.num_plies = 1

    try:
        while True:
            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    result.outcome = "draw"
                else:
                    result.outcome = "win" if outcome.winner == policy_color else "loss"
                return result

            if max_plies is not None and result.num_plies >= max_plies:
                result.cutoff_eval = _cutoff_eval(board, engine, policy_color, cfg.eval_depth)
                return result

            if board.turn == policy_color:
                fen_now = board.fen()
                candidates = _generate_one_move(model, tokenizer, fen_now, cfg)
                try:
                    next_move = chess.Move.from_uci(candidates)
                except ValueError:
                    next_move = None
                if next_move is None or next_move not in board.legal_moves:
                    # Illegal mid-game move forfeits the position.
                    result.outcome = "loss"
                    return result
                if cfg.rollout_vet_enabled:
                    vetted_move = _vet_move(board, next_move, cfg)
                    if vetted_move != next_move:
                        next_move = vetted_move
                        candidates = next_move.uci()
                result.own_prompts.append(format_fen_prompt(fen_now))
                result.own_moves.append(candidates)
                board.push(next_move)
            else:
                opponent_move = _select_opponent_move(board, engine, cfg.opponent_depth)
                board.push(opponent_move)

            result.num_plies += 1
    finally:
        # The engine may have already died mid-rollout (e.g. a Stockfish
        # crash) — every other engine call above already tolerates that,
        # but engine.quit() itself raises EngineTerminatedError on a dead
        # engine, which was left unhandled and took down the entire
        # training process on an incidental engine crash.
        if engine is not None:
            try:
                engine.quit()
            except Exception as exc:
                logger.warning("Failed to cleanly shut down rollout engine: %s", exc)


def _generate_one_move(model, tokenizer, fen: str, cfg: GRPOConfig) -> str:
    """Sample a single move from the policy for one position."""
    from qwen_chess_rl.training.rl.grpo import generate_moves

    moves = generate_moves(model, tokenizer, [fen], cfg, num_generations=1)
    return moves[0][0] if moves and moves[0] else ""
