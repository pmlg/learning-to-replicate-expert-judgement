"""Tests for multi-ply rollouts and episode-outcome reward.

Covers:
- compute_episode_reward: outcome/cutoff-eval -> scalar reward.
- play_rollout: illegal first move, immediate mate, mid-game illegal move,
  capped-mode ply counting, full-game termination bound.

The play_rollout tests need a real Stockfish binary (the opponent's moves
and cutoff evals go through it) — they're skipped if none is found.
"""

from __future__ import annotations

import chess
import pytest

import qwen_chess_rl.training.rl.rollout as rollout_mod
from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.reward import (
    CUTOFF_EVAL_CAP,
    DRAW_REWARD,
    DRAW_REWARD_WHEN_LOSING,
    LOSING_EVAL_THRESHOLD,
    LOSS_REWARD,
    WIN_REWARD,
    _find_stockfish,
    compute_episode_reward,
)
from qwen_chess_rl.training.rl.rollout import play_rollout

STOCKFISH_PATH = _find_stockfish()
requires_stockfish = pytest.mark.skipif(
    STOCKFISH_PATH is None, reason="Stockfish binary not found"
)

# Position one move from mate (Fool's Mate): 1. f3 e5 2. g4, black to move.
FOOLS_MATE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
FOOLS_MATE_MOVE = "d8h4"

# White to move, missing its queen against a full black army — clearly lost.
QUEEN_DOWN_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


class TestComputeEpisodeReward:
    def test_win(self):
        assert compute_episode_reward("win") == WIN_REWARD

    def test_loss(self):
        assert compute_episode_reward("loss") == LOSS_REWARD

    def test_draw(self):
        assert compute_episode_reward("draw") == DRAW_REWARD

    def test_capped_no_eval_is_neutral(self):
        assert compute_episode_reward(None, cutoff_eval=None) == 0.0

    def test_capped_positive_eval_scales_positively(self):
        # This is the bug being fixed relative to the old single-move
        # reward: being ahead should score positively, not just "0 penalty".
        assert compute_episode_reward(None, cutoff_eval=50) == pytest.approx(0.5)

    def test_capped_negative_eval_scales_negatively(self):
        assert compute_episode_reward(None, cutoff_eval=-300) == pytest.approx(-3.0)

    def test_capped_eval_is_capped_at_cutoff_eval_cap(self):
        assert compute_episode_reward(None, cutoff_eval=1_000_000) == CUTOFF_EVAL_CAP
        assert compute_episode_reward(None, cutoff_eval=-1_000_000) == -CUTOFF_EVAL_CAP

    def test_win_dominates_any_single_move_reward(self):
        # Winning must be worth strictly more than the largest score a
        # single move could get under compute_reward() (LEGAL_MOVE_BONUS +
        # CHECKMATE_BONUS = 15) — otherwise game outcome wouldn't actually
        # dominate the training signal.
        assert WIN_REWARD > 15.0

    def test_draw_from_balanced_or_winning_start_stays_discouraged(self):
        assert compute_episode_reward("draw", starting_eval=30.0) == DRAW_REWARD
        assert compute_episode_reward("draw", starting_eval=None) == DRAW_REWARD

    def test_draw_salvaged_from_losing_start_is_rewarded_well(self):
        assert compute_episode_reward("draw", starting_eval=-900.0) == DRAW_REWARD_WHEN_LOSING
        reward = compute_episode_reward("draw", starting_eval=LOSING_EVAL_THRESHOLD - 1)
        assert reward == DRAW_REWARD_WHEN_LOSING

    def test_draw_reward_boundary_is_strictly_below_threshold(self):
        # Exactly at the threshold is not "clearly lost" yet.
        assert compute_episode_reward("draw", starting_eval=LOSING_EVAL_THRESHOLD) == DRAW_REWARD

    def test_draw_reward_when_losing_beats_normal_draw_but_not_a_win(self):
        assert DRAW_REWARD_WHEN_LOSING > DRAW_REWARD
        assert DRAW_REWARD_WHEN_LOSING < WIN_REWARD

    def test_win_and_loss_are_unaffected_by_starting_eval(self):
        assert compute_episode_reward("win", starting_eval=-900.0) == WIN_REWARD
        assert compute_episode_reward("loss", starting_eval=-900.0) == LOSS_REWARD


class TestPlayRolloutStructure:
    """Structural tests that don't need Stockfish (illegal-move paths)."""

    def test_illegal_first_move_returns_none(self):
        cfg = GRPOConfig()
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e5", cfg, max_plies=4,
        )
        assert result is None

    def test_immediate_mate_on_first_move_is_a_win_no_opponent_turn(self):
        cfg = GRPOConfig()
        result = play_rollout(
            None, None, FOOLS_MATE_FEN, FOOLS_MATE_MOVE, cfg, max_plies=None,
        )
        assert result.outcome == "win"
        assert result.own_moves == [FOOLS_MATE_MOVE]
        assert result.num_plies == 1

    def test_mid_game_illegal_move_forfeits_as_a_loss(self, monkeypatch):
        monkeypatch.setattr(rollout_mod, "_generate_one_move", lambda *a, **k: "z9z9")
        cfg = GRPOConfig(opponent_depth=2)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=None,
        )
        assert result.outcome == "loss"


@requires_stockfish
class TestPlayRolloutWithStockfish:
    """Tests exercising the real Stockfish opponent and cutoff evaluation."""

    def test_capped_mode_stops_at_ply_cap(self, monkeypatch):
        def fake_move(model, tokenizer, fen, cfg):
            return str(list(chess.Board(fen).legal_moves)[0])

        monkeypatch.setattr(rollout_mod, "_generate_one_move", fake_move)
        cfg = GRPOConfig(opponent_depth=2, eval_depth=5)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=4,
        )
        assert result.outcome is None
        assert result.cutoff_eval is not None
        assert result.num_plies == 4
        assert len(result.own_moves) == 2
        assert len(result.own_prompts) == 2

    def test_full_game_terminates_with_a_real_outcome(self, monkeypatch):
        def fake_move(model, tokenizer, fen, cfg):
            return str(list(chess.Board(fen).legal_moves)[0])

        monkeypatch.setattr(rollout_mod, "_generate_one_move", fake_move)
        cfg = GRPOConfig(opponent_depth=2)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=None,
        )
        assert result.outcome in ("win", "loss", "draw")

    def test_opponent_move_is_always_legal(self):
        import chess.engine

        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        try:
            board = chess.Board()
            move = rollout_mod._select_opponent_move(board, engine, depth=3)
            assert move in board.legal_moves
        finally:
            engine.quit()

    def test_starting_eval_reflects_a_balanced_position(self):
        cfg = GRPOConfig(eval_depth=5)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=1,
        )
        assert result.starting_eval is not None
        assert abs(result.starting_eval) < 150

    def test_starting_eval_reflects_a_clearly_losing_position(self):
        cfg = GRPOConfig(eval_depth=8)
        result = play_rollout(
            None, None, QUEEN_DOWN_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=1,
        )
        assert result.starting_eval is not None
        assert result.starting_eval < LOSING_EVAL_THRESHOLD
