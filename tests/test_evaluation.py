"""Tests for the reward function and evaluation metrics."""

import chess
import pytest

from qwen_chess_rl.evaluation.metrics import (
    compute_win_rate,
    estimate_elo,
    legal_move_percentage,
    puzzle_accuracy,
)
from qwen_chess_rl.evaluation.stockfish_eval import play_game


class _ScriptedPlayer:
    """Plays a fixed sequence of UCI moves, then fails (returns None)."""

    def __init__(self, moves):
        self.moves = list(moves)

    def choose_move(self, board: chess.Board):
        if not self.moves:
            return None
        uci = self.moves.pop(0)
        return chess.Move.from_uci(uci)


class TestPlayGameIllegalMoveAttribution:
    """Whoever fails to produce a legal move forfeits — not always white.

    Regression test for a bug where any illegal-move termination was scored
    as a "loss" from white's perspective regardless of which side actually
    failed to move.
    """

    def test_black_failure_is_a_win_for_white(self):
        white = _ScriptedPlayer(["e2e4"])
        black = _ScriptedPlayer([])  # fails immediately on its first move
        result = play_game(white, black)
        assert result["termination"] == "illegal_move"
        assert result["result"] == "win"

    def test_white_failure_is_a_loss_for_white(self):
        white = _ScriptedPlayer([])  # fails immediately on its first move
        black = _ScriptedPlayer([])
        result = play_game(white, black)
        assert result["termination"] == "illegal_move"
        assert result["result"] == "loss"


class TestLegalMovePercentage:
    """Test legal move percentage computation."""

    def test_all_legal(self):
        positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e5"),
        ]
        assert legal_move_percentage(positions) == 1.0

    def test_empty(self):
        assert legal_move_percentage([]) == 0.0


class TestWinRate:
    """Test win/rate computation."""

    def test_all_wins(self):
        results = [{"result": "win"}, {"result": "win"}, {"result": "win"}]
        rates = compute_win_rate(results)
        assert rates["win_rate"] == 1.0
        assert rates["loss_rate"] == 0.0

    def test_mixed(self):
        results = [
            {"result": "win"},
            {"result": "draw"},
            {"result": "loss"},
        ]
        rates = compute_win_rate(results)
        assert rates["win_rate"] == pytest.approx(1.0 / 3)
        assert rates["draw_rate"] == pytest.approx(1.0 / 3)
        assert rates["loss_rate"] == pytest.approx(1.0 / 3)


class TestEloEstimation:
    """Test Elo estimation from match results."""

    def test_equal_opponent(self):
        # 50% score vs equal opponent → same Elo
        elo = estimate_elo(5, 0, 5, opponent_elo=1500)
        assert abs(elo - 1500) < 50

    def test_stronger_opponent(self):
        # 0% score vs 2000 Elo → much lower
        elo = estimate_elo(0, 0, 10, opponent_elo=2000)
        assert elo < 1800


class TestPuzzleAccuracy:
    """Test puzzle accuracy computation."""

    def test_all_correct(self):
        puzzles = [{"solution": "e2e4"}, {"solution": "d2d4"}]
        preds = ["e2e4", "d2d4"]
        assert puzzle_accuracy(puzzles, preds) == 1.0

    def test_some_correct(self):
        puzzles = [{"solution": "e2e4"}, {"solution": "d2d4"}]
        preds = ["e2e4", "g1f3"]
        assert puzzle_accuracy(puzzles, preds) == 0.5

    def test_empty(self):
        assert puzzle_accuracy([], []) == 0.0
