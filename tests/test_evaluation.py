"""Tests for the reward function and evaluation metrics."""

import pytest

from qwen_chess_rl.evaluation.metrics import (
    legal_move_percentage,
    compute_win_rate,
    estimate_elo,
    puzzle_accuracy,
)


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
