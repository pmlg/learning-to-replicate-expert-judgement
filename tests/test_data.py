"""Tests for the dataset preprocessing pipeline."""

import tempfile
from pathlib import Path

import pytest

from qwen_chess_rl.data.download import (
    _is_valid_fen,
    _quality_filter,
    split_train_val_test,
)
from qwen_chess_rl.data.dataset import extract_best_move, format_fen_prompt
from qwen_chess_rl.data.preprocess import (
    compute_position_difficulty,
    validate_dataset,
)


class TestFENValidation:
    """Test FEN string validation."""

    def test_valid_initial_position(self):
        assert _is_valid_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_valid_middlegame(self):
        assert _is_valid_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

    def test_invalid_fen_empty(self):
        assert not _is_valid_fen("")

    def test_invalid_fen_garbage(self):
        assert not _is_valid_fen("not a fen string at all")


class TestQualityFilter:
    """Test the quality filter for positions."""

    def test_good_position(self):
        assert _quality_filter({"cp": 50})

    def test_high_cp_loss(self):
        assert not _quality_filter({"cp": 300})

    def test_mate_in_few(self):
        assert _quality_filter({"mate": 3})

    def test_no_cp_no_mate(self):
        assert _quality_filter({})


class TestDatasetSplit:
    """Test train/val/test splitting."""

    def test_split_sizes(self):
        ds = type("FakeDS", (), {"__len__": lambda self: 1000, "select": lambda self, idx: idx})()
        train, val, test = split_train_val_test(ds, train_ratio=0.98, val_ratio=0.01, seed=42)
        assert len(train) == 980
        assert len(val) == 10
        assert len(test) == 10


class TestPromptFormat:
    """Test FEN prompt formatting."""

    def test_format_initial(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        prompt = format_fen_prompt(fen)
        assert "Position:" in prompt
        assert fen in prompt
        assert "Best move:" in prompt


class TestExtractBestMove:
    """Test move extraction from Lichess-style examples."""

    def test_from_line_pv(self):
        assert extract_best_move({"line": "b2e5 c7e5 e1d3"}) == "b2e5"

    def test_prefers_explicit_move(self):
        assert extract_best_move({"move": "e2e4", "line": "b2e5 c7e5"}) == "e2e4"

    def test_empty(self):
        assert extract_best_move({}) == ""


class TestPositionDifficulty:
    """Test position difficulty estimation."""

    def test_initial_position(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        diff = compute_position_difficulty(fen)
        assert 0 <= diff <= 1

    def test_empty_board(self):
        # Empty board should have low difficulty
        diff = compute_position_difficulty("8/8/8/8/8/8/8/8 w - - 0 1")
        assert diff >= 0

    def test_checkmate(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        diff = compute_position_difficulty(fen)
        assert 0 <= diff <= 1
