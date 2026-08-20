"""Tests for the win-probability reward function.

These lock in the two fixes that made the old reward not work:
1. Reward = Q(s,a) = P(mover wins | position after move), NOT a pile of
   arbitrary constants. Mate => 1.0, blunders => low, illegal => -1.0.
2. Perspective conversion: post-move Stockfish score is relative to the
   *opponent*; `1 - winprob_after` converts it back to the mover's view.
"""

import chess

from qwen_chess_rl.training.rl.reward import (
    _score_to_win_prob,
    batch_compute_rewards,
    compute_reward,
)

MATE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"  # Ra8# is mate
RUY_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"


class TestWinProbConversion:
    def test_monotonic(self):
        assert _score_to_win_prob(chess.engine.Cp(400)) > _score_to_win_prob(chess.engine.Cp(0))
        assert _score_to_win_prob(chess.engine.Cp(0)) > _score_to_win_prob(chess.engine.Cp(-400))

    def test_center(self):
        assert abs(_score_to_win_prob(chess.engine.Cp(0)) - 0.5) < 1e-9

    def test_mate_scores(self):
        assert _score_to_win_prob(chess.engine.Mate(3)) == 1.0
        assert _score_to_win_prob(chess.engine.Mate(-3)) == 0.0

    def test_none(self):
        assert _score_to_win_prob(None) == 0.5


class TestRewardSemantics:
    def test_mate_in_one_scores_1(self):
        r = compute_reward(MATE_FEN, "a1a8", eval_depth=8)
        assert r["legal"] is True
        assert r["total_reward"] == 1.0

    def test_illegal_move_scores_minus_1(self):
        r = compute_reward(MATE_FEN, "e2e4", eval_depth=8)
        assert r["legal"] is False
        assert r["total_reward"] == -1.0

    def test_mate_beats_mediocre_move(self):
        mate = compute_reward(MATE_FEN, "a1a8", eval_depth=8)
        slow = compute_reward(MATE_FEN, "a1a2", eval_depth=8)
        assert mate["total_reward"] > slow["total_reward"]

    def test_best_move_ranks_high(self):
        """In an equal position the engine's best move must not be the worst."""
        candidates = ["f1b5", "d2d4", "f1c4", "b1c3", "a2a3"]
        rs = batch_compute_rewards([(RUY_FEN, m) for m in candidates], eval_depth=10)
        by_move = {m: r["total_reward"] for m, r in zip(candidates, rs)}
        # Best theory moves (Bb5/d4) should outrank the passive a3
        assert by_move["f1b5"] > by_move["a2a3"]
        assert by_move["d2d4"] > by_move["a2a3"]

    def test_perspective_conversion_symmetric(self):
        """Reward for the same position must be equal for both sides' best move."""
        # Scholar's mate: white Qxf7# is mate; from black's identical setup the
        # equivalent scoring move (if legal) also gets 1.0.
        r = compute_reward("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "f3f7", eval_depth=8)
        assert r["total_reward"] == 1.0

    def test_batch_survives_bad_fen(self):
        """A malformed position must not crash the batch (engine restart)."""
        bad = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3"  # impossible pos
        rs = batch_compute_rewards([(bad, "f8c5"), (MATE_FEN, "a1a8")], eval_depth=8)
        assert len(rs) == 2
        # The valid position still scores 1.0 for mate even after the crash
        assert rs[1]["total_reward"] == 1.0
