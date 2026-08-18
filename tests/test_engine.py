"""Tests for the classical alpha-beta chess engine."""

import chess
import pytest

from qwen_chess_rl.engine.classical import choose_move, evaluate, evaluate_move


class TestTimeoutSafety:
    """A search interrupted mid-recursion must not leave the board mutated.

    Regression test for a bug where ``_Timeout`` unwound the recursive
    negamax/quiescence stack past ``board.push()`` calls without their
    matching ``pop()``, corrupting the caller's board object so a
    subsequently "legal" move was actually illegal in the real position.
    """

    def test_board_unchanged_after_forced_timeout(self):
        fen = "r1bqkb1r/1ppnnpp1/B3p2p/2N5/3P4/2P2N2/PP3PPP/R1BQK2R b KQkq - 0 8"
        board = chess.Board(fen)
        move = choose_move(board, max_depth=10, time_limit=0.05)
        assert board.fen() == fen
        assert len(board.move_stack) == 0
        assert move is not None
        assert move in board.legal_moves

# Small budgets keep the suite fast while still exercising a few plies.
TEST_DEPTH = 3
TEST_TIME = 2.0


class TestMateDetection:
    """The engine must find and prefer an immediate checkmate."""

    def test_back_rank_mate_in_one(self):
        # White rook delivers Re1-e8#; the Black king is boxed by its own pawns.
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        move = choose_move(board, max_depth=TEST_DEPTH, time_limit=TEST_TIME)
        assert move == chess.Move.from_uci("e1e8")
        board.push(move)
        assert board.is_checkmate()

    def test_queen_mate_in_one(self):
        # Qf7-g7 / here Qf7-f8 style: pick any mating move and confirm it mates.
        board = chess.Board("6k1/4Q1pp/8/8/8/8/5PPP/6K1 w - - 0 1")
        move = choose_move(board, max_depth=TEST_DEPTH, time_limit=TEST_TIME)
        board.push(move)
        assert board.is_checkmate()


class TestBlunderAvoidance:
    """The engine must not throw away a queen for nothing."""

    def test_does_not_hang_queen(self):
        # Qxd5 loses the queen to cxd5/exd5; many safe alternatives exist.
        board = chess.Board("4k3/8/2p1p3/3p4/8/8/P7/3QK3 w - - 0 1")
        move = choose_move(board, max_depth=TEST_DEPTH, time_limit=TEST_TIME)
        assert move != chess.Move.from_uci("d1d5")


class TestAlwaysLegal:
    """choose_move must always return a legal move for playable positions."""

    @pytest.mark.parametrize(
        "fen",
        [
            chess.STARTING_FEN,
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            "8/2k5/8/8/8/5K2/6Q1/8 w - - 0 1",
        ],
    )
    def test_returns_legal_move(self, fen):
        board = chess.Board(fen)
        move = choose_move(board, max_depth=TEST_DEPTH, time_limit=TEST_TIME)
        assert move is not None
        assert move in board.legal_moves


class TestEvaluateMove:
    """evaluate_move must see tactics that only appear in the opponent's reply."""

    # 1. e4 e5 2. Qh5 Nc6 3. Bc4, Black to move. a6 ignores the Qxf7# threat;
    # a static eval of the position right after a6 has no way to know that —
    # it only shows up once the reply is actually searched.
    SCHOLARS_MATE_SETUP = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"

    def test_static_eval_is_blind_to_the_mate_in_one_reply(self):
        board = chess.Board(self.SCHOLARS_MATE_SETUP)
        blunder = chess.Move.from_uci("a7a6")
        sound = chess.Move.from_uci("g7g6")  # covers f7

        def static_score(move):
            board.push(move)
            score = evaluate(board)
            board.pop()
            return score if board.turn == chess.WHITE else -score

        # The static eval actually rates the losing move *higher* — proof
        # a one-ply-deep comparison can't be used to catch this blunder.
        assert static_score(blunder) > static_score(sound)

    def test_searched_eval_catches_the_same_blunder(self):
        board = chess.Board(self.SCHOLARS_MATE_SETUP)
        blunder = chess.Move.from_uci("a7a6")
        sound = chess.Move.from_uci("g7g6")

        blunder_score = evaluate_move(board, blunder, max_depth=3, time_limit=1.0)
        sound_score = evaluate_move(board, sound, max_depth=3, time_limit=1.0)

        assert sound_score > blunder_score
        assert board.fen() == self.SCHOLARS_MATE_SETUP  # board restored

    def test_does_not_mutate_board(self):
        board = chess.Board(self.SCHOLARS_MATE_SETUP)
        fen_before = board.fen()
        evaluate_move(board, chess.Move.from_uci("g7g6"), max_depth=2, time_limit=0.5)
        assert board.fen() == fen_before


class TestNoLegalMoves:
    """Terminal positions must return None rather than crash."""

    def test_checkmate_returns_none(self):
        # Fool's mate final position: Black is checkmated, no legal moves.
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()
        assert choose_move(board, max_depth=TEST_DEPTH, time_limit=TEST_TIME) is None

    def test_stalemate_returns_none(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert board.is_stalemate()
        assert choose_move(board, max_depth=TEST_DEPTH, time_limit=TEST_TIME) is None
