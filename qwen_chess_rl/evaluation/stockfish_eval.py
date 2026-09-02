"""Automated evaluation of chess model playing strength via self-play and benchmarks."""

import logging
import random

import chess
import chess.engine
import numpy as np

from qwen_chess_rl.evaluation.metrics import (
    estimate_elo,
)
from qwen_chess_rl.training.rl.reward import _find_stockfish

logger = logging.getLogger(__name__)


class RandomPlayer:
    """A random chess player (baseline for benchmarking)."""

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None


class StockfishPlayer:
    """Stockfish-based chess player at a fixed depth."""

    def __init__(
        self,
        stockfish_path: str | None = None,
        depth: int = 8,
        skill_level: int | None = None,
    ):
        self.stockfish_path = stockfish_path or _find_stockfish()
        self.depth = depth
        self.skill_level = skill_level
        self.engine = None

    def start(self):
        if not self.stockfish_path:
            raise RuntimeError("Stockfish binary not found; install stockfish or set STOCKFISH_PATH")
        try:
            # python-chess >= 1.0 returns SimpleEngine directly (not a transport tuple)
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to start Stockfish at {self.stockfish_path}: {exc}") from exc
        # Skill Level (0-20, 0 = weakest) lets us play a deliberately weaker
        # opponent; left untouched when None so depth-only behavior is unchanged.
        if self.skill_level is not None:
            self.engine.configure({"Skill Level": self.skill_level})

    def stop(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        if not self.engine:
            return None
        try:
            # time= activates python-chess's hang safety-net (see
            # rollout.py/reward.py) — a depth-only Limit can otherwise
            # block forever on a pathological position.
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth, time=5.0))
            return result.move
        except Exception:
            return None


class HuggingFaceModelPlayer:
    """A chess player backed by a HuggingFace model."""

    def __init__(self, model, tokenizer, temperature: float = 0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        """Get the model's best move for the current board position.

        Args:
            board: Current chess board.

        Returns:
            Chosen move in UCI format, or None if generation fails.
        """
        from qwen_chess_rl.inference import choose_uci_move

        uci = choose_uci_move(
            self.model,
            self.tokenizer,
            board,
            temperature=self.temperature,
        )
        if not uci:
            return None
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        return move if move in board.legal_moves else None


class SearchPlayer:
    """A chess player backed by the classical alpha-beta search engine."""

    def __init__(self, max_depth: int = 4, time_limit: float | None = 1.0):
        self.max_depth = max_depth
        self.time_limit = time_limit

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        from qwen_chess_rl.engine.classical import choose_move

        return choose_move(board, max_depth=self.max_depth, time_limit=self.time_limit)


class HybridPlayer:
    """LLM move suggestion vetted by classical search.

    Asks the language model for a move and keeps it — preserving the model's
    style — only when the search agrees it is within ``margin`` centipawns of
    the search's own best move. Otherwise (or when the LLM proposes an
    illegal/unparseable move) it defers to the search's best move as a tactical
    safety net.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.1,
        max_depth: int = 4,
        time_limit: float | None = 1.0,
        margin: float = 50.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.margin = margin

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        from qwen_chess_rl.engine.classical import choose_move

        search_best = choose_move(board, max_depth=self.max_depth, time_limit=self.time_limit)
        if search_best is None:
            return None

        llm_move = self._llm_move(board)
        if llm_move is None or llm_move == search_best:
            return search_best

        if self._within_margin(board, llm_move, search_best):
            return llm_move
        return search_best

    def _llm_move(self, board: chess.Board) -> chess.Move | None:
        from qwen_chess_rl.inference import choose_uci_move

        uci = choose_uci_move(self.model, self.tokenizer, board, temperature=self.temperature)
        if not uci:
            return None
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        return move if move in board.legal_moves else None

    def _within_margin(
        self,
        board: chess.Board,
        llm_move: chess.Move,
        search_best: chess.Move,
    ) -> bool:
        from qwen_chess_rl.engine.classical import evaluate

        def move_score(move: chess.Move) -> int:
            board.push(move)
            # evaluate() is White-perspective; negate for Black so a higher
            # score always means "better for the side that just moved".
            score = evaluate(board)
            board.pop()
            return score if board.turn == chess.WHITE else -score

        return move_score(search_best) - move_score(llm_move) <= self.margin


def play_game(
    player_white,
    player_black,
    max_moves: int = 200,
    time_limit: float = 2.0,
) -> dict:
    """Play a single chess game between two players.

    Args:
        player_white: Player playing white (must have choose_move method).
        player_black: Player playing black.
        max_moves: Maximum plies before declaring a draw.
        time_limit: Time limit per move in seconds (not enforced for simple players).

    Returns:
        Dictionary with game result:
            - 'result': 'win'/'draw'/'loss' from white's perspective.
            - 'moves': List of UCI moves played.
            - 'final_fen': Final board FEN.
            - 'termination': Reason for game end.
    """
    board = chess.Board()
    move_history = []
    termination = "unknown"
    ply = 0

    for ply in range(max_moves):
        if board.is_game_over():
            termination = "game_over"
            break

        player = player_white if ply % 2 == 0 else player_black
        move = player.choose_move(board)

        if move is None:
            termination = "illegal_move"
            break

        if move not in board.legal_moves:
            termination = "illegal_move"
            break

        board.push(move)
        move_history.append(move.uci())

    if board.is_checkmate():
        result = "win" if board.turn == chess.BLACK else "loss"
    elif board.is_stalemate() or board.is_insufficient_material():
        result = "draw"
    elif board.can_claim_threefold_repetition():
        result = "draw"
    elif termination == "illegal_move":
        # Whichever side was on move when it failed to produce a legal move
        # forfeits — white was on move at even plies, black at odd plies.
        result = "loss" if ply % 2 == 0 else "win"
    elif ply >= max_moves - 1:
        result = "draw"
        termination = "max_moves"
    else:
        result = "draw"

    return {
        "result": result,
        "moves": move_history,
        "final_fen": board.fen(),
        "termination": termination,
    }


def _build_player(
    model,
    tokenizer,
    temperature: float,
    engine: str,
    search_depth: int,
    search_time_limit: float | None,
    hybrid_margin: float,
):
    """Build the policy-side player for a given ``engine`` mode.

    ``"llm"`` is the raw fine-tuned model (no look-ahead); ``"search"`` is the
    classical alpha-beta engine alone; ``"hybrid"`` lets the LLM's move stand
    only when the search agrees it isn't a blunder (see ``HybridPlayer``).
    """
    if engine == "search":
        return SearchPlayer(max_depth=search_depth, time_limit=search_time_limit)
    if engine == "hybrid":
        return HybridPlayer(
            model,
            tokenizer,
            temperature=temperature,
            max_depth=search_depth,
            time_limit=search_time_limit,
            margin=hybrid_margin,
        )
    if engine != "llm":
        raise ValueError(f"Unknown engine: {engine!r} (expected 'llm', 'search', or 'hybrid')")
    return HuggingFaceModelPlayer(model, tokenizer, temperature=temperature)


def evaluate_vs_stockfish(
    model,
    tokenizer,
    num_games: int = 20,
    stockfish_depth: int = 8,
    stockfish_skill_level: int | None = None,
    temperature: float = 0.1,
    engine: str = "llm",
    search_depth: int = 4,
    search_time_limit: float | None = 1.0,
    hybrid_margin: float = 50.0,
) -> dict:
    """Evaluate model playing strength against Stockfish.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        num_games: Number of games to play.
        stockfish_depth: Stockfish analysis depth.
        stockfish_skill_level: Stockfish "Skill Level" (0-20, 0 = weakest),
            or None to leave Stockfish at full strength for its depth.
        temperature: Model sampling temperature.
        engine: Which policy plays: ``"llm"`` (raw model), ``"search"``
            (classical alpha-beta only), or ``"hybrid"`` (LLM move vetted by
            search — see ``HybridPlayer``).
        search_depth: Alpha-beta depth for ``"search"``/``"hybrid"``.
        search_time_limit: Per-move wall-clock budget (seconds) for
            ``"search"``/``"hybrid"``.
        hybrid_margin: Centipawn margin for ``"hybrid"`` — see ``HybridPlayer``.

    Returns:
        Dictionary with win/draw/loss rates and average game stats.
    """
    sf_player = StockfishPlayer(depth=stockfish_depth, skill_level=stockfish_skill_level)
    sf_player.start()
    hf_player = _build_player(
        model, tokenizer, temperature, engine, search_depth, search_time_limit, hybrid_margin
    )

    results = []
    total_moves = []

    for game_idx in range(num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            result = play_game(hf_player, sf_player)
        else:
            result = play_game(sf_player, hf_player)
            # Flip result perspective for consistency
            if result["result"] == "win":
                result["result"] = "loss"
            elif result["result"] == "loss":
                result["result"] = "win"

        results.append(result)
        total_moves.append(len(result["moves"]))
        logger.info(
            "Game %s/%s: %s (%s moves)",
            game_idx + 1, num_games,
            result["result"], len(result["moves"]),
        )

    sf_player.stop()

    wins = sum(1 for r in results if r["result"] == "win")
    draws = sum(1 for r in results if r["result"] == "draw")
    losses = sum(1 for r in results if r["result"] == "loss")

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / num_games if num_games else 0,
        "draw_rate": draws / num_games if num_games else 0,
        "loss_rate": losses / num_games if num_games else 0,
        "avg_game_length": float(np.mean(total_moves)) if total_moves else 0,
        "total_games": num_games,
        "estimated_elo": estimate_elo(wins, draws, losses, 2000),
    }


def evaluate_vs_random(
    model,
    tokenizer,
    num_games: int = 10,
    temperature: float = 0.1,
    engine: str = "llm",
    search_depth: int = 4,
    search_time_limit: float | None = 1.0,
    hybrid_margin: float = 50.0,
) -> dict:
    """Evaluate model against a random player.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        num_games: Number of games.
        temperature: Sampling temperature.
        engine: Which policy plays — see ``evaluate_vs_stockfish``.
        search_depth: Alpha-beta depth for "search"/"hybrid".
        search_time_limit: Per-move wall-clock budget for "search"/"hybrid".
        hybrid_margin: Centipawn margin for "hybrid".

    Returns:
        Win/draw/loss statistics.
    """
    random_player = RandomPlayer()
    hf_player = _build_player(
        model, tokenizer, temperature, engine, search_depth, search_time_limit, hybrid_margin
    )

    results = []
    for _ in range(num_games):
        result = play_game(hf_player, random_player)
        results.append(result)

    wins = sum(1 for r in results if r["result"] == "win")
    draws = sum(1 for r in results if r["result"] == "draw")
    losses = sum(1 for r in results if r["result"] == "loss")

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / num_games if num_games else 0,
        "total_games": num_games,
    }
