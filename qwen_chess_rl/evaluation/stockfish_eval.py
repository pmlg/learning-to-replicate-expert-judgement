"""Automated evaluation of chess model playing strength via self-play and benchmarks."""

import logging
import random
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import numpy as np
import torch

from qwen_chess_rl.evaluation.metrics import (
    compute_all_metrics,
    compute_centipawn_loss,
    legal_move_percentage,
    estimate_elo,
)
from qwen_chess_rl.training.rl.reward import _find_stockfish

logger = logging.getLogger(__name__)


class RandomPlayer:
    """A random chess player (baseline for benchmarking)."""

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None


class StockfishPlayer:
    """Stockfish-based chess player at a fixed depth."""

    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 8):
        self.stockfish_path = stockfish_path or _find_stockfish()
        self.depth = depth
        self.engine = None

    def start(self):
        if not self.stockfish_path:
            raise RuntimeError("Stockfish binary not found; install stockfish or set STOCKFISH_PATH")
        try:
            # python-chess >= 1.0 returns SimpleEngine directly (not a transport tuple)
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to start Stockfish at {self.stockfish_path}: {exc}") from exc

    def stop(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        if not self.engine:
            return None
        try:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            return result.move
        except Exception:
            return None


class HuggingFaceModelPlayer:
    """A chess player backed by a HuggingFace model."""

    def __init__(self, model, tokenizer, temperature: float = 0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
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
    elif ply >= max_moves - 1:
        result = "draw"
        termination = "max_moves"
    else:
        result = "loss" if termination == "illegal_move" else "draw"

    return {
        "result": result,
        "moves": move_history,
        "final_fen": board.fen(),
        "termination": termination,
    }


def evaluate_vs_stockfish(
    model,
    tokenizer,
    num_games: int = 20,
    stockfish_depth: int = 8,
    temperature: float = 0.1,
) -> dict:
    """Evaluate model playing strength against Stockfish.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        num_games: Number of games to play.
        stockfish_depth: Stockfish analysis depth.
        temperature: Model sampling temperature.

    Returns:
        Dictionary with win/draw/loss rates and average game stats.
    """
    sf_player = StockfishPlayer(depth=stockfish_depth)
    sf_player.start()
    hf_player = HuggingFaceModelPlayer(model, tokenizer, temperature=temperature)

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
) -> dict:
    """Evaluate model against a random player.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        num_games: Number of games.
        temperature: Sampling temperature.

    Returns:
        Win/draw/loss statistics.
    """
    random_player = RandomPlayer()
    hf_player = HuggingFaceModelPlayer(model, tokenizer, temperature=temperature)

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
