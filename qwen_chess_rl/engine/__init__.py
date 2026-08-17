"""Classical chess search engine (alpha-beta) used as a tactical safety net."""

from qwen_chess_rl.engine.classical import choose_move, evaluate

__all__ = ["choose_move", "evaluate"]
