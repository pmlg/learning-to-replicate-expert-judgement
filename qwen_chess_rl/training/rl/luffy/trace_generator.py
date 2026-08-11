"""Trace generator — queries Stockfish to build off-policy action traces.

Given a set of FEN positions, the generator:
1. Asks Stockfish for the best move + evaluation at a configurable depth.
2. Records the behaviour log-prob (approximated via a softmax over the
   Stockfish principal-variation evaluations).
3. Applies quality filters (engine depth, evaluation threshold, mate).
4. Returns a list of :class:`OffPolicyTrace` objects.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import chess
import chess.engine

from qwen_chess_rl.training.rl.luffy.schema import OffPolicyTrace
from qwen_chess_rl.training.rl.reward import _find_stockfish, evaluate_position_stockfish

logger = logging.getLogger(__name__)


class TraceGenerator:
    """Generate Stockfish-guided off-policy traces for LUFFY training.

    Parameters:
        stockfish_path: Path to the Stockfish binary (auto-detected if None).
        engine_depth: Search depth for Stockfish analysis.
        min_eval_cp: Minimum absolute evaluation (centipawns) to accept the
            trace — filters out positions where the best move is unclear.
        eval_scale: Temperature-like parameter for the behaviour-log-prob
            softmax over PV evaluation scores.
    """

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        engine_depth: int = 18,
        min_eval_cp: int = 0,
        eval_scale: float = 100.0,
    ):
        self.stockfish_path = stockfish_path or _find_stockfish()
        self.engine_depth = engine_depth
        self.min_eval_cp = min_eval_cp
        self.eval_scale = eval_scale

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_trace(self, fen: str) -> Optional[OffPolicyTrace]:
        """Generate a single off-policy trace for *fen*.

        Returns ``None`` if the trace fails quality filters or Stockfish
        is unavailable.
        """
        eval_result = evaluate_position_stockfish(
            fen,
            stockfish_path=self.stockfish_path,
            depth=self.engine_depth,
        )

        best_move = eval_result.get("bestmove")
        if best_move is None:
            logger.debug("No best move for FEN %s — skipping.", fen)
            return None

        score = eval_result.get("score", 0)
        mate = eval_result.get("mate")
        pv = eval_result.get("pv", [])

        # Quality filter: skip if eval magnitude below threshold and no mate
        if mate is None and abs(score) < self.min_eval_cp:
            logger.debug("FEN %s below eval threshold (%d cp) — skipping.", fen, score)
            return None

        # Approximate behaviour log-prob via softmax over PV scores.
        # We compute this from the principal-variation line that Stockfish
        # returns.  If only one move is available the log-prob is 0.0 (log 1).
        behavior_log_prob = self._compute_behavior_log_prob(fen, pv)

        return OffPolicyTrace(
            fen=fen,
            action=best_move,
            behavior_log_prob=behavior_log_prob,
            stockfish_eval=score,
            stockfish_pv=pv,
            engine_depth=self.engine_depth,
            mate=mate,
            source="stockfish",
        )

    def generate_traces(
        self,
        fens: list[str],
        batch_size: int = 32,
    ) -> list[OffPolicyTrace]:
        """Generate traces for a list of FEN positions.

        Args:
            fens: List of FEN strings.
            batch_size: Number of positions to process before logging
                progress (purely for logging; Stockfish is queried
                sequentially).

        Returns:
            List of valid :class:`OffPolicyTrace` objects (low-quality
            positions are filtered out).
        """
        traces: list[OffPolicyTrace] = []
        for idx, fen in enumerate(fens):
            trace = self.generate_trace(fen)
            if trace is not None:
                traces.append(trace)

            if (idx + 1) % batch_size == 0:
                logger.info(
                    "Generated %d traces from %d positions (%.1f%% pass rate)",
                    len(traces),
                    idx + 1,
                    100.0 * len(traces) / (idx + 1),
                )

        logger.info(
            "Trace generation complete: %d/%d positions (%.1f%% pass rate)",
            len(traces),
            len(fens),
            100.0 * len(traces) / max(len(fens), 1),
        )
        return traces

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_behavior_log_prob(
        self,
        fen: str,
        pv: list[str],
    ) -> float:
        """Approximate the behaviour log-probability of the best move.

        We evaluate the top-k legal moves using Stockfish and compute a
        softmax over centipawn scores. The log-prob of the best move is
        returned.  This is a rough approximation — for a true behaviour
        policy we would need Stockfish's internal move ordering, but this
        captures the key insight: Stockfish is highly confident in its
        top choice (log-prob near 0) in tactical positions and less
        confident in balanced positions.
        """
        if not pv or len(pv) == 0:
            return 0.0

        best_move = pv[0]

        try:
            board = chess.Board(fen)
        except ValueError:
            return 0.0

        # Get evaluations for the top few legal moves
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 1:
            return 0.0  # log(1) = 0 — only one legal move

        # Evaluate each legal move (limit to top-K for efficiency)
        # Use a single persistent engine for all evaluations
        engine = None
        try:
            if self.stockfish_path:
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception:
            return 0.0

        if engine is None:
            return 0.0

        scores: list[float] = []
        try:
            for move in legal_moves[:8]:  # Top-K for efficiency
                board.push(move)
                info = engine.analyse(board, chess.engine.Limit(depth=self.engine_depth))
                board.pop()
                score_obj = info.get("score")
                if score_obj is None:
                    scores.append(0.0)
                    continue
                # PovScore has no .score()/.mate() — use .relative (relative
                # to the side to move, i.e. the opponent after our push).
                relative_score = score_obj.relative
                if relative_score.is_mate():
                    mate_val = relative_score.mate()
                    # mate_val is from the opponent's point of view: positive
                    # means the opponent mates (bad for us), so flip the sign.
                    scores.append(-10000.0 if (mate_val and mate_val > 0) else 10000.0)
                else:
                    # Negate because after our move it's opponent's perspective
                    scores.append(-float(relative_score.score()))
        except Exception as exc:
            logger.debug("Behaviour log-prob computation failed: %s", exc)
            return 0.0
        finally:
            engine.quit()

        if not scores:
            return 0.0

        # Softmax over scores with temperature scaling
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / self.eval_scale) for s in scores]
        total = sum(exp_scores)

        if total == 0:
            return 0.0

        # Probability of the best move (highest score)
        best_prob = max(exp_scores) / total
        return math.log(max(best_prob, 1e-10))


# ---------------------------------------------------------------------- #
# Standalone script entry point
# ---------------------------------------------------------------------- #

def main():
    """CLI entry point for batch trace generation.

    Usage::

        python -m qwen_chess_rl.training.rl.luffy.trace_generator \
            --input data/fens.txt \
            --output data/traces.jsonl \
            --depth 18
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate LUFFY off-policy traces")
    parser.add_argument("--input", type=str, required=True, help="Text file with one FEN per line")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish search depth")
    parser.add_argument("--min-eval", type=int, default=0, help="Minimum eval threshold (centipawns)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.input) as f:
        fens = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d FEN positions from %s", len(fens), args.input)

    gen = TraceGenerator(engine_depth=args.depth, min_eval_cp=args.min_eval)
    traces = gen.generate_traces(fens)

    with open(args.output, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")

    logger.info("Wrote %d traces to %s", len(traces), args.output)


if __name__ == "__main__":
    main()
