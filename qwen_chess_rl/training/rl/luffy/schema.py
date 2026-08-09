"""Trace data schema and dataclasses for LUFFY off-policy traces.

Each trace captures a single state-action-transition with all metadata
needed for importance-sampling-corrected GRPO updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OffPolicyTrace:
    """A single off-policy (state, action) trace from Stockfish.

    Attributes:
        fen: Board position in FEN notation.
        action: Move in UCI notation (e.g. 'e2e4').
        behavior_log_prob: Log-probability of the action under the
            behavior (expert) policy — typically Stockfish's choice.
        stockfish_eval: Centipawn evaluation after the action (relative
            to the side to move).
        stockfish_pv: Principal variation from Stockfish (list of UCI moves).
        engine_depth: Depth at which Stockfish analysed the position.
        mate: Mate-in-N if the position is forced mate, else None.
        reward: Computed reward for this trace (set lazily).
        source: Origin of the trace ('stockfish', 'self_play', 'dataset').
    """

    fen: str
    action: str
    behavior_log_prob: float = 0.0
    stockfish_eval: int = 0
    stockfish_pv: list[str] = field(default_factory=list)
    engine_depth: int = 15
    mate: Optional[int] = None
    reward: float = 0.0
    source: str = "stockfish"

    def to_dict(self) -> dict:
        """Serialise to dictionary (for JSON/Arrow storage)."""
        return {
            "fen": self.fen,
            "action": self.action,
            "behavior_log_prob": self.behavior_log_prob,
            "stockfish_eval": self.stockfish_eval,
            "stockfish_pv": self.stockfish_pv,
            "engine_depth": self.engine_depth,
            "mate": self.mate,
            "reward": self.reward,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OffPolicyTrace:
        """Deserialise from dictionary."""
        return cls(
            fen=d["fen"],
            action=d["action"],
            behavior_log_prob=d.get("behavior_log_prob", 0.0),
            stockfish_eval=d.get("stockfish_eval", 0),
            stockfish_pv=d.get("stockfish_pv", []),
            engine_depth=d.get("engine_depth", 15),
            mate=d.get("mate"),
            reward=d.get("reward", 0.0),
            source=d.get("source", "stockfish"),
        )
