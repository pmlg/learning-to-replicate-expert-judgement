"""Experience buffer for mixed on-policy rollouts and off-policy traces.

Supports:
- Storing both on-policy (model-generated) and off-policy (Stockfish) transitions.
- Configurable mixture ratios (trace_ratio controls the fraction of off-policy data).
- Prioritised sampling for off-policy traces (higher reward → higher priority).
- Efficient batch sampling for GRPO training.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional

import numpy as np

from qwen_chess_rl.training.rl.luffy.schema import OffPolicyTrace

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Replay buffer that holds mixed on-policy and off-policy data.

    Parameters:
        capacity: Maximum number of total items to store.
        trace_ratio: Target fraction of off-policy traces in each sampled
            batch (0.0 = pure on-policy, 1.0 = pure off-policy).
        alpha: Prioritisation exponent for off-policy traces
            (0 = uniform, 1 = full priority).
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        trace_ratio: float = 0.3,
        alpha: float = 0.6,
        seed: int = 42,
    ):
        self.capacity = capacity
        self.trace_ratio = trace_ratio
        self.alpha = alpha
        self._rng = random.Random(seed)

        # On-policy: short-lived deque (cleared after each training batch)
        self.on_policy: deque = deque(maxlen=capacity)

        # Off-policy: persistent list with priority weights
        self.off_policy: list[OffPolicyTrace] = []
        self._priorities: list[float] = []

    # ------------------------------------------------------------------ #
    # Adding data
    # ------------------------------------------------------------------ #

    def add_on_policy(
        self,
        fen: str,
        action: str,
        reward: float,
        advantage: float = 0.0,
    ) -> None:
        """Add an on-policy rollout transition."""
        self.on_policy.append({
            "fen": fen,
            "action": action,
            "reward": reward,
            "advantage": advantage,
            "source": "on_policy",
        })

    def add_off_policy(self, trace: OffPolicyTrace) -> None:
        """Add an off-policy Stockfish trace."""
        if len(self.off_policy) >= self.capacity:
            # Replace a random item (reservoir-style)
            idx = self._rng.randint(0, len(self.off_policy) - 1)
            self.off_policy[idx] = trace
            self._priorities[idx] = max(abs(trace.reward), 1e-6) ** self.alpha
        else:
            self.off_policy.append(trace)
            self._priorities.append(max(abs(trace.reward), 1e-6) ** self.alpha)

    def add_off_policy_batch(self, traces: list[OffPolicyTrace]) -> None:
        """Add a batch of off-policy traces."""
        for trace in traces:
            self.add_off_policy(trace)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample(self, batch_size: int) -> dict:
        """Sample a mixed batch of on-policy and off-policy transitions.

        The fraction of off-policy items is controlled by ``trace_ratio``.

        Returns:
            Dictionary with keys:
                - 'on_policy': list of on-policy dicts
                - 'off_policy': list of OffPolicyTrace objects
                - 'on_policy_count': int
                - 'off_policy_count': int
        """
        n_off = int(batch_size * self.trace_ratio)
        n_on = batch_size - n_off

        # Sample on-policy
        on_items: list[dict] = []
        if n_on > 0 and self.on_policy:
            pool = list(self.on_policy)
            n_on = min(n_on, len(pool))
            on_items = self._rng.sample(pool, n_on)

        # Sample off-policy with priority
        off_items: list[OffPolicyTrace] = []
        if n_off > 0 and self.off_policy:
            n_off = min(n_off, len(self.off_policy))
            off_items = self._priority_sample(n_off)

        return {
            "on_policy": on_items,
            "off_policy": off_items,
            "on_policy_count": len(on_items),
            "off_policy_count": len(off_items),
        }

    def _priority_sample(self, n: int) -> list[OffPolicyTrace]:
        """Sample *n* off-policy traces using prioritised replay."""
        priorities = np.array(self._priorities, dtype=np.float64)
        total = priorities.sum()

        if total <= 0:
            # Uniform fallback
            indices = self._rng.sample(range(len(self.off_policy)), n)
        else:
            probs = priorities / total
            indices = list(
                np.random.choice(len(self.off_policy), size=n, replace=False, p=probs)
            )

        return [self.off_policy[i] for i in indices]

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def clear_on_policy(self) -> None:
        """Clear on-policy buffer (typically after each GRPO update)."""
        self.on_policy.clear()

    def __len__(self) -> int:
        return len(self.on_policy) + len(self.off_policy)

    @property
    def num_on_policy(self) -> int:
        return len(self.on_policy)

    @property
    def num_off_policy(self) -> int:
        return len(self.off_policy)

    def stats(self) -> dict:
        """Return buffer statistics."""
        on_rewards = [x["reward"] for x in self.on_policy] or [0.0]
        off_rewards = [t.reward for t in self.off_policy] or [0.0]
        return {
            "on_policy_count": self.num_on_policy,
            "off_policy_count": self.num_off_policy,
            "total": len(self),
            "trace_ratio_actual": (
                self.num_off_policy / max(len(self), 1)
            ),
            "mean_on_policy_reward": float(np.mean(on_rewards)),
            "mean_off_policy_reward": float(np.mean(off_rewards)),
        }
