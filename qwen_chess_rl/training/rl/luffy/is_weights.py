"""Importance-sampling weight computation and regularisation for LUFFY.

Given off-policy traces with behaviour log-probs and the current policy's
log-probs, this module computes:
1. IS ratios (importance weights).
2. Clipped IS weights (stabilises training).
3. Confidence-based weighting to avoid pure imitation.
4. Entropy regularisation to maintain exploration.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_is_weights(
    policy_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    is_clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute importance-sampling ratios and clipped weights.

    The IS ratio is:

        ratio = exp(log π(a|s) - log μ(a|s))

    where π is the current policy and μ is the behaviour (expert) policy.
    Clipping prevents individual traces from dominating the gradient.

    Args:
        policy_log_probs: Log-probabilities from the current policy π.
            Shape: (N,) or (N, 1).
        behavior_log_probs: Log-probabilities from the behaviour policy μ.
            Shape: (N,) or (N, 1).
        is_clip: Clipping range for IS weights (ratio clamped to
            [1 - is_clip, 1 + is_clip]).

    Returns:
        Tuple of (ratios, clipped_ratios), both shape (N,).
    """
    # Ensure 1-D
    policy_log_probs = policy_log_probs.squeeze()
    behavior_log_probs = behavior_log_probs.squeeze()

    # IS ratio: π(a|s) / μ(a|s)  in log space → exp
    ratios = torch.exp(policy_log_probs - behavior_log_probs)

    # Clip to prevent extreme weights
    clipped_ratios = torch.clamp(ratios, 1.0 - is_clip, 1.0 + is_clip)

    return ratios, clipped_ratios


def confidence_weighted_loss(
    is_loss: torch.Tensor,
    policy_log_probs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply confidence-based weighting to avoid pure imitation.

    Traces where the policy already strongly agrees with the expert
    (high log-prob) get down-weighted — the model already knows these
    moves, so the off-policy signal adds little.  Traces where the
    policy is uncertain or disagrees get up-weighted.

    The weighting function is:

        w = (1 - softmax(log_prob / temperature)) * loss

    This ensures the model learns from its mistakes, not from positions
    it already plays well.

    Args:
        is_loss: Per-trace loss values. Shape: (N,).
        policy_log_probs: Per-trace policy log-probs. Shape: (N,).
        temperature: Controls the sharpness of the confidence weighting.

    Returns:
        Weighted loss scalar.
    """
    # Convert log-probs to normalised confidence
    probs = torch.exp(policy_log_probs / temperature)
    # Confidence: how aligned the policy is with the expert action
    confidence = probs / (probs + 1.0)  # Sigmoid-like squashing

    # Down-weight high-confidence (already-known) traces
    weight = 1.0 - confidence

    return (weight * is_loss).mean()


def entropy_regularization(
    logits: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """Compute entropy bonus to maintain exploration diversity.

    Encourages the policy to maintain a diverse move distribution rather
    than collapsing onto Stockfish moves.

    Args:
        logits: Raw logits from the policy model. Shape: (..., vocab_size).
        beta: Entropy bonus coefficient.

    Returns:
        Negative entropy (to be added to minimisation loss).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    # We subtract entropy because we want to MAXIMISE it (hence minimise -H)
    return -beta * entropy


def compute_luffy_loss(
    policy_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    is_clip: float = 0.2,
    offpolicy_weight: float = 0.5,
    confidence_temp: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the full LUFFY off-policy loss.

    Combines:
    1. IS-corrected policy gradient on off-policy traces.
    2. Confidence-based weighting to focus on uncertain positions.
    3. Scaling by ``offpolicy_weight`` to balance with on-policy GRPO.

    Args:
        policy_log_probs: Log-probs from current policy π. Shape: (N,).
        behavior_log_probs: Log-probs from behaviour policy μ. Shape: (N,).
        advantages: Advantages for each trace. Shape: (N,).
        is_clip: IS clipping range.
        offpolicy_weight: Overall weight for the off-policy loss term.
        confidence_temp: Temperature for confidence weighting.

    Returns:
        Dictionary with 'loss' (scalar), 'is_ratios', 'is_clipped_ratios',
        'confidence_weights'.
    """
    ratios, clipped_ratios = compute_is_weights(
        policy_log_probs, behavior_log_probs, is_clip,
    )

    # Clipped surrogate (same structure as PPO/GRPO)
    surrogate1 = ratios * advantages
    surrogate2 = clipped_ratios * advantages
    is_loss = -torch.min(surrogate1, surrogate2)  # (N,)

    # Confidence-based weighting
    probs = torch.exp(policy_log_probs / confidence_temp)
    confidence = probs / (probs + 1.0)
    confidence_weights = 1.0 - confidence

    weighted_loss = (confidence_weights * is_loss).mean()
    total_loss = offpolicy_weight * weighted_loss

    return {
        "loss": total_loss,
        "is_ratios": ratios,
        "is_clipped_ratios": clipped_ratios,
        "confidence_weights": confidence_weights,
        "raw_is_loss": is_loss.mean(),
    }
