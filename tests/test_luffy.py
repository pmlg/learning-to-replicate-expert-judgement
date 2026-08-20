"""Tests for LUFFY off-policy guidance components.

Covers:
- Trace data schema (OffPolicyTrace) serialisation round-trip.
- ExperienceBuffer mixed sampling and prioritised replay.
- IS-weight computation and clipping correctness.
- Confidence-weighted loss and entropy regularisation.
- Full LUFFY loss composition.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.grpo import combine_onpolicy_offpolicy_loss, compute_luffy_alpha
from qwen_chess_rl.training.rl.luffy.buffer import ExperienceBuffer
from qwen_chess_rl.training.rl.luffy.is_weights import (
    compute_is_weights,
    compute_luffy_loss,
    confidence_weighted_loss,
    entropy_regularization,
)
from qwen_chess_rl.training.rl.luffy.schema import OffPolicyTrace


# ---------------------------------------------------------------------- #
# Schema tests
# ---------------------------------------------------------------------- #

class TestOffPolicyTrace:
    """Test trace data schema serialisation."""

    def test_to_dict_roundtrip(self):
        trace = OffPolicyTrace(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            action="e2e4",
            behavior_log_prob=-0.5,
            stockfish_eval=50,
            stockfish_pv=["e2e4", "e7e5"],
            engine_depth=20,
            mate=None,
            reward=5.0,
            source="stockfish",
        )
        d = trace.to_dict()
        restored = OffPolicyTrace.from_dict(d)
        assert restored.fen == trace.fen
        assert restored.action == trace.action
        assert restored.behavior_log_prob == trace.behavior_log_prob
        assert restored.stockfish_eval == trace.stockfish_eval
        assert restored.stockfish_pv == trace.stockfish_pv
        assert restored.engine_depth == trace.engine_depth
        assert restored.reward == trace.reward
        assert restored.source == trace.source

    def test_from_dict_defaults(self):
        minimal = {"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "action": "a1a2"}
        trace = OffPolicyTrace.from_dict(minimal)
        assert trace.fen == minimal["fen"]
        assert trace.behavior_log_prob == 0.0
        assert trace.stockfish_pv == []
        assert trace.source == "stockfish"

    def test_mate_trace(self):
        trace = OffPolicyTrace(
            fen="rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            action="f6g7",
            behavior_log_prob=0.0,
            mate=1,
            reward=10.0,
        )
        assert trace.mate == 1
        d = trace.to_dict()
        assert d["mate"] == 1


# ---------------------------------------------------------------------- #
# ExperienceBuffer tests
# ---------------------------------------------------------------------- #

class TestExperienceBuffer:
    """Test the mixed experience buffer."""

    def test_add_on_policy(self):
        buf = ExperienceBuffer(capacity=100, seed=42)
        buf.add_on_policy("fen1", "e2e4", reward=5.0)
        assert buf.num_on_policy == 1
        assert buf.num_off_policy == 0

    def test_add_off_policy(self):
        buf = ExperienceBuffer(capacity=100, seed=42)
        trace = OffPolicyTrace(fen="fen1", action="e2e4", reward=5.0)
        buf.add_off_policy(trace)
        assert buf.num_off_policy == 1

    def test_mixed_sampling(self):
        buf = ExperienceBuffer(capacity=100, trace_ratio=0.5, seed=42)
        # Add 10 on-policy + 10 off-policy items
        for i in range(10):
            buf.add_on_policy(f"fen_on_{i}", f"move{i}", reward=float(i))
        for i in range(10):
            buf.add_off_policy(
                OffPolicyTrace(fen=f"fen_off_{i}", action=f"move{i}", reward=float(i))
            )
        sample = buf.sample(batch_size=8)
        assert sample["on_policy_count"] + sample["off_policy_count"] <= 8
        assert sample["on_policy_count"] == 4  # 50% of 8
        assert sample["off_policy_count"] == 4

    def test_default_config_batch_size_yields_off_policy_samples(self):
        # Regression test: luffy_offpolicy_loss used to pre-multiply
        # cfg.batch_size by cfg.trace_ratio before calling buffer.sample(),
        # which internally re-applies trace_ratio — at the GRPOConfig
        # defaults (batch_size=8, trace_ratio=0.3) that double application
        # floored to int(int(8*0.3)*0.3) = 0, silently disabling the LUFFY
        # off-policy loss at the default config. The correct call passes
        # cfg.batch_size straight through.
        cfg = GRPOConfig()
        buf = ExperienceBuffer(capacity=100, trace_ratio=cfg.trace_ratio, seed=cfg.seed)
        for i in range(20):
            buf.add_off_policy(
                OffPolicyTrace(fen=f"fen_off_{i}", action="e2e4", reward=float(i))
            )
        sample = buf.sample(batch_size=cfg.batch_size)
        assert sample["off_policy_count"] > 0

    def test_trace_ratio_zero(self):
        buf = ExperienceBuffer(capacity=100, trace_ratio=0.0, seed=42)
        for i in range(10):
            buf.add_on_policy(f"fen_{i}", "e2e4", reward=5.0)
        for i in range(10):
            buf.add_off_policy(
                OffPolicyTrace(fen=f"fen_off_{i}", action="e2e4", reward=5.0)
            )
        sample = buf.sample(batch_size=8)
        assert sample["off_policy_count"] == 0
        assert sample["on_policy_count"] == 8

    def test_trace_ratio_one(self):
        buf = ExperienceBuffer(capacity=100, trace_ratio=1.0, seed=42)
        for i in range(10):
            buf.add_on_policy(f"fen_{i}", "e2e4", reward=5.0)
        for i in range(10):
            buf.add_off_policy(
                OffPolicyTrace(fen=f"fen_off_{i}", action="e2e4", reward=5.0)
            )
        sample = buf.sample(batch_size=8)
        assert sample["on_policy_count"] == 0
        assert sample["off_policy_count"] == 8

    def test_capacity_limit_off_policy(self):
        buf = ExperienceBuffer(capacity=5, seed=42)
        for i in range(10):
            buf.add_off_policy(
                OffPolicyTrace(fen=f"fen_{i}", action="e2e4", reward=float(i))
            )
        assert buf.num_off_policy == 5  # Capped

    def test_clear_on_policy(self):
        buf = ExperienceBuffer(capacity=100, seed=42)
        buf.add_on_policy("fen", "e2e4", reward=5.0)
        assert buf.num_on_policy == 1
        buf.clear_on_policy()
        assert buf.num_on_policy == 0

    def test_stats(self):
        buf = ExperienceBuffer(capacity=100, seed=42)
        buf.add_on_policy("fen", "e2e4", reward=5.0)
        buf.add_off_policy(OffPolicyTrace(fen="fen2", action="d2d4", reward=3.0))
        stats = buf.stats()
        assert stats["on_policy_count"] == 1
        assert stats["off_policy_count"] == 1
        assert stats["total"] == 2
        assert stats["mean_on_policy_reward"] == 5.0


# ---------------------------------------------------------------------- #
# IS-weight tests
# ---------------------------------------------------------------------- #

class TestISWeights:
    """Test importance-sampling weight computation."""

    def test_ratio_one_when_equal(self):
        # When policy_log_prob == behavior_log_prob, ratio should be 1.0
        log_prob = torch.tensor([-0.5, -0.3, -0.2])
        ratios, clipped = compute_is_weights(log_prob, log_prob, is_clip=0.2)
        assert torch.allclose(ratios, torch.ones(3))
        assert torch.allclose(clipped, torch.ones(3))

    def test_clipping(self):
        # Large ratio should be clipped
        policy = torch.tensor([0.0])  # exp(0 - (-5)) = exp(5) ≈ 148
        behavior = torch.tensor([-5.0])
        ratios, clipped = compute_is_weights(policy, behavior, is_clip=0.2)
        assert ratios.item() > 1.0  # Unclipped is large
        assert clipped.item() == pytest.approx(1.2)  # Clipped to 1 + 0.2

    def test_clipping_lower(self):
        # Very small ratio should be clipped to 1 - is_clip
        policy = torch.tensor([-10.0])
        behavior = torch.tensor([0.0])
        ratios, clipped = compute_is_weights(policy, behavior, is_clip=0.2)
        assert ratios.item() < 1.0
        assert clipped.item() == pytest.approx(0.8)  # Clipped to 1 - 0.2

    def test_shapes(self):
        policy = torch.randn(10)
        behavior = torch.randn(10)
        ratios, clipped = compute_is_weights(policy, behavior)
        assert ratios.shape == (10,)
        assert clipped.shape == (10,)


# ---------------------------------------------------------------------- #
# Confidence-weighted loss tests
# ---------------------------------------------------------------------- #

class TestConfidenceWeightedLoss:
    """Test confidence-based anti-imitation weighting."""

    def test_high_confidence_downweighted(self):
        # High policy log-prob → high confidence → low weight → small loss
        is_loss = torch.ones(5)
        high_conf = torch.zeros(5)  # log_prob=0 → prob=1 → confidence=0.5
        loss_high = confidence_weighted_loss(is_loss, high_conf, temperature=1.0)
        # confidence = exp(0)/(exp(0)+1) = 0.5, weight = 0.5
        assert loss_high.item() < is_loss.mean().item()

    def test_low_confidence_upweighted(self):
        # Low policy log-prob → low confidence → weight near 1
        is_loss = torch.ones(5)
        low_conf = torch.full((5,), -10.0)  # Very negative log-prob
        loss_low = confidence_weighted_loss(is_loss, low_conf, temperature=1.0)
        # weight should be near 1 (1 - small ≈ 1)
        assert loss_low.item() > 0.5  # Should be close to is_loss.mean()

    def test_entropy_regularization(self):
        # Entropy of a uniform distribution should give a positive bonus
        logits = torch.zeros(4, 10)  # Uniform logits → max entropy
        entropy_loss = entropy_regularization(logits, beta=0.01)
        assert entropy_loss.item() < 0  # Negative because we return -beta * H

    def test_entropy_zero_for_peaked(self):
        # Very peaked distribution → near-zero entropy
        logits = torch.tensor([[100.0] + [0.0] * 9])
        entropy_loss = entropy_regularization(logits, beta=0.01)
        assert entropy_loss.item() > -0.01  # Close to zero


# ---------------------------------------------------------------------- #
# Full LUFFY loss tests
# ---------------------------------------------------------------------- #

class TestLuffyLoss:
    """Test the full LUFFY loss composition."""

    def test_loss_is_finite(self):
        policy = torch.randn(8)
        behavior = torch.randn(8)
        advantages = torch.randn(8)
        result = compute_luffy_loss(policy, behavior, advantages)
        assert torch.isfinite(result["loss"])

    def test_loss_increases_with_advantage(self):
        policy = torch.randn(8)
        behavior = policy.clone()  # ratio = 1
        advantages_small = torch.ones(8) * 0.1
        advantages_large = torch.ones(8) * 1.0

        result_small = compute_luffy_loss(policy, behavior, advantages_small)
        result_large = compute_luffy_loss(policy, behavior, advantages_large)

        # Larger advantage magnitude → larger loss (because we minimise)
        assert result_large["loss"].abs() >= result_small["loss"].abs() * 0.9

    def test_returns_all_keys(self):
        policy = torch.randn(4)
        behavior = torch.randn(4)
        advantages = torch.randn(4)
        result = compute_luffy_loss(policy, behavior, advantages)
        expected_keys = {"loss", "is_ratios", "is_clipped_ratios", "confidence_weights", "raw_is_loss"}
        assert set(result.keys()) == expected_keys


class TestCombineOnpolicyOffpolicyLoss:
    """Test the luffy_alpha blend between on-policy GRPO and off-policy LUFFY loss."""

    def test_alpha_zero_is_pure_onpolicy(self):
        onpolicy = torch.tensor(4.0)
        offpolicy = torch.tensor(10.0)
        combined = combine_onpolicy_offpolicy_loss(onpolicy, offpolicy, luffy_alpha=0.0)
        assert combined.item() == pytest.approx(4.0)

    def test_alpha_one_is_pure_offpolicy(self):
        onpolicy = torch.tensor(4.0)
        offpolicy = torch.tensor(10.0)
        combined = combine_onpolicy_offpolicy_loss(onpolicy, offpolicy, luffy_alpha=1.0)
        assert combined.item() == pytest.approx(10.0)

    def test_alpha_blends_proportionally(self):
        onpolicy = torch.tensor(4.0)
        offpolicy = torch.tensor(10.0)
        combined = combine_onpolicy_offpolicy_loss(onpolicy, offpolicy, luffy_alpha=0.5)
        assert combined.item() == pytest.approx(0.5 * 4.0 + 0.5 * 10.0)

    def test_none_onpolicy_falls_back_to_scaled_offpolicy(self):
        offpolicy = torch.tensor(10.0)
        combined = combine_onpolicy_offpolicy_loss(None, offpolicy, luffy_alpha=0.5)
        assert combined.item() == pytest.approx(5.0)

    def test_none_offpolicy_falls_back_to_scaled_onpolicy(self):
        onpolicy = torch.tensor(4.0)
        combined = combine_onpolicy_offpolicy_loss(onpolicy, None, luffy_alpha=0.5)
        assert combined.item() == pytest.approx(2.0)

    def test_both_none_returns_none(self):
        assert combine_onpolicy_offpolicy_loss(None, None, luffy_alpha=0.5) is None


class TestComputeLuffyAlpha:
    """Test the luffy_alpha anneal schedule."""

    def test_disabled_returns_constant(self):
        cfg = GRPOConfig(luffy_alpha=0.85, luffy_alpha_final=0.3, luffy_alpha_anneal_steps=0)
        assert compute_luffy_alpha(cfg, 1) == pytest.approx(0.85)
        assert compute_luffy_alpha(cfg, 10_000) == pytest.approx(0.85)

    def test_starts_at_initial_value(self):
        cfg = GRPOConfig(luffy_alpha=0.85, luffy_alpha_final=0.3, luffy_alpha_anneal_steps=100)
        assert compute_luffy_alpha(cfg, 0) == pytest.approx(0.85)

    def test_midpoint_is_halfway(self):
        cfg = GRPOConfig(luffy_alpha=0.85, luffy_alpha_final=0.3, luffy_alpha_anneal_steps=100)
        assert compute_luffy_alpha(cfg, 50) == pytest.approx((0.85 + 0.3) / 2)

    def test_holds_at_final_value_past_anneal_steps(self):
        cfg = GRPOConfig(luffy_alpha=0.85, luffy_alpha_final=0.3, luffy_alpha_anneal_steps=100)
        assert compute_luffy_alpha(cfg, 100) == pytest.approx(0.3)
        assert compute_luffy_alpha(cfg, 10_000) == pytest.approx(0.3)

    def test_can_anneal_upward(self):
        cfg = GRPOConfig(luffy_alpha=0.2, luffy_alpha_final=0.9, luffy_alpha_anneal_steps=10)
        assert compute_luffy_alpha(cfg, 5) == pytest.approx(0.2 + 0.5 * (0.9 - 0.2))
