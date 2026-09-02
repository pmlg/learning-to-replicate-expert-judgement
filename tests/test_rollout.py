"""Tests for multi-ply rollouts and episode-outcome reward.

Covers:
- compute_episode_reward: outcome/cutoff-eval -> scalar reward.
- play_rollout: illegal first move, immediate mate, mid-game illegal move,
  capped-mode ply counting, full-game termination bound.
- grpo_trajectory_loss: long trajectories must not blow up numerically.

The play_rollout tests need a real Stockfish binary (the opponent's moves
and cutoff evals go through it) — they're skipped if none is found.
"""

from __future__ import annotations

import chess
import pytest

import qwen_chess_rl.training.rl.rollout as rollout_mod
from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.reward import (
    DRAW_REWARD,
    DRAW_REWARD_WHEN_LOSING,
    LEGAL_MOVE_BONUS,
    LOSING_EVAL_THRESHOLD,
    LOSS_REWARD,
    WIN_REWARD,
    _find_stockfish,
    compute_episode_reward,
)
from qwen_chess_rl.training.rl.rollout import RolloutResult, play_rollout

STOCKFISH_PATH = _find_stockfish()
requires_stockfish = pytest.mark.skipif(
    STOCKFISH_PATH is None, reason="Stockfish binary not found"
)

# Position one move from mate (Fool's Mate): 1. f3 e5 2. g4, black to move.
FOOLS_MATE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
FOOLS_MATE_MOVE = "d8h4"

# White to move, missing its queen against a full black army — clearly lost.
QUEEN_DOWN_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"


class TestComputeEpisodeReward:
    def test_win(self):
        assert compute_episode_reward("win") == WIN_REWARD

    def test_loss(self):
        assert compute_episode_reward("loss") == LOSS_REWARD

    def test_draw(self):
        assert compute_episode_reward("draw") == DRAW_REWARD

    def test_capped_outcome_without_cutoff_eval_gets_no_reward(self):
        # No cutoff_eval available (e.g. no engine) -> no signal at all.
        assert compute_episode_reward(None) == 0.0

    def test_capped_outcome_uses_win_probability_from_cutoff_eval(self):
        # Chess-R1 style dense reward: a winning cutoff position scores
        # near WIN_REWARD, a losing one near LOSS_REWARD, a balanced one
        # near zero — instead of flat 0.0 regardless of position quality.
        assert compute_episode_reward(None, cutoff_eval=0.0) == pytest.approx(0.0, abs=1e-6)
        winning = compute_episode_reward(None, cutoff_eval=400.0)
        losing = compute_episode_reward(None, cutoff_eval=-400.0)
        assert 0.0 < winning < WIN_REWARD
        assert LOSS_REWARD < losing < 0.0
        assert winning == pytest.approx(-losing)

    def test_capped_outcome_mate_score_saturates_near_win_reward(self):
        # Mate scores arrive pre-encoded as +/-10000 (see rollout.py) — the
        # logistic conversion should saturate close to WIN_REWARD/LOSS_REWARD.
        assert compute_episode_reward(None, cutoff_eval=10000.0) == pytest.approx(
            WIN_REWARD, abs=1e-3
        )
        assert compute_episode_reward(None, cutoff_eval=-10000.0) == pytest.approx(
            LOSS_REWARD, abs=1e-3
        )

    def test_legal_move_bonus_is_zero(self):
        # Legality alone earns nothing — only game outcome matters.
        assert LEGAL_MOVE_BONUS == 0.0

    def test_draw_from_balanced_start_is_neutral_not_penalized(self):
        assert DRAW_REWARD == 0.0

    def test_win_dominates_any_single_move_reward(self):
        # Winning must be worth strictly more than the largest score a
        # single move could get under compute_reward() (LEGAL_MOVE_BONUS +
        # CHECKMATE_BONUS = 15) — otherwise game outcome wouldn't actually
        # dominate the training signal.
        assert WIN_REWARD > 15.0

    def test_draw_from_balanced_or_winning_start_stays_discouraged(self):
        assert compute_episode_reward("draw", starting_eval=30.0) == DRAW_REWARD
        assert compute_episode_reward("draw", starting_eval=None) == DRAW_REWARD

    def test_draw_salvaged_from_losing_start_is_rewarded_well(self):
        assert compute_episode_reward("draw", starting_eval=-900.0) == DRAW_REWARD_WHEN_LOSING
        reward = compute_episode_reward("draw", starting_eval=LOSING_EVAL_THRESHOLD - 1)
        assert reward == DRAW_REWARD_WHEN_LOSING

    def test_draw_reward_boundary_is_strictly_below_threshold(self):
        # Exactly at the threshold is not "clearly lost" yet.
        assert compute_episode_reward("draw", starting_eval=LOSING_EVAL_THRESHOLD) == DRAW_REWARD

    def test_draw_reward_when_losing_beats_normal_draw_but_not_a_win(self):
        assert DRAW_REWARD_WHEN_LOSING > DRAW_REWARD
        assert DRAW_REWARD_WHEN_LOSING < WIN_REWARD

    def test_win_and_loss_are_unaffected_by_starting_eval(self):
        assert compute_episode_reward("win", starting_eval=-900.0) == WIN_REWARD
        assert compute_episode_reward("loss", starting_eval=-900.0) == LOSS_REWARD


class TestPlayRolloutStructure:
    """Structural tests that don't need Stockfish (illegal-move paths)."""

    def test_illegal_first_move_returns_none(self):
        cfg = GRPOConfig()
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e5", cfg, max_plies=4,
        )
        assert result is None

    def test_immediate_mate_on_first_move_is_a_win_no_opponent_turn(self):
        cfg = GRPOConfig()
        result = play_rollout(
            None, None, FOOLS_MATE_FEN, FOOLS_MATE_MOVE, cfg, max_plies=None,
        )
        assert result.outcome == "win"
        assert result.own_moves == [FOOLS_MATE_MOVE]
        assert result.num_plies == 1

    def test_mid_game_illegal_move_forfeits_as_a_loss(self, monkeypatch):
        monkeypatch.setattr(rollout_mod, "_generate_one_move", lambda *a, **k: "z9z9")
        cfg = GRPOConfig(opponent_depth=2)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=None,
        )
        assert result.outcome == "loss"


class TestVetMove:
    """_vet_move must veto legal-but-losing moves using a searched eval."""

    # 1. e4 e5 2. Qh5 Nc6 3. Bc4, Black to move — a6 ignores the Qxf7# threat.
    SCHOLARS_MATE_SETUP = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"

    def test_overrides_a_blunder_that_hangs_mate(self):
        cfg = GRPOConfig(rollout_vet_depth=3, rollout_vet_time_limit=1.0)
        board = chess.Board(self.SCHOLARS_MATE_SETUP)
        blunder = chess.Move.from_uci("a7a6")

        vetted = rollout_mod._vet_move(board, blunder, cfg)

        assert vetted != blunder
        assert vetted in board.legal_moves
        assert board.fen() == self.SCHOLARS_MATE_SETUP  # board restored

    def test_keeps_a_sound_move_unchanged(self):
        cfg = GRPOConfig(rollout_vet_depth=3, rollout_vet_time_limit=1.0)
        board = chess.Board()
        fine_move = chess.Move.from_uci("g1f3")  # Nf3

        assert rollout_mod._vet_move(board, fine_move, cfg) == fine_move


@requires_stockfish
class TestPlayRolloutWithStockfish:
    """Tests exercising the real Stockfish opponent and cutoff evaluation."""

    def test_capped_mode_stops_at_ply_cap(self, monkeypatch):
        def fake_move(model, tokenizer, fen, cfg):
            return str(list(chess.Board(fen).legal_moves)[0])

        monkeypatch.setattr(rollout_mod, "_generate_one_move", fake_move)
        cfg = GRPOConfig(opponent_depth=2, eval_depth=5)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=4,
        )
        assert result.outcome is None
        assert result.cutoff_eval is not None
        assert result.num_plies == 4
        assert len(result.own_moves) == 2
        assert len(result.own_prompts) == 2

    def test_full_game_terminates_with_a_real_outcome(self, monkeypatch):
        def fake_move(model, tokenizer, fen, cfg):
            return str(list(chess.Board(fen).legal_moves)[0])

        monkeypatch.setattr(rollout_mod, "_generate_one_move", fake_move)
        cfg = GRPOConfig(opponent_depth=2)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=None,
        )
        assert result.outcome in ("win", "loss", "draw")

    def test_opponent_move_is_always_legal(self):
        import chess.engine

        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        try:
            board = chess.Board()
            move = rollout_mod._select_opponent_move(board, engine, depth=3)
            assert move in board.legal_moves
        finally:
            engine.quit()

    def test_starting_eval_reflects_a_balanced_position(self):
        cfg = GRPOConfig(eval_depth=5)
        result = play_rollout(
            None, None, chess.STARTING_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=1,
        )
        assert result.starting_eval is not None
        assert abs(result.starting_eval) < 150

    def test_starting_eval_reflects_a_clearly_losing_position(self):
        cfg = GRPOConfig(eval_depth=8)
        result = play_rollout(
            None, None, QUEEN_DOWN_FEN, "e2e4", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=1,
        )
        assert result.starting_eval is not None
        assert result.starting_eval < LOSING_EVAL_THRESHOLD

    # Black to move after 1. e4 — the model (Black) plays the rest of the
    # Scholar's-mate line itself; the opponent's replies (Qh5, Bc4) are
    # scripted too, so the exact position is reached deterministically
    # without depending on what the real Stockfish opponent happens to play.
    AFTER_1_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    def _play_scholars_mate_setup(self, monkeypatch, own_third_move):
        own_moves_script = iter(["b8c6", own_third_move])
        monkeypatch.setattr(
            rollout_mod, "_generate_one_move",
            lambda model, tokenizer, fen, cfg: next(own_moves_script),
        )
        opponent_script = iter(["d1h5", "f1c4"])
        monkeypatch.setattr(
            rollout_mod, "_select_opponent_move",
            lambda board, engine, depth: chess.Move.from_uci(next(opponent_script)),
        )
        cfg = GRPOConfig(
            rollout_vet_enabled=True, rollout_vet_depth=3, rollout_vet_time_limit=1.0,
        )
        return play_rollout(
            None, None, self.AFTER_1_E4_FEN, "e7e5", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=5,
        )

    def test_vetting_substitutes_a_blundering_own_move(self, monkeypatch):
        # 1. e4 e5 2. Qh5 Nc6 3. Bc4 a6?? — a6 ignores the Qxf7# threat;
        # vetting should replace it before it's ever pushed to the board.
        result = self._play_scholars_mate_setup(monkeypatch, own_third_move="a7a6")
        assert result.own_moves == ["e7e5", "b8c6", result.own_moves[-1]]
        assert result.own_moves[-1] != "a7a6"

    def test_vetting_disabled_keeps_the_blunder(self, monkeypatch):
        own_moves_script = iter(["b8c6", "a7a6"])
        monkeypatch.setattr(
            rollout_mod, "_generate_one_move",
            lambda model, tokenizer, fen, cfg: next(own_moves_script),
        )
        opponent_script = iter(["d1h5", "f1c4"])
        monkeypatch.setattr(
            rollout_mod, "_select_opponent_move",
            lambda board, engine, depth: chess.Move.from_uci(next(opponent_script)),
        )
        cfg = GRPOConfig(rollout_vet_enabled=False)
        result = play_rollout(
            None, None, self.AFTER_1_E4_FEN, "e7e5", cfg,
            stockfish_path=STOCKFISH_PATH, max_plies=5,
        )
        assert result.own_moves[-1] == "a7a6"


class TestGrpoTrajectoryLossStability:
    """Regression tests: a long trajectory must never blow up numerically.

    Summing log-prob differences across a whole trajectory (rather than
    averaging per token) let the importance ratio overflow to inf for
    long full-game rollouts, producing NaN gradients that permanently
    corrupted the model. See grpo_trajectory_loss's docstring.
    """

    @staticmethod
    def _build_tiny_lora_model_and_tokenizer():
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
        config = GPT2Config(
            vocab_size=vocab_size, n_positions=512, n_embd=32, n_layer=2, n_head=2,
        )
        lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")

        model = get_peft_model(GPT2LMHeadModel(config), lora_cfg)
        model.train()

        ref_model = get_peft_model(GPT2LMHeadModel(config), lora_cfg)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        return model, ref_model, tokenizer

    def test_long_trajectory_produces_a_finite_loss_and_gradients(self):
        import torch

        from qwen_chess_rl.data.dataset import format_fen_prompt
        from qwen_chess_rl.training.rl.grpo import grpo_trajectory_loss

        model, ref_model, tokenizer = self._build_tiny_lora_model_and_tokenizer()

        # A 30-ply full-game-length trajectory — long enough that summing
        # (rather than averaging) log-prob differences would overflow.
        fen = chess.STARTING_FEN
        traj = RolloutResult(
            own_prompts=[format_fen_prompt(fen)] * 30,
            own_moves=["e2e4"] * 30,
            outcome="win",
        )
        cfg = GRPOConfig(max_length=256, clip_epsilon=0.2, beta=0.01)

        loss = grpo_trajectory_loss(model, ref_model, tokenizer, [traj], [1.5], cfg)

        assert loss is not None
        assert torch.isfinite(loss)
        loss.backward()
        lora_grads = [p.grad for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
        assert all(g is not None for g in lora_grads)
        assert all(torch.isfinite(g).all() for g in lora_grads)
