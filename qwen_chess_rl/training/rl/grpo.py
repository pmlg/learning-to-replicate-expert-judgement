"""GRPO (Group Relative Policy Optimization) training loop for chess RL."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_chess_rl.data.dataset import format_fen_prompt
from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.reward import compute_episode_reward
from qwen_chess_rl.training.rl.rollout import play_rollout

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FenPromptDataset(Dataset):
    """Minimal dataset that yields raw FEN strings for GRPO rollouts."""

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        max_samples: int = -1,
        seed: int = 42,
    ):
        self.dataset = load_from_disk(str(Path(data_path) / split))
        if max_samples > 0 and len(self.dataset) > max_samples:
            self.dataset = self.dataset.shuffle(seed=seed).select(range(max_samples))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> str:
        return str(self.dataset[idx]["fen"])


def load_rl_model(cfg: GRPOConfig, trainable: bool = True):
    """Load the SFT-fine-tuned model for RL training.

    Loaded in 4-bit (matching the SFT stage) — the policy and reference
    model are each a full copy of the base model, and two full-precision
    9B-param copies don't fit in a single 24-32GB GPU alongside activations.

    Args:
        cfg: GRPO configuration.
        trainable: Whether this copy will be trained (the policy model) or
            only used for frozen inference (the reference model).

    Returns:
        (model, tokenizer) tuple.
    """
    logger.info("Loading SFT checkpoint: %s", cfg.sft_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.sft_checkpoint,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    adapter_config_path = Path(cfg.sft_checkpoint) / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training

        peft_config = PeftConfig.from_pretrained(cfg.sft_checkpoint)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        if trainable:
            base_model = prepare_model_for_kbit_training(base_model)
        # is_trainable=True is required — loading an adapter for inference
        # (the default) freezes it, and loss.backward() would fail with
        # "element 0 of tensors does not require grad and does not have a grad_fn".
        model = PeftModel.from_pretrained(
            base_model,
            cfg.sft_checkpoint,
            is_trainable=trainable,
        )
        logger.info(
            "Loaded PEFT model with %s LoRA adapters.",
            "trainable" if trainable else "frozen",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.sft_checkpoint,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        logger.warning("Model does not appear to have LoRA adapters loaded.")

    model.train() if trainable else model.eval()
    return model, tokenizer


def generate_moves(
    model,
    tokenizer,
    fens: list[str],
    cfg: GRPOConfig,
    num_generations: int = 8,
) -> list[list[str]]:
    """Generate multiple candidate moves for each FEN position.

    Args:
        model: The policy model (with LoRA adapters).
        tokenizer: HuggingFace tokenizer.
        fens: List of FEN positions.
        cfg: GRPO configuration.
        num_generations: Number of moves to sample per position.

    Returns:
        List of lists: for each FEN, list of generated move strings.
    """
    prompts = [format_fen_prompt(fen) for fen in fens]

    # Repeat each prompt num_generations times for batched generation
    repeated_prompts = []
    fen_indices = []
    for i, p in enumerate(prompts):
        for _ in range(num_generations):
            repeated_prompts.append(p)
            fen_indices.append(i)

    # Tokenize
    inputs = tokenizer(
        repeated_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    # Generate — must run in eval mode. The policy model is kept in train()
    # for its backward passes elsewhere in the loop, but generating while
    # train() is active leaves dropout on and, combined with gradient
    # checkpointing from prepare_model_for_kbit_training, corrupts sampled
    # tokens into non-UCI gibberish (verified: the exact same checkpoint on
    # the exact same FEN produces clean legal moves in eval() and garbage
    # in train()) — every rollout would silently fail to produce a legal
    # move, starving GRPO of any on-policy gradient. Restore the prior mode
    # afterward so training resumes normally.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,  # Moves are short UCI strings (e.g., "e2e4")
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        if was_training:
            model.train()

    # Decode only newly generated tokens (not the prompt)
    new_tokens = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Extract UCI move (first whitespace-separated token)
    all_moves = [[] for _ in range(len(fens))]
    for idx, text in zip(fen_indices, decoded):
        move = text.strip().split()[0] if text.strip() else ""
        all_moves[idx].append(move)

    return all_moves


def compute_advantages(
    rewards: list[list[float]],
    epsilon: float = 1e-8,
) -> list[list[float]]:
    """Compute group-normalized advantages for GRPO.

    For each group (all generations for one FEN), normalize rewards
    to zero mean and unit variance.

    Args:
        rewards: List of lists, each inner list has num_generations reward values.
        epsilon: Small constant to prevent division by zero.

    Returns:
        Normalized advantages (same shape as rewards).
    """
    advantages = []
    for group in rewards:
        mean_r = np.mean(group)
        std_r = np.std(group) + epsilon
        adv = [(r - mean_r) / std_r for r in group]
        advantages.append(adv)
    return advantages


def grpo_trajectory_loss(
    model,
    ref_model,
    tokenizer,
    trajectories: list,
    advantages: list[float],
    cfg: GRPOConfig,
) -> torch.Tensor | None:
    """Compute the GRPO loss over full multi-ply rollout trajectories.

    Generalizes the original single-move GRPO loss to credit an entire
    rollout (see rollout.play_rollout): the trajectory-level advantage —
    derived from the game's actual outcome, not just one move's quality —
    is applied to the policy's log-probs for *every* move it made during
    that trajectory. Each ply has its own prompt (the FEN changes move to
    move), so unlike the single-move version this tokenizes each ply
    itself rather than taking pre-tokenized tensors.

    Log-probs are averaged per token across the trajectory, not summed.
    Full-game rollouts can span dozens of the policy's own moves (60-100+
    tokens); summing log-prob differences over that many tokens made the
    importance ratio (exp of the sum) overflow to inf for even mild
    per-token drift, producing NaN gradients that permanently corrupt the
    model. Averaging keeps the ratio's scale independent of trajectory
    length, matching the single-move case when there's only one ply.

    The loss combines, per trajectory:
    1. Clipped importance-sampling policy gradient on the averaged log-probs.
    2. KL divergence penalty against the reference model.

    Any trajectory whose ratio comes out non-finite (defense in depth,
    beyond the averaging fix above) is skipped with a warning rather than
    allowed to poison the whole batch's loss.

    Args:
        model: Current policy model.
        ref_model: Reference model (SFT checkpoint, frozen).
        tokenizer: HuggingFace tokenizer.
        trajectories: One rollout.RolloutResult per sampled candidate.
        advantages: Group-normalized advantage per trajectory (same order
            and length as trajectories).
        cfg: GRPO configuration.

    Returns:
        Scalar loss tensor, or ``None`` when no valid trajectories exist.
    """
    total_loss: torch.Tensor | None = None
    num_samples = 0
    device = next(model.parameters()).device

    for traj, adv in zip(trajectories, advantages):
        gen_log_probs_sum: torch.Tensor | None = None
        ref_log_probs_sum: torch.Tensor | None = None
        token_count = 0

        for prompt, move_uci in zip(traj.own_prompts, traj.own_moves):
            move_id_list = tokenizer.encode(move_uci, add_special_tokens=False)
            if not move_id_list:
                continue
            move_ids = torch.tensor(move_id_list, dtype=torch.long, device=device)

            prompt_inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length,
            ).to(device)
            full_input = torch.cat([prompt_inputs["input_ids"][0], move_ids], dim=-1)
            full_mask = torch.cat(
                [prompt_inputs["attention_mask"][0], torch.ones_like(move_ids)], dim=-1,
            )

            outputs = model(full_input.unsqueeze(0), attention_mask=full_mask.unsqueeze(0))
            log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            ply_log_probs = log_probs[:, -len(move_ids):, :].gather(
                2, move_ids.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).sum()

            with torch.no_grad():
                ref_outputs = ref_model(
                    full_input.unsqueeze(0), attention_mask=full_mask.unsqueeze(0),
                )
                ref_log_probs = F.log_softmax(ref_outputs.logits[:, :-1, :], dim=-1)
                ply_ref_log_probs = ref_log_probs[:, -len(move_ids):, :].gather(
                    2, move_ids.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1).sum()

            gen_log_probs_sum = (
                ply_log_probs if gen_log_probs_sum is None else gen_log_probs_sum + ply_log_probs
            )
            ref_log_probs_sum = (
                ply_ref_log_probs if ref_log_probs_sum is None
                else ref_log_probs_sum + ply_ref_log_probs
            )
            token_count += len(move_ids)

        if gen_log_probs_sum is None or token_count == 0:
            continue

        # Average per token, not summed — keeps the ratio's scale
        # independent of how many plies the trajectory has (see docstring).
        gen_log_probs = gen_log_probs_sum / token_count
        ref_gen_log_probs = ref_log_probs_sum / token_count

        ratio = torch.exp(gen_log_probs - ref_gen_log_probs)
        if not torch.isfinite(ratio):
            logger.warning("Non-finite importance ratio for a trajectory; skipping it.")
            continue
        adv_t = torch.as_tensor(adv, device=ratio.device, dtype=ratio.dtype)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
        policy_loss = -torch.min(ratio * adv_t, clipped_ratio * adv_t)

        # KL penalty (from ref model)
        kl_div = gen_log_probs - ref_gen_log_probs
        kl_penalty = cfg.beta * kl_div

        sample_loss = policy_loss + kl_penalty
        if not torch.isfinite(sample_loss):
            logger.warning("Non-finite loss for a trajectory; skipping it.")
            continue
        total_loss = sample_loss if total_loss is None else total_loss + sample_loss
        num_samples += 1

    if total_loss is None or num_samples == 0:
        return None

    return total_loss / num_samples


def combine_onpolicy_offpolicy_loss(
    onpolicy_loss: torch.Tensor | None,
    offpolicy_loss: torch.Tensor | None,
    luffy_alpha: float,
) -> torch.Tensor | None:
    """Blend the on-policy GRPO loss with the off-policy LUFFY loss.

    total = (1 - luffy_alpha) * onpolicy_loss + luffy_alpha * offpolicy_loss

    Either input may be ``None`` (no on-policy samples this step, or LUFFY
    disabled / nothing sampled from the buffer) — in that case the combined
    loss falls back to whichever term is available, still scaled by
    ``luffy_alpha``.

    Args:
        onpolicy_loss: On-policy GRPO loss, or None.
        offpolicy_loss: Off-policy LUFFY loss, or None.
        luffy_alpha: Ratio of off-policy to on-policy weight, in [0, 1].

    Returns:
        Combined loss tensor, or None if both inputs are None.
    """
    if onpolicy_loss is None and offpolicy_loss is None:
        return None
    if onpolicy_loss is None:
        return luffy_alpha * offpolicy_loss
    if offpolicy_loss is None:
        return (1 - luffy_alpha) * onpolicy_loss
    return (1 - luffy_alpha) * onpolicy_loss + luffy_alpha * offpolicy_loss


def compute_luffy_alpha(cfg: GRPOConfig, step: int) -> float:
    """Compute the current luffy_alpha under the anneal schedule.

    Linearly interpolates from ``cfg.luffy_alpha`` to ``cfg.luffy_alpha_final``
    over ``cfg.luffy_alpha_anneal_steps`` steps, then holds at the final
    value. With ``luffy_alpha_anneal_steps == 0`` (default), returns
    ``cfg.luffy_alpha`` unchanged — the anneal is opt-in.

    Args:
        cfg: GRPO configuration.
        step: Current global training step (1-indexed, as incremented in
            the training loop).

    Returns:
        The luffy_alpha value to use for this step.
    """
    if cfg.luffy_alpha_anneal_steps <= 0:
        return cfg.luffy_alpha
    frac = min(step / cfg.luffy_alpha_anneal_steps, 1.0)
    return cfg.luffy_alpha + frac * (cfg.luffy_alpha_final - cfg.luffy_alpha)


def train(cfg: GRPOConfig):
    """Run the full GRPO training loop.

    When ``cfg.use_luffy`` is True, the loop is augmented with LUFFY
    off-policy guidance: expert (Stockfish) traces are injected during
    policy optimisation via importance-sampling corrections and
    confidence-based regularisation.

    Args:
        cfg: GRPO training configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and reference model
    model, tokenizer = load_rl_model(cfg, trainable=True)
    ref_model, _ = load_rl_model(cfg, trainable=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load raw FENs (do not reuse SFT token tensors — GRPO needs prompt/FEN alignment)
    dataset = FenPromptDataset(
        data_path=cfg.data_path,
        split="train",
        max_samples=cfg.max_train_samples,
        seed=cfg.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: list(batch),
        num_workers=4,
    )

    # --- LUFFY initialisation ---
    luffy_buffer = None
    if cfg.use_luffy:
        from qwen_chess_rl.training.rl.luffy.training import prepare_luffy_buffer

        logger.info("Initialising LUFFY off-policy guidance ...")
        luffy_buffer = prepare_luffy_buffer(cfg)
        logger.info(
            "LUFFY enabled: %d off-policy traces in buffer (trace_ratio=%.2f)",
            luffy_buffer.num_off_policy,
            cfg.trace_ratio,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )

    global_step = 0
    optimizer.zero_grad()

    logger.info("Starting GRPO training for %s steps ...", cfg.num_training_steps)

    while global_step < cfg.num_training_steps:
        for batch in dataloader:
            if global_step >= cfg.num_training_steps:
                break

            fens_batch = [fen for fen in batch if fen]
            if not fens_batch:
                logger.warning("Skipping empty FEN batch")
                continue

            # Generate first-move candidates per FEN
            all_moves = generate_moves(model, tokenizer, fens_batch, cfg)

            if (global_step + 1) % cfg.logging_steps == 0:
                sample_fen = fens_batch[0]
                sample_moves = all_moves[0]
                logger.info(
                    "Step %s sample generations for %s: %s",
                    global_step + 1, sample_fen, sample_moves,
                )

            # Play out each candidate — reward is tied to the game's
            # eventual outcome (win/loss/draw), not just this one move.
            # A random share of candidates play to a natural conclusion;
            # the rest are capped at rollout_max_plies for cost.
            trajectories = []
            all_rewards = []
            for fen, moves in zip(fens_batch, all_moves):
                group_trajectories = []
                group_rewards = []
                for move_uci in moves:
                    traj = None
                    if move_uci:
                        full_game = random.random() < cfg.rollout_full_game_ratio
                        max_plies = None if full_game else cfg.rollout_max_plies
                        traj = play_rollout(model, tokenizer, fen, move_uci, cfg, max_plies=max_plies)
                    if traj is None:
                        group_trajectories.append(None)
                        group_rewards.append(-10.0)  # illegal or empty move
                    else:
                        group_trajectories.append(traj)
                        group_rewards.append(
                            compute_episode_reward(traj.outcome, traj.starting_eval)
                        )
                trajectories.append(group_trajectories)
                all_rewards.append(group_rewards)

            # Compute advantages (group-normalized per starting FEN)
            advantages = compute_advantages(all_rewards)

            # Flatten groups for the loss — rewards are already
            # group-normalized, so each (trajectory, advantage) pair can be
            # treated independently from here.
            flat_trajectories = []
            flat_advantages = []
            for group_traj, group_adv in zip(trajectories, advantages):
                for traj, adv in zip(group_traj, group_adv):
                    if traj is not None:
                        flat_trajectories.append(traj)
                        flat_advantages.append(adv)

            loss = grpo_trajectory_loss(
                model, ref_model, tokenizer, flat_trajectories, flat_advantages, cfg,
            )

            # --- LUFFY off-policy loss term ---
            luffy_metrics = {}
            if luffy_buffer is not None and luffy_buffer.num_off_policy > 0:
                from qwen_chess_rl.training.rl.luffy.training import luffy_offpolicy_loss

                luffy_result = luffy_offpolicy_loss(
                    model, tokenizer, luffy_buffer, cfg, model.device,
                )
                luffy_loss_tensor = luffy_result.get("loss_tensor")
                current_luffy_alpha = compute_luffy_alpha(cfg, global_step + 1)
                if luffy_loss_tensor is not None:
                    loss = combine_onpolicy_offpolicy_loss(
                        loss, luffy_loss_tensor, current_luffy_alpha
                    )
                luffy_metrics = {
                    "luffy_is_ratio": luffy_result.get("mean_is_ratio", 0.0),
                    "luffy_confidence": luffy_result.get("mean_confidence_weight", 0.0),
                    "luffy_num_traces": luffy_result.get("num_traces", 0),
                    "luffy_raw_loss": luffy_result.get("raw_is_loss", 0.0),
                    "luffy_alpha": current_luffy_alpha,
                }

            global_step += 1

            if loss is None or not torch.is_tensor(loss):
                logger.warning(
                    "Step %s: no differentiable GRPO loss (empty generations); skipping update",
                    global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            if not torch.isfinite(loss):
                # Defense in depth: per-trajectory non-finite ratios/losses are
                # already skipped inside grpo_trajectory_loss, but guard the
                # final combined loss too (e.g. after LUFFY blending) — a NaN
                # update here would permanently corrupt the model's weights.
                logger.warning(
                    "Step %s: non-finite loss (%s); skipping update",
                    global_step, loss.item(),
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if global_step % cfg.logging_steps == 0:
                all_rs = [r for group in all_rewards for r in group]
                mean_reward = np.mean(all_rs) if all_rs else 0
                log_msg = (
                    f"Step {global_step}/{cfg.num_training_steps} | "
                    f"Loss: {loss.item():.4f} | Mean Reward: {mean_reward:.2f}"
                )
                if luffy_metrics:
                    log_msg += (
                        f" | IS ratio: {luffy_metrics['luffy_is_ratio']:.3f}"
                        f" | Conf weight: {luffy_metrics['luffy_confidence']:.3f}"
                        f" | Traces: {luffy_metrics['luffy_num_traces']}"
                        f" | luffy_alpha: {luffy_metrics['luffy_alpha']:.3f}"
                    )
                logger.info(log_msg)

            # Save checkpoint
            if global_step % cfg.save_steps == 0:
                save_path = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(str(save_path))
                tokenizer.save_pretrained(str(save_path))
                logger.info("Checkpoint saved at step %s: %s", global_step, save_path)

    # Save final model
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("GRPO training complete. Final model saved to %s", final_path)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO train chess-playing model")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap FEN prompts sampled for rollouts (-1 = full dataset)",
    )
    parser.add_argument(
        "--luffy-alpha",
        type=float,
        default=None,
        help=(
            "Ratio of off-policy (LUFFY) to on-policy (GRPO) loss: "
            "total = (1 - luffy_alpha) * onpolicy + luffy_alpha * offpolicy. "
            f"0 = pure GRPO, 1 = pure LUFFY imitation (default: {GRPOConfig.luffy_alpha})."
        ),
    )
    parser.add_argument(
        "--luffy-alpha-final",
        type=float,
        default=None,
        help=(
            "Target luffy_alpha to anneal towards over --luffy-alpha-anneal-steps "
            f"(default: {GRPOConfig.luffy_alpha_final})."
        ),
    )
    parser.add_argument(
        "--luffy-alpha-anneal-steps",
        type=int,
        default=None,
        help=(
            "Steps over which luffy_alpha linearly anneals to luffy_alpha_final. "
            f"0 disables annealing (default: {GRPOConfig.luffy_alpha_anneal_steps})."
        ),
    )
    args = parser.parse_args()

    cfg = GRPOConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

    if args.sft_checkpoint is not None:
        cfg.sft_checkpoint = args.sft_checkpoint
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.steps is not None:
        cfg.num_training_steps = args.steps
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.luffy_alpha is not None:
        cfg.luffy_alpha = args.luffy_alpha
    if args.luffy_alpha_final is not None:
        cfg.luffy_alpha_final = args.luffy_alpha_final
    if args.luffy_alpha_anneal_steps is not None:
        cfg.luffy_alpha_anneal_steps = args.luffy_alpha_anneal_steps

    train(cfg)


if __name__ == "__main__":
    main()
