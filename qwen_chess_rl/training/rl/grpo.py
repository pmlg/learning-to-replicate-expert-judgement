"""GRPO (Group Relative Policy Optimization) training loop for chess RL."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_chess_rl.data.dataset import ChessPositionDataset, collate_fn, format_fen_prompt
from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.reward import batch_compute_rewards

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


def load_rl_model(cfg: GRPOConfig):
    """Load the SFT-fine-tuned model for RL training.

    Args:
        cfg: GRPO configuration.

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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_checkpoint,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Ensure we're using the LoRA-adapter model
    if hasattr(model, "peft_config"):
        logger.info("Loaded PEFT model with LoRA adapters.")
    else:
        logger.warning("Model does not appear to have LoRA adapters loaded.")

    model.train()
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

    # Generate
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

    # Decode
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract move from generated text (everything after "assistant\n")
    all_moves = [[] for _ in range(len(fens))]
    for idx, text in zip(fen_indices, decoded):
        if "assistant" in text:
            move = text.split("assistant")[-1].strip()
        else:
            move = text.strip()
        # Take only the first word (UCI move)
        move = move.split()[0] if move else ""
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


def grpo_loss(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_ids: list[torch.Tensor],
    advantages: list[list[float]],
    cfg: GRPOConfig,
) -> torch.Tensor:
    """Compute the GRPO loss: clipped surrogate objective + KL penalty.

    The loss combines:
    1. Clipped importance-sampling policy gradient.
    2. KL divergence penalty against the reference model.

    Args:
        model: Current policy model.
        ref_model: Reference model (SFT checkpoint, frozen).
        input_ids: Tokenized input prompts.
        attention_mask: Attention mask for prompts.
        generated_ids: List of tensors, each containing generated token IDs.
        advantages: Group-normalized advantages per generation.
        cfg: GRPO configuration.

    Returns:
        Scalar loss tensor.
    """
    total_loss = 0.0
    num_groups = 0

    current_idx = 0
    for group_gen_ids, group_adv in zip(generated_ids, advantages):
        for gen_ids, adv in zip(group_gen_ids, group_adv):
            if gen_ids.numel() == 0:
                continue

            # Concatenate prompt + generation
            full_input = torch.cat([input_ids[current_idx], gen_ids], dim=-1)
            full_mask = torch.cat([
                attention_mask[current_idx],
                torch.ones_like(gen_ids),
            ], dim=-1)

            # Get log-probs from current policy
            outputs = model(full_input.unsqueeze(0), attention_mask=full_mask.unsqueeze(0))
            logits = outputs.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            gen_log_probs = log_probs[:, -len(gen_ids):, :].gather(
                2, gen_ids.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).sum()

            # Get log-probs from reference model (no gradient)
            with torch.no_grad():
                ref_outputs = ref_model(
                    full_input.unsqueeze(0),
                    attention_mask=full_mask.unsqueeze(0),
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                ref_gen_log_probs = ref_log_probs[:, -len(gen_ids):, :].gather(
                    2, gen_ids.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1).sum()

            # Importance ratio
            ratio = torch.exp(gen_log_probs - ref_gen_log_probs)

            # Clipped surrogate objective
            clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
            policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)

            # KL penalty (from ref model)
            kl_div = gen_log_probs - ref_gen_log_probs
            kl_penalty = cfg.beta * kl_div

            total_loss += policy_loss + kl_penalty
            num_groups += 1

        current_idx += 1

    return total_loss / max(num_groups, 1)


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
    model, tokenizer = load_rl_model(cfg)
    ref_model, _ = load_rl_model(cfg)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load dataset
    dataset = ChessPositionDataset(
        data_path="data/processed",
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
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

            # Move to device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Decode FENs from input_ids for reward computation
            fens_batch = []
            for ids in batch["input_ids"]:
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                # Extract FEN from prompt
                if "Position:" in decoded:
                    fen_part = decoded.split("Position:")[-1].split("\n")[0].strip()
                    fens_batch.append(fen_part)
                else:
                    fens_batch.append("")

            # Generate candidate moves
            all_moves = generate_moves(model, tokenizer, fens_batch, cfg)

            # Compute rewards using Stockfish
            all_rewards = []
            for fen, moves in zip(fens_batch, all_moves):
                positions = [(fen, m) for m in moves if m]
                rewards = batch_compute_rewards(positions, eval_depth=cfg.eval_depth)
                # Extract total_reward values
                r_values = [r["total_reward"] for r in rewards]
                # Pad to num_generations with -10 if no valid moves
                while len(r_values) < len(moves):
                    r_values.append(-10.0)
                all_rewards.append(r_values)

            # Compute advantages
            advantages = compute_advantages(all_rewards)

            # Compute GRPO loss
            generated_ids = []
            for moves in all_moves:
                gen_ids = []
                for move_uci in moves:
                    if move_uci:
                        ids = tokenizer.encode(move_uci, return_tensors="pt")
                        gen_ids.append(ids.squeeze(0).to(model.device))
                    else:
                        gen_ids.append(torch.tensor([], device=model.device))
                generated_ids.append(gen_ids)

            loss = grpo_loss(
                model, ref_model,
                batch["input_ids"], batch["attention_mask"],
                generated_ids, advantages, cfg,
            )

            # --- LUFFY off-policy loss term ---
            luffy_metrics = {}
            if luffy_buffer is not None and luffy_buffer.num_off_policy > 0:
                from qwen_chess_rl.training.rl.luffy.training import luffy_offpolicy_loss

                luffy_result = luffy_offpolicy_loss(
                    model, tokenizer, luffy_buffer, cfg, model.device,
                )
                luffy_loss_tensor = luffy_result.get("loss_tensor")
                if luffy_loss_tensor is not None:
                    loss = loss + luffy_loss_tensor
                luffy_metrics = {
                    "luffy_is_ratio": luffy_result.get("mean_is_ratio", 0.0),
                    "luffy_confidence": luffy_result.get("mean_confidence_weight", 0.0),
                    "luffy_num_traces": luffy_result.get("num_traces", 0),
                    "luffy_raw_loss": luffy_result.get("raw_is_loss", 0.0),
                }

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

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
    parser.add_argument("--sft-checkpoint", type=str, default=GRPOConfig.sft_checkpoint)
    parser.add_argument("--steps", type=int, default=GRPOConfig.num_training_steps)
    parser.add_argument("--lr", type=float, default=GRPOConfig.learning_rate)
    args = parser.parse_args()

    cfg = GRPOConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

    cfg.sft_checkpoint = args.sft_checkpoint or cfg.sft_checkpoint
    cfg.num_training_steps = args.steps or cfg.num_training_steps
    if args.lr:
        cfg.learning_rate = args.lr

    train(cfg)


if __name__ == "__main__":
    main()
