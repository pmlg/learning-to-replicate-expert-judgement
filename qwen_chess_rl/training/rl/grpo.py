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

    Optionally loaded in 4-bit if cfg.use_4bit is True (for memory efficiency
    on larger models). For smaller models with fp16, full precision may fit.

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

    # Prepare quantization config if enabled
    quantization_config = None
    if cfg.use_4bit:
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

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        if trainable and cfg.use_4bit:
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
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(
            cfg.sft_checkpoint,
            **model_kwargs,
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
    tokenizer,
    trajectories: list,
    advantages: list[float],
    cfg: GRPOConfig,
    ref_model=None,
    loss_scale: float = 1.0,
) -> tuple[float | None, int]:
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

    Backwards through each trajectory's loss immediately after computing
    it, rather than summing every trajectory into one tensor before a
    single final backward() call. Full-game rollouts can span 60-100+
    plies, each needing its own kept-alive forward pass for the ratio
    computation above; retaining dozens of those graphs simultaneously
    (one per trajectory in the batch) is what exhausted VRAM at 9B scale
    — backwarding per-trajectory frees each graph before the next
    trajectory's forward pass begins. Gradients accumulate in
    ``model.parameters()[*].grad`` exactly as they would from one combined
    backward() call; the caller still does a single optimizer.step().

    Args:
        model: Current policy model (PEFT-wrapped in the common case).
        tokenizer: HuggingFace tokenizer.
        trajectories: One rollout.RolloutResult per sampled candidate.
        advantages: Group-normalized advantage per trajectory (same order
            and length as trajectories).
        cfg: GRPO configuration.
        ref_model: Optional separate frozen reference model. Only needed
            when ``model`` isn't a PEFT model (no LoRA adapter to disable) —
            with LoRA, the reference forward pass instead runs through
            ``model.disable_adapter()`` so the frozen base weights double as
            the reference without a second full model copy in VRAM (at 9B
            in bf16 that second copy would be another ~18GB).
        loss_scale: Multiplies each trajectory's loss before backward (e.g.
            ``1 - luffy_alpha`` so this on-policy term and LUFFY's
            off-policy term combine to the documented weighted sum even
            though each is backwarded independently rather than summed
            into one tensor first).

    Returns:
        ``(mean loss value across valid trajectories, count of valid
        trajectories)`` — ``(None, 0)`` when no valid trajectories exist.
        The mean is for logging only; gradients have already been applied.
    """
    num_trajectories = len(trajectories)
    if num_trajectories == 0:
        return None, 0

    total_loss_value = 0.0
    num_samples = 0
    device = next(model.parameters()).device
    use_disable_adapter = ref_model is None and hasattr(model, "disable_adapter")
    if ref_model is None and not use_disable_adapter:
        raise ValueError(
            "grpo_trajectory_loss needs either a PEFT model (for "
            "disable_adapter) or an explicit ref_model."
        )

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

            # Gradient-checkpointed: each ply is an independent forward pass
            # (fresh FEN -> fresh tokenization, no shared KV-cache across
            # plies), so checkpointing it discards its activations
            # immediately rather than keeping them alive until this
            # trajectory's single backward() call below — without this, a
            # long full-game trajectory (60-100+ plies) keeps every one of
            # its plies' activations resident simultaneously, which is what
            # was driving peak VRAM at 9B scale, independent of batch_size.
            def _policy_logits(input_ids, attention_mask):
                return model(input_ids, attention_mask=attention_mask).logits

            policy_logits = torch.utils.checkpoint.checkpoint(
                _policy_logits,
                full_input.unsqueeze(0),
                full_mask.unsqueeze(0),
                use_reentrant=False,
            )
            log_probs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
            ply_log_probs = log_probs[:, -len(move_ids):, :].gather(
                2, move_ids.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).sum()

            with torch.no_grad():
                if use_disable_adapter:
                    with model.disable_adapter():
                        ref_outputs = model(
                            full_input.unsqueeze(0), attention_mask=full_mask.unsqueeze(0),
                        )
                else:
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

        # Backward now, while this trajectory's graph is the only one
        # alive — normalized by num_trajectories (not num_samples, which
        # isn't known until the loop finishes) so a handful of skipped
        # trajectories don't require rescaling already-applied gradients.
        (sample_loss * (loss_scale / num_trajectories)).backward()
        total_loss_value += sample_loss.item()
        num_samples += 1

    if num_samples == 0:
        return None, 0

    return total_loss_value / num_samples, num_samples


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


def compute_opponent_depth(cfg: GRPOConfig, step: int) -> int:
    """Compute the current rollout opponent depth under the anneal schedule.

    Linearly interpolates (rounded to the nearest integer depth) from
    ``cfg.opponent_depth`` to ``cfg.opponent_depth_final`` over
    ``cfg.opponent_depth_anneal_steps`` steps, then holds at the final
    value. With ``opponent_depth_anneal_steps == 0`` (default), returns
    ``cfg.opponent_depth`` unchanged — the anneal is opt-in.

    Args:
        cfg: GRPO configuration.
        step: Current global training step (1-indexed, as incremented in
            the training loop).

    Returns:
        The Stockfish search depth to use for the rollout opponent this step.
    """
    if cfg.opponent_depth_anneal_steps <= 0:
        return cfg.opponent_depth
    frac = min(step / cfg.opponent_depth_anneal_steps, 1.0)
    return round(cfg.opponent_depth + frac * (cfg.opponent_depth_final - cfg.opponent_depth))


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

    # Load the policy model. When it's a PEFT/LoRA model, the frozen base
    # weights double as the reference model via disable_adapter() inside
    # grpo_trajectory_loss — a second full model copy would cost another
    # ~18GB of VRAM at 9B scale for no benefit, since the "reference" and
    # "policy" share the exact same base weights anyway. Only load a
    # genuinely separate reference model for a non-PEFT (full fine-tune)
    # checkpoint, where there's no adapter to disable.
    model, tokenizer = load_rl_model(cfg, trainable=True)
    ref_model = None
    if not hasattr(model, "disable_adapter"):
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
            current_opponent_depth = compute_opponent_depth(cfg, global_step + 1)
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
                        try:
                            traj = play_rollout(
                                model, tokenizer, fen, move_uci, cfg, max_plies=max_plies,
                                opponent_depth=current_opponent_depth,
                            )
                        except Exception as exc:
                            # A rollout crash (e.g. the Stockfish subprocess
                            # dying mid-game) must not take down a training
                            # run that's hours into progress — treat it the
                            # same as an illegal/empty move.
                            logger.warning("Rollout failed for %s: %s", fen, exc)
                    if traj is None:
                        group_trajectories.append(None)
                        group_rewards.append(-10.0)  # illegal or empty move
                    else:
                        group_trajectories.append(traj)
                        group_rewards.append(
                            compute_episode_reward(
                                traj.outcome, traj.starting_eval, traj.cutoff_eval,
                            )
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

            # current_luffy_alpha is needed as grpo_trajectory_loss's
            # loss_scale, so it's computed before that call rather than
            # alongside the LUFFY block below (it only depends on the step
            # number, not on anything computed in between).
            current_luffy_alpha = compute_luffy_alpha(cfg, global_step + 1) if luffy_buffer is not None else 0.0

            # Backwards per-trajectory internally (see docstring) rather
            # than returning one tensor for a single backward() call here —
            # full-game trajectories are too numerous/long to keep every
            # one's graph alive simultaneously at 9B scale.
            onpolicy_loss_value, onpolicy_num_samples = grpo_trajectory_loss(
                model, tokenizer, flat_trajectories, flat_advantages, cfg,
                ref_model=ref_model, loss_scale=(1 - current_luffy_alpha),
            )

            # --- LUFFY off-policy loss term ---
            luffy_metrics = {}
            luffy_loss_value = None
            if luffy_buffer is not None and luffy_buffer.num_off_policy > 0:
                from qwen_chess_rl.training.rl.luffy.training import luffy_offpolicy_loss

                luffy_result = luffy_offpolicy_loss(
                    model, tokenizer, luffy_buffer, cfg, model.device,
                )
                luffy_loss_tensor = luffy_result.get("loss_tensor")
                if luffy_loss_tensor is not None and torch.isfinite(luffy_loss_tensor):
                    (current_luffy_alpha * luffy_loss_tensor).backward()
                    luffy_loss_value = luffy_loss_tensor.item()
                luffy_metrics = {
                    "luffy_is_ratio": luffy_result.get("mean_is_ratio", 0.0),
                    "luffy_confidence": luffy_result.get("mean_confidence_weight", 0.0),
                    "luffy_num_traces": luffy_result.get("num_traces", 0),
                    "luffy_raw_loss": luffy_result.get("raw_is_loss", 0.0),
                    "luffy_alpha": current_luffy_alpha,
                }

            global_step += 1

            if onpolicy_num_samples == 0 and luffy_loss_value is None:
                logger.warning(
                    "Step %s: no differentiable GRPO loss (empty generations); skipping update",
                    global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # Non-finite trajectories/LUFFY batches are already filtered
            # out before their individual backward() calls above, so
            # whatever gradients are in .grad at this point are safe to
            # apply.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if global_step % cfg.logging_steps == 0:
                all_rs = [r for group in all_rewards for r in group]
                mean_reward = np.mean(all_rs) if all_rs else 0
                # Combined value is for display only — gradients from each
                # term were already applied via their own backward() calls.
                display_loss = 0.0
                if onpolicy_num_samples:
                    display_loss += (1 - current_luffy_alpha) * onpolicy_loss_value
                if luffy_loss_value is not None:
                    display_loss += current_luffy_alpha * luffy_loss_value
                log_msg = (
                    f"Step {global_step}/{cfg.num_training_steps} | "
                    f"Loss: {display_loss:.4f} | Mean Reward: {mean_reward:.2f} | "
                    f"Opponent depth: {current_opponent_depth}"
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
    parser.add_argument("--output-dir", type=str, default=None)
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
    parser.add_argument(
        "--opponent-depth-final",
        type=int,
        default=None,
        help=(
            "Target rollout opponent Stockfish depth to anneal towards over "
            f"--opponent-depth-anneal-steps (default: {GRPOConfig.opponent_depth_final})."
        ),
    )
    parser.add_argument(
        "--opponent-depth-anneal-steps",
        type=int,
        default=None,
        help=(
            "Steps over which the rollout opponent depth linearly anneals to "
            f"opponent_depth_final. 0 disables annealing (default: {GRPOConfig.opponent_depth_anneal_steps})."
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
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
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
    if args.opponent_depth_final is not None:
        cfg.opponent_depth_final = args.opponent_depth_final
    if args.opponent_depth_anneal_steps is not None:
        cfg.opponent_depth_anneal_steps = args.opponent_depth_anneal_steps

    train(cfg)


if __name__ == "__main__":
    main()
