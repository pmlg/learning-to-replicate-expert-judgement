"""GRPO (Group Relative Policy Optimization) training configuration."""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """Configuration for GRPO reinforcement learning training.

    Attributes:
        sft_checkpoint: Path to the SFT-fine-tuned model checkpoint.
        output_dir: Directory for saving RL checkpoints.
        data_path: Path to the processed dataset used for GRPO rollout
            prompts (FENs). Independent of SFT's data_path — the two stages
            can be trained on different processed datasets.
        max_train_samples: Cap on FEN prompts sampled for rollouts
            (-1 = use full set). Training still runs for num_training_steps
            steps, re-shuffling and looping over this subset each epoch.
        use_4bit: Whether to load the policy/reference models in 4-bit (QLoRA).
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4').
        bnb_4bit_use_double_quant: Use double quantization for memory savings.
        num_generations: Number of responses to generate per prompt for advantage estimation.
        clip_epsilon: Clipping parameter for the surrogate loss (ε in GRPO paper).
        beta: KL penalty coefficient.
        learning_rate: Learning rate for the policy update.
        batch_size: Number of prompts per training batch.
        ppo_epochs: Number of PPO epochs per batch (training iterations on same batch).
        max_grad_norm: Gradient clipping value.
        max_length: Maximum sequence length for generation.
        temperature: Sampling temperature for move generation.
        top_p: Nucleus sampling parameter.
        eval_depth: Stockfish analysis depth for reward computation.
        num_training_steps: Total training steps.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate playing strength every N steps.
        save_total_limit: Max checkpoints to retain.
        use_luffy: Enable LUFFY off-policy guidance augmentation.
        trace_ratio: Fraction of off-policy traces in each sampled batch.
        is_clip: IS weight clipping range for off-policy corrections.
        luffy_alpha: Ratio of off-policy (LUFFY) to on-policy (GRPO) loss in
            the combined update: total = (1 - luffy_alpha) * onpolicy +
            luffy_alpha * offpolicy. 0 = pure on-policy GRPO, 1 = pure
            off-policy imitation of the LUFFY traces. Used as the starting
            value of the anneal (see luffy_alpha_final / luffy_alpha_anneal_steps).
        luffy_alpha_final: Value luffy_alpha linearly anneals *to* over
            luffy_alpha_anneal_steps. Rationale: early training benefits from
            leaning on Stockfish traces (luffy_alpha high) to quickly learn
            legal, sound moves; but staying at a high imitation weight caps
            the policy at roughly the trace engine's own strength. Annealing
            down hands more weight to on-policy GRPO once the policy is
            competent, letting reward-driven search exceed the trace
            engine's ceiling instead of just cloning it — the same
            bootstrap-then-refine curriculum used to push win rate toward
            100% rather than plateauing at "as good as the traces."
        luffy_alpha_anneal_steps: Number of steps over which luffy_alpha
            linearly interpolates from luffy_alpha to luffy_alpha_final.
            0 (default) disables annealing — luffy_alpha stays constant,
            matching the original static-blend behaviour.
        trace_engine_depth: Stockfish depth for trace generation.
        confidence_temp: Temperature for confidence-based anti-imitation weighting.
        entropy_beta: Entropy bonus coefficient (encourages exploration).
        min_eval_cp: Minimum Stockfish eval (centipawns) to accept a trace.
        trace_path: Path to pre-generated traces (JSONL); empty = generate on-the-fly.
        buffer_capacity: Experience buffer capacity for off-policy replay.
        seed: Random seed.
        report_to: Logging destination (wandb, tensorboard, none).
        rollout_full_game_ratio: Fraction of GRPO candidates per step that
            get played out to a natural conclusion (checkmate/draw) rather
            than truncated at rollout_max_plies. Winning the game is scored
            far higher than any single-move quality signal (see
            reward.WIN_REWARD), so this is what actually teaches the model
            to play for the win rather than just avoid blunders.
        rollout_max_plies: Ply cap for the non-full-game share of rollouts.
            At the cutoff, Stockfish's eval of the final position stands in
            for the outcome. Bounds per-step cost — full games can run
            40-80+ plies, each needing a model forward pass and a
            Stockfish call.
        opponent_depth: Stockfish search depth for the opponent's moves
            during rollouts. Deliberately shallow — a full-strength
            opponent would rarely lose to a still-learning policy, starving
            training of win signal.
        rollout_vet_enabled: Whether a shallow classical search vets the
            policy's own moves during a rollout (see
            :func:`qwen_chess_rl.training.rl.rollout._vet_move`), replacing
            a legal-but-losing move with the search's own choice. Applies
            only to moves generated *during* the rollout — never the seed
            first move, which must stay pure-LLM so GRPO's group-relative
            advantage still has genuine diversity to compare. Cleans up
            trajectories so the win/loss reward reflects sound play rather
            than mid-game blunders, and when the search's move is kept,
            the trajectory's reward directly reinforces the model
            producing it — a soft, reward-weighted form of imitation.
        rollout_vet_depth: Alpha-beta depth for the vetting search.
        rollout_vet_time_limit: Per-move wall-clock budget (seconds) for
            the vetting search. Kept shallow/short since it runs on every
            one of the policy's own plies across every rollout.
        rollout_vet_margin: Centipawn margin within which the policy's own
            move is kept over the search's move (matches HybridPlayer's
            default — see evaluation/stockfish_eval.py).
    """

    # Model
    sft_checkpoint: str = "checkpoints/sft/final"
    output_dir: str = "checkpoints/rl"

    # Data — separate from SFT's data_path; the two stages can use
    # different processed datasets if desired.
    data_path: str = "data/processed"
    max_train_samples: int = 500_000  # -1 = use full set

    # Quantization — the policy and reference model are each a full copy of
    # the base model; without 4-bit quantization, two bf16 9B-param copies
    # (~18GB each) don't fit on a single 24-32GB GPU.
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # GRPO hyperparameters
    num_generations: int = 8
    clip_epsilon: float = 0.2
    beta: float = 0.01
    learning_rate: float = 5e-6
    batch_size: int = 8
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5

    # Generation
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9

    # Reward / evaluation
    eval_depth: int = 15

    # Training loop
    num_training_steps: int = 500_000
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 200
    save_total_limit: int = 5

    # LUFFY off-policy guidance
    use_luffy: bool = True  # Master toggle for LUFFY augmentation
    trace_ratio: float = 0.3  # Fraction of off-policy traces in each batch
    is_clip: float = 0.2  # IS weight clipping range
    luffy_alpha: float = 0.85  # Ratio of off-policy (LUFFY) to on-policy (GRPO) loss
    luffy_alpha_final: float = 0.85  # Annealed target for luffy_alpha (see docstring)
    luffy_alpha_anneal_steps: int = 0  # 0 = constant luffy_alpha (no anneal)
    trace_engine_depth: int = 18  # Stockfish depth for trace generation
    confidence_temp: float = 1.0  # Temperature for confidence-based weighting
    entropy_beta: float = 0.01  # Entropy bonus coefficient (anti-imitation)
    min_eval_cp: int = 0  # Minimum Stockfish eval to accept a trace
    trace_path: str = ""  # Path to pre-generated traces (JSONL); empty = generate on-the-fly
    buffer_capacity: int = 10_000  # Experience buffer capacity

    # Multi-ply rollouts — reward is tied to actually winning the game,
    # not just per-move soundness. See reward.py's WIN/LOSS/DRAW_REWARD.
    rollout_full_game_ratio: float = 0.5  # fraction played to conclusion vs. capped
    rollout_max_plies: int = 10  # ply cap for the capped share
    opponent_depth: int = 3  # shallow Stockfish opponent during rollouts

    # Rollout move vetting — a shallow classical search corrects legal-but-
    # losing moves made *during* a rollout (never the seed first move; see
    # docstring). See rollout.py's _vet_move.
    rollout_vet_enabled: bool = True
    rollout_vet_depth: int = 4
    rollout_vet_time_limit: float = 0.3
    rollout_vet_margin: float = 50.0  # centipawns

    # General
    seed: int = 42
    report_to: str = "none"
