"""GRPO (Group Relative Policy Optimization) training configuration."""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """Configuration for GRPO reinforcement learning training.

    Attributes:
        sft_checkpoint: Path to the SFT-fine-tuned model checkpoint.
        output_dir: Directory for saving RL checkpoints.
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
        offpolicy_weight: Weight for the off-policy loss term.
        trace_engine_depth: Stockfish depth for trace generation.
        confidence_temp: Temperature for confidence-based anti-imitation weighting.
        entropy_beta: Entropy bonus coefficient (encourages exploration).
        min_eval_cp: Minimum Stockfish eval (centipawns) to accept a trace.
        trace_path: Path to pre-generated traces (JSONL); empty = generate on-the-fly.
        buffer_capacity: Experience buffer capacity for off-policy replay.
        seed: Random seed.
        report_to: Logging destination (wandb, tensorboard, none).
    """

    # Model
    sft_checkpoint: str = "checkpoints/sft/final"
    output_dir: str = "checkpoints/rl"

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
    num_training_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 200
    save_total_limit: int = 5

    # LUFFY off-policy guidance
    use_luffy: bool = False  # Master toggle for LUFFY augmentation
    trace_ratio: float = 0.3  # Fraction of off-policy traces in each batch
    is_clip: float = 0.2  # IS weight clipping range
    offpolicy_weight: float = 0.5  # Weight for off-policy loss term
    trace_engine_depth: int = 18  # Stockfish depth for trace generation
    confidence_temp: float = 1.0  # Temperature for confidence-based weighting
    entropy_beta: float = 0.01  # Entropy bonus coefficient (anti-imitation)
    min_eval_cp: int = 0  # Minimum Stockfish eval to accept a trace
    trace_path: str = ""  # Path to pre-generated traces (JSONL); empty = generate on-the-fly
    buffer_capacity: int = 10_000  # Experience buffer capacity

    # General
    seed: int = 42
    report_to: str = "wandb"
