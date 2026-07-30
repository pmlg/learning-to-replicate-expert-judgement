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

    # General
    seed: int = 42
    report_to: str = "wandb"
