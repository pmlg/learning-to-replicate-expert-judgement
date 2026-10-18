"""SFT training configuration for QLoRA fine-tuning of Qwen chess models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning of a Qwen model on chess positions.

    Attributes:
        model_name: HuggingFace model identifier.
        lora_r: LoRA rank dimension.
        lora_alpha: LoRA scaling parameter.
        lora_dropout: Dropout rate for LoRA layers.
        target_modules: Modules to apply LoRA to.
        use_4bit: Whether to use 4-bit quantization (QLoRA).
        bnb_4bit_compute_dtype: Compute dtype for 4-bit (bfloat16 recommended).
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4').
        bnb_4bit_use_double_quant: Use double quantization for memory savings.
        output_dir: Directory for saving checkpoints.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Steps for gradient accumulation.
        learning_rate: Peak learning rate.
        lr_scheduler_type: Learning rate schedule ('cosine' or 'cosine_with_restarts').
        warmup_steps: Linear warmup steps.
        max_grad_norm: Gradient clipping value.
        num_train_epochs: Number of training epochs.
        max_steps: Max training steps (overrides epochs if set).
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        save_total_limit: Max number of checkpoints to keep.
        bf16: Use bfloat16 precision.
        fp16: Use float16 precision (fallback if bf16 not supported).
        gradient_checkpointing: Enable gradient checkpointing for memory.
        optim: Optimizer to use (e.g. 'paged_adamw_8bit').
        seed: Random seed.
        data_path: Path to processed dataset.
        max_train_samples: Cap on training examples (-1 = use full set).
        max_eval_samples: Cap on validation examples (-1 = use full set).
        max_length: Maximum sequence length.
        report_to: Logging destination (wandb, tensorboard, none).
    """

    # Model — switched from Qwen3.5-9B to Qwen3-1.7B (operator goal change):
    # dramatically smaller, so training fits comfortably in bf16 without
    # quantization and runs much faster per step, at the cost of raw
    # capacity. Qwen3-1.7B is a standard (non-hybrid-attention) Qwen3
    # architecture — none of the Qwen3.5 hybrid-attention special-casing
    # elsewhere in this codebase (get_base_hidden_size's nested text_config
    # fallback, flash-linear-attention) is needed for it, but those paths
    # degrade gracefully to the plain case so nothing had to change there.
    model_name: str = "Qwen/Qwen3-1.7B"
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ]
    )

    # Quantization
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training
    output_dir: str = "checkpoints/sft-1.7b"
    # At 1.7B, per-device activations are tiny relative to the 9B recipe's
    # batch_size=2 — raised to 16 (with accumulation cut 24->3, keeping the
    # same effective batch size of 48) purely for throughput; VRAM headroom
    # was confirmed empirically before launching the real run.
    batch_size: int = 16
    gradient_accumulation_steps: int = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 means use epochs
    logging_steps: int = 25
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3

    # Precision — RTX 5090 (Blackwell) natively supports bf16; it has a wider
    # exponent range than fp16 so training stays stable without loss scaling.
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"

    # Auxiliary value loss — the Lichess dataset already carries a Stockfish
    # `cp` eval per position, otherwise unused by SFT. Predicting it from the
    # same trunk the policy head reads from (à la AlphaZero's value head)
    # gives the model an auxiliary chess-understanding signal on top of pure
    # move-imitation. Uses hl-gauss-pytorch's histogram loss (Imani et al.,
    # https://arxiv.org/abs/1806.04613) rather than MSE regression, which the
    # HL-Gauss paper (https://arxiv.org/abs/2403.03950) found regresses
    # scalar targets more accurately for this kind of value prediction.
    use_hl_gauss_aux_loss: bool = True
    hl_gauss_aux_weight: float = 0.1
    hl_gauss_num_bins: int = 128
    hl_gauss_cp_clip: float = 1000.0

    # Data
    seed: int = 42
    data_path: str = "data/processed"
    # Use the full dataset — this task needs broad exposure to board
    # patterns; a 50k slice (2.5% of 1.96M) left the model undertrained.
    max_train_samples: int = 500_000
    max_eval_samples: int = 500
    # Real examples (prompt + move + EOS) are ~90 tokens; 512 wasted ~5.5x
    # compute on padding every example to a fixed length. 128 covers actual
    # content with margin and lets the full dataset run in the same budget.
    max_length: int = 128
    report_to: str = "none"
