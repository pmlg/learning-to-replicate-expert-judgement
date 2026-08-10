"""SFT training configuration for QLoRA fine-tuning of Qwen-8B."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning of Qwen-8B on chess positions.

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

    # Model
    model_name: str = "Qwen/Qwen-8B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training
    output_dir: str = "checkpoints/sft"
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 means use epochs
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3

    # Precision
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # Data
    seed: int = 42
    data_path: str = "data/processed"
    # Full 2M is not required for a useful LoRA run; 50k is a strong default.
    max_train_samples: int = 50_000
    max_eval_samples: int = 2_000
    max_length: int = 512
    report_to: str = "none"
