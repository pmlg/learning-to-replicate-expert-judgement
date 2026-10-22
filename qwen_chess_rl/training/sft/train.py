"""Supervised Fine-Tuning (SFT) for Qwen on chess positions using QLoRA."""

import logging
import sys
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from qwen_chess_rl.training.sft.config import SFTConfig
from qwen_chess_rl.data.dataset import tokenize_example

logger = logging.getLogger(__name__)


def get_base_hidden_size(model) -> int:
    """Get the text-decoder hidden size, handling Qwen3.5's nested text_config."""
    config = model.config
    text_config = getattr(config, "text_config", None)
    return (text_config or config).hidden_size


class ChessDataCollator:
    """Wraps a base collator, batching ``cp_target`` separately.

    ``DataCollatorForSeq2Seq`` only knows how to pad token-sequence fields
    (input_ids/attention_mask/labels); the scalar ``cp_target`` auxiliary
    label has to be pulled out before delegating to it and re-attached as a
    plain tensor afterward.
    """

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: list[dict]) -> dict:
        cp_targets = [f.pop("cp_target") for f in features]
        batch = self.base_collator(features)
        batch["cp_target"] = torch.tensor(cp_targets, dtype=torch.float32)
        return batch


class ChessSFTTrainer(Trainer):
    """Trainer that adds an auxiliary HL-Gauss value loss to the move CE loss.

    The value head reads the hidden state at the last prompt token (the same
    position whose logits predict the first move token) and predicts the
    position's Stockfish eval — an auxiliary chess-understanding signal
    alongside pure move imitation. See SFTConfig.use_hl_gauss_aux_loss.
    """

    def __init__(self, *args, value_head=None, aux_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_head = value_head
        self.aux_weight = aux_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cp_target = inputs.pop("cp_target", None)
        outputs = model(**inputs, output_hidden_states=self.value_head is not None)
        loss = outputs.loss

        if self.value_head is not None and cp_target is not None:
            labels = inputs["labels"]
            # First non-masked label position == first generated (move) token;
            # the hidden state one position earlier is what predicts it.
            first_label_idx = (labels != -100).float().argmax(dim=-1)
            value_idx = (first_label_idx - 1).clamp(min=0)
            hidden = outputs.hidden_states[-1]
            value_embed = hidden[torch.arange(hidden.size(0), device=hidden.device), value_idx]
            value_loss = self.value_head(value_embed.float(), cp_target.to(hidden.device))
            loss = loss + self.aux_weight * value_loss

        return (loss, outputs) if return_outputs else loss


def create_quantization_config(cfg: SFTConfig) -> BitsAndBytesConfig:
    """Create 4-bit quantization config for QLoRA.

    Args:
        cfg: Training configuration.

    Returns:
        BitsAndBytesConfig for 4-bit quantization.
    """
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def create_lora_config(cfg: SFTConfig) -> LoraConfig:
    """Create LoRA configuration for QLoRA fine-tuning.

    Args:
        cfg: Training configuration.

    Returns:
        LoraConfig for PEFT.
    """
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(cfg: SFTConfig):
    """Load the configured Qwen model with optional 4-bit quantization and LoRA.

    Args:
        cfg: Training configuration.

    Returns:
        (model, tokenizer) tuple.
    """
    logger.info("Loading model: %s", cfg.model_name)

    # Determine torch dtype
    if cfg.bf16:
        torch_dtype = torch.bfloat16
    elif cfg.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load model with optional quantization
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    if cfg.use_4bit:
        quantization_config = create_quantization_config(cfg)
        kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    hidden_size = get_base_hidden_size(model)

    # Prepare for k-bit training if using quantization
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA, or train every parameter directly (see use_lora docstring).
    if cfg.use_lora:
        lora_config = create_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Full fine-tuning: trainable params: %d || all params: %d || trainable%%: %.4f",
            trainable, total, 100 * trainable / total,
        )

    # Auxiliary HL-Gauss value head — a submodule of `model` so its params
    # ride along in model.parameters() (picked up by Trainer's optimizer)
    # without needing a separate optimizer/param group. Kept in float32
    # regardless of the base model's dtype; HLGaussLoss forces fp32 math
    # internally anyway (see hl_gauss_pytorch.hl_gauss.HLGaussLossFromSupport).
    if cfg.use_hl_gauss_aux_loss:
        from hl_gauss_pytorch import HLGaussLayer

        model.value_head = HLGaussLayer(
            dim=hidden_size,
            hl_gauss_loss=dict(
                min_value=-cfg.hl_gauss_cp_clip,
                max_value=cfg.hl_gauss_cp_clip,
                num_bins=cfg.hl_gauss_num_bins,
            ),
        ).to(device=next(model.parameters()).device, dtype=torch.float32)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_datasets(cfg: SFTConfig, tokenizer):
    """Load and tokenize the train/validation datasets.

    Args:
        cfg: Training configuration.
        tokenizer: HuggingFace tokenizer.

    Returns:
        (train_dataset, eval_dataset) as tokenized datasets.
    """
    data_path = Path(cfg.data_path)
    logger.info("Loading datasets from %s", data_path)

    train_ds = load_from_disk(str(data_path / "train"))
    val_ds = load_from_disk(str(data_path / "val"))

    if cfg.max_train_samples > 0 and len(train_ds) > cfg.max_train_samples:
        train_ds = train_ds.shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))
        logger.info("Using %d / full train examples", len(train_ds))
    else:
        logger.info("Using full train set (%d examples)", len(train_ds))

    if cfg.max_eval_samples > 0 and len(val_ds) > cfg.max_eval_samples:
        val_ds = val_ds.shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))
        logger.info("Using %d / full val examples", len(val_ds))

    # Tokenize datasets
    def tokenize_fn(examples):
        return tokenize_example(
            examples,
            tokenizer,
            max_length=cfg.max_length,
        )

    logger.info("Tokenizing training data (this may take a while) ...")
    train_ds = train_ds.map(
        tokenize_fn,
        remove_columns=train_ds.column_names,
        batched=False,
        desc="Tokenizing training data",
        load_from_cache_file=True,
    )

    logger.info("Tokenizing validation data ...")
    val_ds = val_ds.map(
        tokenize_fn,
        remove_columns=val_ds.column_names,
        batched=False,
        desc="Tokenizing validation data",
        load_from_cache_file=True,
    )

    return train_ds, val_ds


def _disable_wandb(reason: str) -> str:
    """Force Trainer away from wandb even if env still has a key."""
    import os

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    logger.warning("%s; disabling wandb (report_to=none)", reason)
    return "none"


def resolve_report_to(report_to: str) -> str:
    """Return a usable report_to, falling back when wandb auth is unavailable.

    ``wandb.login()`` can return True when a key is present even if the key is
    invalid; verify with the API before enabling Trainer's wandb callback.
    """
    if report_to != "wandb":
        return report_to

    import os

    api_key = (os.environ.get("WANDB_API_KEY") or "").strip()
    if not api_key:
        return _disable_wandb("WANDB_API_KEY unset")

    try:
        import wandb

        # Avoid interactive login prompts during training.
        os.environ["WANDB_API_KEY"] = api_key
        api = wandb.Api(api_key=api_key, timeout=15)
        # Forces a real authenticated request; raises if the key is bad.
        _ = api.viewer
    except Exception as exc:
        return _disable_wandb(f"wandb auth failed ({exc})")

    return "wandb"


def train(cfg: SFTConfig, resume_from_checkpoint: str | bool | None = None):
    """Run the full SFT training loop.

    Args:
        cfg: Training configuration.
        resume_from_checkpoint: Path to a specific checkpoint directory to
            resume from, ``True`` to auto-resume from the latest checkpoint
            in ``cfg.output_dir``, or ``None`` to start fresh. A run this
            long (hours) losing all progress to a crash or restart otherwise
            would be costly.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = resolve_report_to(cfg.report_to)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load datasets
    train_dataset, eval_dataset = load_datasets(cfg, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        seed=cfg.seed,
        report_to=report_to,
        # cp_target isn't a model.forward() arg, so the Trainer's default
        # column-pruning would drop it before it ever reaches the collator.
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    data_collator = ChessDataCollator(
        DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            padding=True,
            label_pad_token_id=-100,
        )
    )

    # Initialize trainer
    trainer = ChessSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        value_head=getattr(model, "value_head", None),
        aux_weight=cfg.hl_gauss_aux_weight,
    )

    # Train
    logger.info("Starting SFT training (resume_from_checkpoint=%s) ...", resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("SFT training complete. Model saved to %s", final_path)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT train Qwen on chess positions")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, default=SFTConfig.model_name)
    parser.add_argument("--data-path", type=str, default=SFTConfig.data_path)
    parser.add_argument("--output-dir", type=str, default=SFTConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=SFTConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SFTConfig.learning_rate)
    parser.add_argument("--max-steps", type=int, default=SFTConfig.max_steps)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=SFTConfig.max_train_samples,
        help="Cap training examples (-1 = full dataset)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=SFTConfig.max_eval_samples,
        help="Cap validation examples (-1 = full dataset)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir to resume from, or 'auto' for the latest checkpoint in --output-dir",
    )
    args = parser.parse_args()

    cfg = SFTConfig()
    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

    # CLI overrides
    if args.model_name:
        cfg.model_name = args.model_name
    if args.data_path:
        cfg.data_path = args.data_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.max_steps:
        cfg.max_steps = args.max_steps
    cfg.max_train_samples = args.max_train_samples
    cfg.max_eval_samples = args.max_eval_samples
    if args.no_wandb:
        cfg.report_to = "none"

    resume = args.resume_from_checkpoint
    if resume == "auto":
        resume = True
    train(cfg, resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
