"""Supervised Fine-Tuning (SFT) for Qwen-8B on chess positions using QLoRA."""

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
    """Load Qwen-8B with 4-bit quantization and prepare for LoRA training.

    Args:
        cfg: Training configuration.

    Returns:
        (model, tokenizer) tuple.
    """
    logger.info("Loading model: %s", cfg.model_name)
    quantization_config = create_quantization_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = create_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
    )

    logger.info("Tokenizing validation data ...")
    val_ds = val_ds.map(
        tokenize_fn,
        remove_columns=val_ds.column_names,
        batched=False,
        desc="Tokenizing validation data",
    )

    return train_ds, val_ds


def train(cfg: SFTConfig):
    """Run the full SFT training loop.

    Args:
        cfg: Training configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        report_to=cfg.report_to,
        logging_dir=str(output_dir / "logs"),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        label_pad_token_id=-100,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training ...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("SFT training complete. Model saved to %s", final_path)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT train Qwen-8B on chess positions")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, default=SFTConfig.model_name)
    parser.add_argument("--data-path", type=str, default=SFTConfig.data_path)
    parser.add_argument("--output-dir", type=str, default=SFTConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=SFTConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SFTConfig.learning_rate)
    parser.add_argument("--max-steps", type=int, default=SFTConfig.max_steps)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
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
    if args.no_wandb:
        cfg.report_to = "none"

    train(cfg)


if __name__ == "__main__":
    main()
