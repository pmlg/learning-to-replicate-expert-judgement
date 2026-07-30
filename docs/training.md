# Training Guide

## Two-Stage Training

### Stage 1: SFT (Supervised Fine-Tuning)

The model learns to predict the best move from FEN positions using cross-entropy loss.

**Setup:**
- Model: Qwen/Qwen-8B
- Method: QLoRA (4-bit, lora_r=16)
- Batch size: 4 per GPU (effective: 32 with grad accumulation)
- Learning rate: 2e-4 (cosine schedule)
- Hardware: 1× RTX 5090 (~24GB VRAM)

**Monitoring:**
```bash
wandb login  # optional
bash scripts/train_sft.sh
```

### Stage 2: GRPO (Reinforcement Learning)

Improves the SFT model using GRPO with Stockfish as the reward function.

**Key concepts:**
- For each position, the model generates 8 candidate moves
- Stockfish evaluates each move's quality
- Rewards are normalized within the group (zero mean, unit variance)
- The policy is updated with a clipped surrogate objective + KL penalty
