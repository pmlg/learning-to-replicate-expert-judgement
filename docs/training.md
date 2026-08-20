# Training Guide

## Two-Stage Training

### Stage 1: SFT (Supervised Fine-Tuning)

The model learns to predict the best move from FEN positions using cross-entropy loss.

**Setup:**
- Model: Qwen/Qwen3-4B (override with `--model` / `model_name`)
- Method: QLoRA (4-bit, lora_r=16)
- Batch size: 4 per GPU (effective: 32 with grad accumulation)
- Learning rate: 2e-4 (cosine schedule)
- Hardware: GPU with ~16GB+ VRAM for 4B QLoRA

**Monitoring:**
```bash
wandb login  # optional
bash scripts/train_sft.sh
```

### Stage 2: GRPO (Reinforcement Learning)

Improves the SFT model using GRPO, with reward tied to actually winning
games against a shallow Stockfish opponent — not just per-move quality.

**Key concepts:**
- For each position, the model generates 8 candidate first moves
- Each candidate is played out into a full rollout (see
  [rollout.py](../qwen_chess_rl/training/rl/rollout.py)) against Stockfish
  at a deliberately weak search depth (`opponent_depth`); a share of
  rollouts (`rollout_full_game_ratio`) play to a natural conclusion, the
  rest are capped at `rollout_max_plies` for cost
- Only a decisive outcome earns reward (see `reward.compute_episode_reward`):
  win is the only positive score, loss is a heavy penalty, a draw is
  neutral unless salvaged from a clearly lost position (then rewarded).
  A capped rollout that never reached a decisive result — "just a normal
  move" — scores 0, same as mere legality
- Rewards are normalized within the group (zero mean, unit variance)
- The policy is updated with a clipped surrogate objective + KL penalty,
  applied to the summed log-probs of every move the policy made during the
  rollout (`grpo_trajectory_loss`), not just the first

### Stage 2b: LUFFY Off-Policy Guidance (Optional)

LUFFY augments GRPO by injecting expert (Stockfish) move traces as off-policy
data during policy optimisation. This helps the model learn from positions
where pure on-policy exploration stalls, while importance-sampling
corrections and confidence-based regularisation prevent collapse into a
Stockfish clone.

**Key concepts:**
- A trace generator queries Stockfish for best moves + evaluations
- Traces are stored in an experience buffer with mixture-ratio sampling
- IS weights correct the distribution mismatch between policy and expert
- Confidence weighting down-weights traces the model already plays well
- An entropy bonus encourages exploration diversity

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_luffy` | `False` | Master toggle for LUFFY augmentation |
| `trace_ratio` | `0.3` | Fraction of off-policy traces per batch |
| `is_clip` | `0.2` | IS weight clipping range |
| `luffy_alpha` | `0.85` | Ratio of off-policy (LUFFY) to on-policy (GRPO) loss: `total = (1 - luffy_alpha) * onpolicy + luffy_alpha * offpolicy` |
| `luffy_alpha_final` | `0.85` | Target value luffy_alpha anneals to over `luffy_alpha_anneal_steps` (bootstrap on traces early, hand more weight to on-policy GRPO later) |
| `luffy_alpha_anneal_steps` | `0` | Steps over which luffy_alpha linearly anneals to `luffy_alpha_final`; `0` disables annealing |
| `trace_engine_depth` | `18` | Stockfish depth for trace generation |
| `confidence_temp` | `1.0` | Temperature for anti-imitation weighting |
| `entropy_beta` | `0.01` | Entropy bonus (encourages exploration) |

**Usage:**

```bash
# Pre-generate traces
python -m qwen_chess_rl.training.rl.luffy.trace_generator \
    --input data/fens.txt \
    --output data/traces.jsonl \
    --depth 18

# Train with LUFFY (via config YAML)
cat > configs/luffy_grpo.yaml << 'EOF'
use_luffy: true
trace_ratio: 0.3
is_clip: 0.2
luffy_alpha: 0.85
trace_path: data/traces.jsonl
num_training_steps: 1000
EOF

python -m qwen_chess_rl.training.rl.grpo --config configs/luffy_grpo.yaml

# Or enable on-the-fly trace generation
cat > configs/luffy_onthefly.yaml << 'EOF'
use_luffy: true
trace_ratio: 0.3
trace_engine_depth: 18
trace_path: ""  # Empty = generate from dataset
num_training_steps: 1000
EOF
```

**Running ablations:**

```bash
python -m qwen_chess_rl.training.rl.luffy.ablation \
    --output results/luffy_ablation \
    --steps 500
```

This runs a grid of configurations (baseline GRPO, low/medium/high trace
ratios, aggressive/no IS clipping) and produces a comparison table.
