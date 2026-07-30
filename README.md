# ♟ Qwen Chess RL

**Train Qwen-8B to play chess via Supervised Fine-Tuning (SFT) and Reinforcement Learning (GRPO)**

This project implements an end-to-end training pipeline for fine-tuning **Qwen-8B** to play chess, using ~2 million labelled chess positions from Lichess. It includes:

- 📥 **Data Pipeline** — Download, filter, and sample 2M chess positions from the [Lichess chess position evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) dataset
- 🎯 **SFT Training** — Supervised fine-tuning with QLoRA to predict best moves from FEN positions
- 🧠 **GRPO RL** — Group Relative Policy Optimization with Stockfish as the reward function
- 📊 **Evaluation Suite** — Benchmark against Stockfish, random players, and puzzle datasets
- 🌐 **Web Application** — Interactive chess UI similar to the [Qwen Chess demo](https://qwen-chess.replit.app/)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)

---

## 🏗 Architecture

```mermaid
flowchart TD
    A[Lichess Dataset] --> B[Data Pipeline]
    B --> C[2M Filtered Positions]
    C --> D[SFT Training<br/>QLoRA Qwen-8B]
    D --> E[SFT Checkpoint]
    E --> F[GRPO RL Training<br/>Stockfish Reward]
    F --> G[RL Checkpoint]
    G --> H[Web App<br/>FastAPI + React]
    E --> I[Evaluation Suite]
    G --> I
    I --> J[Benchmark Report]
```

### Training Pipeline

```
┌─────────────────────────────────────────────────────┐
│                   Data Pipeline                      │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌────────┐ │
│  │ Download │→ │ Filter   │→ │ Sample │→ │ Split  │ │
│  └──────────┘  └──────────┘  └────────┘  └────────┘ │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              SFT (Supervised Fine-Tuning)            │
│  ┌──────────────────────────────────────────────┐   │
│  │  Qwen-8B + QLoRA (lora_r=16, 4-bit)         │   │
│  │  Input: FEN → Output: Best Move (UCI)        │   │
│  │  Loss: CrossEntropy(next_token)              │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│           GRPO (Group Relative Policy Optimization)  │
│  ┌──────────────────────────────────────────────┐   │
│  │  For each position: generate 8 moves         │   │
│  │  Evaluate via Stockfish reward function      │   │
│  │  Normalize advantages within group           │   │
│  │  Update policy (clipped surrogate + KL)      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Reward Function

The GRPO reward function considers multiple chess-quality factors:

| Factor | Reward | Description |
|--------|--------|-------------|
| Legal move | +5 | The proposed move is legal |
| Centipawn loss | 0 to -10 | Scaled penalty proportional to move quality loss |
| Checkmate | +10 | The move delivers checkmate |
| Blunder | -5 | Additional penalty if centipawn loss > 100 |
| Draw avoidance | -3 | Moving into drawing position (stalemate/insufficient material) |
| Repetition | -2 | Moving into threefold repetition |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: RTX 5090 for 8B model)
- Stockfish chess engine (`sudo apt-get install stockfish`)

### Installation

```bash
# Clone the repository
git clone https://github.com/deep-science-ai/qwen-chess-rl.git
cd qwen-chess-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev,train]"

# Optional: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verify Setup

```bash
# Check Stockfish is available
stockfish --version

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📥 Dataset Preparation

### Download and Process

```bash
# Run the full pipeline: download → filter → sample 2M → split
python -m qwen_chess_rl.data.download
```

This will:
1. Download the Lichess chess position evaluations dataset (~100GB+ on disk)
2. Filter positions by quality (max centipawn loss < 200, valid FEN)
3. Sample ~2 million positions
4. Split into train (98%), validation (1%), test (1%)
5. Save to `data/processed/{train,val,test}` in Arrow format

### Custom Options

```python
from qwen_chess_rl.data.download import run_pipeline

paths = run_pipeline(
    cache_dir="data/raw",
    output_dir="data/processed",
    target_size=2_000_000,
    num_proc=8,
    seed=42,
)
```

---

## 🎯 Training

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune Qwen-8B using QLoRA to predict the best move from FEN positions.

```bash
# Default training
bash scripts/train_sft.sh

# With custom options
bash scripts/train_sft.sh \
    --model Qwen/Qwen-8B \
    --data data/processed \
    --output checkpoints/sft \
    --batch-size 4 \
    --lr 2e-4 \
    --max-steps 10000
```

**Configuration** (see `qwen_chess_rl/training/sft/config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora_r` | 16 | LoRA rank dimension |
| `lora_alpha` | 32 | LoRA scaling parameter |
| `use_4bit` | True | 4-bit quantization (QLoRA) |
| `batch_size` | 4 | Per-device batch size |
| `gradient_accumulation_steps` | 8 | Effective batch size: 32 |
| `learning_rate` | 2e-4 | Peak learning rate |
| `optim` | `paged_adamw_8bit` | Memory-efficient optimizer |
| `max_length` | 512 | Max sequence length |

**Memory requirements:** ~24GB GPU RAM (QLoRA 4-bit, batch_size=4 on RTX 5090)

### Stage 2: GRPO Reinforcement Learning

Improve the SFT model using GRPO with Stockfish as the reward function.

```bash
# Default training
bash scripts/train_grpo.sh

# With custom options
bash scripts/train_grpo.sh \
    --sft-checkpoint checkpoints/sft/final \
    --output checkpoints/rl \
    --steps 1000 \
    --lr 5e-6
```

**Configuration** (see `qwen_chess_rl/training/rl/config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_generations` | 8 | Moves sampled per position |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `beta` | 0.01 | KL penalty coefficient |
| `batch_size` | 8 | Prompts per training batch |
| `ppo_epochs` | 4 | Inner PPO update iterations |
| `temperature` | 1.0 | Move sampling temperature |

---

## 📊 Evaluation

### Run Full Benchmark

```bash
# Evaluate a checkpoint
bash scripts/evaluate.sh checkpoints/sft/final

# Compare multiple checkpoints
bash scripts/evaluate.sh checkpoints/sft/final \
    --compare checkpoints/rl/checkpoint-500 checkpoints/rl/final
```

### Metrics Reported

| Metric | Description |
|--------|-------------|
| **Avg Centipawn Loss** | Average score difference vs Stockfish's best move |
| **Legal Move %** | Fraction of generated moves that are legal |
| **Win Rate vs Stockfish** | Win/loss/draw rate at depth 8 |
| **Win Rate vs Random** | Baseline performance |
| **Estimated Elo** | Rating estimated from Stockfish match results |
| **Puzzle Accuracy** | % correct on chess puzzle datasets |

### Programmatic Evaluation

```python
from qwen_chess_rl.evaluation.benchmark import run_full_benchmark

results = run_full_benchmark(
    checkpoint_path="checkpoints/sft/final",
    stockfish_games=20,
    stockfish_depth=8,
)
print(results["vs_stockfish"]["estimated_elo"])
```

---

## 🌐 Web Application

### Start Backend

```bash
# Start the FastAPI server
QWEN_CHESS_CHECKPOINT=checkpoints/sft/final \
    uvicorn qwen_chess_rl.web.backend.main:app --host 0.0.0.0 --port 8000
```

### Start Frontend (Development)

```bash
cd web/frontend
npm install
npm run dev
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/new_game` | POST | Start new game session |
| `/move` | POST | Make a move + get AI response |
| `/game/{session_id}` | GET | Get current game state |
| `/undo/{session_id}` | POST | Undo last move pair |
| `/import_fen` | POST | Import FEN position |
| `/fen_export/{session_id}` | GET | Export FEN + PGN |

### Features

- Interactive chessboard (drag-and-drop)
- Play as white or black
- FEN import/export for analysis
- PGN move history display
- Adjustable model temperature
- Responsive UI (desktop + mobile)
- Move explanations (AI thinking indicator)

---

## 🐳 Docker Deployment

### Full Stack

```bash
# Start the web application
docker compose up web

# Start training
docker compose --profile train up train-sft
docker compose --profile train up train-grpo

# Run evaluation
docker compose --profile eval up evaluate
```

### Build Custom Images

```bash
# Web application
docker build --target full -t qwen-chess-web .

# Training
docker build --target training -t qwen-chess-train .
```

---

## 📁 Project Structure

```
qwen-chess-rl/
├── qwen_chess_rl/              # Main package
│   ├── data/                   # Data pipeline
│   │   ├── download.py         # Download + filter + sample
│   │   ├── preprocess.py       # FEN tokenization
│   │   └── dataset.py          # PyTorch Dataset
│   ├── training/
│   │   ├── sft/                # Supervised Fine-Tuning
│   │   │   ├── config.py       # QLoRA hyperparameters
│   │   │   └── train.py        # SFT training loop
│   │   └── rl/                 # Reinforcement Learning
│   │       ├── config.py       # GRPO hyperparameters
│   │       ├── reward.py       # Stockfish reward function
│   │       └── grpo.py         # GRPO training loop
│   ├── evaluation/
│   │   ├── metrics.py          # Centipawn loss, legal %, Elo
│   │   ├── stockfish_eval.py   # Play games vs Stockfish
│   │   └── benchmark.py        # Full benchmark orchestrator
│   └── web/
│       ├── backend/
│       │   └── main.py         # FastAPI server
│       └── frontend/
│           ├── src/            # React components
│           └── package.json
├── scripts/                    # Shell scripts
│   ├── train_sft.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
├── docs/                       # Documentation
├── tests/                      # Unit tests
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device IDs |
| `QWEN_CHESS_CHECKPOINT` | — | Model checkpoint path for web app |
| `STOCKFISH_PATH` | `stockfish` | Stockfish binary location |
| `WANDB_API_KEY` | — | Weights & Biases logging key |
| `OMP_NUM_THREADS` | `8` | OpenMP threads for data loading |

### YAML Config (Advanced)

Create a `config.yaml`:
```yaml
model_name: Qwen/Qwen-8B
lora_r: 32
lora_alpha: 64
batch_size: 8
learning_rate: 3e-4
```

Then:
```bash
bash scripts/train_sft.sh --config config.yaml
```

---

## 📈 Results

> *Results will be populated after training runs complete.*

### SFT Results (Baseline)

| Metric | Value |
|--------|-------|
| Training Loss | — |
| Eval Loss | — |
| Legal Move % | — |
| Avg Centipawn Loss | — |

### GRPO Improvement Over SFT

| Metric | SFT | GRPO | Improvement |
|--------|-----|------|-------------|
| Win Rate vs Stockfish (d8) | — | — | — |
| Estimated Elo | — | — | — |
| Puzzle Accuracy | — | — | — |

---

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Dataset Preparation](docs/dataset.md)
- [Training Guide](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Deployment Guide](docs/deployment.md)
- [Architecture Overview](docs/architecture.md)

---

## 📖 Citation

```bibtex
@software{qwen_chess_rl,
  title = {Qwen Chess RL: Fine-tuning Qwen-8B for Chess via SFT and GRPO},
  author = {El Fayoumi, Ibrahim},
  year = {2026},
  url = {https://github.com/deep-science-ai/qwen-chess-rl}
}
```

---

## 📝 License

MIT

---

*Built with [Qwen-8B](https://huggingface.co/Qwen/Qwen-8B), [HuggingFace Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), and [python-chess](https://python-chess.readthedocs.io/).*
