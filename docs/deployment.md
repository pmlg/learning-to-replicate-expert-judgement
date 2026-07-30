# Deployment Guide

## Docker Deployment

### Prerequisites
- Docker with NVIDIA Container Toolkit
- Stockfish installed

### Commands

```bash
# Web application
docker compose up web

# Training
docker compose --profile train up train-sft

# Evaluation
docker compose --profile eval up evaluate
```

## Manual Deployment

### Backend
```bash
QWEN_CHESS_CHECKPOINT=/path/to/checkpoint \
    uvicorn qwen_chess_rl.web.backend.main:app \
    --host 0.0.0.0 --port 8000 \
    --workers 2
```

### Frontend
```bash
cd web/frontend
npm install
npm run build
npm run preview  # production preview on :3000
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `QWEN_CHESS_CHECKPOINT` | Path to model checkpoint |
| `STOCKFISH_PATH` | Stockfish binary path |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |
