# =============================================================================
# Qwen Chess RL — Training & Web App Dockerfile
# =============================================================================
# Multi-stage build: training environment + web application

# ---- Stage 1: Base environment ----
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    git \
    curl \
    wget \
    stockfish \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv venv /opt/venv --python python3.10

# ---- Stage 2: Dependencies ----
FROM base AS deps
WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --no-cache -e ".[dev,train]" && \
    uv pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ---- Stage 3: Full image ----
FROM deps AS full

COPY . .

RUN uv pip install --no-cache -e ".[dev,train]"

EXPOSE 8000

# Default: run the web app (override CMD for training)
CMD ["uv", "run", "--no-sync", "uvicorn", "qwen_chess_rl.web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Stage 4: Training ----
FROM deps AS training

COPY . .
RUN uv pip install --no-cache -e ".[train]"

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m"]
CMD ["qwen_chess_rl.training.sft.train"]
