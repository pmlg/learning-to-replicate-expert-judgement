# Notebooks

| Notebook | What it shows |
|----------|---------------|
| [`grpo_and_onpolicy_distillation.ipynb`](grpo_and_onpolicy_distillation.ipynb) | Teaching walkthrough of the GRPO loss (ratio, clip, KL, aggregation) and on-policy distillation (per-token reverse KL) on one chess position, using Qwen3-0.6B and Stockfish. No training. Figures land in [`figures/`](figures/). |

## Running (CPU only)

The notebook uses its own lightweight CPU environment, separate from the training stack
(no vLLM / flash-attn / CUDA):

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers accelerate matplotlib pandas python-chess nbformat nbclient nbconvert jupyter-core ipykernel
uv run --no-project python -m ipykernel install --user --name dat341

uv run --no-project jupyter nbconvert --to notebook --execute --inplace \
    notebooks/grpo_and_onpolicy_distillation.ipynb --ExecutePreprocessor.timeout=900
```

`--no-project` stops `uv run` from syncing the repo's full (GPU) dependency set into the venv.

Requirements:

- Stockfish binary at `~/.local/bin/stockfish` (Stockfish 17+).
- First run downloads `Qwen/Qwen3-0.6B` (~1.2 GB) into the Hugging Face cache.
- Runtime: about 2–4 minutes on a laptop CPU once the model is cached.
