"""LUFFY-augmented GRPO training loop.

Extends the standard GRPO training loop with off-policy Stockfish traces:
1. Pre-generates (or loads) a pool of expert traces.
2. Maintains an :class:`ExperienceBuffer` with mixed on-/off-policy data.
3. During each GRPO step, samples off-policy traces and computes an
   IS-corrected, confidence-weighted loss term alongside the on-policy
   GRPO loss.
4. Applies an entropy bonus to prevent collapse to Stockfish behaviour.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen_chess_rl.data.dataset import format_fen_prompt
from qwen_chess_rl.training.rl.config import GRPOConfig
from qwen_chess_rl.training.rl.luffy.buffer import ExperienceBuffer
from qwen_chess_rl.training.rl.luffy.is_weights import compute_luffy_loss
from qwen_chess_rl.training.rl.luffy.schema import OffPolicyTrace
from qwen_chess_rl.training.rl.luffy.trace_generator import TraceGenerator

logger = logging.getLogger(__name__)


def load_traces_from_jsonl(path: str | Path) -> list[OffPolicyTrace]:
    """Load pre-generated traces from a JSONL file.

    Args:
        path: Path to JSONL file with one trace dict per line.

    Returns:
        List of :class:`OffPolicyTrace` objects.
    """
    traces: list[OffPolicyTrace] = []
    path = Path(path)
    if not path.exists():
        logger.warning("Trace file %s does not exist.", path)
        return traces

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(OffPolicyTrace.from_dict(json.loads(line)))

    logger.info("Loaded %d traces from %s", len(traces), path)
    return traces


def compute_trace_rewards(
    traces: list[OffPolicyTrace],
    stockfish_path: str | None = None,
    eval_depth: int = 15,
) -> None:
    """Compute and attach rewards to a list of traces (in-place).

    Uses the same reward function as the on-policy GRPO loop for
    consistency.

    Args:
        traces: List of traces to annotate.
        stockfish_path: Path to Stockfish binary.
        eval_depth: Evaluation depth for reward computation.
    """
    from qwen_chess_rl.training.rl.reward import compute_reward

    for trace in traces:
        reward = compute_reward(
            trace.fen,
            trace.action,
            stockfish_path=stockfish_path,
            eval_depth=eval_depth,
        )
        trace.reward = reward.get("total_reward", 0.0)


def prepare_luffy_buffer(cfg: GRPOConfig) -> ExperienceBuffer:
    """Build and populate the experience buffer for LUFFY training.

    If ``cfg.trace_path`` is set, loads pre-generated traces.
    Otherwise, generates traces on-the-fly from the dataset FENs.

    Args:
        cfg: GRPO configuration with LUFFY hyperparameters.

    Returns:
        Populated :class:`ExperienceBuffer`.
    """
    buffer = ExperienceBuffer(
        capacity=cfg.buffer_capacity,
        trace_ratio=cfg.trace_ratio,
        seed=cfg.seed,
    )

    if cfg.trace_path:
        traces = load_traces_from_jsonl(cfg.trace_path)
        logger.info("Loaded %d off-policy traces from %s", len(traces), cfg.trace_path)
    else:
        # Generate traces on-the-fly from the training data FENs
        logger.info("Generating traces on-the-fly (depth=%d) ...", cfg.trace_engine_depth)
        gen = TraceGenerator(
            engine_depth=cfg.trace_engine_depth,
            min_eval_cp=cfg.min_eval_cp,
        )

        # Try to load FENs from the processed dataset
        fens = _load_dataset_fens(cfg.data_path)
        if fens:
            # Subsample to buffer capacity for efficiency
            max_traces = min(cfg.buffer_capacity, len(fens))
            rng = np.random.default_rng(cfg.seed)
            sampled_fens = rng.choice(fens, size=max_traces, replace=False).tolist()
            traces = gen.generate_traces(sampled_fens)
        else:
            logger.warning("No FENs found for trace generation — LUFFY buffer will be empty.")
            traces = []

    # Compute rewards for all traces
    if traces:
        logger.info("Computing rewards for %d traces ...", len(traces))
        compute_trace_rewards(traces, eval_depth=cfg.eval_depth)

    buffer.add_off_policy_batch(traces)
    logger.info("LUFFY buffer ready: %d off-policy traces", buffer.num_off_policy)
    return buffer


def _load_dataset_fens(data_path: str = "data/processed") -> list[str]:
    """Load FEN strings from the processed dataset.

    Args:
        data_path: Path to the HuggingFace datasets directory.

    Returns:
        List of FEN strings.
    """
    try:
        import datasets as hf_datasets

        ds = hf_datasets.load_from_disk(Path(data_path) / "train")
        fens = list(ds["fen"])
        logger.info("Loaded %d FENs from %s", len(fens), data_path)
        return fens
    except Exception as exc:
        logger.debug("Could not load dataset FENs: %s", exc)
        return []


def luffy_offpolicy_loss(
    model,
    tokenizer,
    buffer: ExperienceBuffer,
    cfg: GRPOConfig,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int | None]:
    """Compute the LUFFY off-policy loss for a single training step.

    Samples a batch of off-policy traces, computes the current policy's
    log-prob for each trace action, and applies IS-corrected,
    confidence-weighted loss.

    Args:
        model: Current policy model.
        tokenizer: HuggingFace tokenizer.
        buffer: Experience buffer with off-policy traces.
        cfg: GRPO configuration with LUFFY hyperparameters.
        device: Torch device.

    Returns:
        Dictionary with 'loss' (tensor or float), 'mean_is_ratio',
        'mean_confidence_weight', 'num_traces'.
    """
    if buffer.num_off_policy == 0:
        return {"loss": 0.0, "mean_is_ratio": 0.0, "mean_confidence_weight": 0.0, "num_traces": 0}

    # Sample off-policy traces
    sample_size = max(int(cfg.batch_size * cfg.trace_ratio), 1)
    sample = buffer.sample(sample_size)
    traces = sample["off_policy"]

    if not traces:
        return {"loss": 0.0, "mean_is_ratio": 0.0, "mean_confidence_weight": 0.0, "num_traces": 0}

    policy_log_probs_list = []
    behavior_log_probs_list = []
    advantages_list = []

    for trace in traces:
        # Compute current policy log-prob for the expert action
        prompt = format_fen_prompt(trace.fen)
        action_ids = tokenizer.encode(trace.action, return_tensors="pt").to(device)

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length,
        ).to(device)

        # Forward pass WITH gradients (we need gradients for policy update)
        full_input = torch.cat([inputs["input_ids"], action_ids], dim=-1)
        full_mask = torch.cat([
            inputs["attention_mask"],
            torch.ones_like(action_ids),
        ], dim=-1)

        outputs = model(full_input, attention_mask=full_mask)
        logits = outputs.logits[:, :-1, :]  # Shift for next-token prediction
        log_probs = F.log_softmax(logits, dim=-1)

        # Extract log-prob of the action tokens
        action_len = action_ids.shape[-1]
        gen_log_probs = log_probs[:, -action_len:, :].gather(
            2, action_ids.unsqueeze(-1),
        ).squeeze(-1).sum()

        policy_log_probs_list.append(gen_log_probs)
        behavior_log_probs_list.append(trace.behavior_log_prob)
        advantages_list.append(trace.reward)  # Use reward as advantage for off-policy

    # Stack into tensors
    policy_log_probs = torch.stack(policy_log_probs_list)
    behavior_log_probs = torch.tensor(
        behavior_log_probs_list, dtype=torch.float32, device=device,
    )
    advantages = torch.tensor(
        advantages_list, dtype=torch.float32, device=device,
    )
    # Normalise advantages
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute LUFFY loss
    result = compute_luffy_loss(
        policy_log_probs,
        behavior_log_probs,
        advantages,
        is_clip=cfg.is_clip,
        confidence_temp=cfg.confidence_temp,
    )

    return {
        "loss": result["loss"].item() if isinstance(result["loss"], torch.Tensor) else float(result["loss"]),
        "loss_tensor": result["loss"] if isinstance(result["loss"], torch.Tensor) else None,
        "mean_is_ratio": float(result["is_ratios"].mean().item()),
        "mean_confidence_weight": float(result["confidence_weights"].mean().item()),
        "raw_is_loss": float(result["raw_is_loss"].item()),
        "num_traces": len(traces),
    }


# ---------------------------------------------------------------------- #
# Convenience: trace generation script
# ---------------------------------------------------------------------- #

def generate_traces_cli():
    """CLI entry point for generating traces from a FEN file.

    Usage::

        python -m qwen_chess_rl.training.rl.luffy.training \
            generate-traces --input data/fens.txt --output data/traces.jsonl
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate LUFFY traces")
    parser.add_argument("--input", type=str, required=True, help="FEN text file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL")
    parser.add_argument("--depth", type=int, default=18, help="Stockfish depth")
    parser.add_argument("--min-eval", type=int, default=0, help="Min eval threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    gen = TraceGenerator(engine_depth=args.depth, min_eval_cp=args.min_eval)

    with open(args.input) as f:
        fens = [line.strip() for line in f if line.strip()]

    traces = gen.generate_traces(fens)

    # Compute rewards
    compute_trace_rewards(traces, eval_depth=15)

    with open(args.output, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")

    logger.info("Wrote %d traces to %s", len(traces), args.output)


if __name__ == "__main__":
    generate_traces_cli()
