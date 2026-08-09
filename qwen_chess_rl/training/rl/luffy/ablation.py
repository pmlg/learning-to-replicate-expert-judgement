"""Evaluation and ablation runner for LUFFY experiments.

Runs multiple GRPO configurations with different LUFFY settings
(trace_ratio, IS clipping, offpolicy_weight) and collects comparison
metrics across all configurations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from qwen_chess_rl.training.rl.config import GRPOConfig

logger = logging.getLogger(__name__)


# Default ablation grid — covers the key LUFFY hyperparameters
DEFAULT_ABLATION_GRID = {
    "baseline_grpo": {
        "use_luffy": False,
        "trace_ratio": 0.0,
        "offpolicy_weight": 0.0,
    },
    "luffy_low_trace": {
        "use_luffy": True,
        "trace_ratio": 0.1,
        "offpolicy_weight": 0.3,
        "is_clip": 0.2,
    },
    "luffy_medium_trace": {
        "use_luffy": True,
        "trace_ratio": 0.3,
        "offpolicy_weight": 0.5,
        "is_clip": 0.2,
    },
    "luffy_high_trace": {
        "use_luffy": True,
        "trace_ratio": 0.5,
        "offpolicy_weight": 0.7,
        "is_clip": 0.2,
    },
    "luffy_aggressive_clipping": {
        "use_luffy": True,
        "trace_ratio": 0.3,
        "offpolicy_weight": 0.5,
        "is_clip": 0.1,
    },
    "luffy_no_clipping": {
        "use_luffy": True,
        "trace_ratio": 0.3,
        "offpolicy_weight": 0.5,
        "is_clip": 0.9,
    },
}


def build_config(
    name: str,
    overrides: dict,
    base_config: Optional[GRPOConfig] = None,
) -> GRPOConfig:
    """Build a GRPOConfig from a named ablation configuration.

    Args:
        name: Name of the ablation configuration.
        overrides: Dictionary of config overrides.
        base_config: Base configuration to override (defaults to GRPOConfig()).

    Returns:
        Configured GRPOConfig instance.
    """
    cfg = base_config or GRPOConfig()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logger.warning("Unknown config key '%s' in ablation '%s'", key, name)

    # Set output directory per experiment
    cfg.output_dir = f"checkpoints/luffy_ablation/{name}"
    return cfg


def run_ablation(
    output_dir: str = "results/luffy_ablation",
    config_grid: Optional[dict] = None,
    base_config: Optional[GRPOConfig] = None,
    skip_training: bool = False,
) -> dict:
    """Run the full LUFFY ablation experiment.

    For each configuration in the grid:
    1. Build the config.
    2. (Optionally) run training.
    3. Evaluate the resulting checkpoint.
    4. Collect metrics.

    Results are saved to ``output_dir/results.json``.

    Args:
        output_dir: Directory for saving results.
        config_grid: Custom ablation grid (defaults to DEFAULT_ABLATION_GRID).
        base_config: Base GRPO configuration.
        skip_training: If True, only evaluate existing checkpoints.

    Returns:
        Dictionary with all experiment results.
    """
    config_grid = config_grid or DEFAULT_ABLATION_GRID
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for name, overrides in config_grid.items():
        logger.info("=== Running ablation: %s ===", name)
        cfg = build_config(name, overrides, base_config)

        experiment_result: dict = {
            "config_name": name,
            "config": asdict(cfg),
            "training": {},
            "evaluation": {},
        }

        if not skip_training:
            try:
                start_time = time.time()
                from qwen_chess_rl.training.rl.grpo import train as grpo_train

                grpo_train(cfg)
                training_time = time.time() - start_time
                experiment_result["training"] = {
                    "elapsed_seconds": training_time,
                    "output_dir": cfg.output_dir,
                    "status": "completed",
                }
            except Exception as exc:
                logger.error("Training failed for %s: %s", name, exc)
                experiment_result["training"] = {
                    "status": "failed",
                    "error": str(exc),
                }

        # Evaluate checkpoint
        checkpoint_path = Path(cfg.output_dir) / "final"
        if checkpoint_path.exists():
            try:
                from qwen_chess_rl.evaluation.benchmark import run_full_benchmark

                eval_results = run_full_benchmark(
                    checkpoint_path=str(checkpoint_path),
                    stockfish_games=20,
                    stockfish_depth=8,
                )
                experiment_result["evaluation"] = eval_results
            except Exception as exc:
                logger.error("Evaluation failed for %s: %s", name, exc)
                experiment_result["evaluation"] = {"error": str(exc)}
        else:
            experiment_result["evaluation"] = {"status": "no_checkpoint"}

        all_results[name] = experiment_result

        # Save incremental results
        with open(out_path / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate comparison table
    comparison = generate_comparison_table(all_results)
    with open(out_path / "comparison.md", "w") as f:
        f.write(comparison)

    logger.info("Ablation study complete. Results saved to %s", out_path)
    return all_results


def generate_comparison_table(results: dict[str, dict]) -> str:
    """Generate a Markdown comparison table from ablation results.

    Args:
        results: Results dictionary from :func:`run_ablation`.

    Returns:
        Markdown-formatted comparison table.
    """
    rows: list[list[str]] = []
    headers = [
        "Config",
        "trace_ratio",
        "is_clip",
        "offpolicy_weight",
        "Legal Move %",
        "Avg CP Loss",
        "Win Rate (SF)",
        "Puzzle Acc.",
        "Est. Elo",
    ]

    for name, data in results.items():
        cfg = data.get("config", {})
        eval_data = data.get("evaluation", {})

        # Extract metrics from nested evaluation results
        metrics = eval_data.get("metrics", {})
        vs_sf = eval_data.get("vs_stockfish", {})

        rows.append([
            name,
            str(cfg.get("trace_ratio", 0)),
            str(cfg.get("is_clip", 0)),
            str(cfg.get("offpolicy_weight", 0)),
            f"{metrics.get('legal_move_pct', 0):.1%}",
            f"{metrics.get('avg_centipawn_loss', 0):.1f}",
            f"{vs_sf.get('win_rate', 0):.1%}",
            f"{metrics.get('puzzle_accuracy', 0):.1%}",
            f"{vs_sf.get('estimated_elo', 0):.0f}",
        ])

    # Build Markdown table
    md = "# LUFFY Ablation Comparison\n\n"
    md += "| " + " | ".join(headers) + " |\n"
    md += "|" + "---|" * len(headers) + "\n"
    for row in rows:
        md += "| " + " | ".join(row) + " |\n"

    return md


def main():
    """CLI entry point for running LUFFY ablations."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LUFFY ablation experiments")
    parser.add_argument("--output", type=str, default="results/luffy_ablation")
    parser.add_argument("--skip-training", action="store_true", help="Only evaluate existing checkpoints")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base = GRPOConfig(num_training_steps=args.steps)
    run_ablation(
        output_dir=args.output,
        base_config=base,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()
