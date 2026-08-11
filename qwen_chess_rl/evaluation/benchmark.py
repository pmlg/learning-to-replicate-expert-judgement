"""Full benchmark suite orchestrator for chess model evaluation."""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_chess_rl.evaluation.metrics import compute_all_metrics, puzzle_accuracy
from qwen_chess_rl.evaluation.stockfish_eval import (
    evaluate_vs_stockfish,
    evaluate_vs_random,
)

logger = logging.getLogger(__name__)


def load_model_for_eval(checkpoint_path: str):
    """Load a model for evaluation (inference mode).

    Supports full checkpoints and PEFT/LoRA adapter directories.

    Args:
        checkpoint_path: Path to the model checkpoint.

    Returns:
        (model, tokenizer) tuple in eval mode.
    """
    import json

    ckpt = Path(checkpoint_path)
    adapter_cfg_path = ckpt / "adapter_config.json"
    logger.info("Loading model from %s for evaluation ...", checkpoint_path)

    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            base_name = json.load(f).get("base_model_name_or_path") or "Qwen/Qwen3-1.7B"
        logger.info("PEFT adapter detected; base model: %s", base_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(ckpt))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_full_benchmark(
    checkpoint_path: str,
    output_dir: str = "evaluation_results",
    stockfish_games: int = 20,
    random_games: int = 10,
    stockfish_depth: int = 8,
    temperature: float = 0.1,
) -> dict:
    """Run the complete evaluation benchmark suite.

    Evaluates the model against:
    - Stockfish at fixed depth
    - Random player
    - Previous checkpoints (if available)

    Reports:
    - Average centipawn loss
    - Legal move percentage
    - Win rate
    - Estimated Elo
    - Puzzle accuracy (if puzzle dataset available)

    Args:
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory for evaluation results.
        stockfish_games: Number of games vs Stockfish.
        random_games: Number of games vs random player.
        stockfish_depth: Stockfish analysis depth.
        temperature: Model sampling temperature.

    Returns:
        Dictionary with all evaluation results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_for_eval(checkpoint_path)

    results = {
        "checkpoint": checkpoint_path,
        "temperature": temperature,
    }

    # 1. Evaluate vs Stockfish
    logger.info("=== Evaluating vs Stockfish (depth=%s, %s games) ===", stockfish_depth, stockfish_games)
    sf_results = evaluate_vs_stockfish(
        model, tokenizer,
        num_games=stockfish_games,
        stockfish_depth=stockfish_depth,
        temperature=temperature,
    )
    results["vs_stockfish"] = sf_results
    logger.info("Stockfish results: %s", sf_results)

    # 2. Evaluate vs Random
    logger.info("=== Evaluating vs Random (%s games) ===", random_games)
    random_results = evaluate_vs_random(
        model, tokenizer,
        num_games=random_games,
        temperature=temperature,
    )
    results["vs_random"] = random_results
    logger.info("Random results: %s", random_results)

    # 3. Centipawn loss on test set (if available)
    test_path = Path("data/processed/test")
    if test_path.exists():
        logger.info("=== Computing centipawn loss on test set ===")
        from datasets import load_from_disk

        test_ds = load_from_disk(str(test_path))
        positions = []
        from qwen_chess_rl.data.dataset import extract_best_move

        for example in test_ds.select(range(min(1000, len(test_ds)))):
            fen = example.get("fen", "")
            move = extract_best_move(example)
            if fen and move:
                positions.append((fen, move))

        if positions:
            metrics = compute_all_metrics(positions)
            results["test_set_metrics"] = metrics
            logger.info("Test set metrics: %s", metrics)

    # Save results
    results_path = output_path / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Benchmark results saved to %s", results_path)

    return results


def compare_checkpoints(
    checkpoint_paths: list[str],
    output_dir: str = "evaluation_results",
    **kwargs,
) -> dict:
    """Compare multiple model checkpoints.

    Args:
        checkpoint_paths: List of checkpoint paths to evaluate.
        output_dir: Directory for results.
        **kwargs: Additional args passed to run_full_benchmark.

    Returns:
        Dictionary mapping checkpoint names to their results.
    """
    comparisons = {}
    for ckpt in checkpoint_paths:
        # Use parent/name so sft/final vs rl/final don't collide as "final"
        path = Path(ckpt)
        name = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
        logger.info("Benchmarking checkpoint: %s", name)
        comparisons[name] = run_full_benchmark(ckpt, output_dir=output_dir, **kwargs)

    # Save combined comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    # Print summary table
    logger.info("\n=== CHECKPOINT COMPARISON ===")
    for name, result in comparisons.items():
        sf = result.get("vs_stockfish", {})
        logger.info(
            "%s: vs Stockfish — W:%s D:%s L:%s (Elo: %s)",
            name,
            sf.get("wins", 0), sf.get("draws", 0), sf.get("losses", 0),
            sf.get("estimated_elo", "N/A"),
        )

    return comparisons


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run chess model evaluation benchmark")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--stockfish-games", type=int, default=20)
    parser.add_argument("--random-games", type=int, default=10)
    parser.add_argument("--stockfish-depth", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--compare", type=str, nargs="+", help="Additional checkpoints to compare")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    bench_kwargs = {
        "output_dir": args.output_dir,
        "stockfish_games": args.stockfish_games,
        "random_games": args.random_games,
        "stockfish_depth": args.stockfish_depth,
        "temperature": args.temperature,
    }
    if args.compare:
        all_checkpoints = [args.checkpoint] + list(args.compare)
        compare_checkpoints(all_checkpoints, **bench_kwargs)
    else:
        run_full_benchmark(args.checkpoint, **bench_kwargs)


if __name__ == "__main__":
    main()
