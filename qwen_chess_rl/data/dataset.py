"""PyTorch Dataset for chess position training."""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
import datasets

logger = logging.getLogger(__name__)

# Special tokens for move format
MOVE_SEP = " -> "
EOS_TOKEN = "<|im_end|>"


def extract_best_move(example: dict) -> str:
    """Extract the best UCI move from a dataset example.

    Lichess position dumps store the Stockfish PV in ``line`` (space-separated
    UCI moves). Older schemas may use ``move`` / ``bestmove``.
    """
    for key in ("move", "bestmove", "uci"):
        value = example.get(key)
        if value:
            return str(value).strip().split()[0]
    line = example.get("line")
    if line:
        return str(line).strip().split()[0]
    return ""


def format_fen_prompt(fen: str) -> str:
    """Format a FEN position as a prompt for the model.

    The model receives the FEN and outputs the best move in UCI notation.

    Args:
        fen: FEN string of the position.

    Returns:
        Formatted prompt string.
    """
    return f"<|im_start|>system\nYou are a chess engine. Given a FEN position, output the best move in UCI format (e.g., e2e4).\n<|im_end|>\n<|im_start|>user\nPosition: {fen}\nBest move:<|im_end|>\n<|im_start|>assistant\n"


def format_move_target(move: str) -> str:
    """Format the target move as model output.

    Args:
        move: UCI move string (e.g., 'e2e4').

    Returns:
        Target string with EOS token.
    """
    return f"{move}{EOS_TOKEN}"


def tokenize_example(
    example: dict,
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokenize a single chess position example for training.

    Args:
        example: Dataset example with 'fen' and a move in move/bestmove/line.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Tokenized example with input_ids, attention_mask, and labels.
    """
    fen = str(example.get("fen", ""))
    move = extract_best_move(example)
    if not move:
        # Avoid training on empty assistant targets (teaches immediate EOS).
        move = "0000"

    prompt = format_fen_prompt(fen)
    target = format_move_target(move)
    full_text = prompt + target

    tokens = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length" if max_length else False,
        return_tensors=None,
    )

    # Compute labels: mask out the prompt tokens and padding (set to -100)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels = tokens["input_ids"].copy()
    labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
    labels = [
        -100 if mask == 0 else label
        for label, mask in zip(labels, tokens["attention_mask"])
    ]

    tokens["labels"] = labels
    return tokens


class ChessPositionDataset(Dataset):
    """PyTorch dataset for chess position training.

    Loads preprocessed positions from HuggingFace Datasets arrow format
    and provides tokenized (input_ids, attention_mask, labels) tensors.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        stream: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to the processed dataset directory.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum sequence length.
            split: Dataset split name (train/val/test).
            stream: Whether to stream data (lower memory usage).
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        if stream:
            self.dataset = datasets.load_dataset(
                str(self.data_path),
                split=split,
                streaming=True,
            )
        else:
            self.dataset = datasets.load_from_disk(str(self.data_path / split))

    def __len__(self) -> int:
        if hasattr(self.dataset, "nrows"):
            return self.dataset.nrows
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        # Fallback for streaming datasets
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.dataset[idx]

        tokens = tokenize_example(
            example,
            self.tokenizer,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(tokens["labels"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Stacks individual examples into batched tensors.

    Args:
        batch: List of tokenized examples.

    Returns:
        Batched tensors.
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
