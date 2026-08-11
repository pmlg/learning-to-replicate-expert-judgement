"""Shared chess-move inference helpers."""

from __future__ import annotations

from typing import Optional

import chess
import torch
import torch.nn.functional as F

from qwen_chess_rl.data.dataset import format_fen_prompt


def choose_uci_move(
    model,
    tokenizer,
    board: chess.Board,
    *,
    temperature: float = 0.0,
) -> Optional[str]:
    """Pick a legal UCI move for ``board`` using the policy model.

    Strategy:
    1. Free-generate a short completion and accept it if legal.
    2. Otherwise score every legal UCI string by teacher-forced log-prob and
       pick the best (avoids the catastrophic "first legal move" fallback).
    """
    if model is None or tokenizer is None:
        return None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    prompt = format_fen_prompt(board.fen())
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature is not None and temperature > 0
    gen_kwargs = {
        "max_new_tokens": 8,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip common chat/thinking leftovers.
    for marker in ("</think>", "assistant", "\n"):
        if marker in decoded:
            decoded = decoded.split(marker)[-1].strip()
    parts = decoded.split()
    if parts:
        try:
            move = chess.Move.from_uci(parts[0])
            if move in board.legal_moves:
                return move.uci()
        except ValueError:
            pass

    return _best_legal_by_logprob(model, tokenizer, prompt, legal_moves, device)


def _best_legal_by_logprob(model, tokenizer, prompt: str, legal_moves, device) -> Optional[str]:
    """Rank legal UCI moves by conditional log-probability under the model."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
        device
    )
    best_uci = None
    best_score = float("-inf")

    with torch.no_grad():
        for move in legal_moves:
            uci = move.uci()
            move_ids = tokenizer(uci, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
                device
            )
            if move_ids.numel() == 0:
                continue
            full = torch.cat([prompt_ids, move_ids], dim=1)
            logits = model(input_ids=full).logits
            # Log-probs for the move tokens conditioned on the prompt prefix.
            move_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
            log_probs = F.log_softmax(move_logits, dim=-1)
            token_lp = log_probs.gather(2, move_ids.unsqueeze(-1)).squeeze(-1)
            score = token_lp.mean().item()
            if score > best_score:
                best_score = score
                best_uci = uci

    return best_uci
