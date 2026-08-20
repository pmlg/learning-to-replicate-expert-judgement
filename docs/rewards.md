# Reward Design: Why Q(s,a) Win-Probability

Status: **active** (implemented in `qwen_chess_rl/training/rl/reward.py`, PR #13)
Last reviewed: 2026-08-20

## TL;DR

The reward is

```
r = Q(s, a) = P(mover wins | position after the proposed move)   in [0, 1]
```

- Mate ⇒ 1.0, blunder ⇒ ~0, solid move ⇒ 0.5–0.8, illegal ⇒ −1.0 floor.
- Win probability comes from Stockfish centipawn eval via the Lichess-scale
  sigmoid `p = 1 / (1 + 10^(−cp/400))`.
- The perspective is **always the mover's**: post-move Stockfish scores are
  relative to the *opponent* (side to move), so the reward uses
  `1 − winprob_after`.

## Why the old reward didn't work

Four compounding defects were found in the pre-PR-13 implementation:

1. **It didn't reward winning.** It summed arbitrary constants
   (`+5 legal, +10 checkmate, −5 blunder, +3 draw-avoid, +2 repetition-avoid,
   −10 max cp penalty`). A winning tactic and a passive move could score
   nearly the same. Nothing measured "does this move win the game."
2. **A sign bug corrupted the signal.** Stockfish scores are relative to the
   side to move. The old code compared pre-move score (mover's perspective)
   against post-move score (opponent's perspective) **without negation**, so
   a *good* move looked like a ~100cp loss and was penalized.
3. **It compared the wrong quantity.** `abs(score_before − score_after)`
   conflates "position already won" with "move quality". It also spawned a
   fresh Stockfish process per evaluation — dominant cost in RL rollouts.
4. **No crash safety.** A malformed FEN segfaults the python-chess UCI
   subprocess (verified: exit −11). The old code would kill the whole
   training run.

## The strategic-reward question (dense vs sparse)

### Claim under discussion

> "Tactical reward is easy, strategic reward in sequential decision-making is
> tough. The real obstacle to converting expert knowledge to RL is data and
> rewards — strategic, not tactical rewards."

### Position taken (evidence-backed)

The problem is real, but the fix is **not** to reward "strategy" explicitly.

- **Dense Q-value reward IS the strategic reward.** Chess-R1
  (Krafton, COLM SCALR 2025, arXiv:2507.00726) showed dense Q-value rewards
  (win-probability from a chess-pretrained critic) **beat sparse outcome
  rewards**, and sparse binary rewards **failed completely** on small models
  (Qwen2.5-3B/8B). A Q-function has already propagated value through the game
  tree — the long-horizon "strategy" is baked into the value, so the model
  learns it by regression, not by solving credit assignment over 40+ plies.
- **Hand-crafted strategic rewards (piece activity, king safety, pawn
  structure bonuses) are the classic reward-shaping trap**: fragile,
  hackable, and empirically outperformed by dense Q. Don't shape "strategy";
  let the evaluator be the strategy.
- **The real ceiling is the evaluator's own vision, not the reward.**
  Chess-R1's root-cause analysis: RL plateaus (~25–30% puzzle accuracy vs
  60–80% expert) because LLMs lack *board-state comprehension* (FEN→FEN
  simulation, MATE detection), and no reward fixes a model that can't read
  the board. ChessLLM (NAACL 2025, arXiv:2501.17186) reached ~1788 Elo with
  **pure SFT on complete games** (+350 Elo from long-game data over short
  fragments) — data with strategic content, not reward shaping, carries the
  load.
- **To cross into expert territory you need search at inference**
  (MCTS/rollout-assisted decoding), because a depth-12 eval cannot teach the
  model strategy the evaluator itself cannot see. The reward is a means; the
  board-understanding + search is the end.

### Sharpened formulation

> The obstacle is (a) **data that contains strategic content** (complete
> games, not fragments) and (b) **reward density** — a Q-function that has
> already done long-horizon credit assignment. Hand-crafting "strategic
> rewards" is a trap; the evidence says dense Q-value reward is the correct
> design, and board comprehension + search at inference is what crosses into
> expert territory.

## Implementation notes

- One persistent Stockfish process per batch (was: one process per eval).
- Crash-safe: engine death mid-batch triggers restart + single retry, never
  a failed training run. Malformed FENs are scored −1.0 without evaluation.
- `eval_depth` default 12 (was 15) — same signal at ~2× speed.
- Back-compat: `evaluate_position_stockfish()` (raw eval dict), the episode
  constants `WIN/LOSS/DRAW_REWARD*`, `LOSING_EVAL_THRESHOLD`, and
  `LEGAL_MOVE_BONUS = 0.0` are retained because `grpo.py` / `metrics.py` /
  `luffy/trace_generator.py` import them.
- Illegal / invalid moves: −1.0 floor (hard gate); `grpo.py` padding was
  changed from −10 to −1 to match the new scale.

## Sources

- Chess-R1 — arXiv:2507.00726 (dense vs sparse reward ablation, plateau
  diagnosis)
- ChessLLM — arXiv:2501.17186 (SFT ceiling, long-game data value)
- Searchless Chess — arXiv:2402.04494 (270M chess-only transformer, GM-level
  without search; evaluation contested)
- dynomight, "More LLM chess" (Nov 2024) — prompt format effects; legal-move
  listing is format-dependent (hurts chat API, helps FEN RL)

## Metrics (SMART)

- M1: legal-move rate ≥ 99.9% on held-out positions (reward must never
  reward illegal moves)
- M2: SFT checkpoint beats raw base by ≥ 300 Elo vs Stockfish L1–5
- M3: dense-reward GRPO beats sparse-reward GRPO by ≥ 5% puzzle accuracy
  (replicates Chess-R1 on our stack)
- M4: LUFFY on/off ablation on Qwen3-1.7B (expect LUFFY > GRPO-only in the
  weak-model regime)
- M5: ≥ 1700 Elo vs Stockfish (strong club level; titled-expert territory
  requires inference-time search)
