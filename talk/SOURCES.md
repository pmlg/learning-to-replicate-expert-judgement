# DAT-340 — Verified sources for slide claims

All quantitative claims in the deck must trace to one of these. Verified 2026-10-18.

## Thinking Machines Lab × Bridgewater (primary source)

**"Learning to Replicate Expert Judgment in Financial Tasks"** — Sarah Su, Kevin Zhu,
Emily Xiao, Rohan Alur, Daniel Kang (Bridgewater AIA Labs) in collaboration with
Thinking Machines Lab, published **30 Jun 2026**.
<https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/>

Verified figures (from the primary source, not press coverage):

- Frontier models, naive task prompt: **~50% avg accuracy** across 6 tasks
  (Opus 4.6: 47.2, Gemini 3.1 Pro: 50.1, GPT 5.4: 47.2, GPT 5.5: 48.5, Opus 4.8: 45.6).
- Expert-written prompts: **mid-70s** (77.2, 74.3, 75.8, 78.2, 78.0) — still under the 80% trust threshold.
- Base **Qwen3-235B**, trained on **Tinker**.
  - Qwen base: **44.8%** avg accuracy
  - Qwen + GRPO: **73.48%**
  - Full recipe: **84.66%** vs best frontier **78.2%** → **29.8% fewer errors**
- Cost: **13.8× reduction** in inference cost per task.
- Leave-one-out ablations: w/o interleaved batching **72.18%**; w/o CISPO+asymmetric clips **74.56%**; w/o OPD (on-policy distillation) **72.39%**; OPD w/ best-val teacher **81.55%**.
- Label strategy: contractor labels insufficient → model-vs-labeler disagreement routed to expert investors for re-review; final eval on held-out test set.
- Caveats stated by authors: internal methodology, not an independent benchmark; six tasks only; inference cost only (excludes labelling/training cost).

Secondary corroboration (do not quote numbers from these; primary only):
winbuzzer.com (2026-07-04), zenml.io/llmops-database, megaoneai.com analysis.

## Repo results (our chess numbers)

Canonical file: `evaluation_results/comparison.json` (n=1000 test positions, temperature 0.1).

| Metric | SFT | RL (GRPO) |
|---|---|---|
| Legal-move rate | 100% | 100% |
| Median CPL | 81.0 | 90.0 |
| Mean CPL (blunder-dominated, std ≈ 2900) | 1045.7 | 1022.7 |
| vs Stockfish (20 games) | 0–20 | 0–20 |
| Avg game length (plies) | 24.3 | 27.7 |
| vs random (10 games) | 0W / 9D / 1L | 0W / 9D / 1L |

Resolved issues:
1. **Canonical file decided (2026-10-19): `comparison.json`** — it is the only file with both
   checkpoints from one harness run. `benchmark_results.json` is a stale/duplicate rl/final run
   (29.9 plies, median CPL 102.5, 0W/10D vs random) and is NOT canonical; do not quote from it.
2. No GRPO+LUFFY result exists in the repo — the stage-3 slide presents LUFFY as
   designed-with-external-evidence, not measured-here.
3. vs-random 9/10 draws anomaly: likely draw adjudication / move-limit artifact; present as-is.

## Other citations on the final slide

- LUFFY — "RL with Mixing Guidance from Off-policy Data" — arXiv:2504.14945.
- On-policy distillation — Thinking Machines Lab blog, thinkingmachines.ai.
