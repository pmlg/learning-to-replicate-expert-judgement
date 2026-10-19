# PMLG talk — full rehearsal script (26 slides, ~40:00)

**How to practise today (3 passes, ~2 h 15 m total):**
1. **Pass 1 — read-through (25 min):** read each section aloud once, no timer. Mark any sentence you stumble on.
2. **Pass 2 — timed (45 min):** run the deck with `cue_sheet.md` timings. Target 40:00; note where you drift >30 s. The 0.5-min transition slides (3, 5, 8, 10, 16, 19) are where drift hides.
3. **Pass 3 — full dress (45 min + 20 min Q&A):** no notes except slide cues. Record on your phone and listen back once at 1.5×.

**Numbers you must be able to say without looking:**
- Frontier naive ~50%; expert prompts mid-70s (77.2–78.2)
- Bridgewater: Qwen base 44.8 → GRPO 73.5 → full recipe 84.7; best frontier 78.2; 13.8× cheaper
- Our chess: CPL 1045.7 → 1022.7; median 81 → 90; game length 24.3 → 27.7; 0W-0D-20L both; legal moves 100%
- Ablations: no interleaved batching 72.2; no OPD 72.4; no CISPO 74.6


## Slide 1 — Teaching LLMs Expert Judgment: A Chess RL Case Study
[0:00-1:30] Welcome. One sentence up front: expert judgment is tacit, so it has to be trained into a model, not prompted into it. Tonight: chess with Qwen3-1.7B as the controlled testbed, then the Bridgewater result that shows the same recipe works at production scale.

## Slide 2 — The hook: prompting can't transfer judgment
[1:30-4:00] Start with the strongest evidence, not ours. Frontier models averaged ~50% (Opus 4.6: 47.2, Gemini 3.1 Pro: 50.1, GPT 5.4: 47.2...). Expert prompts lifted to mid-70s (77.2, 74.3, 75.8, 78.2, 78.0) - coin flip to C+ student. Prompt optimisation hit a wall. Quote is from the TML/Bridgewater post.

## Slide 3 — The thesis
[4:00-4:30] Transition.

## Slide 4 — Thesis: tacit judgment is trained in, not prompted
[4:30-6:30] Chess gives us a perfect expert (Stockfish ~3200 Elo), an exact reward, and infinite labelled positions. Financial judgment at Bridgewater is the noisy real-world version of the same problem. Everything we learn about reward design and credit assignment transfers.

## Slide 5 — Setup: chess as the controlled testbed
[6:30-7:00] Transition to setup.

## Slide 6 — Setup: Qwen3-1.7B + Stockfish reward
[7:00-9:00] Deliberately the smallest credible model. Question we're really asking: how small can a model be and still reason? Three training stages mirror the talk structure.

## Slide 7 — How we measure
[9:00-10:30] Explain CPL briefly: 0 = engine's move, 100 cp = roughly a pawn of lost value. Legal-move rate is the floor: an LLM that emits 'e4 e5 Nf3' as one token or hallucinates squares fails here. Our headline floor stat: 100% legal moves on 1000 held-out positions.

## Slide 8 — Stage 1 - SFT on game transcripts
[10:30-11:00] Transition.

## Slide 9 — Stage 1 - SFT: what the model learns
[11:00-13:00] SFT buys competence, not judgment. The odd result to flag honestly: 9/10 draws vs random. Likely draw-adjudication / move-limit artifacts - we show the numbers as they are. Data format note: transcripts teach the token patterns of chess; the model still has no internal critic.

## Slide 10 — Stage 2 - GRPO with Stockfish as the reward
[13:00-13:30] Transition.

## Slide 11 — Stage 2 - GRPO: reward from the engine
[13:30-15:30] Keep GRPO math light here (the notebook does the step-by-step). Key intuition: within a group of sampled moves, the model learns which of ITS OWN candidate moves the engine prefers. That is expert signal, per decision.

## Slide 12 — Failure modes: sparse reward, brutal credit assignment
[15:30-18:00] This is the honest heart of the talk. Compare: a chat model gets per-token feedback; our policy gets one scalar after ~30 decisions. That is why RL from game outcomes alone stalls - and it motivates both per-move engine shaping and the distillation stage.

## Slide 13 — Strategic vs tactical: why chess is hard for a 1.7B model
[18:00-20:00] Tie back to the thesis: strategic judgment in chess is tacit even for humans - club players can't write down why a move 'feels' right. That's the same structure as Bridgewater investors filtering documents.

## Slide 14 — GRPO results - the honest read
[20:00-22:30] Numbers are real, from the repo. Say it straight: RL did NOT move the headline evals - median CPL slightly up (81->90), still 0-20 vs Stockfish. What changed: longer games (24.3 -> 27.7 plies) - the policy survives longer, plays differently. Discuss benchmark_results.json discrepancy: different rl/final run; we must freeze one canonical file.

## Slide 15 — So did RL teach anything new?
[22:30-24:30] This is the 'RL: new capabilities or sharpened SFT?' key question answered for OUR case: sharpening. Bridge to LUFFY/on-policy distillation: the problem isn't the model, it's the reward density.

## Slide 16 — Stage 3 - off-policy learning: LUFFY and distillation
[24:30-25:00] Transition.

## Slide 17 — LUFFY: off-policy guidance from a stronger teacher
[25:00-27:00] Be explicit about repo status: our LUFFY eval is not yet run - the comparison-table cell is empty (audit 2026-10-18). Present LUFFY as the designed next stage, with the TML/Bridgewater ablation as external evidence the recipe class works.

## Slide 18 — Why on-policy distillation helps
[27:00-29:00] The key mental model: RL asks 'which of my moves was good?' (sparse, noisy). Distillation asks 'match the teacher's distribution HERE' (dense, per-token, on the states I actually visit). Examples: SFT model sees only Stockfish games, not its own blunders; OPD trains on the model's own states with teacher targets.

## Slide 19 — Evidence at scale: Thinking Machines x Bridgewater
[29:00-29:30] Transition to the payoff.

## Slide 20 — The same recipe, at production scale
[29:30-31:30] This is our thesis proven outside chess. Walk the pipeline: contractor labels failed (models trained on them performed poorly) - the fix was routing contested examples to real investors. Note the overlap with our recipe: GRPO + on-policy distillation, same component names.

## Slide 21 — Bridgewater numbers
[31:30-33:00] Read the bars: 44.8 base -> 73.5 after GRPO alone (the biggest single jump, same as our SFT->RL story) -> 84.7 after the full recipe. Their ablations: removing interleaved batching drops to 72.2%; removing OPD drops to 72.4%; CISPO+asymmetric clips 74.6%. A 13.8x cost cut with better accuracy is the production argument.

## Slide 22 — Caveats (say them before the audience does)
[33:00-34:30] Credibility slide. Both the Bridgewater result and our chess numbers are self-reported on internal evals. State limits, then move to takeaways. This honesty is itself part of the talk's argument about expert judgment.

## Slide 23 — Key questions this testbed answers
[34:30-37:00] The three questions from the abstract, answered with our data plus the external evidence. Keep each to ~45 seconds.

## Slide 24 — Takeaways
[37:00-38:30] Five lines, each proven earlier in the deck. Slow down here.

## Slide 25 — Demo & materials
[38:30-39:30] Decision still open: live vs recorded demo (ticket task). The notebook §6 runs offline from committed outputs - the safe default. Re-run on the training box so outputs show the trained policy, not the base-model fallback.

## Slide 26 — Thank you - questions?
[39:30-40:00+] Q&A. Likely questions: why chess (objective reward), why not Leela (we study LLM training, not chess engines), will the recipe transfer (Bridgewater evidence), what's next (LUFFY eval on our repo).
