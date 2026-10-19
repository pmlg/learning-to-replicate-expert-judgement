# PMLG dry-run cue sheet

Generated from `pmlg_chess_rl_talk_draft.pptx` speaker notes. **Total speaking time: 40:00** (40 min) across 26 timed slides.

| # | Start | End | Dur | Slide |
|---|-------|-----|-----|-------|
| 1 | 0:00 | 1:30 | 1.5m | Teaching LLMs Expert Judgment: A Chess RL Case Study |
| 2 | 1:30 | 4:00 | 2.5m | The hook: prompting can't transfer judgment |
| 3 | 4:00 | 4:30 | 0.5m | The thesis |
| 4 | 4:30 | 6:30 | 2.0m | Thesis: tacit judgment is trained in, not prompted |
| 5 | 6:30 | 7:00 | 0.5m | Setup: chess as the controlled testbed |
| 6 | 7:00 | 9:00 | 2.0m | Setup: Qwen3-1.7B + Stockfish reward |
| 7 | 9:00 | 10:30 | 1.5m | How we measure |
| 8 | 10:30 | 11:00 | 0.5m | Stage 1 - SFT on game transcripts |
| 9 | 11:00 | 13:00 | 2.0m | Stage 1 - SFT: what the model learns |
| 10 | 13:00 | 13:30 | 0.5m | Stage 2 - GRPO with Stockfish as the reward |
| 11 | 13:30 | 15:30 | 2.0m | Stage 2 - GRPO: reward from the engine |
| 12 | 15:30 | 18:00 | 2.5m | Failure modes: sparse reward, brutal credit assignment |
| 13 | 18:00 | 20:00 | 2.0m | Strategic vs tactical: why chess is hard for a 1.7B model |
| 14 | 20:00 | 22:30 | 2.5m | GRPO results - the honest read |
| 15 | 22:30 | 24:30 | 2.0m | So did RL teach anything new? |
| 16 | 24:30 | 25:00 | 0.5m | Stage 3 - off-policy learning: LUFFY and distillation |
| 17 | 25:00 | 27:00 | 2.0m | LUFFY: off-policy guidance from a stronger teacher |
| 18 | 27:00 | 29:00 | 2.0m | Why on-policy distillation helps |
| 19 | 29:00 | 29:30 | 0.5m | Evidence at scale: Thinking Machines x Bridgewater |
| 20 | 29:30 | 31:30 | 2.0m | The same recipe, at production scale |
| 21 | 31:30 | 33:00 | 1.5m | Bridgewater numbers |
| 22 | 33:00 | 34:30 | 1.5m | Caveats (say them before the audience does) |
| 23 | 34:30 | 37:00 | 2.5m | Key questions this testbed answers |
| 24 | 37:00 | 38:30 | 1.5m | Takeaways |
| 25 | 38:30 | 39:30 | 1.0m | Demo & materials |
| 26 | 39:30 | 40:00+ | 0.5m | Thank you - questions? |

## Pacing notes

- Q&A opens at 39:30 — 5:30 left of a 45-min slot.
- Sub-30s transition slides: #3, #5, #8, #10, #16, #19, #26
