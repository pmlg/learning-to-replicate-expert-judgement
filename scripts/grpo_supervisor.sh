#!/bin/bash
# Supervises GRPO training: restarts from the latest saved checkpoint on
# crash (observed: sporadic CUDA/CUBLAS errors at 9B scale operating near
# the VRAM ceiling, transient network errors, and occasional environment
# resets that kill this script itself). Each attempt writes to its own
# from-<offset> subdirectory (GRPO's own checkpoint-N numbering is always
# local to a single invocation, starting at 0 — reusing one output dir
# across restarts would silently collide/overwrite checkpoint names from
# different attempts).
set -u

BASE_DIR="checkpoints/rl-9b"
SFT_CHECKPOINT="checkpoints/sft-9b/final"
# Steps 1-300: fixed depth-3 opponent. Steps 300-600: opponent annealed
# 3 -> 6. Both phases plateaued at ~1600 Elo (0/20 vs depth-8 Stockfish,
# 0 draws even with search-vetted "hybrid" eval). Diagnosing the step-600
# checkpoint's losses showed every one traced back to a single simple
# mid-game blunder (a hung piece to an immediate recapture), not slow
# positional decline — so this next 600-step phase holds the opponent
# steady at depth 6 (now config.py's default) and instead strengthens
# rollout_vet_depth/time_limit/margin (also now in config.py) to correct
# more of those blunders during training, reinforcing the policy away
# from them via the vetted-move imitation signal (see rollout.py).
TOTAL_STEPS=1200
MAX_RETRIES=60
LOG="$BASE_DIR/train.log"
mkdir -p "$BASE_DIR"

# A `model.save_pretrained()`/`tokenizer.save_pretrained()` call killed
# mid-write (observed: an environment reset killed training between the
# safetensors temp-file write and its atomic rename) leaves a checkpoint-N
# directory that EXISTS but has none of its actual contents — just leaves
# a stray .tmp<hash> file. The dir still matches the checkpoint-* glob
# below, so without this check a corrupted checkpoint gets silently picked
# as the resume point and fails downstream with a confusing "need
# sentencepiece" tokenizer error that has nothing to do with the real
# problem (no tokenizer.json at all).
is_valid_checkpoint() {
  [ -f "$1/adapter_model.safetensors" ] && [ -f "$1/tokenizer.json" ] && [ -f "$1/adapter_config.json" ]
}

done_steps=0
ckpt="$SFT_CHECKPOINT"

# Resume across supervisor restarts too: scan all prior attempts for the
# checkpoint representing the most true cumulative steps. A "final" dir's
# true total is NOT necessarily today's TOTAL_STEPS — a prior phase may
# have run with a smaller target (e.g. this script bumped TOTAL_STEPS
# 300 -> 600 for a second training phase after the first 300 completed).
# Each live completion below writes a .true_total_steps marker recording
# the actual count; fall back to the highest sibling checkpoint-N (always
# a true lower bound, and exact whenever the final step lands on a
# save_steps boundary) for older "final" dirs saved before this marker
# existed.
for d in "$BASE_DIR"/from-*/checkpoint-* "$BASE_DIR"/from-*/final; do
  [ -d "$d" ] || continue
  is_valid_checkpoint "$d" || continue
  base=$(basename "$(dirname "$d")")
  offset=${base#from-}
  n=$(basename "$d")
  if [ "$n" = "final" ]; then
    if [ -f "$d/.true_total_steps" ]; then
      total=$(cat "$d/.true_total_steps")
    else
      sibling_max=$(ls -d "$(dirname "$d")"/checkpoint-* 2>/dev/null | sed -E 's/.*checkpoint-([0-9]+)/\1/' | sort -n | tail -1)
      total=${sibling_max:-$offset}
    fi
  else
    local_step=${n#checkpoint-}
    total=$((offset + local_step))
  fi
  if [ "$total" -gt "$done_steps" ]; then
    done_steps=$total
    ckpt="$d"
  fi
done

retry=0
while [ "$retry" -lt "$MAX_RETRIES" ]; do
  remaining=$((TOTAL_STEPS - done_steps))
  if [ "$remaining" -le 0 ]; then
    echo "[supervisor] Reached $done_steps/$TOTAL_STEPS steps. Done. Final checkpoint: $ckpt" >> "$LOG"
    break
  fi

  attempt_dir="$BASE_DIR/from-$done_steps"
  echo "[supervisor] Attempt $((retry+1))/$MAX_RETRIES: resuming from $ckpt (true step $done_steps), $remaining steps remaining -> $attempt_dir" >> "$LOG"

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m qwen_chess_rl.training.rl.grpo \
    --sft-checkpoint "$ckpt" \
    --steps "$remaining" \
    --output-dir "$attempt_dir" \
    >> "$LOG" 2>&1
  exit_code=$?

  if [ -d "$attempt_dir/final" ] && is_valid_checkpoint "$attempt_dir/final"; then
    done_steps="$TOTAL_STEPS"
    ckpt="$attempt_dir/final"
    echo "$done_steps" > "$attempt_dir/final/.true_total_steps"
    echo "[supervisor] Training process exited cleanly (code $exit_code), reached $done_steps/$TOTAL_STEPS." >> "$LOG"
    break
  fi

  latest_local=""
  for d in $(ls -d "$attempt_dir"/checkpoint-* 2>/dev/null | sed -E 's/.*checkpoint-([0-9]+)/\1/' | sort -rn); do
    if is_valid_checkpoint "$attempt_dir/checkpoint-$d"; then
      latest_local="$d"
      break
    fi
  done
  if [ -n "${latest_local:-}" ]; then
    done_steps=$((done_steps + latest_local))
    ckpt="$attempt_dir/checkpoint-$latest_local"
  fi

  echo "[supervisor] Attempt exited (code $exit_code) after reaching true step $done_steps. Retrying in 20s..." >> "$LOG"
  retry=$((retry+1))
  sleep 20
done

if [ "$retry" -ge "$MAX_RETRIES" ]; then
  echo "[supervisor] Hit max retries ($MAX_RETRIES). Stopping at true step $done_steps (checkpoint: $ckpt)." >> "$LOG"
fi
