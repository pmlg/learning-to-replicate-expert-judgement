"""DAT-340: build the honest results charts from evaluation_results/comparison.json.

Two-point comparisons only (SFT final vs RL final) — no training curves exist
(no checkpoints/logs in the repo), so nothing here invents a trajectory.
Outputs: talk/charts/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "talk" / "charts"
OUT.mkdir(exist_ok=True)

cmp_ = json.load(open(ROOT / "evaluation_results" / "comparison.json"))
sft = cmp_["sft/final"]
rl = cmp_["rl/final"]

BLUE, RED, GREY = "#1f6fd6", "#c0392b", "#6b7078"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


# --- chart 1: centipawn loss (the honest headline) --------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
x = [0, 1]
w = 0.38
# mean CPL with +-1 std as error bars (heavy-tailed: std ~2,900)
ax.bar([0 - w / 2], [sft["test_set_metrics"]["avg_centipawn_loss"]], w, yerr=[sft["test_set_metrics"]["std_centipawn_loss"] / 1000],
       color=BLUE, capsize=5, label="SFT final")
ax.bar([0 + w / 2], [sft["test_set_metrics"]["median_centipawn_loss"]], w, color="#7fb2e5",
       label="SFT median")
ax.bar([1 - w / 2], [rl["test_set_metrics"]["avg_centipawn_loss"]], w, yerr=[rl["test_set_metrics"]["std_centipawn_loss"] / 1000],
       color=RED, capsize=5, label="RL (GRPO) final")
ax.bar([1 + w / 2], [rl["test_set_metrics"]["median_centipawn_loss"]], w, color="#e59898",
       label="RL median")
for xi, v in ((0 - w / 2, sft["test_set_metrics"]["avg_centipawn_loss"]),
              (0 + w / 2, sft["test_set_metrics"]["median_centipawn_loss"]),
              (1 - w / 2, rl["test_set_metrics"]["avg_centipawn_loss"]),
              (1 + w / 2, rl["test_set_metrics"]["median_centipawn_loss"])):
    ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=11)
ax.set_xticks([0, 1])
ax.set_xticklabels(["SFT final", "RL (GRPO) final"])
ax.set_ylabel("centipawn loss vs Stockfish\n(1,000-position test set)")
ax.set_title("Centipawn loss: -2.2% mean after RL\n(median moved 81 -> 90; both heavy-tailed)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "cpl_sft_vs_rl.png", bbox_inches="tight")

# --- chart 2: behaviour changed, headline didn't ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), dpi=150)
axes[0].bar([0, 1], [sft["vs_stockfish"]["avg_game_length"], rl["vs_stockfish"]["avg_game_length"]], 0.5, color=[BLUE, RED])
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["SFT", "RL"])
axes[0].set_title("Avg game length vs Stockfish\n24.3 -> 27.7 plies")
for xi, v in zip([0, 1], [sft["vs_stockfish"]["avg_game_length"], rl["vs_stockfish"]["avg_game_length"]]):
    axes[0].text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=11)

axes[1].bar([0, 1], [sft["test_set_metrics"]["legal_move_pct"] * 100,
                    rl["test_set_metrics"]["legal_move_pct"] * 100], 0.5, color=[BLUE, RED])
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["SFT", "RL"])
axes[1].set_ylim(0, 115)
axes[1].set_title("Legal-move rate\n100% both (n=1,000)")
for xi, v in zip([0, 1], [100, 100]):
    axes[1].text(xi, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=11)

axes[2].bar([0, 1], [20, 20], 0.5, color=[GREY, GREY])
axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["SFT", "RL"])
axes[2].set_title("vs Stockfish (20 games)\nL 20-0 both; est. Elo floor 1600")
axes[2].set_yticks([])
axes[2].text(0, 10, "0W 0D\n20L", ha="center", fontsize=13, color="white")
axes[2].text(1, 10, "0W 0D\n20L", ha="center", fontsize=13, color="white")

fig.suptitle("Behaviour changed; the headline didn't", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "behaviour_vs_headline.png", bbox_inches="tight")

print("charts written:", sorted(p.name for p in OUT.glob("*.png")))
print("SPY check:", sft["test_set_metrics"]["avg_centipawn_loss"], rl["test_set_metrics"]["avg_centipawn_loss"])
