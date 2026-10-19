import json

nb_path = "notebooks/grpo_and_onpolicy_distillation.ipynb"
nb = json.load(open(nb_path))

cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Why this is very hard\n",
        "\n",
        "Lichess hands us **trillions of positions** and Stockfish is **close to an oracle** — so the raw signal exists. The difficulty is everything around it (Adithya Bhaskar's thread on the Queen paper, arXiv:2610.03695, makes the same points from a much bigger project):\n",
        "\n",
        "1. **The data is not in the right format to learn explanations.** Engines emit centipawn evaluations and principal variations — not natural-language reasoning. There is no corpus of position → expert *explanation*; you have to manufacture one (Queen seeds theirs from a frontier model, then improves it).\n",
        "2. **Even legality is hard for pretrained LMs.** Before GPT-5.6-Sol, no language model except Gemini could play legal moves consistently — the action space (~35 moves/ply, strict syntax) is adversarial to next-token prediction. Our own eval hits the same wall: legal-move rate is the *floor*, not the achievement.\n",
        "3. **Explanations need search; LMs have none.** A good explanation states the best move, the predicted principal variation, *and why the alternatives are worse*. That is look-ahead — an LM's weights do not do it natively, it has to be distilled in (Queen's natural-language Bellman update; our GRPO + distillation stages).\n",
        "4. **Playing strength is a prerequisite.** An explanation of a weak move is a confident wrong answer. This is why headline playing strength barely moved for us while behaviour changed — and why Queen needed 7 iterations (+915 Elo) before its prose was worth trusting.\n",
        "5. **The expert's knowledge is latent, not verbal.** Stockfish (or a chess encoder) 'knows' the position but cannot speak; the LM can speak but does not know. Coupling the two is an architecture problem (cross-attention to a silent encoder in Queen) — a different failure surface than pure training.\n",
        "\n",
        "So the recipe anyone converges on — labelled expert signal, group-relative RL from that signal, then distillation back into the policy — is not a fashion choice. It is the workaround for the five walls above. Sources: [arXiv:2610.03695](https://arxiv.org/abs/2610.03695) (Queen, Princeton PLI), [the thread](https://x.com/AdithyaNLP/status/2107123927525077505).\n",
    ],
    "id": "a7f3c2d1e9b4",
}

# insert after cell 7 ("Where this fits") so it frames the technical sections
insert_at = 8
assert nb["cells"][7]["cell_type"] == "markdown"
nb["cells"].insert(insert_at, cell)

json.dump(nb, open(nb_path, "w"), indent=1)
print(f"inserted why-hard cell at position {insert_at}; notebook now has {len(nb['cells'])} cells")
