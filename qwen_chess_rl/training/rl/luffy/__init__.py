"""LUFFY off-policy guidance for GRPO training.

Implements the approach from the LUFFY paper: inject expert (Stockfish)
move sequences as off-policy traces during policy optimization, using
importance-sampling corrections and regularization so the model benefits
from demonstrations without collapsing into a Stockfish clone.
"""
