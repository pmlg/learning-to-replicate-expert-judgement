"""qwen-chess-rl: Train Qwen (default Qwen3-4B) to play chess via SFT + GRPO."""

import os

from dotenv import load_dotenv

load_dotenv()

# Strip whitespace/CRLF that editors on Windows sometimes leave in .env values.
for _key in ("HF_TOKEN", "WANDB_API_KEY", "STOCKFISH_PATH", "QWEN_CHESS_CHECKPOINT"):
    _value = os.environ.get(_key)
    if _value is not None:
        os.environ[_key] = _value.strip()

__version__ = "0.1.0"
