"""Pytest configuration for test imports."""
import sys
from pathlib import Path

# Add project root to sys.path so `import qwen_chess_rl` works
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
