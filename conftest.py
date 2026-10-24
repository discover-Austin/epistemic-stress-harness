# Puts the repo root on sys.path so `pytest` finds the harness and runners packages.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
