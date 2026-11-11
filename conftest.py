"""Test defaults: DATA_ROOT falls back to this repo (the dataset is committed at its root)."""

import os
from pathlib import Path

os.environ.setdefault("DATA_ROOT", str(Path(__file__).resolve().parent))
