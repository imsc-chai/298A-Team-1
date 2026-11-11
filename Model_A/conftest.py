"""Test defaults: DATA_ROOT falls back to the repo root (where the dataset is committed)."""

import os
from pathlib import Path

os.environ.setdefault("DATA_ROOT", str(Path(__file__).resolve().parent.parent))
