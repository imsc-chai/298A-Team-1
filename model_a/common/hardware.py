"""Hardware fingerprint recorded with every result so latency numbers are traceable.

The client may run on a laptop while Ollama runs on a Colab GPU, so we record both the local
machine and the GPU visible here (Colab runs Ollama on the same VM).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "gpu": None,
    }
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            info["gpu"] = [line.strip() for line in proc.stdout.strip().splitlines()]
        else:
            info["gpu_error"] = proc.stderr.strip()[:200] or f"nvidia-smi exit {proc.returncode}"
    return info
