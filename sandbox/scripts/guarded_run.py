"""Run a long, resumable command on an 8 GB Mac without letting the model server fill memory and disk.

Found during the v2 render: Ollama's llama-server grew to 10 GB over a few thousand requests; the
excess went to swap, swap files filled the disk, and the render died with "No space left on device".
Unloading the model releases the memory.

This wrapper runs the command and checks every 20 s:
- the model server's memory (resident + compressed, from `top`);
- the free disk space.
If the server is over --max-server-gb or the disk is under --min-free-gb, it stops the command,
unloads every loaded model, waits for memory to settle and starts the command again. The command
must be resumable (p17sim render2 resumes from its cache; p17sim eval resumes per persona).

Usage: python scripts/guarded_run.py [--max-server-gb 5] [--min-free-gb 2.5] -- p17sim render2 --data data/v2
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

OLLAMA = "http://localhost:11434"


def log(msg: str) -> None:
    print(f"[guard {datetime.now():%H:%M:%S}] {msg}", flush=True)


def server_gb() -> float:
    """Memory of the largest llama-server process, resident + compressed, in GB."""
    out = subprocess.run(["top", "-l", "1", "-o", "mem", "-n", "30", "-stats", "command,mem"], capture_output=True,
                         text=True).stdout
    sizes = []
    for line in out.splitlines():
        if line.startswith("llama-server"):
            m = re.search(r"([\d.]+)([KMG])\+?$", line.split()[-1])
            if m:
                sizes.append(float(m.group(1)) * {"K": 1e-6, "M": 1e-3, "G": 1.0}[m.group(2)])
    return max(sizes, default=0.0)


def free_gb() -> float:
    return shutil.disk_usage("/System/Volumes/Data").free / 1e9


def unload_all() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
        for n in names:
            req = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps({"model": n, "keep_alive": 0}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
            log(f"unloaded {n}")
    except OSError as exc:
        log(f"could not unload models: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-server-gb", type=float, default=5.0)
    ap.add_argument("--min-free-gb", type=float, default=2.5)
    ap.add_argument("--max-restarts", type=int, default=200)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("give the command after --")
    restarts = 0
    while True:
        log(f"start: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        reason = None
        while proc.poll() is None:
            time.sleep(20)
            s, f = server_gb(), free_gb()
            if s > a.max_server_gb:
                reason = f"model server at {s:.1f} GB"
            elif f < a.min_free_gb:
                reason = f"only {f:.1f} GB disk free"
            if reason:
                log(f"{reason}: stopping the command to free memory")
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        if reason is None:
            log(f"command finished with exit code {proc.returncode}")
            return proc.returncode
        restarts += 1
        if restarts > a.max_restarts:
            log("too many restarts, giving up")
            return 1
        unload_all()
        time.sleep(30)  # let swap shrink before starting again
        log(f"after unload: server {server_gb():.1f} GB, disk free {free_gb():.1f} GB (restart {restarts})")


if __name__ == "__main__":
    sys.exit(main())
