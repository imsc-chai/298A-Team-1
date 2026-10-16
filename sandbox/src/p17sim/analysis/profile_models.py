"""Model profiling: each local model measured alone on this machine.

For every model: disk size, parameters, quantization, resident memory once loaded (and the share on
the GPU), cold-load time, and for prompt lengths of about 256, 1,024 and 4,096 tokens: time to first
token, prefill tokens/s and decode tokens/s (3 repetitions, each prompt made unique so Ollama's
prompt cache cannot help). The embedding model gets texts/s instead.
Energy per 1,000 output tokens uses the pre-registered 20 W draw.
Run it when nothing else is using the model server, so the numbers are not shared with a render.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

HOST = "http://localhost:11434"
FILLER = ("The assistant keeps notes about calendars, messages and files on the device. ") * 20
WATTS = 20.0


def _post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(f"{HOST}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{HOST}{path}", timeout=30) as r:
        return json.loads(r.read())


def _unload_all() -> None:
    for m in _get("/api/ps").get("models", []):
        _post("/api/generate", {"model": m["name"], "keep_alive": 0})
    time.sleep(2)


def _prompt(n_tokens: int, tag: int) -> str:
    words = (FILLER * (n_tokens // 200 + 1)).split()[: int(n_tokens * 0.75)]
    return f"[{tag}] " + " ".join(words) + "\nSummarize the notes above in two sentences."


def profile_chat_model(model: str, lengths=(256, 1024, 4096), reps: int = 3) -> dict[str, Any]:
    _unload_all()
    show = _post("/api/show", {"model": model})
    t = time.perf_counter()
    cold = _post("/api/generate", {"model": model, "prompt": "Hi", "stream": False, "think": False,
                                   "options": {"num_predict": 1, "num_ctx": 8192}})
    cold_wall = time.perf_counter() - t
    ps = next((m for m in _get("/api/ps")["models"] if m["name"].startswith(model.split(":")[0])), {})
    rows = []
    for n in lengths:
        for k in range(reps):
            r = _post("/api/generate", {"model": model, "prompt": _prompt(n, 1000 * n + k), "stream": False,
                                        "think": False, "options": {"num_predict": 128, "num_ctx": 8192,
                                                                    "temperature": 0.6, "seed": k}})
            rows.append({"target_prompt_tokens": n, "prompt_tokens": r.get("prompt_eval_count", 0),
                         "ttft_s": (r.get("load_duration", 0) + r.get("prompt_eval_duration", 0)) / 1e9,
                         "prefill_tok_s": r.get("prompt_eval_count", 0) / max(1e-9, r.get("prompt_eval_duration", 1) / 1e9),
                         "decode_tok_s": r.get("eval_count", 0) / max(1e-9, r.get("eval_duration", 1) / 1e9),
                         "output_tokens": r.get("eval_count", 0)})
    by_len = {}
    for n in lengths:
        rs = [r for r in rows if r["target_prompt_tokens"] == n]
        by_len[str(n)] = {k: round(float(np.median([r[k] for r in rs])), 3)
                          for k in ("prompt_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s")}
    decode = float(np.median([r["decode_tok_s"] for r in rows]))
    details = show.get("details", {})
    return {"model": model, "kind": "chat", "parameters": details.get("parameter_size"),
            "quantization": details.get("quantization_level"), "family": details.get("family"),
            "disk_bytes": next((m["size"] for m in _get("/api/tags")["models"] if m["name"] == model), None),
            "resident_bytes": ps.get("size"), "gpu_bytes": ps.get("size_vram"),
            "cold_load_s": round(cold.get("load_duration", 0) / 1e9, 2), "cold_first_reply_wall_s": round(cold_wall, 2),
            "by_prompt_length": by_len, "decode_tok_s_median": round(decode, 2),
            "energy_wh_per_1k_output_tokens": round(1000 / decode * WATTS / 3600, 3) if decode else None,
            "reps": reps}


def profile_embedder(model: str = "all-minilm", batch: int = 64, reps: int = 3) -> dict[str, Any]:
    _unload_all()
    texts = [f"note {i}: " + FILLER[:300] for i in range(batch)]
    t = time.perf_counter()
    _post("/api/embed", {"model": model, "input": ["warm up"]})
    cold = time.perf_counter() - t
    speeds = []
    for k in range(reps):
        t = time.perf_counter()
        _post("/api/embed", {"model": model, "input": [f"{k} {x}" for x in texts]})
        speeds.append(batch / (time.perf_counter() - t))
    ps = next((m for m in _get("/api/ps")["models"] if m["name"].startswith(model)), {})
    return {"model": model, "kind": "embedding",
            "disk_bytes": next((m["size"] for m in _get("/api/tags")["models"] if m["name"].startswith(model)), None),
            "resident_bytes": ps.get("size"), "cold_first_call_s": round(cold, 2),
            "texts_per_s_median": round(float(np.median(speeds)), 1), "batch": batch}


def run(out: Path, chat_models=("qwen3.5:4b", "llama3.2:latest", "phi3:mini")) -> list[dict]:
    results = [profile_chat_model(m) for m in chat_models] + [profile_embedder()]
    _unload_all()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    return results
