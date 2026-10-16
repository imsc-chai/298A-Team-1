"""Real-world reference statistics for the v2 generator and the bias audit.

Source: the Enron email corpus (released by FERC; parsed copy `Hellisotherpeople/enron_emails_parsed`
on Hugging Face). Rows are fetched at random offsets through the datasets-server API and reduced to
aggregate statistics in memory. No message text, address or name is written to disk.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np

DATASET = "Hellisotherpeople/enron_emails_parsed"
ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=train&offset={o}&length={n}"
N_ROWS = 535_703
OUT = Path(__file__).with_name("reference_stats.json")
# python.org builds ship without CA certificates; use the system bundle when it exists.
_SSL = ssl.create_default_context(cafile="/etc/ssl/cert.pem") if Path("/etc/ssl/cert.pem").exists() else None
_QUOTED = re.compile(r"(-{3,}\s*Original Message|-{3,}\s*Forwarded by|\nFrom: )", re.I)


def _fetch(offset: int, length: int) -> list[dict]:
    url = ROWS_URL.format(ds=DATASET, o=offset, n=length)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60, context=_SSL) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except (OSError, ValueError, KeyError) as exc:  # network hiccup: back off and retry
            last = exc
            time.sleep(2 ** attempt)
    print(f"page at offset {offset} failed: {last}")
    return []


def _own_words(body: str) -> int:
    """Words the sender wrote, before any quoted or forwarded block."""
    own = _QUOTED.split(body or "", maxsplit=1)[0]
    return len(own.split())


def collect(n_pages: int = 100, page: int = 100, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    offsets = sorted(int(o) for o in rng.choice(N_ROWS - page, size=n_pages, replace=False))
    hours, weekdays, words, senders = Counter(), Counter(), [], Counter()
    replies = rows = bad_dates = 0
    for o in offsets:
        for row in _fetch(o, page):
            rows += 1
            try:
                dt = parsedate_to_datetime(row["date"])
            except (TypeError, ValueError):
                bad_dates += 1
                continue
            hours[dt.hour] += 1  # sender's local clock, as written in the header
            weekdays[dt.weekday()] += 1
            words.append(_own_words(row["body"]))
            senders[row["from"]] += 1
            replies += (row["subject"] or "").lower().startswith("re:")
    n = sum(hours.values())
    if n < 1000:
        raise SystemExit(f"only {n} dated rows fetched; refusing to write reference stats")
    w = np.array(words)
    # Enron Date headers carry a clock offset: taken at face value, mail peaks at 07:00-09:00 with
    # 5% an hour at 01:00-05:00. We shift the curve by the k that puts the most mail inside
    # 08:00-18:00 and keep the raw curve next to it.
    raw_pmf = np.array([hours[h] / n for h in range(24)])
    shift = int(max(range(24), key=lambda k: np.roll(raw_pmf, k)[8:18].sum()))
    per_sender = np.array(sorted(senders.values(), reverse=True))
    return {
        "source": f"huggingface.co/datasets/{DATASET} (Enron corpus, FERC public release)",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": {"rows": rows, "dated": n, "bad_dates": bad_dates, "pages": n_pages, "page_size": page,
                   "offset_seed": seed},
        "header_hour_pmf": raw_pmf.tolist(),
        "hour_shift_estimate": shift,
        "hour_of_day_pmf": np.roll(raw_pmf, shift).tolist(),
        "weekday_pmf_mon0": [weekdays[d] / n for d in range(7)],
        "words_quantiles": {str(q): float(np.quantile(w, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "words_log_mean": float(np.log1p(w).mean()),
        "words_log_sd": float(np.log1p(w).std()),
        "reply_share": replies / n,
        "sender_volume": {"senders": int(len(per_sender)),
                          "top10pct_share": float(per_sender[:max(1, len(per_sender) // 10)].sum() / per_sender.sum())},
        "note": "Aggregates only. Corporate mail from 1999-2002, so it is a reference for timing and length "
                "shape, not for content or topics.",
    }


if __name__ == "__main__":
    stats = collect()
    OUT.write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({k: stats[k] for k in ("sample", "words_quantiles", "reply_share", "sender_volume")}, indent=1))
