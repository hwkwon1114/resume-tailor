"""Eval cache — opt-in (default OFF) so eval-as-regression sees stochastic variance.

Key: sha256(jd + canonical_resume_json + prompt_version).
Eviction: LRU at 100 entries by mtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(".cache/eval")
CAPACITY = 100


def cache_key(jd: str, resume_dict: dict, prompt_version: str = "v1") -> str:
    canonical = json.dumps(resume_dict, sort_keys=True)
    h = hashlib.sha256()
    h.update(jd.encode("utf-8"))
    h.update(b"||")
    h.update(canonical.encode("utf-8"))
    h.update(b"||")
    h.update(prompt_version.encode("utf-8"))
    return h.hexdigest()[:32]


def get(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def put(key: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload))
    _evict_lru()


def _evict_lru() -> None:
    if not CACHE_DIR.exists():
        return
    entries = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    while len(entries) > CAPACITY:
        oldest = entries.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass


def clear_cache() -> int:
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.json"):
        p.unlink()
        n += 1
    return n


def seconds_since(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return float("inf")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--clear":
        n = clear_cache()
        print(f"cleared {n} entries from {CACHE_DIR}")
        sys.exit(0)
    print(f"cache dir: {CACHE_DIR.resolve()}")
    if CACHE_DIR.exists():
        print(f"entries: {len(list(CACHE_DIR.glob('*.json')))}")
