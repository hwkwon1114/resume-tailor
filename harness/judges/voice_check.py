"""VoiceCheck — mechanical stylistic comparison (no LLM call in v1).

Computes per-bullet voice_drift_score = normalized distance between input and
output bullet stylistic signature. Surfaced informationally; not blocking
in v1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.schema import Resume, iter_bullets

_JARGON_PATH = Path(__file__).parent / "jargon_words.txt"
_JARGON: frozenset[str] = frozenset(w.strip().lower() for w in _JARGON_PATH.read_text().splitlines() if w.strip())

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_TOKEN_SPLIT = re.compile(r"\W+")


@dataclass(slots=True)
class _Signature:
    avg_sentence_len: float
    jargon_density: float  # jargon hits / total tokens
    passive_ratio: float  # crude: count of " was "/" were "/" by " patterns / sentences

    def distance(self, other: "_Signature") -> float:
        a = abs(self.avg_sentence_len - other.avg_sentence_len) / max(self.avg_sentence_len, 1.0)
        b = abs(self.jargon_density - other.jargon_density) * 5.0  # weight jargon shifts heavily
        c = abs(self.passive_ratio - other.passive_ratio) * 2.0
        return min(a + b + c, 1.0)


@dataclass(slots=True)
class _BulletDrift:
    output_id: str
    drift_score: float
    output_signature: _Signature
    flagged_input_ids: list[str]


@dataclass(slots=True)
class VoiceCheckResult:
    per_bullet: list[_BulletDrift] = field(default_factory=list)

    @property
    def mean_drift(self) -> float:
        if not self.per_bullet:
            return 0.0
        return sum(b.drift_score for b in self.per_bullet) / len(self.per_bullet)

    @property
    def max_drift(self) -> float:
        if not self.per_bullet:
            return 0.0
        return max(b.drift_score for b in self.per_bullet)


def _signature(text: str) -> _Signature:
    text = text or ""
    sents = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()] or [text]
    tokens = [t.lower() for t in _TOKEN_SPLIT.split(text) if t]
    if not tokens:
        return _Signature(0.0, 0.0, 0.0)
    avg_sent = len(tokens) / len(sents)
    jargon_hits = sum(1 for t in tokens if t in _JARGON)
    jargon_density = jargon_hits / len(tokens)
    passive_hits = len(re.findall(r"\b(was|were)\b\s+\w+ed\b|\bby\s+the\b", text.lower()))
    passive_ratio = passive_hits / len(sents)
    return _Signature(avg_sent, jargon_density, passive_ratio)


class VoiceCheck:
    name = "voice_check"

    def run(self, *, input_resume: Resume, output_resume: Resume) -> VoiceCheckResult:
        in_sig_by_id = {b.id: _signature(b.text) for b in iter_bullets(input_resume)}
        per_bullet: list[_BulletDrift] = []
        for b in iter_bullets(output_resume):
            out_sig = _signature(b.text)
            if not b.source_ids:
                per_bullet.append(_BulletDrift(b.id, 1.0, out_sig, []))
                continue
            distances = []
            for sid in b.source_ids:
                in_sig = in_sig_by_id.get(sid)
                if in_sig is None:
                    continue
                distances.append(in_sig.distance(out_sig))
            drift = min(distances) if distances else 1.0
            per_bullet.append(_BulletDrift(b.id, drift, out_sig, list(b.source_ids)))
        return VoiceCheckResult(per_bullet=per_bullet)
