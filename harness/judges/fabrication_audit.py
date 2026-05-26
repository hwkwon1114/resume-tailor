"""FabricationAudit — LLM-as-judge per-bullet support scoring.

Splits output bullets into BATCH_SIZE chunks and audits chunks in parallel,
then merges. Sized so each call's payload stays well under the Gemini CLI
120s timeout wall (a single-call audit on 13-15 bullets consistently hit it).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from harness.models import ModelClient
from harness.prompts import load_prompt
from harness.schema import Resume, iter_bullets

THRESHOLD = 0.6
# Per the judge rubric: scores below this are "named specific not in input"
# (0.2 — invented tool/quantity) or worse — hard fabrications. Scores in
# [BORDERLINE_FLOOR, THRESHOLD) are "interpretation drift" (0.4) — softer
# judge calls where the run-to-run noise dominates.
BORDERLINE_FLOOR = 0.4
# Tolerate up to this many borderline (0.4-level) bullets before failing the
# audit. Hard fabrications (< 0.4) always count toward failure. The selected
# attempt's audit needs `not flagged_bullets` to clear, so this gives the
# strict-mode gate a small noise budget without letting through real invention.
BORDERLINE_TOLERANCE = 1
BATCH_SIZE = 4
_MAX_PARALLEL_CHUNKS = 5


class _BulletAudit(BaseModel):
    bullet_id: str
    support_score: float = Field(ge=0.0, le=1.0)
    reason: str


class _AuditReport(BaseModel):
    audits: list[_BulletAudit]


@dataclass(slots=True)
class FabricationAuditResult:
    per_bullet: dict[str, _BulletAudit]
    flagged_bullets: list[_BulletAudit] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.per_bullet:
            return 1.0
        ok = sum(1 for a in self.per_bullet.values() if a.support_score >= THRESHOLD)
        return ok / len(self.per_bullet)


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class FabricationAudit:
    name = "fabrication_audit"

    def run(self, *, input_resume: Resume, output_resume: Resume, model: ModelClient) -> FabricationAuditResult:
        input_bullets = {b.id: b.text for b in iter_bullets(input_resume)}
        output_bullets = list(iter_bullets(output_resume))
        if not output_bullets:
            return FabricationAuditResult(per_bullet={})

        items = []
        for b in output_bullets:
            cited = [{"id": sid, "text": input_bullets.get(sid, "<UNKNOWN>")} for sid in b.source_ids]
            items.append({"bullet_id": b.id, "output_text": b.text, "cited_inputs": cited})

        system = load_prompt("system_fabrication_judge")
        chunks = _chunk(items, BATCH_SIZE)

        def _audit_chunk(chunk_items: list) -> list[_BulletAudit]:
            prompt = (
                "Audit each output bullet against its cited input bullets. "
                "Return a JSON object {\"audits\": [{bullet_id, support_score, reason}, ...]} "
                "with EXACTLY one entry per output bullet.\n\n"
                f"OUTPUT BULLETS:\n{json.dumps(chunk_items, indent=2)}"
            )
            resp = model.generate_structured(system=system, prompt=prompt, schema=_AuditReport)
            # Filter to this chunk's bullet ids — defends against a model that
            # hallucinates extra audits or accidentally returns audits from
            # other chunks (test fixtures that share a single canned response
            # exhibit this; real Gemini occasionally does too).
            wanted = {item["bullet_id"] for item in chunk_items}
            return [a for a in resp.data.audits if a.bullet_id in wanted]

        if len(chunks) == 1:
            chunk_audits = [_audit_chunk(chunks[0])]
        else:
            workers = min(len(chunks), _MAX_PARALLEL_CHUNKS)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                chunk_audits = list(pool.map(_audit_chunk, chunks))

        all_audits = [a for chunk in chunk_audits for a in chunk]
        per_bullet = {a.bullet_id: a for a in all_audits}
        hard_fails = [a for a in all_audits if a.support_score < BORDERLINE_FLOOR]
        # Closest-to-pass borderlines first so the tolerated ones are the
        # judge's least-confident calls; the worst borderlines flag.
        borderlines = sorted(
            (a for a in all_audits if BORDERLINE_FLOOR <= a.support_score < THRESHOLD),
            key=lambda a: -a.support_score,
        )
        flagged = hard_fails + borderlines[BORDERLINE_TOLERANCE:]
        return FabricationAuditResult(per_bullet=per_bullet, flagged_bullets=flagged)
