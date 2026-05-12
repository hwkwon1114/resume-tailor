"""FabricationAudit — LLM-as-judge per-bullet support scoring.

Single batched call: emits a list of {bullet_id, support_score, reason} for
every output bullet so we don't pay per-bullet round-trips.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from harness.models import ModelClient
from harness.prompts import load_prompt
from harness.schema import Resume, iter_bullets

THRESHOLD = 0.6


class _BulletAudit(BaseModel):
    bullet_id: str
    support_score: float = Field(ge=0.0, le=1.0)
    reason: str


class _AuditReport(BaseModel):
    audits: list[_BulletAudit]


@dataclass(slots=True)
class FabricationAuditResult:
    per_bullet: dict[str, _BulletAudit]
    flagged_bullets: list[_BulletAudit] = field(default_factory=list)  # below THRESHOLD

    @property
    def pass_rate(self) -> float:
        if not self.per_bullet:
            return 1.0
        ok = sum(1 for a in self.per_bullet.values() if a.support_score >= THRESHOLD)
        return ok / len(self.per_bullet)


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
        prompt = (
            "Audit each output bullet against its cited input bullets. "
            "Return a JSON object {\"audits\": [{bullet_id, support_score, reason}, ...]} "
            "with EXACTLY one entry per output bullet.\n\n"
            f"OUTPUT BULLETS:\n{json.dumps(items, indent=2)}"
        )

        resp = model.generate_structured(system=system, prompt=prompt, schema=_AuditReport)
        report: _AuditReport = resp.data
        per_bullet = {a.bullet_id: a for a in report.audits}
        flagged = [a for a in report.audits if a.support_score < THRESHOLD]
        return FabricationAuditResult(per_bullet=per_bullet, flagged_bullets=flagged)
