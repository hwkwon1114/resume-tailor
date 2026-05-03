"""AC-5 fabrication-audit threshold + pruning behavior (uses EchoModelClient)."""
from __future__ import annotations

from harness.judges.fabrication_audit import _AuditReport, FabricationAudit, THRESHOLD
from harness.models import EchoModelClient
from harness.schema import Bullet, Contact, ExperienceEntry, Resume


def _resume_with_bullets(bullet_texts):
    bullets = [Bullet(id=f"exp-0-out-{i}", text=t, source_ids=["exp-0-b0"]) for i, t in enumerate(bullet_texts)]
    return Resume(
        contact=Contact(name="X", email="x@y.z"),
        summary="s",
        experience=[ExperienceEntry(
            id="exp-0", employer="E", title="T", start_date="2020-01", end_date="2021-01",
            bullets=bullets,
        )],
        education=[], projects=[], skills=[],
    )


def test_threshold_constant_is_06():
    assert THRESHOLD == 0.6


def test_calibration_5_bullets_threshold_split():
    """5 bullets at known support levels (0.9/0.7/0.6/0.5/0.3); only those < 0.6 are flagged."""
    out = _resume_with_bullets(["a", "b", "c", "d", "e"])
    audit = _AuditReport.model_validate({"audits": [
        {"bullet_id": "exp-0-out-0", "support_score": 0.9, "reason": "supported"},
        {"bullet_id": "exp-0-out-1", "support_score": 0.7, "reason": "supported"},
        {"bullet_id": "exp-0-out-2", "support_score": 0.6, "reason": "borderline"},
        {"bullet_id": "exp-0-out-3", "support_score": 0.5, "reason": "weak"},
        {"bullet_id": "exp-0-out-4", "support_score": 0.3, "reason": "fabricated"},
    ]})
    model = EchoModelClient(structured_response=audit)
    result = FabricationAudit().run(input_resume=out, output_resume=out, model=model)
    flagged_ids = sorted(a.bullet_id for a in result.flagged_bullets)
    assert flagged_ids == ["exp-0-out-3", "exp-0-out-4"]
