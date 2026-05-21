"""AC-5 fabrication-audit threshold + pruning behavior (uses EchoModelClient)."""
from __future__ import annotations

from harness.judges.fabrication_audit import (
    BATCH_SIZE,
    _AuditReport,
    FabricationAudit,
    THRESHOLD,
)
from harness.models import EchoModelClient, ModelResponse
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


def test_batches_audit_by_chunk_size_and_merges_results():
    """The audit splits >BATCH_SIZE bullets into parallel LLM calls, then merges.

    Bench evidence (2026-05-15 AWS run): a single fab-audit call on 13-15
    bullets consistently hit the 120s Gemini CLI timeout wall. Splitting
    into chunks of BATCH_SIZE keeps each call's payload small enough to
    return well under the wall. Contract: with N bullets and chunk size B,
    expect ceil(N/B) model calls; merged per_bullet dict equals what a
    single-shot audit would have produced (modulo call ordering).
    """
    n_bullets = BATCH_SIZE * 2 + 1  # forces 3 chunks
    out = _resume_with_bullets([f"text {i}" for i in range(n_bullets)])

    call_log: list[list[str]] = []

    class _ChunkRecordingModel:
        def generate_structured(self, *, system, prompt, schema):
            import re
            # The prompt embeds the JSON items as `"bullet_id": "exp-0-out-N"`.
            ids = re.findall(r'"bullet_id":\s*"([^"]+)"', prompt)
            call_log.append(ids)
            audits = [{"bullet_id": bid, "support_score": 0.9, "reason": "ok"} for bid in ids]
            return ModelResponse(
                data=_AuditReport.model_validate({"audits": audits}),
                model_name="chunk",
            )
        def generate_text(self, **kw):
            return ModelResponse(data="", model_name="chunk")

    model = _ChunkRecordingModel()
    result = FabricationAudit().run(input_resume=out, output_resume=out, model=model)

    # 3 chunks expected: BATCH_SIZE + BATCH_SIZE + 1.
    assert len(call_log) == 3, f"expected 3 chunks, got {len(call_log)}: {call_log}"
    assert [len(c) for c in call_log] == [BATCH_SIZE, BATCH_SIZE, 1]
    # No bullet was audited twice, none was skipped.
    flat = [bid for chunk in call_log for bid in chunk]
    assert sorted(flat) == sorted(b.id for b in out.experience[0].bullets)
    # Merged result spans every bullet.
    assert len(result.per_bullet) == n_bullets
    assert not result.flagged_bullets  # all 0.9


def test_audit_flags_bullets_from_any_chunk():
    """A bullet flagged in chunk 1 must appear in the merged flagged_bullets list."""
    n_bullets = BATCH_SIZE * 2  # two chunks
    out = _resume_with_bullets([f"text {i}" for i in range(n_bullets)])

    # Chunk-aware mock: flag the LAST bullet in each chunk to test cross-chunk merge.
    class _MixedScoreModel:
        def generate_structured(self, *, system, prompt, schema):
            import re
            ids = re.findall(r'"bullet_id":\s*"([^"]+)"', prompt)
            audits = []
            for i, bid in enumerate(ids):
                # Last bullet of each chunk is fabricated (0.2); others clean (0.9).
                score = 0.2 if i == len(ids) - 1 else 0.9
                audits.append({"bullet_id": bid, "support_score": score, "reason": "x"})
            return ModelResponse(
                data=_AuditReport.model_validate({"audits": audits}),
                model_name="mixed",
            )
        def generate_text(self, **kw):
            return ModelResponse(data="", model_name="mixed")

    result = FabricationAudit().run(input_resume=out, output_resume=out, model=_MixedScoreModel())
    assert len(result.flagged_bullets) == 2, (
        "one flagged bullet from each of two chunks must survive the merge"
    )
