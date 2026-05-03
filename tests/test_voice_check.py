"""AC-13 VoiceCheck mechanical drift detection."""
from __future__ import annotations

from harness.judges.voice_check import VoiceCheck
from harness.schema import Bullet, Contact, ExperienceEntry, Resume


def _resume(input_text: str, output_text: str) -> tuple[Resume, Resume]:
    inp = Resume(
        contact=Contact(name="X", email="x@y.z"),
        summary="s",
        experience=[ExperienceEntry(
            id="exp-0", employer="E", title="T", start_date="2020-01", end_date="2021-01",
            bullets=[Bullet(id="exp-0-b0", text=input_text, source_ids=[])],
        )],
        education=[], projects=[], skills=[],
    )
    out = Resume(
        contact=Contact(name="X", email="x@y.z"),
        summary="s",
        experience=[ExperienceEntry(
            id="exp-0", employer="E", title="T", start_date="2020-01", end_date="2021-01",
            bullets=[Bullet(id="exp-0-out-0", text=output_text, source_ids=["exp-0-b0"])],
        )],
        education=[], projects=[], skills=[],
    )
    return inp, out


def test_low_drift_for_matching_voice():
    inp, out = _resume("Built a Python service for data ingestion.", "Built a Python tool for data ingestion.")
    res = VoiceCheck().run(input_resume=inp, output_resume=out)
    assert res.max_drift < 0.5


def test_high_drift_for_corporate_speak():
    inp, out = _resume(
        "Built a Python service for data ingestion.",
        "Spearheaded the leveraging of Python paradigms to evangelize a synergized data-ingestion ecosystem.",
    )
    res = VoiceCheck().run(input_resume=inp, output_resume=out)
    assert res.max_drift > 0.2, f"drift={res.max_drift}"
