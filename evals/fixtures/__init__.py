"""Fixture loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from harness.schema import Resume, autopopulate_bullet_ids

_FIXTURE_DIR = Path(__file__).parent
_RESUME_DIR = _FIXTURE_DIR / "resume"
_JD_DIR = _FIXTURE_DIR / "jd"


@dataclass(slots=True)
class Fixture:
    name: str
    jd: str
    resume: Resume
    forbidden_terms: list[str] = field(default_factory=list)


def load_fixture(name: str, *, resume_filename: str = "me.json", forbidden_terms: list[str] | None = None) -> Fixture:
    jd_text = (_JD_DIR / f"{name}.txt").read_text(encoding="utf-8")
    raw = json.loads((_RESUME_DIR / resume_filename).read_text(encoding="utf-8"))
    resume = Resume.model_validate(autopopulate_bullet_ids(raw))
    return Fixture(name=name, jd=jd_text, resume=resume, forbidden_terms=forbidden_terms or [])


def all_baselines() -> list[Fixture]:
    return [load_fixture("swe-1"), load_fixture("ml-1"), load_fixture("mismatch-1")]


def temptation_fixture() -> Fixture:
    return load_fixture(
        "temptation-1",
        forbidden_terms=["rust", "webassembly", "wasm", "cap'n proto", "capnproto"],
    )
