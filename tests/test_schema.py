"""Schema invariants and helpers."""
from __future__ import annotations

from copy import deepcopy

import pytest

from harness.schema import Resume, autopopulate_bullet_ids, iter_bullet_ids, locked_fields


_MIN_RAW = {
    "contact": {"name": "X", "email": "x@y.z"},
    "summary": "S",
    "experience": [{"employer": "E", "title": "T", "start_date": "2020-01", "end_date": "2021-01",
                    "bullets": [{"text": "did a thing"}, {"text": "did another"}]}],
    "education": [{"school": "U", "degree": "BS", "start_date": "2016-01", "end_date": "2020-01"}],
    "projects": [],
    "skills": ["Python"],
}


def test_autopopulate_assigns_ids():
    raw = autopopulate_bullet_ids({**_MIN_RAW, "experience": list(_MIN_RAW["experience"])})
    raw["experience"][0]["bullets"][0].setdefault("source_ids", [])
    raw["experience"][0]["bullets"][1].setdefault("source_ids", [])
    raw["education"][0]["bullets"] = []
    r = Resume.model_validate(raw)
    ids = list(iter_bullet_ids(r))
    assert ids == ["exp-0-b0", "exp-0-b1"]


def test_duplicate_bullet_ids_rejected():
    raw = autopopulate_bullet_ids(deepcopy(_MIN_RAW))
    raw["experience"][0]["bullets"] = [
        {"id": "dup", "text": "a", "source_ids": []},
        {"id": "dup", "text": "b", "source_ids": []},
    ]
    raw["education"][0]["bullets"] = []
    with pytest.raises(Exception):
        Resume.model_validate(raw)


def test_locked_fields_keys():
    raw = autopopulate_bullet_ids(deepcopy(_MIN_RAW))
    raw["education"][0]["bullets"] = []
    r = Resume.model_validate(raw)
    locked = locked_fields(r)
    assert "experience.exp-0.employer" in locked
    assert locked["experience.exp-0.employer"] == "E"
    assert "education.edu-0.school" in locked
    assert "education.edu-0.degree" in locked
