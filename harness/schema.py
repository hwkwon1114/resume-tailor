"""Pydantic resume schema with provenance tracking.

Bullet.source_ids points back to input bullets the output bullet derives from,
enabling mechanical fabrication detection. NO field defaults are used on the
schema (google-genai issue #699 rejects schemas with defaults); callers
construct input bullets with explicit source_ids=[].
"""
from __future__ import annotations

from typing import Iterator

from pydantic import BaseModel, Field, model_validator


class Contact(BaseModel):
    name: str
    email: str
    phone: str | None = None
    location: str | None = None
    links: list[str] = Field(default_factory=list)


class Bullet(BaseModel):
    id: str
    text: str
    source_ids: list[str]


class ExperienceEntry(BaseModel):
    id: str
    employer: str
    title: str
    start_date: str
    end_date: str | None = None
    location: str | None = None
    bullets: list[Bullet]


class EducationEntry(BaseModel):
    id: str
    school: str
    degree: str
    start_date: str
    end_date: str | None = None
    gpa: str | None = None
    bullets: list[Bullet]


class Project(BaseModel):
    id: str
    name: str
    description: str
    bullets: list[Bullet]


class Resume(BaseModel):
    contact: Contact
    summary: str
    experience: list[ExperienceEntry]
    education: list[EducationEntry]
    projects: list[Project]
    skills: list[str]

    @model_validator(mode="after")
    def _enforce_unique_ids(self) -> "Resume":
        seen: set[str] = set()
        for bid in iter_bullet_ids(self):
            if bid in seen:
                raise ValueError(f"Duplicate bullet id: {bid!r}")
            seen.add(bid)
        section_ids = (
            [e.id for e in self.experience]
            + [e.id for e in self.education]
            + [p.id for p in self.projects]
        )
        if len(section_ids) != len(set(section_ids)):
            raise ValueError(f"Duplicate section id among {section_ids}")
        return self


def iter_bullet_ids(resume: Resume) -> Iterator[str]:
    """Yield every bullet id reachable in the resume (any section)."""
    for entry in resume.experience:
        for b in entry.bullets:
            yield b.id
    for edu in resume.education:
        for b in edu.bullets:
            yield b.id
    for proj in resume.projects:
        for b in proj.bullets:
            yield b.id


def iter_bullets(resume: Resume) -> Iterator[Bullet]:
    """Yield every Bullet object in the resume."""
    for entry in resume.experience:
        yield from entry.bullets
    for edu in resume.education:
        yield from edu.bullets
    for proj in resume.projects:
        yield from proj.bullets


def locked_fields(resume: Resume) -> dict[str, str]:
    """Return the byte-comparable field set that must never change.

    Keys are dotted paths like 'experience.exp-1.employer' so a diff identifies
    which field drifted.
    """
    out: dict[str, str] = {}
    for e in resume.experience:
        out[f"experience.{e.id}.employer"] = e.employer
        out[f"experience.{e.id}.start_date"] = e.start_date
        if e.end_date is not None:
            out[f"experience.{e.id}.end_date"] = e.end_date
    for ed in resume.education:
        out[f"education.{ed.id}.school"] = ed.school
        out[f"education.{ed.id}.degree"] = ed.degree
        out[f"education.{ed.id}.start_date"] = ed.start_date
        if ed.end_date is not None:
            out[f"education.{ed.id}.end_date"] = ed.end_date
        if ed.gpa is not None:
            out[f"education.{ed.id}.gpa"] = ed.gpa
    return out


def autopopulate_bullet_ids(raw: dict) -> dict:
    """Stamp deterministic ids onto a raw resume dict.

    Section ids: '{kind}-{idx}'.  Bullet ids: '{section_id}-b{bullet_idx}'.
    Existing ids are preserved. Mutates and returns the dict.
    """
    for kind, key in (("exp", "experience"), ("edu", "education"), ("proj", "projects")):
        for sec_idx, section in enumerate(raw.get(key, [])):
            section.setdefault("id", f"{kind}-{sec_idx}")
            for b_idx, bullet in enumerate(section.get("bullets", [])):
                bullet.setdefault("id", f"{section['id']}-b{b_idx}")
                bullet.setdefault("source_ids", [])
    return raw
