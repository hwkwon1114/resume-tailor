"""Resume intake: PDF bytes or raw pasted text → validated Resume.

Two paths:
  extract_text_from_pdf(bytes)  — pdfplumber page-by-page text
  text_to_resume(text, model)   — LLM extraction into Resume schema
"""
from __future__ import annotations

import io

import pdfplumber
from pydantic import BaseModel, Field, model_validator

from harness.models import ModelClient
from harness.schema import Resume, autopopulate_bullet_ids

_SYSTEM = """\
You are a resume parser. Extract the structured information from the resume text below and output it as a
Resume JSON object. Follow these rules exactly:

1. contact: extract name, email, phone, location (city/state or "Remote"), linkedin URL if present.
2. summary: copy the professional summary verbatim if present; otherwise write a single neutral sentence.
3. experience: one entry per job. employer, title, location, start_date, end_date (or "Present").
   bullets: each distinct accomplishment as a separate bullet. text must start with a strong action verb.
   Set id="" and source_ids=[] — they will be auto-populated.
4. education: one entry per degree. institution, degree, field, start_date, end_date, gpa if shown.
   bullets: any honors, thesis, or relevant coursework lines.
5. projects: any standalone project section. name, description, start_date, end_date.
   bullets: key accomplishment lines.
6. skills: flat list of skill strings exactly as written.

Do not invent, infer, or embellish anything that is not in the source text.
"""

_PROMPT_TEMPLATE = "RESUME TEXT:\n{text}"


class _ResumeEnvelope(BaseModel):
    """Thin envelope so we can ask for structured output without nested schema issues."""
    resume: Resume = Field(description="The fully extracted Resume object.")

    @model_validator(mode="before")
    @classmethod
    def _populate_ids(cls, data: dict) -> dict:
        if isinstance(data, dict) and "resume" in data and isinstance(data["resume"], dict):
            raw = data["resume"]
            # setdefault only fills missing keys, not "". Clear empty-string ids first
            # so autopopulate_bullet_ids can assign unique deterministic ones.
            for key in ("experience", "education", "projects"):
                for section in raw.get(key, []):
                    if section.get("id") == "":
                        del section["id"]
                    for bullet in section.get("bullets", []):
                        if bullet.get("id") == "":
                            del bullet["id"]
            data["resume"] = autopopulate_bullet_ids(raw)
        return data


def extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def text_to_resume(text: str, model: ModelClient) -> Resume:
    """Convert raw resume text (pasted or extracted from PDF) to a validated Resume."""
    if not text.strip():
        raise ValueError("Resume text is empty.")

    resp = model.generate_structured(
        system=_SYSTEM,
        prompt=_PROMPT_TEMPLATE.format(text=text.strip()),
        schema=_ResumeEnvelope,
    )
    return resp.data.resume
