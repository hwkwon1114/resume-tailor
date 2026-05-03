"""One-time converter: resume.json (custom schema) → resume_converted.json (harness schema).

Usage:
    .venv/bin/python convert_resume.py
    # writes resume_converted.json — upload via the JSON tab in the app
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "resume.json"
DST = Path(__file__).parent / "resume_converted.json"


def _parse_duration(duration: str) -> tuple[str, str | None]:
    """Return (start_date, end_date | None).

    Handles: "Aug 2022-Dec 2025", "Aug 2022 - May 2024",
             "Dec 2020-Current", "Jan 2025 May 2025"
    """
    s = duration.strip()
    # Try dash separator (with optional spaces)
    m = re.split(r"\s*[-–]\s*", s, maxsplit=1)
    if len(m) == 2:
        start, end = m[0].strip(), m[1].strip()
    else:
        # Space-only separator: "Jan 2025 May 2025"
        parts = s.split()
        if len(parts) == 4:
            start = f"{parts[0]} {parts[1]}"
            end = f"{parts[2]} {parts[3]}"
        else:
            return s, None

    end_clean = None if end.lower() in ("current", "present", "") else end
    return start, end_clean


def _bullets(responsibilities: list[str], section_id: str) -> list[dict]:
    return [
        {"id": f"{section_id}-b{i}", "text": r.strip(), "source_ids": []}
        for i, r in enumerate(responsibilities)
    ]


def convert(raw: dict) -> dict:
    ci = raw.get("contact_information", {})
    links = []
    if li := ci.get("linkedin"):
        links.append(li)
    if ws := ci.get("website"):
        links.append(ws)

    contact = {
        "name": ci.get("name", ""),
        "email": ci.get("email", ""),
        "phone": ci.get("phone", ""),
        "location": ci.get("location", ""),
        "links": links,
    }

    # Skills: flatten all competency lists
    cc = raw.get("core_competencies", {})
    skills: list[str] = []
    for lst in cc.values():
        skills.extend(lst)

    # Experience
    experience = []
    for i, job in enumerate(raw.get("professional_experience", [])):
        sec_id = f"exp-{i}"
        start, end = _parse_duration(job.get("duration", ""))
        experience.append({
            "id": sec_id,
            "employer": job.get("company", ""),
            "title": job.get("role", ""),
            "location": job.get("location", ""),
            "start_date": start,
            "end_date": end,
            "bullets": _bullets(job.get("responsibilities", []), sec_id),
        })

    # Projects
    projects = []
    for i, proj in enumerate(raw.get("technical_and_creative_projects", [])):
        sec_id = f"proj-{i}"
        desc = proj.get("role") or proj.get("technologies") or ""
        start, end = _parse_duration(proj.get("duration", "")) if proj.get("duration") else ("", None)
        projects.append({
            "id": sec_id,
            "name": proj.get("name", ""),
            "description": desc,
            "start_date": start,
            "end_date": end,
            "bullets": _bullets(proj.get("responsibilities", []), sec_id),
        })

    # Education
    education = []
    for i, ed in enumerate(raw.get("education", [])):
        sec_id = f"edu-{i}"
        start, end = _parse_duration(ed.get("duration", ""))
        # Any noteworthy lines as bullets (grant, notable coursework)
        edu_bullets = []
        if grant := ed.get("grant"):
            # Keep just the first sentence to avoid wall-of-text
            first_sent = grant.split("(")[0].strip().rstrip(".")
            edu_bullets.append({"id": f"{sec_id}-b0", "text": first_sent, "source_ids": []})
        education.append({
            "id": sec_id,
            "school": ed.get("institution", ""),
            "degree": ed.get("degree", ""),
            "field": "",
            "start_date": start,
            "end_date": end,
            "gpa": ed.get("gpa", ""),
            "bullets": edu_bullets,
        })

    return {
        "contact": contact,
        "summary": "",
        "experience": experience,
        "education": education,
        "projects": projects,
        "skills": skills,
    }


def _clean(text: str) -> str:
    """Strip [span_N\n...](start_span) / [span_N](end_span) viewer annotations."""
    text = re.sub(r"\[span_\d+\s*\]\(end_span\)", "", text)
    text = re.sub(r"\[span_\d+\s*\n\s*\]\(start_span\)", "", text)
    return text


def main() -> None:
    raw = json.loads(_clean(SRC.read_text()))
    out = convert(raw)
    DST.write_text(json.dumps(out, indent=2))
    print(f"Written to {DST}")
    print(f"  {len(out['experience'])} experience entries")
    print(f"  {len(out['projects'])} project entries")
    print(f"  {len(out['education'])} education entries")
    print(f"  {len(out['skills'])} skills")

    # Quick validation via harness schema
    try:
        from harness.schema import Resume, autopopulate_bullet_ids
        Resume.model_validate(autopopulate_bullet_ids(out))
        print("  Schema validation: OK")
    except Exception as exc:
        print(f"  Schema validation FAILED: {exc}")


if __name__ == "__main__":
    main()
