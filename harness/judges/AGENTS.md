<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# judges

## Purpose
Quality judges that go beyond mechanical validation. Two kinds live here:
1. **LLM-backed semantic judges** (JD coverage judge, orphan fixer) — they call Gemini to evaluate or repair the resume.
2. **Mechanical stylistic judges** (voice check, fabrication audit) — pure Python, used for informational signals or sanity checks.

The split matters because LLM judges cost quota; the orchestrator gates them carefully.

## Key Files
| File | Description |
|------|-------------|
| `jd_coverage_judge.py` | `JDCoverageJudge` — calls Gemini to score 0–1 how well the output covers the JD and list uncovered requirements. Result feeds back into the next retry's prompt. |
| `orphan_fixer.py` | The big one. Detects bullets that spill 1–3 words onto a second line (`detect_orphans`, `detect_orphans_geometry`) and rewrites them via the LLM (`fix_orphans`, `fix_bullets`, `llm_clean_truncated`). Geometry-aware variant uses the rendered box tree. Only invoked on the final candidate. |
| `voice_check.py` | `VoiceCheck` — mechanical stylistic-drift signal (e.g. tense consistency, jargon density). Informational, not retry-triggering. |
| `fabrication_audit.py` | Lightweight cross-check that bullet content stays grounded in the input. Complements `SourceAttributionValidator`'s structural check with a content-level heuristic. |
| `jargon_words.txt` | Word list used by `voice_check.py` to flag jargon density. |

## For AI Agents

### Working In This Directory
- **Cost discipline.** Only `jd_coverage_judge.py` runs on every attempt. `orphan_fixer.py` runs **once**, on the final candidate, and **only** when orphans are detected. Don't move LLM judges into the per-attempt hot path without a quota plan.
- The orphan fixer has two detection modes: token-count heuristic (`detect_orphans`) and rendered-geometry-aware (`detect_orphans_geometry`). The geometry mode is more accurate but requires a real WeasyPrint render; the orchestrator chooses based on whether a render is already available.
- The orphan fixer's `llm_clean_truncated` path handles bullets that the LLM truncated mid-sentence — it must produce a clean ending (period, no dangling connectives). See the `_HANGING_CONNECTIVES` set in `harness/orchestrator.py`.
- Voice check and fabrication audit must not call the LLM — they're the fast, mechanical companions to the slower LLM judges.

### Testing Requirements
- LLM judges are tested with `EchoModelClient` stubs that return canned scoring JSON or canned rewrites.
- `tests/test_orphan_fixer.py` covers the detection logic (heuristic + geometry) thoroughly — extend it whenever you change orphan thresholds or rewrite policies.
- `tests/test_voice_check.py` and `tests/test_fabrication_audit.py` cover the mechanical judges.

### Common Patterns
- Judges return structured result dataclasses (`JDCoverageResult`, `VoiceCheckResult`, `OrphanCategory`) — never raw strings.
- The orphan fixer rewrites bullets but **preserves bullet ids and source_ids** — provenance survives the fix. Breaking this is a regression.

## Dependencies

### Internal
- `harness/models/` — all LLM judges accept a `ModelClient` injected from above.
- `harness/schema.py` — judges operate on `Resume` / `Bullet`.
- `render.pdf` — geometry-aware orphan detection needs a real render.

### External
- **weasyprint** (via `render.pdf`) — for the geometry-aware orphan detection path.

<!-- MANUAL: -->
