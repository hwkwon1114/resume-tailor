<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# models

## Purpose
LLM client abstraction. Defines the `ModelClient` protocol the rest of the harness depends on, plus a stub for tests and a Gemini CLI subprocess implementation for production. Swap implementations by passing a different `ModelClient` instance to the orchestrator — nothing else in the harness should reference a specific model.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | `ModelClient` protocol, `ModelResponse` dataclass, `ModelOutputError` exception, and `EchoModelClient` (returns canned JSON for tests). |
| `gemini_subprocess.py` | `GeminiSubprocessClient` — shells out to the local `gemini` binary, captures stdout, surfaces non-zero exits as `ModelOutputError`. |
| `__init__.py` | Re-exports `EchoModelClient`, `ModelClient`, `ModelOutputError`, `ModelResponse`. |

## For AI Agents

### Working In This Directory
- The `ModelClient` protocol is the only contract the orchestrator should know about. New providers (e.g. an HTTP-API Gemini client, or a different LLM) go here as new classes implementing the protocol.
- `GeminiSubprocessClient` depends on `gemini` being on `PATH` and authenticated. There is no API key path; auth is the user's Google account via `gemini` CLI's browser flow.
- `EchoModelClient` is the test substrate. Tests construct it with a queue of canned responses; orchestrator pulls from it as if it were a real LLM.

### Testing Requirements
- A new client implementation should have a smoke test that asserts the protocol shape and at least one happy-path call. Live-LLM smoke tests should be gated behind `RUN_LIVE_EVALS=1` or similar.

### Common Patterns
- All clients return `ModelResponse(text=...)`. The orchestrator/generator is responsible for JSON-parsing the text.
- Subprocess errors and CLI quota errors are normalized to `ModelOutputError` so the orchestrator only needs to catch one exception type.

## Dependencies

### Internal
- Consumed by `harness/generate.py`, `harness/orchestrator.py`, `harness/pdf_intake.py`, and the judges in `harness/judges/`.

### External
- Standard library only (subprocess, dataclasses, typing) — no third-party SDKs in production. google-genai is a transitive dep used for type hints, not request flow.

<!-- MANUAL: -->
