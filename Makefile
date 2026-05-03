# macOS needs DYLD_FALLBACK_LIBRARY_PATH so WeasyPrint can find Homebrew GLib/Pango.
export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib

PY := .venv/bin/python
PYTEST := .venv/bin/pytest
STREAMLIT := .venv/bin/streamlit

.PHONY: app test evals evals-live clear-cache

app:
	$(STREAMLIT) run app.py

test:
	$(PYTEST) -q

evals:
	$(PY) evals/run_evals.py

evals-live:
	RUN_LIVE_EVALS=1 $(PY) evals/run_evals.py

clear-cache:
	$(PY) evals/cache.py --clear
