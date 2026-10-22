# Hybrid Gateway

Local metering proxy for Hermes Agent → llama-server / Telnyx Inference. See README.md for behaviour.

- Keep it small: `gateway.py` (proxy + CLI, PEP 723 inline deps, run with `uv run`), `dashboard.html` (no build), `prices.json`, `README.md`, `test_gateway.py`.
- Never store prompt/response text; never write or return `TELNYX_API_KEY`.
- Everything bound for Telnyx goes through `redact_body` in `forward()`. Log and print only redaction counts and kinds, never values; keep redaction deterministic so prompt caching holds. Test fakes are built in pieces (`fake_secrets()`) so no file holds a whole key.
- `/auto` router: fixed rules first, the router model only on new user turns; log only route, source and a fixed-list category, never the router's reasoning.
- Lint/type-check: `uvx ruff check --line-length 120 gateway.py test_gateway.py` and
  `uvx --with fastapi --with httpx --with tiktoken --with pathspec --with uvicorn --with keyring mypy --python-version 3.12 --ignore-missing-imports gateway.py test_gateway.py`.
- E2E: `uv run test_gateway.py` against a running proxy (cleans up its own rows).
- British English in UI copy and docs.
