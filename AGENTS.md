# Hybrid Gateway

Local metering proxy for Hermes Agent → llama-server / Telnyx Inference. See README.md for behaviour.

- Keep it small: `gateway.py` (proxy + CLI, PEP 723 inline deps, run with `uv run`), `dashboard.html` (no build), `prices.json`, `README.md`, `test_gateway.py`, `test_redaction.py`.
- Never store prompt/response text; never write or return `TELNYX_API_KEY`.
- Everything bound for Telnyx goes through `redact_body` in `forward()`: credentials and personal data (`iter_secrets`, `iter_pii`), every field but tool schemas. Non-JSON POST bodies are refused (fail closed), never forwarded unredacted. Credentials pin `/auto` to the Mac (`rule:secret`), and so does personal data (`rule:pii`). Log and print only redaction counts and kinds, never values; keep redaction deterministic so prompt caching holds. Test fakes are built in pieces (`fake_secrets()`) so no file holds a whole key.
- `/auto` router: fixed rules first, the router model only on new user turns; log only route, source and a fixed-list category, never the router's reasoning.
- Lint/type-check: `uvx ruff check --line-length 120 gateway.py test_gateway.py test_redaction.py` and
  `uvx --with fastapi --with httpx --with tiktoken --with pathspec --with uvicorn --with keyring --with pytest mypy --python-version 3.12 --ignore-missing-imports gateway.py test_gateway.py test_redaction.py`.
- E2E: `uv run test_gateway.py` against a running proxy (cleans up its own rows).
- Offline: `uv run test_redaction.py` (pytest) drives the real app with Telnyx, the Mac model and the router mocked through `httpx.MockTransport`; it asserts on the bytes Telnyx would receive, and on size routing (Mac / Flash / K3). Any change to `redact_body`, `forward()` or `decide()` needs a test there. PII fakes use reserved or published test values (Ofcom drama numbers, card test numbers, the example IBAN).
- CI (`.github/workflows/ci.yml`): lint, mypy, the offline `paths` and `secrets` tests and `test_redaction.py` on every push to `main` and every PR; keep its commands in step with the ones above.
- British English in UI copy and docs.
