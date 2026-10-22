# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["httpx>=0.27", "pathspec>=0.12", "tiktoken>=0.8"]  # the last two for gateway pack
# ///
"""End-to-end check against a running Hybrid Gateway.

For each backend: list models, send one non-streaming and one streaming chat
completion, then confirm matching rows landed in gateway.db with token counts.
For /auto: one routed request, then dry runs of the tool-loop, secret and context rules.
For secrets: detection, redaction, scan and pack on fake values in temp folders (no proxy needed).
For settings: the API shape, a round trip that is restored afterwards, and the error
cases. For paths: relative startup paths resolve against this folder from any cwd
(no proxy needed). Secrets are never changed. Rows created by this script are deleted afterwards
unless --keep is passed.

    uv run test_gateway.py                       # paths, settings, local, Telnyx and auto
    uv run test_gateway.py --only local          # skip the rest
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
PROMPT = [{"role": "user", "content": "Reply with the single word: ok"}]
SETTING_KEYS = [
    "local_url", "local_api_key", "local_model", "router_model", "router_timeout", "local_context",
    "remote_url", "remote_api_key", "flash_model", "k3_model", "flash_context", "gateway_token", "read_timeout",
]


def check(cond: bool, msg: str, failures: list[str]) -> None:
    """Print a pass/fail line and collect failures."""
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def run_backend(client: httpx.Client, base: str, backend: str, model: str | None, headers: dict[str, str],
                db: sqlite3.Connection, failures: list[str]) -> None:
    """Exercise one backend through the proxy and verify its log rows."""
    print(f"\n[{backend}]")
    before = db.execute("SELECT COALESCE(MAX(id), 0) FROM requests").fetchone()[0]

    r = client.get(f"{base}/{backend}/v1/models", headers=headers)
    ids = [m.get("id") for m in r.json().get("data", [])] if r.status_code == 200 else []
    check(r.status_code == 200 and bool(ids), f"GET /v1/models -> {r.status_code}, {len(ids)} models", failures)
    model = model or (ids[0] if ids else None)
    if not model:
        check(False, "no model available to test with", failures)
        return
    print(f"  using model {model}")

    body = {"model": model, "messages": PROMPT, "max_tokens": 16}
    r = client.post(f"{base}/{backend}/v1/chat/completions", headers=headers, json=body)
    check(r.status_code == 200 and "choices" in r.json(), f"non-streaming completion -> {r.status_code}", failures)

    events: list[dict[str, object]] = []
    with client.stream("POST", f"{base}/{backend}/v1/chat/completions", headers=headers,
                       json={**body, "stream": True}) as s:
        ok = s.status_code == 200 and s.headers.get("content-type", "").startswith("text/event-stream")
        for line in s.iter_lines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                events.append(json.loads(line[5:]))
    check(ok and len(events) > 1, f"streaming completion -> {s.status_code}, {len(events)} SSE events", failures)
    check(any(e.get("usage") for e in events), "stream carried a usage chunk (include_usage injected)", failures)

    rows = db.execute(
        "SELECT * FROM requests WHERE id > ? AND backend = ? ORDER BY id", (before, backend)
    ).fetchall()
    check(len(rows) == 2, f"{len(rows)} rows logged (expected 2)", failures)
    for row in rows:
        check(row["prompt_tokens"] > 0 and row["completion_tokens"] > 0 and row["status"] == 200,
              f"row {row['id']}: in={row['prompt_tokens']} cached={row['cached_tokens']} "
              f"out={row['completion_tokens']} {row['wall_ms']}ms status={row['status']} "
              f"source={row['token_source']} job={row['job_label']}", failures)
    cols = {c[1] for c in db.execute("PRAGMA table_info(requests)")}
    check(not cols & {"prompt", "response", "messages", "content"}, "no prompt/response text columns", failures)


def run_auto(client: httpx.Client, base: str, token: str, db: sqlite3.Connection, failures: list[str]) -> None:
    """Exercise /auto: model listing, a routed request, tool-loop stickiness and the secret and context rules."""
    print("\n[auto]")
    router = client.get(f"{base}/api/summary").json().get("router") or {}
    if not router.get("enabled"):
        print("  SKIP no Mac model (local_model) is set on the proxy")
        return
    auth = {"authorization": f"Bearer {token}"}
    before = db.execute("SELECT COALESCE(MAX(id), 0) FROM requests").fetchone()[0]

    r = client.get(f"{base}/auto/v1/models", headers=auth)
    if r.status_code == 503:
        print(f"  SKIP /auto is not configured: {r.json().get('error')}")
        return
    check(r.status_code == 200 and r.json()["data"][0]["id"] == "auto", f"GET /auto/v1/models -> {r.status_code}",
          failures)
    r = client.get(f"{base}/auto/v1/models", headers={"authorization": "Bearer wrong"})
    check(r.status_code == 401, f"wrong token rejected -> {r.status_code}", failures)

    messages: list[dict[str, object]] = [
        {"role": "system", "content": f"Hybrid Gateway self-test {time.time()}"},
        {"role": "user", "content": "Reply with the single word: ok"},
    ]
    r = client.post(f"{base}/auto/v1/chat/completions", headers=auth,
                    json={"model": "auto", "messages": messages, "max_tokens": 16})
    check(r.status_code == 200 and "choices" in r.json(), f"routed completion -> {r.status_code}", failures)
    work = db.execute("SELECT * FROM requests WHERE id > ? AND purpose = 'work' ORDER BY id DESC LIMIT 1",
                      (before,)).fetchone()
    calls = db.execute("SELECT * FROM requests WHERE id > ? AND purpose = 'router'", (before,)).fetchall()
    check(work is not None and work["route"] in ("mac", "flash", "k3"),
          f"work row logged: route={work and work['route']} source={work and work['route_source']} "
          f"category={work and work['route_category']} model={work and work['model']}", failures)
    check(bool(work) and work["route_source"] == "router", "decided by the router model (not a fallback)", failures)
    check(len(calls) == 1 and calls[0]["status"] == 200,
          f"router call logged: {len(calls)} row(s), {calls[0]['wall_ms'] if calls else '-'} ms", failures)

    # Mid tool loop: the next request must stay on the same route without asking the router.
    loop = [*messages,
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"}]
    d = client.post(f"{base}/api/route", json={"messages": loop}).json()
    check(d["source"] == "rule:tool-loop" and work is not None and d["route"] == work["route"],
          f"tool result stays on {d['route']} ({d['source']})", failures)

    # A credential in the conversation keeps it on the Mac, without asking the router (dry run).
    leak = "Why won't my app connect? DATABASE_URL=postgres://admin:not-a-real-pass@10.0.0.1:5432/app"
    d = client.post(f"{base}/api/route", json={"messages": [{"role": "user", "content": leak}]}).json()
    check(d["route"] == "mac" and d["source"] == "rule:secret", f"credential stays on {d['route']} ({d['source']})",
          failures)

    # Reading a secrets file keeps the conversation on the Mac, whatever the file holds (dry run).
    env_read = [{"role": "user", "content": "What port does the app use?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_env", "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"path": "app/.env.local"})}}]},
                {"role": "tool", "tool_call_id": "call_env", "content": "PORT=3000"}]
    d = client.post(f"{base}/api/route", json={"messages": env_read}).json()
    check(d["route"] == "mac" and d["source"] == "rule:secret", f".env.local tool result stays on {d['route']} "
          f"({d['source']})", failures)

    # Bigger than the local window: must not run on the Mac (dry run, so nothing is billed).
    big = "word " * (router["local_context"] + 2000)
    d = client.post(f"{base}/api/route", json={"messages": [{"role": "user", "content": big}]}).json()
    check(d["route"] != "mac", f"{router['local_context']:,}+ token prompt goes to {d['route']} ({d['source']})",
          failures)


def run_settings(client: httpx.Client, base: str, failures: list[str]) -> None:
    """Check the settings API without touching secrets; any value changed is put back."""
    print("\n[settings]")
    r = client.get(f"{base}/api/settings")
    data = r.json() if r.status_code == 200 else {}
    fields = {f["key"]: f for f in data.get("fields", [])}
    check(r.status_code == 200 and [f["key"] for f in data.get("fields", [])] == SETTING_KEYS,
          f"GET /api/settings -> {r.status_code}, {len(fields)} fields in contract order", failures)
    secrets = [f for f in fields.values() if f["type"] == "secret"]
    check(len(secrets) == 3 and all(f["value"] is None and f["env_value"] is None and isinstance(f.get("set"), bool)
                                    and isinstance(f.get("env_set"), bool) for f in secrets),
          "secrets have value and env_value null, and set and env_set flags", failures)
    check(all(f["env_origin"] in ("shell", "file", None) and (f["env_origin"] is None) == (f["env_set_by"] is None)
              for f in fields.values()), "env_origin matches env_set_by on every field", failures)
    startup = data.get("startup") or {}
    check(isinstance(data.get("keychain"), bool) and {"flash", "k3"} <= set(data.get("prices") or {})
          and {"port", "env_file", "env_file_loaded", "keychain_service"} <= set(startup),
          "keychain, prices and startup (with env file) present", failures)
    token = os.environ.get("HYBRID_GATEWAY_TOKEN")
    check(not token or token not in r.text, "the token never appears in GET /api/settings", failures)

    field = fields.get("router_timeout")
    if not field:
        check(False, "router_timeout is missing", failures)
    else:
        original = field["value"] if field["source"] == "app" else None
        trial = 9.5 if field["value"] != 9.5 else 10.5
        r = client.post(f"{base}/api/settings", json={"values": {"router_timeout": trial}})
        after = {f["key"]: f for f in r.json().get("fields", [])} if r.status_code == 200 else {}
        check(after.get("router_timeout", {}).get("value") == trial
              and after["router_timeout"]["source"] == "app", f"POST router_timeout={trial} -> {r.status_code}",
              failures)
        r = client.post(f"{base}/api/settings", json={"values": {"router_timeout": original}})
        after = {f["key"]: f for f in r.json().get("fields", [])} if r.status_code == 200 else {}
        check(r.status_code == 200 and after["router_timeout"]["value"] == field["value"]
              and after["router_timeout"]["source"] == field["source"],
              f"router_timeout restored to {field['value']} ({field['source']})", failures)

    r = client.post(f"{base}/api/settings", json={"values": {"no_such_setting": 1}})
    check(r.status_code == 422 and r.json().get("field") == "no_such_setting",
          f"unknown key rejected -> {r.status_code}, field={r.json().get('field')}", failures)
    numeric = next((f["key"] for f in fields.values() if f["type"] in ("seconds", "tokens")), None)
    if numeric:  # 0 is below every numeric field's minimum
        r = client.post(f"{base}/api/settings", json={"values": {numeric: 0}})
        check(r.status_code == 422 and r.json().get("field") == numeric,
              f"out-of-range {numeric} rejected -> {r.status_code}, field={r.json().get('field')}", failures)
    free = [f["key"] for f in fields.values() if f["type"] == "url"]
    if len(free) == 2:  # both URLs bad: the first is named, both are described, nothing is saved
        r = client.post(f"{base}/api/settings", json={"values": {free[0]: "ftp://x", free[1]: "nope"}})
        err = r.json().get("error", "")
        check(r.status_code == 422 and r.json().get("field") == free[0] and all(k in err for k in free)
              and "ftp://x" not in err, f"two bad URLs -> {r.status_code}, field={r.json().get('field')}", failures)
    r = client.post(f"{base}/api/settings", json={"prices": {"flash": {"input": -1}}})
    check(r.status_code == 422 and r.json().get("field") == "prices.flash.input",
          f"negative price rejected -> {r.status_code}, field={r.json().get('field')}", failures)
    r = client.post(f"{base}/api/settings", content=b"{}", headers={"content-type": "text/plain"})
    check(r.status_code == 415, f"non-JSON content type rejected -> {r.status_code}", failures)

    # A saved value overrides env; clearing it falls back to env. Non-secrets only, least disruptive type first.
    by_env = sorted((f for f in fields.values() if f["source"] == "env" and f["type"] != "secret"),
                    key=lambda f: ["seconds", "tokens", "model", "url"].index(f["type"]))
    if not by_env:
        print("  SKIP drawer-over-env (no non-secret field is set by env)")
    else:
        over = by_env[0]
        trials: dict[str, list[str | int]] = {
            "seconds": [30, 31], "tokens": [50_000, 50_001], "model": ["hybrid-gateway-selftest", "hybrid-gateway-x"],
            "url": ["http://127.0.0.1:9/v1", "http://127.0.0.1:8/v1"]}
        pick = next(t for t in trials[over["type"]] if t != over["env_value"])
        name = f"{over['key']} (env {over['env_set_by']})"
        r = client.post(f"{base}/api/settings", json={"values": {over["key"]: pick}})
        after = {f["key"]: f for f in r.json().get("fields", [])} if r.status_code == 200 else {}
        got = after.get(over["key"], {})
        check(r.status_code == 200 and got.get("source") == "app" and got.get("value") == pick
              and got.get("env_set_by") == over["env_set_by"], f"{name}: saved value overrides env -> "
              f"{r.status_code}, source={got.get('source')}", failures)
        r = client.post(f"{base}/api/settings", json={"values": {over["key"]: None}})
        after = {f["key"]: f for f in r.json().get("fields", [])} if r.status_code == 200 else {}
        got = after.get(over["key"], {})
        check(r.status_code == 200 and got.get("source") == "env" and got.get("value") == over["env_value"],
              f"{name}: cleared falls back to env -> {r.status_code}, source={got.get('source')}", failures)

    r = client.get(f"{base}/api/settings/models", params={"backend": "local"})
    m = r.json() if r.status_code == 200 else {}
    ok = isinstance(m.get("ok"), bool) and isinstance(m.get("models"), list) \
        and (m["error"] is None if m.get("ok") else isinstance(m.get("error"), str))
    check(r.status_code == 200 and ok, f"GET /api/settings/models?backend=local -> ok={m.get('ok')}, "
          f"{len(m.get('models') or [])} models{'' if m.get('ok') else f' ({m.get("error")})'}", failures)


def fake_secrets() -> dict[str, str]:
    """Fake credentials in real formats, built in pieces so this file never holds one whole."""
    return {
        "telnyx-key": "KEY0" + "19A3F7C2" * 3 + "_" + "q8Wm2Lx9Zp4R",
        "aws-key": "AKIA" + "Q7XW" * 4,
        "anthropic-key": "sk-ant-" + "api03-" + "x7Kq" * 6,
        "stripe-key": "sk_" + "live_" + "a1B2c3D4" * 3,
        "github-token": "ghp_" + "Zx9Yw8Vu7Ts6" * 2,
        "resend-key": "re_" + "Ab3dE6gH" + "_" + "x9Y8z7W6v5U4t3S2r1Q0",
        "credential": "q9" + "Lm2Xz7Pw4Kd8",
    }


def run_secrets(failures: list[str]) -> None:
    """Secret detection, redaction, `gateway scan` and `gateway pack`, on fake values in temp folders. Needs no proxy."""
    import contextlib
    import io
    import tempfile
    from argparse import Namespace

    print("\n[secrets]")
    sys.path.insert(0, str(HERE))
    sys.dont_write_bytecode = True
    import gateway

    fakes = fake_secrets()
    lines = {kind: (f"export API_SECRET={v}" if kind == "credential" else f"key = {v}") for kind, v in fakes.items()}
    for kind, line in lines.items():
        text, kinds = gateway.redact_text(line)
        check(fakes[kind] not in text and kinds == [kind], f"{kind} redacted -> {text}", failures)
    for line in ['"api_key": "' + fakes["credential"] + '"', "password: hunter2hunter2",
                 "DATABASE_URL=postgres://app:" + "s3cret" + "@db:5432/app"]:
        check(bool(gateway.find_secrets(line)), f"found in {gateway.redact_text(line)[0]}", failures)
    for line in ["max_tokens: 16", "api_key: str | None = field(repr=False)", "API_KEY=your-api-key-here",
                 "secret = has_secret(body)", "apiKey: process.env.API_KEY", 'token_url = "https://auth.x/token"']:
        check(not gateway.find_secrets(line), f"not a credential: {line}", failures)

    def call(cid: str, path: str) -> dict[str, object]:
        return {"id": cid, "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}

    body: dict[str, object] = {
        "model": "m", "tools": [{"type": "function", "function": {"name": "read_file", "description": "token: "
                                                                  + fakes["credential"]}}],
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Deploy with " + fakes["aws-key"]},
            {"role": "assistant", "content": None, "tool_calls": [
                call("c1", ".env"), call("c2", ".env.example"),
                {"id": "c3", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(
                    {"path": "deploy.sh", "content": "export TELNYX_API_KEY=" + fakes["telnyx-key"]})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "PORT=3000"},
            {"role": "tool", "tool_call_id": "c2", "content": "PORT=3000"},
            {"role": "tool", "tool_call_id": "c3", "content": "ok"},
        ],
    }
    check(gateway.has_secret(body), "has_secret: credential or secrets file in the conversation", failures)
    check(not gateway.has_secret({"messages": [body["messages"][0], {"role": "user", "content": "hi"},  # type: ignore
                                             {"role": "assistant", "tool_calls": [call("c2", ".env.example")]},
                                             {"role": "tool", "tool_call_id": "c2", "content": "PORT=3000"}]}),
          "has_secret: .env.example is not a secrets file", failures)
    first = json.loads(json.dumps(body))
    kinds = gateway.redact_body(first)
    second = json.loads(json.dumps(body))
    gateway.redact_body(second)
    msgs = first["messages"]
    sent = json.dumps(msgs)
    check(fakes["aws-key"] not in sent and fakes["telnyx-key"] not in sent
          and sorted(kinds) == ["aws-key", "secrets-file", "telnyx-key"], f"redact_body replaced {kinds}", failures)
    check("withheld" in msgs[3]["content"] and msgs[4]["content"] == "PORT=3000",
          "the .env tool result is withheld; the .env.example one is kept", failures)
    args = json.loads(msgs[2]["tool_calls"][2]["function"]["arguments"])
    check(args["content"] == "export TELNYX_API_KEY=[redacted:telnyx-key]", "tool call arguments redacted, still JSON",
          failures)
    check(gateway.REDACTION_NOTE in msgs[0]["content"] and first["tools"] == body["tools"],
          "placeholder note added to the system prompt; tool schemas untouched", failures)
    check(first == second, "redaction is deterministic (prompt cache safe)", failures)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        (root / "config").mkdir(parents=True)
        (root / ".env").write_text("PORT=3000\n")
        (root / ".env.example").write_text("TELNYX_API_KEY=\n")
        (root / "config" / "app.json").write_text('{\n  "name": "app",\n  "apiKey": "' + fakes["credential"] + '"\n}\n')
        (root / "main.py").write_text("print('hello')\n")
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            code = gateway.cmd_scan(Namespace(paths=[str(root)]))
        report = out_buf.getvalue()
        check(code == 1 and ".env  secrets file" in report and "config/app.json:3  credential" in report
              and ".env.example" not in report and "main.py" not in report, f"scan reports {report.splitlines()}",
              failures)
        check(fakes["credential"] not in report + err_buf.getvalue(), "scan never prints a value", failures)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            clean = gateway.cmd_scan(Namespace(paths=[str(root / "main.py")]))
        check(clean == 0, f"clean file scans clean -> exit {clean}", failures)

        packed = Path(tmp) / "out.packed.txt"
        with contextlib.redirect_stdout(io.StringIO()):
            gateway.cmd_pack(Namespace(path=str(root), output=str(packed), limit=1_000_000, no_redact=False))
        text = packed.read_text()
        check("===== .env =====" not in text and "===== .env.example =====" in text
              and fakes["credential"] not in text and "[redacted:credential]" in text,
              "pack skips .env and redacts config/app.json", failures)


def run_redaction(client: httpx.Client, base: str, model: str, token: str, db: sqlite3.Connection,
                  failures: list[str]) -> None:
    """A credential sent to /telnyx is replaced before it leaves the machine, and the row counts it."""
    print("\n[telnyx redaction]")
    before = db.execute("SELECT COALESCE(MAX(id), 0) FROM requests").fetchone()[0]
    fake = fake_secrets()["aws-key"]
    r = client.post(f"{base}/telnyx/v1/chat/completions", headers={"authorization": f"Bearer {token}"}, json={
        "model": model, "max_tokens": 40,
        "messages": [{"role": "user", "content": f"Repeat this exactly and nothing else: {fake}"}]})
    reply = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "" if r.is_success else ""
    check(r.status_code == 200 and fake not in reply, f"reply never holds the key -> {r.status_code} {reply!r}",
          failures)
    row = db.execute("SELECT redactions FROM requests WHERE id > ? AND backend = 'telnyx' ORDER BY id DESC LIMIT 1",
                     (before,)).fetchone()
    check(row is not None and row["redactions"] == 1, f"row logged {row and row['redactions']} redaction(s)", failures)


def run_paths(failures: list[str]) -> None:
    """Relative startup paths resolve against the folder holding gateway.py, whatever the cwd. Needs no proxy."""
    import tempfile

    print("\n[paths]")
    sys.path.insert(0, str(HERE))
    sys.dont_write_bytecode = True  # leave no __pycache__ in the project
    import gateway

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        absolute = str(Path(tmp).resolve() / "abs-settings.json")
        environ = {"HOME": os.environ.get("HOME", ""), "HYBRID_GATEWAY_DB": "data/test.db",
                   "HYBRID_GATEWAY_PRICES": "~/selftest-prices.json", "HYBRID_GATEWAY_SETTINGS": absolute,
                   "HYBRID_GATEWAY_ENV_FILE": "selftest-missing.env"}
        try:
            os.chdir(tmp)  # nothing is created: Startup only resolves paths
            env = gateway.load_env(environ, quiet=True)
            s = gateway.Startup.from_env(env)
        finally:
            os.chdir(cwd)
    check(s.db_path == HERE / "data" / "test.db", f"HYBRID_GATEWAY_DB=data/test.db from another cwd -> {s.db_path}",
          failures)
    check(s.prices_path == Path.home() / "selftest-prices.json", f"~ expanded -> {s.prices_path}", failures)
    check(str(s.settings_path) == absolute, "absolute HYBRID_GATEWAY_SETTINGS kept", failures)
    check(s.env_file == HERE / "selftest-missing.env" and not s.env_file_loaded,
          f"relative HYBRID_GATEWAY_ENV_FILE -> {s.env_file} (not found)", failures)
    # Defaults with no env vars, built without reading the real .env.
    d = gateway.Startup.from_env(gateway.Env({}, {}, gateway.env_file_path({}), False))
    check(d.env_file == HERE / ".env" and d.settings_path == HERE / "settings.json"
          and d.db_path == HERE / "gateway.db" and d.prices_path == HERE / "prices.json",
          "no env vars: .env, settings.json, gateway.db and prices.json all next to gateway.py", failures)


def proxy_token(client: httpx.Client, base: str) -> str | None:
    """HYBRID_GATEWAY_TOKEN from this shell, else the proxy's own (revealed, never changed)."""
    if token := os.environ.get("HYBRID_GATEWAY_TOKEN"):
        return token
    r = client.post(f"{base}/api/settings/token", json={})
    return r.json().get("token") if r.status_code == 200 else None


def main() -> int:
    """Run the checks and clean up the rows they create."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("HYBRID_GATEWAY_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--db", default=str(HERE / Path(os.environ.get("HYBRID_GATEWAY_DB") or "gateway.db").expanduser()))
    ap.add_argument("--only", choices=["paths", "secrets", "settings", "local", "telnyx", "auto"])
    ap.add_argument("--local-model", help="default: first model llama-server lists")
    ap.add_argument("--telnyx-model", help="default: the proxy's flash model")
    ap.add_argument("--keep", action="store_true", help="keep the rows this script creates")
    args = ap.parse_args()
    failures: list[str] = []

    if args.only in (None, "paths"):
        run_paths(failures)
    if args.only in (None, "secrets"):
        run_secrets(failures)
    if args.only in ("paths", "secrets"):
        print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) failed.'}")
        return 1 if failures else 0

    db = sqlite3.connect(args.db, isolation_level=None)
    db.row_factory = sqlite3.Row
    start_id = db.execute("SELECT COALESCE(MAX(id), 0) FROM requests").fetchone()[0]

    with httpx.Client(timeout=120) as client:
        try:
            client.get(f"{args.url}/api/summary").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"Hybrid Gateway is not reachable at {args.url}: {exc}")
            return 2

        if args.only in (None, "settings"):
            run_settings(client, args.url, failures)

        if args.only in (None, "local"):
            run_backend(client, args.url, "local", args.local_model, {}, db, failures)

        if args.only in (None, "telnyx"):
            token = proxy_token(client, args.url)
            if not token:
                check(False, "no hybrid gateway token (shell or proxy settings)", failures)
            else:
                print("\n[telnyx auth]")
                r = client.get(f"{args.url}/telnyx/v1/models", headers={"authorization": "Bearer wrong"})
                check(r.status_code == 401, f"wrong token rejected -> {r.status_code}", failures)
                fields = {f["key"]: f for f in client.get(f"{args.url}/api/settings").json()["fields"]}
                run_backend(client, args.url, "telnyx", args.telnyx_model or fields["flash_model"]["value"],
                            {"authorization": f"Bearer {token}"}, db, failures)
                run_redaction(client, args.url, args.telnyx_model or fields["flash_model"]["value"], token, db,
                              failures)

        if args.only in (None, "auto"):
            token = proxy_token(client, args.url)
            if not token:
                check(False, "no hybrid gateway token (shell or proxy settings)", failures)
            else:
                run_auto(client, args.url, token, db, failures)

    if not args.keep:
        n = db.execute("DELETE FROM requests WHERE id > ?", (start_id,)).rowcount
        print(f"\nCleaned up {n} test rows (use --keep to leave them).")

    print(f"\n{'All checks passed.' if not failures else f'{len(failures)} check(s) failed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
