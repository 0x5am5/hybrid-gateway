# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fastapi>=0.115", "httpx>=0.27", "tiktoken>=0.8", "pathspec>=0.12", "keyring>=25", "pytest>=8"]
# ///
"""Offline proof that nothing private leaves the Mac unredacted.

Builds the real FastAPI app from gateway.py with its upstream client swapped for an httpx.MockTransport, so Telnyx,
the local model and the router model are all fakes. Every assertion is about the bytes the fake Telnyx actually
received, plus what landed in gateway.db and on stderr. No proxy, model server, network or Keychain is needed
(tiktoken may fetch its encoding once).

    uv run test_redaction.py              # all tests
    uv run test_redaction.py -k auto      # pytest arguments pass straight through
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # leave no __pycache__ in the project
import gateway

JSON = dict[str, Any]

LOCAL_URL = "http://local.test/v1"
REMOTE_URL = "https://telnyx.test/v2/ai/openai"
LOCAL_MODEL = "mac-model"
ROUTER_MODEL = "router-model"
FLASH_MODEL = "zai-org/GLM-5.3-Flash"
K3_MODEL = "moonshotai/Kimi-K3"
# Plain strings that match no secret pattern, so they only show up where the gateway itself puts them.
GATEWAY_TOKEN = "hybrid-gateway-test-token"
TELNYX_KEY = "telnyx-test-upstream-key"
LOCAL_KEY = "local-test-upstream-key"
BASE = "http://127.0.0.1:8787"


def fake_secrets() -> dict[str, str]:
    """Fake credentials in real formats, one per kind, built in pieces so this file never holds one whole."""
    return {
        "telnyx-key": "KEY0" + "19A3F7C2" * 3 + "_" + "q8Wm2Lx9Zp4R",
        "aws-key": "AKIA" + "Q7XW" * 4,
        "anthropic-key": "sk-ant-" + "api03-" + "x7Kq" * 6,
        "openai-key": "sk-" + "proj-" + "Ab12Cd34Ef56Gh78Ij90Kl",
        "stripe-key": "sk_" + "live_" + "a1B2c3D4" * 3,
        "github-token": "ghp_" + "Zx9Yw8Vu7Ts6" * 2,
        "gitlab-token": "glpat-" + "Xy12Ab34Cd56Ef78Gh90",
        "slack-token": "xoxb-" + "1234567890-" + "abcdefABCDEF",
        "google-key": "AIza" + ("Xb7Q" * 9)[:35],
        "huggingface-token": "hf_" + "aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF",
        "npm-token": "npm_" + "Np4X" * 9,
        "sendgrid-key": "SG." + "Ab12Cd34Ef56Gh78" + "." + "Ij90Kl12Mn34Op56Qr78",
        "resend-key": "re_" + "Ab3dE6gH" + "_" + "x9Y8z7W6v5U4t3S2r1Q0",
        "sentry-token": "sntrys_" + "Qm9vYmFyQmF6UXV4" + "Tm9yZg",
        "shopify-token": "shpat_" + "a1b2c3d4" * 4,
        "jwt": "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "." + "SflKxwRJSMeKKF2QT4fwpM",
        "private-key": "-----BEGIN " + "RSA PRIVATE KEY-----\n" + "MIIEow" + "IBAAKCAQEA7xQ9" + "\n-----END RSA "
                       + "PRIVATE KEY-----",
    }


# Credentials that only count in context: (kind, the text sent, the secret part that must not leave).
CONTEXTUAL = [
    ("url-credentials", "DATABASE_URL=postgres://admin:" + "Sup3rS3cr3t" + "@db.internal:5432/app",
     "admin:" + "Sup3rS3cr3t"),
    ("bearer-token", "curl -H 'Authorization: Bearer " + "tokA1b2C3d4" + "E5f6G7h8I9j0' https://api.x",
     "tokA1b2C3d4" + "E5f6G7h8I9j0"),
    ("basic-auth", "Authorization: Basic " + "dXNlcjpwYXNz" + "d29yZDEyMw==", "dXNlcjpwYXNz" + "d29yZDEyMw=="),
    ("credential", "export API_SECRET=" + "q9" + "Lm2Xz7Pw4Kd8", "q9" + "Lm2Xz7Pw4Kd8"),
    ("credential", '{"apiKey": "' + "Zr8" + "Vt3Np6Qw1Hs4" + '"}', "Zr8" + "Vt3Np6Qw1Hs4"),
    ("credential", "password: " + "hunter2" + "hunter2", "hunter2" + "hunter2"),
]


# ---------------------------------------------------------------------------
# Fake upstreams
# ---------------------------------------------------------------------------


@dataclass
class Sent:
    """One request the gateway sent upstream."""

    url: str
    headers: dict[str, str]
    raw: bytes

    @property
    def json(self) -> JSON:
        """The body parsed as JSON."""
        parsed = json.loads(self.raw)
        assert isinstance(parsed, dict)
        return parsed


RouterReply = Callable[[JSON], httpx.Response]


def unread(status: int, content: bytes, content_type: str = "application/json") -> httpx.Response:
    """A response the gateway can stream from, like a real upstream (a `content=` response is already read)."""
    return httpx.Response(status, headers={"content-type": content_type}, stream=httpx.ByteStream(content))


def completion(model: str, text: str = "ok") -> httpx.Response:
    """A minimal non-streaming chat completion with usage."""
    return unread(200, json.dumps({
        "id": "cmpl-test", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }).encode())


def sse(model: str) -> httpx.Response:
    """A minimal streamed chat completion ending in a usage chunk."""
    events = [
        {"model": model, "choices": [{"index": 0, "delta": {"content": "ok"}}]},
        {"model": model, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return unread(200, body.encode(), "text/event-stream")


def router_says(route: str, category: str = "chat") -> RouterReply:
    """A router model that always picks `route`."""
    return lambda _: completion(ROUTER_MODEL, json.dumps({"route": route, "category": category}))


@dataclass
class Upstreams:
    """The fake Telnyx, local model and router model, recording everything they are sent."""

    telnyx: list[Sent] = field(default_factory=list)
    local: list[Sent] = field(default_factory=list)
    router: list[Sent] = field(default_factory=list)
    router_reply: RouterReply = field(default_factory=lambda: router_says("mac"))

    def handle(self, request: httpx.Request) -> httpx.Response:
        """MockTransport handler: record the request and answer like the real backend would."""
        sent = Sent(str(request.url), dict(request.headers), request.content)
        body: JSON = {}
        try:
            parsed = json.loads(sent.raw) if sent.raw else {}
            body = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        model = str(body.get("model") or "")
        if sent.url.startswith(REMOTE_URL):
            self.telnyx.append(sent)
            return sse(model) if body.get("stream") is True else completion(model)
        if sent.url.startswith(LOCAL_URL):
            if model == ROUTER_MODEL:
                self.router.append(sent)
                return self.router_reply(body)
            self.local.append(sent)
            return sse(model) if body.get("stream") is True else completion(model)
        raise AssertionError(f"unexpected upstream: {sent.url}")


@dataclass
class Gateway:
    """A gateway app wired to fake upstreams, plus where its files live."""

    client: TestClient
    upstreams: Upstreams
    db_path: Path

    def post(self, path: str, body: JSON | None = None, **kwargs: Any) -> httpx.Response:
        """POST as Hermes would, with the hybrid gateway token."""
        headers = {"authorization": f"Bearer {GATEWAY_TOKEN}", **kwargs.pop("headers", {})}
        if body is not None:
            kwargs["json"] = body
        return self.client.post(path, headers=headers, **kwargs)

    def rows(self) -> list[sqlite3.Row]:
        """Every logged request row."""
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        try:
            return db.execute("SELECT * FROM requests ORDER BY id").fetchall()
        finally:
            db.close()

    def db_dump(self) -> str:
        """The whole database as SQL text, for checking that no value was stored."""
        db = sqlite3.connect(self.db_path)
        try:
            return "\n".join(db.iterdump())
        finally:
            db.close()


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def make_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[GatewayFactory]:
    """Build gateways on temp files with fake upstreams; extra keyword arguments become env vars."""
    clients: list[TestClient] = []
    monkeypatch.setattr(gateway, "keychain_usable", lambda: False)  # never read the real Keychain

    def build(**env: str) -> Gateway:
        values = {
            "HYBRID_GATEWAY_LOCAL_URL": LOCAL_URL, "HYBRID_GATEWAY_LOCAL_API_KEY": LOCAL_KEY,
            "HYBRID_GATEWAY_LOCAL_MODEL": LOCAL_MODEL, "HYBRID_GATEWAY_ROUTER_MODEL": ROUTER_MODEL,
            "HYBRID_GATEWAY_REMOTE_URL": REMOTE_URL, "TELNYX_API_KEY": TELNYX_KEY,
            "HYBRID_GATEWAY_FLASH_MODEL": FLASH_MODEL, "HYBRID_GATEWAY_K3_MODEL": K3_MODEL,
            "HYBRID_GATEWAY_TOKEN": GATEWAY_TOKEN, **env,
        }
        folder = tmp_path / f"gw{len(clients)}"
        folder.mkdir()
        startup = gateway.Startup(
            host="127.0.0.1", port=8787, db_path=folder / "gateway.db", prices_path=HERE / "prices.json",
            settings_path=folder / "settings.json", keychain_service="hybrid-gateway-test",
            env_file=folder / ".env", env_file_loaded=False,
        )
        settings = gateway.Settings(startup, gateway.Env(values, {k: "shell" for k in values}, startup.env_file, False))
        upstreams = Upstreams()
        real_client = httpx.AsyncClient
        with monkeypatch.context() as m:  # only the gateway's own upstream client gets the fake transport
            m.setattr(httpx, "AsyncClient",
                      lambda **kw: real_client(transport=httpx.MockTransport(upstreams.handle), **kw))
            app = gateway.create_app(settings)
        client = TestClient(app, base_url=BASE)
        client.__enter__()
        clients.append(client)
        return Gateway(client, upstreams, startup.db_path)

    yield build
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def gw(make_gateway: GatewayFactory) -> Gateway:
    """A gateway with default settings."""
    return make_gateway()


def assert_absent(sent: Sent | bytes, *values: str) -> None:
    """None of the values appear in what was sent, raw or JSON-decoded (escaping can't hide one)."""
    raw = sent.raw if isinstance(sent, Sent) else sent
    text = raw.decode("utf-8", "replace")
    decoded = text
    try:
        decoded = json.dumps(json.loads(raw), ensure_ascii=False)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    for value in values:
        assert value not in text and value not in decoded, "a credential reached Telnyx"
        for line in value.splitlines():  # multi-line keys: no line of the body survives either
            if len(line) >= 12:
                assert line not in decoded, "part of a credential reached Telnyx"


def chat(*messages: JSON, **extra: Any) -> JSON:
    """A chat completion body for the Flash model."""
    return {"model": FLASH_MODEL, "messages": list(messages), **extra}


def read_call(cid: str, path: str, name: str = "read_file") -> JSON:
    """An assistant tool call that reads a file."""
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps({"path": path})}}


# ---------------------------------------------------------------------------
# /telnyx: everything bound for the cloud is redacted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(fake_secrets()))
def test_every_known_format_is_redacted_before_telnyx(gw: Gateway, kind: str) -> None:
    secret = fake_secrets()[kind]
    r = gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": f"Why is this failing? {secret}"}))

    assert r.status_code == 200
    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert f"[redacted:{kind}]" in sent.json["messages"][-1]["content"]
    assert gw.rows()[-1]["redactions"] == 1


@pytest.mark.parametrize(("kind", "text", "secret"), CONTEXTUAL, ids=[f"{k}-{i}" for i, (k, _, _) in
                                                                      enumerate(CONTEXTUAL)])
def test_contextual_credentials_are_redacted_before_telnyx(gw: Gateway, kind: str, text: str, secret: str) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": text}))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert f"[redacted:{kind}]" in sent.json["messages"][-1]["content"]


def test_many_credentials_in_one_message_are_all_redacted(gw: Gateway) -> None:
    fakes = fake_secrets()
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": "\n".join(fakes.values())}))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, *fakes.values())
    assert gw.rows()[-1]["redactions"] == len(fakes)


def test_every_message_role_and_shape_is_redacted(gw: Gateway) -> None:
    f = fake_secrets()
    body = chat(
        {"role": "system", "content": f"You are Hermes. Deploy key: {f['github-token']}"},
        {"role": "user", "content": [
            {"type": "text", "text": f"Look at {f['aws-key']}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]},
        {"role": "assistant", "content": f"I'll set {f['stripe-key']}", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(
                {"path": "deploy.sh", "content": f"export TELNYX_API_KEY={f['telnyx-key']}",
                 "nested": {"list": [f"token {f['slack-token']}"]}})}},
            {"id": "c2", "type": "function", "function": {"name": "bash",
                                                          "arguments": f"not json {f['openai-key']}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": f"wrote {f['anthropic-key']}"},
        {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": f"out: {f['jwt']}"}]},
        {"role": "developer", "content": f"Also {f['huggingface-token']}"},
    )
    gw.post("/telnyx/v1/chat/completions", body)

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, *f.values())
    msgs = sent.json["messages"]
    args = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])  # still valid JSON for the model
    assert args["content"] == "export TELNYX_API_KEY=[redacted:telnyx-key]"
    assert args["nested"]["list"] == ["token [redacted:slack-token]"]
    assert msgs[2]["tool_calls"][1]["function"]["arguments"] == "not json [redacted:openai-key]"
    assert msgs[1]["content"][1] == body["messages"][1]["content"][1], "image parts pass through untouched"
    assert [m.get("tool_call_id") for m in msgs] == [m.get("tool_call_id") for m in body["messages"]]


@pytest.mark.parametrize(("call", "label"), [
    (read_call("c1", ".env"), ".env"),
    (read_call("c1", "/srv/app/.env.production"), ".env.production"),
    (read_call("c1", "~/.ssh/id_ed25519"), "id_ed25519"),
    (read_call("c1", "certs/server.pem"), "server.pem"),
    (read_call("c1", "infra/prod.tfvars"), "prod.tfvars"),
    (read_call("c1", "gcp/service-account-prod.json"), "service-account-prod.json"),
    ({"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
        {"command": "cat config/.env.local | grep DB_"})}}, ".env.local"),
    ({"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps(
        {"command": "cat ~/.aws/credentials"})}}, "credentials"),
])
def test_tool_result_from_a_secrets_file_is_withheld_whole(gw: Gateway, call: JSON, label: str) -> None:
    contents = "DB_HOST=db.internal\nDB_USER=app\nDB_NAME=billing_prod"  # matches no pattern on purpose
    gw.post("/telnyx/v1/chat/completions", chat(
        {"role": "user", "content": "What database does the app use?"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": contents},
    ))

    [sent] = gw.upstreams.telnyx
    assert "db.internal" not in sent.raw.decode() and "billing_prod" not in sent.raw.decode()
    tool = sent.json["messages"][-1]
    assert tool["tool_call_id"] == "c1" and "withheld" in tool["content"] and label in tool["content"]
    assert gw.rows()[-1]["redactions"] == 1


@pytest.mark.parametrize("path", [".env.example", ".env.sample", ".env.template", "README.md", "src/env.ts"])
def test_tool_results_from_ordinary_files_are_kept(gw: Gateway, path: str) -> None:
    gw.post("/telnyx/v1/chat/completions", chat(
        {"role": "user", "content": "What port?"},
        {"role": "assistant", "content": None, "tool_calls": [read_call("c1", path)]},
        {"role": "tool", "tool_call_id": "c1", "content": "PORT=3000"},
    ))

    [sent] = gw.upstreams.telnyx
    assert sent.json["messages"][-1]["content"] == "PORT=3000"
    assert gw.rows()[-1]["redactions"] == 0


def test_streaming_requests_are_redacted(gw: Gateway) -> None:
    secret = fake_secrets()["aws-key"]
    with gw.client.stream("POST", "/telnyx/v1/chat/completions", headers={"authorization": f"Bearer {GATEWAY_TOKEN}"},
                          json=chat({"role": "user", "content": secret}, stream=True)) as r:
        streamed = b"".join(r.iter_bytes())

    assert r.status_code == 200 and b"data: [DONE]" in streamed
    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert sent.json["stream_options"] == {"include_usage": True}
    assert gw.rows()[-1]["redactions"] == 1


@pytest.mark.parametrize("prompt", ["str", "list"])
def test_legacy_completions_prompt_is_redacted(gw: Gateway, prompt: str) -> None:
    secret = fake_secrets()["stripe-key"]
    value: str | list[str] = f"Charge with {secret}" if prompt == "str" else ["Charge with", secret]
    gw.post("/telnyx/v1/completions", {"model": FLASH_MODEL, "prompt": value})

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert "[redacted:stripe-key]" in json.dumps(sent.json["prompt"])


def test_responses_api_input_is_redacted(gw: Gateway) -> None:
    secret = fake_secrets()["github-token"]
    gw.post("/telnyx/v1/responses", {"model": FLASH_MODEL, "input": [
        {"role": "user", "content": [{"type": "input_text", "text": f"Push with {secret}"}]}]})

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert "[redacted:github-token]" in sent.raw.decode()


def test_embeddings_input_is_redacted(gw: Gateway) -> None:
    secret = fake_secrets()["openai-key"]
    gw.post("/telnyx/v1/embeddings", {"model": "embed", "input": [f"config: {secret}", "hello"]})

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert sent.json["input"] == ["config: [redacted:openai-key]", "hello"]


def test_redaction_is_deterministic_so_the_prompt_cache_holds(gw: Gateway) -> None:
    f = fake_secrets()
    body = chat({"role": "system", "content": "You are Hermes."},
                {"role": "user", "content": f"{f['aws-key']} and {f['telnyx-key']}"},
                {"role": "assistant", "content": None, "tool_calls": [read_call("c1", ".env")]},
                {"role": "tool", "tool_call_id": "c1", "content": "PORT=1"})
    gw.post("/telnyx/v1/chat/completions", body)
    gw.post("/telnyx/v1/chat/completions", body)

    first, second = gw.upstreams.telnyx
    assert first.raw == second.raw


def test_a_placeholder_note_is_added_to_the_system_prompt(gw: Gateway) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "system", "content": "You are Hermes."},
                                                {"role": "user", "content": fake_secrets()["aws-key"]}))
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": fake_secrets()["aws-key"]}))

    with_system, without = (s.json["messages"] for s in gw.upstreams.telnyx)
    assert with_system[0]["content"] == f"You are Hermes.\n\n{gateway.REDACTION_NOTE}" and len(with_system) == 2
    assert without[0] == {"role": "system", "content": gateway.REDACTION_NOTE} and len(without) == 2


def test_a_clean_request_is_forwarded_byte_for_byte(gw: Gateway) -> None:
    raw = json.dumps(chat({"role": "user", "content": "Explain the max_tokens: 16 setting and process.env.API_KEY"}))
    gw.post("/telnyx/v1/chat/completions", content=raw.encode(), headers={"content-type": "application/json"})

    [sent] = gw.upstreams.telnyx
    assert sent.raw == raw.encode()
    assert gateway.REDACTION_NOTE not in sent.raw.decode()
    assert gw.rows()[-1]["redactions"] == 0


@pytest.mark.parametrize("text", [
    "API_KEY=your-api-key-here", "apiKey: process.env.API_KEY", "token = ${GITHUB_TOKEN}", "max_tokens: 16",
    "secret = has_secret(body)", 'token_url = "https://auth.example.com/token"', "password: <your password>",
    "Use sk-... as the format", "cache_key = user_id",
])
def test_placeholders_and_code_are_not_redacted(gw: Gateway, text: str) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": text}))

    [sent] = gw.upstreams.telnyx
    assert sent.json["messages"] == [{"role": "user", "content": text}]


def test_tool_schemas_are_left_alone(gw: Gateway) -> None:
    """Documented behaviour: the agent's own tool schemas are not redacted (README, Secrets)."""
    tools = [{"type": "function", "function": {"name": "deploy", "description": "Deploys the app",
                                               "parameters": {"type": "object", "properties": {}}}}]
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": fake_secrets()["aws-key"]}, tools=tools))

    [sent] = gw.upstreams.telnyx
    assert sent.json["tools"] == tools


def test_telnyx_gets_its_own_key_and_never_the_gateway_token_or_local_key(gw: Gateway) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": "hi"}))

    [sent] = gw.upstreams.telnyx
    assert sent.headers["authorization"] == f"Bearer {TELNYX_KEY}"
    everything = json.dumps(sent.headers) + sent.raw.decode()
    assert GATEWAY_TOKEN not in everything and LOCAL_KEY not in everything


@pytest.mark.parametrize("auth", [None, "Bearer wrong", f"Bearer {TELNYX_KEY}"])
def test_nothing_reaches_telnyx_without_the_gateway_token(gw: Gateway, auth: str | None) -> None:
    headers = {"authorization": auth} if auth else {}
    r = gw.client.post("/telnyx/v1/chat/completions", headers=headers,
                       json=chat({"role": "user", "content": fake_secrets()["aws-key"]}))

    assert r.status_code == 401
    assert gw.upstreams.telnyx == []


def test_cross_site_pages_cannot_use_the_proxy(gw: Gateway) -> None:
    r = gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": "hi"}),
                headers={"origin": "https://evil.example"})

    assert r.status_code == 403
    assert gw.upstreams.telnyx == []


def test_local_requests_are_not_redacted(gw: Gateway) -> None:
    """Redaction is for the cloud only: the Mac sees the real conversation."""
    secret = fake_secrets()["aws-key"]
    gw.post("/local/v1/chat/completions", {"model": LOCAL_MODEL, "messages": [{"role": "user", "content": secret}]})

    [sent] = gw.upstreams.local
    assert secret in sent.raw.decode()
    assert gw.upstreams.telnyx == []
    assert gw.rows()[-1]["redactions"] == 0


def test_values_never_reach_the_log_database_or_stderr(gw: Gateway, capsys: pytest.CaptureFixture[str]) -> None:
    f = fake_secrets()
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": f"{f['aws-key']} {f['aws-key']}"}))
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": f["telnyx-key"]}))

    err = capsys.readouterr().err
    assert "redacted 2 private value(s) before sending to Telnyx: aws-key x2" in err
    assert "redacted 1 private value(s) before sending to Telnyx: telnyx-key" in err
    dump = gw.db_dump()
    for value in (f["aws-key"], f["telnyx-key"]):
        assert value not in err and value not in dump
    assert [row["redactions"] for row in gw.rows()] == [2, 1]
    cols = {c[1] for c in sqlite3.connect(gw.db_path).execute("PRAGMA table_info(requests)")}
    assert not cols & {"prompt", "response", "messages", "content"}


def test_the_csv_export_holds_only_counts(gw: Gateway) -> None:
    secret = fake_secrets()["aws-key"]
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": secret}))

    csv = gw.client.get("/api/export.csv").text
    assert secret not in csv
    assert csv.splitlines()[1].split(",")[14] == "1"  # the redactions column


# ---------------------------------------------------------------------------
# /auto: credentials keep work on the Mac; anything that still goes to the cloud is redacted
# ---------------------------------------------------------------------------


def auto(gw: Gateway, *messages: JSON) -> httpx.Response:
    """One /auto request."""
    return gw.post("/auto/v1/chat/completions", {"model": "auto", "messages": list(messages)})


def test_auto_credential_keeps_the_request_on_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("k3", "debug")  # would go to the cloud if asked
    r = auto(gw, {"role": "user", "content": f"Why can't I connect? {CONTEXTUAL[0][1]}"})

    assert r.status_code == 200
    assert gw.upstreams.telnyx == [] and gw.upstreams.router == []
    [sent] = gw.upstreams.local
    assert sent.json["model"] == LOCAL_MODEL
    row = gw.rows()[-1]
    assert (row["route"], row["route_source"], row["backend"]) == ("mac", "rule:secret", "local")


def test_auto_secrets_file_read_keeps_the_conversation_on_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("flash")
    r = auto(gw, {"role": "user", "content": "What port does the app use?"},
             {"role": "assistant", "content": None, "tool_calls": [read_call("c1", "app/.env.local")]},
             {"role": "tool", "tool_call_id": "c1", "content": "PORT=3000"})

    assert r.status_code == 200
    assert gw.upstreams.telnyx == []
    assert gw.rows()[-1]["route_source"] == "rule:secret"


def test_auto_credential_keeps_later_turns_on_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("k3", "refactor")
    first = {"role": "user", "content": f"Use {fake_secrets()['stripe-key']} for billing"}
    history = [first, {"role": "assistant", "content": "Done."}]
    for follow_up in ("Now refactor the whole billing module across the monorepo", "And write the tests"):
        auto(gw, *history, {"role": "user", "content": follow_up})
        history += [{"role": "user", "content": follow_up}, {"role": "assistant", "content": "ok"}]

    assert gw.upstreams.telnyx == []
    assert [row["route_source"] for row in gw.rows()] == ["rule:secret", "rule:secret"]


def test_auto_credential_in_a_tool_result_pulls_a_cloud_loop_back_to_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("flash", "feature")
    system = {"role": "system", "content": "You are Hermes."}
    user = {"role": "user", "content": "Add a health check endpoint"}
    auto(gw, system, user)
    assert len(gw.upstreams.telnyx) == 1  # a clean turn goes to Flash

    auto(gw, system, user,
         {"role": "assistant", "content": None, "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "printenv"}'}}]},
         {"role": "tool", "tool_call_id": "c1", "content": f"AWS_ACCESS_KEY_ID={fake_secrets()['aws-key']}"})

    assert len(gw.upstreams.telnyx) == 1, "the tool result holding a key must not go to Flash"
    assert [row["route_source"] for row in gw.rows() if row["purpose"] == "work"] == ["router", "rule:secret"]


@pytest.mark.parametrize("failure", ["error", "timeout", "garbage", "bad-route"])
def test_auto_router_failure_never_falls_back_to_the_cloud(gw: Gateway, failure: str) -> None:
    def reply(_: JSON) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("router timed out")
        if failure == "error":
            return unread(500, b'{"error": "boom"}')
        content = "I think k3 would be best" if failure == "garbage" else '{"route": "cloud", "category": "chat"}'
        return completion(ROUTER_MODEL, content)

    gw.upstreams.router_reply = reply
    r = auto(gw, {"role": "user", "content": "Refactor the billing system"})

    assert r.status_code == 200
    assert gw.upstreams.telnyx == []
    assert [row["route_source"] for row in gw.rows() if row["purpose"] == "work"] == ["fallback"]


def test_auto_clean_request_goes_to_the_cloud_unchanged(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("flash", "feature")
    messages = [{"role": "system", "content": "You are Hermes."}, {"role": "user", "content": "Add a /health route"}]
    auto(gw, *messages)

    [sent] = gw.upstreams.telnyx
    assert sent.json["model"] == FLASH_MODEL and sent.json["messages"] == messages
    assert sent.headers["authorization"] == f"Bearer {TELNYX_KEY}"
    assert GATEWAY_TOKEN not in json.dumps(sent.headers)


def test_auto_router_model_runs_locally(gw: Gateway) -> None:
    """The router sees the user's request, so it must only ever be called on the Mac."""
    gw.upstreams.router_reply = router_says("mac")
    auto(gw, {"role": "user", "content": "What does git rebase do?"})

    assert len(gw.upstreams.router) == 1 and gw.upstreams.router[0].url.startswith(LOCAL_URL)
    assert gw.upstreams.telnyx == []


@pytest.mark.parametrize(("env", "route", "model"), [
    ({"HYBRID_GATEWAY_LOCAL_CONTEXT": "1000"}, "flash", FLASH_MODEL),  # too big for the Mac
    ({"HYBRID_GATEWAY_LOCAL_CONTEXT": "1000", "HYBRID_GATEWAY_FLASH_CONTEXT": "1000"}, "k3", K3_MODEL),  # and Flash
])
def test_auto_oversized_credential_goes_to_the_cloud_redacted(make_gateway: GatewayFactory,
                                                             capsys: pytest.CaptureFixture[str],
                                                             env: dict[str, str], route: str, model: str) -> None:
    gw = make_gateway(**env)
    gw.upstreams.router_reply = router_says("mac")  # forced off the Mac by the context rule
    f = fake_secrets()
    big = "Summarise this log.\n" + "the service restarted cleanly at noon\n" * 300
    r = auto(gw, {"role": "system", "content": "You are Hermes."},
             {"role": "user", "content": f"{big}\nkey={f['aws-key']}\n{CONTEXTUAL[0][1]}"},
             {"role": "assistant", "content": None, "tool_calls": [read_call("c1", ".env")]},
             {"role": "tool", "tool_call_id": "c1", "content": "PORT=3000"})

    assert r.status_code == 200
    assert gw.upstreams.local == []
    [sent] = gw.upstreams.telnyx
    assert sent.json["model"] == model
    assert_absent(sent, f["aws-key"], CONTEXTUAL[0][2])
    assert "withheld" in sent.json["messages"][-1]["content"]
    assert gateway.REDACTION_NOTE in sent.json["messages"][0]["content"]
    row = gw.rows()[-1]
    assert (row["route"], row["route_source"], row["redactions"]) == (route, "rule:context", 3)
    assert "too big for the Mac" in capsys.readouterr().err


def test_auto_oversized_streaming_credential_is_redacted(make_gateway: GatewayFactory) -> None:
    gw = make_gateway(HYBRID_GATEWAY_LOCAL_CONTEXT="1000")
    secret = fake_secrets()["github-token"]
    big = "the service restarted cleanly at noon\n" * 300
    with gw.client.stream("POST", "/auto/v1/chat/completions", headers={"authorization": f"Bearer {GATEWAY_TOKEN}"},
                          json={"model": "auto", "stream": True,
                                "messages": [{"role": "user", "content": f"{big}{secret}"}]}) as r:
        b"".join(r.iter_bytes())

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)


def test_auto_route_probe_never_contacts_telnyx(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("k3")
    d = gw.client.post("/api/route", json={"messages": [{"role": "user", "content": fake_secrets()["aws-key"]}]})

    assert d.json()["route"] == "mac" and d.json()["source"] == "rule:secret"
    assert gw.upstreams.telnyx == [] and gw.rows() == []


# ---------------------------------------------------------------------------
# /auto routing by size: small on the Mac, medium on Flash, large on K3, redacted whenever it leaves the Mac
# ---------------------------------------------------------------------------

MAC_CONTEXT = 2_000
FLASH_CONTEXT = 8_000
SIZES = {"small": 300, "medium": 4_000, "large": 12_000}  # target prompt tokens for each band
LINE = "The billing service restarted cleanly at noon after the deploy.\n"


def tokens_of(messages: list[JSON]) -> int:
    """Prompt size exactly as the gateway counts it."""
    return gateway.count_tokens(gateway.prompt_text({"messages": messages}))


def sized(target: int, ask: str = "Summarise this log.", extra: str = "") -> list[JSON]:
    """A system prompt plus one user message of about `target` tokens."""
    system = {"role": "system", "content": "You are Hermes."}
    head = f"{ask}\n{extra}\n" if extra else f"{ask}\n"
    per_line = gateway.count_tokens(LINE)
    base = tokens_of([system, {"role": "user", "content": head}])
    return [system, {"role": "user", "content": head + LINE * max((target - base) // per_line, 0)}]


@pytest.fixture
def sized_gw(make_gateway: GatewayFactory) -> Gateway:
    """A gateway with small, known Mac and Flash windows."""
    return make_gateway(HYBRID_GATEWAY_LOCAL_CONTEXT=str(MAC_CONTEXT), HYBRID_GATEWAY_FLASH_CONTEXT=str(FLASH_CONTEXT))


def work_rows(gw: Gateway) -> list[sqlite3.Row]:
    """Logged work requests, without the router's own calls."""
    return [row for row in gw.rows() if row["purpose"] == "work"]


def test_size_bands_are_what_the_tests_assume() -> None:
    small, medium, large = (tokens_of(sized(n)) for n in SIZES.values())
    assert small <= MAC_CONTEXT < medium <= FLASH_CONTEXT < large


def test_small_request_runs_on_the_mac(sized_gw: Gateway) -> None:
    sized_gw.upstreams.router_reply = router_says("mac", "question")
    r = auto(sized_gw, *sized(SIZES["small"], "What does this log say?"))

    assert r.status_code == 200 and r.json()["model"] == LOCAL_MODEL
    assert sized_gw.upstreams.telnyx == []
    [sent] = sized_gw.upstreams.local
    assert sent.json["model"] == LOCAL_MODEL
    assert sent.headers["authorization"] == f"Bearer {LOCAL_KEY}", "the Mac gets its own key, not the gateway token"
    [row] = work_rows(sized_gw)
    assert (row["backend"], row["route"], row["route_source"], row["route_category"]) == \
        ("local", "mac", "router", "question")


@pytest.mark.parametrize("router", ["mac", "flash"])
def test_medium_request_runs_on_flash(sized_gw: Gateway, router: str) -> None:
    """Too big for the Mac: even when the router picks the Mac, it goes to Flash."""
    sized_gw.upstreams.router_reply = router_says(router, "writing")
    messages = sized(SIZES["medium"])
    r = auto(sized_gw, *messages)

    assert r.status_code == 200
    assert sized_gw.upstreams.local == []
    [sent] = sized_gw.upstreams.telnyx
    assert sent.json["model"] == FLASH_MODEL and sent.json["messages"] == messages
    assert sent.headers["authorization"] == f"Bearer {TELNYX_KEY}"
    [row] = work_rows(sized_gw)
    assert (row["backend"], row["route"]) == ("telnyx", "flash")
    assert row["route_source"] == ("rule:context" if router == "mac" else "router")


def test_medium_request_can_still_be_escalated_to_k3_by_the_router(sized_gw: Gateway) -> None:
    """Size sets a floor, not a ceiling: hard work the router sends to K3 goes there."""
    sized_gw.upstreams.router_reply = router_says("k3", "debug")
    auto(sized_gw, *sized(SIZES["medium"], "Find the race condition in this log."))

    [sent] = sized_gw.upstreams.telnyx
    assert sent.json["model"] == K3_MODEL
    assert work_rows(sized_gw)[0]["route_source"] == "router"


@pytest.mark.parametrize("router", ["mac", "flash", "k3"])
def test_large_request_runs_on_k3_without_asking_the_router(sized_gw: Gateway, router: str) -> None:
    sized_gw.upstreams.router_reply = router_says(router)
    messages = sized(SIZES["large"])
    r = auto(sized_gw, *messages)

    assert r.status_code == 200
    assert sized_gw.upstreams.router == [] and sized_gw.upstreams.local == []
    [sent] = sized_gw.upstreams.telnyx
    assert sent.json["model"] == K3_MODEL and sent.json["messages"] == messages
    [row] = work_rows(sized_gw)
    assert (row["backend"], row["route"], row["route_source"]) == ("telnyx", "k3", "rule:context")


@pytest.mark.parametrize("over", [0, 1])
def test_mac_boundary(make_gateway: GatewayFactory, over: int) -> None:
    """Exactly the Mac context still fits on the Mac; one token more does not."""
    messages = sized(1_500)
    gw = make_gateway(HYBRID_GATEWAY_LOCAL_CONTEXT=str(tokens_of(messages) - over))
    gw.upstreams.router_reply = router_says("mac")
    auto(gw, *messages)

    assert work_rows(gw)[0]["route"] == ("mac" if over == 0 else "flash")


@pytest.mark.parametrize("over", [0, 1])
def test_flash_boundary(make_gateway: GatewayFactory, over: int) -> None:
    """Exactly the Flash context stays on Flash; one token more goes to K3."""
    messages = sized(3_000)
    gw = make_gateway(HYBRID_GATEWAY_LOCAL_CONTEXT="1000", HYBRID_GATEWAY_FLASH_CONTEXT=str(tokens_of(messages) - over))
    gw.upstreams.router_reply = router_says("flash")
    auto(gw, *messages)

    [sent] = gw.upstreams.telnyx
    assert sent.json["model"] == (FLASH_MODEL if over == 0 else K3_MODEL)


@pytest.mark.parametrize(("size", "route", "model"), [("medium", "flash", FLASH_MODEL), ("large", "k3", K3_MODEL)])
def test_cloud_tiers_redact_credentials(sized_gw: Gateway, size: str, route: str, model: str) -> None:
    f = fake_secrets()
    sized_gw.upstreams.router_reply = router_says("mac")
    leak = f"AWS_ACCESS_KEY_ID={f['aws-key']}\n{CONTEXTUAL[0][1]}\n{CONTEXTUAL[1][1]}"
    messages = [*sized(SIZES[size], "Why does the deploy fail?", leak),
                {"role": "assistant", "content": None, "tool_calls": [read_call("c1", "deploy/.env.production")]},
                {"role": "tool", "tool_call_id": "c1", "content": "STAGE=prod\nREGION=eu-west-2"}]
    r = auto(sized_gw, *messages)

    assert r.status_code == 200
    assert sized_gw.upstreams.local == []
    [sent] = sized_gw.upstreams.telnyx
    assert sent.json["model"] == model
    assert_absent(sent, f["aws-key"], CONTEXTUAL[0][2], CONTEXTUAL[1][2])
    assert "eu-west-2" not in sent.raw.decode() and "withheld" in sent.json["messages"][-1]["content"]
    assert sent.json["messages"][1]["content"].count(LINE) > 0, "the rest of the prompt is sent intact"
    row = work_rows(sized_gw)[0]
    assert (row["route"], row["route_source"], row["redactions"]) == (route, "rule:context", 4)


def test_small_request_with_a_credential_stays_on_the_mac_unredacted(sized_gw: Gateway) -> None:
    """Small enough for the Mac: the credential never leaves, so there is nothing to redact."""
    secret = fake_secrets()["aws-key"]
    sized_gw.upstreams.router_reply = router_says("k3")
    auto(sized_gw, *sized(SIZES["small"], "Why does the deploy fail?", f"AWS_ACCESS_KEY_ID={secret}"))

    assert sized_gw.upstreams.telnyx == [] and sized_gw.upstreams.router == []
    [sent] = sized_gw.upstreams.local
    assert secret in sent.raw.decode()
    assert work_rows(sized_gw)[0]["route_source"] == "rule:secret"


def tool_turn(cid: str, output: str) -> list[JSON]:
    """An assistant tool call and its result."""
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "bash", "arguments": '{"command": "tail log"}'}}]},
            {"role": "tool", "tool_call_id": cid, "content": output}]


def test_a_mac_tool_loop_moves_to_flash_when_it_outgrows_the_mac(sized_gw: Gateway) -> None:
    sized_gw.upstreams.router_reply = router_says("mac", "ops")
    messages = sized(SIZES["small"], "Tail the log and tell me what happened.")
    auto(sized_gw, *messages)
    auto(sized_gw, *messages, *tool_turn("c1", "short output"))
    auto(sized_gw, *messages, *tool_turn("c1", LINE * 400))

    assert len(sized_gw.upstreams.router) == 1, "the router is only asked on the user's turn"
    assert [(r["route"], r["route_source"]) for r in work_rows(sized_gw)] == \
        [("mac", "router"), ("mac", "rule:tool-loop"), ("flash", "rule:context")]
    [sent] = sized_gw.upstreams.telnyx
    assert sent.json["model"] == FLASH_MODEL


def test_a_flash_tool_loop_stays_on_flash_then_moves_to_k3_when_it_outgrows_flash(sized_gw: Gateway) -> None:
    sized_gw.upstreams.router_reply = router_says("flash", "feature")
    messages = sized(SIZES["medium"], "Add a health check.")
    auto(sized_gw, *messages)
    auto(sized_gw, *messages, *tool_turn("c1", "ok"))
    auto(sized_gw, *messages, *tool_turn("c1", LINE * 1_000))

    assert [s.json["model"] for s in sized_gw.upstreams.telnyx] == [FLASH_MODEL, FLASH_MODEL, K3_MODEL]
    assert [r["route_source"] for r in work_rows(sized_gw)] == ["router", "rule:tool-loop", "rule:context"]
    assert sized_gw.upstreams.local == []


def test_a_tool_result_that_pushes_a_secret_conversation_past_the_mac_is_redacted(sized_gw: Gateway) -> None:
    """A credential keeps work on the Mac until it no longer fits; then it goes to the cloud redacted."""
    secret = fake_secrets()["telnyx-key"]
    messages = sized(SIZES["small"], "Check the deploy.", f"TELNYX_API_KEY={secret}")
    auto(sized_gw, *messages)
    auto(sized_gw, *messages, *tool_turn("c1", LINE * 400))

    assert [(r["route"], r["route_source"]) for r in work_rows(sized_gw)] == \
        [("mac", "rule:secret"), ("flash", "rule:context")]
    [sent] = sized_gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert work_rows(sized_gw)[1]["redactions"] == 1


def test_auto_replaces_the_model_and_keeps_everything_else(sized_gw: Gateway) -> None:
    sized_gw.upstreams.router_reply = router_says("flash")
    body = {"model": "auto", "messages": sized(SIZES["medium"]), "temperature": 0.2, "max_tokens": 512,
            "tools": [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]}
    sized_gw.post("/auto/v1/chat/completions", body)

    [sent] = sized_gw.upstreams.telnyx
    assert sent.json == {**body, "model": FLASH_MODEL}


# ---------------------------------------------------------------------------
# Fields and bodies that used to slip through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["instructions", "metadata", "user"])
def test_every_top_level_field_is_redacted(gw: Gateway, key: str) -> None:
    secret = fake_secrets()["aws-key"]
    value: str | JSON = {"note": secret} if key == "metadata" else f"Deploy with {secret}"
    gw.post("/telnyx/v1/responses", {"model": FLASH_MODEL, "input": "hi", key: value})

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    assert "[redacted:aws-key]" in json.dumps(sent.json[key])


@pytest.mark.parametrize("key", ["name", "id", "type", "role", "file", "image_url"])
def test_free_form_metadata_is_checked_under_any_key(gw: Gateway, key: str) -> None:
    """Codex review on PR #1: protocol key names inside metadata used to be skipped."""
    email, ssn = fake_pii()["email"], fake_pii()["us-ssn"]
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": "hi"},
                                                metadata={key: email, "nested": [{key: ssn}]}))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, email, ssn)
    assert sent.json["metadata"] == {key: "[redacted:email]", "nested": [{key: "[redacted:us-ssn]"}]}


def test_protocol_objects_keep_their_ids_and_binary_parts(gw: Gateway) -> None:
    image = {"url": "data:image/png;base64," + "AKIA" + "Q7XW" * 4}  # base64 that happens to look like a key
    item = {"type": "message", "id": "msg_1", "role": "user",
            "content": [{"type": "input_text", "text": fake_pii()["email"]},
                        {"type": "input_image", "image_url": image}]}
    gw.post("/telnyx/v1/responses", {"model": FLASH_MODEL, "input": [item]})

    [sent] = gw.upstreams.telnyx
    [got] = sent.json["input"]
    assert got["content"][1] == item["content"][1], "binary payloads in protocol objects are never altered"
    assert (got["id"], got["type"], got["content"][0]["text"]) == ("msg_1", "message", "[redacted:email]")


def test_response_format_schemas_are_left_alone(gw: Gateway) -> None:
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": fake_secrets()["aws-key"]},
                                                response_format=schema))

    [sent] = gw.upstreams.telnyx
    assert sent.json["response_format"] == schema


@pytest.mark.parametrize("key", ["name", "id", "type", "file"])
def test_tool_argument_values_under_any_key_are_redacted(gw: Gateway, key: str) -> None:
    secret = fake_secrets()["aws-key"]
    gw.post("/telnyx/v1/chat/completions", chat(
        {"role": "user", "content": "Store it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "set_secret",
                                                          "arguments": json.dumps({key: secret})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, secret)
    call = next(m for m in sent.json["messages"] if m.get("tool_calls"))["tool_calls"][0]
    assert (call["id"], call["function"]["name"]) == ("c1", "set_secret"), "the call itself is untouched"
    assert json.loads(call["function"]["arguments"]) == {key: "[redacted:aws-key]"}


@pytest.mark.parametrize(("content", "content_type"), [
    (b"AWS_ACCESS_KEY_ID=" + fake_secrets()["aws-key"].encode(), "text/plain"),
    (b'--b\r\nContent-Disposition: form-data; name="file"\r\n\r\njane.doe@acme-test.co.uk\r\n--b--',
     "multipart/form-data; boundary=b"),
    (b'["a JSON array", "' + fake_secrets()["aws-key"].encode() + b'"]', "application/json"),
])
def test_bodies_that_cannot_be_redacted_are_refused(gw: Gateway, content: bytes, content_type: str) -> None:
    r = gw.post("/telnyx/v1/files", content=content, headers={"content-type": content_type})

    assert r.status_code == 415 and "only forwards JSON" in r.json()["error"]
    assert gw.upstreams.telnyx == []


def test_bodyless_posts_still_reach_telnyx(gw: Gateway) -> None:
    r = gw.post("/telnyx/v1/batches/b1/cancel")

    assert r.status_code == 200 and len(gw.upstreams.telnyx) == 1


# ---------------------------------------------------------------------------
# Personal data (PII): redacted on the way to the cloud, and kept on the Mac by /auto when it fits
# ---------------------------------------------------------------------------


def fake_pii() -> dict[str, str]:
    """Personal data in real formats, using reserved or published test values so none belongs to anyone."""
    return {
        "email": "jane.doe" + "@acme-test.co.uk",
        "phone": "+44 20 7946 " + "0958",  # Ofcom drama range
        "card-number": "4111 1111 " + "1111 1111",  # Visa test card
        "iban": "GB82 WEST " + "1234 5698 7654 32",  # the standard example IBAN
        "uk-ni-number": "AB 12 " + "34 56 C",
        "us-ssn": "123-45-" + "6789",
    }


PII_FORMATS = [
    ("email", "sam.o'neill+hermes@sub.acme-test.io"),  # the quote ends the match; the rest must go
    ("email", "JANE@ACME-TEST.COM"),
    ("phone", "+1 (415) 555-0132"),
    ("phone", "07700 900123"),
    ("phone", "07700900123"),
    ("phone", "(020) 7946 0958"),
    ("phone", "415-555-0132"),
    ("card-number", "5555555555554444"),  # Mastercard test card
    ("card-number", "3782 822463 10005"),  # Amex test card
    ("card-number", "6011-1111-1111-1117"),  # Discover test card
    ("card-number", "2223003122003222"),  # Mastercard 2-series test card
    ("iban", "DE89370400440532013000"),
    ("uk-ni-number", "AB123456C"),
]


@pytest.mark.parametrize("kind", list(fake_pii()))
def test_every_kind_of_personal_data_is_redacted_before_telnyx(gw: Gateway, kind: str) -> None:
    value = fake_pii()[kind]
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": f"Update the customer record: {value}."}))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, value)
    assert sent.json["messages"][-1]["content"] == f"Update the customer record: [redacted:{kind}]."
    assert gw.rows()[-1]["redactions"] == 1


@pytest.mark.parametrize(("kind", "value"), PII_FORMATS, ids=[v for _, v in PII_FORMATS])
def test_common_personal_data_formats_are_redacted(gw: Gateway, kind: str, value: str) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": f"Contact: {value}"}))

    [sent] = gw.upstreams.telnyx
    secret = value.split("'")[-1] if "'" in value else value
    assert_absent(sent, secret)
    assert f"[redacted:{kind}]" in sent.json["messages"][-1]["content"]


@pytest.mark.parametrize(("text", "kind", "value"), [
    ("jane@company.md", "email", "jane@company.md"),  # .md is Moldova, not just Markdown
    ("jane@company.py", "email", "jane@company.py"),  # .py is Paraguay
    ("IBAN AT611904300234573201 EUR", "iban", "AT611904300234573201"),
    ("IBAN GB82 WEST 1234 5698 7654 32 GBP", "iban", "GB82 WEST 1234 5698 7654 32"),
    ("card 4111 1111 1111 1111 123", "card-number", "4111 1111 1111 1111"),  # followed by its CVV
    ("ref 22 4111 1111 1111 1111", "card-number", "4111 1111 1111 1111"),  # preceded by another number
    ("call +44 20 7946 0958 12345", "phone", "+44 20 7946 0958"),
], ids=["moldova", "paraguay", "iban-currency", "iban-spaced-currency", "card-cvv", "card-after-ref", "phone-then-ref"])
def test_personal_data_next_to_other_text_is_still_found(gw: Gateway, text: str, kind: str, value: str) -> None:
    """Codex review on PR #1: greedy matches and file-like email suffixes used to hide these."""
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": text}))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, value)
    assert sent.json["messages"][-1]["content"] == text.replace(value, f"[redacted:{kind}]")


def test_two_cards_in_a_row_are_both_redacted(gw: Gateway) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": "4111 1111 1111 1111 5555 5555 5555 4444"}))

    [sent] = gw.upstreams.telnyx
    assert sent.json["messages"][-1]["content"] == "[redacted:card-number] [redacted:card-number]"


def test_auto_personal_data_next_to_other_text_keeps_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("k3")
    auto(gw, {"role": "user", "content": "Why did the transfer to AT611904300234573201 EUR fail?"})

    assert gw.upstreams.telnyx == []
    assert work_rows(gw)[0]["route_source"] == "rule:pii"


def test_personal_data_is_redacted_everywhere_a_credential_would_be(gw: Gateway) -> None:
    p = fake_pii()
    gw.post("/telnyx/v1/chat/completions", chat(
        {"role": "system", "content": f"The user is Jane, {p['email']}."},
        {"role": "user", "content": [{"type": "text", "text": f"Ring me on {p['phone']}"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "query_db", "arguments": json.dumps(
                {"sql": f"SELECT * FROM customers WHERE ni = '{p['uk-ni-number']}'"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": f"card={p['card-number']}, iban={p['iban']}"},
        user=p["email"],
    ))

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, *(v for k, v in p.items() if k != "us-ssn"))
    assert gw.rows()[-1]["redactions"] == 6


def test_a_customer_export_read_by_a_tool_is_redacted_row_by_row(gw: Gateway) -> None:
    rows = ["name,email,phone"] + [f"Customer {i},customer{i}@acme-test.co.uk,07700 900{i:03d}" for i in range(20)]
    gw.post("/telnyx/v1/chat/completions", chat(
        {"role": "user", "content": "How many customers are there?"},
        {"role": "assistant", "content": None, "tool_calls": [read_call("c1", "exports/customers.csv")]},
        {"role": "tool", "tool_call_id": "c1", "content": "\n".join(rows)},
    ))

    [sent] = gw.upstreams.telnyx
    content = sent.json["messages"][-1]["content"]
    assert "@acme-test.co.uk" not in content and "07700 900" not in content
    assert content.count("[redacted:email]") == 20 and content.count("[redacted:phone]") == 20
    assert content.splitlines()[0] == "name,email,phone", "the shape of the data survives for the model"


@pytest.mark.parametrize("text", [
    "Contact user@example.com for help", "logo@2x.png", "icon@1.5x.md", "app.min@3x.webp",
    "git@github.com:0x5am5/hybrid-gateway.git",
    "npm i react@18.2.0 @types/node", "Co-Authored-By: Bot <12345+bot@users.noreply.github.com>",
    "from: noreply@acme-test.co.uk", "admin@localhost", "dev@app.test",
    "2026-10-30T12:00:00Z", "10.0.0.1:5432", "v1.12.3", "ts=1730246400000", "id 0123456789", "@@ -10,7 +20,8 @@",
    "4111111111111112", "GB82 WEST 1234 5698 7654 33 GBP", "550e8400-e29b-41d4-a716-446655440000",
    "GB82 WEST 1234 5698 7654 33", "000-12-3456",
    "QQ 12 34 56 C", "port 8787 and 65535", "total 1,234,567.89", "sha 3f2a9c1d4e5b6a7f0c1d",
])
def test_lookalikes_are_not_treated_as_personal_data(gw: Gateway, text: str) -> None:
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": text}))

    [sent] = gw.upstreams.telnyx
    assert sent.json["messages"] == [{"role": "user", "content": text}]


def test_personal_data_never_reaches_the_log_database_or_stderr(gw: Gateway,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    p = fake_pii()
    gw.post("/telnyx/v1/chat/completions", chat({"role": "user", "content": f"{p['email']} {p['phone']}"}))

    err = capsys.readouterr().err
    assert "redacted 2 private value(s) before sending to Telnyx: email, phone" in err
    for value in (p["email"], p["phone"]):
        assert value not in err and value not in gw.db_dump()


def test_auto_personal_data_keeps_the_request_on_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("k3")
    value = fake_pii()["email"]
    r = auto(gw, {"role": "user", "content": f"Draft a reply to {value} about their refund"})

    assert r.status_code == 200
    assert gw.upstreams.telnyx == [] and gw.upstreams.router == []
    [sent] = gw.upstreams.local
    assert value in sent.raw.decode(), "the Mac sees the real conversation"
    row = gw.rows()[-1]
    assert (row["route"], row["route_source"]) == ("mac", "rule:pii")


def test_auto_personal_data_in_a_tool_result_pulls_a_cloud_loop_back_to_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("flash", "feature")
    system = {"role": "system", "content": "You are Hermes."}
    user = {"role": "user", "content": "Find the customer whose order failed"}
    auto(gw, system, user)
    auto(gw, system, user,
         {"role": "assistant", "content": None, "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "sql", "arguments": '{"q": "SELECT 1"}'}}]},
         {"role": "tool", "tool_call_id": "c1", "content": f"phone: {fake_pii()['phone']}"})

    assert len(gw.upstreams.telnyx) == 1
    assert [r["route_source"] for r in work_rows(gw)] == ["router", "rule:pii"]


def test_auto_personal_data_keeps_later_turns_on_the_mac(gw: Gateway) -> None:
    gw.upstreams.router_reply = router_says("k3", "refactor")
    history = [{"role": "user", "content": f"My card is {fake_pii()['card-number']}, why was it declined?"},
               {"role": "assistant", "content": "Let me look."}]
    auto(gw, *history, {"role": "user", "content": "Now refactor the whole payments module"})

    assert gw.upstreams.telnyx == []
    assert work_rows(gw)[0]["route_source"] == "rule:pii"


def test_auto_credentials_take_precedence_over_personal_data(gw: Gateway) -> None:
    auto(gw, {"role": "user", "content": f"{fake_pii()['email']} {fake_secrets()['aws-key']}"})

    assert work_rows(gw)[0]["route_source"] == "rule:secret"


def test_auto_personal_data_in_the_system_prompt_alone_does_not_pin_the_mac(gw: Gateway) -> None:
    """Like credentials, the agent's own system prompt doesn't decide the route, but it is still redacted."""
    gw.upstreams.router_reply = router_says("flash", "feature")
    email = fake_pii()["email"]
    auto(gw, {"role": "system", "content": f"The user's email is {email}."}, {"role": "user", "content": "Add a test"})

    [sent] = gw.upstreams.telnyx
    assert_absent(sent, email)
    assert work_rows(gw)[0]["route_source"] == "router"


@pytest.mark.parametrize(("size", "model"), [("medium", FLASH_MODEL), ("large", K3_MODEL)])
def test_auto_personal_data_too_big_for_the_mac_goes_to_the_cloud_redacted(sized_gw: Gateway, size: str,
                                                                          model: str) -> None:
    p = fake_pii()
    sized_gw.upstreams.router_reply = router_says("mac")
    leak = f"Customer {p['email']}, {p['phone']}, card {p['card-number']}"
    r = auto(sized_gw, *sized(SIZES[size], "Summarise this support thread.", leak))

    assert r.status_code == 200 and sized_gw.upstreams.local == []
    [sent] = sized_gw.upstreams.telnyx
    assert sent.json["model"] == model
    assert_absent(sent, p["email"], p["phone"], p["card-number"])
    assert gateway.REDACTION_NOTE in sent.json["messages"][0]["content"]
    assert work_rows(sized_gw)[0]["redactions"] == 3


def test_small_request_with_personal_data_stays_on_the_mac(sized_gw: Gateway) -> None:
    sized_gw.upstreams.router_reply = router_says("flash")
    auto(sized_gw, *sized(SIZES["small"], "Who is this?", fake_pii()["phone"]))

    assert sized_gw.upstreams.telnyx == []
    assert work_rows(sized_gw)[0]["route_source"] == "rule:pii"


def test_scan_and_pack_cover_personal_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from argparse import Namespace

    p = fake_pii()
    root = tmp_path / "proj"
    root.mkdir()
    (root / "customers.csv").write_text(f"name,email\nJane,{p['email']}\n")
    (root / "main.py").write_text("print('hello')\n")

    assert gateway.cmd_scan(Namespace(paths=[str(root)])) == 1
    out = capsys.readouterr()
    assert "customers.csv:2  email" in out.out and "main.py" not in out.out
    assert p["email"] not in out.out + out.err

    packed = tmp_path / "out.packed.txt"
    gateway.cmd_pack(Namespace(path=str(root), output=str(packed), limit=1_000_000, no_redact=False))
    assert p["email"] not in packed.read_text() and "Jane,[redacted:email]" in packed.read_text()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-p", "no:cacheprovider", "-W", "ignore::pytest.PytestAssertRewriteWarning",
                          *sys.argv[1:]]))
