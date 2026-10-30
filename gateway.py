# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "httpx>=0.27",
#   "tiktoken>=0.8",
#   "pathspec>=0.12",
#   "keyring>=25",
# ]
# ///
"""Hybrid Gateway: a local metering proxy between Hermes and its model backends.

Usage:
    uv run gateway.py serve          # proxy + dashboard on 127.0.0.1:8787
    uv run gateway.py route "<text>" # dry-run the /auto router
    uv run gateway.py pack <path>    # pack a repo into one text file, print tokens
    uv run gateway.py scan <path>    # list files and lines that look like secrets or personal data
"""

import argparse
import csv
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

HERE = Path(__file__).resolve().parent

Backend = Literal["local", "telnyx"]
JSON = dict[str, Any]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


STARTUP_ENV = ("HYBRID_GATEWAY_PORT", "HYBRID_GATEWAY_DB", "HYBRID_GATEWAY_PRICES", "HYBRID_GATEWAY_SETTINGS",
               "HYBRID_GATEWAY_KEYCHAIN_SERVICE")
EnvOrigin = Literal["shell", "file"]


def project_path(value: str) -> Path:
    """A path from env with `~` expanded; a relative one is taken from the folder holding gateway.py, not the cwd."""
    return (HERE / Path(value.strip()).expanduser()).resolve()


@dataclass(frozen=True)
class Env:
    """The shell environment over the env file (.env by default), merged once at startup. os.environ is never changed."""

    values: dict[str, str] = field(repr=False)
    origin: dict[str, EnvOrigin]
    file: Path
    file_loaded: bool

    def get(self, name: str) -> str | None:
        """The variable's value, or None if it is unset or blank."""
        value = self.values.get(name, "")
        return value if value.strip() else None

    def describe(self, name: str) -> str:
        """The variable's name, plus the env file if the value came from there. Never the value."""
        return f"{name} in {self.file}" if self.origin.get(name) == "file" else name


def read_env_file(path: Path) -> tuple[dict[str, str], bool]:
    """Parse KEY=VALUE lines from the env file; returns (known non-empty values, loaded).

    Allows # comment lines, blank lines, a leading `export ` and one matching pair of quotes. No interpolation.
    Warnings name the line or key, never a value.
    """
    import re

    if not path.is_file():
        return {}, False
    try:
        mode = path.stat().st_mode
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"hybrid-gateway: could not read {path} ({type(exc).__name__})", file=sys.stderr)
        return {}, False
    if mode & 0o077:
        print(f"hybrid-gateway: {path} can be read by other users and may hold keys; run: chmod 600 {path}",
              file=sys.stderr)
    known = {*STARTUP_ENV, *(name for spec in FIELDS for name in spec.env)}
    out: dict[str, str] = {}
    unknown: set[str] = set()
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = re.sub(r"^export\s+", "", line).partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            print(f"hybrid-gateway: ignoring line {n} of {path}: expected KEY=VALUE", file=sys.stderr)
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key not in known:
            unknown.add(key)
        elif value.strip():  # an empty value counts as unset
            out[key] = value
    if unknown:
        print(f"hybrid-gateway: ignoring unknown keys in {path}: {', '.join(sorted(unknown))}", file=sys.stderr)
    return out, True


def env_file_path(shell: dict[str, str]) -> Path:
    """HYBRID_GATEWAY_ENV_FILE from the shell, else .env next to gateway.py."""
    raw = shell.get("HYBRID_GATEWAY_ENV_FILE", "")
    return project_path(raw) if raw.strip() else (HERE / ".env").resolve()


def load_env(environ: dict[str, str] | None = None, quiet: bool = False) -> Env:
    """Merge the shell over the env file (HYBRID_GATEWAY_ENV_FILE, shell only). A blank shell value counts as unset."""
    shell = dict(os.environ if environ is None else environ)
    path = env_file_path(shell)
    from_file, loaded = read_env_file(path)
    from_shell = {k: v for k, v in shell.items() if v.strip()}
    origin: dict[str, EnvOrigin] = {**{k: "file" for k in from_file}, **{k: "shell" for k in from_shell}}
    if not quiet:
        print(f"hybrid-gateway: env file {path} {'loaded' if loaded else 'not found'}")
    return Env({**shell, **from_file, **from_shell}, origin, path, loaded)


@dataclass(frozen=True)
class Startup:
    """Settings read once at startup (shell > env file > default); changing these needs a restart."""

    host: str
    port: int
    db_path: Path
    prices_path: Path
    settings_path: Path
    keychain_service: str
    env_file: Path
    env_file_loaded: bool

    @classmethod
    def from_env(cls, env: Env) -> "Startup":
        """Read HYBRID_GATEWAY_PORT / _DB / _PRICES / _SETTINGS / _KEYCHAIN_SERVICE."""
        port = (env.get("HYBRID_GATEWAY_PORT") or "8787").strip()
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError(f"{env.describe('HYBRID_GATEWAY_PORT')} must be a port number")

        def path(name: str, default: str) -> Path:
            return project_path(env.get(name) or default)

        return cls(
            host="127.0.0.1",  # never bind wider than loopback
            port=int(port),
            db_path=path("HYBRID_GATEWAY_DB", "gateway.db"),
            prices_path=path("HYBRID_GATEWAY_PRICES", "prices.json"),
            settings_path=path("HYBRID_GATEWAY_SETTINGS", "settings.json"),
            keychain_service=(env.get("HYBRID_GATEWAY_KEYCHAIN_SERVICE") or "hybrid-gateway").strip(),
            env_file=env.file,
            env_file_loaded=env.file_loaded,
        )


@dataclass(frozen=True)
class Config:
    """Live settings, resolved app (settings.json, Keychain) > env > default. Rebuilt on every save."""

    local_url: str
    local_api_key: str | None = field(repr=False)
    local_model: str | None
    router_model: str
    router_timeout: float
    local_context: int
    remote_url: str
    remote_api_key: str | None = field(repr=False)
    flash_model: str
    k3_model: str
    flash_context: int
    gateway_token: str | None = field(repr=False)
    read_timeout: float


# ---------------------------------------------------------------------------
# Settings: app (settings.json / Keychain) > shell env > env file > default
# ---------------------------------------------------------------------------

FieldType = Literal["url", "secret", "model", "seconds", "tokens"]
Value = str | int | float
Source = Literal["env", "app", "default"]


@dataclass(frozen=True)
class FieldSpec:
    """One setting the dashboard can edit, and the env vars it falls back to."""

    key: str
    group: Literal["local", "remote", "proxy"]
    label: str
    type: FieldType
    env: tuple[str, ...]  # primary name first, then legacy aliases
    default: Value | None
    help: str
    bounds: tuple[float, float] = (0, 0)  # seconds and tokens only

    @property
    def secret(self) -> bool:
        """Secrets live in the Keychain and are never returned."""
        return self.type == "secret"


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("local_url", "local", "Server URL", "url", ("HYBRID_GATEWAY_LOCAL_URL", "LLAMA_SERVER_URL"),
              "http://127.0.0.1:8080/v1", "OpenAI-compatible base URL of the local server, ending in /v1."),
    FieldSpec("local_api_key", "local", "API key", "secret", ("HYBRID_GATEWAY_LOCAL_API_KEY", "LLAMA_SERVER_API_KEY"),
              None, "Replaces the client's auth on /local requests; leave unset if the server needs none."),
    FieldSpec("local_model", "local", "Mac model", "model",
              ("HYBRID_GATEWAY_LOCAL_MODEL",), None,
              "The model /auto uses for the Mac route; /auto returns 503 until it is set."),
    FieldSpec("router_model", "local", "Router model", "model", ("HYBRID_GATEWAY_ROUTER_MODEL",),
              "gemma-4-e2b-it-4bit", "Small model on the local server that picks a route for each new user turn."),
    FieldSpec("router_timeout", "local", "Router timeout", "seconds", ("HYBRID_GATEWAY_ROUTER_TIMEOUT",), 8.0,
              "After this long the request falls back to the Mac, or to its current route.", (0.5, 120)),
    FieldSpec("local_context", "local", "Mac context", "tokens", ("HYBRID_GATEWAY_LOCAL_CONTEXT",), 64_000,
              "Requests bigger than this never run on the Mac.", (1000, 10_000_000)),
    FieldSpec("remote_url", "remote", "Base URL", "url", ("HYBRID_GATEWAY_REMOTE_URL", "TELNYX_BASE_URL"),
              "https://api.telnyx.com/v2/ai/openai", "OpenAI-compatible base URL for Telnyx Inference."),
    FieldSpec("remote_api_key", "remote", "API key", "secret", ("TELNYX_API_KEY",), None,
              "Your Telnyx key; it is kept in the Keychain and never shown again."),
    FieldSpec("flash_model", "remote", "Flash model", "model", ("HYBRID_GATEWAY_FLASH_MODEL",),
              "zai-org/GLM-5.3-Flash", "The cheap, fast cloud tier."),
    FieldSpec("k3_model", "remote", "K3 model", "model", ("HYBRID_GATEWAY_K3_MODEL",),
              "moonshotai/Kimi-K3", "The strongest cloud tier, for hard or large work."),
    FieldSpec("flash_context", "remote", "Flash context", "tokens", ("HYBRID_GATEWAY_FLASH_CONTEXT",), 200_000,
              "Requests bigger than this always go to the K3 tier; match the Flash model's real window.",
              (1000, 10_000_000)),
    FieldSpec("gateway_token", "proxy", "Hybrid Gateway token", "secret", ("HYBRID_GATEWAY_TOKEN",), None,
              "Shared secret Hermes sends to /telnyx and /auto."),
    FieldSpec("read_timeout", "proxy", "Read timeout", "seconds", ("HYBRID_GATEWAY_READ_TIMEOUT",), 900.0,
              "Longest wait between upstream bytes before a request fails.", (10, 3600)),
)
FIELD_BY_KEY = {f.key: f for f in FIELDS}
PRICE_FIELDS = ("label", "input", "cached", "output")


def coerce(spec: FieldSpec, raw: object) -> Value:
    """Validate and normalise one setting value. Raises TypeError/ValueError; messages never echo the value."""
    if spec.type in ("seconds", "tokens"):
        try:
            if isinstance(raw, bool) or not isinstance(raw, int | float | str):
                raise TypeError
            num: float = float(raw)
            if spec.type == "tokens":
                if not num.is_integer():
                    raise ValueError
                num = int(num)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{spec.key} must be a number of {spec.type}") from None
        lo, hi = spec.bounds
        if not lo <= num <= hi:
            raise ValueError(f"{spec.key} must be between {lo:g} and {hi:g} {spec.type}")
        return num
    if not isinstance(raw, str):
        raise TypeError(f"{spec.key} must be a string")
    text = raw.strip()
    if spec.type == "url":
        from urllib.parse import urlsplit

        try:
            parts = urlsplit(text)
            ok = parts.scheme in ("http", "https") and bool(parts.hostname) and len(text) <= 2000
        except ValueError:
            ok = False
        if not ok or any(c.isspace() for c in text):
            raise ValueError(f"{spec.key} must be an http(s) URL with a host")
        return text.rstrip("/")
    if spec.type == "model":
        if not 1 <= len(text) <= 200 or any(c.isspace() for c in text):
            raise ValueError(f"{spec.key} must be 1–200 characters with no spaces")
        return text
    if not 1 <= len(text) <= 4096 or not text.isprintable() or any(c.isspace() for c in text):
        raise ValueError(f"{spec.key} must be 1–4096 printable characters with no spaces")
    return text


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then rename it over the target."""
    import stat
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def keychain_usable() -> bool:
    """True if keyring has a real backend (the macOS Keychain here)."""
    try:
        import keyring

        return float(keyring.get_keyring().priority) > 0
    except Exception:  # noqa: BLE001 - any keyring failure just means no Keychain
        return False


class SettingsError(Exception):
    """A rejected settings change: HTTP status, message, and the field it belongs to (a setting key or
    "prices.<tier>[.<name>]"), so the dashboard can show it inline."""

    def __init__(self, status: int, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.field = field


@dataclass(frozen=True)
class Resolved:
    """A setting's current value and where it came from."""

    value: Value | None
    source: Source


@dataclass(frozen=True)
class EnvValue:
    """A setting's env value (validated at startup), the variable that set it and where that variable lives."""

    name: str
    origin: EnvOrigin
    value: Value = field(repr=False)


@dataclass
class PricesEdit:
    """Tier price changes from the dashboard: tier ('flash' | 'k3') -> the fields given."""

    verified: str | None = None
    tiers: dict[str, dict[str, str | float]] = field(default_factory=dict)


class Settings:
    """Merges app (settings.json, Keychain) > env (shell, then env file) > default and caches the resulting Config."""

    def __init__(self, startup: Startup, env: Env) -> None:
        self.startup = startup
        self._lock = threading.Lock()
        self.keychain = keychain_usable()
        self._env = self._read_env(env)  # raises ValueError for a bad env value; fixed for the process
        self._app = self._read_file()
        self._secrets = {f.key: self._keychain_get(f.key) for f in FIELDS if f.secret}
        self._resolved = self._resolve(self._app)
        self.config = self._build()

    # -- reading -------------------------------------------------------------

    @staticmethod
    def _read_env(env: Env) -> dict[str, EnvValue]:
        out: dict[str, EnvValue] = {}
        for spec in FIELDS:
            name = next((n for n in spec.env if env.get(n)), None)
            if name is None:
                continue
            try:
                value = coerce(spec, env.values[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{env.describe(name)}: {exc}") from None
            out[spec.key] = EnvValue(name, env.origin.get(name, "shell"), value)
        return out

    def _read_file(self) -> dict[str, Value]:
        path = self.startup.settings_path
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            print(f"hybrid-gateway: ignoring {path}: {type(exc).__name__}", file=sys.stderr)
            return {}
        out: dict[str, Value] = {}
        for key, raw in (data.items() if isinstance(data, dict) else []):
            spec = FIELD_BY_KEY.get(key)
            if spec is None or spec.secret:
                continue
            try:
                out[key] = coerce(spec, raw)
            except (TypeError, ValueError) as exc:
                print(f"hybrid-gateway: ignoring {key} in {path.name}: {exc}", file=sys.stderr)
        return out

    def _keychain_get(self, key: str) -> str | None:
        if not self.keychain:
            return None
        try:
            import keyring

            value = keyring.get_password(self.startup.keychain_service, key)
            return value if isinstance(value, str) and value else None
        except Exception as exc:  # noqa: BLE001 - treat an unreadable entry as unset
            print(f"hybrid-gateway: could not read {key} from the Keychain ({type(exc).__name__})", file=sys.stderr)
            return None

    def _resolve(self, app: dict[str, Value]) -> dict[str, Resolved]:
        out: dict[str, Resolved] = {}
        for spec in FIELDS:
            mine = self._secrets.get(spec.key) if spec.secret else app.get(spec.key)
            env = self._env.get(spec.key)
            if mine is not None:
                out[spec.key] = Resolved(mine, "app")
            elif env is not None:
                out[spec.key] = Resolved(env.value, "env")
            else:
                out[spec.key] = Resolved(spec.default, "default")
        return out

    def _build(self) -> Config:
        r = self._resolved

        def text(key: str) -> str | None:
            value = r[key].value
            return value if isinstance(value, str) else None

        def num(key: str) -> float:
            value = r[key].value
            assert isinstance(value, int | float)
            return value

        def required(key: str) -> str:
            value = text(key)
            assert value is not None
            return value

        return Config(
            local_url=required("local_url"), local_api_key=text("local_api_key"), local_model=text("local_model"),
            router_model=required("router_model"), router_timeout=float(num("router_timeout")),
            local_context=int(num("local_context")), remote_url=required("remote_url"),
            remote_api_key=text("remote_api_key"), flash_model=required("flash_model"),
            k3_model=required("k3_model"), flash_context=int(num("flash_context")),
            gateway_token=text("gateway_token"), read_timeout=float(num("read_timeout")),
        )

    def view(self) -> JSON:
        """GET /api/settings: every field (secrets as set/unset only), tier prices and startup paths."""
        fields: list[JSON] = []
        for spec in FIELDS:
            res = self._resolved[spec.key]
            env = self._env.get(spec.key)
            item: JSON = {
                "key": spec.key, "group": spec.group, "label": spec.label, "type": spec.type,
                "value": None if spec.secret else res.value, "default": spec.default, "source": res.source,
                "env": spec.env[0], "env_set_by": env.name if env else None,
                "env_origin": env.origin if env else None,
                "env_value": None if spec.secret or env is None else env.value, "help": spec.help,
            }
            if spec.secret:
                item["set"] = res.value is not None
                item["env_set"] = env is not None
            fields.append(item)
        prices = load_prices(self.startup.prices_path)
        cfg = self.config
        tiers: JSON = {"verified": prices.get("verified")}
        for tier, model in (("flash", cfg.flash_model), ("k3", cfg.k3_model)):
            rates = price_for(prices, model) or {}
            tiers[tier] = {"model": model, "label": tier_label(prices, model),
                           **{k: rates.get(k) for k in PRICE_FIELDS[1:]}}
        s = self.startup
        return {
            "keychain": self.keychain,
            "fields": fields,
            "prices": tiers,
            "startup": {"port": s.port, "db": str(s.db_path), "prices": str(s.prices_path),
                        "settings": str(s.settings_path), "keychain_service": s.keychain_service,
                        "env_file": str(s.env_file), "env_file_loaded": s.env_file_loaded},
        }

    # -- writing -------------------------------------------------------------

    def apply(self, values: dict[str, object], clear: list[str], prices: PricesEdit | None = None) -> None:
        """Validate a change in full, then write Keychain, settings.json and prices.json. Raises SettingsError."""
        with self._lock:
            for key in [*values, *clear]:
                if key not in FIELD_BY_KEY:
                    raise SettingsError(422, f"unknown setting: {key[:60]}", key[:60])
            if both := set(values) & set(clear):
                raise SettingsError(422, f"{min(both)} is in both values and clear", min(both))

            app = dict(self._app)
            put: dict[str, str] = {}
            bad: dict[str, str] = {}  # every invalid value, reported together
            for key, raw in values.items():
                spec = FIELD_BY_KEY[key]
                if spec.secret and raw in (None, ""):
                    continue  # secrets are never sent back, so a blank one means "unchanged"
                if raw is None:
                    app.pop(key, None)
                    continue
                try:
                    value = coerce(spec, raw)
                except (TypeError, ValueError) as exc:
                    bad[key] = str(exc)
                    continue
                if spec.secret:
                    put[key] = str(value)
                else:
                    app[key] = value
            if bad:
                raise SettingsError(422, "; ".join(bad.values()), next(iter(bad)))
            drop = [k for k in clear if FIELD_BY_KEY[k].secret]
            for key in clear:
                app.pop(key, None)
            if (put or drop) and not self.keychain:
                raise SettingsError(503, "the Keychain is not available, so secrets cannot be saved", [*put, *drop][0])

            after = self._resolve(app)
            new_prices = self._edit_prices(prices, str(after["flash_model"].value), str(after["k3_model"].value)) \
                if prices else None

            try:
                self._keychain_write(put, drop)
                if app != self._app:
                    text = json.dumps({f.key: app[f.key] for f in FIELDS if f.key in app}, indent=2) + "\n"
                    path = self.startup.settings_path
                    try:
                        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        write_atomic(path, text)
                    except OSError as exc:
                        raise SettingsError(500, f"could not write settings.json ({type(exc).__name__})") from None
                    self._app = app
                if new_prices is not None:
                    try:
                        write_atomic(self.startup.prices_path, json.dumps(new_prices, indent=2) + "\n")
                    except OSError as exc:
                        raise SettingsError(500, f"could not write prices.json ({type(exc).__name__})") from None
            finally:
                self._resolved = self._resolve(self._app)
                self.config = self._build()

    def _keychain_write(self, put: dict[str, str], drop: list[str]) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError

        service = self.startup.keychain_service
        for key, secret in put.items():
            try:
                keyring.set_password(service, key, secret)
            except Exception as exc:  # noqa: BLE001 - report the type only, never the value
                raise SettingsError(503, f"could not save {key} to the Keychain ({type(exc).__name__})", key) from None
            self._secrets[key] = secret
        for key in drop:
            try:
                keyring.delete_password(service, key)
            except PasswordDeleteError:
                pass  # already gone
            except Exception as exc:  # noqa: BLE001
                raise SettingsError(503, f"could not remove {key} from the Keychain ({type(exc).__name__})",
                                    key) from None
            self._secrets[key] = None

    def _edit_prices(self, edit: PricesEdit, flash_model: str, k3_model: str) -> JSON:
        path = self.startup.prices_path
        try:
            data = json.loads(path.read_text()) if path.exists() else \
                {"verified": None, "currency": "USD", "unit": "per 1M tokens", "models": {}}
        except (OSError, json.JSONDecodeError):
            raise SettingsError(500, f"{path.name} could not be read; fix it by hand first") from None
        models = data.setdefault("models", {}) if isinstance(data, dict) else None
        if not isinstance(models, dict):
            raise SettingsError(500, f"{path.name} has no models object; fix it by hand first")
        for tier, model in (("flash", flash_model), ("k3", k3_model)):
            given = edit.tiers.get(tier)
            if not given:
                continue
            name = next((n for n in models if n.lower() == model.lower()), None)
            if name is None or not isinstance(models[name], dict):
                if missing := [k for k in PRICE_FIELDS if k not in given]:
                    raise SettingsError(422, f"{tier} prices for a new model need {', '.join(missing)}",
                                        f"prices.{tier}.{missing[0]}")
                name = name or model
                models[name] = {}
            models[name].update(given)
        if edit.verified:
            data["verified"] = edit.verified
        return data


# ---------------------------------------------------------------------------
# Storage (metadata only: never prompt or response text)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY,
    label       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT
);
CREATE TABLE IF NOT EXISTS requests (
    id                 INTEGER PRIMARY KEY,
    ts                 TEXT NOT NULL,
    backend            TEXT NOT NULL,
    model              TEXT,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    cached_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    wall_ms            INTEGER NOT NULL,
    status             INTEGER NOT NULL,
    job_id             INTEGER REFERENCES jobs(id),
    job_label          TEXT NOT NULL DEFAULT 'untagged',
    token_source       TEXT NOT NULL DEFAULT 'usage'  -- 'usage' | 'tiktoken' | 'none'
);
CREATE INDEX IF NOT EXISTS requests_job ON requests(job_id);
"""

# Columns added after the first release; ALTERed onto databases created before them.
ADDED_COLUMNS = {
    "purpose": "TEXT NOT NULL DEFAULT 'work'",  # 'work' | 'router'
    "route": "TEXT",  # 'mac' | 'flash' | 'k3' for requests that came in on /auto
    "route_source": "TEXT",  # 'router' | 'rule:secret' | 'rule:pii' | 'rule:context' | 'rule:tool-loop' | 'fallback'
    "route_category": "TEXT",  # fixed list from the router, never free text
    "redactions": "INTEGER NOT NULL DEFAULT 0",  # credentials and personal data replaced before going to the cloud
}


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Store:
    """Thread-safe wrapper around the SQLite log."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)  # e.g. HYBRID_GATEWAY_DB=data/gateway.db
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(requests)")}
        for name, decl in ADDED_COLUMNS.items():
            if name not in existing:
                self._db.execute(f"ALTER TABLE requests ADD COLUMN {name} {decl}")

    def query(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Run a read query and return all rows."""
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def execute(self, sql: str, args: tuple[Any, ...] = ()) -> None:
        """Run a write statement."""
        with self._lock:
            self._db.execute(sql, args)

    def open_job(self) -> sqlite3.Row | None:
        """The currently open job, if any."""
        rows = self.query("SELECT id, label, started_at FROM jobs WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1")
        return rows[0] if rows else None

    def start_job(self, label: str) -> None:
        """Close any open job and open a new one."""
        ts = now_iso()
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("UPDATE jobs SET ended_at = ? WHERE ended_at IS NULL", (ts,))
            self._db.execute("INSERT INTO jobs (label, started_at) VALUES (?, ?)", (label, ts))
            self._db.execute("COMMIT")

    def end_job(self) -> None:
        """Close the open job, if any."""
        self.execute("UPDATE jobs SET ended_at = ? WHERE ended_at IS NULL", (now_iso(),))

    def reset(self) -> None:
        """Delete every logged request and job."""
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM requests")
            self._db.execute("DELETE FROM jobs")
            self._db.execute("COMMIT")

    def log(self, rec: "RequestRecord") -> None:
        """Insert one request row, tagged with the open job."""
        job = self.open_job()
        self.execute(
            """INSERT INTO requests (ts, backend, model, prompt_tokens, cached_tokens, completion_tokens,
                                     wall_ms, status, job_id, job_label, token_source,
                                     purpose, route, route_source, route_category, redactions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.ts, rec.backend, rec.model, rec.prompt_tokens, rec.cached_tokens, rec.completion_tokens,
                rec.wall_ms, rec.status, job["id"] if job else None, job["label"] if job else "untagged",
                rec.token_source, rec.purpose, rec.route, rec.route_source, rec.route_category, rec.redactions,
            ),
        )


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def load_prices(path: Path) -> JSON:
    """Read prices.json fresh each call so edits apply without a restart."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"hybrid-gateway: could not read {path}: {exc}", file=sys.stderr)
        return {"verified": None, "models": {}}


def price_for(prices: JSON, model: str | None) -> JSON | None:
    """Look up a model's rates, case-insensitively."""
    if not model:
        return None
    for name, rates in prices.get("models", {}).items():
        if name.lower() == model.lower():
            return rates
    return None


def tier_label(prices: JSON, model: str) -> str:
    """A tier's display name from prices.json, falling back to its model id."""
    label = (price_for(prices, model) or {}).get("label")
    return label if isinstance(label, str) and label.strip() else model


def cost_usd(rates: JSON, prompt: int, cached: int, completion: int) -> float:
    """Uncached input x input rate + cached x cached rate + output x output rate, per 1M tokens."""
    uncached = max(prompt - cached, 0)
    return (uncached * rates["input"] + cached * rates["cached"] + completion * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Token estimation (fallback only)
# ---------------------------------------------------------------------------

_enc = None


def count_tokens(text: str) -> int:
    """Approximate token count with tiktoken's o200k_base encoding."""
    global _enc
    if _enc is None:
        import tiktoken

        _enc = tiktoken.get_encoding("o200k_base")
    return len(_enc.encode(text, disallowed_special=()))


def prompt_text(body: JSON) -> str:
    """Flatten chat messages (and tool schemas) into text for estimation."""
    parts: list[str] = []
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(msg, dict) and msg.get("tool_calls"):
            parts.append(json.dumps(msg["tool_calls"]))
    if body.get("tools"):
        parts.append(json.dumps(body["tools"]))
    if isinstance(body.get("prompt"), str):
        parts.append(body["prompt"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """Metadata for one proxied completion request."""

    backend: Backend
    model: str | None
    ts: str = field(default_factory=now_iso)
    started: float = field(default_factory=time.perf_counter)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    status: int = 0
    wall_ms: int = 0
    token_source: str = "none"
    purpose: str = "work"
    route: str | None = None
    route_source: str | None = None
    route_category: str | None = None
    redactions: int = 0

    def apply_usage(self, usage: JSON) -> None:
        """Copy token counts from an OpenAI-style usage object."""
        self.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.completion_tokens = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens = int(details.get("cached_tokens") or 0)
        self.token_source = "usage"

    def estimate(self, body: JSON, completion: str) -> None:
        """Fill token counts with tiktoken when the backend sent no usage."""
        try:
            self.prompt_tokens = count_tokens(prompt_text(body))
            self.completion_tokens = count_tokens(completion)
            self.token_source = "tiktoken"
        except Exception as exc:  # noqa: BLE001 - a tiktoken failure must never break the proxy
            print(f"hybrid-gateway: token estimate failed: {exc}", file=sys.stderr)

    def finish(self) -> None:
        """Stamp the wall time."""
        self.wall_ms = int((time.perf_counter() - self.started) * 1000)


HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade", "host", "content-length", "content-encoding", "accept-encoding",
}
METERED_PATHS = {"chat/completions", "completions"}


def upstream_headers(headers: Any, backend: Backend, cfg: Config) -> dict[str, str]:
    """Copy client headers minus hop-by-hop ones, swapping in the backend's own key where configured."""
    out = {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}
    out["accept-encoding"] = "identity"  # keep SSE bytes readable for usage parsing
    key = cfg.remote_api_key if backend == "telnyx" else cfg.local_api_key
    if key:
        out.pop("authorization", None)  # starlette lower-cases header names
        out["authorization"] = f"Bearer {key}"
    return out


def client_headers(headers: Any) -> dict[str, str]:
    """Upstream response headers safe to return to the client."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


class SSEUsageTap:
    """Watches SSE bytes as they pass through, picking out usage and output text."""

    def __init__(self) -> None:
        self._buf = b""
        self.usage: JSON | None = None
        self.model: str | None = None
        self._text: list[str] = []

    def feed(self, chunk: bytes) -> None:
        """Consume a raw chunk; complete lines are parsed, partial lines are kept."""
        self._buf += chunk
        *lines, self._buf = self._buf.split(b"\n")
        for line in lines:
            self._line(line.strip())

    def close(self) -> None:
        """Parse any trailing line without a newline."""
        if self._buf.strip():
            self._line(self._buf.strip())
        self._buf = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        self.model = event.get("model") or self.model
        if event.get("usage"):
            self.usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    self._text.append(delta[key])
            if delta.get("tool_calls"):
                self._text.append(json.dumps(delta["tool_calls"]))
            if isinstance(choice.get("text"), str):
                self._text.append(choice["text"])

    @property
    def text(self) -> str:
        """Output text seen so far (held in memory only, never stored)."""
        return "".join(self._text)


def response_text(payload: JSON) -> str:
    """Output text from a non-streaming completion, for estimation only."""
    parts: list[str] = []
    for choice in payload.get("choices") or []:
        msg = choice.get("message") or {}
        for key in ("content", "reasoning_content", "reasoning"):
            if isinstance(msg.get(key), str):
                parts.append(msg[key])
        if msg.get("tool_calls"):
            parts.append(json.dumps(msg["tool_calls"]))
        if isinstance(choice.get("text"), str):
            parts.append(choice["text"])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Router (/auto): choose mac, flash or k3 once per user turn
# ---------------------------------------------------------------------------

Route = Literal["mac", "flash", "k3"]
ROUTES: tuple[Route, ...] = ("mac", "flash", "k3")
ROUTE_CATEGORIES = ("chat", "question", "edit", "feature", "debug", "refactor", "research", "writing", "ops")


def router_prompt(flash: str, k3: str) -> str:
    """The router's system prompt, naming the cloud tiers by their price-list labels.

    With the default labels ("GLM-5.3 Flash", "Kimi K3") this is byte-identical to the tuned, evaluated prompt.
    """
    return f"""You are the router for a coding agent. Read the user's latest request and choose where the agent \
should run it.

mac: the local model on the user's Mac. Free and private, but slower and less capable. Use it for conversation, quick \
questions, explaining code or what an error message means, small edits to one or two files, running commands or \
tests and reporting the output, short summaries, simple scripts, and anything involving secrets or personal data.
flash: {flash} in the cloud. Cheap and fast. Use it for moderate work a small local model may get wrong: \
multi-file features, writing tests, substantial documentation, summarising long documents, and fixing a bug that \
comes with a clear error message or stack trace to follow.
k3: {k3} in the cloud. The strongest choice and about 20 times the price of flash. Use it only for hard or large \
work: architecture and system design, large refactors or migrations, security audits, long multi-step plans, very \
large context, and bugs with no clear error to follow, such as race conditions, intermittent or concurrency \
failures, memory leaks, stale or inconsistent data, and exactly-once or ordering guarantees.

When asked to fix a bug: a clear error to follow means flash; an unknown, intermittent or systemic cause means k3. \
Only explaining an error, or running something and reporting back, stays on mac.
Questions that only ask to explain or compare tools, commands or concepts stay on mac, even when they mention merges, \
migrations, refactors or architecture: judge by the work requested, not the topic.
Prefer mac when unsure. If the conversation already has a current route and the new message is a simple follow-up \
(thanks, yes, continue, a small tweak), keep the current route.
Reply with JSON only: {{"route": "mac" | "flash" | "k3", "category": one of """ + ", ".join(ROUTE_CATEGORIES) + "}"


ROUTER_SCHEMA: JSON = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(ROUTES)},
        "category": {"type": "string", "enum": list(ROUTE_CATEGORIES)},
    },
    "required": ["route", "category"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    """Where an /auto request runs, and why."""

    route: Route
    source: str  # 'router' | 'rule:secret' | 'rule:pii' | 'rule:context' | 'rule:tool-loop' | 'fallback'
    category: str | None = None
    router_ms: int | None = None


def message_text(msg: Any) -> str:
    """Plain text of one chat message (text parts only)."""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def has_images(body: JSON) -> bool:
    """True if any message carries an image part."""
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
            return True
    return False


def user_messages(body: JSON) -> list[JSON]:
    """The conversation's user messages, oldest first."""
    return [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]


def session_key(body: JSON) -> str | None:
    """Stable id for a conversation: a hash of its system prompt and first user message (never stored)."""
    import hashlib

    msgs = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    system = next((message_text(m) for m in msgs if m.get("role") in ("system", "developer")), "")
    first_user = next((message_text(m) for m in msgs if m.get("role") == "user"), None)
    if first_user is None:
        return None
    return hashlib.sha256(json.dumps([system, first_user]).encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Secrets: find credentials, keep them on the Mac, redact them from anything bound for the cloud
# ---------------------------------------------------------------------------

# Credential shapes by kind. A false positive keeps work on the Mac (or hides a harmless value from the cloud); a miss
# sends a secret to the cloud. Where a pattern has a "v" group, only that part is the secret.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple((kind, re.compile(rx)) for kind, rx in (
    ("url-credentials", r"\b[a-z][a-z0-9+.-]*://(?P<v>[^\s:/@]+:[^\s/@]+)@"),  # postgres://user:pass@host
    ("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
    ("stripe-key", r"\b(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{16,}"),
    ("anthropic-key", r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai-key", r"\bsk-[A-Za-z0-9_-]{20,}"),
    ("github-token", r"\b(?:ghp|gho|ghs|ghu|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    ("gitlab-token", r"\bglpat-[A-Za-z0-9_-]{20,}"),
    ("slack-token", r"\bxox[abposr]-[A-Za-z0-9-]{10,}|https://hooks\.slack\.com/services/[A-Za-z0-9/]{20,}"),
    ("aws-key", r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
    ("google-key", r"\bAIza[0-9A-Za-z_-]{35}"),
    ("telnyx-key", r"\bKEY0[0-9A-Za-z_]{20,}"),
    ("huggingface-token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("npm-token", r"\bnpm_[A-Za-z0-9]{36}\b"),
    ("sendgrid-key", r"\bSG\.[\w-]{16,}\.[\w-]{16,}"),
    ("resend-key", r"\bre_[A-Za-z0-9]{6,}_[A-Za-z0-9]{16,}"),
    ("sentry-token", r"\bsntry[su]_[A-Za-z0-9+/=_-]{20,}"),
    ("shopify-token", r"\bshp(?:at|ss|ca|pa)_[a-fA-F0-9]{32}"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ("bearer-token", r"\bBearer\s+(?P<v>[A-Za-z0-9._~+/-]{20,}=*)"),
    ("basic-auth", r"(?i)\bauthorization:\s*basic\s+(?P<v>[A-Za-z0-9+/]{12,}={0,2})"),
))

# Generic `name = value` credentials (env, JSON, YAML, TOML, code), in the style of gitleaks' generic-api-key rule.
# The match starts at the keyword; the value is checked by credential_value() before it counts.
ASSIGNMENT_RE = re.compile(
    r"(?i)(?:secret|token|passw(?:or)?d|pwd|api[_.-]?key|apikey|credentials?|creds|[_.-]key)[\w.-]{0,20}?"
    r"[\\'\"`]{0,3}[ \t]*(?::=|=>|=|:)[ \t]*[\\'\"`]{0,3}(?P<v>[^\s\"'`\\,;<>(){}\[\]]{6,300}+)(?!\()"
)
NAME_CHARS = re.compile(r"[\w.-]*$")
SAFE_NAME_RE = re.compile(
    r"(?i)(?:public|primary|foreign|sort|partition|cache|idempotency|unique|composite|row|object|hot|max|min|num"
    r"|total|input|output|prompt|completion|cached|next|page|csrf)[_.-]?(?:key|tokens?)$"
    r"|(?:key|token|secret|password)s?[_.-]?(?:ids?|names?|path|file|size|length|type|url|uri|endpoint|count|limit"
    r"|hash|field|ref|policy|reset|env|var|header|source|expires?|expiry|ttl)$"
)
PASSWORD_NAME_RE = re.compile(r"(?i)passw|pwd|secret")
IDENTIFIER_RE = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+|[a-z]+(?:_[a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+")
PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:x{3,}.*|\*{3,}|\.{3,}.*|<[^>]*>?|your[_-].*|changeme|change[_-]me|example.*|dummy.*|sample.*"
    r"|placeholder.*|redacted.*|\[redacted.*|null|none|nil|true|false|undefined|todo|required|optional|string|str"
    r"|number|boolean|secret|password|token)$"
)
CODE_REF_RE = re.compile(r"^(?:\$|%\(|\.{0,2}/|~/|(?:process|import|os|env|self|this|cfg|config|settings|args|opts"
                         r"|options|ctx|context|req|request|props|params)\.)|://")


def shannon_entropy(text: str) -> float:
    """Bits per character; random keys score 4-6, words and repeated text far less."""
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values()) if n else 0.0


def credential_value(name: str, value: str) -> bool:
    """Whether `name = value` looks like a real credential rather than a placeholder, code or a setting."""
    if SAFE_NAME_RE.search(name) or PLACEHOLDER_RE.match(value) or CODE_REF_RE.search(value):
        return False
    if IDENTIFIER_RE.fullmatch(value):  # f.type, settings.token, has_secret
        return False
    if PASSWORD_NAME_RE.search(name):  # passwords can be plain words, but short words are usually prose
        return len(set(value)) >= 4 and (len(value) >= 16 or not re.fullmatch(r"[A-Za-z_.-]+", value))
    if len(value) < 8 or re.fullmatch(r"[A-Za-z_.-]+", value):  # identifiers, words and dotted names
        return False
    return shannon_entropy(value) >= 3.0


def iter_secrets(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield (start, end, kind) for each likely credential in text, unordered and possibly overlapping."""
    for kind, rx in SECRET_PATTERNS:
        for m in rx.finditer(text):
            group = "v" if "v" in rx.groupindex else 0
            yield m.start(group), m.end(group), kind
    for m in ASSIGNMENT_RE.finditer(text):
        prefix = NAME_CHARS.search(text, max(m.start() - 60, 0), m.start())
        name = (prefix.group() if prefix else "") + text[m.start():m.start("v")]
        if credential_value(name.strip("\\'\"`:=> \t"), m.group("v")):
            yield m.start("v"), m.end("v"), "credential"


# Personal data with a recognisable shape. Like credentials, matching favours precision: each kind has a checksum,
# a known prefix or a strict layout, and documentation-style values (example.com, @2x.png, git@github.com) are skipped.
EMAIL_RE = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"(?P<tld>[A-Za-z]{2,24})\b(?![@-])"
)
SAFE_EMAIL_DOMAIN_RE = re.compile(
    r"(?i)(?:^|\.)(?:example\.(?:com|org|net)|[\w-]+\.(?:example|test|invalid|localhost|local)"
    r"|users\.noreply\.github\.com)$"
)
SAFE_EMAIL_LOCAL_RE = re.compile(r"(?i)^(?:git|no-?reply|do-?not-?reply)$")
FILE_SUFFIXES = {"png", "jpg", "jpeg", "gif", "svg", "webp", "avif", "ico", "css", "scss", "js", "mjs", "cjs", "ts",
                 "tsx", "jsx", "json", "py", "md", "html", "txt", "tgz"}
URL_USERINFO_RE = re.compile(r"://[^\s/@]*$")
PHONE_RE = re.compile(
    r"(?<![\w+./-])(?:"
    r"\+\d{1,3}[ .-]?(?:\(0?\d{1,4}\)[ .-]?)?\d{1,4}(?:[ .-]?\d{2,4}){1,4}"  # +44 20 7946 0958, +1 (415) 555-0132
    r"|\(0\d{2,4}\)[ .-]?\d{3,4}[ .-]?\d{3,4}"  # (020) 7946 0958
    r"|0\d{2,4}[ -]\d{3,4}[ -]?\d{3,4}|07\d{9}"  # 020 7946 0958, 07700 900123, 07700900123
    r"|\(\d{3}\)[ .-]?\d{3}[ .-]\d{4}|\d{3}[.-]\d{3}[.-]\d{4}"  # (415) 555-0132, 415-555-0132
    r")(?!\w|[.-]\d)"
)
CARD_RE = re.compile(r"(?<![\w.-])[2-6](?:[ -]?\d){12,18}(?!\w|[.-]\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
UK_NI_RE = re.compile(r"\b(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b")
US_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum every payment card number carries."""
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def card_number(text: str) -> bool:
    """A Visa, Mastercard, Amex or Discover number: right prefix, right length and a valid Luhn checksum."""
    d = re.sub(r"[ -]", "", text)
    brand = (d[0] == "4" and len(d) in (13, 16, 19)) or (len(d) == 16 and (51 <= int(d[:2]) <= 55
                                                                         or 2221 <= int(d[:4]) <= 2720)) \
        or (len(d) == 15 and d[:2] in ("34", "37")) \
        or (16 <= len(d) <= 19 and (d.startswith(("6011", "65")) or 644 <= int(d[:3]) <= 649))
    return brand and luhn_valid(d)


def iban_valid(text: str) -> bool:
    """An IBAN with a valid ISO 13616 mod-97 checksum."""
    s = text.replace(" ", "")
    return 15 <= len(s) <= 34 and int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def phone_number(text: str) -> bool:
    """A phone number with a plausible digit count: 9-15 for international (E.164), 10-11 for national."""
    n = sum(c.isdigit() for c in text)
    return 9 <= n <= 15 if text.startswith("+") else 10 <= n <= 11


def iter_pii(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield (start, end, kind) for each likely piece of personal data in text."""
    for m in EMAIL_RE.finditer(text):
        local, _, domain = m.group().partition("@")
        if SAFE_EMAIL_DOMAIN_RE.search(domain) or SAFE_EMAIL_LOCAL_RE.match(local) \
                or m.group("tld").lower() in FILE_SUFFIXES or URL_USERINFO_RE.search(text, 0, m.start()):
            continue
        yield m.start(), m.end(), "email"
    for kind, rx, valid in (("phone", PHONE_RE, phone_number), ("card-number", CARD_RE, card_number),
                            ("iban", IBAN_RE, iban_valid), ("uk-ni-number", UK_NI_RE, None),
                            ("us-ssn", US_SSN_RE, None)):
        for m in rx.finditer(text):
            if valid is None or valid(m.group()):
                yield m.start(), m.end(), kind


def find_secrets(text: str, pii: bool = True) -> list[tuple[int, int, str]]:
    """Sorted, non-overlapping (start, end, kind) spans of likely credentials and, unless pii is False, personal
    data; overlaps keep the most specific kind."""
    found = [*iter_secrets(text), *(iter_pii(text) if pii else ())]
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in sorted(found, key=lambda s: (s[0], s[2] == "credential")):
        if merged and start < merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], end), prev[2])
        else:
            merged.append((start, end, kind))
    return merged


def redact_text(text: str) -> tuple[str, list[str]]:
    """Text with each credential and piece of personal data replaced by `[redacted:<kind>]`, and the kinds replaced."""
    spans = find_secrets(text)
    if not spans:
        return text, []
    out: list[str] = []
    last = 0
    for start, end, kind in spans:
        out += [text[last:start], f"[redacted:{kind}]"]
        last = end
    out.append(text[last:])
    return "".join(out), [kind for _, _, kind in spans]


# Files that hold secrets by convention. Templates such as .env.example are fine.
SECRET_FILE_RE = re.compile(
    r"(?i)^(?:\.env(?:\.(?!(?:example|sample|template|dist|defaults?)$)[\w.-]+)?|\.envrc|\.dev\.vars|\.netrc|_netrc"
    r"|\.npmrc|\.pypirc|\.pgpass|\.git-credentials|\.dockercfg|\.htpasswd|id_(?:rsa|dsa|ecdsa|ed25519)"
    r"|[\w.*-]+\.(?:pem|key|p12|pfx|jks|keystore|ppk|kdbx|tfvars|tfstate)|credentials(?:\.json|\.ya?ml)?"
    r"|secrets?(?:\.[\w-]+)?\.(?:json|ya?ml|toml|env|txt)|service[-_]account[\w.-]*\.json)$"
)
PATH_SPLIT_RE = re.compile(r"[\s'\"`;|&<>()=,]+")


def is_secret_file(path: str) -> bool:
    """True if a path's file name is one that conventionally holds secrets (.env, id_rsa, *.pem, ...)."""
    return bool(SECRET_FILE_RE.match(path.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]))


def tool_arguments(call: Any) -> Any:
    """A tool call's arguments, decoded from their JSON string when possible."""
    args = (call.get("function") or {}).get("arguments") if isinstance(call, dict) else None
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def strings_in(value: Any) -> Iterator[str]:
    """Every string nested in a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)


def secret_file_calls(body: JSON) -> dict[str, str]:
    """tool_call_id -> secrets file name, for each tool call whose arguments name one (read_file .env, cat id_rsa)."""
    found: dict[str, str] = {}
    for msg in body.get("messages") or []:
        for call in (msg.get("tool_calls") or []) if isinstance(msg, dict) else []:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            for text in strings_in(tool_arguments(call)):
                name = next((t for t in PATH_SPLIT_RE.split(text) if t and is_secret_file(t)), None)
                if name:
                    found[call["id"]] = name.replace("\\", "/").rsplit("/", 1)[-1]
                    break
    return found


def conversation_texts(body: JSON) -> Iterator[str]:
    """Text of every user, assistant and tool message, tool call arguments included. The system prompt and tool
    schemas are skipped: they are the agent's own text, and often show example URLs and addresses."""
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") in ("system", "developer"):
            continue
        yield message_text(msg)
        for call in msg.get("tool_calls") or []:
            yield from strings_in(tool_arguments(call))


def has_secret(body: JSON) -> bool:
    """True if a user, assistant or tool message looks like it holds a credential, or is the result of reading a
    secrets file."""
    files = secret_file_calls(body)
    if any(isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id") in files
           for m in body.get("messages") or []):
        return True
    return any(next(iter_secrets(t), None) for t in conversation_texts(body))


def has_pii(body: JSON) -> bool:
    """True if a user, assistant or tool message looks like it holds personal data (an email address, phone number,
    card number, IBAN, NI number or SSN)."""
    return any(next(iter_pii(t), None) for t in conversation_texts(body))


# Keys whose values are never prose: binary payloads, ids and enums. Inside tool call arguments every key is checked,
# since those are the model's own choice of names.
REDACT_SKIP_KEYS = {"role", "type", "id", "tool_call_id", "name", "image_url", "input_audio", "file"}
# Top-level fields that are the agent's own schemas, never conversation text.
REDACT_SKIP_TOP = {"tools", "functions", "tool_choice", "response_format"}
REDACTION_NOTE = (
    "Hybrid Gateway replaced credentials and personal data in this conversation with [redacted:<kind>] placeholders "
    "before it left the user's machine. Never write a placeholder into a file, command or config; leave lines "
    "holding one unchanged."
)


def redact_body(body: JSON) -> list[str]:
    """Redact credentials and personal data in place in a request bound for the cloud and return the kinds replaced.

    Every field is covered (messages and the system prompt, prompt, input, instructions, metadata, ...); a tool result
    that read a secrets file is withheld whole. Tool schemas are left alone. The output is deterministic, so the
    provider's prompt cache still hits.
    """
    files = secret_file_calls(body)
    kinds: list[str] = []

    def walk(value: Any, key: str | None = None, in_args: bool = False) -> Any:
        if key in REDACT_SKIP_KEYS and not in_args:
            return value
        if isinstance(value, str):
            if key == "arguments" and not in_args:  # tool call arguments are JSON in a string: redact the values
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, (dict, list)):
                    before = len(kinds)
                    redacted = walk(decoded, in_args=True)
                    return json.dumps(redacted, ensure_ascii=False) if len(kinds) > before else value
            new, found = redact_text(value)
            kinds.extend(found)
            return new
        if isinstance(value, list):
            return [walk(v, in_args=in_args) for v in value]
        if isinstance(value, dict):
            return {k: walk(v, k, in_args) for k, v in value.items()}
        return value

    messages = body.get("messages")
    if isinstance(messages, list):
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id") in files:
                name = files[msg["tool_call_id"]]
                messages[i] = {**msg, "content": f"[withheld by Hybrid Gateway: this tool result came from {name}, "
                                                 "a secrets file, and stays on the user's machine]"}
                kinds.append("secrets-file")
            else:
                messages[i] = walk(msg)
    for key in list(body):  # prompt, input, instructions, metadata, user and anything else a client sends
        if key not in REDACT_SKIP_TOP and key != "messages":
            body[key] = walk(body[key])
    if kinds and isinstance(messages, list):
        first = messages[0] if messages else None
        if isinstance(first, dict) and first.get("role") in ("system", "developer") \
                and isinstance(first.get("content"), str):
            messages[0] = {**first, "content": f"{first['content']}\n\n{REDACTION_NOTE}"}
        else:
            messages.insert(0, {"role": "system", "content": REDACTION_NOTE})
    return kinds


def json_object(raw: bytes) -> JSON | None:
    """A request body parsed as a JSON object, or None for anything else."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def clip(text: str, limit: int) -> str:
    """Keep the start and end of long text; instructions often come last in a long paste."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    return f"{text[:head]}\n[…]\n{text[-(limit - head):]}"


def router_input(body: JSON, tokens: int, current: str | None) -> str:
    """What the router model sees: the request itself plus a few numbers, not the transcript."""
    users = user_messages(body)
    latest = message_text(users[-1]) if users else ""
    original = message_text(users[0]) if users else ""
    parts = []
    if len(users) > 1 and original.strip() != latest.strip():
        parts.append(f"Original request:\n{clip(original, 600)}")
    parts.append(f"Latest message:\n{clip(latest, 2000)}")
    parts.append(
        f"Context: about {tokens:,} tokens; {len(body.get('tools') or [])} tools available; turn {len(users)}; "
        f"current route: {current or 'none'}; images: {'yes' if has_images(body) else 'no'}"
    )
    return "\n\n".join(parts)


def parse_decision(content: str) -> tuple[Route, str | None] | None:
    """Pull route and category out of the router's reply, tolerating stray text around the JSON."""
    for candidate in (content, *re.findall(r"\{[^{}]*\}", content)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("route") in ROUTES:
            category = data.get("category")
            return data["route"], category if category in ROUTE_CATEGORIES else None
    return None


class Sessions:
    """Last route per conversation, in memory only, so tool loops stay on one model."""

    def __init__(self, idle_s: float = 1800, cap: int = 2000) -> None:
        from collections import OrderedDict

        self._routes: OrderedDict[str, tuple[Route, float]] = OrderedDict()
        self._idle_s = idle_s
        self._cap = cap

    def get(self, key: str | None) -> Route | None:
        """The conversation's current route, unless it has gone idle."""
        if key is None or key not in self._routes:
            return None
        route, seen = self._routes[key]
        if time.monotonic() - seen > self._idle_s:
            del self._routes[key]
            return None
        return route

    def set(self, key: str | None, route: Route) -> None:
        """Remember the route, evicting the least recently used conversation when full."""
        if key is None:
            return
        self._routes[key] = (route, time.monotonic())
        self._routes.move_to_end(key)
        while len(self._routes) > self._cap:
            self._routes.popitem(last=False)


def create_app(settings: Settings) -> Any:
    """Build the FastAPI app: proxy routes, dashboard, dashboard API and settings API."""
    from contextlib import asynccontextmanager
    from datetime import date

    import anyio
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
    from starlette.background import BackgroundTask

    startup = settings.startup
    store = Store(startup.db_path)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0),  # overridden per request
        follow_redirects=False,
    )

    def upstream_timeout(cfg: Config) -> httpx.Timeout:
        return httpx.Timeout(connect=10.0, read=cfg.read_timeout, write=60.0, pool=10.0)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(title="Hybrid Gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    allowed_hosts = {f"127.0.0.1:{startup.port}", f"localhost:{startup.port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    @app.middleware("http")
    async def loopback_only(request: Request, call_next: Any) -> Any:
        # Block DNS rebinding and cross-site requests from other pages in the browser.
        if request.headers.get("host") not in allowed_hosts:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in allowed_origins:
            return JSONResponse({"error": "forbidden origin"}, status_code=403)
        return await call_next(request)

    # -- proxy ---------------------------------------------------------------

    sessions = Sessions()

    def token_error(request: Request, cfg: Config) -> Response | None:
        """401/503 unless the request carries the hybrid gateway token and Telnyx is configured."""
        if not cfg.gateway_token or not cfg.remote_api_key:
            return JSONResponse({"error": "the Hybrid Gateway token and Telnyx API key must be set "
                                          "(dashboard settings, or HYBRID_GATEWAY_TOKEN and TELNYX_API_KEY)"},
                                status_code=503)
        presented = request.headers.get("authorization", "")
        if not hmac.compare_digest(presented.encode(), f"Bearer {cfg.gateway_token}".encode()):
            return JSONResponse({"error": "invalid or missing hybrid gateway token"}, status_code=401)
        return None

    async def proxy(request: Request, backend: Backend, path: str) -> Response:
        cfg = settings.config
        if backend == "telnyx" and (err := token_error(request, cfg)):
            return err
        return await forward(request, cfg, backend, path, await request.body())

    async def forward(request: Request, cfg: Config, backend: Backend, path: str, raw: bytes,
                      decision: Decision | None = None) -> Response:
        base = cfg.local_url if backend == "local" else cfg.remote_url
        url = f"{base}/{path}"
        headers = upstream_headers(request.headers, backend, cfg)
        if decision is not None and backend == "local" and not cfg.local_api_key:
            headers.pop("authorization", None)  # it carries the hybrid gateway token
        metered = request.method == "POST" and path in METERED_PATHS

        body: JSON = {}
        if metered:
            try:
                parsed = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return JSONResponse({"error": "request body must be JSON"}, status_code=400)
            if not isinstance(parsed, dict):
                return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
            body = parsed

        redacted: list[str] = []
        if backend == "telnyx" and request.method == "POST":
            target = body if metered else json_object(raw)
            if target is None and raw.strip():
                # Fail closed: a body that can't be read (plain text, a multipart upload) can't be redacted.
                return JSONResponse({"error": "Hybrid Gateway only forwards JSON object bodies to Telnyx, so it can "
                                              "redact credentials and personal data first"}, status_code=415)
            if target is not None and (redacted := redact_body(target)):
                raw = json.dumps(target).encode()
                counts = ", ".join(f"{k} x{n}" if n > 1 else k for k, n in Counter(redacted).items())
                print(f"hybrid-gateway: redacted {len(redacted)} private value(s) before sending to Telnyx: {counts}",
                      file=sys.stderr)

        streaming = metered and body.get("stream") is True
        if streaming:
            opts = body.get("stream_options")
            if not isinstance(opts, dict):
                body["stream_options"] = {"include_usage": True}
            elif "include_usage" not in opts:
                opts["include_usage"] = True
            raw = json.dumps(body).encode()

        rec = RequestRecord(backend=backend, model=body.get("model")) if metered else None
        if rec:
            rec.redactions = len(redacted)
        if rec and decision:
            rec.route, rec.route_source, rec.route_category = decision.route, decision.source, decision.category
        upstream_req = client.build_request(request.method, url, headers=headers, content=raw,
                                            params=list(request.query_params.multi_items()),
                                            timeout=upstream_timeout(cfg))
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            if rec:
                rec.status = 502
                rec.finish()
                store.log(rec)
            return JSONResponse({"error": f"upstream unavailable: {type(exc).__name__}"}, status_code=502)

        if not metered:
            return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                     headers=client_headers(upstream.headers),
                                     background=BackgroundTask(upstream.aclose))

        assert rec is not None
        rec.status = upstream.status_code

        if streaming and upstream.status_code < 400:
            tap = SSEUsageTap()

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream.aiter_raw():
                        tap.feed(chunk)
                        yield chunk
                except httpx.HTTPError:
                    rec.status = 502
                except BaseException:  # client went away mid-stream
                    rec.status = 499
                    raise
                finally:
                    # Log before any await: anyio re-raises cancellation at every await in this block.
                    rec.finish()
                    tap.close()
                    rec.model = tap.model or rec.model
                    if tap.usage:
                        rec.apply_usage(tap.usage)
                    else:
                        rec.estimate(body, tap.text)
                    store.log(rec)
                    with anyio.CancelScope(shield=True):
                        await upstream.aclose()

            out_headers = client_headers(upstream.headers)
            out_headers.update({"cache-control": "no-cache", "x-accel-buffering": "no"})
            return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers,
                                     media_type=upstream.headers.get("content-type", "text/event-stream"))

        # Non-streaming (or an error response): read it whole, meter, return unchanged.
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        payload: JSON = {}
        try:
            maybe = json.loads(content)
            payload = maybe if isinstance(maybe, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        rec.finish()
        rec.model = payload.get("model") or rec.model
        if isinstance(payload.get("usage"), dict):
            rec.apply_usage(payload["usage"])
        elif upstream.status_code < 400:
            rec.estimate(body, response_text(payload))
        store.log(rec)
        return Response(content=content, status_code=upstream.status_code, headers=client_headers(upstream.headers))

    @app.api_route("/local/v1/{path:path}", methods=["GET", "POST"])
    async def local_route(request: Request, path: str) -> Response:
        return await proxy(request, "local", path)

    @app.api_route("/telnyx/v1/{path:path}", methods=["GET", "POST"])
    async def telnyx_route(request: Request, path: str) -> Response:
        return await proxy(request, "telnyx", path)

    # -- auto routing --------------------------------------------------------

    async def context_tokens(body: JSON) -> int:
        text = prompt_text(body)
        if len(text) > 200_000:  # big prompts: count off the event loop
            return await anyio.to_thread.run_sync(count_tokens, text)
        return count_tokens(text)

    async def ask_router(cfg: Config, body: JSON, tokens: int, current: Route | None, log: bool) -> Decision:
        """One call to the small local model; falls back to the current route (or mac) on any failure."""
        prices = load_prices(startup.prices_path)
        system = router_prompt(tier_label(prices, cfg.flash_model), tier_label(prices, cfg.k3_model))
        req = {
            "model": cfg.router_model, "temperature": 0, "max_tokens": 60, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": router_input(body, tokens, current)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "route", "strict": True, "schema": ROUTER_SCHEMA}},
        }
        headers = {"authorization": f"Bearer {cfg.local_api_key}"} if cfg.local_api_key else {}
        rec = RequestRecord(backend="local", model=cfg.router_model, purpose="router")
        parsed = None
        try:
            r = await client.post(f"{cfg.local_url}/chat/completions", json=req, headers=headers,
                                  timeout=cfg.router_timeout)
            rec.status = r.status_code
            payload = r.json() if r.status_code < 400 else {}
            if isinstance(payload.get("usage"), dict):
                rec.apply_usage(payload["usage"])
            content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            parsed = parse_decision(content)
        except (httpx.HTTPError, ValueError, AttributeError, IndexError) as exc:
            rec.status = rec.status or 504
            print(f"hybrid-gateway: router call failed: {type(exc).__name__}", file=sys.stderr)
        rec.finish()
        if log:
            store.log(rec)
        if parsed is None:
            return Decision(current or "mac", "fallback", router_ms=rec.wall_ms)
        return Decision(parsed[0], "router", parsed[1], router_ms=rec.wall_ms)

    async def decide(cfg: Config, body: JSON, dry_run: bool = False) -> Decision:
        """Fixed rules first, then the router model, once per user turn."""
        key = session_key(body)
        current = sessions.get(key)
        tokens = await context_tokens(body)
        msgs = body.get("messages") or []
        last_role = msgs[-1].get("role") if msgs and isinstance(msgs[-1], dict) else None
        mac_fits = tokens <= cfg.local_context

        secret = has_secret(body)
        pii = not secret and has_pii(body)
        if (secret or pii) and not mac_fits:
            print("hybrid-gateway: /auto request looks like it holds a credential or personal data but is too big for "
                  "the Mac; routing it normally, redacted", file=sys.stderr)
        if mac_fits and secret:
            # Credentials stay on the Mac, for this turn and (since they stay in the history) the rest of the chat.
            decision = Decision("mac", "rule:secret")
        elif mac_fits and pii:
            # Personal data likewise.
            decision = Decision("mac", "rule:pii")
        elif tokens > cfg.flash_context:
            decision = Decision("k3", "rule:context")
        elif last_role != "user" and current is not None:
            # Mid tool loop: the result goes back to the model that asked for it, unless it no longer fits.
            decision = Decision(current, "rule:tool-loop") if current != "mac" or mac_fits \
                else Decision("flash", "rule:context")
        else:
            decision = await ask_router(cfg, body, tokens, current, log=not dry_run)
            if decision.route == "mac" and not mac_fits:
                decision = Decision("flash", "rule:context", decision.category, decision.router_ms)
        if not dry_run:
            sessions.set(key, decision.route)
        return decision

    @app.api_route("/auto/v1/{path:path}", methods=["GET", "POST"])
    async def auto_route(request: Request, path: str) -> Response:
        cfg = settings.config
        if err := token_error(request, cfg):
            return err
        if request.method == "GET" and path == "models":
            return JSONResponse({"object": "list", "data": [{"id": "auto", "object": "model", "owned_by": "hybrid-gateway"}]})
        if request.method != "POST" or path != "chat/completions":
            return JSONResponse({"error": "/auto supports GET /models and POST /chat/completions"}, status_code=404)
        if not cfg.local_model:
            return JSONResponse({"error": "a Mac model must be set for /auto "
                                          "(dashboard settings, or HYBRID_GATEWAY_LOCAL_MODEL)"}, status_code=503)
        try:
            body = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse({"error": "request body must be a JSON object with messages"}, status_code=400)

        decision = await decide(cfg, body)
        backend: Backend = "local" if decision.route == "mac" else "telnyx"
        body["model"] = {"mac": cfg.local_model, "flash": cfg.flash_model, "k3": cfg.k3_model}[decision.route]
        return await forward(request, cfg, backend, path, json.dumps(body).encode(), decision)

    # -- dashboard -----------------------------------------------------------

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(HERE / "dashboard.html", headers={"cache-control": "no-store"})

    class StartJob(BaseModel):
        label: str = Field(min_length=1, max_length=80)

    async def json_body(request: Request) -> JSON:
        # Requiring a JSON content type forces a CORS preflight for any cross-site caller.
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "content-type must be application/json")
        try:
            data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(400, "invalid JSON")
        if not isinstance(data, dict):
            raise HTTPException(400, "expected a JSON object")
        return data

    @app.post("/api/jobs/start")
    async def start_job(request: Request) -> JSON:
        try:
            req = StartJob.model_validate(await json_body(request))
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_context=False))
        label = req.label.strip()
        if not label:
            raise HTTPException(422, "label must not be blank")
        store.start_job(label)
        return {"ok": True}

    @app.post("/api/jobs/end")
    async def end_job(request: Request) -> JSON:
        await json_body(request)
        store.end_job()
        return {"ok": True}

    @app.post("/api/reset")
    async def reset(request: Request) -> JSON:
        await json_body(request)
        store.reset()
        return {"ok": True}

    class RouteProbe(BaseModel):
        messages: list[dict[str, Any]] = Field(min_length=1, max_length=1000)
        tools: list[dict[str, Any]] | None = None

    @app.post("/api/route")
    async def route_probe(request: Request) -> JSON:
        """Dry run: what /auto would decide for these messages. Nothing is forwarded or logged."""
        try:
            probe = RouteProbe.model_validate(await json_body(request))
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_context=False))
        d = await decide(settings.config, probe.model_dump(exclude_none=True), dry_run=True)
        return {"route": d.route, "source": d.source, "category": d.category, "router_ms": d.router_ms}

    @app.get("/api/summary")
    async def summary() -> JSON:
        return build_summary(store, load_prices(startup.prices_path), settings.config)

    @app.get("/api/export.csv")
    async def export_csv() -> Response:
        prices = load_prices(startup.prices_path)
        rows = store.query("SELECT * FROM requests ORDER BY id")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["ts", "backend", "model", "job", "purpose", "route", "route_source", "route_category",
                    "prompt_tokens", "cached_tokens", "completion_tokens", "wall_ms", "status", "token_source",
                    "redactions", "cost_usd"])
        for r in rows:
            rates = price_for(prices, r["model"]) if r["backend"] == "telnyx" else None
            cost = (f"{cost_usd(rates, r['prompt_tokens'], r['cached_tokens'], r['completion_tokens']):.6f}"
                    if rates else ("0" if r["backend"] == "local" else ""))
            w.writerow([r["ts"], r["backend"], r["model"], r["job_label"], r["purpose"], r["route"],
                        r["route_source"], r["route_category"], r["prompt_tokens"], r["cached_tokens"],
                        r["completion_tokens"], r["wall_ms"], r["status"], r["token_source"], r["redactions"], cost])
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"content-disposition": f'attachment; filename="hybrid-gateway-{stamp}.csv"'})

    # -- settings ------------------------------------------------------------

    class TierPrices(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        label: str | None = Field(None, min_length=1, max_length=40)
        input: float | None = Field(None, ge=0, le=10_000)
        cached: float | None = Field(None, ge=0, le=10_000)
        output: float | None = Field(None, ge=0, le=10_000)

    class PricesBody(BaseModel):
        model_config = ConfigDict(extra="forbid")
        verified: str | None = None
        flash: TierPrices | None = None
        k3: TierPrices | None = None

        @field_validator("verified")
        @classmethod
        def iso_date(cls, v: str | None) -> str | None:
            if v is not None:
                if len(v) != 10:
                    raise ValueError("must be YYYY-MM-DD")
                date.fromisoformat(v)  # ValueError -> 422
            return v

    class SettingsBody(BaseModel):
        model_config = ConfigDict(extra="forbid")
        values: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=len(FIELDS))
        clear: list[str] = Field(default_factory=list, max_length=len(FIELDS))
        prices: PricesBody | None = None

    def error(status: int, message: str, field: str | None = None) -> JSONResponse:
        return JSONResponse({"error": message, "field": field}, status_code=status)

    def invalid(exc: ValidationError) -> JSONResponse:
        """422 listing every envelope error; "field" names the first (a setting key or prices.<tier>.<name>)."""
        found: dict[str, tuple[str | None, str]] = {}  # label -> (field, message), one per field
        # include_input=False: a rejected secret must never be echoed back.
        for e in exc.errors(include_url=False, include_input=False, include_context=False):
            loc = [str(part) for part in e["loc"]]
            msg = e["msg"]
            if loc[0] == "values" and len(loc) > 1:
                field: str | None = loc[1]
                if len(loc) > 2:  # a type error inside the value union names only its first branch
                    msg = "must be a string, a number or null"
            elif loc[0] == "prices":
                field = ".".join(loc[:3])
            else:
                field = None
            found.setdefault(field or ".".join(loc), (field, msg))
        first = next((f for f, _ in found.values() if f), None)
        return error(422, "; ".join(f"{label}: {msg}" for label, (_, msg) in found.items()), first)

    @app.get("/api/settings")
    async def get_settings() -> JSON:
        return settings.view()

    @app.post("/api/settings", response_model=None)
    async def post_settings(request: Request) -> JSON | JSONResponse:
        try:
            req = SettingsBody.model_validate(await json_body(request))
        except ValidationError as exc:
            return invalid(exc)
        edit = None
        if req.prices:
            tiers = {t: p.model_dump(exclude_none=True) for t, p in (("flash", req.prices.flash), ("k3", req.prices.k3))
                     if p is not None}
            edit = PricesEdit(verified=req.prices.verified, tiers=tiers)
        try:
            values: dict[str, object] = dict(req.values)
            await anyio.to_thread.run_sync(settings.apply, values, req.clear, edit)
        except SettingsError as exc:
            return error(exc.status, exc.message, exc.field)
        return settings.view()

    @app.post("/api/settings/token")
    async def reveal_token(request: Request) -> JSON:
        await json_body(request)
        return {"token": settings.config.gateway_token}

    @app.post("/api/settings/token/rotate", response_model=None)
    async def rotate_token(request: Request) -> JSON | JSONResponse:
        await json_body(request)
        change: dict[str, object] = {"gateway_token": secrets.token_urlsafe(32)}
        try:
            await anyio.to_thread.run_sync(settings.apply, change, [])
        except SettingsError as exc:
            return error(exc.status, exc.message, exc.field)
        return {"token": settings.config.gateway_token}

    @app.get("/api/settings/models")
    async def list_models(backend: Literal["local", "remote"]) -> JSON:
        """Model ids the backend lists, using the current settings. Errors never include a key."""
        cfg = settings.config
        base, key = (cfg.local_url, cfg.local_api_key) if backend == "local" else (cfg.remote_url, cfg.remote_api_key)

        def failed(reason: str) -> JSON:
            return {"ok": False, "models": [], "error": reason}

        if backend == "remote" and not key:
            return failed("Telnyx API key is not set")
        try:
            r = await client.get(f"{base}/models", headers={"authorization": f"Bearer {key}"} if key else {},
                                 timeout=5.0)
        except httpx.TimeoutException:
            return failed("timed out after 5 s")
        except httpx.HTTPError as exc:
            return failed(f"could not connect ({type(exc).__name__})")
        if r.status_code >= 400:
            return failed(f"server returned HTTP {r.status_code}")
        try:
            data = r.json().get("data")
            ids = [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]
        except (ValueError, AttributeError, TypeError):
            return failed("unexpected response from /models")
        return {"ok": True, "models": ids, "error": None}

    return app


def build_summary(store: Store, prices: JSON, cfg: Config) -> JSON:
    """Aggregate the log into dashboard columns (local, flash tier, k3 tier) and per-job rows."""
    columns: dict[str, JSON] = {"local": {"key": "local", "tier": "local", "label": "On the Mac", "cost": None}}
    for tier, model in (("flash", cfg.flash_model), ("k3", cfg.k3_model)):
        columns.setdefault(model.lower(), {"key": model, "tier": tier, "label": tier_label(prices, model),
                                           "cost": 0.0 if price_for(prices, model) else None})
    for col in columns.values():
        col.update(jobs=0, requests=0, tokens_in=0, tokens_out=0, _jobs=set())

    groups = store.query(
        """SELECT r.job_id, r.job_label, r.backend, r.model, r.purpose,
                  COUNT(*) AS requests,
                  SUM(r.prompt_tokens) AS tokens_in, SUM(r.cached_tokens) AS cached,
                  SUM(r.completion_tokens) AS tokens_out, SUM(r.wall_ms) AS wall_ms,
                  SUM(r.token_source = 'tiktoken') AS estimated,
                  MIN(r.id) AS first_id, j.ended_at
           FROM requests r LEFT JOIN jobs j ON j.id = r.job_id
           GROUP BY r.job_id, r.backend, r.model, r.purpose
           ORDER BY first_id DESC"""
    )
    jobs: list[JSON] = []
    total = 0.0
    for g in groups:
        rates = price_for(prices, g["model"]) if g["backend"] == "telnyx" else None
        cost = cost_usd(rates, g["tokens_in"], g["cached"], g["tokens_out"]) if rates else None
        jobs.append({
            "id": f"{g['job_id'] or 'untagged'}|{g['backend']}|{g['model']}|{g['purpose']}",
            "label": g["job_label"],
            "purpose": g["purpose"],
            "open": g["job_id"] is not None and g["ended_at"] is None,
            "backend": g["backend"],
            "model": g["model"],
            "requests": g["requests"],
            "tokens_in": g["tokens_in"],
            "tokens_out": g["tokens_out"],
            "wall_ms": g["wall_ms"],
            "estimated": g["estimated"] > 0,
            "cost": cost,
        })
        total += cost or 0.0
        col_key = "local" if g["backend"] == "local" else (g["model"] or "").lower()
        target = columns.get(col_key)
        if target is None:
            continue
        target["requests"] += g["requests"]
        target["tokens_in"] += g["tokens_in"]
        target["tokens_out"] += g["tokens_out"]
        if g["job_id"] is not None:
            target["_jobs"].add(g["job_id"])
        if cost is not None and target["cost"] is not None:
            target["cost"] += cost
    for col in columns.values():
        col["jobs"] = len(col.pop("_jobs"))

    routed = {r["route"]: r["n"] for r in store.query(
        "SELECT route, COUNT(*) AS n FROM requests WHERE purpose = 'work' AND route IS NOT NULL GROUP BY route")}
    calls = store.query(
        "SELECT COUNT(*) AS n, AVG(wall_ms) AS avg_ms FROM requests WHERE purpose = 'router' AND status < 400")[0]
    fallbacks = store.query("SELECT COUNT(*) AS n FROM requests WHERE route_source = 'fallback'")[0]["n"]
    redactions = store.query("SELECT COUNT(*) AS requests, COALESCE(SUM(redactions), 0) AS n FROM requests "
                             "WHERE redactions > 0")[0]

    job = store.open_job()
    return {
        "total_cost": total,
        "columns": list(columns.values()),
        "jobs": jobs,
        "router": {
            "model": cfg.router_model,
            "enabled": cfg.local_model is not None,
            "local_context": cfg.local_context,
            "flash_context": cfg.flash_context,
            "routed": {r: routed.get(r, 0) for r in ROUTES},
            "decisions": calls["n"],
            "avg_ms": round(calls["avg_ms"]) if calls["avg_ms"] is not None else None,
            "fallbacks": fallbacks,
        },
        "redactions": {"requests": redactions["requests"], "credentials": redactions["n"]},
        "open_job": {"label": job["label"], "started_at": job["started_at"]} if job else None,
        "prices_verified": prices.get("verified"),
    }


# ---------------------------------------------------------------------------
# CLI: pack
# ---------------------------------------------------------------------------

LOCKFILES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lock", "bun.lockb",
    "Cargo.lock", "poetry.lock", "uv.lock", "Pipfile.lock", "pdm.lock", "composer.lock", "Gemfile.lock",
    "go.sum", "mix.lock", "pubspec.lock", "Podfile.lock", "Package.resolved", "flake.lock", "deno.lock",
    "packages.lock.json", "gradle.lockfile",
}


def is_lockfile(path: Path) -> bool:
    """True for dependency lockfiles."""
    return path.name in LOCKFILES or path.suffix == ".lock"


def git_files(root: Path) -> list[Path] | None:
    """Tracked + untracked-but-not-ignored files, if root is inside a git work tree."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted({root / p for p in out.decode().split("\0") if p})


def walk_files(root: Path) -> list[Path]:
    """Walk root honouring every .gitignore found on the way down (non-git fallback)."""
    import pathspec

    found: list[Path] = []

    def ignored(path: Path, specs: list[tuple[Path, Any]], is_dir: bool) -> bool:
        for base, spec in specs:
            rel = path.relative_to(base).as_posix() + ("/" if is_dir else "")
            if spec.match_file(rel):
                return True
        return False

    def visit(d: Path, specs: list[tuple[Path, Any]]) -> None:
        gi = d / ".gitignore"
        if gi.is_file():
            specs = [*specs, (d, pathspec.GitIgnoreSpec.from_lines(gi.read_text(errors="ignore").splitlines()))]
        for entry in sorted(d.iterdir()):
            if entry.name == ".git" or entry.is_symlink():
                continue
            if entry.is_dir():
                if not ignored(entry, specs, True):
                    visit(entry, specs)
            elif entry.is_file() and not ignored(entry, specs, False):
                found.append(entry)

    visit(root, [])
    return found


def read_text_file(path: Path) -> str | None:
    """File contents as text, or None if it looks binary."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def cmd_pack(args: argparse.Namespace) -> int:
    """Pack a repo into one text file and print its token count."""
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    out_path = Path(args.output or f"{root.name}.packed.txt").resolve()
    files = git_files(root)
    files = walk_files(root) if files is None else [f for f in files if f.is_file() and not f.is_symlink()]

    skipped_binary = skipped_lock = 0
    skipped_secret: list[str] = []
    redacted: list[str] = []
    parts: list[str] = [f"# Packed from {root.name} on {now_iso()}\n"]
    included = 0
    for f in files:
        if f.resolve() == out_path:
            continue
        if is_lockfile(f):
            skipped_lock += 1
            continue
        if not args.no_redact and is_secret_file(f.name):
            skipped_secret.append(f.relative_to(root).as_posix())
            continue
        text = read_text_file(f)
        if text is None:
            skipped_binary += 1
            continue
        if not args.no_redact:
            text, found = redact_text(text)
            redacted.extend(found)
        parts.append(f"\n===== {f.relative_to(root).as_posix()} =====\n{text}")
        if not text.endswith("\n"):
            parts.append("\n")
        included += 1

    packed = "".join(parts)
    out_path.write_text(packed)
    tokens = count_tokens(packed)
    limit = args.limit
    verdict = "fits" if tokens <= limit else "DOES NOT FIT"
    print(f"Wrote {out_path}")
    print(f"Files: {included} included, {skipped_binary} binary skipped, {skipped_lock} lockfiles skipped")
    if skipped_secret or redacted:
        print(f"Secrets: {len(skipped_secret)} secrets file(s) skipped ({', '.join(skipped_secret[:5])}"
              f"{', ...' if len(skipped_secret) > 5 else ''}), {len(redacted)} value(s) redacted. "
              f"Run `gateway scan {args.path}` to see where.")
    print(f"Tokens: {tokens:,} (o200k_base estimate) — {verdict} in {limit:,} ({tokens / limit:.1%})")
    return 0 if tokens <= limit else 1


SCAN_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache",
                  ".pytest_cache", ".turbo", ".next", ".wrangler"}
SCAN_MAX_BYTES = 5_000_000


def scan_files(root: Path) -> list[Path]:
    """Every regular file under root, ignored ones included (an upload doesn't read .gitignore), minus caches."""
    if root.is_file():
        return [root]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SCAN_SKIP_DIRS)
        found.extend(Path(dirpath) / name for name in sorted(filenames))
    return [f for f in found if f.is_file() and not f.is_symlink()]


def cmd_scan(args: argparse.Namespace) -> int:
    """Report files and lines that look like they hold secrets or personal data, without printing any value."""
    files = scanned = 0
    findings = 0
    for arg in args.paths:
        root = Path(arg).expanduser().resolve()
        if not root.exists():
            print(f"not found: {root}", file=sys.stderr)
            return 2
        for f in scan_files(root):
            files += 1
            rel = f.relative_to(root.parent if root.is_file() else root).as_posix()
            hits: list[str] = []
            if is_secret_file(f.name):
                hits.append(f"{rel}  secrets file: withheld from the cloud whole")
            text = read_text_file(f) if f.stat().st_size <= SCAN_MAX_BYTES else None
            if text is not None:
                scanned += 1
                hits += [f"{rel}:{text.count(chr(10), 0, start) + 1}  {kind}" for start, _, kind in find_secrets(text)]
            findings += len(hits)
            for line in hits:
                print(line)
    verdict = f"{findings} finding(s)" if findings else "nothing found"
    print(f"\nScanned {scanned} text file(s) of {files}: {verdict}. Values are never printed. Matching favours "
          "precision, so a clean result is not a guarantee.", file=sys.stderr)
    return 1 if findings else 0


def cmd_serve(_: argparse.Namespace) -> int:
    """Run the proxy and dashboard."""
    import uvicorn

    try:
        env = load_env()
        startup = Startup.from_env(env)
        settings = Settings(startup, env)
    except ValueError as exc:
        print(f"hybrid-gateway: {exc}", file=sys.stderr)
        return 2
    cfg = settings.config
    keychain = f"Keychain service {startup.keychain_service}" if settings.keychain else "no Keychain (env secrets only)"
    print(f"hybrid-gateway: settings -> {startup.settings_path}, {keychain}")
    if not cfg.gateway_token or not cfg.remote_api_key:
        print("hybrid-gateway: Hybrid Gateway token or Telnyx API key not set; /telnyx and /auto will return 503 "
              "until both are set in the dashboard or env", file=sys.stderr)
    print(f"hybrid-gateway: local -> {cfg.local_url}")
    print(f"hybrid-gateway: telnyx -> {cfg.remote_url} (flash {cfg.flash_model}, k3 {cfg.k3_model})")
    if cfg.local_model:
        print(f"hybrid-gateway: auto -> router {cfg.router_model}; mac = {cfg.local_model}, "
              f"local context {cfg.local_context:,}, flash context {cfg.flash_context:,}")
    else:
        print("hybrid-gateway: no Mac model set; /auto will return 503 until one is set", file=sys.stderr)
    print(f"hybrid-gateway: dashboard http://{startup.host}:{startup.port}/")
    uvicorn.run(create_app(settings), host=startup.host, port=startup.port, log_level="warning")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """Ask the running proxy where /auto would send a request (dry run: nothing forwarded or logged)."""
    import urllib.error
    import urllib.request

    try:
        port = Startup.from_env(load_env(quiet=True)).port
    except ValueError as exc:
        print(f"hybrid-gateway: {exc}", file=sys.stderr)
        return 2
    for text in args.text:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/route", method="POST",
            data=json.dumps({"messages": [{"role": "user", "content": text}]}).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.load(resp)
        except (urllib.error.URLError, OSError) as exc:
            print(f"Hybrid Gateway is not reachable on port {port}: {exc}", file=sys.stderr)
            return 2
        ms = f"{d['router_ms']} ms" if d.get("router_ms") is not None else "no router call"
        print(f"{d['route']:<5}  {d.get('category') or '-':<9} {d['source']:<15} {ms:>14}  {clip(text, 70)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="gateway", description="Hybrid Gateway")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the proxy and dashboard").set_defaults(func=cmd_serve)
    r = sub.add_parser("route", help="dry-run the /auto router on one or more requests")
    r.add_argument("text", nargs="+", help="request text, one argument per request")
    r.set_defaults(func=cmd_route)
    p = sub.add_parser("pack", help="pack a repo into one text file and count tokens")
    p.add_argument("path")
    p.add_argument("-o", "--output", help="output file (default: <repo>.packed.txt in the current directory)")
    p.add_argument("--limit", type=int, default=1_000_000, help="token budget to check against")
    p.add_argument("--no-redact", action="store_true", help="keep secrets files and credential values")
    p.set_defaults(func=cmd_pack)
    sc = sub.add_parser("scan", help="list files and lines that look like they hold secrets (values never shown)")
    sc.add_argument("paths", nargs="+", help="files or folders to check before handing them to an agent")
    sc.set_defaults(func=cmd_scan)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
