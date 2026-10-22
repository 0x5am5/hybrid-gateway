# Hybrid Gateway

A small local proxy between Hermes Agent and its model backends: a local OpenAI-compatible server on the Mac, and Telnyx Inference. It can also choose the backend for each user turn. It logs metadata for every completion request, never prompt or response text, and shows where the work ran and what it cost at <http://127.0.0.1:8787/>.

```
Hermes ──► 127.0.0.1:8787/local/v1/*  ──► local server (llama-server, oMLX, …)
       ├─► 127.0.0.1:8787/telnyx/v1/* ──► Telnyx Inference
       └─► 127.0.0.1:8787/auto/v1/*   ──► router model picks the Mac, Flash or K3
                    │
                    └── gateway.db (metadata only) ──► dashboard
```

| File | What it is |
| --- | --- |
| `gateway.py` | Proxy, auto-router, secret redaction, dashboard and settings API, and the `serve`/`route`/`pack`/`scan` CLI. One file with inline [uv](https://docs.astral.sh/uv/) dependencies. |
| `dashboard.html` | The dashboard and Settings drawer. No build step. |
| `prices.json` | USD per 1M tokens for each Telnyx model, its display label, and the `verified` date. |
| `test_gateway.py` | End-to-end checks; all but the path check need the running proxy. |
| `.env.example` | Template for `.env`, the optional [env file](#the-env-file). |
| `.env`, `settings.json`, `gateway.db` | Yours: created locally next to `gateway.py` and gitignored. `.env` is your env file, `settings.json` holds values saved in the Settings drawer, `gateway.db` is the request log. |

## Quick start

You need [uv](https://docs.astral.sh/uv/). It fetches Python 3.12 and the dependencies on first run.

```sh
cp .env.example .env && chmod 600 .env   # optional: fill in any values you'd rather keep in a file
uv run gateway.py serve
```

Open <http://127.0.0.1:8787/> and press **Settings**. Set the local server URL (and its API key, if it has one), your Telnyx API key and the Mac model, then press **Create token** under *Hybrid Gateway token* and copy the value into `~/.hermes/.env`:

```sh
HYBRID_GATEWAY_TOKEN=<the token>
```

Then [point Hermes at the proxy](#hermes-setup). Prefer environment variables? Every setting below has one, which you can set in your shell or in [`.env`](#the-env-file). Anything saved in the app overrides them.

## Settings

| Setting | Env var | Default | Stored in |
| --- | --- | --- | --- |
| Mac: Server URL. OpenAI-compatible base URL, ending in `/v1`. | `HYBRID_GATEWAY_LOCAL_URL` | `http://127.0.0.1:8080/v1` | settings.json |
| Mac: API key. Replaces the client's auth on `/local` requests. | `HYBRID_GATEWAY_LOCAL_API_KEY` | none | Keychain |
| Mac: Mac model. The model `/auto` uses for the Mac route; `/auto` returns 503 until it is set. | `HYBRID_GATEWAY_LOCAL_MODEL` | none | settings.json |
| Mac: Router model. Small model on the local server that picks a route. | `HYBRID_GATEWAY_ROUTER_MODEL` | `gemma-4-e2b-it-4bit` | settings.json |
| Mac: Router timeout (0.5–120 s). After this the request falls back. | `HYBRID_GATEWAY_ROUTER_TIMEOUT` | `8` | settings.json |
| Mac: Mac context (tokens). Bigger requests never run on the Mac. | `HYBRID_GATEWAY_LOCAL_CONTEXT` | `64000` | settings.json |
| Telnyx: Base URL. | `HYBRID_GATEWAY_REMOTE_URL` | `https://api.telnyx.com/v2/ai/openai` | settings.json |
| Telnyx: API key. | `TELNYX_API_KEY` | none | Keychain |
| Telnyx: Flash model. The cheap, fast tier. | `HYBRID_GATEWAY_FLASH_MODEL` | `zai-org/GLM-5.3-Flash` | settings.json |
| Telnyx: K3 model. The strongest tier. | `HYBRID_GATEWAY_K3_MODEL` | `moonshotai/Kimi-K3` | settings.json |
| Telnyx: Flash context (tokens). Bigger requests always go to K3; match Flash's real window. | `HYBRID_GATEWAY_FLASH_CONTEXT` | `200000` | settings.json |
| Proxy: Hybrid Gateway token. See [below](#the-hybrid-gateway-token). | `HYBRID_GATEWAY_TOKEN` | none | Keychain |
| Proxy: Read timeout (10–3600 s). Longest wait between upstream bytes. | `HYBRID_GATEWAY_READ_TIMEOUT` | `900` | settings.json |
| Prices: label, input, cached and output rates for the Flash and K3 models, and the verified date. | none | — | prices.json |

- **Precedence:** app > shell env > env file > default. The app means `settings.json`, or the Keychain for secrets. Each field shows an *App*, *Env* or *Default* badge, and a saved value that overrides an env var says so under the field. Leave a field empty (or press **Clear** on a secret) to fall back to the env var, then the default; the empty field's placeholder shows which.
- **Secrets** (the two API keys and the token) live in the macOS Keychain (service `hybrid-gateway` by default). They are never written to `settings.json`, never logged and never returned by the API, except the token through **Reveal**. Keys are safer there than in a plain-text env file. If no Keychain is available, set secrets with env vars instead.
- **No restart.** Saved changes apply to the next request. `prices.json` is re-read on every refresh, so hand edits apply too.
- **Startup only** (shell > env file > default, shown read-only in Settings): `HYBRID_GATEWAY_PORT` (`8787`), `HYBRID_GATEWAY_DB` (`gateway.db`), `HYBRID_GATEWAY_PRICES` (`prices.json`), `HYBRID_GATEWAY_SETTINGS` (`settings.json`), `HYBRID_GATEWAY_KEYCHAIN_SERVICE` (`hybrid-gateway`), and `HYBRID_GATEWAY_ENV_FILE` (shell only, `.env`).
- **Paths.** Files default to the folder holding `gateway.py`, and relative paths in these variables start from there, not from where you run the command. `~` is expanded. See [Moving the project](#moving-the-project).
- **Legacy names** still work, in the shell and the env file: `LLAMA_SERVER_URL`, `LLAMA_SERVER_API_KEY`, `HYBRID_GATEWAY_AUTO_LOCAL_MODEL` and `TELNYX_BASE_URL`.
- An invalid env value stops `serve` with an error naming the variable (and the env file, if it came from there), never the value. An invalid value in `settings.json` is ignored with a warning.

The Settings drawer also checks both backends' model lists and offers them as suggestions in the model fields.

### The env file

Hybrid Gateway reads `.env` next to `gateway.py` at startup, if it exists (set `HYBRID_GATEWAY_ENV_FILE` in your shell to use another file). Start from the template with `cp .env.example .env`. Use it for any variable above, so you don't have to export them in every shell:

```sh
# .env
HYBRID_GATEWAY_LOCAL_URL=http://127.0.0.1:8006/v1
HYBRID_GATEWAY_LOCAL_MODEL=Qwen3.8-27B-4bit
export HYBRID_GATEWAY_LOCAL_API_KEY="your-omlx-key"
TELNYX_API_KEY='your-telnyx-key'
HYBRID_GATEWAY_TOKEN=your-hybrid-gateway-token
```

- One `KEY=VALUE` per line. Blank lines and lines starting with `#` are skipped, a leading `export ` is allowed, and one matching pair of single or double quotes around the value is removed. Nothing is interpolated, and comments go on their own line.
- An empty value counts as unset. Unknown keys are ignored with a warning naming them.
- Your shell wins over the file. Changes need a restart.
- It may hold keys, so keep it private: `chmod 600 .env`. Git ignores it, and so does `gateway pack`. Hybrid Gateway warns at startup if other users can read it. It never logs values from the file, and never returns secret ones.

### Moving the project

Everything Hybrid Gateway uses sits in the project folder, so you can move or copy the folder as a whole.

- **Same Mac, new folder:** move it and run `uv run gateway.py serve` from anywhere. Keys saved in the drawer are still in the Keychain.
- **Another Mac:** copy the folder. A git clone won't include `.env`, `settings.json` or `gateway.db`, so run `cp .env.example .env && chmod 600 .env` and fill it in. Keys saved through the drawer stay in the old Mac's Keychain, so put them in `.env` or save them in the drawer again. Then set up Hermes on that Mac as below, with the same token in both places.

## The Hybrid Gateway token

The token is a shared password between Hermes and Hybrid Gateway, separate from your Telnyx key. Hybrid Gateway holds the real Telnyx key, so the token stops anything else on the Mac, a web page included, from spending on your Telnyx account through the proxy. `/telnyx` and `/auto` refuse requests without it (401) and return 503 until both the token and the Telnyx key are set.

Hermes reads it from `HYBRID_GATEWAY_TOKEN` in `~/.hermes/.env`. To change it, press **Rotate** in Settings and copy the new value into that file; Hermes is refused until you do. Keep `TELNYX_API_KEY` out of `~/.hermes/.env`.

## Hermes setup

In `~/.hermes/config.yaml`:

```yaml
model:
  provider: custom
  default: auto
  base_url: http://127.0.0.1:8787/auto/v1
  key_env: HYBRID_GATEWAY_TOKEN
  context_length: 1000000   # K3's window, so Hermes doesn't compress long jobs to fit the Mac

model_aliases:
  mac:
    model: Qwen3.8-27B-4bit   # yours, from Settings or curl http://127.0.0.1:8787/local/v1/models
    provider: custom
    base_url: http://127.0.0.1:8787/local/v1
  flash:
    model: zai-org/GLM-5.3-Flash
    provider: custom
    base_url: http://127.0.0.1:8787/telnyx/v1
    key_env: HYBRID_GATEWAY_TOKEN
  k3:
    model: moonshotai/Kimi-K3
    provider: custom
    base_url: http://127.0.0.1:8787/telnyx/v1
    key_env: HYBRID_GATEWAY_TOKEN
```

The aliases are manual overrides: in a session, `/model mac`, `/model flash` or `/model k3` pins the work to one backend. Config changes apply to new sessions.

Don't use `provider: llamacpp`. Hermes resolves it to its own managed llama-server and ignores `base_url`, so local traffic would bypass the meter.

## Local servers

`/local` works with any OpenAI-compatible server. Set its URL and key in Settings (or the env vars above).

- **oMLX:** `http://127.0.0.1:8006/v1`. It needs its API key, and it reports cached tokens.
- **Hermes's managed llama-server:** `http://127.0.0.1:18434/v1`. The key is in `~/.hermes/runtimes/llamacpp/.api_key`.

## Auto-routing

For each request on `/auto`, Hybrid Gateway picks the Mac, Flash or K3 in this order:

1. **Secret rule.** If a user, assistant or tool message looks like it holds a credential, or a tool read a secrets file, the request stays on the Mac. The credential stays in the history, so the rest of the conversation stays on the Mac too. The system prompt and tool schemas aren't checked. If the conversation is too big for the Mac, the rule can't apply: Hybrid Gateway prints a warning and routes the request normally, with the credentials redacted (see [Secrets](#secrets)).
2. **Context rule.** Bigger than the Flash context: K3.
3. **Tool-loop stickiness.** If the last message isn't from the user, stay on the conversation's current route, so a tool result goes back to the model that asked for it and the prompt cache stays warm. A Mac conversation that has outgrown the Mac context moves to Flash.
4. **Router model, on new user turns.** It answers in constrained JSON with a route and a category from a fixed list. If it picks the Mac for a prompt bigger than the Mac context, the request goes to Flash.
5. **Fallback.** On a timeout, error or unreadable reply, stay on the current route, or the Mac for a new conversation. It never silently moves to the cloud.

The router sees the original request and the latest message (start and end, if long), plus the context size, tool count, turn number, current route and whether there are images. It doesn't see the rest of the transcript, the system prompt or tool output. Its prompt names the tiers by their labels in `prices.json`.

Each routed request logs its route, how it was chosen (`router`, `rule:secret`, `rule:context`, `rule:tool-loop` or `fallback`) and the category; the router's own text is never stored. Each router call is logged as a Mac request marked *Router*. Conversations are recognised by a hash of the system prompt and first user message, held in memory only and forgotten after 30 minutes idle.

Dry run against the running proxy (nothing is forwarded or logged):

```sh
uv run gateway.py route "Fix the typo in README" "Refactor billing across the monorepo"
```

Gemma 4 E2B (`gemma-4-e2b-it-4bit`) routed 41 of 42 hand-labelled requests correctly, at about 0.12 s per decision after a cold load of about 4.5 s.

## Secrets

Hybrid Gateway looks for credentials in two ways:

- **Known formats:** a password in a URL (`postgres://user:pass@host`), private keys, and API keys and tokens from Stripe, Anthropic, OpenAI, GitHub, GitLab, Slack, AWS, Google, Telnyx, Hugging Face, npm, SendGrid, Resend, Sentry and Shopify, plus JWTs, bearer tokens and basic auth headers.
- **Assignments:** `name = value` in env files, JSON, YAML, TOML or code, where the name says secret, token, password, credential or key, and the value looks random rather than a placeholder (`your-api-key`), an env reference (`process.env.X`, `${X}`) or an identifier.

It also treats some files as secrets by name: `.env` and its variants (but not `.env.example`, `.sample` or `.template`), `.dev.vars`, `.netrc`, `.npmrc`, `.pypirc`, `.pgpass`, SSH keys, `*.pem`, `*.key`, `*.p12`, `*.tfvars`, `*.tfstate`, `credentials*`, `secrets.*` and service account JSON.

**Everything sent to Telnyx is redacted first**, on `/telnyx` and on `/auto`. Each credential becomes `[redacted:<kind>]` and a tool result that read a secrets file is withheld whole. A short note in the system prompt tells the model not to write placeholders back into files. Redaction covers every message, including the system prompt and tool call arguments, but not tool schemas. The same input always gives the same output, so the provider's prompt cache still hits. Nothing changes on the Mac or in Hermes's own history. Each row logs how many credentials were redacted, and the server prints their kinds, never their values. The dashboard shows the total.

Matching favours precision, so unusual formats can slip through. Treat it as a safety net, not the only safeguard. Before giving an agent a folder, check it:

```sh
uv run gateway.py scan ~/Downloads/client-files      # exit 1 if anything is found
```

`scan` reads every file, including ignored ones (an upload doesn't read `.gitignore`), except `.git`, `node_modules`, virtualenvs and caches. It prints the file, line and kind for each finding, never the value.

## Dashboard

- **Total cost** and the share of tokens processed on the Mac, Flash and K3.
- **Columns** for the Mac, the Flash model and the K3 model (labels from `prices.json`): cost, jobs, requests, tokens in and out.
- **Auto-routed** strip: requests per route, the router model, decision count, average decision time and fallbacks.
- **Jobs.** Type a label and press **Start job** (or Enter); every request is tagged with it until **End job**. Starting a new job closes the open one. Untagged requests show as *untagged*. The open job survives a restart.
- **Table** grouped by job, backend and model. Click a row (or focus it and press Enter) to highlight it; click again or press Esc to clear.
- **Reset day** deletes every logged request and job after a confirmation. **Export CSV** downloads every request with its route and cost.
- **Settings** opens the drawer described above.

It refreshes every 2 seconds.

## How it behaves

- **Passthrough.** GET and POST requests under `/local/v1` and `/telnyx/v1` are forwarded unchanged and SSE is relayed as it arrives. The one change: a streaming request without `stream_options.include_usage` gets it set to `true`, so the final chunk carries usage. `/auto/v1` serves `GET /models` (one model, `auto`) and `POST /chat/completions`.
- **Logging.** One row per completion request: timestamp, backend, model, prompt, cached and completion tokens, wall time, status, job and route. Model listings aren't logged. With no usage from the backend, tokens are estimated with tiktoken (`o200k_base`) and marked `≈` and *est.* on the dashboard. A client that disconnects mid-stream is logged as `499`.
- **Cost.** `(prompt − cached) × input + cached × cached rate + completion × output`, per 1M tokens, from `prices.json`. Mac rows cost nothing and show *on the Mac*; a Telnyx model with no price shows *no price*.
- **Browser safety.** The proxy binds to `127.0.0.1` only and rejects requests whose `Host` or `Origin` isn't `127.0.0.1`/`localhost` on its port. Dashboard and settings writes require a JSON body, so other web pages can't change settings, read the token or reset your data.

## Pack a repo

```sh
uv run gateway.py pack ~/Sites/some-repo            # writes ./some-repo.packed.txt
uv run gateway.py pack ~/Sites/some-repo -o job.txt --limit 1000000
```

Inside a git work tree it uses `git ls-files --cached --others --exclude-standard`, which honours every `.gitignore` plus your global excludes. Outside one, it reads each `.gitignore` on the way down. It skips binaries, lockfiles and secrets files, redacts credentials in the rest (`--no-redact` keeps both), prints file and token counts, and exits with `1` if the result is over `--limit` (default 1,000,000). The count uses `o200k_base`; Kimi and GLM tokenisers differ by a few per cent, so leave headroom.

To type `gateway pack …` or `gateway route …`, add an alias:

```sh
alias gateway="uv run $PWD/gateway.py"
```

## Test

With the proxy running and a model loaded on the local server:

```sh
uv run test_gateway.py                         # paths, settings, local, Telnyx and auto
uv run test_gateway.py --only settings         # or paths, secrets, local, telnyx, auto
uv run test_gateway.py --telnyx-model moonshotai/Kimi-K3
```

- **local / telnyx:** lists models, sends one non-streaming and one streaming request, checks the stream carried usage and that two rows landed with token counts. Telnyx also checks that a wrong token is rejected.
- **auto:** one routed request decided by the router model, then dry runs of tool-loop stickiness, the secret rule (a credential, and a `.env.local` tool result) and the context rule.
- **secrets:** detection of fake keys in real formats, things that aren't credentials, `redact_body` (secrets file results withheld, tool call arguments, the system note, determinism), `scan` and `pack` on a temp folder. Needs no proxy.
- **telnyx** also sends a fake AWS key and checks the row logged one redaction. Skipped if the Mac model, the token or the Telnyx key isn't set.
- **paths:** relative `HYBRID_GATEWAY_DB`, `HYBRID_GATEWAY_ENV_FILE` and friends resolve against the project folder from another working directory. Needs no proxy.
- **settings:** the API shape, that secrets are never returned, a `router_timeout` round trip that is put back afterwards, the error cases, and, if a non-secret field is set by env, that a saved value overrides it and clearing falls back to it. Secrets are never changed.

The token comes from `HYBRID_GATEWAY_TOKEN` in your shell, or is read from the proxy. Other flags: `--url`, `--db`, `--local-model`. A full run costs a fraction of a cent on Telnyx. Afterwards it deletes every request row added during the run, including any Hermes traffic in that window; pass `--keep` to leave them.
