# HyperNix T1 API

A controlled HTTP gateway into HyperNix-pip. Built as `hypernix.t1api`,
mountable into any Python server. The client requests an operation; the
server decides what exists, what's available, and how much is left — see
[Design principle](#design-principle).

**Status: released — T1 v1.0.26.8.1.1** (long form `1.0.2026.8.1.1`).
The betas ended here. This page is the living contract for what's actually
implemented vs. planned; cross-reference against the
[Roadmap](#roadmap) before assuming an endpoint exists.

The T1 API no longer tracks the `hypernix` package version. The two ship
together but answer different questions — "which pip release is this"
versus "which API contract is this" — and a client pinning a contract
could never derive one from `0.71.5rc2`. See
[Versioning](#versioning).

## Contents

- [Quickstart](#quickstart)
- [Versioning](#versioning)
- [Installation](#installation)
- [Architecture](#architecture)
- [Model registry](#model-registry)
- [Authentication](#authentication)
- [Quota & usage](#quota--usage)
- [Model routing & quota cascade](#model-routing--quota-cascade)
- [Keys, plans, and assignment](#keys-plans-and-assignment)
- [Servers](#servers)
- [Modules](#modules)
- [Remote multi-server deployment](#remote-multi-server-deployment)
- [Jobs](#jobs)
- [Events](#events)
- [Billing](#billing)
- [Cost, estimates, and forecasts](#cost-estimates-and-forecasts)
- [Audit log](#audit-log)
- [Rate limiting](#rate-limiting)
- [Network policy](#network-policy)
- [TLS and mTLS](#tls-and-mtls)
- [PostgreSQL](#postgresql)
- [Production deployment](#production-deployment)
- [The LM Studio bridge](#the-lm-studio-bridge)
- [HyperLink](#hyperlink)
- [Hugging Face link merging](#hugging-face-link-merging)
- [The SDK](#the-sdk)
- [Endpoint reference](#endpoint-reference)
- [Configuration](#configuration)
- [Security](#security)
- [Roadmap](#roadmap)
- [Design principle](#design-principle)

## Quickstart

```bash
pip install 'hypernix[t1api]'
cp .env.t1api.example .env   # fill in T1_TOKEN_SECRET at minimum
python3 -m uvicorn hypernix.t1api.app:create_app --factory --reload
# → http://127.0.0.1:8000/docs (Swagger UI)
```

Mint yourself a T1 key with the existing `gkey` CLI (T1 keys are
`hypernix.keymaster` keys — the T1 API doesn't create its own key format):

```bash
gkey create --type admin --scopes admin,read,write
```

Then drive the server with `waiter` (see
[wiki/Waiter-TUI.md](Waiter-TUI.md)):

```bash
waiter serv -A -I http://127.0.0.1:8000 -K <the T1 key from gkey create> -E
waiter models
```

## Versioning

The T1 API has its own version, in six parts:

```
1   .   0    .   2026  .  8    .   0     .  1
│       │        │        │        │        │
│       │        │        │        │        └── bug fix + assorted minor features
│       │        │        │        └─────────── new feature
│       │        │        └──────────────────── month of the release
│       │        └───────────────────────────── year of the release
│       └────────────────────────────────────── major update
└────────────────────────────────────────────── T1 API generation
```

Two spellings of the same version:

| | | |
|---|---|---|
| **short** | `1.0.26.8.1.1` | two-digit year. The wire form — what `__t1api_version__`, `GET /status`, `waiter --version` and every response carry, because it is the form people type. |
| **long** | `1.0.2026.8.1.1` | four-digit year. The changelog form. |

Both parse, with or without a `v` / `t1 v` prefix, and they compare
equal:

```python
from hypernix.t1api.version import T1Version, T1_VERSION

T1Version.parse("t1 v1.0.2026.8.1.1") == T1Version.parse("1.0.26.8.1.1")   # True
T1Version.parse("1.0.26.8.1.1") < "1.0.26.9.0.0"                            # True
T1_VERSION.generation                                                        # "1.0"
T1_VERSION.release                                                           # "2026-08"
```

A three-digit year (`1.0.202.8.0.1`) raises rather than being guessed at:
a typo that parses is worse than one that does not.

**What a client should pin.** `generation` (`api.major`) is the
compatibility boundary. Within a generation the API only adds; across one
it may remove. `T1Version.compatible_with()` is that check.

`GET /status` reports all of it:

```json
{
  "t1_api_version": "1.0.26.8.1.1",
  "t1_api_version_long": "1.0.2026.8.1.1",
  "beta": "t1-1.0",
  "t1_version": {"generation": "1.0", "release": "2026-08", "year": 2026, "month": 8}
}
```

The `beta` field keeps its name — it used to say `beta4` and now says
`t1-1.0` — because Beta 3 clients read it, and renaming a field is a
breaking change for a cosmetic win.

**Package version.** The pip package (`hypernix`) versions
independently: `0.72.3` ships T1 v1.0.26.8.1.1. `GET /status` reports
both.

## Installation

### The installer

`install-t1.sh` is the guided path: it installs the package, asks what
kind of deployment this is, and writes a configuration that matches the
answers.

```bash
./install-t1.sh                     # interactive
curl -fsSL <raw-url>/install-t1.sh | bash
./install-t1.sh --dry-run           # show what it would do, write nothing
./install-t1.sh --non-interactive   # defaults, no prompts (CI, images)
```

It asks about identity and bind address, deployment kind, which key
families may connect, the T2 admin password, the connection allowlist,
rate limits, cost accounting, the model source, HyperLink, and the
`waiter` manager TUI — then writes `.env` (mode 0600), a start script,
optionally a systemd unit and a model-registry template, mints an admin
key, and seeds the allowlist.

Three properties worth knowing:

- **It is re-runnable.** An existing `.env` is backed up with a timestamp
  rather than overwritten.
- **Secrets never reach your shell history or a world-readable file.**
  The T2 admin password is read with terminal echo off, and generated
  files are created at their final mode *before* any content is written.
- **It verifies rather than assumes.** The allowlist is read back from
  the database after seeding, because being told the whitelist is
  configured when it is not means locking yourself out of your own
  server.

`--config-dir` puts everything, including the key store, under one
directory, so two T1 servers on one machine do not share credentials.

Requires bash 3.2+ (the stock macOS shell), python3 3.10+, and pip.

### By hand

`hypernix.t1api`'s **core** (registry, storage, usage, auth, config,
errors) is pure Python + stdlib and imports fine with a base `hypernix`
install — same as `hypernix.keymaster` / `hypernix.gatekeeper`, which it
wraps rather than duplicates. The **HTTP layer**
(`hypernix.t1api.create_app`, the routers, Pydantic schemas) needs the
optional extra:

```bash
pip install 'hypernix[t1api]'   # fastapi, uvicorn, pydantic, python-dotenv
```

Importing `hypernix.t1api.create_app` without the extra raises a clear
`ImportError` telling you to install it, instead of failing deep inside
FastAPI's own import chain.

### Local / Tailscale deployment

No separate deployment mode needed — a Tailscale or LAN deployment is the
same `create_app()` + `uvicorn` command as anything else. Two things
change:

1. **Bind to the Tailscale/LAN interface**, not just loopback:
   ```bash
   uvicorn hypernix.t1api.app:create_app --factory --host 0.0.0.0 --port 8000
   ```
2. **Pass `allow_private_address=True`** wherever you register something
   at a private address — `POST /servers/register`'s
   `allow_private_address` field, or `ServerRegistry.register(...,
   allow_private_address=True)` directly. Without it, `t1api.security`'s
   SSRF guard rejects RFC1918/loopback/Tailscale (`100.64.0.0/10`)
   addresses by default (see [Security](#security)) — that's the correct
   default for "register a remote server" in general, and the explicit
   opt-in is what makes a local deployment's private addresses expected
   rather than suspicious.

`waiter serv -L` (`--local-only`) records this intent client-side in the
saved config (see [Waiter-TUI.md](Waiter-TUI.md)) but doesn't itself
change server-side behavior — the two flags above are what actually
matter on the server.

## Architecture

```
hypernix/t1api/
  errors.py     stable T1ErrorCode enum + T1APIError             (stdlib only)
  registry.py   ModelRegistry / ModelEntry — the model registry  (stdlib only)
  db.py         SQLiteBackend / PostgresBackend + translation    (stdlib only*)
  storage.py    UsageStore — usage events, history, aggregates   (stdlib only*)
  usage.py      UsageMeter — per-key/model usage + exhaustion    (stdlib only)
  cost.py       CostCalculator — pricing, estimates, forecasts   (stdlib only)
  auth.py       T1AuthService — wraps Keymaster + Gatekeeper     (stdlib only)
  keys.py       KeyDirectory — plans, assignment, model narrowing(stdlib only)
  routing.py    RoutingEngine — plan cascades, quota fallback    (stdlib only)
  servers.py    ServerRegistry — registration + trust levels     (stdlib only)
  modules.py    ModuleRegistry — blobs, checksums, versioning    (stdlib only)
  transport.py  ModuleTransport — signed push, guarded fetch     (stdlib only)
  deploy.py     DeploymentCoordinator — the job handlers         (stdlib only)
  jobs.py       JobQueue — queued/running/succeeded/failed/...   (stdlib only)
  events.py     EventBus — in-process pub/sub                    (stdlib only)
  billing.py    BillingLedger — balances, payment tokens         (stdlib only)
  audit.py      AuditLog — durable, queryable, secret-scrubbed   (stdlib only)
  ratelimit.py  RateLimiter — token bucket + sliding window      (stdlib only)
  netpolicy.py  NetworkPolicy — IP allow/block, forced limits    (stdlib only)
  mtls.py       TLSSettings / ClientCertVerifier                 (stdlib only)
  security.py   SSRF + path-traversal guards                     (stdlib only)
  config.py     T1APIConfig — env vars + production validation   (stdlib only)
  schemas.py    Pydantic request/response models                  (needs [t1api])
  deps.py       FastAPI Depends() wiring                           (needs [t1api])
  app.py        create_app() factory + middleware stack            (needs [t1api])
  routers/      health auth models usage config servers modules
                jobs events billing keys audit security           (needs [t1api])

hypernix/t1sdk/    the client SDK          (stdlib only, no server extra needed)
hypernix/waiter/   the waiter TUI/CLI      (stdlib only, sits on the SDK)
```

\* `db.py` and `storage.py` import psycopg lazily, and only when
`T1_DATABASE_URL` is actually set.

The core/HTTP split is deliberate: routing, quota math, and registry
enforcement are testable (and reusable, e.g. from `waiter` or a future
non-HTTP embedding) without pulling in FastAPI. `tests/test_t1api_core.py`
and `tests/test_t1api_auth.py` exercise the core directly and need no
extra beyond base `hypernix`; only `tests/test_t1api_http.py` needs
`pip install hypernix[t1api-test]` (fastapi/pydantic/uvicorn *and* httpx,
which Starlette's `TestClient` requires and which plain `[t1api]` doesn't
pull in).

### Mounting into an existing server

```python
from hypernix.t1api import create_app

# Standalone
app = create_app()

# Mounted under a prefix inside an existing FastAPI app
existing_app.mount("/t1", create_app(mount_prefix="/t1"))

# Or merge routes directly into an existing app's own router
t1 = create_app()
for route in t1.routes:
    existing_app.router.routes.append(route)
```

Every dependency (`Keymaster`, `Gatekeeper`, `ModelRegistry`,
`UsageStore`) is injectable into `create_app(...)` — useful for tests, or
for sharing one `Keymaster` instance between the T1 API and another
service in the same process.

## Model registry

**Hard requirement: the T1 API only exposes models explicitly registered
in the model registry.** Nothing discovers models from HyperNix-pip's
checkpoint cache, HuggingFace Hub, or a client-supplied path. An
unregistered `model_id` always fails the same way:

```json
{"error": {"code": "MODEL_NOT_SUPPORTED", "message": "...", "details": {"model_id": "..."}}, "request_id": "..."}
```

### Example registry data

`hypernix/t1api/data/model_registry.example.json` ships the nine models
from the original T1 API spec (HyperNix 1, Ryiver 1, nanoNix, ...). The
spec is explicit that these are placeholders, so every seed entry loads
with `status: "example"` / `is_example_entry: true` and **is invisible**
to `GET /models`, `require()`, etc. unless:

- `T1_ENABLE_EXAMPLE_MODELS=1` is set, or
- the registry is constructed with `ModelRegistry.load(include_examples=True)`.

This is what makes the "only registered models" rule mean something in
practice — until you swap in real entries, the registry is empty from the
API's point of view, not silently populated with spec placeholders.

### Adding real models

Data change, not a code change. Either:

1. Write a JSON file in the same shape as `model_registry.example.json`
   and point `T1_MODEL_REGISTRY_PATH` (or `ModelRegistry.load(path=...)`)
   at it, or
2. Call `registry.register(entry)` from an admin-only code path (the
   registry itself doesn't check scopes — the caller must).

Required fields, per the spec: `model_id` (a stable slug — **never** a
parameter-count string, e.g. `nanonix-mini-lite` not `85b-25.25b`),
`display_name`, `version`, `total_parameters`, `active_parameters`,
`architecture`, `supported_tasks`, `availability`, `minimum_plan`,
`free_tier_available`, `api_available`, `local_available`,
`remote_available`, `context_limit`, `input_token_limit`,
`output_token_limit`, `tool_call_limit`, `pricing`, `routing_priority`,
`fallback_model`, `license`, `status`.

`routing_priority` and `fallback_model` are recorded starting in Beta 1
but not yet *acted on* — the router/cascade engine that walks the
fallback chain is Beta 2 (see [Roadmap](#roadmap)).

## Authentication

Two credential types, both accepted on every authenticated route via
`Authorization: Bearer <credential>`:

1. **Raw T1 keys** — `hypernix.keymaster.Keymaster` keys (`T1_...`
   strings), created with `gkey create` or `Keymaster.create()`.
   `POST /auth/t1/validate` checks one without consuming it for anything
   else.
2. **Scoped tokens** — short-lived, obtained by exchanging a raw key via
   `POST /auth/token`. Format: `T1S.<payload_b64>.<sig_b64>`, HMAC-SHA256
   signed with `T1_TOKEN_SECRET`. Deliberately *not* a full JWT — no new
   required dependency for something this small. A token's scopes must be
   a subset of the underlying key's scopes (narrowing only, never
   widening). Rotating `T1_TOKEN_SECRET` invalidates every outstanding
   token immediately — that's the revocation story for a stateless token.

### Which key families this server accepts

T1 v1.0.26.8.1.0 accepts T2 keys alongside T1 ones. Two switches control
which spellings a given deployment will take, and both default to on —
the release's whole point is that existing clients keep working:

| Setting | Default | Effect when `0` |
| --- | --- | --- |
| `T1_ACCEPT_T2_KEYS` | `1` | T2 and T2S keys are refused. For a deployment staying strictly T1 through a migration window. |
| `T1_ACCEPT_T1_KEYS` | `1` | The bare T1 spelling is refused. For a deployment that has finished moving to T2. |

Setting both to `0` is refused at startup rather than served: nothing
could authenticate, and every request would then fail with a message
about the key rather than about the configuration.

`T1_ACCEPT_T1_KEYS=0` narrows the accepted **spelling**, not the key
store. A T2 key authenticates against the T1 key behind it either way, so
turning it off orphans no existing key — it requires the holder to present
the T2 form, and the refusal says exactly that. Wrap an existing key with
`T2KeyGenerator.from_t1(key, access_level=...)`.

One caveat worth stating plainly: **admin authority comes from the key
store** (key type plus scopes), not from the T2 password component. A key
wrapped from an admin T1 key is still an admin. That is what makes
`T1_ACCEPT_T1_KEYS=0` survivable — otherwise turning it on with only a T1
admin key in hand would lock you out with no way to change the setting
back. `install-t1.sh` handles this by handing you the T2 form of the
admin key it minted whenever you choose the T2-only policy.

### Minting keys in each format

`gkey create -v` chooses which spelling a new key is issued in, and
`gkey version` reports what this install can mint:

```bash
gkey create -v v1                       # T1_…            (default)
gkey create -v v2 --level 5             # T2_…-5
gkey create -v v2 --type admin          # T2_<password>_…-9
gkey create -v v2short                  # T2S_…-1         for HyperLink
gkey version                            # package, T1 API, and key formats
```

| `-v` | Prefix | What it is |
| --- | --- | --- |
| `v1` | `T1_` | The long-standing key. Accepted everywhere. |
| `v2` | `T2_` | Access level 1–9, optional admin password, SSPKID. |
| `v2short` | `T2S_` | A 26-character body so it can be typed. Never an admin. |

`v2.1` (T2C) is named but not issuable, and asking for it says so rather
than reporting an unknown version.

The mechanism is worth stating because it explains the constraints: **a
v2 key is a spelling of a v1 key, not a separate credential.** `gkey`
mints the key into the store in its v1 form and then presents it in the
requested one; the server converts it back on every request. So both
spellings of a key work, `gkey revoke <key-id>` kills both, and the
access level and password ride on the presentation rather than on the
stored record.

Two consequences fall out of that:

- A `v2short` key's body length is fixed at mint time. Presentation
  cannot shorten a body that is already in the store, so `--body-len`
  conflicting with `v2short` is refused rather than quietly ignored.
- `--type admin -v v2` attaches a password component, but the authority
  it grants comes from the key store, not the password. The two are
  checked together: `gkey` refuses to present a non-admin key in the
  admin form, because a key that looks administrative in its prefix and
  is refused by every admin endpoint is worse than one that never
  claimed to be.

Every impossible combination is refused *before* the key is minted. A key
created and then found unpresentable would still be in the store — valid,
usable, and known to nobody, since the operator saw only an error.

### Billing keys (T2P), and refusing them

A **T2P** key is an ordinary T2 key with a billing binding attached, so a
key can be issued to someone who pays for their own usage rather than
drawing on the operator's budget.

Two things are deliberately not in the credential:

- **No card data, anywhere.** A binding holds provider-issued references —
  a customer token and a method token — and the store refuses anything
  shaped like a card number at the boundary. This server is not in the
  cardholder data path.
- **No binding in the key.** Keys get pasted into terminals and config
  files and land in shell history. The key says only *that* it is
  billing-bearing; the server looks the binding up by key ID.

A T2P key is **never an administrator**. Spending money and reconfiguring
a server are separate authorities with no reason to travel together.

#### A server does not have to accept one

Somebody else's payment arrangement is somebody else's business
relationship. `T1_BILLING_KEY_POLICY`:

| Policy | Behaviour |
| --- | --- |
| `allow` (default) | Accept the key and use its binding. Nothing changes for existing servers. |
| `deny` | Refuse with `402 BILLING_KEY_REFUSED`, pointing at `T1_PAYMENT_URL` — for an operator who sells access through their own site. |
| `separate` | Accept the request but never bill the authenticating key. Payment must arrive as a distinct T2P key in `X-Payment-Key`. |

`separate` exists so the credential that identifies a caller and the one
that spends money are different objects with different lifetimes: either
can be rotated without disturbing the other, and a leaked auth key does
not spend anything.

The policy is enforced at **authentication**, not at charge time.
Refusing after the work is done is a refund, not a policy — the operator
has already paid for the inference.

#### Spend caps

A cap is in currency, not tokens, and is checked against the *estimated*
cost before the work runs, in the same cascade that already tracks
per-key cost. Revoking a key releases its binding: a binding that
outlives its key is a standing authorisation to charge someone for a
credential that no longer exists.

### The first key a new server has

A fresh server starts with an empty key store, and every route that could
put something in it is admin-only. That is a closed loop, and it is why
`waiter hyperlink pair` could not be run on a new install: pairing is
admin-only and there was no admin.

So a new server issues itself one credential and prints it, once:

```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃  First start — here is a key to set this server up with.     ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    T2_Nc2npsJZgGXD_kNg7ZHWo2TZgsT9yggwWAyP3kn@[[*^/2-9

    Works only from this machine, and only for 72 more hours.
```

Three limits are what make printing an admin key acceptable:

| Limit | What enforces it |
| --- | --- |
| **Loopback only** | Checked on every request against the address the network policy already resolved. A key that leaks off the box authenticates nowhere — 403 `bootstrap_key_is_local_only`. |
| **Three days** | The key store's own expiry. Nothing extra runs to enforce it. |
| **Once** | Minted only when there is no live bootstrap key, so restarts do not accumulate admin credentials. On day four, an expired one is replaced rather than leaving you locked out. |

It is a T2 admin key because you are going to type it into `waiter`, and
because admin on a T2 key is carried by the password component — so the
credential itself says which one is the powerful one.

```bash
waiter serv -A -I http://127.0.0.1:8000 -K '<the key>' -L
waiter hyperlink pair --label "my iPhone"       # now works on a new server
```

Then mint a real key and let this one expire:

```bash
gkey create --type admin --scopes admin,read,write
```

`T1_BOOTSTRAP_KEY=0` turns it off for a deployment that provisions
credentials some other way.

The banner does not print a port unless the deployment was configured
with one (`T1_HYPERLINK_PUBLIC_URL`): uvicorn owns the bind address, so
anything else would be a guess printed as an instruction.

### Using a T2S key

A T2S key is 26 typeable characters, meant for a phone. Two things about
it surprise people, and both are by design:

**It must be minted on the server.** A T2 key is a spelling of a T1 key,
not a separate credential — it is converted back to its T1 form and
looked up in the key store. A key produced by `T2KeyGenerator.generate()`
belongs to no key store and authenticates as nothing, however well-formed
it is. Mint one with:

```bash
gkey create -v v2short --scopes read,write
```

**It can never be an administrator.** Admin authority rides on the
password component of a T2 prefix, and the T2S format has no room for
one. Widening the underlying key's scopes does not change this. So a T2S
key cannot mint pairing codes (`POST /hyperlink/pair` is admin-only);
pair from the PC with an admin key, and use the T2S key afterwards.

Outside HyperLink a T2S key is further narrowed to read and non-admin
write, whatever the underlying key allows. That narrowing is what makes a
typeable credential acceptable rather than a liability.

Both failures name themselves. An unregistered key gets `AUTH_INVALID_KEY`
with `details.reason = "not_in_key_store"` and the command that mints a
real one; the admin refusal gets `AUTH_ADMIN_REQUIRED` with
`details.reason = "t2s_is_never_admin"` and the route that works.

### Admin rotate / promote

`POST /auth/t1/admin/rotate` is admin-only (`AUTH_ADMIN_REQUIRED`
otherwise) and implements the spec's "convert a normal T1 token into an
admin token only when the authenticated user has the required
permission": the *requester* must already hold an admin-scoped key. It
rotates the target key (never mutates a live key's scopes in place —
consistent with how `Keymaster.rotate()` already treats rotation as
"replace, don't mutate") and, if `promote_to_admin: true`, reissues it as
`KeyType.ADMIN`.

### Undoing an authentication change

```
POST /t1/auth/undo      reverse the last reversible change
POST /t1/auth/redo      re-apply the one just undone
GET  /t1/auth/history   what is on record, and whether undo/redo is possible
```

(Also reachable as `/auth/t1/undo`, `/auth/t1/redo`, `/auth/t1/history`.)

A rotation is recorded with the previous key material, so undoing it puts
the old key back — **without changing the key ID**, so anything holding a
reference to the key keeps working. Redo re-applies the rotation. A new
operation clears the redo stack: once history diverges, replaying the old
future produces a state nobody asked for.

The history stores key material for rotations, and encrypts it with
Keymaster's own Fernet when the `security` extra is installed. `GET
/t1/auth/history` never returns it — `describe()` withholds the secret
fields.

Two refusals worth knowing:

- **Nothing to undo** (`404`) — no reversible change on record.
- **The key is gone** (`409`) — the history outlives the keys it refers
  to, so a rotation whose key has since been revoked cannot be reversed.
  The entry is left in place rather than silently discarded.

Recording is best-effort: a rotation that succeeds is not failed
afterwards because the history could not be written. The new key is
already in the response, and reporting a failure that did not happen
would lose it.

## Quota & usage

Beta 1 ships **basic per-key/per-model usage tracking**, not the full
routing/cascade engine (Beta 2). What's enforced today:

- Every usage record validates its `model_id` against the registry first
  — you cannot record usage against an unregistered model.
- A model's allowance is **fully exhausted** the moment *either* its input
  token cap *or* its output token cap is reached — independently, not
  averaged. `assert_not_exhausted()` raises `MODEL_QUOTA_EXHAUSTED`;
  `GET /usage/remaining` reports it without raising.
- Accounting is per-key **and** per-model — exhausting one model never
  touches another model's quota, matching the spec's "each model has
  independent accounting."
- The reset window is a fixed rolling period (`T1_USAGE_RESET_PERIOD_SECONDS`,
  default 24h) — a config value, not hard-coded logic.

**Not yet implemented** (Beta 2): automatic fallback when a model is
exhausted, plan-aware cascade hierarchies (free vs. paired-plan fallback
chains), manual-selection-of-an-exhausted-model handling beyond the raw
error, multi-tier reset windows.

## Model routing & quota cascade

`hypernix.t1api.routing` — not in the spec's literal endpoint list, but
its "MODEL ROUTING" / "QUOTA CASCADE" sections describe the engine in
detail, so Beta 2 gives it one: `POST /models/route`.

```bash
curl -X POST $HOST/models/route -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"plan": "free", "input_tokens": 2000}'
# → {"model_id": "nanonix-mini-plus", "reason": "...", "cascade_position": 0, ...}
```

Cascades are data, not code — `t1api/data/routing_policies.example.json`
ships the spec's own free-tier and paired-plan examples verbatim (short
prompts route to `nanonanonano-n3`, long prompts to `nanonix-mini-lite`,
then `nanonix-nano`, then `nanonanonanonano-n4`; the paired plan falls
back to `nanonix-mini` instead of `nanonix-mini-lite` after N^3). Point
`T1_MODEL_REGISTRY_PATH`/a custom `RoutingTable.load(path=...)` at your
own file to replace it.

Manual selection (`model_id` set in the request body) never silently
substitutes a different model — an exhausted model raises
`MODEL_QUOTA_EXHAUSTED` unless `automatic_fallback: true` is set, in
which case it walks the cascade starting after the requested model's
position. Every model, whether reached automatically or manually, still
goes through `ModelRegistry.require()` — a routing request can't name an
unregistered model any more than any other endpoint can.

## Servers

`hypernix.t1api.servers` — tracks which servers exist and whether
they're trusted. A server starts `untrusted` on registration; only an
admin can promote it (`PATCH /servers/{id}`) to `trusted`, or mark it
`local` at registration time for the operator's own loopback/Tailscale
address. `require_trusted()` is the enforcement point other subsystems
(module sync) call before treating a server as a valid target — see
[Security](#security) for the SSRF guard on `address`.

## Modules

`hypernix.t1api.modules` — module creation, local upload, remote-source
*registration* (not fetching — see below), versioning (`(name, version)`
is unique), and sync-tracking against the server registry.

**What "sync" means in this beta**: `POST /modules/{id}/sync` queues a
`module_sync` job. The handler checks the target server is trusted
(`ServerRegistry.require_trusted`) and then records the association
(`deployed_servers`) — there is no real network transport pushing module
bytes to a live remote server, because Beta 2 has no second real server
to test that against safely. The trust-gating and job-tracking pipeline
is real and tested end-to-end (`tests/test_t1api_http_beta2.py`); actual
byte transport is Beta 3 scope.

**What "remote upload" means in this beta**: `POST /modules/upload/remote`
validates the source URL (SSRF guard, same as server registration) and
marks the module `pending_fetch`. It does not fetch the URL. Fetching is
deliberately not wired to a job kind yet — there's no safe way to exercise
an actual outbound HTTP fetch in the sandbox this was built in, and doing
it without testing it felt worse than shipping the validation/tracking
half honestly and leaving the fetch itself for Beta 3.

The registry never executes anything it stores — see the module docstring
in `t1api/modules.py` for why that's a hard design constraint, not an
oversight.

## Jobs

`hypernix.t1api.jobs` — `queued → running → succeeded|failed|cancelled`,
exactly the state machine the spec specifies. Job *kinds* are a plugin-
style handler registry (`JobQueue.register_handler`); submitting an
unregistered kind returns `NOT_SUPPORTED` rather than inventing behavior.
Beta 2 registers exactly one real handler, `module_sync` (composed in
`t1api/app.py` from `ModuleRegistry` + `ServerRegistry`, since it's the
one place allowed to depend on both).

Execution is a small in-process `ThreadPoolExecutor`, not an external task
queue — consistent with "SQLite for development" scale. Cancellation is
cooperative (`threading.Event`) and was tested against a real
in-flight background job, not just simulated.

## Events

`hypernix.t1api.events` — an in-process pub/sub bus. `GET /events`
matches the spec's literal endpoint (polling, with `since_id`/`type`/
`limit` filters). `GET /events/stream` is an addition beyond the spec's
list — a Server-Sent-Events live tail, since "event streaming" reads most
naturally as a stream and a pure poll can't be that.

Every job state transition auto-publishes a `job.<status>` event with no
per-handler code required (`JobQueue(event_bus=...)`); server/module
mutations publish from their routers directly
(`server.registered`, `module.uploaded`, etc). `GET /events` therefore
gives a unified timeline across all of Beta 2's subsystems, not one feed
per subsystem.

## Billing

`hypernix.t1api.billing` — **an internal ledger, not a payment processor
integration.** There is no Stripe/card-network call anywhere in this
module. A "payment token" is an admin-minted, single-use redeemable
credit code (think: a gift-card code), matching the spec's own wording
("payment token"). Real money-in (charging an actual card) is a
different, much larger integration this spec doesn't describe and this
implementation doesn't attempt.

- `POST /billing/payment-token` (admin-only) mints a token and returns
  the raw value exactly once — it is never retrievable again; only its
  SHA-256 hash is persisted.
- `POST /billing/redeem` credits an account and marks the token spent;
  redeeming twice returns `PAYMENT_TOKEN_ALREADY_REDEEMED`.
- `POST /billing/add-balance` (admin-only) is a direct credit with no
  token trail — the most trusted operation in this module.
- Every transaction is masked in API responses (`txn_abcd1234…`) — see
  `Transaction.to_dict()` / `PaymentTokenRecord.to_dict()`.
- `BillingLedger.charge()` exists (a debit that raises
  `INSUFFICIENT_BALANCE` rather than going negative) but is **not**
  automatically wired to usage metering yet — cost-per-request billing
  integration (the spec's "actual cost"/"estimated cost" usage endpoints)
  is Beta 3 scope.

## Keys, plans, and assignment

`GET /keys`, `POST /keys/import`, `POST /keys/assign` — and the thing they
imply that matters more than the endpoints.

**A plan is not something a client can name.** It is a property of an
*assignment* an administrator records against a key:

```bash
waiter keys --assign <KEY_ID> --plan paired --account acct-42
waiter keys --assign <KEY_ID> --models nanonix-nano,nanonix-mini-lite
```

`POST /models/route` then resolves the plan from that assignment. Passing
`plan` in the request body is an *assertion*, not a selection: matching is
accepted, mismatching returns `AUTH_INSUFFICIENT_SCOPE`. That makes the
body field useful as a guard ("fail if this key isn't on the plan I
expect") and useless as an escalation.

`allowed_models` narrows a key to a subset of the registry. Every entry is
validated against the registry when the assignment is written, so an
assignment can never smuggle in a model id the registry never heard of.
The narrowing is checked on manual selection *and* on the model automatic
routing lands on — the cascade is plan-shaped and the narrowing is
key-shaped, so the two can disagree and the outcome is checked too.

**No endpoint here ever returns a raw key.** `GET /keys` returns masked
ids and metadata; `POST /keys/import` returns counts and masked ids even
though it must read key material to do its job. The only endpoint in the
entire API that hands back a key string is rotation, which mints a new
one.

Division of responsibility: `hypernix.security.keymaster` owns key
*lifecycle* (creation, the raw string, encryption at rest, rotation,
revocation). `t1api.keys` owns T1-specific *assignment* in its own table.
A key can be rotated without this module knowing, and the assignment
survives, because it is keyed by `key_id`.

## Remote multi-server deployment

Beta 2's module "sync" was bookkeeping: it trust-gated a target and
recorded that a sync happened, moving no bytes, and said so. Beta 3 moves
the bytes.

```bash
waiter servers --register deploy-01 --address https://deploy-01.example.com
waiter servers                          # note the server_id
# promote it — an admin action, audited, and the gate every push checks
waiter security --help                  # (trust promotion is PATCH /servers/{id})
waiter deploy <MODULE_ID> --to <SERVER_ID_1>,<SERVER_ID_2> --wait
```

Four rules, each enforced in code rather than documented as an
expectation:

1. **The target must already be trusted.** A push goes only to a server an
   admin promoted to `trusted`/`local`. The caller names a *server_id*;
   the address comes from the registry. There is no code path anywhere
   that pushes to a URL a client supplied.
2. **Every transfer is authenticated and integrity-checked.** Pushes carry
   an HMAC-SHA256 signature over `method|path|timestamp|body-digest` using
   `T1_DEPLOY_SECRET`, plus the payload's SHA-256. The receiver
   (`POST /modules/receive`) recomputes both. A signature that is missing,
   stale (outside a 300-second window), or wrong is rejected before the
   body is stored. The sender then compares the digest the receiver echoes
   back: a cheerful `200` with the wrong digest fails the deployment.
3. **Bytes are never executed, imported, or interpreted** — on either
   side. A module is an opaque blob with a checksum. That is the whole
   answer to "prevent arbitrary remote code execution through module
   upload/deployment": there is no deserialization step to attack.
4. **Fetches are bounded and pinned.** `POST /modules/{id}/fetch` stages a
   *registered* remote source — the URL comes from the registry entry that
   `POST /modules/upload/remote` validated, never from the job payload.
   Redirects are refused outright, because following one is precisely how
   an SSRF check gets bypassed: the first hop passes validation, the
   second goes wherever the attacker wants.

`POST /modules/receive` authenticates with the shared deployment secret
rather than a T1 key, because a peer server is not a user and giving every
deployment target an admin key just to receive bytes would be a far larger
grant than it needs.

## Cost, estimates, and forecasts

`GET /usage/cost`, `POST /usage/estimate`, and the cost half of
`GET /usage/history`.

Every number comes from two places and nowhere else: recorded usage events
and the registry entry's `pricing` block. There is no second price list — a
model's price is a registry fact, so changing it is a registry change, and
a model that isn't registered has no price and cannot be costed.

### Reporting what was actually spent

`POST /usage/report` (Beta 4) is the counterpart to routing. The T1 API is
a control plane with no inference endpoint, so for any client that runs the
model itself the server would otherwise never learn what the call cost —
per-model counters would stay at zero, the quota cascade would never
advance past its first model, and every per-model limit would be
unenforceable in practice. `UsageMeter.record` existed from Beta 1; Beta 4
gives it an HTTP surface.

Three rules make it safe to expose to every authenticated key:

- **Usage is recorded against the caller's own key.** There is no
  `key_id` field in the request body. A body-supplied one would let any
  valid key burn another key's quota — the same class of mistake as Beta
  2's client-supplied `plan`.
- **The model must be registered *and* allowed for that key.** A report
  can't invent a model or a budget line.
- **Counts are non-negative and capped.** A report can add usage, never
  subtract it. A client that could report negative tokens could refund
  itself quota, which would make every limit in the system advisory.

A report that pushes a model past its cap still succeeds and is recorded —
the tokens really were spent — and the response says `exhausted: true`. The
refusal belongs on the *next* `POST /models/route`, not on the accounting
for work already done.

Two honesty rules shape the API:

- **Estimates are labelled estimates.** `POST /usage/estimate` records
  nothing and reserves nothing. Omitting `output_tokens` gives an *upper*
  bound using the model's `output_token_limit`, so an estimate is never an
  optimistic guess a caller could be surprised by.
- **Forecasts state their basis.** A forecast returns the observation
  window it extrapolated from and a confidence band, because "you will
  spend $340 this month" derived from eleven minutes of history is a
  number that should arrive with that context attached.

Usage recorded against a model that has since left the registry is still
reported — priced at zero and listed in `unpriced_models`, so the gap is
visible rather than silently folded into the total.

```bash
waiter cost --range 30d --group-by model_id --forecast
waiter cost --estimate-model nanonix-nano --input-tokens 5000
```

## Audit log

`GET /audit`, admin-only, and reading it is itself audited — otherwise
"who has been reading the audit trail" is the one question it cannot
answer.

Three categories are recorded: **administrator actions** (anything gated by
`require_admin`), **security events** (auth failures, scope denials,
rate-limit trips, blocked IPs, SSRF and path-traversal rejections, mTLS
failures), and **state-changing writes**.

What is never recorded is secrets. `mask_identifier` is the only way an
identifier reaches a record, and `scrub_details` drops any field whose
*name* looks secret-ish — `key`, `token`, `secret`, `password`,
`authorization`, `dsn`, `credential` — regardless of what the caller
passed. That is deny-by-name at write time, so a future call site that
accidentally hands over a raw key cannot write it to disk. Identifiers
that merely *look* like secrets by name (`key_id`, `payment_token_id`) are
carved out, because stripping them would remove exactly what makes a
record useful.

An audit write never takes down the request it describes: `record()`
catches and logs storage failures. `record_strict()` exists for the
handful of cases where an unrecorded success is worse than a failed
request.

```bash
waiter audit --category admin --limit 20
waiter audit --outcome denied
```

## Rate limiting

Middleware, running **before** the route handler — which is what the
spec's "apply rate limits before expensive model operations" means in
practice. A limiter that runs after routing and metering have already
happened has not protected anything.

Two algorithms, because they answer different questions:

- A **token bucket** answers "may this caller make a request right now?",
  allowing a short burst above the steady rate. That is the right shape
  for interactive clients like the waiter TUI, which idles and then fires
  several calls when you open a pane.
- A **sliding window** answers "has this caller exceeded N in the last M
  seconds?" with no burst allowance. That is the right shape for the
  forced limits an operator sets with `waiter -r`, where a hard ceiling is
  the entire point.

A request is checked against every rule that matches it and denied by the
first refusal, so a caller is bounded by both their key's rate and their
address's rate — neither can be escaped by varying the other. Expensive
endpoints declare a higher `cost`, so one module upload consumes more
budget than ten `GET /health`s.

Forced limits (`waiter -r`, `POST /security/limits`) apply *in addition
to* the configured rules, so `-r` can only ever tighten.

Callers can see their own remaining budget at `GET /security/rate-limits`
and back off before a 429 rather than after one.

**Multi-worker caveat, stated rather than papered over:** limits are
per-process. Four uvicorn workers means a configured 120/min behaves as
480/min in aggregate. Either run one worker, or set the configured value
to your intended limit divided by the worker count.

## Network policy

`GET/POST/DELETE /security/network*` — the server side of `waiter`'s
`-B`/`-W`/`-a` flags, which Beta 1/2 could only record locally.

The decision order is deliberate and the tests pin it:

1. **Blocklist wins over everything.** A blocked address is denied even if
   it also appears on the allowlist — a later allowlist entry must never
   silently un-block something an operator explicitly blocked. To restore
   access you *appeal* (remove the block); that is what an appeal is.
2. **Allowlisted → allowed**, regardless of `allow_unlisted`.
3. **Neither** → allowed only if `T1_ALLOW_UNLISTED_CLIENTS` is on.

That third case is the design principle's own bullet — "if the non
whitelisted user/person trying to get access is both, not black listed and
that the server allows non whitelisted users to connect" — which is why
`allow_unlisted` is a first-class server-side setting rather than an
implied default.

Entries are single addresses or CIDR ranges, matched with `ipaddress`, so
`10.1.2.3` matches a `10.0.0.0/8` rule without string prefix guesswork.
Anything unparseable is rejected at write time rather than stored as a
rule that can never match. Blocking a range that covers your own address
is refused: locking yourself out has no undo through this API.

```bash
waiter security --block 203.0.113.9 --reason abuse
waiter security --allow 100.64.0.0/10 --reason tailnet
waiter security --appeal 203.0.113.9
waiter security --allow-unlisted off          # allowlist-only mode
```

## TLS and mTLS

Two genuinely different deployments, both supported explicitly because
conflating them is the usual way mTLS ends up decorative.

**Direct termination** — uvicorn holds the certificates.
`TLSSettings.uvicorn_kwargs()` produces exactly what `uvicorn.run()` needs,
including `ssl_cert_reqs=CERT_REQUIRED`. The client's certificate is
verified by the TLS stack before a byte of HTTP is parsed. Strongest
position, and the one this module steers toward.

**Reverse-proxy termination** — nginx/Caddy/Traefik holds the certificates
and forwards `X-Client-*` headers. The handshake happens before the
request reaches Python, so the app can only learn about the certificate
from headers, and that is trustworthy only if:

- the connection genuinely came from a trusted proxy (`T1_TRUSTED_PROXIES`),
  because otherwise any client able to reach the app directly can simply
  *send* `X-Client-Verify: SUCCESS`; and
- the proxy said verification actually succeeded — a present-but-unverified
  certificate is a failure, not a pass.

Getting the first wrong turns mTLS into a header anyone can set, so the
verifier **refuses to trust proxy headers at all when `T1_TRUSTED_PROXIES`
is empty**. That combination is a misconfiguration and it fails closed.
`waiter doctor` reports it.

`/health` is exempt from the client-certificate requirement so load
balancers keep working; it exposes nothing but liveness.

Optional allowlisting by certificate subject or fingerprint is supported
for the "only these three machines" case, checked *in addition to* CA
verification and never instead of it. Fingerprints are normalized before
comparison (`AA:BB:CC`, `sha256/aabbcc` and `aabbcc` all match), because
proxies format them inconsistently and an allowlist that silently never
matches is worse than none.

See `examples/t1api/nginx.conf` for a working proxy configuration.

## PostgreSQL

"SQLite for development, PostgreSQL for production", switched with one
environment variable:

```bash
pip install 'hypernix[t1api-pg]'
export T1_DATABASE_URL=postgresql://t1:...@localhost:5432/t1api
```

That moves **every** store — usage, servers, modules, jobs, billing,
audit, network policy, key assignments — and changes nothing else. It
works because the portability lives in one place (`t1api/db.py`) rather
than being sprinkled through five modules as `if postgres:` branches: a
connection wrapper normalizes placeholders (`?` → `%s`, skipping string
literals), row-by-name access, DDL dialect, and transaction/close
semantics, so every store's queries are written once against one contract.

Connection pooling comes from `psycopg_pool` when installed; without it
the backend opens a connection per call, which is the same short-lived
model SQLite already uses. Correctness never depends on the pool being
present, only throughput.

Schema creation is idempotent and runs at startup, including the online
migration that adds Beta 3's `user_id`/`account_id` columns to an existing
Beta 1/2 `usage_events` table — an upgrade in place, not a dump and
reload.

## Production deployment

`T1_ENVIRONMENT=production` turns on **configuration validation**:
`create_app()` refuses to start when the deployment is unsafe, and lists
every problem at once rather than one per restart.

```
T1APIError: [CONFIG_INVALID] This configuration is not safe for a production deployment:
  - T1_TOKEN_SECRET is not set. Scoped tokens would be signed with an ephemeral...
  - T1_DATABASE_URL is not set. SQLite is the documented development backend...
  - No TLS configured (T1_TLS_CERTFILE/T1_TLS_KEYFILE) and not marked as running behind...
```

A bad production config should fail the deploy, not surface later as a
puzzling 500. The same list is available without the raising — over HTTP
at `GET /status` (`production_warnings`), or from a client:

```bash
waiter doctor          # exits non-zero if a production server has warnings
waiter smoke           # end-to-end: auth, registry, quota, limits, authorization
waiter smoke --write   # also creates and removes a scratch module
```

Worked examples, all runnable:

| File | Deployment |
|---|---|
| `examples/t1api/run_local.sh` | Development: SQLite, loopback, docs on |
| `examples/t1api/run_tailscale.sh` | Tailnet: binds the 100.x address, allowlist-only, private deploy targets enabled |
| `examples/t1api/Dockerfile` | Production image: two-stage, non-root, CPU torch, healthcheck |
| `examples/t1api/docker-compose.yml` | API + PostgreSQL + nginx; the API is never published to the host |
| `examples/t1api/nginx.conf` | TLS termination and optional mTLS header forwarding |
| `examples/t1api/hypernix-t1api.service` | systemd unit with a real sandbox |
| `examples/t1api/.env.example` | Every variable, with why it matters |
| `examples/t1api/API-EXAMPLES.md` | Request/response for every endpoint, generated from a real server |
| `examples/t1api/openapi.json` | Exported schema |

Regenerate the last two after changing a response shape:

```bash
python scripts/t1api_examples.py
```

## The LM Studio bridge

*New in T1 v1.0.26.8.0.1.*

LM Studio serves an OpenAI-compatible API. The bridge borrows whatever
model is loaded in it — on localhost, across the LAN, or over a tailnet —
and exposes it through the T1 API.

```bash
export T1_LMSTUDIO_URL=http://localhost:1234
waiter lmstudio status      # reachable? anything loaded? CORS?
waiter lmstudio models      # what it has, loaded ones marked
waiter lmstudio chat "explain SIMD in one line"
waiter lmstudio local       # probe from *this* machine, no T1 server
```

**Why it goes through the T1 API rather than being called directly.**
Authentication, scopes, rate limiting, the audit log and usage accounting
all sit in front of this router because it is a router. LM Studio has
none of those. And LM Studio only has to be reachable from the *server*,
not from every client — which is what lets it stay bound to loopback on
the desktop while a phone still uses it.

**What it does not do.** Load, unload, or download models in LM Studio.
LM Studio owns its models; the bridge borrows what is loaded. When
nothing is loaded, the error says exactly that, with the address it asked
and the models it saw.

Two details worth knowing:

* The bridge prefers LM Studio's native `/api/v0/models` over `/v1/models`
  for the one fact the OpenAI shape cannot express: whether a model is
  actually **loaded**. `/v1/models` lists everything downloaded, and a
  chat against an unloaded model either stalls on a just-in-time load or
  fails outright.
* `waiter lmstudio status` reports the **CORS** state explicitly. It only
  matters for a browser or WKWebView talking to LM Studio directly — a
  Python client is not subject to it — but "works from curl, not from the
  app" is otherwise a long afternoon.

Over a LAN, plain HTTP means prompts cross the network in the clear, so
`T1_ENVIRONMENT=production` refuses to start with `T1_LMSTUDIO_URL`
pointing at a non-loopback, non-Tailscale `http://` address. Tailscale is
exempt: WireGuard already encrypted it.

| Variable | Default | |
|---|---|---|
| `T1_LMSTUDIO_ENABLED` | `1` | master switch |
| `T1_LMSTUDIO_URL` | — | where LM Studio is |
| `T1_LMSTUDIO_API_KEY` | — | for a reverse proxy in front of it |
| `T1_LMSTUDIO_DISCOVERY` | `0` | allow admin-triggered tailnet sweeps |
| `T1_LMSTUDIO_TIMEOUT` | `300` | read timeout, seconds |

## HyperLink

*New in T1 v1.0.26.8.0.1.*

The phone-facing surface, and the backend for the
[HyperLink iOS app](../ios/README.md): pair a device, hold conversations
that live on the PC, attach images and code, and resolve model links.

### Pairing

A long `T1_...` key is not typeable on a phone. So enrolment is a
two-step exchange:

```bash
waiter hyperlink pair --label "my iPhone"
```

which prints a six-character code and the addresses this machine answers
on. The phone posts the code to `POST /hyperlink/pair/redeem` and gets a
device token back, once.

Properties, and why each:

* **The code is not the credential.** Six characters from an alphabet
  with no `0/O/1/I/L`, ten minutes, single use, five attempts. The
  credential it produces is 32 bytes of `secrets.token_urlsafe`.
* **Tokens are stored hashed.** A stolen `hypernix.db` does not hand over
  anyone's phone. Verification is `compare_digest` over the SHA-256.
* **Revocation is per-device.** Losing a phone revokes that phone.
* **A device is never an admin.** Whatever key paired it, a device token
  cannot mint pairing codes, list other devices, or unpair one — a stolen
  phone must not be able to enrol a second one. It *can* unpair itself:
  that is the app's "sign out", and requiring an admin would leave a wiped
  phone's token valid until somebody noticed.

`POST /hyperlink/pair/redeem` is the only unauthenticated endpoint in
HyperLink. Every failure against it is audited with the client address,
because a run of failures is what guessing looks like.

### Ownership

A device's **owner is the key that paired it**, not the device id. That
one decision is what makes a conversation started on the desktop
continue on the phone, while another operator's stays invisible.
Unpairing a phone does not orphan the threads started on it.

### Sessions

Conversations live on the server (`/hyperlink/sessions`), append-only,
with the answering model recorded per message — people switch models
mid-thread and "which model said this" is the first question asked when
re-reading one.

Context is trimmed by **token budget**, not message count: a fixed "last
20 messages" either overflows a small context window or wastes a large
one. The system prompt is always kept and charged first, so a long one
eats into history rather than being dropped — dropping it changes the
assistant's behaviour mid-conversation.

`GET /hyperlink/sessions/{id}/messages?after_seq=N` is the incremental
sync hook.

### Attachments

Content-addressed by SHA-256 (`/hyperlink/files`). Re-sending the same
screenshot costs nothing, ids cannot be enumerated, and nothing is ever
overwritten. Deletion is reference-counted, so removing one message's
copy does not take another's bytes.

At the moment of inference each attachment is expanded differently:

| kind | becomes |
|---|---|
| image | an `image_url` part with an inline data URL — what vision models take |
| text/code | a fenced block in the message, with the filename in the fence info |
| anything else | a one-line note naming the file and its type, so the model can decline rather than hallucinate |

Content type is decided by **magic bytes first**, then the filename, then
the client's claim — a `.png` that is really a zip is labelled a zip.
Downloads are always `Content-Disposition: attachment` with `nosniff`:
this server can be reached from a WKWebView, and a stored file rendering
as HTML in the app's origin would be stored XSS.

### Endpoint advertisement

`GET /hyperlink/endpoints` returns every address this machine answers on,
ranked: Tailscale DNS name, Tailscale IP, LAN, loopback. The app tries
them in order and keeps the first that answers, so nothing has to change
when the phone leaves the house. Authenticated, despite looking
innocuous — a list of a machine's internal addresses is reconnaissance.

## Hugging Face link merging

*New in T1 v1.0.26.8.0.1.* `POST /hyperlink/models/resolve`, and
`waiter fetch`.

Somebody finds a GGUF they want. What they have is one or two URLs, and
neither is sufficient:

* a **model page** says which repository and what is in it, but not which
  of its fourteen quantisations you meant;
* a **direct download link** — the one behind the download arrow — says
  exactly which bytes, but nothing about the rest of the repo, and for a
  vision model or a split GGUF those bytes alone will not load.

Give it either or both and it returns one complete download plan:

```bash
waiter fetch https://huggingface.co/bartowski/Qwen3-8B-GGUF
waiter fetch --page <model page> --file <download-arrow link>
waiter fetch bartowski/Qwen3-8B-GGUF:Q5_K_M
waiter fetch <any link> --local --offline      # resolve here, no server
```

Three pieces of knowledge go into "so it runs properly":

1. **Split GGUFs.** `model-00001-of-00003.gguf` is one third of a model.
   The part someone happened to click gives a file llama.cpp refuses. The
   whole set is pulled, whichever part was named, and a missing part on
   the hub is reported.
2. **Vision projectors.** A VLM's `mmproj-*.gguf` is a separate file, and
   without it the model loads and then cannot see images — a much more
   confusing failure than not loading at all. The projector matching the
   weights' quantisation is preferred.
3. **Repository conflicts.** When the page and the file link name
   different repositories, that is two tabs open and the wrong one
   copied. It raises, unless `prefer` says which side wins (`file` for
   the bytes, `page` for the repository).

Accepted forms: full page URLs, `/tree/`, `/blob/`, `/resolve/` links,
`hf.co` and `hf-mirror.com`, `hf://owner/repo/file.gguf`, bare
`owner/repo`, and the Ollama-style `owner/repo:Q4_K_M` shorthand.

With no network — or with `offline: true` — a plan is still built from a
direct file link alone, split part names included, marked
`metadata_from_api: false`. A phone on a bad connection should still be
able to start a download it has the exact URL for.

## The SDK

`hypernix.t1sdk` — zero dependencies beyond the standard library, because
an SDK is a *client* and must import on machines that never install the
server extra.

```python
from hypernix.t1sdk import T1Client, T1QuotaError

client = T1Client("https://t1.example.com", credential=key)

for model in client.list_models():
    print(model.model_id, model.status, model.is_routable)

try:
    decision = client.route(input_tokens=1200)
except T1QuotaError as exc:
    print("exhausted:", exc.code, "retry after", exc.retry_after)

job = client.deploy_and_wait(module_id, ["srv-1", "srv-2"])
```

Errors are a hierarchy mapped from the server's stable codes, so you catch
the category you care about (`T1AuthError`, `T1QuotaError`,
`T1NotFoundError`) rather than matching strings — while `exc.code` keeps
the exact code for when the distinction matters. Every exception carries
the server's `request_id`, which is the single most useful thing to put in
a bug report.

Retries are automatic on connection failures, 502/503/504 and 429, honour
a server-sent `Retry-After` over the backoff curve, and are **never**
applied to a non-idempotent POST: replaying `POST /billing/redeem` after a
timeout could look like a double redemption, so the SDK would rather
surface the timeout.

mTLS and private CAs:

```python
from hypernix.t1sdk import T1Client, TLSConfig

client = T1Client("https://t1.internal", credential=key, tls=TLSConfig(
    client_certfile="client.crt", client_keyfile="client.key",
    ca_certs="internal-ca.pem",
))
```

`client.call(method, path, ...)` is an escape hatch for an endpoint a
given SDK version doesn't wrap, with the same error handling and retries —
so a deployment running a newer server is never blocked on an SDK release.

## Endpoint reference

All responses include `request_id`. Errors use the envelope shown in
[Model registry](#model-registry) above with a code from
`hypernix.t1api.errors.T1ErrorCode`.

**Beta 1**

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | none | liveness |
| GET | `/status` | none | version, model count, storage backend |
| GET | `/config` | none | explicit allowlist only — never secrets |
| POST | `/auth/t1/validate` | none (key in body) | validate a raw T1 key |
| POST | `/auth/token` | none (key in body) | exchange a key for a scoped token |
| POST | `/auth/t1/rotate` | bearer | rotate the caller's own key |
| POST | `/auth/t1/admin/rotate` | bearer, admin | rotate/promote another key |
| GET | `/models` | none | list registry entries |
| GET | `/models/{model_id}` | none | full entry detail |
| GET | `/models/{model_id}/availability` | none | registry-level availability |
| GET | `/models/{model_id}/usage` | bearer | caller's usage for this model |
| GET | `/usage/current` | bearer | all-time + current-window totals |
| GET | `/usage/remaining?model_id=` | bearer | remaining allowance for one model |

**Beta 2**

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/models/route` | bearer | routing/quota-cascade decision (addition, not in the spec's literal list) |
| GET | `/servers` | bearer | list registered servers |
| POST | `/servers/register` | bearer | register a server (starts `untrusted`) |
| PATCH | `/servers/{server_id}` | bearer, admin | trust-level/status/capability changes |
| DELETE | `/servers/{server_id}` | bearer, admin | deregister |
| GET | `/modules` | bearer | list modules (own, or all for admins) |
| POST | `/modules/create` | bearer | create a draft module entry |
| POST | `/modules/upload/local?module_id=` | bearer | multipart file upload, activates the module |
| POST | `/modules/upload/remote?module_id=` | bearer | register (not fetch) a remote source |
| GET | `/modules/{module_id}` | bearer | module detail |
| PATCH | `/modules/{module_id}` | bearer, owner/admin | metadata/status |
| DELETE | `/modules/{module_id}` | bearer, owner/admin | delete entry + stored file |
| POST | `/modules/{module_id}/sync` | bearer, owner/admin | queues a `module_sync` job |
| POST | `/jobs` | bearer | submit a job (`NOT_SUPPORTED` for unregistered kinds) |
| GET | `/jobs/{job_id}` | bearer | job status/result/error |
| POST | `/jobs/{job_id}/cancel` | bearer | cooperative cancellation |
| GET | `/events` | bearer | poll recent events (`since_id`/`type`/`limit`) |
| GET | `/events/stream` | bearer | SSE live tail (addition, not in the spec's literal list) |
| GET | `/billing/balance` | bearer | caller's own balance |
| GET | `/billing/transactions` | bearer | caller's own transaction history |
| POST | `/billing/payment-token` | bearer, admin | mint a redeemable token |
| POST | `/billing/redeem` | bearer | redeem a token into an account |
| POST | `/billing/add-balance` | bearer, admin | direct credit |

**Beta 3**

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/usage/history` | bearer | raw events, date-ranged and filtered; non-admins are pinned to their own |
| GET | `/usage/by?group_by=` | bearer | usage by model/key/server/module/user/account/endpoint |
| GET | `/usage/cost` | bearer | actual cost, breakdowns, optional forecast |
| POST | `/usage/estimate` | bearer | what an operation *would* cost; records nothing |
| GET | `/keys` | bearer | admins see all, everyone else sees only their own |
| POST | `/keys/import` | bearer, admin | import a Keymaster export; returns masked ids only |
| POST | `/keys/assign` | bearer, admin | bind a key to a plan/account/user/servers/models |
| POST | `/modules/{module_id}/deploy` | bearer, owner/admin | push to several trusted servers as one job |
| POST | `/modules/{module_id}/fetch` | bearer, owner/admin | stage the module's registered remote source |
| POST | `/modules/receive` | HMAC (deploy secret) | inbound server-to-server push |
| GET | `/audit` | bearer, admin | the audit trail; reading it is audited |
| GET | `/security/network` | bearer, admin | allow/block entries and the unlisted-client posture |
| POST | `/security/network/blacklist` | bearer, admin | `waiter -B` |
| POST | `/security/network/whitelist` | bearer, admin | `waiter -W` |
| DELETE | `/security/network/{cidr}` | bearer, admin | `waiter -a` (appeal) |
| POST | `/security/network/allow-unlisted` | bearer, admin | open-unless-blocked vs allowlist-only |
| GET | `/security/limits` | bearer, admin | forced per-key/per-server limits |
| POST | `/security/limits` | bearer, admin | `waiter -r`; only ever tightens |
| DELETE | `/security/limits/{type}/{id}` | bearer, admin | clear a forced limit |
| GET | `/security/rate-limits` | bearer | the caller's own remaining budget |

**Beta 4**

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/usage/report` | bearer | report tokens actually consumed; always against the caller's own key |

Note the two `module_id`-as-query-parameter endpoints
(`/modules/upload/local`, `/modules/upload/remote`) — deliberate, matching
the spec's literal paths (`POST /modules/upload/local`, no `{module_id}`
in the template), unlike every other module endpoint which puts
`module_id` in the path.

Destructive operations (`DELETE /servers/{id}`, `DELETE /modules/{id}`)
require `?confirm=true` unless `T1_REQUIRE_DESTRUCTIVE_CONFIRMATION=0`.

**Full request/response examples** for every endpoint:
[`examples/t1api/API-EXAMPLES.md`](../examples/t1api/API-EXAMPLES.md).
That file is *generated by driving a real server*
(`scripts/t1api_examples.py`), not written by hand — a hand-written
example is a claim about the API, a generated one is a recording of it,
and regenerating shows the behaviour change as a diff. The exported schema
is `examples/t1api/openapi.json`; a running server also serves it at
`/openapi.json` with Swagger UI at `/docs`.

**T1 v1.0.26.8.0.1**

*The LM Studio bridge.* `base_url` is an admin-only per-request override:
useful when moving a model between two machines, and a server-side
request forgery primitive if anyone may set it.

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/bridge/lmstudio/status` | bearer | reachable? loaded? CORS? (`?discover=true` is admin + opt-in) |
| GET | `/bridge/lmstudio/models` | bearer | every model, loaded ones marked |
| POST | `/bridge/lmstudio/chat` | bearer | one completion; reply lifted out of the envelope, full one in `raw` |
| POST | `/bridge/lmstudio/chat/stream` | bearer | SSE, relayed near-verbatim |

*HyperLink.* Every path accepts a device token **or** a T1 key, except
where noted.

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/hyperlink/endpoints` | bearer | addresses this machine answers on, ranked |
| POST | `/hyperlink/pair` | bearer, **admin, not a device** | mint a pairing code |
| POST | `/hyperlink/pair/redeem` | **none** — the code is the credential | one device token, once |
| DELETE | `/hyperlink/pair/{code}` | bearer, admin | cancel an unredeemed code |
| GET | `/hyperlink/devices` | bearer, admin | paired devices |
| GET | `/hyperlink/devices/me` | device token | what this token resolves to |
| PATCH | `/hyperlink/devices/{id}` | admin, or the device itself | rename |
| DELETE | `/hyperlink/devices/{id}` | admin, or the device itself | unpair |
| GET/POST | `/hyperlink/sessions` | bearer | list / create |
| GET/PATCH/DELETE | `/hyperlink/sessions/{id}` | bearer | one session |
| GET | `/hyperlink/sessions/{id}/messages` | bearer | `?after_seq=` for incremental sync |
| POST | `/hyperlink/sessions/{id}/chat` | bearer | one turn; both messages persisted |
| POST | `/hyperlink/sessions/{id}/chat/stream` | bearer | the same turn, streamed |
| POST | `/hyperlink/files` | bearer | multipart upload |
| GET | `/hyperlink/files` | bearer | list; `?session_id=` to scope |
| GET/DELETE | `/hyperlink/files/{id}` | bearer | download (always `attachment`) / delete |
| POST | `/hyperlink/models/resolve` | bearer | merge a Hugging Face page + file link |

**0.72.4**

*Training administration.* Admin-only unless the server is in
[trusted-network mode](#trusted-network-mode); the destructive controls
need the second opt-in on top of that. See [Training](#training).

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/training/runs` | bearer, **admin or trusted** | every run, newest first; `?active=true` |
| GET | `/training/runs/{run_id}` | bearer, **admin or trusted** | one run; the job name resolves too |
| GET | `/training/runs/{run_id}/logs` | bearer, **admin or trusted** | tail of its output; `?tail=` (default 200) |
| GET | `/training/runs/{run_id}/checkpoints` | bearer, **admin or trusted** | with `exists`, so a deleted one shows |
| GET | `/training/resources` | bearer, **admin or trusted** | GPU/CPU/RAM right now |
| POST | `/training/runs/{run_id}/pause` | bearer, **admin or partial-admin** | SIGSTOP; does *not* free VRAM |
| POST | `/training/runs/{run_id}/resume` | bearer, **admin or partial-admin** | SIGCONT |
| POST | `/training/runs/{run_id}/stop` | bearer, **admin or partial-admin** | SIGTERM, so it can checkpoint |

## Trusted network mode

Off by default, and that default is the whole feature. A private address
is not consent: nothing is keyless until an administrator says so, with
`T1_TRUSTED_NETWORK=1` or `install-t1.sh --trusted-network`.

```bash
T1_TRUSTED_NETWORK=1                 # loopback + LAN + tailnet, keyless
T1_TRUSTED_NETWORK_LAN=1             # narrow it: 0 for tailnet-only
T1_TRUSTED_NETWORK_TAILNET=1
T1_TRUSTED_NETWORK_PARTIAL_ADMIN=0   # a second, separate opt-in
```

`hypernix.system.nettrust` classifies each connection as `loopback`,
`tailnet`, `lan` or `public`. The LAN ranges are enumerated explicitly
rather than taken from `ipaddress.is_private`, which is far broader than
RFC 1918 — it is true for the documentation ranges (192.0.2/24,
198.51.100/24, 203.0.113/24), for `0.0.0.0/8`, and for benchmarking and
reserved space. Treating all of that as "the LAN" would have granted
keyless access to addresses nobody would call local.

Three boundaries this mode never crosses:

**A public origin is never keyless.** `TrustPolicy.allows_keyless`
refuses `PUBLIC` before the configuration is consulted, so a hand-built
policy cannot express it either.

**A same-host reverse proxy cannot launder a caller.** nginx or caddy in
front of this process makes every request in the world arrive from
`127.0.0.1` — the *most* trusted level. An unverifiable `X-Forwarded-For`
therefore collapses the origin to public: set `T1_TRUSTED_PROXIES` before
enabling any of this. Failing closed here is what stops an operator
publishing a sofa-only server to the internet while believing otherwise.

**A bad key is not "no key".** The keyless path runs only when a request
brings no credential at all. If a rejected key fell through to it,
revoking a key would stop working from the LAN — the opposite of what
revoking means.

Partial admin never grants the `ADMIN` scope. A keyless caller gets
`READ`, plus `WRITE` when partial admin is on, so every endpoint that
checks for admin the ordinary way stays unreachable without a key. The
one surface that consults it directly is [Training](#training).

## Training

`hypernix.training.monitor` — a training run is the longest-lived and
least observable thing this package starts. It goes for hours, it is
usually launched over a connection that will not survive it, and reading
a log was the only way to know how it was going.

Two halves, deliberately separate. `ProgressReporter` is written *by* the
trainer: one atomic rewrite of a small JSON file per update, no lock and
no append, because the reader is a web request that can arrive halfway
through an epoch and a half-written status is worse than a stale one.
`TrainingMonitor` is read by everything else, and merges the file with
the actual process state — a crashed trainer leaves a record still saying
`running`, and believing it shows a healthy run that has not existed
since Tuesday. That case is reported as `stale`, with the last update's
age in `error`.

Nothing has to be wired up by the caller. `hypernix-t1 launch-script`
exports `HNX_RUN_ID` and `HNX_LOG_PATH`, and `train()` reads them, so a
detached run appears on its own:

```bash
hypernix-t1 launch-script ./train.py --name qwen-sft --detach
hypernix-t1 training                 # every run
hypernix-t1 training --show qwen-sft # one, in full
hypernix-t1 training --stop qwen-sft # SIGTERM, so it can checkpoint
```

The CLI takes no key. Its controls signal processes the invoking user
already owns and its records are files that user can already read, so
the access control is the filesystem's — the same one that decided
whether the command could run at all. Credentials are for the network,
and that path is the API below.

### Who may call the endpoints

| | admin key | trusted mode | + partial admin | public |
|---|---|---|---|---|
| read runs, logs, checkpoints, resources | ✅ | ✅ | ✅ | ❌ |
| stop / pause / resume | ✅ | ❌ | ✅ | ❌ |

Two tiers because the destructive half earns a second opt-in: killing six
hours of training is not something a device that presented no credential
gets to do because it happens to be on the same wifi. A public origin
never qualifies for either, whatever the configuration says —
`TrustPolicy.allows_keyless` refuses `PUBLIC` before the policy is
consulted. Every control is audited under `admin`, the same category as
rotating a key.

Presenting an ordinary read key does not lose you access you would have
had keyless from the same address; the stronger of the two decides.

### Two things worth saying out loud

**Pausing does not free the card.** `pause` is SIGSTOP, which freezes the
process with its GPU allocations intact. That is what makes `resume`
instant, and it is also why the card does not become available to
anything else. Both the API's `note` and the CLI say so, because it is
the thing people assume the opposite of.

**Stop is terminate, not kill.** A trainer that handles SIGTERM gets to
write a final checkpoint. Stopping a *paused* run sends SIGCONT first — a
stopped process cannot handle SIGTERM, so without it "stop" reports
success and leaves the run frozen forever. Stopping a run that has
already ended is a `409`, not a rewrite: overwriting `finished` with
`stopped` would leave the history saying an operator killed a job that in
fact completed.

`progress` and `eta_seconds` are `null` rather than `0` whenever they are
unknown — no declared schedule for the first, a run that is no longer
going for the second. A bar at 0% for a job two hours in, or `eta 6s` for
a trainer that died an hour ago, are both lies a dashboard tells
confidently.

## Configuration

See [`examples/t1api/.env.example`](../examples/t1api/.env.example) for
every variable, with a note on why each one matters. Each maps 1:1 to a
`T1APIConfig` field.

Nothing under `T1APIConfig.public_dict()` (what `GET /config` returns) can
leak a secret: it is an explicit allowlist, not "everything except a
blocklist", specifically so a newly-added secret field cannot quietly
start being served. `GET /status` reports which secrets are *set* —
booleans only, never values.

Setting `T1_ENVIRONMENT=production` additionally makes the configuration
**validated rather than assumed** — see
[Production deployment](#production-deployment).

T1 v1.0.26.8.0.1 added:

| Variable | Default | |
|---|---|---|
| `T1_LMSTUDIO_ENABLED` | `1` | the LM Studio bridge |
| `T1_LMSTUDIO_URL` | — | where LM Studio is; plaintext non-loopback is a production error |
| `T1_LMSTUDIO_API_KEY` | — | secret; reported as set/unset only |
| `T1_LMSTUDIO_DISCOVERY` | `0` | allow admin-triggered tailnet sweeps |
| `T1_LMSTUDIO_TIMEOUT` | `300` | read timeout, seconds |
| `T1_HYPERLINK_ENABLED` | `1` | the phone-facing surface |
| `T1_HYPERLINK_PUBLIC_URL` | — | a reverse proxy / tunnel address; ranked first |
| `T1_HYPERLINK_PORT` | `8000` | the port advertised to clients |
| `T1_HYPERLINK_FILES_DIR` | `~/.hypernix/hyperlink/files` | attachment blobs |
| `T1_HYPERLINK_MAX_UPLOAD_BYTES` | `67108864` | enforced on bytes read, not `Content-Length` |
| `T1_HYPERLINK_PAIRING_TTL` | `600` | how long a pairing code lives |
| `T1_HF_TOKEN` / `HF_TOKEN` | — | secret; for resolving gated repositories |

T1 v1.0.26.8.1.1 adds:

| Variable | Default | |
|---|---|---|
| `T1_BOOTSTRAP_KEY` | `1` | mint a local-only three-day admin key on a first start with an empty key store. `0` on a server that is provisioned some other way. |
| `T1_BILLING_KEY_POLICY` | `allow` | `allow` \| `deny` \| `separate` — whether a [T2P key](#billing-keys-t2p-and-refusing-them) may pay for its own request, must not, or must hand payment to a distinct key in `X-Payment-Key`. |
| `T1_PAYMENT_URL` | — | where a `deny`/`separate` server sends a caller instead. Returned in the refusal, so the error is actionable. |
| `T1_KEYMASTER_DIR` | `~/.hypernix/keymaster` | the key store. `gkey` reads the same variable, so a server and the tool that mints its keys cannot disagree about where keys live. |

## Security

- API keys/tokens are never logged. The exception-handling path logs the
  error *code*, not the credential.
- `GET /config` is an allowlist (see above).
- Scoped tokens can only narrow a key's scopes, never widen them.
- Admin operations (`/auth/t1/admin/rotate`, server trust promotion,
  billing mint/add-balance) require the *requester* to already hold
  `KeyType.ADMIN` + `KeyScope.ADMIN` — checked before anything else runs.
- **SSRF guard** (`t1api.security.validate_remote_address`): only
  `http`/`https` schemes are ever accepted; the cloud-metadata SSRF target
  (`169.254.169.254`) is always rejected; private/loopback/link-local
  addresses require explicit `allow_private=True` (surfaced as
  `allow_private_address` on server registration, `allow_private` on
  remote module source registration) — the knob a Tailscale/local
  deployment needs, without silently allowing SSRF for everyone else.
- **Path-traversal guard** (`t1api.security.sanitize_module_path`): every
  local module upload resolves its filename against a fixed storage
  directory and rejects anything that would escape it.
- **No code execution**: the module system never imports, executes, or
  interprets anything it stores — see the module docstring in
  `t1api/modules.py`.
- **Server trust gating**: `ServerRegistry.require_trusted()` blocks
  module sync to any server that isn't `trusted`/`local` — a fresh
  registration is always `untrusted` until an admin promotes it.
- **mTLS** — direct or proxy-terminated, with the trusted-proxy check that
  keeps forwarded headers from being forgeable. See
  [TLS and mTLS](#tls-and-mtls).
- **IP allow/blocklists** with an explicit unlisted-client posture. See
  [Network policy](#network-policy).
- **Rate limiting** in middleware, before the handler, with per-key,
  per-IP, per-endpoint rules and operator-forced ceilings. See
  [Rate limiting](#rate-limiting).
- **Audit logging** that is durable, queryable, and scrubs secret-shaped
  fields by name at write time. See [Audit log](#audit-log).
- **Signed module transfer** between servers, checksum-verified on both
  ends, with redirects refused on fetch. See
  [Remote multi-server deployment](#remote-multi-server-deployment).
- **Explicit confirmation** on destructive operations (`?confirm=true`).
- **Production configuration validation** refuses to start an unsafe
  production deployment, listing every problem at once.
- Security response headers (`X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, `Cache-Control: no-store`, and HSTS when TLS is on)
  are applied to every response, including error responses raised from
  middleware.

### Known limitation

**Module blobs are not encrypted at rest.** They are checksummed and
path-sanitized, and the store relies on filesystem permissions — the
systemd unit in `examples/t1api/` confines writes to a single directory
owned by the service user, and the container image runs as a non-root user
with the blob store on its own volume. Everything else that needs
protecting at rest already has it: T1 keys are encrypted by Keymaster,
payment tokens are stored only as SHA-256 hashes, and the waiter's local
config is Fernet-encrypted with `-E`. Encrypting the blob store itself is
the one Beta 3 line item deliberately left open rather than half-done, and
it is listed as such in the [Roadmap](#roadmap).

### Security audit checklist

Before exposing a deployment:
[`wiki/T1-API-Security-Checklist.md`](T1-API-Security-Checklist.md).

## Roadmap

Matches the spec's own beta breakdown exactly — nothing here is
renumbered or reinterpreted.

| Beta | Scope | Status |
|---|---|---|
| **1** | Core FastAPI server, T1 auth + scoped tokens, model registry, basic per-key/model usage tracking, `/health` `/status` `/models` + auth/usage/config endpoints, basic `waiter` CLI, SQLite, OpenAPI docs | **Shipped** |
| **2** | Module registry, module upload/sync, server registry, async jobs, event streaming, quota cascade, model routing engine, billing/payment-token support, Tailscale/local deployment guide | **Shipped** |
| **3** | Production hardening, PostgreSQL backend, complete audit logging, mTLS, advanced rate limiting, remote multi-server deployment (real module byte transport + remote-source fetch), full `waiter` TUI (curses `-G`), complete SDK, complete test suite, deployment docs, security audit checklist, production configuration validation | **Shipped** (this doc). One item is deliberately still open: module blobs are checksummed but not encrypted at rest — the store relies on filesystem permissions. See [Security](#security). |
| **4** | `hyped-pro` T1-key support against a local T1 API server; `hyped-pro` auto-displaying the current public release version; `qwen3.8-27b` registry entry | **Shipped** |

The betas end there. Releases from **T1 v1.0.26.8.0.1** onward use the
six-part scheme in [Versioning](#versioning) rather than a beta number.

| Release | Scope | Status |
|---|---|---|
| **1.0.26.8.0.1** | The [LM Studio bridge](#the-lm-studio-bridge); [HyperLink](#hyperlink) pairing, sessions, attachments and endpoint advertisement; [Hugging Face link merging](#hugging-face-link-merging); the [HyperLink iOS app](../ios/README.md) | **Shipped** |
| **1.0.26.8.1.1** | The [bootstrap admin key](#the-first-key-a-new-server-has) a fresh server issues itself; [T2P billing keys](#billing-keys-t2p-and-refusing-them) and the server-side policy that can refuse them; the three HyperLink reachability fixes (advertised port, tailnet ATS, Tailscale diagnosis) and [T2S key](#using-a-t2s-key) entry from the app; [undoable](#undoing-an-authentication-change) auth changes and durable server ids | **Shipped** |

## Design principle

> The T1 API is a controlled gateway into HyperNix. The client must
> never be trusted to decide what it is allowed to access.

The server determines which models exist, which are available, which
limits apply, which fallbacks are allowed, how much usage remains, and
what an operation costs. The client only requests an operation.

That is not a slogan; it is a list of enforcement points, and each one is
a specific function you can go read:

| The server decides… | Enforced by |
|---|---|
| which models exist | `ModelRegistry.require()` — an unregistered id is `MODEL_NOT_SUPPORTED`, always |
| which models *this key* may use | `KeyDirectory.assert_model_allowed()` |
| which plan the caller is on | `KeyDirectory.resolve_plan()` — **not** the request body |
| which fallbacks are allowed | `RoutingEngine` walking a plan-scoped, data-driven cascade |
| how much usage remains | `UsageMeter.assert_not_exhausted()` |
| what an operation costs | `CostCalculator`, priced from the registry entry and nowhere else |
| which servers may receive a module | `ServerRegistry.require_trusted()` |
| whether a client may connect at all | `NetworkPolicy.require_allowed()` |
| how hard a caller may hit the API | `RateLimiter.check()`, in middleware, before the handler |

The Beta 3 change worth calling out is the third row. Beta 2's
`POST /models/route` took `plan` from the request body, which let a client
name the most generous plan it could think of. A plan is now a property of
an administrator-recorded assignment, and a client that asserts a
different one is refused rather than believed.
