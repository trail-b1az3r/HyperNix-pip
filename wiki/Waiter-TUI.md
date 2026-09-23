# waiter — the T1 API TUI/CLI

`waiter` is the official client for the [HyperNix T1 API](T1-API.md).
Console script, installed with base `hypernix` (no `[t1api]` extra
needed — it's a client, built on stdlib `urllib`, not the server).

**Status: Beta 3 — complete.** Every flag in the spec's `serv` list is
wired to real behaviour, and the full curses TUI (`-G`) ships. `waiter`
gets its model list, availability, limits, quota and fallback chain from
the server at request time — it never hard-codes a model list, and it
never decides what you may use.

As of Beta 3 the client sits on [`hypernix.t1sdk`](T1-API.md#the-sdk)
rather than being a second implementation of the same HTTP calls; the
subcommand surface is unchanged.


## Versions, help, and connection failures

```bash
waiter version            # package, waiter, T1 API, key formats, the server
waiter version --json
waiter help               # list the longer help topics
waiter help connect       # pointing waiter at a server
waiter help keys          # which credential to use
waiter help hyperlink     # pairing a phone
waiter help find          # locating a server you did not configure
```

`waiter version` prints four numbers that move independently — the
package, waiter's protocol version, the T1 API (with the oldest client it
speaks to), and the key formats — plus the connected server's version
when one is reachable. Comparing them is how "my key is refused" and "my
client is too old" get diagnosed. The server line is fetched
unauthenticated, and omitted rather than guessed at when the server
cannot be reached.

### When a connection fails

waiter explains an unreachable server instead of restating the errno. It
names the address, **where that address came from**, and the command that
fixes it:

```
✗ Could not reach http://127.0.0.1:1234/hyperlink/pair ([Errno 111] Connection refused)
  Address from: the saved config (~/.hypernix/waiter/waiter.config.jsonl)

Port 1234 is LM Studio's default, not the T1 API's (8000).
  `waiter lmstudio` reaches it through the T1 server, not directly.
  If you meant the T1 API:  waiter serv -A -I http://127.0.0.1:8000 -K <key>
```

The distinction that matters: **LM Studio (1234) and Ollama (11434) are
model backends the T1 server talks to, not addresses waiter should point
at.** `waiter lmstudio` reaches LM Studio *through* the server. Pointing
waiter straight at one gives a connection refused, or a puzzling 404 from
something that is not a T1 API.

On failure waiter probes whether the port accepts a connection at all,
and if it does, what answers there — a T1 API, an OpenAI-compatible
backend, or some other web server — so the three cases get three
different answers:

| What it found | What it says |
| --- | --- |
| Nothing listening | Nothing is listening on `host:port`, and how to start the server |
| A known bridge port | That port belongs to LM Studio / Ollama, and the T1 address to use instead |
| Something that is not a T1 API | What answered, and how to re-point waiter |

These probes deliberately bypass the proxy environment: a diagnostic asks
whether *this* host is up, and a proxy in between answers a different
question. Ordinary API traffic still honours `HTTP_PROXY` as before.

## Quickstart

```bash
waiter serv -A -I "https://myserver.ts.net:8000" -K "T1_..." -E
waiter models
waiter model nanonix-nano
waiter usage
waiter route --plan free --input-tokens 2000
```

`-A` validates the key against the server and saves the config in one
step. `-E` seals the key as a [v2.1 key](T1-API.md#v21-t2c-keys-and-rotorvault)
when the server supports it, so the config holds a kit and the wire only
ever carries that day's key. It also encrypts the config at rest (Fernet,
the same pattern as `hypernix.keymaster`'s own key-storage encryption).
Against an older server, or with a key format that cannot be sealed, `-E`
keeps the key as it is and still encrypts at rest. Without
`cryptography`, waiter falls back to plain JSON with a warning, and
`pip install hypernix[security]` enables it.

The letters can be grouped (0.72.6), so the same setup is:

```bash
waiter serv -AEK "T1_..." -I "https://myserver.ts.net:8000"
```

Config is saved to `~/.hypernix/waiter/waiter.config.jsonl` by default;
override with `-F <path>`.

## Subcommands

| Command | Does |
|---|---|
| `waiter serv` | One-shot setup / refresh / sync — see [flags](#serv-flags) |
| `waiter models` | List models visible in the registry |
| `waiter model <id>` | Detail + availability + your usage for one model |
| `waiter route` | Ask the routing engine which model to use — `--plan`, `--model` (manual), `--input-tokens`, `--auto-fallback` |
| `waiter status` | Server status (version, model count, storage backend) |
| `waiter health` | Liveness check |
| `waiter whoami` | Validate the configured key, show scopes |
| `waiter usage` | Usage summary; `--model <id>` for remaining allowance on one model |
| `waiter servers` | List servers, or `--register NAME --address ADDR [--allow-private]` |
| `waiter modules` | List modules, or `--create NAME [--version]` / `--upload ID --file PATH` / `--sync ID --server-id ID` |
| `waiter jobs get\|cancel <job_id>` | Check or cancel an async job |
| `waiter events` | Poll recent events — `--limit`, `--since <event_id>` |
| `waiter billing` | Balance, or `--transactions`, or `--redeem <token>` |
| `waiter config` | Show the locally saved config (key masked) |
| `waiter tui` | Open the full curses dashboard (same as `serv -G`) |
| `waiter cost` | Spend, per-model/server breakdowns, forecasts, and `--estimate-model` |
| `waiter keys` | List keys; `--assign` a plan/account/models; `--import-file` |
| `waiter deploy <id> --to a,b` | Push a module to trusted servers, `--wait` to follow the job |
| `waiter security` | Network policy (`--block`/`--allow`/`--appeal`), forced limits, your own rate-limit budget |
| `waiter audit` | Read the server's audit trail (admin) |
| `waiter doctor` | Check a server's configuration; exits non-zero on production warnings |
| `waiter smoke` | Run smoke tests against a server (`--write` for a self-cleaning write test) |
| `waiter version` | Package, waiter, T1 API and key format versions; `--json` |
| `waiter kits` | Installed kits: `list [--json]`, `remove NAME`, `run NAME COMMAND [ARGS...]` — install with `serv -k` |
| `waiter help <topic>` | Longer help — `connect`, `keys`, `hyperlink`, `find` |

Every subcommand accepts `-I`/`-K`/`-F`/`-P`/`-H` to override the saved
config for just that call, and `--json` for raw JSON instead of a table.

`waiter modules --upload` sends a real multipart/form-data body built by
hand with the standard library (no `requests` dependency) — see
`T1Client.upload_module_local` in `hypernix/waiter/client.py` if you're
curious how; it was tested against a real HTTP server, not mocked.

## `serv` flags

| Flag | Meaning | Status |
|---|---|---|
| `-A` | Automatic configuration: validate + save in one step | ✅ full |
| `-I <addr>` | Server IP / Tailscale IP / public or localhost URL | ✅ full |
| `-K <token>` | T1 token | ✅ full |
| `-E` | Encrypt local config/secrets at rest | ✅ full |
| `-F <path>` | Local config file path | ✅ full |
| `-s` | Save current server/local config to a `.jsonl` file | ✅ full |
| `-L` | Local/Tailscale/localhost-only mode | ✅ chooses the URL scheme; the server-side counterpart is `T1_ALLOW_UNLISTED_CLIENTS=0` plus an allowlisted tailnet — see `examples/t1api/run_tailscale.sh` |
| `-P <port>` | Server port | ✅ full |
| `-H <url>` | Home page URL | ✅ stored |
| `-R` | Quick refresh: re-validate + re-fetch models | ✅ full |
| `-Rf` | Force full refresh | ✅ full — models, servers, modules, events, and the server's config |
| `-y` | Synchronize local config against the server | ✅ full — mirrors the server's `/config` and model count into the local config so `waiter config` reflects the server rather than what was typed weeks ago |
| `-g` | Open an interactive CLI session | ✅ REPL (`models`/`status`/`usage`/`whoami`/`quit`) |
| `-G` | Open the full TUI | ✅ full — see [The TUI](#the-tui) |
| `-B <ip\|cidr>` | Blacklist an address or range (repeatable) | ✅ full — applied to the server (admin key required) *and* saved locally |
| `-W <ip\|cidr>` | Allowlist an address or range (repeatable) | ✅ full — same |
| `-r <subject>=<n>/<window>` | Force a limit on a key/server, e.g. `key:abc123=60/60s` or `server:s1=1000t/1h` | ✅ full — `t` suffix means tokens; only ever tightens |
| `-a <ip\|cidr>` | Appeal: remove an entry from the server's lists | ✅ full |
| `-C <key>=<value>` | Additional configuration settings (repeatable) | ✅ stored locally; `-y` populates it from the server's own settings |
| `--promote-admin` | After validating, request admin promotion for this key | ✅ full — requires the *authenticating* key to already be admin-scoped (`POST /auth/t1/admin/rotate`); not in the original flag list, added as the operational hook for the "-K supports conversion to admin" requirement |
| `-b` | Give bare strings to the options they look like — see [grouping and `-b`](#grouping-letters-and--b) | ✅ full (0.72.6) |
| `-u` | Update hypernix to at least the server's version (`pip install --upgrade`) | ✅ full (0.72.6); `--dry-run` prints the command |
| `-ud` | Update hypernix to exactly the server's version, down as well as up | ✅ full (0.72.6) |
| `-k <path>` | Install a kit: a folder or `.zip` with a `kit.json` — see [Kits](#kits) | ✅ full (0.72.6) |
| `-c` / `--no-conceal` | Turn the server's [conceal mode](T1-API.md#conceal-mode-and-36-hour-retention) on or off for this key: address masked, data kept 36 hours (access level 3+) | ✅ full (0.72.6) |
| `-T` | Open the TUI with the [control pane](#control-mode--t) | ✅ full (0.72.6) — a verified administrator holding a level-9 T2 or v2.1 key |
| `-Y` | Print the server's public card (`GET /server/info`) | ✅ full (0.72.6) |
| `-S` | Check this client, then the server, for security problems | ✅ full (0.72.6) — public endpoints only |
| `-e` / `--unlock` | Lock the config with a password (Rotorvault + scrypt), or remove the lock | ✅ full (0.72.6) — every command then asks, or reads `HNX_WAITER_PASSWORD` |
| `-h` | Help | ✅ (argparse default) |

`-B`/`-W`/`-a`/`-r` write to the server **and** to the local config. That
is deliberate rather than redundant: the local copy is what `waiter
config` shows and what a re-run of `waiter serv -A` re-applies against a
rebuilt server, and it is the only record available when your key is not
admin — in which case the server call is refused and the CLI says so
plainly instead of pretending it worked.

### Grouping letters and `-b`

Single letters can be run together in any order (0.72.6):
`waiter serv -ArEK T2C_… -I 100.64.0.7` is `-A -R -E -K T2C_… -I
100.64.0.7`. A letter that takes a value (`I K F P H B W r a C k`) must
end its group, because its value is the next word, or stand on its own.
waiter refuses `-AKE key` and says how to write it, rather than guessing.

Inside a group, `r` means refresh (`-R`). On its own, `-r` keeps its
old meaning, `-r SUBJECT=LIMIT`, because only a lone `-r` has room for
its value. `Rf` and `ud` inside a group are the full refresh and the
exact update.

Without `-b`, a word with no option in front of it is an error. With
`-b`, waiter works out which option each one belongs to and prints what
it decided:

| Looks like | Goes to |
|---|---|
| `T1_…`, `T2_…`, `T2C_…`, `T2CK_…` and the other key prefixes | `-K` |
| `http(s)://…`, an IP address, a host name | `-I` |
| a whole number from 1 to 65535 | `-P` |
| `key:…=N/Ns` or `N/Ns` | `-r` |
| `KEY=VALUE` | `-C` |
| a `.zip`, or a folder with a `kit.json` | `-k` |
| any other existing file | `-F` |

```bash
waiter serv -Ab T1_abc123 https://myserver.ts.net 8000
```

A CIDR range is refused, because it could be `-B`, `-W` or `-a`, and
blocking an address is not something to guess. Two strings that look
like the same option are refused too, as is a string for an option that
is already given.

## Kits

A kit is a user-made mod: a folder, or a `.zip` of one, with a
`kit.json` at its top.

```json
{
  "name": "fancy-status",
  "version": "1.2.0",
  "kind": "waiter",
  "description": "A status line with colours",
  "commands": {"status": "fancy_status:main"}
}
```

`kind` is `waiter`, `t1api-client`, `hyped-pro` or `other`, and only a
`waiter` kit has `commands` (`module:function`, called with the
arguments). `waiter serv -k PATH` installs one to
`~/.hypernix/waiter/kits/<name>/`, replacing a kit with the same name.
`waiter kits` lists them, `waiter kits remove NAME` removes one, and
`waiter kits run NAME COMMAND [ARGS...]` runs a command.

waiter checks that an archive is well-formed, unpacks to no more than
50 MB, contains no symbolic links and writes nothing outside the kit's
folder. It does nothing else to vet the contents. **A kit is code, and
it runs as you**: installing one is the same trust decision as
`pip install`.

## Interactive session (`-g`)

```
waiter serv -g -F ~/.hypernix/waiter/waiter.config.jsonl
waiter> models
waiter> usage
waiter> whoami
waiter> quit
```

Not the full TUI — a thin REPL over the same one-shot subcommands, useful
when you're going to run several commands against the same server back to
back.

## The TUI

```bash
waiter tui                 # or: waiter serv -G
```

Eight panes, `TAB` between them, everything sourced from the API (nine
in [control mode](#control-mode--t)):

| Pane | Shows |
|---|---|
| **Models** | every registered model, availability, input/output limits, minimum plan — `ENTER` selects one |
| **Quota** | per-model input and output bars, with `EXHAUSTED` when a cap is hit |
| **Usage** | window and all-time totals, spend, account balance, forecast, per-model breakdown |
| **Jobs** | recent jobs with live progress; `c` cancels the selected one |
| **Servers** | registered servers, trust level, status, address |
| **Modules** | modules with status and size; `u` uploads a local file, `d` deploys to a server |
| **Events** | live event tail |
| **Settings** | server version, backend, which protections are on, your key's scopes, and any production warnings the server reports |

The Models pane also renders the **fallback chain** — the cascade the
server actually walked on the last routing decision, with each step marked
`EXHAUSTED`, skipped, or selected. That comes from `POST /models/route`'s
own `considered` list; the TUI never reconstructs a chain from registry
`fallback_model` fields, because the plan's policy decides the order, not
the model entry.

`a` toggles automatic routing. Turning it on immediately asks the server
to route, so the header shows the model that would actually be used rather
than the last one you picked by hand. Selecting an exhausted model tells
you it is exhausted rather than silently substituting another — the server
returns `MODEL_QUOTA_EXHAUSTED` and the TUI reports it.

Refreshes happen on a background thread, so an unreachable server shows
stale data with an error banner rather than freezing the terminal.

`?` lists the keys. `q` quits.

**Windows:** `curses` is not in the standard library there. `waiter tui`
says so and points at the one-shot subcommands, which work everywhere;
`pip install windows-curses` enables the TUI.

### Control mode (`-T`)

```bash
waiter serv -T -K "T2_..."        # a level-9 T2 or v2.1 administrator key
```

`-T` opens the TUI with a **Control** pane first. The pane shows the
server's configuration warnings, the network policy (blocked and allowed
ranges, and whether unlisted clients are let in), the keys, and recent
security events. `b`, `w` and `x` block,
allow or remove an address or range, and `l` toggles unlisted clients.

It opens only after the server confirms the key is an administrator key
of the T2 or v2.1 family at access level 9. A T1 admin key, a level-8
admin key and a level-9 key that is not an admin are all refused. The
pane only decides what the screen offers: the server checks every action
on its own. Without `-T` there is no Control pane and none of its keys
do anything.

## Checking a server

```bash
waiter serv -Y     # the server's public card: name, owner, versions, features
waiter serv -S     # this client, then the server, for security problems
waiter doctor      # configuration: what would block a production start
waiter smoke       # behaviour: auth enforced, registry gated, limits on
waiter smoke --write   # also creates and removes a scratch module
```

`-S` checks this client first: the config file's mode, a key stored as
itself rather than sealed, whether the config is locked, whether
`cryptography` is installed, and the hypernix version. Then it checks the
server: plain HTTP on a public address, production warnings, missing
security headers, whether it answers without a key, and whether this
client is older than the server. It uses only the endpoints any client
can see. It checks your own setup and does not scan the server.

`doctor` reads `GET /status` and prints every production warning the
server reports, exiting non-zero on a production server with warnings — so
it works as a deployment gate. `smoke` treats *expected refusals as
passes*: a non-admin key being refused `/audit` is a pass, and being
served it is a failure.

## Errors

`waiter` surfaces the T1 API's stable error codes directly (e.g.
`MODEL_NOT_SUPPORTED`, `MODEL_QUOTA_EXHAUSTED`, `AUTH_INVALID_KEY`,
`IP_BLOCKED`, `RATE_LIMITED`) rather than a generic "request failed" — see
[T1-API.md#endpoint-reference](T1-API.md#endpoint-reference).

Underneath, the SDK maps those codes to an exception hierarchy, so code
built on `hypernix.t1sdk` can catch `T1QuotaError` or `T1AuthError`
instead of matching strings. `waiter`'s own `T1ClientError` is an alias
for the SDK's base `T1Error`, so a single `except` still catches
everything.
