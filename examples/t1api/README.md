# HyperNix T1 API — deployment examples

Runnable deployments, smallest first. Full contract in
[`wiki/T1-API.md`](../../wiki/T1-API.md); check
[`wiki/T1-API-Security-Checklist.md`](../../wiki/T1-API-Security-Checklist.md)
before exposing any of them to something you don't control.

| File | Deployment |
|---|---|
| [`run_local.sh`](run_local.sh) | Development: SQLite, loopback, docs on, no TLS |
| [`run_tailscale.sh`](run_tailscale.sh) | Tailnet: binds the 100.x address, allowlist-only, private deploy targets enabled |
| [`Dockerfile`](Dockerfile) | Production image: two-stage, non-root, CPU torch, healthcheck |
| [`docker-compose.yml`](docker-compose.yml) | API + PostgreSQL + nginx; the API is never published to the host |
| [`nginx.conf`](nginx.conf) | TLS termination and optional mTLS header forwarding |
| [`hypernix-t1api.service`](hypernix-t1api.service) | systemd unit with a real sandbox |
| [`.env.example`](.env.example) | Every configuration variable, with why it matters |
| [`claude_code.sh`](claude_code.sh) | Claude Code on a model this server runs (`hnx claude`) |
| [`mcp-servers.example.json`](mcp-servers.example.json) | Remote MCP servers T1 connects to (Hugging Face's, and any other) |
| [`hyperlink_datasets_and_code.sh`](hyperlink_datasets_and_code.sh) | A dataset excerpt and a code sandbox in one HyperLink message, with curl |
| [`../../bin/hypernix-t1`](../../bin/hypernix-t1) | Not an example: the service manager most single machines want instead of any of the above |
| [`API-EXAMPLES.md`](API-EXAMPLES.md) | Request/response for every endpoint — generated, not written |
| [`openapi.json`](openapi.json) | Exported OpenAPI schema |

## Fastest path

On one machine you own, `hypernix-t1` does all of this for you:

```bash
pip install 'hypernix[t1api]'
hnx-t1 setup              # the installer (b1): asks, writes the config, starts the server
hnx-t1 status             # address, version, HyperLink on the web, models folder
hnx-t1 upgrade -S all     # later: HyperNix in the server's Python, then its .env
hnx list-all              # every command there is
```

The scripts here are for when you want to see or change each piece:

```bash
pip install 'hypernix[t1api]'
./run_local.sh

# in another shell
gkey create --type admin --scopes admin,read,write
waiter serv -A -I http://127.0.0.1:8000 -K "<the key>" -E
waiter smoke
waiter tui
```

## Claude Code (0.72.8a3)

T1 answers the Anthropic Messages API (`/v1/messages`, streaming, tools
and images included), so Claude Code can work with a model HyperNix
runs:

```bash
hnx claude                                   # this machine's T1, the model it has loaded
hnx claude --server http://100.x.y.z:8000 --key "$T1_API_KEY" --model qwen3.5-9b
hnx claude --print-env                       # just the ANTHROPIC_* variables
./claude_code.sh -- -p "explain this repo"   # the same, as a script
```

The model names Claude Code asks for by itself (`claude-*`) are
answered by `T1_CLAUDE_CODE_MODEL`, or by the loaded model.
`T1_ANTHROPIC_ENABLED=0` turns the API off. See
[The Anthropic Messages API](../../wiki/T1-API.md#the-anthropic-messages-api).

## MCP, both ways

T1 is an MCP server at `/mcp` (Streamable HTTP, protocol 2025-11-25 back
to 2024-11-05), and a client of other MCP servers, whose tools it then
offers on `/mcp` and to HyperLink's model:

```bash
cp mcp-servers.example.json ~/.hypernix/t1api/mcp-servers.json   # or T1_MCP_SERVERS_FILE
export HF_TOKEN=hf_...          # read by "token_env"; T1_MCP_HF=1 adds Hugging Face's alone
curl -s "$T1/mcp/remote" -H "Authorization: Bearer $KEY"           # what is connected
claude mcp add --transport http hypernix "$T1/mcp" --header "Authorization: Bearer $KEY"
```

Adding a server over the API (`POST /mcp/remote`) needs an admin key.
See [Remote MCP servers](../../wiki/T1-API.md#remote-mcp-servers-hugging-faces-and-others).

## Net

`hnx net` is deprecated: its status, peers, expose, setup, sync and
remote-run features are the admin-only `/net` routes now, and the
command prints which route replaced each subcommand. Remote run stays
off unless `T1_NET_REMOTE_RUN=1`. See [Net](../../wiki/T1-API.md#net).

## The public-server counter

A server that says it is public (`T1_PUBLIC=1`, or a public
`T1_SERVER_URL` or `T1_HYPERLINK_PUBLIC_URL`) and runs T1
v1.1.26.10.1.0 or newer sends one heartbeat an hour: a hashed random id,
the T1 version and the package version, never an address. The badge in the repository's README counts them.
`T1_PUBLIC_COUNTER=0` turns it off; `T1_COUNTER_HOST=1` makes a server
the counter others report to. See
[The public-server counter](../../wiki/T1-API.md#the-public-server-counter).

## HyperLink: datasets and code

A HyperLink message can carry parts of dataset files (JSONL, Parquet,
CSV, text, Markdown, ...) and a `/code` sandbox the model works in, with
git. The web page does it with the 📎 button and the Code tab;
[`hyperlink_datasets_and_code.sh`](hyperlink_datasets_and_code.sh) does it
with curl. Running code needs `T1_CODE_SANDBOX=1`. See
[Datasets, code sessions and Markdown](../../wiki/T1-API.md#datasets-code-sessions-and-markdown-0728a3).

## Production

```bash
cp .env.example ../../.env          # then fill in the secrets
docker compose up -d
waiter doctor -I https://your-host -K "$T1_ADMIN_KEY"
```

`waiter doctor` exits non-zero on a production server with configuration
warnings, so it works as a deployment gate. The server itself refuses to
start on an unsafe production configuration and lists every problem at
once — see
[Production deployment](../../wiki/T1-API.md#production-deployment).

## Regenerating the API examples

`API-EXAMPLES.md` and `openapi.json` are generated by driving a real
server, so they record the API's behaviour rather than asserting it. After
changing a response shape:

```bash
pip install 'hypernix[t1api-test]'
python ../../scripts/t1api_examples.py
```

Run it with a throwaway `HOME` and `T1_CONFIG_DIR` if you have a real
server on the machine: the generator starts its own, and a fresh one with
an empty key store prints a bootstrap key. The machine's name, like ids
and timestamps, is replaced with a placeholder.

The diff is then the behaviour change, which is the point. Volatile values
(ids, timestamps) and anything credential-shaped are replaced with
placeholders before the file is written.

## Models

Put GGUFs, or HyperNix checkpoint folders, in `~/.hypernix/models`;
HyperLink's downloads land there too. With `T1_MODEL_SYNC=1` (which
`hypernix-t1 create` and `install-t1.sh` write) the server mirrors that
folder into its own, `~/.hypernix/t1api/models`, one symlink per file,
and serves from there: at startup, after each download and before each
model list. `hypernix-sync` (also `t1-sync`, `hypernix-t1 sync`) does it
by hand; `--dry-run` shows what it would change.

```bash
hypernix-sync --dry-run
hypernix-sync --index     # and write the model registry from the result
```

Under the systemd unit, whose sandbox can only write `/var/lib/hypernix`,
put both folders there:

```
T1_MODELS_SOURCE=/var/lib/hypernix/models
T1_MODELS_DIR=/var/lib/hypernix/t1api/models
```

In a container, mount the models folder and point `T1_HF_DOWNLOAD_DIR`
at it rather than syncing: a symlink made on the host points at a host
path, which does not exist inside the container.

## The `config/` directory

`docker-compose.yml` mounts `./config` at `/etc/hypernix` read-only. Put
your real model registry and routing policy there:

```
config/
  models.json     # T1_MODEL_REGISTRY_PATH
  routing.json    # T1_ROUTING_POLICY_PATH
```

Shapes to copy: `src/hypernix/t1api/data/model_registry.example.json` and
`routing_policies.example.json`. Those shipped files are the spec's
placeholder models — deliberately invisible and unroutable unless
`T1_ENABLE_EXAMPLE_MODELS=1`, which production validation flags.
