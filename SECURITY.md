# Security policy

## Supported versions

Security fixes go into the newest release line only. Older releases are
not patched; upgrade to get a fix.

| Version | Supported |
| --- | --- |
| 0.72.8 (alphas, and the release when it ships) | ✅ |
| 0.72.7 | ✅ until 0.72.8 is released |
| 0.72.6 and older | ❌ |

The T1 API is versioned separately (`GET /version`); a fix to it ships in
the next package release and raises the T1 version.

## Reporting a vulnerability

**Please do not open a public issue, discussion or pull request for a
vulnerability.**

Report it privately through GitHub's private vulnerability reporting:
[open a security advisory](https://github.com/trail-b1az3r/hypernix-pip/security/advisories/new).

Include what you can of:

- the component (T1 API, HyperLink app or web, `waiter`, a CLI command,
  hyprslug, a runner, the Firefox extension…) and the version;
- what an attacker can do, and from where (the internet, the LAN or
  Tailscale, a paired device, a T1 key with which scopes, a local user);
- steps or a proof of concept, and any configuration it needs
  (`T1_*` settings, trusted network on or off, accounts on or off);
- whether you have told anyone else.

What to expect:

- an acknowledgement within **7 days**;
- an assessment, and a fix or a plan, within **30 days** for anything
  rated high or critical;
- a published advisory once a fixed release is out, crediting you unless
  you ask not to be named.

This is a volunteer project; those are targets, not guarantees.

## Scope

In scope: code in this repository — the `hypernix` package, `waiter`,
`hypernix-t1` / `hnx-t1`, the T1 API and HyperLink server, the HyperLink
iOS and web clients, the native `ggml-hnx` patch, the Firefox extension
and the install scripts.

Out of scope:

- vulnerabilities in third-party projects this one uses (llama.cpp and its
  forks, LM Studio, PyTorch, FastAPI…) — report those upstream; tell us too
  if HyperNix's use of them makes things worse;
- a model's output (what it says or writes) unless HyperNix acts on it in
  a way it should not — for example a tool call that escapes its workspace;
- attacks that need an admin T1 key or root on the server: those already
  control the server;
- denial of service by sending a server more work than its hardware can do.

## Running it safely

- Keep the T1 API off the open internet unless it is behind TLS and every
  client uses a key. The trusted-network settings (`T1_TRUSTED_NETWORK`)
  are for a LAN or Tailscale you control.
- Keyless trusted-network access does not reach chats or memories unless
  `T1_HYPERLINK_SHARED_KEYLESS_CHATS=1` is set; leave it off on a shared
  network.
- Set `T1_HYPERLINK_REQUIRE_SIGNED=1` once every client is 0.72.8 or later,
  so a prompt without a valid signature is refused.
- Give model tool access (`allow_execute`, mutating T1 tools) only to keys
  that need it.
- Pin what is fetched from the Hugging Face Hub: put `"owner/repo":
  "<commit>"` entries in `~/.hypernix/hf-pins.json` (or the file
  `HYPERNIX_HF_PINS` names), or set `HYPERNIX_HF_REVISION_<REPO>`. Every
  model, tokenizer and dataset download honours them; unpinned ones fetch
  the repo's default branch.

## Scanning

Pushes to main, pull requests and a weekly schedule run Bandit, Semgrep and the HyperNix T1
audit (`.github/workflows/hypernix-security-scan.yml`), and GitHub code
scanning reports to the repository's Security tab.
