# HyperNix T1 API — security audit checklist

Work through this before exposing a T1 API deployment to anything you
don't control. It is ordered by blast radius, not by effort.

Two things make it shorter than it looks:

- **`waiter doctor` checks the configuration half automatically.** It
  reads `GET /status`, prints every production warning the server
  reports, and exits non-zero on a production server with warnings — so
  it works as a deployment gate.
- **`waiter smoke` checks the behaviour half.** It verifies that
  authentication is enforced, that unregistered models are refused, that
  rate limiting is on, and that a non-admin key is correctly refused the
  admin surface. Expected refusals count as passes, so "non-admin served
  the audit log" is a *failure* — which is the direction a
  security-relevant tool should be sensitive in.

```bash
waiter doctor -I https://t1.example.com -K "$T1_ADMIN_KEY"
waiter smoke  -I https://t1.example.com -K "$T1_ADMIN_KEY"
```

Everything below that those two cover is marked ✅ **automated**.

---

## 1. Secrets

- [ ] ✅ **`T1_TOKEN_SECRET` is set and ≥32 characters.** Unset means
      scoped tokens are signed with an ephemeral per-process secret: they
      break on restart and differ between workers.
- [ ] **It is not in the unit file, the compose file, or the image.**
      Unit files are world-readable and image layers are permanent. Use
      `EnvironmentFile=`, a secrets mount, or your platform's secret
      store.
- [ ] **Rotating it is a known procedure.** Changing the value
      invalidates every outstanding scoped token immediately — that *is*
      the revocation mechanism for a stateless token, so know that it
      logs everyone out and that this is intended.
- [ ] ✅ **`T1_DEPLOY_SECRET` is set only if you actually do
      server-to-server deployment**, is ≥32 characters, and is identical
      on both ends. Unset is correct and safe for a single server.
- [ ] **The database password is not in `T1_DATABASE_URL` on a command
      line.** It appears in `ps` output. Use an environment file.
- [ ] **`cryptography` is installed** (`pip install 'hypernix[security]'`).
      Without it Keymaster stores keys as plain JSON and warns — a warning
      is easy to miss in a container log.

## 2. Transport

- [ ] ✅ **TLS is terminated somewhere.** Either the API holds the
      certificates (`T1_TLS_CERTFILE`/`T1_TLS_KEYFILE`) or a proxy does
      and `T1_MTLS_BEHIND_PROXY=1` says so. Production validation refuses
      a deployment that is neither.
- [ ] **If a proxy terminates TLS, `T1_TRUSTED_PROXIES` names it.**
      Without it, `X-Forwarded-For` is ignored (so IP blocklists and
      per-IP rate limits see the proxy, not the client) and client-cert
      headers are refused outright. Both fail closed, but both fail.
- [ ] **`T1_TRUSTED_PROXIES` is as narrow as your topology allows.** It
      is the list of addresses permitted to assert who the client is.
- [ ] **If mTLS is on, verify a request without a client certificate is
      actually refused.** A proxy that forwards no `X-Client-*` headers
      at all produces the same "no certificate" state as a proxy that is
      not configured for mTLS — test it, don't assume it.
- [ ] **The API is not separately reachable around the proxy.** In the
      compose example the API has no `ports:` at all, only `expose:`.

## 3. Access

- [ ] **Decide the unlisted-client posture deliberately.**
      `T1_ALLOW_UNLISTED_CLIENTS=1` is "open unless blocked" (a public
      API); `0` is "allowlist only" (a private one). There is no
      third option and no default that is right for both.
- [ ] **If allowlist-only, the allowlist is populated before you switch
      it on.** Otherwise the first thing it blocks is you.
- [ ] ✅ **`T1_NETWORK_POLICY_ENABLED=1`.** Disabling it turns off
      allow/blocklist enforcement entirely.
- [ ] **Blocklist entries that were meant to be temporary have a TTL.**
      `ttl_seconds` on the blacklist call; otherwise they outlive the
      incident and nobody remembers why.
- [ ] **Understand that the blocklist wins over the allowlist.** Adding
      an address to the allowlist does not un-block it. Appeal
      (`DELETE /security/network/{cidr}`) is the un-block.

## 4. Authentication and authorization

- [ ] ✅ **Authenticated endpoints reject an unauthenticated request.**
- [ ] ✅ **A non-admin key is refused the admin surface** — `/audit`,
      `/keys/import`, `/keys/assign`, `/security/*`, billing mint and
      add-balance, server trust promotion.
- [ ] **Admin keys are few, and you know who holds each one.**
      `waiter keys` lists them with their scopes; `GET /audit
      ?category=admin` shows what they have done.
- [ ] **Scoped tokens are used where the client shouldn't hold the raw
      key.** A token can only narrow scopes, never widen them, and it
      expires.
- [ ] **Key rotation is exercised at least once before you need it.**
      `POST /auth/t1/rotate` invalidates the old key immediately; find
      out what that breaks in a drill rather than an incident.
- [ ] **Every key has an assignment.** Without one it falls back to
      `T1_DEFAULT_PLAN`, which is a decision worth making explicitly
      rather than inheriting.

## 5. The model registry

- [ ] ✅ **`T1_ENABLE_EXAMPLE_MODELS=0` in production.** The shipped
      entries are placeholders the spec describes as not real.
- [ ] **`T1_MODEL_REGISTRY_PATH` points at your data**, and that file is
      read-only to the service user.
- [ ] ✅ **Model ids are slugs, not parameter counts** —
      `nanonix-mini-lite`, never `85b-25.25b`. `waiter smoke` checks this.
- [ ] **Pricing is correct in the registry**, because it is the only
      place cost comes from. A zero price means free, everywhere,
      silently.
- [ ] **Routing policies reference models that exist.** A cascade step
      naming an unregistered model is skipped, so a typo degrades
      routing quietly instead of failing loudly.

## 6. Limits

- [ ] ✅ **`T1_RATE_LIMIT_ENABLED=1`.**
- [ ] **The configured limits account for your worker count.** Limits are
      per-process: four workers means a 120/min rule allows 480/min in
      aggregate. Divide, or run one worker.
- [ ] **Per-model token caps in the registry are the numbers you meant.**
      Hitting *either* the input or the output cap exhausts a model for
      the whole reset period — that is the spec's rule, and it surprises
      people who expect a combined budget.
- [ ] **`T1_USAGE_RESET_PERIOD_SECONDS` matches your billing period.**

## 7. Module handling

- [ ] **The module store is on its own volume, owned by the service
      user.** Blobs are checksummed but *not encrypted at rest* — see the
      known limitation in [T1-API.md](T1-API.md#known-limitation).
- [ ] **`T1_MAX_TRANSFER_BYTES` is set to something your disk can
      survive** being filled with.
- [ ] **Nothing downstream executes what the API stores.** The API never
      imports, executes, or interprets a module, and any consumer that
      does is where the risk moved to — not where it was removed.
- [ ] **Servers are promoted to `trusted` deliberately, one at a time.**
      Trust is what gates module transfer. A fresh registration is always
      `untrusted`.
- [ ] **Remote source URLs are reviewed before registration.** The SSRF
      guard blocks non-http(s) schemes, the cloud-metadata address, and
      private ranges without an explicit opt-in — but a *public* URL you
      did not intend is still fetched.
- [ ] **`T1_ALLOW_PRIVATE_DEPLOY_TARGETS=1` only on a tailnet/LAN
      deployment**, where private targets are the point.

## 8. Auditing and observability

- [ ] ✅ **`T1_AUDIT_ENABLED=1`.**
- [ ] **The audit table is backed up with the rest of the database**, and
      retention is a decision you have made (`AuditLog.purge_before`).
- [ ] **Somebody reads it.** Start with
      `waiter audit --outcome denied` — refusals are the interesting
      records, and a spike in them is the earliest signal you get.
- [ ] **Logs are checked for anything credential-shaped.** Nothing in the
      API logs a key, token, payment token, or DSN password; a `grep` for
      `T1_` in your log store is a cheap way to confirm your *own* code
      does not either.

## 9. Data

- [ ] ✅ **`T1_DATABASE_URL` is set** (PostgreSQL) rather than running
      production on SQLite.
- [ ] **The database is backed up, and a restore has been tested.**
- [ ] **The database user has only the privileges it needs** — the API
      creates its own tables at startup, so it needs DDL on its own
      schema and nothing beyond it.
- [ ] **Payment tokens are understood to be single-use secrets.** The raw
      value is returned exactly once at mint time and stored only as a
      hash; there is no way to recover one, by design.

## 10. Surface

- [ ] **`T1_EXPOSE_DOCS=0` unless the OpenAPI surface should be public.**
- [ ] ✅ **`T1_CORS_ALLOW_ORIGINS` is empty or a specific list — never
      `*`.** A wildcard origin on an authenticated API lets any site
      drive it with a user's credentials.
- [ ] **`T1_REQUIRE_DESTRUCTIVE_CONFIRMATION=1`** unless a scripted
      environment genuinely needs it off.
- [ ] **`GET /config` and `GET /status` have been eyeballed on the live
      deployment** for anything you did not expect to be public.

## 11. Opt-in features

Each of these is off, or narrow, by default. Every one that is on should
be on because somebody decided it, not because an installer answer was
never revisited. `hypernix-t1 status` and `GET /status` show what is on.

- [ ] **Trusted-network mode (`T1_TRUSTED_NETWORK=1`) only where every
      machine on that network is yours.** Keyless callers from the LAN
      and the tailnet get read access; turn off whichever of
      `T1_TRUSTED_NETWORK_LAN` / `T1_TRUSTED_NETWORK_TAILNET` you do not
      need. Leave `T1_TRUSTED_NETWORK_TAILNET_VERIFY=1`: with it off,
      anything that numbers itself 100.x is treated as a tailnet peer.
- [ ] **`T1_TRUSTED_NETWORK_PARTIAL_ADMIN` stays `0`** unless keyless
      callers should pause and stop training, read the hardware and load
      models. That is a separate decision from letting them read.
- [ ] **`T1_HYPERLINK_SHELL=0`** unless paired phones should run shell
      commands on this machine; if on, `T1_HYPERLINK_SHELL_ROOT` is as
      narrow as it can be.
- [ ] **noodle: `T1_NOODLE_ENABLED` and `T1_NOODLE_ALLOW_EXECUTE` are
      separate switches.** Writing files in a workspace is the first;
      running them is the second, and it is running code on this server.
- [ ] **`T1_RUNNER_SWITCH_PERM=0`** unless non-admins should change the
      model everybody on the server is using.
- [ ] **`T1_MCP_ENABLED=0`** unless an assistant is going to use it. It
      is the same capabilities by another route, and a route nobody uses
      is one nobody watches.
- [ ] **Web accounts: registration is `closed`, `invite` or
      `first-user`, not `open`,** unless strangers should mint keys. In
      `site` or `cloudflare` mode, `T1_ACCOUNTS_PUBLIC_URL` is set and the
      proxy replaces the client's `X-Forwarded-For` rather than
      appending to it.
- [ ] **HyperLink on the web answers only on loopback and the tailnet.**
      From a LAN address, `http://<lan-ip>:37965/` and the API port's `/`
      should both refuse. `T1_WEB_ENABLED=0` if nobody uses it.
- [ ] **The bootstrap admin key is replaced.** A first start with an
      empty key store mints a local-only three-day admin key; mint real
      keys and let it expire, or set `T1_BOOTSTRAP_KEY=0` on a server
      provisioned another way.
- [ ] **`T1_BILLING_KEY_POLICY` is a deliberate choice** on a server that
      takes payment keys.
- [ ] **`HYPERNIX_TRUST_PICKLE` is not set on the server.** Checkpoints
      load with `weights_only=True`; that variable turns full unpickling
      (arbitrary code) back on for every load in the process.

## 12. Staying current

- [ ] **`hypernix-t1 version` shows the release you think it does.** The
      server runs from its own venv, which a `pip install -U` in your
      shell does not touch; `hypernix-t1 upgrade` upgrades the right one
      and restarts. `status` and `start` warn when it has fallen behind.
- [ ] **Security fixes in the changelog have been read against this
      deployment** — the 🔒 and 𖢥 entries under Security in
      [the changelog](Changelog.md).

---

## After an incident

1. `waiter security --block <address>` — takes effect immediately and
   survives restart.
2. `waiter audit --category security` — what was refused, and from where.
3. `waiter audit --category admin` — what succeeded, and by whom.
4. `waiter keys` — scopes and assignments of every key.
5. Rotate `T1_TOKEN_SECRET` to invalidate every outstanding scoped token
   at once; rotate individual keys with `POST /auth/t1/admin/rotate`.
6. `waiter security --appeal <address>` when the block should be lifted.

Every one of those is recorded in the audit log, including the reads.
