"""t1api.routers — one module per resource group, matching the spec's
'PRIMARY ENDPOINT SET'.

Beta 1 wired: health, auth, models, usage, config.
Beta 2 added: servers, modules, jobs, events, billing.
Beta 3 adds: keys (the spec's 28–30, previously unimplemented), audit,
and security (network policy + forced limits).
T1 v1.0.26.8.0.1 adds: bridge (the LM Studio bridge) and hyperlink (the
phone-facing surface — pairing, sessions, files, Hugging Face
resolution). Both are documented in ``wiki/T1-API.md``; neither is in
the original spec's endpoint list because neither existed when it was
written.
T1 v1.0.26.8.1.0 adds: authundo (``/t1/auth/undo`` and ``/t1/auth/redo``,
aliased under ``/auth/t1/``) and backup (``/backup/list``,
``/backup/restore``).
T1 v1.0.26.9.2.1 adds: inference — the governed generation surface.
``/bridge/lmstudio/*`` is a pass-through that never consults the
registry, the cascade or the quota; ``/inference/*`` is the same
capability with every one of those applied.
0.72.4 adds: training — progress, logs, checkpoints, machine
resources and the stop/pause/resume controls, admin-gated unless the
server is explicitly in trusted-network mode.

Three routers expose endpoints beyond the spec's literal list, each for
a stated reason rather than by accident:

* ``POST /models/route`` — the spec describes the routing engine in
  detail but names no endpoint for it (routers/models.py).
* ``GET /events/stream`` — "event streaming" reads as a live tail, and
  ``GET /events`` alone can only poll (routers/events.py).
* ``GET /audit`` and ``/security/*`` — "complete audit logging" and the
  IP allowlist/blocklist requirements need a read and a write surface
  respectively (routers/audit.py, routers/security.py).
"""
from __future__ import annotations

from . import (
    audit,
    auth,
    authundo,
    backup,
    billing,
    bridge,
    config,
    events,
    health,
    hyperlink,
    inference,
    jobs,
    keys,
    models,
    modules,
    security,
    servers,
    training,
    usage,
)

ALL_ROUTERS = (
    health.router,
    auth.router,
    models.router,
    usage.router,
    config.router,
    servers.router,
    modules.router,
    jobs.router,
    events.router,
    billing.router,
    keys.router,
    audit.router,
    security.router,
    # T1 v1.0.26.8.0.1
    bridge.router,
    hyperlink.router,
    # T1 v1.0.26.8.1.0
    authundo.router,
    authundo.alias_router,
    backup.router,
    # T1 v1.0.26.9.2.1
    inference.router,
    # 0.72.4
    training.router,
)

__all__ = [
    "ALL_ROUTERS",
    "audit",
    "auth",
    "authundo",
    "backup",
    "billing",
    "bridge",
    "config",
    "events",
    "health",
    "hyperlink",
    "inference",
    "jobs",
    "keys",
    "models",
    "modules",
    "security",
    "servers",
    "training",
    "usage",
]
