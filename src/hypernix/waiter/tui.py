"""waiter.tui — the full curses TUI behind ``waiter serv -G``.

The spec's Beta 3 line is "full waiter TUI", and its WAITER TUI section
lists what it has to show. Each pane below maps to those bullets:

===================================  ==========================================
Spec requirement                     Where it lives
===================================  ==========================================
list supported models                Models pane
display model availability           Models pane, ``avail`` column
display model limits                 Models pane, ``in``/``out`` columns
display model quota                  Quota pane, per-model bars
display current model                Header, ``model:`` field
manually select a model              Models pane, ``Enter``
enable/disable automatic routing     Header, ``a`` toggles
show fallback chain                  Routing pane
show quota exhaustion                Quota pane, EXHAUSTED marker
show remaining usage                 Usage pane
show spent cost                      Usage pane
show account balance                 Usage pane
upload a local module                Modules pane, ``u``
push modules to remote servers       Modules pane, ``d``
monitor jobs / watch live progress   Jobs pane, auto-refreshing
watch server status                  Servers pane
watch events                         Events pane, live tail
manage permitted settings            Settings pane
===================================  ==========================================

**Everything shown comes from the API.** There is no hard-coded model
list, no client-side quota arithmetic, and no local guess at what the
caller may do — the spec is explicit that "the TUI MUST obtain model
information from the API's model registry instead of maintaining a
separate hard-coded model list", and the design principle extends that
to every other decision. When the TUI greys out a model, it is because
``GET /models/{id}/availability`` said so; when it shows a fallback
chain, that chain came from ``POST /models/route``'s ``considered``
list.

Implementation notes
--------------------
* ``curses`` is stdlib on POSIX and absent on Windows. :func:`run` checks
  and returns a clear message rather than throwing an ImportError at
  someone who just wanted a dashboard — the one-shot subcommands work
  everywhere.
* Network calls happen on a background refresh thread, never in the draw
  path, so a slow or unreachable server makes the TUI show stale data and
  an error line instead of freezing the terminal.
* The screen redraws from a :class:`DashboardState` snapshot. Fetch and
  render are fully separated, which is what makes the state machine
  testable without a terminal — see ``tests/test_t1api_beta3_tui.py``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .client import T1Client, T1ClientError

REFRESH_INTERVAL = 5.0

PANES = ("models", "quota", "usage", "jobs", "servers", "modules", "events", "settings")

#: The extra pane `waiter serv -T` opens with (0.72.6). Only in control
#: mode, which only a verified level-9 T2 administrator reaches — see
#: waiter.servops.control_allowed. The server checks every action anyway.
CONTROL_PANE = "control"

CONTROL_HELP = [
    "b                 block an address or range (Control pane)",
    "w                 allow an address or range (Control pane)",
    "x                 remove an address from both lists (Control pane)",
    "l                 let unlisted addresses in, or not (Control pane)",
]

HELP_LINES = [
    "TAB / shift-TAB   next / previous pane",
    "UP DOWN           move selection",
    "ENTER             select model (Models) / open detail",
    "a                 toggle automatic routing",
    "r                 refresh now",
    "u                 upload a local module (Modules pane)",
    "d                 deploy selected module to a server (Modules pane)",
    "c                 cancel selected job (Jobs pane)",
    "?                 toggle this help",
    "q                 quit",
]


@dataclass
class DashboardState:
    """Everything the TUI draws, refreshed as one snapshot.

    Held as plain data with no curses in sight so the refresh logic can
    be exercised in tests against a fake client. ``error`` is a banner,
    not an exception: a failed refresh must leave the previous snapshot
    on screen rather than blanking it, because stale data with a visible
    warning is more useful than an empty dashboard.
    """

    models: list[dict[str, Any]] = field(default_factory=list)
    availability: dict[str, dict[str, Any]] = field(default_factory=dict)
    quota: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    balance: dict[str, Any] = field(default_factory=dict)
    jobs: list[dict[str, Any]] = field(default_factory=list)
    servers: list[dict[str, Any]] = field(default_factory=list)
    modules: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    # Control mode (waiter serv -T) only.
    network: dict[str, Any] = field(default_factory=dict)
    keys: list[dict[str, Any]] = field(default_factory=list)
    security_events: list[dict[str, Any]] = field(default_factory=list)

    selected_model: str | None = None
    automatic_routing: bool = True
    last_refresh: float = 0.0
    error: str | None = None
    notice: str | None = None

    @property
    def fallback_chain(self) -> list[dict[str, Any]]:
        """The cascade the server walked on the last routing decision.

        Comes straight from ``POST /models/route``'s ``considered``
        list — the TUI never reconstructs a chain from the registry's
        ``fallback_model`` fields, because the plan's policy, not the
        model entry, is what actually decides the order.
        """
        return list(self.routing.get("considered", []))

    def model_row(self, model_id: str) -> dict[str, Any] | None:
        for model in self.models:
            if model.get("model_id") == model_id:
                return model
        return None


class DashboardController:
    """Fetches state and applies user actions. No curses dependency."""

    def __init__(self, client: T1Client, *, state: DashboardState | None = None,
                 control: bool = False) -> None:
        self.client = client
        self.state = state or DashboardState()
        self.control = control
        self._lock = threading.Lock()

    @property
    def panes(self) -> tuple[str, ...]:
        return (CONTROL_PANE, *PANES) if self.control else PANES

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> DashboardState:
        """Pull one full snapshot.

        Each section is fetched independently and a failure in one is
        recorded rather than aborting the rest: an operator whose key
        can't read ``/servers`` should still see their models and quota,
        not an empty screen. Only a total failure surfaces as the banner.
        """
        errors: list[str] = []

        def attempt(label: str, fn, default):
            try:
                return fn()
            except T1ClientError as exc:
                errors.append(f"{label}: {exc}")
                return default
            except Exception as exc:  # noqa: BLE001 - a draw loop must not die on a surprise
                errors.append(f"{label}: {exc}")
                return default

        models_payload = attempt("models", self.client.list_models, {"models": []})
        models = models_payload.get("models", [])

        availability: dict[str, dict[str, Any]] = {}
        quota: dict[str, dict[str, Any]] = {}
        for model in models:
            model_id = model.get("model_id")
            if not model_id:
                continue
            availability[model_id] = attempt(
                f"availability[{model_id}]",
                lambda mid=model_id: self.client.model_availability(mid),
                {},
            )
            if self.client.credential:
                quota[model_id] = attempt(
                    f"quota[{model_id}]",
                    lambda mid=model_id: self.client.model_usage(mid),
                    {},
                )

        authed = bool(self.client.credential)
        usage = attempt("usage", self.client.usage_current, {}) if authed else {}
        cost = (
            attempt("cost", lambda: self.client.usage_cost(forecast=True).raw, {}) if authed else {}
        )
        balance = attempt("balance", self.client.billing_balance, {}) if authed else {}
        servers = attempt("servers", self.client.list_servers, {"servers": []}).get("servers", [])
        modules = attempt("modules", self.client.list_modules, {"modules": []}).get("modules", [])
        events = attempt("events", lambda: self.client.list_events(limit=30), {"events": []}).get(
            "events", []
        )
        status = attempt("status", self.client.status, {})
        identity = attempt("whoami", self.client.validate, {}) if authed else {}
        if self.control:
            network = attempt("network policy", self.client.network_policy, {})
            keys = [
                {"key_id": k.key_id, "key_type": k.key_type, "active": k.active,
                 "scopes": list(k.scopes)}
                for k in attempt("keys", self.client.list_keys, [])
            ]
            security = attempt(
                "security events",
                lambda: self.client.audit_events(category="security", limit=20), {},
            ).get("events", [])

        with self._lock:
            state = self.state
            if models_payload.get("models") is not None:
                state.models = models
                state.availability = availability
                state.quota = quota
            state.usage = usage or state.usage
            state.cost = cost or state.cost
            state.balance = balance or state.balance
            state.servers = servers or state.servers
            state.modules = modules or state.modules
            state.events = events or state.events
            state.status = status or state.status
            state.identity = identity or state.identity
            if self.control:
                state.network = network or state.network
                state.keys = keys or state.keys
                state.security_events = security or state.security_events
            state.last_refresh = time.time()
            # Only report a banner when something actually failed, and cap
            # it — a screen full of per-model errors helps nobody.
            state.error = "; ".join(errors[:3]) + (" …" if len(errors) > 3 else "") if errors else None
            if state.selected_model is None:
                first_available = next(
                    (
                        m["model_id"]
                        for m in models
                        if availability.get(m.get("model_id", ""), {}).get("available")
                    ),
                    None,
                )
                state.selected_model = first_available
        return self.state

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def select_model(self, model_id: str) -> str:
        """Manually select a model, asking the server whether it's usable.

        The TUI does not decide this. It calls ``POST /models/route`` with
        the model named, and reports whatever the server says — including
        ``MODEL_QUOTA_EXHAUSTED``, which is exactly the outcome the spec
        wants surfaced rather than papered over with a silent
        substitution.
        """
        try:
            decision = self.client.route(model_id=model_id, automatic_fallback=False)
        except T1ClientError as exc:
            if exc.code == "MODEL_QUOTA_EXHAUSTED":
                self.state.notice = (
                    f"{model_id} is exhausted for this reset period. "
                    "Enable automatic routing (a) to fall through the cascade."
                )
            else:
                self.state.notice = f"Cannot select {model_id}: {exc}"
            return self.state.notice
        self.state.selected_model = decision["model_id"]
        self.state.routing = decision
        self.state.automatic_routing = False
        self.state.notice = f"Model set to {decision['model_id']}."
        return self.state.notice

    def toggle_automatic_routing(self) -> str:
        """Turn automatic routing on or off.

        Turning it *on* immediately asks the server to route, so the
        header shows the model that would actually be used rather than
        the last one the operator picked by hand.
        """
        self.state.automatic_routing = not self.state.automatic_routing
        if not self.state.automatic_routing:
            self.state.notice = "Automatic routing OFF — the selected model is used as-is."
            return self.state.notice
        try:
            decision = self.client.route()
        except T1ClientError as exc:
            self.state.automatic_routing = False
            self.state.notice = f"Automatic routing unavailable: {exc}"
            return self.state.notice
        self.state.routing = decision
        self.state.selected_model = decision["model_id"]
        self.state.notice = f"Automatic routing ON — server chose {decision['model_id']}."
        return self.state.notice

    # -- control mode (waiter serv -T) ---------------------------------

    def _control_action(self, label: str, fn) -> str:
        if not self.control:
            return self._note("control actions need waiter serv -T")
        try:
            fn()
        except T1ClientError as exc:
            return self._note(f"{label} refused: {exc}")
        self.refresh()
        return self._note(f"{label}: done")

    def block_address(self, cidr: str) -> str:
        return self._control_action(f"block {cidr}",
                                    lambda: self.client.blacklist_ip(cidr, reason="waiter -T"))

    def allow_address(self, cidr: str) -> str:
        return self._control_action(f"allow {cidr}",
                                    lambda: self.client.whitelist_ip(cidr, reason="waiter -T"))

    def appeal_address(self, cidr: str) -> str:
        return self._control_action(f"remove {cidr}", lambda: self.client.appeal_ip(cidr))

    def toggle_allow_unlisted(self) -> str:
        current = bool(self.state.network.get("allow_unlisted_clients", True))
        label = "let unlisted addresses in" if not current else "refuse unlisted addresses"
        return self._control_action(label, lambda: self.client.set_allow_unlisted(not current))

    def _note(self, text: str) -> str:
        with self._lock:
            self.state.notice = text
        return text

    def cancel_job(self, job_id: str) -> str:
        try:
            job = self.client.cancel_job(job_id)["job"]
        except T1ClientError as exc:
            self.state.notice = f"Could not cancel {job_id[:8]}: {exc}"
            return self.state.notice
        self.state.notice = f"Job {job_id[:8]} is now {job['status']}."
        return self.state.notice

    def upload_module(self, module_id: str, file_path: str) -> str:
        try:
            module = self.client.upload_module_local(module_id, file_path)["module"]
        except T1ClientError as exc:
            self.state.notice = f"Upload failed: {exc}"
            return self.state.notice
        self.state.notice = (
            f"Uploaded {module['size_bytes']} bytes to {module['name']}@{module['version']}."
        )
        return self.state.notice

    def deploy_module(self, module_id: str, server_id: str) -> str:
        try:
            queued = self.client.sync_module(module_id, server_id)
        except T1ClientError as exc:
            self.state.notice = f"Deploy failed: {exc}"
            return self.state.notice
        self.state.notice = f"Queued deployment job {queued['job_id'][:8]} — watch the Jobs pane."
        return self.state.notice

    def track_jobs(self) -> list[dict[str, Any]]:
        """Refresh the job list from recent job events.

        There is no ``GET /jobs`` list endpoint (the spec defines
        ``/jobs/{id}`` and ``/jobs/{id}/cancel`` only), so the TUI derives
        the set of interesting jobs from the event stream — every job
        publishes ``job.<status>`` events — and then fetches each one's
        current state. That keeps the dashboard working against the
        documented contract instead of requiring an endpoint the spec
        doesn't have.
        """
        job_ids: list[str] = []
        for event in self.state.events:
            if str(event.get("type", "")).startswith("job."):
                job_id = event.get("data", {}).get("job_id")
                if job_id and job_id not in job_ids:
                    job_ids.append(job_id)
        jobs = []
        for job_id in job_ids[:20]:
            try:
                jobs.append(self.client.get_job(job_id)["job"])
            except T1ClientError:
                continue
        self.state.jobs = jobs
        return jobs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _bar(used: int, limit: int, width: int = 18) -> str:
    """A text quota bar. Full bar = exhausted, and the caller decides how
    to colour it."""
    if limit <= 0:
        return "─" * width
    filled = min(width, int(round(width * min(1.0, used / limit))))
    return "█" * filled + "░" * (width - filled)


def render_pane(state: DashboardState, pane: str, width: int = 100) -> list[str]:
    """Render one pane to plain text lines.

    Split out from the curses drawing so every pane's content can be
    asserted in a test without a terminal, and so a future non-curses
    front end (a plain ``--watch`` mode, say) can reuse it verbatim.
    """
    if pane == CONTROL_PANE:
        return _render_control(state)
    if pane == "models":
        lines = [f"{'model_id':<26}{'status':<11}{'avail':<7}{'in':>9}{'out':>9}  plan"]
        for model in state.models:
            model_id = model.get("model_id", "?")
            available = state.availability.get(model_id, {}).get("available")
            marker = "yes" if available else "no"
            selected = "▶ " if model_id == state.selected_model else "  "
            lines.append(
                f"{selected}{model_id:<24}{model.get('status', '?'):<11}{marker:<7}"
                f"{_fmt_int(model.get('input_token_limit')):>9}"
                f"{_fmt_int(model.get('output_token_limit')):>9}  {model.get('minimum_plan', '')}"
            )
        if not state.models:
            lines.append("  (no models visible — the server's registry is empty for this caller)")
        return lines

    if pane == "quota":
        lines = [f"{'model_id':<26}{'input':<22}{'output':<22}state"]
        for model_id, snap in state.quota.items():
            if not snap:
                continue
            exhausted = "EXHAUSTED" if snap.get("is_exhausted") else "ok"
            lines.append(
                f"  {model_id:<24}"
                f"{_bar(snap.get('input_tokens_used', 0), snap.get('input_token_limit', 0)):<22}"
                f"{_bar(snap.get('output_tokens_used', 0), snap.get('output_token_limit', 0)):<22}"
                f"{exhausted}"
            )
        if not any(state.quota.values()):
            lines.append("  (quota needs an authenticated key — pass -K)")
        return lines

    if pane == "usage":
        window = state.usage.get("current_window", {})
        all_time = state.usage.get("all_time", {})
        lines = [
            f"  window requests : {window.get('requests', 0)}",
            f"  window tokens   : {window.get('total_tokens', 0)}",
            f"  all-time tokens : {all_time.get('total_tokens', 0)}",
            f"  spent           : {state.cost.get('total_cost', 0.0)} {state.cost.get('currency', '')}",
            f"  balance         : {state.balance.get('balance', '-')}",
        ]
        forecast = state.cost.get("forecast")
        if forecast:
            lines.append(
                f"  forecast        : {forecast.get('projected_cost', 0.0)} "
                f"{forecast.get('currency', '')} over {forecast.get('horizon_seconds', 0) / 86400:.0f}d "
                f"({forecast.get('confidence', '?')} confidence)"
            )
        by_model = state.usage.get("by_model", [])
        if by_model:
            lines.append("")
            lines.append(f"  {'model_id':<26}{'requests':>10}{'in':>12}{'out':>12}")
            for row in by_model:
                lines.append(
                    f"  {row['model_id']:<26}{row['requests']:>10}"
                    f"{row['input_tokens']:>12}{row['output_tokens']:>12}"
                )
        return lines

    if pane == "jobs":
        lines = [f"{'job_id':<12}{'kind':<16}{'status':<12}detail"]
        for job in state.jobs:
            detail = job.get("error") or _job_progress(job)
            lines.append(
                f"  {job.get('job_id', '')[:10]:<10}{job.get('kind', ''):<16}"
                f"{job.get('status', ''):<12}{detail}"
            )
        if not state.jobs:
            lines.append("  (no recent jobs — they appear here as job.* events arrive)")
        return lines

    if pane == "servers":
        lines = [f"{'server_id':<14}{'name':<20}{'trust':<12}{'status':<10}address"]
        for server in state.servers:
            lines.append(
                f"  {server.get('server_id', '')[:12]:<12}{server.get('name', ''):<20}"
                f"{server.get('trust_level', ''):<12}{server.get('status', ''):<10}"
                f"{server.get('address', '')}"
            )
        if not state.servers:
            lines.append("  (no servers registered)")
        return lines

    if pane == "modules":
        lines = [f"{'module_id':<14}{'name@version':<28}{'status':<14}{'size':>10}  deployed"]
        for module in state.modules:
            lines.append(
                f"  {module.get('module_id', '')[:12]:<12}"
                f"{module.get('name', '')}@{module.get('version', ''):<18}"
                f"{module.get('status', ''):<14}{_fmt_int(module.get('size_bytes')):>10}  "
                f"{len(module.get('deployed_servers', []))} server(s)"
            )
        if not state.modules:
            lines.append("  (no modules — create one with `waiter modules --create NAME`)")
        return lines

    if pane == "events":
        lines = []
        for event in reversed(state.events[-25:]):
            stamp = time.strftime("%H:%M:%S", time.localtime(event.get("ts", 0)))
            lines.append(f"  {stamp}  {event.get('type', ''):<32}{str(event.get('data', ''))[:60]}")
        if not lines:
            lines.append("  (no events yet)")
        return lines

    if pane == "settings":
        status = state.status
        lines = [
            f"  server            : {status.get('environment', '?')} / {status.get('beta', '?')}",
            f"  t1 api version    : {status.get('t1_api_version', '?')}",
            f"  hypernix version  : {status.get('hypernix_version', '?')}",
            f"  storage backend   : {status.get('storage_backend', '?')}",
            f"  TLS / mTLS        : {status.get('tls_enabled', False)} / {status.get('mtls_mode', '?')}",
            f"  rate limiting     : {status.get('rate_limit_enabled', '?')}",
            f"  audit logging     : {status.get('audit_enabled', '?')}",
            f"  network policy    : {status.get('network_policy_enabled', '?')} "
            f"(unlisted clients: {status.get('allow_unlisted_clients', '?')})",
            f"  remote deployment : {status.get('remote_deployment_enabled', '?')}",
            f"  production ready  : {status.get('production_ready', '?')}",
        ]
        warnings = status.get("production_warnings", [])
        if warnings:
            lines.append("")
            lines.append("  configuration warnings from the server:")
            lines.extend(f"    • {w[:width - 8]}" for w in warnings[:6])
        identity = state.identity
        if identity:
            lines.extend(
                [
                    "",
                    f"  your key          : {identity.get('key_id', '?')[:12]}… "
                    f"({identity.get('key_type', '?')})",
                    f"  your scopes       : {', '.join(identity.get('scopes', []))}",
                ]
            )
        return lines

    if pane == "routing":
        lines = [
            f"  automatic routing : {'ON' if state.automatic_routing else 'OFF'}",
            f"  current model     : {state.selected_model or '(none)'}",
            f"  policy            : {state.routing.get('policy_name', '(not routed yet)')}",
            f"  resolved plan     : {state.routing.get('resolved_plan', '?')}",
            "",
            "  fallback chain (as the server walked it):",
        ]
        chain = state.fallback_chain
        if not chain:
            lines.append("    (press 'a' to ask the server to route and reveal the chain)")
        for index, step in enumerate(chain):
            if step.get("skipped"):
                note = f"skipped — {step['skipped']}"
            elif step.get("exhausted"):
                note = "EXHAUSTED"
            else:
                note = "→ selected"
            lines.append(f"    {index + 1}. {step.get('model_id', '?'):<26}{note}")
        return lines

    return [f"  (unknown pane: {pane})"]


def _render_control(state: DashboardState) -> list[str]:
    status, network = state.status, state.network
    lines = ["SERVER"]
    lines.append(f"  {status.get('server_name', '?')}  hypernix {status.get('hypernix_version', '?')}  "
                 f"environment {status.get('environment', '?')}")
    warnings = status.get("production_warnings") or []
    lines.append(f"  {len(warnings)} configuration warning(s)" if warnings else "  no configuration warnings")
    lines.extend(f"    - {w}" for w in warnings[:5])
    lines.append("")
    lines.append("NETWORK POLICY  (b block · w allow · x remove · l unlisted)")
    lines.append(f"  unlisted addresses: {'allowed' if network.get('allow_unlisted_clients', True) else 'refused'}")
    for entry in (network.get("entries") or [])[:12]:
        lines.append(f"  {entry.get('kind', '?'):<10}{entry.get('cidr', '?'):<22}{entry.get('reason', '')}")
    lines.append("")
    lines.append(f"KEYS  {len(state.keys)}")
    for key in state.keys[:10]:
        flag = "" if key.get("active", True) else "  (inactive)"
        lines.append(f"  {key.get('key_id', '?')[:12]:<14}{key.get('key_type', '?'):<10}"
                     f"{','.join(key.get('scopes') or [])}{flag}")
    lines.append("")
    lines.append("RECENT SECURITY EVENTS")
    if not state.security_events:
        lines.append("  none")
    for event in state.security_events[:10]:
        lines.append(f"  {event.get('action', '?'):<34}{event.get('outcome', '?'):<9}"
                     f"{event.get('client_ip', '')}")
    return lines


def _job_progress(job: dict[str, Any]) -> str:
    result = job.get("result") or {}
    delivered = result.get("delivered")
    if delivered is not None:
        requested = len(result.get("requested_servers", []) or [])
        return f"{len(delivered)}/{requested} server(s) delivered"
    if job.get("started_at") and not job.get("finished_at"):
        return f"running for {time.time() - job['started_at']:.0f}s"
    return ""


def _fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def render_header(state: DashboardState, base_url: str) -> str:
    age = time.time() - state.last_refresh if state.last_refresh else None
    routing = "auto" if state.automatic_routing else "manual"
    return (
        f" waiter │ {base_url} │ model: {state.selected_model or '-'} │ routing: {routing} │ "
        f"refreshed {f'{age:.0f}s ago' if age is not None else 'never'} "
    )


# ---------------------------------------------------------------------------
# curses driver
# ---------------------------------------------------------------------------


def run(client: T1Client, *, refresh_interval: float = REFRESH_INTERVAL, control: bool = False) -> int:
    """Run the TUI. Returns a process exit code.

    Returns 1 with an explanation when curses isn't available (notably on
    Windows) rather than raising — the one-shot ``waiter`` subcommands
    still work there, and that is what the message says.
    """
    try:
        import curses
    except ImportError:
        print(
            "The waiter TUI needs the standard-library `curses` module, which isn't available "
            "on this platform (Windows ships Python without it).\n"
            "Use the one-shot subcommands instead — `waiter models`, `waiter usage`, "
            "`waiter jobs get <id>` — or `waiter serv -g` for an interactive prompt.\n"
            "On Windows, `pip install windows-curses` also enables this TUI."
        )
        return 1

    controller = DashboardController(client, control=control)
    return curses.wrapper(lambda stdscr: _loop(stdscr, controller, client, refresh_interval))


def _loop(stdscr, controller: DashboardController, client: T1Client, refresh_interval: float) -> int:
    import curses

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)  # headers
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # ok
        curses.init_pair(3, curses.COLOR_RED, -1)  # errors / exhausted
        curses.init_pair(4, curses.COLOR_YELLOW, -1)  # notices

    panes = controller.panes
    pane_index = 0
    selection = 0
    show_help = False
    stop = threading.Event()

    def refresher() -> None:
        """Background refresh. Network work never happens in the draw
        path, so an unreachable server degrades to stale data plus an
        error banner rather than a frozen terminal."""
        while not stop.is_set():
            controller.refresh()
            controller.track_jobs()
            stop.wait(refresh_interval)

    thread = threading.Thread(target=refresher, daemon=True, name="waiter-tui-refresh")
    thread.start()

    try:
        while True:
            state = controller.state
            pane = panes[pane_index]
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            stdscr.addnstr(0, 0, render_header(state, client.base_url).ljust(width), width,
                           curses.color_pair(1) | curses.A_REVERSE)

            tabs = "  ".join(
                f"[{name.upper()}]" if i == pane_index else f" {name} " for i, name in enumerate(panes)
            )
            stdscr.addnstr(1, 0, tabs, width, curses.color_pair(1))

            body = render_pane(state, pane, width=width)
            if pane == "models" and state.fallback_chain:
                body = body + [""] + render_pane(state, "routing", width=width)
            elif pane == "models":
                body = body + [""] + render_pane(state, "routing", width=width)

            visible = height - 5
            top = max(0, min(selection - visible + 2, len(body) - visible)) if len(body) > visible else 0
            for row, line in enumerate(body[top : top + visible]):
                absolute = top + row
                attr = curses.A_NORMAL
                if "EXHAUSTED" in line:
                    attr = curses.color_pair(3)
                elif absolute == selection and pane in ("models", "jobs", "modules", "servers"):
                    attr = curses.A_REVERSE
                stdscr.addnstr(3 + row, 0, line, width - 1, attr)

            footer_row = height - 1
            if state.error:
                stdscr.addnstr(footer_row, 0, f" ! {state.error}"[: width - 1], width - 1,
                               curses.color_pair(3))
            elif state.notice:
                stdscr.addnstr(footer_row, 0, f" {state.notice}"[: width - 1], width - 1,
                               curses.color_pair(4))
            else:
                stdscr.addnstr(footer_row, 0, "  TAB panes · ENTER select · a auto-route · r refresh · ? help · q quit"[: width - 1], width - 1)

            if show_help:
                _draw_help(stdscr, curses, height, width,
                           HELP_LINES + CONTROL_HELP if controller.control else HELP_LINES)

            stdscr.refresh()

            key = stdscr.getch()
            if key == -1:
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("?"):
                show_help = not show_help
            elif key == ord("\t"):
                pane_index = (pane_index + 1) % len(panes)
                selection = 0
            elif key == curses.KEY_BTAB:
                pane_index = (pane_index - 1) % len(panes)
                selection = 0
            elif key == curses.KEY_DOWN:
                selection = min(selection + 1, max(0, len(body) - 1))
            elif key == curses.KEY_UP:
                selection = max(0, selection - 1)
            elif key == ord("r"):
                controller.refresh()
                controller.track_jobs()
            elif key == ord("a"):
                controller.toggle_automatic_routing()
            elif key in (curses.KEY_ENTER, 10, 13):
                _activate(controller, pane, selection, state)
            elif pane == CONTROL_PANE and key in (ord("b"), ord("w"), ord("x")):
                label = {ord("b"): "Block", ord("w"): "Allow", ord("x"): "Remove"}[key]
                cidr = _prompt(stdscr, curses, height, width, f"{label} address or range: ")
                if cidr:
                    {ord("b"): controller.block_address, ord("w"): controller.allow_address,
                     ord("x"): controller.appeal_address}[key](cidr)
            elif pane == CONTROL_PANE and key == ord("l"):
                controller.toggle_allow_unlisted()
            elif key == ord("c") and pane == "jobs":
                job = _row_item(state.jobs, selection)
                if job:
                    controller.cancel_job(job["job_id"])
            elif key == ord("u") and pane == "modules":
                module = _row_item(state.modules, selection)
                if module:
                    path = _prompt(stdscr, curses, height, width, "Local file to upload: ")
                    if path:
                        controller.upload_module(module["module_id"], path)
            elif key == ord("d") and pane == "modules":
                module = _row_item(state.modules, selection)
                if module:
                    server_id = _prompt(stdscr, curses, height, width, "Target server_id: ")
                    if server_id:
                        controller.deploy_module(module["module_id"], server_id)
    finally:
        stop.set()
    return 0


def _row_item(rows: list[dict[str, Any]], selection: int) -> dict[str, Any] | None:
    """Map a screen row back to its record.

    Every pane's body starts with one header line, so row N on screen is
    record N-1. Returns None for the header row or an out-of-range
    selection instead of guessing.
    """
    index = selection - 1
    if 0 <= index < len(rows):
        return rows[index]
    return None


def _activate(controller: DashboardController, pane: str, selection: int, state: DashboardState) -> None:
    if pane == "models":
        model = _row_item(state.models, selection)
        if model:
            controller.select_model(model["model_id"])


def _draw_help(stdscr, curses, height: int, width: int, lines: list[str] | None = None) -> None:
    lines = lines or HELP_LINES
    box_height = len(lines) + 4
    box_width = min(width - 4, 74)
    top = max(1, (height - box_height) // 2)
    left = max(0, (width - box_width) // 2)
    for row in range(box_height):
        stdscr.addnstr(top + row, left, " " * box_width, box_width, curses.A_REVERSE)
    stdscr.addnstr(top + 1, left + 2, "waiter TUI — keys", box_width - 4, curses.A_REVERSE | curses.A_BOLD)
    for row, line in enumerate(lines):
        stdscr.addnstr(top + 3 + row, left + 2, line, box_width - 4, curses.A_REVERSE)


def _prompt(stdscr, curses, height: int, width: int, label: str) -> str:
    """Read a line of text from the user at the bottom of the screen."""
    curses.echo()
    curses.curs_set(1)
    stdscr.nodelay(False)
    try:
        stdscr.addnstr(height - 2, 0, label.ljust(width - 1), width - 1, curses.A_REVERSE)
        stdscr.refresh()
        raw = stdscr.getstr(height - 2, len(label), 200)
        return raw.decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 - a cancelled prompt must not kill the TUI
        return ""
    finally:
        curses.noecho()
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(200)


__all__ = [
    "DashboardController",
    "DashboardState",
    "PANES",
    "render_header",
    "render_pane",
    "run",
]
