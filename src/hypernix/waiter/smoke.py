"""waiter.smoke — the CLI smoke-testing tool (spec deliverable #11).

``waiter smoke`` answers one question an operator asks constantly and
that no single endpoint answers: *is this deployment actually working?*
Not "is the process up" (that's ``/health``) but "does authentication
work, is the registry populated, do quota and cost report sensibly, are
the protections on".

Design rules, because a smoke tester that damages a live system is worse
than no smoke tester:

* **Read-only by default.** Every check in the default set is a GET or a
  validation call. ``--write`` opts into the small set that creates
  something, and each of those cleans up after itself.
* **A failed check is data, not a crash.** Every check is trapped
  individually and reported with its error; one failure never hides the
  results of the others. The exit code summarizes.
* **Expected refusals count as passes.** A non-admin key getting 403
  from ``/audit`` means authorization is *working*. Checks declare what
  they expect, so "correctly refused" is a pass and "unexpectedly
  allowed" is a failure — which is the more useful direction for a
  security-relevant tool to be sensitive in.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..t1sdk.errors import T1AuthError, T1Error


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False
    duration_ms: float = 0.0

    @property
    def symbol(self) -> str:
        if self.skipped:
            return "–"
        return "✓" if self.passed else "✗"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class SmokeReport:
    base_url: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "checks": [r.to_dict() for r in self.results],
        }


class _Runner:
    def __init__(self, report: SmokeReport, *, fail_fast: bool) -> None:
        self.report = report
        self.fail_fast = fail_fast
        self.aborted = False

    def check(self, name: str, fn: Callable[[], str], *, skip_if: bool = False, skip_reason: str = "") -> bool:
        """Run one check. Returns whether it passed.

        A check reports success by returning a detail string and failure
        by raising — including :class:`AssertionError`, so a check body
        can be a plain ``assert`` about what the server returned.
        """
        if self.aborted:
            return False
        if skip_if:
            self.report.results.append(CheckResult(name, passed=True, skipped=True, detail=skip_reason))
            return True
        start = time.monotonic()
        try:
            detail = fn()
            result = CheckResult(name, passed=True, detail=detail or "")
        except AssertionError as exc:
            result = CheckResult(name, passed=False, detail=str(exc) or "assertion failed")
        except T1Error as exc:
            result = CheckResult(name, passed=False, detail=f"{exc.code or 'error'}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a smoke tester reports, it doesn't crash
            result = CheckResult(name, passed=False, detail=f"{type(exc).__name__}: {exc}")
        result.duration_ms = (time.monotonic() - start) * 1000
        self.report.results.append(result)
        if not result.passed and self.fail_fast:
            self.aborted = True
        return result.passed


def run_smoke_tests(
    client: Any,
    *,
    base_url: str = "",
    include_write: bool = False,
    fail_fast: bool = False,
    as_json: bool = False,
) -> int:
    """Run the smoke suite. Returns a process exit code (0 = all passed)."""
    report = SmokeReport(base_url=base_url or getattr(client, "base_url", "?"))
    runner = _Runner(report, fail_fast=fail_fast)
    authenticated = bool(getattr(client, "credential", None))

    # --- Reachability -------------------------------------------------
    def check_health() -> str:
        body = client.health()
        if not (body.get("status") == "ok"):
            raise AssertionError(f"/health returned status={body.get('status')!r}")
        if not (body.get("request_id")):
            raise AssertionError("/health response has no request_id")
        return "server is up and returns a request_id"

    runner.check("health", check_health)

    status_holder: dict[str, Any] = {}

    def check_status() -> str:
        raw = client._get("/status")
        status_holder.update(raw)
        if not (raw.get("beta")):
            raise AssertionError("/status did not report a beta level")
        return f"{raw['beta']} · {raw.get('storage_backend')} · {raw.get('model_count')} model(s)"

    runner.check("status", check_status)

    # --- Contract -----------------------------------------------------
    def check_config_has_no_secrets() -> str:
        raw = client._get("/config")
        text = json.dumps(raw)
        for forbidden in ("token_secret", "deploy_secret", "database_url", "password"):
            if not (forbidden not in text):
                raise AssertionError(f"/config leaked a {forbidden} field")
        return f"{len(raw.get('config', {}))} public setting(s), no secret fields"

    runner.check("config exposes no secrets", check_config_has_no_secrets)

    def check_models() -> str:
        payload = client.list_models_raw() if hasattr(client, "list_models_raw") else client.list_models()
        models = payload.get("models", [])
        for model in models:
            if not (model.get("model_id")):
                raise AssertionError("a model entry has no model_id")
            # The spec is explicit that a model_id must be a slug, not a
            # parameter count ("nanonix-mini-lite", not "85b-25.25b").
            if not (not model["model_id"][0].isdigit()):
                raise AssertionError(
                    f"model_id {model['model_id']!r} starts with a digit — parameter counts "
                    "must not be used as identifiers"
                )
        return f"{len(models)} registered model(s), all with slug identifiers"

    runner.check("model registry", check_models)

    def check_unregistered_model_rejected() -> str:
        sentinel = f"definitely-not-a-model-{uuid.uuid4().hex[:8]}"
        try:
            client._get(f"/models/{sentinel}")
        except T1Error as exc:
            if exc.code != "MODEL_NOT_SUPPORTED":
                raise AssertionError(
                    f"expected MODEL_NOT_SUPPORTED for an unregistered model, got {exc.code}"
                ) from exc
            return "unregistered model_id correctly returns MODEL_NOT_SUPPORTED"
        raise AssertionError("an unregistered model_id was accepted — the registry gate is not working")

    runner.check("unregistered model rejected", check_unregistered_model_rejected)

    def check_auth_required() -> str:
        try:
            client.transport.request("GET", "/usage/current")
        except T1AuthError as exc:
            if exc.status not in (401, 403):
                raise AssertionError(f"expected 401/403 without credentials, got {exc.status}") from exc
            return "authenticated endpoints reject an unauthenticated request"
        raise AssertionError("/usage/current served an unauthenticated request")

    runner.check("auth is enforced", check_auth_required)

    # --- Authenticated ------------------------------------------------
    def check_identity() -> str:
        identity = client.validate()
        if not (identity.get("key_id")):
            raise AssertionError("validate() returned no key_id")
        if not ("key" not in identity):
            raise AssertionError("validate() echoed the raw key back")
        return f"key {identity['key_id'][:12]}… ({identity.get('key_type')}), scopes: {', '.join(identity.get('scopes', []))}"

    runner.check(
        "credential validates",
        check_identity,
        skip_if=not authenticated,
        skip_reason="no credential configured (pass -K)",
    )

    def check_usage() -> str:
        usage = client._get("/usage/current", auth=True)
        if "current_window" not in usage:
            raise AssertionError("/usage/current has no current_window")
        return f"{usage['current_window']['requests']} request(s) in the current window"

    runner.check("usage reporting", check_usage, skip_if=not authenticated, skip_reason="no credential")

    def check_cost() -> str:
        cost = client._get("/usage/cost", query={"range": "24h"}, auth=True)
        if "total_cost" not in cost:
            raise AssertionError("/usage/cost has no total_cost")
        return f"{cost['total_cost']} {cost.get('currency')} over 24h"

    runner.check("cost reporting", check_cost, skip_if=not authenticated, skip_reason="no credential")

    def check_rate_limit_headers() -> str:
        payload = client._get("/security/rate-limits", auth=True)
        if "rules" not in payload:
            raise AssertionError("/security/rate-limits has no rules")
        if not payload.get("enabled", True):
            raise AssertionError("rate limiting is DISABLED on this server")
        return f"{len(payload['rules'])} rule(s) active"

    runner.check("rate limiting active", check_rate_limit_headers,
                 skip_if=not authenticated, skip_reason="no credential")

    def check_admin_surface() -> str:
        """Reaching /audit should either work (admin) or be refused
        (non-admin). Both are correct; being *served* to a non-admin is
        not."""
        identity = client.validate()
        is_admin = bool(identity.get("is_admin"))
        try:
            client._get("/audit", query={"limit": 1}, auth=True)
        except T1AuthError:
            if is_admin:
                raise AssertionError("an admin key was refused access to /audit") from None
            return "non-admin key correctly refused by /audit"
        if not (is_admin):
            raise AssertionError("a NON-ADMIN key was served the audit log — authorization is broken")
        return "admin key can read the audit log"

    runner.check("admin authorization", check_admin_surface,
                 skip_if=not authenticated, skip_reason="no credential")

    # --- Write checks (opt-in) ----------------------------------------
    def check_module_lifecycle() -> str:
        name = f"smoke-{uuid.uuid4().hex[:8]}"
        created = client._post(
            "/modules/create", body={"name": name, "version": "0.0.1"}, auth=True
        )["module"]
        module_id = created["module_id"]
        try:
            fetched = client._get(f"/modules/{module_id}")["module"]
            if not (fetched["name"] == name):
                raise AssertionError("created module did not read back")
            # Confirmation gate must actually bite.
            try:
                client.transport.request("DELETE", f"/modules/{module_id}", auth=True)
            except T1Error as exc:
                if exc.code != "CONFIRMATION_REQUIRED":
                    raise AssertionError(
                        f"unconfirmed delete returned {exc.code}, expected CONFIRMATION_REQUIRED"
                    ) from exc
            else:
                raise AssertionError("an unconfirmed destructive delete succeeded")
        finally:
            # Always clean up, including after an assertion failure — a
            # smoke run must not leave scratch objects behind.
            try:
                client.transport.request(
                    "DELETE", f"/modules/{module_id}", query={"confirm": "true"}, auth=True
                )
            except T1Error:
                pass
        return f"created, read back, confirmation-gated, and removed {name}"

    runner.check(
        "module lifecycle (write)",
        check_module_lifecycle,
        skip_if=not (include_write and authenticated),
        skip_reason="--write not given" if authenticated else "no credential",
    )

    # --- Output --------------------------------------------------------
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1

    _render(report, status_holder)
    return 0 if report.ok else 1


def _render(report: SmokeReport, status: dict[str, Any]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"waiter smoke — {report.base_url}", header_style="bold cyan", border_style="dim")
        table.add_column("")
        table.add_column("check")
        table.add_column("detail", overflow="fold")
        table.add_column("ms", justify="right")
        for result in report.results:
            style = "dim" if result.skipped else ("green" if result.passed else "red")
            table.add_row(
                result.symbol, result.name, result.detail, f"{result.duration_ms:.0f}", style=style
            )
        console.print(table)
        summary = f"{report.passed} passed, {report.failed} failed, {report.skipped} skipped"
        console.print(f"[{'green' if report.ok else 'red'}]{summary}[/]")
    except ImportError:
        print(f"waiter smoke — {report.base_url}")
        for result in report.results:
            print(f"  {result.symbol} {result.name:<32}{result.detail}")
        print(f"{report.passed} passed, {report.failed} failed, {report.skipped} skipped")

    warnings = status.get("production_warnings") or []
    if warnings and status.get("environment") == "production":
        print(f"\n{len(warnings)} production configuration warning(s) — run `waiter doctor` for detail.")


__all__ = ["CheckResult", "SmokeReport", "run_smoke_tests"]
