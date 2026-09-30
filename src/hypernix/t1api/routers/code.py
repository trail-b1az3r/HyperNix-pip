"""``/code`` — run code in sandboxes on this server.

The tree, each level describing the one below it on a GET::

    GET    /code                                   what this server runs, and how to ask
    GET    /code/create                            what can be created
    POST   /code/create                            run a snippet once; the sandbox is removed
    GET    /code/create/sandbox                    your sandboxes
    POST   /code/create/sandbox                    a sandbox that stays
    GET    /code/create/sandbox/perms              what every sandbox may do
    GET    /code/create/sandbox/perms/<directives> read or change it: s2?:=on|s3?:=120
    GET    /code/create/sandbox/perms|<directives> the same, written straight after `perms`
    GET    /code/sandbox/{id}                      one sandbox and its files
    DELETE /code/sandbox/{id}
    GET    /code/sandbox/{id}/files/{path}         read a file
    PUT    /code/sandbox/{id}/files/{path}         write one: {"content": "..."}
    DELETE /code/sandbox/{id}/files/{path}
    POST   /code/sandbox/{id}/run                  {"path": "main.py"} or {"code": "...", "language": "python"}

Who may do what:

* Reading -- ``GET /code``, ``/create``, ``perms``, and a perms request
  that only reads (``s1=?``) -- is any credential HyperLink accepts.
* Creating a sandbox and running code need the operator's switch,
  ``T1_CODE_SANDBOX=1``, and an admin key or a key with ``write``.
  Running code is a shell with extra steps, and a read-only token is
  not a shell.
* Changing the permissions needs an admin key, as changing the web
  search settings does.

The permissions are the server's, held on the app like ``/web/v1``'s
settings: a restart puts back the cautious defaults. Everything that is
not HTTP lives in :mod:`hypernix.t1api.codebox`.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse

from .. import codebox as cb
from ..audit import AuditCategory, AuditOutcome
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_audit_log,
    get_config,
    get_hyperlink_principal,
    get_request_id,
)
from ..errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code", tags=["code"])

HOW_TO_ENABLE = "set T1_CODE_SANDBOX=1 in the server's .env and restart it"

_ERROR_CODES = {
    404: T1ErrorCode.NOT_FOUND,
    403: T1ErrorCode.NOT_SUPPORTED,
    409: T1ErrorCode.CONFLICT,
    413: T1ErrorCode.QUOTA_EXCEEDED,
    501: T1ErrorCode.NOT_SUPPORTED,
}


# -- shared state --------------------------------------------------------


def _perms(request: Request) -> cb.CodePerms:
    current = getattr(request.app.state, "t1_code_perms", None)
    if current is None:
        current = cb.CodePerms()
        request.app.state.t1_code_perms = current
    return current


def _store(request: Request, config: T1APIConfig) -> cb.SandboxStore:
    store = getattr(request.app.state, "t1_code_store", None)
    if store is None:
        store = cb.SandboxStore(getattr(config, "code_sandbox_dir", None))
        request.app.state.t1_code_store = store
    return store


def _who(principal: HyperLinkPrincipal) -> str:
    return principal.device_id or principal.owner or "keyless"


def _may_run(principal: HyperLinkPrincipal) -> bool:
    return principal.is_admin or "write" in principal.scopes or "admin" in principal.scopes


def _require_runner(principal: HyperLinkPrincipal, config: T1APIConfig) -> None:
    if not getattr(config, "code_sandbox", False):
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            f"Running code is off on this server. Its operator can turn it on: {HOW_TO_ENABLE}.",
            http_status=403,
            details={"how_to_enable": HOW_TO_ENABLE},
        )
    if not _may_run(principal):
        raise T1APIError(
            T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
            "Creating sandboxes and running code needs an admin key or a key with "
            "the 'write' scope.",
            http_status=403,
        )


def _refused(exc: cb.SandboxError) -> T1APIError:
    return T1APIError(
        _ERROR_CODES.get(exc.status, T1ErrorCode.VALIDATION_ERROR),
        str(exc),
        http_status=exc.status,
        details={"reason": exc.code},
    )


def _audit(audit, action: str, principal, request_id: str, details: dict[str, Any]) -> None:
    # Before the run, so code that takes the server down is still in the log.
    try:
        audit.record(
            f"code.{action}", category=AuditCategory.ADMIN, outcome=AuditOutcome.SUCCESS,
            actor_key_id=_who(principal), request_id=request_id, resource_type="code",
            details=details,
        )
    except Exception:  # noqa: BLE001 - auditing must not be why a run fails
        logger.debug("code: audit record failed", exc_info=True)


def _machine(config: T1APIConfig) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(config, "code_sandbox", False)),
        "how_to_enable": HOW_TO_ENABLE,
        "installed_languages": sorted(cb.available_languages()),
        "network_isolation": cb.network_isolation() is not None,
    }


# -- the tree ------------------------------------------------------------


@router.get("")
@router.get("/")
def code_index(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Whether this server runs code, what with, and where to go next."""
    perms = _perms(request)
    return {
        **_machine(config),
        "you_may_run": _may_run(principal),
        "perms": perms.to_dict(),
        "sandboxes": len(_store(request, config).list(_who(principal))),
        "next": {
            "create": "/code/create",
            "sandbox": "/code/create/sandbox",
            "perms": "/code/create/sandbox/perms",
        },
        "request_id": request_id,
    }


@router.get("/create")
def code_create_help(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """What can be created, and the body each takes."""
    return {
        **_machine(config),
        "run_once": {"method": "POST", "path": "/code/create",
                     "body": {"code": "print('hi')", "language": "python", "stdin": ""}},
        "sandbox": {"method": "POST", "path": "/code/create/sandbox",
                    "body": {"name": "optional", "files": {"main.py": "print('hi')"}}},
        "perms": "/code/create/sandbox/perms",
        "request_id": request_id,
    }


@router.post("/create")
def code_run_once(
    request: Request,
    body: dict[str, Any] = Body(...),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    audit=Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Run ``code`` once, in a sandbox that is deleted before this returns."""
    _require_runner(principal, config)
    code = body.get("code")
    if not isinstance(code, str) or not code.strip():
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "send the code to run as `code`")
    language = str(body.get("language") or "python")
    perms = _perms(request)
    store = _store(request, config)
    _audit(audit, "run_once", principal, request_id,
           {"language": language, "code": code[:500]})
    try:
        box = store.create(_who(principal), perms, files=body.get("files") or None)
        try:
            result = store.run(box, perms, language=language, code=code,
                               stdin=str(body.get("stdin") or ""))
        finally:
            store.delete(box)
    except cb.SandboxError as exc:
        raise _refused(exc) from exc
    return {**result.to_dict(), "request_id": request_id}


@router.get("/create/sandbox")
def code_sandboxes(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Your sandboxes (every sandbox, for an admin)."""
    perms = _perms(request)
    store = _store(request, config)
    store.expire(perms)
    boxes = store.list(_who(principal), is_admin=principal.is_admin)
    return {"sandboxes": [b.to_dict(perms) for b in boxes],
            "max_per_caller": cb.MAX_PER_OWNER, "request_id": request_id}


@router.post("/create/sandbox")
def code_create_sandbox(
    request: Request,
    body: dict[str, Any] | None = Body(default=None),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    audit=Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> JSONResponse:
    """A sandbox that stays until it is deleted or sits idle past s7."""
    _require_runner(principal, config)
    body = body or {}
    files = body.get("files") or {}
    if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR,
                         "`files` is {path: text content}")
    perms = _perms(request)
    try:
        box = _store(request, config).create(_who(principal), perms,
                                             name=str(body.get("name") or ""), files=files)
    except cb.SandboxError as exc:
        raise _refused(exc) from exc
    _audit(audit, "sandbox_created", principal, request_id, {"sandbox": box.id})
    return JSONResponse({**box.to_dict(perms), "request_id": request_id}, status_code=201)


# -- permissions ---------------------------------------------------------


def _perms_body(perms: cb.CodePerms, request_id: str, **extra: Any) -> dict[str, Any]:
    return {"perms": perms.to_dict(), "grammar": cb.GRAMMAR, **extra, "request_id": request_id}


@router.get("/create/sandbox/perms")
def code_perms_read(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """What every sandbox may do, and what this machine can enforce."""
    return _perms_body(_perms(request), request_id, machine=_machine(config))


@router.get("/create/sandbox/perms{directives:path}")
def code_perms_write(
    directives: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    audit=Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """``perms/s2?:=on|s3?:=120``, or ``perms|s1=?;s2?:=on``.

    As with ``/web/v1/config``, everything from the first ``?`` on is the
    query string to HTTP, so the clause is rebuilt from the path and the
    raw query before it is read. A request that only reads (``s1=?``,
    ``s1?=k``) is open to anyone who may read the perms; one that
    changes something needs an admin key.
    """
    raw = directives
    if request.url.query:
        raw = f"{directives}?{request.url.query}"
    current = _perms(request)
    try:
        updated, changed, asked = cb.apply_perms(current, raw)
    except cb.GrammarError as exc:
        raise T1APIError(T1ErrorCode.CONFIG_INVALID, str(exc), http_status=400,
                         details={"received": raw[:200], "grammar": cb.GRAMMAR}) from exc
    if updated != current:
        if not principal.is_admin:
            raise T1APIError(T1ErrorCode.AUTH_ADMIN_REQUIRED,
                             "changing the sandbox permissions needs an admin key.",
                             http_status=403)
        request.app.state.t1_code_perms = updated
        _audit(audit, "perms_changed", principal, request_id,
               {"changed": changed, "directives": raw[:200]})
        logger.info("code: perms changed %s by %s", changed, principal.label)
    wanted = {label: updated.to_dict()[label] for label in asked}
    return _perms_body(updated, request_id, changed=changed, asked=wanted)


# -- one sandbox ---------------------------------------------------------


def _box(request: Request, config: T1APIConfig, principal: HyperLinkPrincipal,
         box_id: str) -> tuple[cb.SandboxStore, cb.Sandbox]:
    store = _store(request, config)
    store.expire(_perms(request))
    try:
        return store, store.get(box_id, _who(principal), is_admin=principal.is_admin)
    except cb.SandboxError as exc:
        raise _refused(exc) from exc


@router.get("/sandbox/{box_id}")
def code_sandbox_read(
    box_id: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    _store_, box = _box(request, config, principal, box_id)
    return {**box.to_dict(_perms(request)), "request_id": request_id}


@router.delete("/sandbox/{box_id}")
def code_sandbox_delete(
    box_id: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    store, box = _box(request, config, principal, box_id)
    store.delete(box)
    return {"deleted": box.id, "request_id": request_id}


@router.get("/sandbox/{box_id}/files/{path:path}")
def code_file_read(
    box_id: str,
    path: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    store, box = _box(request, config, principal, box_id)
    try:
        content, truncated = store.read(box, path)
    except cb.SandboxError as exc:
        raise _refused(exc) from exc
    return {"path": path, "content": content, "truncated": truncated, "request_id": request_id}


@router.put("/sandbox/{box_id}/files/{path:path}")
def code_file_write(
    box_id: str,
    path: str,
    request: Request,
    body: dict[str, Any] = Body(...),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    _require_runner(principal, config)
    content = body.get("content")
    if not isinstance(content, str):
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "send the file's text as `content`")
    store, box = _box(request, config, principal, box_id)
    try:
        written = store.write(box, path, content, _perms(request))
    except cb.SandboxError as exc:
        raise _refused(exc) from exc
    return {"path": path, "bytes": written.stat().st_size, "request_id": request_id}


@router.delete("/sandbox/{box_id}/files/{path:path}")
def code_file_delete(
    box_id: str,
    path: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    store, box = _box(request, config, principal, box_id)
    try:
        store.remove(box, path)
    except cb.SandboxError as exc:
        raise _refused(exc) from exc
    return {"deleted": path, "request_id": request_id}


@router.post("/sandbox/{box_id}/run")
def code_sandbox_run(
    box_id: str,
    request: Request,
    body: dict[str, Any] = Body(...),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    audit=Depends(get_audit_log),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Run a file in the sandbox (``path``), or new code (``code``)."""
    _require_runner(principal, config)
    store, box = _box(request, config, principal, box_id)
    code = body.get("code")
    args = body.get("args") or []
    if not isinstance(args, list):
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "`args` is a list of strings")
    _audit(audit, "run", principal, request_id,
           {"sandbox": box.id, "path": str(body.get("path") or ""),
            "code": code[:500] if isinstance(code, str) else ""})
    try:
        result = store.run(box, _perms(request), language=str(body.get("language") or ""),
                           code=code if isinstance(code, str) else None,
                           path=str(body.get("path") or ""), stdin=str(body.get("stdin") or ""),
                           args=[str(a) for a in args])
    except cb.SandboxError as exc:
        raise _refused(exc) from exc
    return {**result.to_dict(), "sandbox": box.id, "request_id": request_id}


__all__ = ["router"]
