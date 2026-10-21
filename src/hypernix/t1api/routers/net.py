"""``/net`` — everything ``hnx net`` did, as the T1 API (0.72.8a3).

``hnx net`` is replaced by these routes: its Tailscale and peer operations are these
routes, answering JSON instead of printing, on the machine the T1
server runs on. The work is :mod:`hypernix.system.netops`, shared with
the deprecated CLI so the two cannot drift.

Every route needs an **admin** key: the tailnet's membership, its
addresses and SSH to its peers are this machine's network, not a
caller's. Routes that change something take ``?confirm=true`` (where the
server requires confirmation for destructive operations), and starting
or stopping a script on the peers is off unless ``T1_NET_REMOTE_RUN=1``.

| ``hnx net`` | Route |
|---|---|
| ``status`` | ``GET /net/status`` |
| ``m-ip`` | ``GET /net/ip`` |
| ``connect IP`` | ``GET /net/peers/{peer}/ping`` (an SSH session is not an API call) |
| ``config`` | ``GET`` / ``PUT /net/config`` |
| ``mutli-a-port PORT`` | ``POST /net/ports``, ``DELETE /net/ports/{port}`` |
| ``ex-port PORT [--apply]`` | ``POST /net/expose`` |
| ``a-il`` | ``POST /net/peers/trust`` |
| ``s-storage`` | ``POST /net/storage/sync`` |
| ``onef-all`` | ``GET`` / ``POST /net/hub`` |
| ``m-setup`` / ``auto-setup`` | ``GET`` / ``POST /net/setup`` |
| ``tail acheck FILE [-r]`` | ``POST /net/peers/check``, ``POST /net/peers/run`` |
| ``tail stop FILE`` | ``POST /net/peers/stop`` |
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ...system import netops
from ..auth import AuthContext
from ..config import T1APIConfig
from ..deps import get_auth_context, get_config, require_admin, require_confirmation
from ..errors import T1APIError, T1ErrorCode

router = APIRouter(prefix="/net", tags=["net"])


def _admin(principal: AuthContext = Depends(get_auth_context)) -> AuthContext:
    return require_admin(principal)


async def _thread(fn, *args, **kwargs):
    import anyio

    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


def _bad(exc: Exception) -> T1APIError:
    return T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc), http_status=400)


class ConfigBody(BaseModel):
    storage_dir: str | None = None
    ports: list[int] | None = None
    hub: bool | None = None


class PortBody(BaseModel):
    port: int = Field(..., ge=1, le=65535)


class ExposeBody(BaseModel):
    port: int = Field(..., ge=1, le=65535)
    apply: bool = False


class SyncBody(BaseModel):
    peers: list[str] | None = None
    dry_run: bool = False


class FileBody(BaseModel):
    file: str = Field(..., min_length=1, max_length=4096)


@router.get("/status")
async def net_status(_: AuthContext = Depends(_admin)):
    """This node, its peers, and whether Tailscale is up."""
    return await _thread(netops.status)


@router.get("/ip")
async def net_ip(_: AuthContext = Depends(_admin)):
    return await _thread(netops.tailnet_ip)


@router.get("/peers")
async def net_peers(online: bool = True, _: AuthContext = Depends(_admin)):
    return {"peers": await _thread(netops.peers, online_only=online)}


@router.get("/peers/{peer}/ping")
async def net_ping(peer: str, _: AuthContext = Depends(_admin)):
    result = await _thread(netops.ping, peer)
    return {"peer": peer, "reachable": result.ok, **result.to_dict()}


@router.get("/config")
async def net_config(_: AuthContext = Depends(_admin)):
    return netops.load_config()


@router.put("/config")
async def net_config_set(body: ConfigBody, _: AuthContext = Depends(_admin)):
    config = netops.load_config()
    config.update({k: v for k, v in body.model_dump().items() if v is not None})
    try:
        return netops.save_config(config)
    except netops.NetError as exc:
        raise _bad(exc) from exc


@router.post("/ports")
async def net_port_add(body: PortBody, _: AuthContext = Depends(_admin)):
    config = netops.load_config()
    config["ports"] = sorted({*config["ports"], body.port})
    return netops.save_config(config)


@router.delete("/ports/{port}")
async def net_port_remove(port: int, _: AuthContext = Depends(_admin)):
    config = netops.load_config()
    if port not in config["ports"]:
        raise T1APIError(T1ErrorCode.NOT_FOUND, f"port {port} is not registered", http_status=404)
    config["ports"] = [p for p in config["ports"] if p != port]
    return netops.save_config(config)


@router.post("/expose")
async def net_expose(body: ExposeBody, request: Request, _: AuthContext = Depends(_admin)):
    """``tailscale serve --bg PORT``; without ``apply``, the command it would run."""
    if body.apply:
        require_confirmation(request, action=f"Serving port {body.port} on the tailnet")
    return await _thread(netops.expose, body.port, apply=body.apply)


@router.post("/peers/trust")
async def net_trust(request: Request, _: AuthContext = Depends(_admin)):
    """Add every online peer's SSH host keys to known_hosts."""
    require_confirmation(request, action="Trusting the SSH host keys of every peer")
    return {"peers": await _thread(netops.trust_peers)}


@router.post("/storage/sync")
async def net_sync(body: SyncBody, request: Request, _: AuthContext = Depends(_admin)):
    """rsync the storage folder to the online peers (or the ones named)."""
    if not body.dry_run:
        require_confirmation(request, action="Copying the storage folder to the peers")
    return {"peers": await _thread(netops.sync_storage, only=body.peers, dry_run=body.dry_run),
            "storage_dir": netops.load_config()["storage_dir"]}


@router.get("/hub")
async def net_hub(_: AuthContext = Depends(_admin)):
    """Whether this node is the central storage node, and the address peers use for it."""
    config = netops.load_config()
    return {"hub": config["hub"], "storage_dir": config["storage_dir"], **await _thread(netops.tailnet_ip)}


@router.post("/hub")
async def net_hub_set(_: AuthContext = Depends(_admin)):
    config = netops.load_config()
    config["hub"] = True
    saved = netops.save_config(config)
    return {"hub": True, "storage_dir": saved["storage_dir"], **await _thread(netops.tailnet_ip)}


@router.get("/setup")
async def net_setup_steps(_: AuthContext = Depends(_admin)):
    return {"steps": netops.setup_steps()}


@router.post("/setup")
async def net_setup(request: Request, _: AuthContext = Depends(_admin)):
    """``tailscale up`` (its login link, if it needs one) and an SSH key."""
    require_confirmation(request, action="Bringing Tailscale up")
    return await _thread(netops.setup)


@router.post("/peers/check")
async def net_check(body: FileBody, _: AuthContext = Depends(_admin)):
    return {"file": body.file, "peers": await _thread(netops.check_file, body.file)}


def _remote_run_allowed(config: T1APIConfig) -> None:
    if not config.net_remote_run:
        raise T1APIError(T1ErrorCode.NOT_SUPPORTED,
                         "Running scripts on the peers is off. Set T1_NET_REMOTE_RUN=1 on this server "
                         "to allow it (it starts programs on other machines).", http_status=501)


@router.post("/peers/run")
async def net_run(body: FileBody, request: Request, _: AuthContext = Depends(_admin),
                  config: T1APIConfig = Depends(get_config)):
    """Start ``python3 FILE`` on every online peer that has it."""
    _remote_run_allowed(config)
    require_confirmation(request, action=f"Running {body.file} on the peers")
    return {"file": body.file, "peers": await _thread(netops.run_file, body.file)}


@router.post("/peers/stop")
async def net_stop(body: FileBody, request: Request, _: AuthContext = Depends(_admin),
                   config: T1APIConfig = Depends(get_config)):
    _remote_run_allowed(config)
    require_confirmation(request, action=f"Stopping {body.file} on the peers")
    return {"file": body.file, "peers": await _thread(netops.stop_file, body.file)}
