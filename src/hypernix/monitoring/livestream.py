"""monitoring.livestream — stream a run to a browser or a terminal.

A WebSocket server that broadcasts four kinds of thing as they happen:
execution logs, subagent thoughts (from :mod:`hypernix.interfaces.noodle`),
hardware metrics, and script-generation or quantisation progress.

Why a hand-written WebSocket
----------------------------
The handshake is a SHA-1 and a base64, and the frame format is a length
prefix and an optional mask. That is roughly 120 lines, and it means
this works on a machine with nothing installed — which is the machine it
is for. A training box gets torch and nothing else, and "pip install
websockets before you can watch your run" is a bad trade for 120 lines.

Only what a viewer needs is implemented: text frames, close, ping/pong,
and client-to-server frames read and discarded. Fragmented and binary
frames are rejected explicitly rather than mis-parsed.

Back-pressure, which is the actual hard part
--------------------------------------------
A training loop emits far faster than a browser renders, and the naive
version — write to every socket on every event — turns one slow viewer
into a stalled trainer. So each client has a bounded queue and a
:class:`Client` that falls behind is **dropped**, not waited for. A
dropped viewer reconnects and sees current state; a stalled training run
is an hour of GPU time. The ring buffer means a reconnecting viewer gets
recent history rather than starting blank.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Event",
    "EventKind",
    "LiveStreamServer",
    "HardwareSampler",
    "sample_hardware",
    "WS_GUID",
]

#: RFC 6455. Concatenated with the client's key and hashed for the accept.
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_OP_TEXT, _OP_CLOSE, _OP_PING, _OP_PONG = 0x1, 0x8, 0x9, 0xA


class EventKind:
    LOG = "log"
    THOUGHT = "thought"
    TOOL = "tool"
    METRICS = "metrics"
    PROGRESS = "progress"
    STATUS = "status"


@dataclass
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)
    source: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {"kind": self.kind, "at": self.at, "source": self.source, **self.payload},
            default=str,
        )


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def sample_hardware() -> dict[str, Any]:
    """GPU, CPU and RAM, best-effort and never raising.

    Reads ``/proc`` directly for CPU and RAM rather than requiring
    psutil: this runs every second inside a training process and the
    dependency is not worth it. Falls back to blanks on non-Linux, where
    the GPU numbers are the ones people actually want anyway.
    """
    out: dict[str, Any] = {"gpus": [], "cpu_percent": 0.0, "ram": {}}

    # hypernix.system.gpus rather than nvidia-smi: a stream watched by
    # somebody with a Radeon showed no GPUs at all, which reads as "the
    # stream is broken" rather than "this tool only supports one vendor".
    try:
        from ..system import gpus as _gpus

        for card in _gpus.detect():
            used, total = card.memory_used_mb, card.memory_total_mb
            out["gpus"].append(
                {
                    "index": card.index,
                    "name": card.name,
                    "utilization": card.utilization_pct,
                    "vram_used_mb": used,
                    "vram_total_mb": total,
                    "vram_percent": (
                        round(used / total * 100, 1)
                        if used is not None and total else 0.0
                    ),
                    # None, not 0.0, when the vendor tool declined to
                    # answer. nvidia-smi reports "[N/A]" for power draw
                    # on several consumer cards, and a stream showing
                    # 0 W reads as an idle GPU rather than as a missing
                    # measurement.
                    "temperature_c": card.temperature_c,
                    "power_w": card.power_w,
                }
            )
    except Exception:  # noqa: BLE001 - a sample must never kill the stream
        logger.debug("livestream: GPU sampling failed", exc_info=True)

    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            meminfo = {
                parts[0].rstrip(":"): _f(parts[1])
                for parts in (line.split() for line in handle)
                if len(parts) >= 2
            }
        total_kb = meminfo.get("MemTotal", 0.0)
        available_kb = meminfo.get("MemAvailable", 0.0)
        out["ram"] = {
            "total_mb": round(total_kb / 1024, 1),
            "used_mb": round((total_kb - available_kb) / 1024, 1),
            "percent": round((total_kb - available_kb) / total_kb * 100, 1) if total_kb else 0.0,
        }
    except OSError:
        pass

    try:
        out["cpu_percent"] = _cpu_percent()
        out["load"] = list(os.getloadavg())
    except OSError:
        pass
    return out


_LAST_CPU: tuple[float, float] | None = None


def _cpu_percent() -> float:
    """CPU busy fraction since the previous call.

    Two samples are required for a meaningful number, so the first call
    returns 0 rather than the since-boot average — which is a real number
    and never the one anyone wants.
    """
    global _LAST_CPU
    with open("/proc/stat", encoding="utf-8") as handle:
        fields = [float(v) for v in handle.readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
    total = sum(fields)
    previous, _LAST_CPU = _LAST_CPU, (idle, total)
    if previous is None:
        return 0.0
    idle_delta, total_delta = idle - previous[0], total - previous[1]
    if total_delta <= 0:
        return 0.0
    return round((1.0 - idle_delta / total_delta) * 100, 1)


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


class HardwareSampler(threading.Thread):
    """Publishes a metrics event on an interval."""

    def __init__(self, publish: Callable[[Event], None], *, interval: float = 2.0) -> None:
        super().__init__(daemon=True, name="livestream-hw")
        self.publish = publish
        self.interval = max(0.25, float(interval))
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.publish(Event(EventKind.METRICS, sample_hardware(), source="hardware"))
            except Exception:  # noqa: BLE001 - a sampler must never kill the run
                logger.debug("livestream: hardware sample failed", exc_info=True)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# WebSocket framing
# ---------------------------------------------------------------------------


def _encode_frame(payload: bytes, opcode: int = _OP_TEXT) -> bytes:
    """Server-to-client frame. Never masked, per RFC 6455."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


def _read_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("client closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(sock: socket.socket) -> tuple[int, bytes]:
    """Read one client frame. Raises on anything this server does not do."""
    first, second = _read_exactly(sock, 2)
    fin, opcode = first & 0x80, first & 0x0F
    masked, length = second & 0x80, second & 0x7F
    if not fin:
        # Fragmentation is legal and unnecessary here. Rejecting it beats
        # reassembling it wrong.
        raise ConnectionError("fragmented frames are not supported")
    if length == 126:
        length = struct.unpack(">H", _read_exactly(sock, 2))[0]
    elif length == 127:
        length = struct.unpack(">Q", _read_exactly(sock, 8))[0]
    if length > 1 << 20:
        raise ConnectionError("client frame too large")
    mask = _read_exactly(sock, 4) if masked else b""
    payload = _read_exactly(sock, length) if length else b""
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


@dataclass
class Client:
    """One connected viewer, with a bounded queue.

    ``dropped`` is why the queue is bounded: a viewer that cannot keep up
    is disconnected rather than allowed to block the publisher. See the
    module docstring.
    """

    sock: socket.socket
    address: str
    queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=256))
    dropped: int = 0
    connected_at: float = field(default_factory=time.time)


class LiveStreamServer:
    """A WebSocket server that broadcasts :class:`Event` objects.

    Bound to loopback by default. This publishes training logs, model
    output and hardware details with no authentication; exposing it on
    ``0.0.0.0`` should be a deliberate act, and the default should not
    make it one by accident.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        history: int = 500,
        sample_hardware_every: float = 2.0,
        token: str = "",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.token = token
        self._history: deque[Event] = deque(maxlen=int(history))
        self._clients: list[Client] = []
        self._lock = threading.Lock()
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sampler = (
            HardwareSampler(self.publish, interval=sample_hardware_every)
            if sample_hardware_every > 0
            else None
        )

        if host not in ("127.0.0.1", "localhost", "::1") and not token:
            logger.warning(
                "livestream: bound to %s with no token. This stream carries training logs, "
                "model output and hardware details, and has no authentication. Pass a token, "
                "or bind to loopback and use an SSH tunnel.",
                host,
            )

    # -- lifecycle ----------------------------------------------------

    def start(self) -> int:
        """Start listening. Returns the port actually bound."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(16)
        self.port = self._server.getsockname()[1]

        accept = threading.Thread(target=self._accept_loop, daemon=True, name="livestream")
        accept.start()
        self._threads.append(accept)
        if self._sampler is not None:
            self._sampler.start()
        logger.info("livestream: ws://%s:%d", self.host, self.port)
        return self.port

    def stop(self) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.stop()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        with self._lock:
            clients, self._clients = list(self._clients), []
        for client in clients:
            try:
                client.sock.close()
            except OSError:
                pass

    def __enter__(self) -> LiveStreamServer:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self.stop()
        return False

    # -- publishing ---------------------------------------------------

    def publish(self, event: Event) -> None:
        """Broadcast an event. Never blocks the caller.

        This is called from a training loop's inner iteration. If it
        could block, a slow viewer would slow the training, which is the
        exact inversion of what a monitoring tool is for.
        """
        self._history.append(event)
        message = _encode_frame(event.to_json().encode("utf-8"))
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.queue.put_nowait(message)
            except queue.Full:
                client.dropped += 1
                if client.dropped > 32:
                    logger.info(
                        "livestream: dropping %s — %d frames behind",
                        client.address, client.dropped,
                    )
                    self._remove(client)

    def log(self, message: str, *, level: str = "info", source: str = "") -> None:
        self.publish(Event(EventKind.LOG, {"message": message, "level": level}, source=source))

    def progress(self, label: str, fraction: float, **extra: Any) -> None:
        self.publish(
            Event(
                EventKind.PROGRESS,
                {"label": label, "fraction": max(0.0, min(1.0, float(fraction))), **extra},
            )
        )

    def agent_event(self, event: Any) -> None:
        """Adapter for :class:`hypernix.interfaces.noodle.AgentEvent`."""
        kind = EventKind.THOUGHT if getattr(event, "kind", "") == "thought" else EventKind.TOOL
        self.publish(
            Event(kind, dict(getattr(event, "detail", {}) or {}),
                  source=getattr(event, "agent", ""))
        )

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # -- connections --------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                sock, address = self._server.accept()
            except OSError:
                break
            thread = threading.Thread(
                target=self._handle, args=(sock, f"{address[0]}:{address[1]}"),
                daemon=True, name="livestream-client",
            )
            thread.start()
            self._threads.append(thread)

    def _handle(self, sock: socket.socket, address: str) -> None:
        try:
            request = self._read_request(sock)
        except (OSError, ConnectionError):
            sock.close()
            return

        if "upgrade: websocket" not in request.lower():
            # A browser hitting the port directly gets the page rather
            # than a blank connection — the difference between "this is
            # broken" and "oh, that is what it is".
            self._serve_page(sock)
            return

        key = ""
        supplied_token = ""
        for line in request.split("\r\n"):
            lowered = line.lower()
            if lowered.startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()
            elif lowered.startswith("get "):
                target = line.split(" ")[1] if " " in line else ""
                if "token=" in target:
                    supplied_token = target.split("token=", 1)[1].split("&")[0]
        if not key:
            sock.close()
            return
        if self.token and supplied_token != self.token:
            sock.sendall(b"HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n")
            sock.close()
            return

        accept = base64.b64encode(
            hashlib.sha1(  # noqa: S324 - RFC 6455 mandates SHA-1 here
                (key + WS_GUID).encode("ascii"), usedforsecurity=False
            ).digest()
        ).decode("ascii")
        sock.sendall(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode("ascii")
        )

        client = Client(sock=sock, address=address)
        with self._lock:
            self._clients.append(client)
        logger.info("livestream: %s connected", address)

        # Replay recent history so a viewer that connects mid-run sees
        # context rather than a blank pane until the next event.
        for event in list(self._history):
            try:
                client.queue.put_nowait(_encode_frame(event.to_json().encode("utf-8")))
            except queue.Full:
                break

        reader = threading.Thread(target=self._read_loop, args=(client,), daemon=True)
        reader.start()
        self._write_loop(client)

    def _read_loop(self, client: Client) -> None:
        """Read client frames: answer pings, notice closes, discard the rest."""
        while not self._stop.is_set():
            try:
                opcode, payload = _read_frame(client.sock)
            except (OSError, ConnectionError, struct.error):
                break
            if opcode == _OP_CLOSE:
                break
            if opcode == _OP_PING:
                try:
                    client.queue.put_nowait(_encode_frame(payload, _OP_PONG))
                except queue.Full:
                    break
        self._remove(client)

    def _write_loop(self, client: Client) -> None:
        while not self._stop.is_set():
            try:
                frame = client.queue.get(timeout=1.0)
            except queue.Empty:
                try:
                    client.sock.sendall(_encode_frame(b"", _OP_PING))
                except OSError:
                    break
                continue
            try:
                client.sock.sendall(frame)
            except OSError:
                break
        self._remove(client)

    def _remove(self, client: Client) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
        try:
            client.sock.close()
        except OSError:
            pass

    @staticmethod
    def _read_request(sock: socket.socket) -> str:
        sock.settimeout(10.0)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("client closed during handshake")
            data += chunk
            if len(data) > 65536:
                raise ConnectionError("handshake too large")
        sock.settimeout(None)
        return data.decode("utf-8", "replace")

    def _serve_page(self, sock: socket.socket) -> None:
        body = PAGE.replace("__PORT__", str(self.port)).encode("utf-8")
        sock.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        sock.close()


#: The viewer. Served from the same port so there is nothing to install
#: and nothing to configure — open the port in a browser and it is there.
#: Palette matches hypernix.scriptgen.theme: dark neutrals, HyperNix red,
#: no purple.
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>hypernix livestream</title>
<style>
  :root{--obsidian:#0e0e0d;--charcoal:#181817;--slate:#212120;--edge:#3e3e3b;
        --text:#e8e6e3;--dim:#a9a6a1;--faint:#7c7975;--red:#c62828;--redb:#e53935;
        --ok:#4a9d5f;--warn:#c9922e}
  *{box-sizing:border-box}
  body{margin:0;background:var(--obsidian);color:var(--text);
       font:13px/1.5 "DejaVu Sans Mono",Menlo,Consolas,monospace}
  header{display:flex;align-items:center;gap:12px;padding:10px 16px;
         background:var(--charcoal);border-bottom:1px solid var(--edge)}
  h1{font-size:15px;margin:0;font-weight:700}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--faint)}
  .dot.on{background:var(--ok)}
  main{display:grid;grid-template-columns:1fr 340px;gap:10px;padding:10px;
       height:calc(100vh - 46px)}
  section{background:var(--charcoal);border:1px solid var(--edge);border-radius:4px;
          display:flex;flex-direction:column;min-height:0}
  h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);
     margin:0;padding:8px 12px;border-bottom:1px solid var(--edge)}
  .feed{overflow-y:auto;padding:8px 12px;flex:1;min-height:0}
  .row{padding:2px 0;border-bottom:1px solid rgba(255,255,255,.03);word-break:break-word}
  .t{color:var(--faint);margin-right:8px}
  .src{color:var(--redb);margin-right:8px}
  .log .msg{color:var(--text)} .warn .msg{color:var(--warn)} .error .msg{color:var(--redb)}
  .thought .msg{color:var(--dim);font-style:italic}
  .tool .msg{color:var(--text)}
  .metric{display:flex;justify-content:space-between;padding:3px 12px}
  .metric b{font-weight:400;color:var(--dim)}
  .bar{height:5px;background:var(--slate);border-radius:3px;margin:2px 12px 8px;overflow:hidden}
  .bar i{display:block;height:100%;background:var(--red)}
  .prog{padding:6px 12px;border-bottom:1px solid var(--edge)}
</style></head><body>
<header><span class="dot" id="dot"></span><h1>hypernix livestream</h1>
  <span style="color:var(--faint)" id="stat">connecting…</span></header>
<main>
  <section><h2>events</h2><div class="feed" id="feed"></div></section>
  <section><h2>hardware</h2><div id="hw" class="feed"></div>
    <h2>progress</h2><div id="prog" class="feed" style="flex:0 0 140px"></div></section>
</main>
<script>
const feed=document.getElementById('feed'),hw=document.getElementById('hw'),
      prog=document.getElementById('prog'),dot=document.getElementById('dot'),
      stat=document.getElementById('stat');
const progs={};
function time(ts){return new Date(ts*1000).toLocaleTimeString()}
function connect(){
  const ws=new WebSocket(`ws://${location.host}${location.search}`);
  ws.onopen=()=>{dot.classList.add('on');stat.textContent='connected'};
  ws.onclose=()=>{dot.classList.remove('on');stat.textContent='disconnected — retrying';
                  setTimeout(connect,2000)};
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.kind==='metrics'){render_hw(d);return}
    if(d.kind==='progress'){render_prog(d);return}
    const row=document.createElement('div');
    row.className='row '+(d.kind==='log'?(d.level||'log'):d.kind);
    row.innerHTML=`<span class="t">${time(d.at)}</span>`+
      (d.source?`<span class="src">${d.source}</span>`:'')+
      `<span class="msg">${escape_(d.message||d.text||d.tool||JSON.stringify(d))}</span>`;
    // Trim rather than grow without bound: a long run would otherwise
    // make the tab unresponsive after a few hours.
    feed.appendChild(row); while(feed.children.length>800)feed.removeChild(feed.firstChild);
    feed.scrollTop=feed.scrollHeight;
  };
}
function escape_(s){const d=document.createElement('div');d.textContent=String(s);
                    return d.innerHTML}
function render_hw(d){
  let html='';
  (d.gpus||[]).forEach(g=>{
    html+=`<div class="metric"><b>GPU${g.index} ${escape_(g.name)}</b><span>${g.utilization}%</span></div>`;
    html+=`<div class="bar"><i style="width:${g.utilization}%"></i></div>`;
    html+=`<div class="metric"><b>VRAM</b><span>${(g.vram_used_mb/1024).toFixed(1)} / ${(g.vram_total_mb/1024).toFixed(1)} GB</span></div>`;
    html+=`<div class="bar"><i style="width:${g.vram_percent}%"></i></div>`;
    if(g.temperature_c)html+=`<div class="metric"><b>temp</b><span>${g.temperature_c}°C  ${g.power_w||0}W</span></div>`;
  });
  if(d.cpu_percent!==undefined){
    html+=`<div class="metric"><b>CPU</b><span>${d.cpu_percent}%</span></div>`;
    html+=`<div class="bar"><i style="width:${d.cpu_percent}%"></i></div>`;}
  if(d.ram&&d.ram.percent!==undefined){
    html+=`<div class="metric"><b>RAM</b><span>${(d.ram.used_mb/1024).toFixed(1)} / ${(d.ram.total_mb/1024).toFixed(1)} GB</span></div>`;
    html+=`<div class="bar"><i style="width:${d.ram.percent}%"></i></div>`;}
  hw.innerHTML=html||'<div class="metric"><b>no metrics yet</b></div>';
}
function render_prog(d){
  progs[d.label]=d;
  prog.innerHTML=Object.values(progs).map(p=>
    `<div class="prog"><div class="metric"><b>${escape_(p.label)}</b>`+
    `<span>${(p.fraction*100).toFixed(0)}%</span></div>`+
    `<div class="bar" style="margin:2px 0"><i style="width:${p.fraction*100}%"></i></div></div>`
  ).join('');
}
connect();
</script></body></html>
"""
