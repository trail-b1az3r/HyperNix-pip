"""hyperlink.managed — running a model without LM Studio.

HyperLink could reach exactly one thing: an LM Studio server somebody had
started by hand, on the machine, with the right settings. So the server
could hold forty GGUFs and serve none of them, and "switch model" meant
alt-tabbing on the PC.

This owns a llama.cpp server process instead. It starts it, knows what it
is serving, and stops it — which is what makes load, unload and switch
real operations rather than instructions to go and do something manually.

Where the weights go
--------------------
The request was specific about this and it is the part that decides
whether a model runs at all on a given machine: a configurable number of
layers on the GPU with the remainder in system RAM, and placement across
GPU, RAM, swap and CPU.

``Placement`` is that decision, and it can be:

* **explicit** — "31 layers on the GPU", because somebody who has tuned
  their own machine should not have their number second-guessed, and
* **worked out** — from the model's size, the free VRAM and the free RAM,
  for somebody who has not.

The automatic path is deliberately conservative. Guessing one layer too
many does not degrade gracefully: CUDA returns out-of-memory at load and
the model does not run at all, so the headroom below is not timidity, it
is the difference between "slower than it could be" and "does not work".

Swap is reported and never targeted. A model paging through swap produces
tokens at a rate that reads as a hang, and offering it as a placement
would be offering something nobody wants; what swap is for here is
explaining *why* a model was refused.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

logger = logging.getLogger(__name__)

__all__ = [
    "Placement",
    "plan_placement",
    "ManagedModel",
    "ManagedRunner",
    "ManagedError",
    "BACKENDS",
]

#: Backends a caller may ask for. `auto` picks by what the machine has.
#:
#: The llama.cpp ones differ by compute API rather than by code path:
#: CUDA on NVIDIA, Vulkan on nearly anything with a GPU including AMD and
#: Intel, CPU when there is nothing to offload to.
BACKENDS = ("auto", "cuda", "vulkan", "cpu", "hnx-cuda", "hnx-cpu")

#: Leave this much VRAM alone. The display server wants some, the CUDA
#: context itself wants some, and fragmentation takes more than the
#: arithmetic suggests. 12% plus a fixed floor is what stops an automatic
#: plan from being one layer too optimistic — which is not a small
#: degradation, it is a load that fails outright.
VRAM_HEADROOM_FRACTION = 0.12
VRAM_HEADROOM_BYTES = 512 * 1024 * 1024

#: The same for system RAM, and larger: everything else on the machine is
#: also using it, and an OOM killer takes the whole process.
RAM_HEADROOM_FRACTION = 0.20

#: How long to wait for a freshly started server to answer. Loading a 70B
#: off a spinning disk genuinely takes this long, and a timeout that
#: fires early looks exactly like a crash.
DEFAULT_START_TIMEOUT = 300.0


def _library_env(bin_dir: Path) -> dict[str, str]:
    """The environment llama-server needs to find its own libraries.

    A build tree keeps libggml, libllama and the rest next to the binary
    rather than installing them, and the binary does not always carry an
    rpath that finds them. Inheriting the parent environment unchanged
    gets "error while loading shared libraries: libggml.so" — which reads
    like a broken build and is a missing search path.

    Prepended, not replaced: an operator who has set one for a reason
    keeps it.
    """
    import os

    environment = dict(os.environ)
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    existing = environment.get(variable, "")
    environment[variable] = (
        f"{bin_dir}{os.pathsep}{existing}" if existing else str(bin_dir)
    )
    return environment


class ManagedError(RuntimeError):
    """A model could not be started, stopped, or asked about."""


@dataclass
class Placement:
    """Where a model's weights are going to sit."""

    #: Layers to put on the GPU. 0 means everything in system RAM.
    gpu_layers: int = 0
    #: What the model has in total, so "31 of 33" can be shown rather
    #: than a bare number that means nothing without it.
    total_layers: int = 0
    backend: str = "cpu"
    #: Bytes expected on the card and in RAM. Estimates — the real split
    #: is llama.cpp's to make — but close enough to refuse a model that
    #: cannot fit anywhere.
    vram_bytes: int = 0
    ram_bytes: int = 0
    #: Swap in use on this machine. Reported, never planned into: a model
    #: paging through swap produces tokens at a rate that reads as a
    #: hang.
    swap_used_bytes: int = 0
    #: Why this plan, in a sentence somebody can disagree with.
    reason: str = ""
    #: True when the caller said so rather than this working it out.
    explicit: bool = False

    @property
    def fully_offloaded(self) -> bool:
        return bool(self.total_layers) and self.gpu_layers >= self.total_layers

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_layers": self.gpu_layers,
            "total_layers": self.total_layers,
            "backend": self.backend,
            "vram_bytes": self.vram_bytes,
            "ram_bytes": self.ram_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "fully_offloaded": self.fully_offloaded,
            "explicit": self.explicit,
            "reason": self.reason,
        }


def _free_memory() -> tuple[int, int, int]:
    """``(free_ram, total_ram, swap_used)``, or zeros when unknowable."""
    try:
        import psutil
    except ImportError:
        return 0, 0, 0
    try:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return int(virtual.available), int(virtual.total), int(swap.used)
    except Exception:  # noqa: BLE001 - a missing reading is not an error
        return 0, 0, 0


def _free_vram() -> tuple[int, int]:
    """``(free, total)`` VRAM in bytes across every card."""
    try:
        from ..system import gpus

        cards = gpus.detect()
    except Exception as exc:  # noqa: BLE001 - no vendor tool is guaranteed
        logger.debug("managed: no GPU detected: %s", exc)
        return 0, 0
    total = used = 0
    for card in cards:
        card_total = getattr(card, "memory_total_mb", None)
        card_used = getattr(card, "memory_used_mb", None)
        if card_total:
            total += int(card_total) * 1024 * 1024
        if card_used:
            used += int(card_used) * 1024 * 1024
    return max(0, total - used), total


def _detect_backend(vram_total: int) -> str:
    """What this machine can actually use.

    CUDA when there is an NVIDIA card, Vulkan when there is some other
    one, CPU otherwise. Asked of the hardware rather than configured,
    because a configured backend that is not there fails at load with a
    message about a shared library.
    """
    try:
        from ..system import gpus

        cards = gpus.detect()
    except Exception:  # noqa: BLE001
        return "cpu"
    vendors = {str(getattr(c, "vendor", "")).lower() for c in cards}
    if "nvidia" in vendors:
        return "cuda"
    if vendors & {"amd", "intel"}:
        return "vulkan"
    return "cpu" if not vram_total else "vulkan"


def plan_placement(
    *,
    file_bytes: int,
    total_layers: int = 0,
    gpu_layers: int | None = None,
    backend: str = "auto",
    context_length: int = 0,
    vram_free: int | None = None,
    ram_free: int | None = None,
) -> Placement:
    """Decide where the weights go.

    *gpu_layers* is honoured when given — somebody who has tuned their
    own machine should not have their number second-guessed — and worked
    out otherwise.
    """
    if backend not in BACKENDS:
        raise ManagedError(
            f"Unknown backend {backend!r}. Available: {', '.join(BACKENDS)}"
        )

    ram_available, _ram_total, swap_used = _free_memory()
    if ram_free is not None:
        ram_available = ram_free
    vram_available, vram_total = _free_vram()
    if vram_free is not None:
        vram_available, vram_total = vram_free, max(vram_total, vram_free)

    resolved = _detect_backend(vram_total) if backend == "auto" else backend
    on_cpu = resolved in ("cpu", "hnx-cpu")

    if on_cpu or not vram_available:
        reason = (
            "running entirely on the CPU"
            if on_cpu
            else "no GPU memory is free, so everything is in system RAM"
        )
        # The tightest fit there is, and the one most worth warning
        # about: everything is in RAM, so if it does not fit there the
        # machine starts swapping and generation slows to something that
        # reads as a hang. Said rather than refused -- somebody may know
        # something this does not -- but not silent.
        if ram_available and file_bytes > ram_available * (1 - RAM_HEADROOM_FRACTION):
            reason += (
                f" — and this will be tight: {file_bytes} bytes against "
                f"{ram_available} free, so expect swapping"
            )
        return Placement(
            gpu_layers=0, total_layers=total_layers, backend=resolved,
            ram_bytes=file_bytes, swap_used_bytes=swap_used,
            explicit=gpu_layers is not None,
            reason=reason,
        )

    # The KV cache lives on the card too, and it scales with context. A
    # plan that budgets only for weights is the plan that loads and then
    # runs out of memory on a long conversation.
    kv_bytes = int(context_length * 0.5 * 1024) if context_length else 0
    usable = max(
        0,
        int(vram_available * (1 - VRAM_HEADROOM_FRACTION))
        - VRAM_HEADROOM_BYTES
        - kv_bytes,
    )

    if gpu_layers is not None:
        chosen = max(0, int(gpu_layers))
        if total_layers:
            chosen = min(chosen, total_layers)
        per_layer = file_bytes / total_layers if total_layers else 0
        return Placement(
            gpu_layers=chosen, total_layers=total_layers, backend=resolved,
            vram_bytes=int(per_layer * chosen),
            ram_bytes=int(file_bytes - per_layer * chosen),
            swap_used_bytes=swap_used, explicit=True,
            reason=f"{chosen} layer(s) on the GPU, as asked",
        )

    if not total_layers:
        # No layer count to divide by. All or nothing, decided on whether
        # the whole file fits — a partial offload needs a denominator.
        if file_bytes and usable >= file_bytes:
            return Placement(
                gpu_layers=999, total_layers=0, backend=resolved,
                vram_bytes=file_bytes, swap_used_bytes=swap_used,
                reason="the whole model fits in VRAM (layer count unknown, so all of it)",
            )
        return Placement(
            gpu_layers=0, total_layers=0, backend=resolved,
            ram_bytes=file_bytes, swap_used_bytes=swap_used,
            reason="layer count unknown and the model does not clearly fit, so RAM",
        )

    per_layer = file_bytes / total_layers
    fit = int(usable // per_layer) if per_layer else 0
    fit = max(0, min(fit, total_layers))

    if fit >= total_layers:
        reason = f"all {total_layers} layers fit in VRAM with room for the KV cache"
    elif fit:
        reason = (
            f"{fit} of {total_layers} layers fit in the VRAM that is free; "
            f"the rest run from system RAM"
        )
    else:
        reason = "not enough free VRAM for even one layer, so everything is in RAM"

    remaining = int(file_bytes - per_layer * fit)
    if ram_available and remaining > ram_available * (1 - RAM_HEADROOM_FRACTION):
        reason += (
            f" — and this will be tight: {remaining} bytes in RAM against "
            f"{ram_available} free"
        )

    return Placement(
        gpu_layers=fit, total_layers=total_layers, backend=resolved,
        vram_bytes=int(per_layer * fit), ram_bytes=remaining,
        swap_used_bytes=swap_used, reason=reason,
    )


@dataclass
class ManagedModel:
    """What is loaded right now."""

    model_id: str
    path: str
    port: int
    placement: Placement
    started_at: float
    context_length: int = 0
    pid: int = 0
    #: What a client talks to. OpenAI-compatible, so the existing bridge
    #: client works against it unchanged — which is the point: nothing
    #: downstream has to learn a second protocol.
    base_url: str = ""
    #: The vision projector loaded with it, "" for none. A model takes
    #: images through the runner exactly when this is set.
    mmproj: str = ""

    @property
    def supports_images(self) -> bool:
        return bool(self.mmproj)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": self.path,
            "port": self.port,
            "pid": self.pid,
            "base_url": self.base_url,
            "context_length": self.context_length,
            "started_at": self.started_at,
            "uptime_seconds": round(max(0.0, time.time() - self.started_at), 1),
            "placement": self.placement.to_dict(),
            "mmproj": self.mmproj,
            "supports_images": self.supports_images,
        }


class ManagedRunner:
    """The llama.cpp server process this server owns.

    One model at a time, on purpose. Two would need VRAM accounting
    between them and a policy for what to evict, and "switch model" is
    what was asked for — not "run several and route between them", which
    the registry and the routing cascade already do for backends that
    exist.
    """

    def __init__(self, *, port: int = 8781, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._current: ManagedModel | None = None

    # -- state ----------------------------------------------------------

    @property
    def current(self) -> ManagedModel | None:
        """What is loaded, or None.

        Checks the process is still alive rather than trusting the
        record: llama.cpp exits on its own when a load fails after the
        fork — a GGUF it cannot read, a CUDA allocation it cannot make —
        and a runner that kept reporting it as loaded would send every
        request to a socket nobody is listening on.
        """
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                logger.warning(
                    "managed: the server for %s exited with %s",
                    self._current.model_id if self._current else "?",
                    self._process.returncode,
                )
                self._process = None
                self._current = None
            return self._current

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- lifecycle ------------------------------------------------------

    def load(
        self,
        path: str | Path,
        *,
        model_id: str = "",
        gpu_layers: int | None = None,
        backend: str = "auto",
        context_length: int = 0,
        total_layers: int = 0,
        timeout: float = DEFAULT_START_TIMEOUT,
        mmproj: str | None = None,
    ) -> ManagedModel:
        """Start serving *path*, replacing whatever was running.

        *mmproj* is the vision projector: ``None`` finds the one beside
        the model (``mmproj-*.gguf``), ``""`` loads none.

        Unloads first rather than starting a second process: two
        llama.cpp servers on one machine will each try to take the VRAM
        the other has, and the failure is an out-of-memory on the one
        that was working.
        """
        model_path = Path(path).expanduser()
        from .brewed import is_brewed_dir

        if is_brewed_dir(model_path):
            converted = brewed_gguf_for(model_path)
            if converted is None:
                return self._load_brewed(model_path, model_id=model_id, backend=backend,
                                         timeout=timeout)
            # The same model through llama.cpp: a KV cache, quantised
            # weights, and the GPU layer split, none of which the PyTorch
            # server has.
            import json as _json

            config = _json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            model_id = model_id or str(config.get("name") or model_path.name).lower()
            total_layers = total_layers or int(config.get("n_layers") or 0)
            model_path = converted
        if not model_path.is_file():
            raise ManagedError(f"No such model: {model_path}")

        from ..quant.gguf import GGUFError, GGUFFile
        from ..quant.runtime_bridge import BridgeError, find_build, find_mmproj, serve_argv

        try:
            header = GGUFFile.read(model_path)
        except (GGUFError, OSError):
            header = None           # llama-server says what is wrong with it
        types = (frozenset(int(t.ggml_type) for t in header.tensors)
                 if header is not None else frozenset())
        if header is not None and header.metadata.get("general.architecture") == "hypernix":
            raise ManagedError(_hypernix_architecture_message(model_path))
        try:
            build = find_build(need=types)
        except BridgeError as exc:
            raise ManagedError(
                f"{exc}\n\nThis is what HyperNix serves models with when LM "
                f"Studio is not involved."
            ) from exc
        missing = build.missing(types)
        if missing:
            # Refused here, in words, rather than handed to llama-server
            # to fail on as "invalid ggml type 210. should be in [0, 43)".
            raise ManagedError(_unreadable_types_message(model_path, build, missing))

        placement = plan_placement(
            file_bytes=model_path.stat().st_size,
            total_layers=total_layers,
            gpu_layers=gpu_layers,
            backend=backend,
            context_length=context_length,
        )

        self.unload()

        projector = find_mmproj(model_path) if mmproj is None else (
            Path(mmproj).expanduser() if mmproj else None)
        if projector is not None and not projector.is_file():
            raise ManagedError(f"No vision projector at {projector}.")
        argv = serve_argv(
            build, model_path, host=self.host, port=self.port,
            gpu_layers=placement.gpu_layers, context=context_length,
            alias=model_id or model_path.stem, mmproj=projector,
        )
        logger.info("managed: starting %s", " ".join(argv))
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, never a shell
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=_library_env(build.bin_dir),
            )
        except OSError as exc:
            raise ManagedError(f"Could not start {argv[0]}: {exc}") from exc

        record = ManagedModel(
            model_id=model_id or model_path.stem,
            path=str(model_path),
            port=self.port,
            placement=placement,
            started_at=time.time(),
            context_length=context_length,
            pid=process.pid,
            base_url=self.base_url,
            mmproj=str(projector) if projector else "",
        )

        with self._lock:
            self._process = process
            self._current = record

        if not self._wait_until_ready(process, timeout):
            code = process.poll()
            output = self._drain(process)
            self.unload()
            what = (f"llama-server exited (code {code}) while loading {record.model_id}"
                    if code is not None
                    else f"{record.model_id} did not start within {timeout:.0f}s")
            raise ManagedError(
                f"{what}.\n"
                f"Placement was: {placement.reason}\n"
                f"Last output:\n{output[-2000:]}"
            )
        return record

    def _load_brewed(self, folder: Path, *, model_id: str, backend: str,
                     timeout: float) -> ManagedModel:
        """Serve a native HyperNix checkpoint (see :mod:`.brewed_server`).

        Same contract as a GGUF load: whatever was running is unloaded,
        the process is watched until it answers ``/health``, and its last
        output is in the error if it never does. PyTorch holds the whole
        model on one device, so there is no layer split to plan.
        """
        import json as _json

        if backend not in BACKENDS:
            raise ManagedError(f"Unknown backend {backend!r}. Available: {', '.join(BACKENDS)}")
        device = ("cpu" if backend in ("cpu", "hnx-cpu")
                  else "cuda" if backend in ("cuda", "hnx-cuda") else "auto")
        try:
            config = _json.loads((folder / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ManagedError(f"{folder}/config.json is unreadable: {exc}") from exc
        layers = int(config.get("n_layers") or 0)
        context = int(config.get("max_seq_len") or 0)
        placement = Placement(
            gpu_layers=0 if device == "cpu" else layers, total_layers=layers,
            backend=device, explicit=device != "auto",
            reason=(f"native HyperNix model: all {layers} layers in PyTorch on "
                    f"{'the CPU' if device == 'cpu' else 'the GPU when there is one'}, "
                    f"no llama.cpp"),
        )
        self.unload()
        alias = model_id or str(config.get("name") or folder.name).lower()
        argv = [sys.executable, "-m", "hypernix.hyperlink.brewed_server",
                "--model", str(folder), "--host", self.host, "--port", str(self.port),
                "--alias", alias, "--device", device]
        logger.info("managed: starting %s", " ".join(argv))
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, never a shell
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except OSError as exc:
            raise ManagedError(f"Could not start the HyperNix model server: {exc}") from exc
        record = ManagedModel(
            model_id=alias, path=str(folder), port=self.port, placement=placement,
            started_at=time.time(), context_length=context, pid=process.pid,
            base_url=self.base_url,
        )
        with self._lock:
            self._process = process
            self._current = record
        if not self._wait_until_ready(process, timeout):
            code = process.poll()
            output = self._drain(process)
            self.unload()
            what = (f"the HyperNix model server exited (code {code}) while loading {alias}"
                    if code is not None else f"{alias} did not start within {timeout:.0f}s")
            raise ManagedError(
                f"{what}.\n"
                f"Last output:\n{output[-2000:]}"
            )
        return record

    def unload(self) -> bool:
        """Stop whatever is running. True if something was."""
        with self._lock:
            process, self._process, self._current = self._process, None, None
        if process is None:
            return False
        if process.poll() is not None:
            return True
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # A server mid-load holds the GPU allocation it has already
            # made. Waiting politely for ever means the next load fails
            # on memory that belongs to a process nobody is using.
            logger.warning("managed: server did not stop; killing it")
            process.kill()
            process.wait(timeout=10)
        return True

    # -- readiness ------------------------------------------------------

    def _wait_until_ready(self, process: subprocess.Popen, timeout: float) -> bool:
        """Poll until the server answers, or it dies, or time runs out.

        Polling the health endpoint rather than sleeping a fixed time:
        loading a 70B off a slow disk takes minutes and loading a 1B
        takes a second, and a fixed wait is either a stall or a race.
        """
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                with safe_urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(0.5)
        return False

    @staticmethod
    def _drain(process: subprocess.Popen) -> str:
        """Whatever the server said before it gave up.

        The reason a load failed is in here — "cannot allocate", "unknown
        model architecture" — and throwing it away leaves a caller with
        "it did not start" and nothing to act on.
        """
        if process.stdout is None:
            return ""
        try:
            process.terminate()
            return process.stdout.read() or ""
        except Exception:  # noqa: BLE001
            return ""


# ---------------------------------------------------------------------------
# Several instances of one model
# ---------------------------------------------------------------------------


class ManagedPool:
    """N copies of one model, on N consecutive ports.

    :class:`ManagedRunner`'s one-at-a-time rule is the right default and
    the docstring above says why. This is the opt-in exception, for the
    case it does not cover: several people (or one person with several
    tabs) waiting on one server, where the serialisation is the whole
    of the latency.

    Three things it has to get right, none of which a loop over
    ``ManagedRunner`` gets for free.

    **Ports.** Instances sit on ``port, port+1, …`` and none of them may
    land on the T1 server's own port — a collision there does not fail
    loudly, it takes the API down and leaves llama.cpp answering on it.

    **All or nothing.** If instance three of four fails to start, the
    two that did are orphans: holding VRAM, answering nothing, invisible
    to the next load's placement arithmetic. A partial load unwinds
    completely and reports the failure.

    **Eviction is the same operation.** ``load`` replaces whatever the
    pool was running, on every instance, exactly as the single runner
    does — so "switch model" means the same thing whichever mode the
    server is in.
    """

    def __init__(
        self,
        *,
        instances: int = 2,
        port: int = 8781,
        host: str = "127.0.0.1",
        avoid_ports: Iterable[int] = (),
    ) -> None:
        if instances < 1:
            raise ValueError("a pool needs at least one instance")
        self.host = host
        self.base_port = port
        self.avoid = {int(p) for p in avoid_ports}
        self._lock = threading.Lock()
        self._runners = [
            ManagedRunner(port=p, host=host)
            for p in allocate_ports(port, instances, avoid=self.avoid)
        ]

    @property
    def instances(self) -> int:
        return len(self._runners)

    @property
    def ports(self) -> list[int]:
        return [r.port for r in self._runners]

    @property
    def current(self) -> ManagedModel | None:
        """What the pool is serving, from the first live instance."""
        for runner in self._runners:
            found = runner.current
            if found is not None:
                return found
        return None

    @property
    def live(self) -> list[ManagedRunner]:
        return [r for r in self._runners if r.current is not None]

    @property
    def base_urls(self) -> list[str]:
        return [r.base_url for r in self.live]

    def load(self, path: str | Path, **kwargs: Any) -> list[ManagedModel]:
        """Start every instance on *path*, or leave none of them running."""
        with self._lock:
            self._unload_all()
            started: list[ManagedModel] = []
            try:
                for runner in self._runners:
                    started.append(runner.load(path, **kwargs))
            except Exception as exc:
                # The unwind is the point. Without it a failed third
                # instance leaves two holding VRAM that the next load's
                # placement arithmetic cannot see and will not get.
                logger.warning(
                    "managed: instance %d of %d failed to start (%s); "
                    "stopping the %d that did",
                    len(started) + 1, len(self._runners), exc, len(started),
                )
                self._unload_all()
                raise ManagedError(
                    f"instance {len(started) + 1} of {len(self._runners)} "
                    f"did not start: {exc}. The pool was rolled back, so "
                    f"nothing is running and no memory is held."
                ) from exc
            return started

    def unload(self) -> bool:
        with self._lock:
            return self._unload_all()

    def _unload_all(self) -> bool:
        stopped = False
        for runner in self._runners:
            stopped = runner.unload() or stopped
        return stopped

    def to_dict(self) -> dict[str, Any]:
        current = self.current
        return {
            "mode": "pool",
            "instances": self.instances,
            "ports": self.ports,
            "live": len(self.live),
            "base_urls": self.base_urls,
            "model": current.to_dict() if current else None,
        }


def _hypernix_architecture_message(model: Path) -> str:
    """Why a GGUF labelled `hypernix` will not load, and how to remake it."""
    return (
        f"{model.name} says its architecture is 'hypernix', which no llama.cpp "
        f"knows -- the HyperNix patch adds number formats, not architectures, so "
        f"a patched build refuses it too (\"unknown model architecture: "
        f"'hypernix'\").\n"
        f"It was written by the generic converter, which older versions also used "
        f"for hyperNix0x-v2 models saved without a config.json. Export it again "
        f"from the model folder -- it is now written as 'llama', as "
        f"HyperNix.3-mini is:\n"
        f"  hnx convert /path/to/model-folder -o {model.stem}.gguf\n"
        f"or, for a training run's checkpoint: hnx brew gguf runs/<name>/checkpoints/latest.pt"
    )


def _unreadable_types_message(model: Path, build: Any, missing: list[int]) -> str:
    """Why *build* cannot open *model*, and the one command that fixes it."""
    from ..quant.ggufcheck import _type_name  # noqa: PLC2701

    names = ", ".join(f"{_type_name(t)} ({t})" for t in missing)
    if build.patched:
        why = (f"The llama.cpp at {build.bin_dir} was patched before these "
               f"types were added.")
        fix = "Rebuild it: ./native/ggml-hnx/build.sh"
    else:
        why = (f"The llama.cpp at {build.bin_dir} is a stock build, which reads "
               f"no HyperNix types.")
        fix = ("Build the patched one: ./native/ggml-hnx/build.sh (it records "
               "where it built, so this server finds it), or point at one with "
               "HNX_LLAMA_BUILD=/path/to/build")
    return (f"{model.name} uses {names}, and no llama.cpp build here can read "
            f"{'it' if len(missing) == 1 else 'them'}.\n{why}\n{fix}")


def allocate_ports(
    start: int, count: int, *, avoid: Iterable[int] = (), ceiling: int = 65535
) -> list[int]:
    """*count* ports from *start* upwards, skipping *avoid*.

    Skipping rather than failing: the port to avoid is normally the T1
    server's own, and a pool that refuses to start because its third
    instance would have landed on it is worse than one that takes the
    next port up.
    """
    if count < 1:
        raise ValueError("need at least one port")
    blocked = {int(p) for p in avoid}
    found: list[int] = []
    port = int(start)
    while len(found) < count:
        if port > ceiling:
            raise ManagedError(
                f"ran out of ports above {start} before finding {count}"
            )
        if port not in blocked:
            found.append(port)
        port += 1
    return found


#: ``auto`` serves a brewed model through llama.cpp when there is a build,
#: converting it once; ``torch`` always uses the PyTorch server; ``llama``
#: requires llama.cpp and says why when it cannot.
BREWED_BACKEND_ENV = "HYPERNIX_BREWED_BACKEND"


def brewed_cache_dir() -> Path:
    """``<T1_CONFIG_DIR>/cache/brewed-gguf``: beside the server, not in the model."""
    import os

    base = os.environ.get("T1_CONFIG_DIR") or str(Path.home() / ".hypernix" / "t1api")
    return Path(base) / "cache" / "brewed-gguf"


def brewed_gguf_for(folder: Path) -> Path | None:
    """A GGUF of the brewed model in *folder* for llama.cpp, or ``None``.

    ``None`` means "serve it with the PyTorch server": the setting says so,
    there is no llama.cpp build, or the model cannot be converted (no
    tokenizer, for one). The conversion is cached by the weights' size
    and modification time, outside the model folder, which may be a
    read-only mirror of symlinks.
    """
    import os

    mode = os.environ.get(BREWED_BACKEND_ENV, "auto").strip().lower() or "auto"
    if mode == "torch":
        return None
    try:
        from ..quant.runtime_bridge import BridgeError, find_build

        find_build()
    except BridgeError as exc:
        if mode == "llama":
            raise ManagedError(f"{BREWED_BACKEND_ENV}=llama, but there is no llama.cpp build: {exc}") from exc
        return None

    weights = next((folder / n for n in ("model.safetensors", "model.pt", "pytorch_model.bin",
                                         "weights.pt") if (folder / n).is_file()), None)
    if weights is None:
        return None
    stat = weights.stat()
    import hashlib

    key = hashlib.sha256(
        f"{folder.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    target = brewed_cache_dir() / f"{folder.name}-{key}.f16.gguf"
    if target.is_file():
        return target
    try:
        from ..models.brewer_gguf import export_gguf

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        report = export_gguf(folder, partial, outtype="f16")
        partial.replace(target)
        logger.info("managed: %s converted for llama.cpp: %s", folder, report.summary())
        for stale in target.parent.glob(f"{folder.name}-*.f16.gguf"):
            if stale != target:
                stale.unlink(missing_ok=True)
        return target
    except Exception as exc:  # noqa: BLE001 - the PyTorch server still runs it
        if mode == "llama":
            raise ManagedError(f"{folder} could not be converted for llama.cpp: {exc}") from exc
        logger.warning("managed: %s stays on the PyTorch server: %s", folder, exc)
        return None
