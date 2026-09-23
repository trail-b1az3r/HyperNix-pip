"""magnesium — keep other apps out of the model's way.

Element 12. Light, and it burns hot so nothing else has to.

What it does
------------
Lowers the scheduling priority of other running applications, and can
confine them to a subset of cores, so a model that is generating gets
the machine. Everything it changes it records, and :meth:`deactivate`
puts back exactly — the original niceness and the original affinity,
per process — rather than "resetting to normal", which is a guess about
what normal was.

What it never touches
---------------------
Terminals and consoles, shells, Python, HyperNix and llama.cpp, the
desktop's own compositor and audio, the init system, kernel threads,
this process and everything above and below it in the process tree, and
anything owned by another user. That list is checked *before* anything
is changed, by name and by relationship, and a process that matches any
of it is reported as protected rather than silently skipped — so "why
did my editor not slow down" has an answer on screen.

Lowering priority is the default because it is the safe one: a nice'd
process still runs, it just yields. Confining affinity is opt-in
(``cores``), and freezing processes is not offered at all — a frozen
browser holding a lock the model's download needs is a deadlock with a
friendly name.

Plan first
----------
:meth:`Magnesium.plan` says what would be changed and changes nothing.
The CLI shows it before asking, because this is the one element that
reaches outside HyperNix.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from ..system import errorcatalogue as codes
from ..system.errorcodes import HyperNixError
from .hydrogen import Element, ElementSpec

__all__ = ["Magnesium", "Target", "Plan", "PROTECTED_NAMES", "is_protected_name"]

#: Never limited, matched case-insensitively against the process name
#: with any `.exe` removed. Prefixes end in `*`.
PROTECTED_NAMES: frozenset[str] = frozenset({
    # terminals and consoles
    "gnome-terminal", "gnome-terminal-server", "konsole", "kitty",
    "alacritty", "wezterm", "wezterm-gui", "foot", "xterm", "urxvt",
    "tilix", "terminator", "st", "ghostty", "iterm2", "terminal",
    "windowsterminal", "conhost", "openconsole", "cmd", "powershell", "pwsh",
    # shells and multiplexers
    "bash", "zsh", "fish", "sh", "dash", "ksh", "nu", "tmux", "screen",
    "zellij", "ssh", "sshd", "mosh-server",
    # python, and us
    "python*", "pypy*", "hypernix*", "hnx", "llama-server", "llama-cli",
    "llama*", "hyped*", "bun", "node",
    # the machine keeping itself alive
    "systemd*", "init", "launchd", "kernel_task", "wininit", "csrss",
    "smss", "services", "lsass", "svchost", "dwm", "explorer",
    "xorg", "xwayland", "gnome-shell", "kwin_wayland", "kwin_x11",
    "sway", "hyprland", "mutter", "windowserver", "loginwindow",
    "pipewire*", "pulseaudio", "wireplumber", "dbus*", "polkitd",
    "networkmanager", "wpa_supplicant", "udevd", "systemd-udevd",
})

#: The lowest priority magnesium will set, and never lower than this:
#: 19 on POSIX is "only when nothing else wants the CPU", which for an
#: app with a UI is indistinguishable from hung.
DEFAULT_NICE = 10


def is_protected_name(name: str) -> bool:
    key = (name or "").lower().removesuffix(".exe")
    if not key:
        return True        # a nameless process is a kernel thread, or gone
    if key in PROTECTED_NAMES:
        return True
    return any(key.startswith(p[:-1]) for p in PROTECTED_NAMES if p.endswith("*"))


@dataclass
class Target:
    pid: int
    name: str
    #: "limit" | "protected" | "not-ours" | "gone" | "denied"
    decision: str
    reason: str = ""
    original_nice: int | None = None
    original_affinity: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "name": self.name, "decision": self.decision,
                "reason": self.reason}


@dataclass
class Plan:
    targets: list[Target] = field(default_factory=list)

    def of(self, decision: str) -> list[Target]:
        return [t for t in self.targets if t.decision == decision]

    def summary(self) -> str:
        counts = {}
        for t in self.targets:
            counts[t.decision] = counts.get(t.decision, 0) + 1
        return ", ".join(f"{n} {d}" for d, n in sorted(counts.items())) or "nothing"


def _is_kernel_thread(info: dict[str, Any]) -> bool:
    """A kernel thread, told apart from a program by its command line.

    Names cannot do this — there are hundreds (`kworker/3:1H`,
    `ksoftirqd/0`, `irq/24-ACPI:Ged`) and they change between kernels.
    What they share is an empty command line and kthreadd (pid 2) as
    parent. Checked on Linux only; `cmdline` is empty for other reasons
    elsewhere. Owner does not help: run as root, which is how magnesium
    reaches other users' apps, every kernel thread is "ours".
    """
    if not sys.platform.startswith("linux"):
        return False
    cmdline = info.get("cmdline")
    if cmdline is None:
        return False       # not asked for, or unreadable: decide by name
    return not cmdline or info.get("ppid") == 2


def _psutil():
    try:
        import psutil
    except ImportError:
        raise HyperNixError(
            codes.NO_PERMISSION,
            "magnesium needs psutil to see other processes",
        ) from None
    return psutil


class Magnesium(Element):
    spec = ElementSpec(
        symbol="Mg",
        summary="Lower other apps' priority (and optionally their cores) "
                "while a model runs. Never touches terminals, shells, "
                "Python or HyperNix, and puts everything back.",
        version="1.0.0",
        permissions=frozenset({"processes"}),
    )

    def __init__(self, context) -> None:
        super().__init__(context)
        self._changed: dict[int, Target] = {}

    # -- deciding --------------------------------------------------------

    def _family(self, psutil) -> set[int]:
        """This process, every ancestor and every descendant."""
        me = psutil.Process()
        family = {me.pid}
        try:
            family.update(p.pid for p in me.parents())
            family.update(p.pid for p in me.children(recursive=True))
        except psutil.Error:
            pass
        return family

    def plan(self, *, processes=None) -> Plan:
        """What would be changed. Changes nothing."""
        psutil = _psutil()
        family = self._family(psutil)
        me = psutil.Process()
        try:
            my_user = me.username()
        except psutil.Error:
            my_user = None
        extra = {n.lower() for n in self.context.config.get("protect", [])}

        plan = Plan()
        for proc in processes if processes is not None else psutil.process_iter(
            ["pid", "name", "username", "cmdline", "ppid"]
        ):
            info = getattr(proc, "info", None) or {}
            pid = info.get("pid", getattr(proc, "pid", 0))
            name = info.get("name") or ""
            if pid <= 1 or pid in family:
                plan.targets.append(Target(pid, name, "protected",
                                           "this process, its tree, or init"))
                continue
            if _is_kernel_thread(info):
                plan.targets.append(Target(pid, name, "protected",
                                           "kernel thread"))
                continue
            if is_protected_name(name) or name.lower().removesuffix(".exe") in extra:
                plan.targets.append(Target(pid, name, "protected",
                                           "on the never-touch list"))
                continue
            owner = info.get("username")
            if my_user is not None and owner not in (None, my_user):
                plan.targets.append(Target(pid, name, "not-ours",
                                           f"owned by {owner}"))
                continue
            plan.targets.append(Target(pid, name, "limit"))
        return plan

    # -- doing -----------------------------------------------------------

    def activate(self) -> None:
        self.context.require("processes")
        psutil = _psutil()
        nice = int(self.context.config.get("nice", DEFAULT_NICE))
        nice = max(1, min(nice, DEFAULT_NICE if sys.platform == "win32" else 15))
        cores = self.context.config.get("cores")

        for target in self.plan().of("limit"):
            try:
                proc = psutil.Process(target.pid)
                if proc.name() != target.name:
                    # The pid was reused between planning and now. Not the
                    # process we decided about, so not one we touch.
                    target.decision, target.reason = "gone", "pid reused"
                    continue
                target.original_nice = proc.nice()
                if sys.platform == "win32":
                    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                elif target.original_nice < nice:
                    proc.nice(nice)
                if cores and hasattr(proc, "cpu_affinity"):
                    target.original_affinity = list(proc.cpu_affinity())
                    proc.cpu_affinity(list(cores))
                self._changed[target.pid] = target
            except psutil.NoSuchProcess:
                target.decision = "gone"
            except psutil.AccessDenied:
                target.decision = "denied"

    def deactivate(self) -> None:
        """Put back what was changed, per process, exactly."""
        if not self._changed:
            return
        psutil = _psutil()
        for pid, target in list(self._changed.items()):
            try:
                proc = psutil.Process(pid)
                if proc.name() == target.name:
                    if target.original_nice is not None:
                        proc.nice(target.original_nice)
                    if target.original_affinity is not None:
                        proc.cpu_affinity(target.original_affinity)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass   # exited, or already gone from our reach
            self._changed.pop(pid, None)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "limited": len(self._changed),
                "pids": sorted(self._changed)}

    @property
    def changed(self) -> dict[int, Target]:
        return dict(self._changed)
