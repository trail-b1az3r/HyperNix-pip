"""protect — monitor blanking with a wake word, and hardware health logging.

What it does
------------
Blanks the monitor, puts the terminal into raw mode, and watches for a
wake word typed blind. Typing it turns the screen back on and exits.
Optionally logs CPU and memory to a file while it waits, because the
screen is off and printing would be both invisible and destructive to
raw mode.

Why this file is longer than "run xset"
---------------------------------------
It used to be exactly that::

    subprocess.run(["xset", "dpms", "force", state], check=False,
                   stdout=DEVNULL, stderr=DEVNULL)

which does nothing at all under four common conditions, and says nothing
about any of them:

* **DPMS is off.** ``xset dpms force off`` is a request to the DPMS
  extension, and if DPMS is disabled — which it is on a lot of desktops,
  because power management is handled by the desktop environment
  instead — the server takes the request and ignores it. The screen
  stays on. This is the most common one.
* **The session is Wayland.** There is no X server to ask. ``xset``
  either is not installed or fails to open a display, and either way the
  screen stays on.
* **There is no graphical session at all** — over SSH, in a TTY, in a
  container. Nothing to blank.
* **The platform is macOS.** The old code checked for Linux and did
  nothing otherwise, so ``hnx prot`` on a Mac printed "Monitor will
  sleep", left the screen on, and waited for a wake word. macOS has
  ``pmset displaysleepnow``, which is exactly this feature.

``check=False`` plus ``except Exception: pass`` plus two ``DEVNULL``\\s
meant every one of those failed identically and silently. The screen
stayed on, the terminal went into raw mode, and the only evidence was a
message promising the opposite. So the rule here is: find a method that
works on *this* session, use it, and when there isn't one say which of
the four it is and what to install.

It does not lock the screen
---------------------------
The GNOME/KDE path used to blank by activating the screensaver, and an
active screensaver locks the session: the wake word then woke a lock
screen. The screen is now turned off with the compositor's own power
setting, and lock-on-idle (GNOME, Cinnamon, MATE, KDE, Xfce, the X
screensaver) is switched off for as long as protection lasts and put
back afterwards (:class:`hypernix.system.blanking.LockGuard`). If a crash
left it off, ``hnx prot restore`` puts it back. On a Mac set to lock
immediately when the display sleeps, it refuses unless ``--allow-lock``.

Usage:
  hnx prot [start] [--force] [--allow-lock]
  hnx prot restore
  hnx prot bind set <word>
  hnx prot bind reset
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from .blanking import (
    BLANKERS,
    COMMAND_TIMEOUT,
    Blanker,
    BlankResult,
    LockGuard,
    _no_method,
    choose_blanker,
    detect_session,
    dpms_enabled,
    macos_locks_on_sleep,
    set_monitor_state,
)

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

CONFIG_DIR = Path.home() / ".hypernix"
CONFIG_FILE = CONFIG_DIR / "protect.json"
#: What the lock guard changed, so `hnx prot restore` can undo it after a crash.
LOCK_STATE_FILE = CONFIG_DIR / "protect-lock-state.json"
DEFAULT_WAKE_WORD = "bon"

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {"wake_word": DEFAULT_WAKE_WORD, "health_checks": True}
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"wake_word": DEFAULT_WAKE_WORD, "health_checks": True}


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def set_wake_word(word: str) -> None:
    cfg = load_config()
    cfg["wake_word"] = word
    save_config(cfg)
    print(f"[protect] Wake word set to '{word}'.")


def reset_wake_word() -> None:
    cfg = load_config()
    cfg["wake_word"] = DEFAULT_WAKE_WORD
    save_config(cfg)
    print(f"[protect] Wake word reset to default '{DEFAULT_WAKE_WORD}'.")


# ---------------------------------------------------------------------------
# Turning the screen off
# ---------------------------------------------------------------------------
#
# Imported from :mod:`hypernix.system.blanking`, not written here.
# `outage` needs the same answers and used to carry its own slightly
# different copy, which is how both of them ended up missing the DPMS
# enable that this whole fix turns on.

__all__ = [
    "BLANKERS",
    "COMMAND_TIMEOUT",
    "BlankResult",
    "Blanker",
    "choose_blanker",
    "cli_main",
    "detect_session",
    "dpms_enabled",
    "load_config",
    "reset_wake_word",
    "save_config",
    "set_monitor_state",
    "set_wake_word",
    "start_protection",
]


def _health_monitor_thread(stop_event: threading.Event) -> None:
    """Optional periodic health check logging."""
    try:
        import psutil
    except ImportError:
        return

    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        # Log to file to avoid breaking raw mode terminal
        log_file = CONFIG_DIR / "protect_health.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - CPU: {cpu}% MEM: {mem}%\n")
        except Exception:
            pass
        
        # Check every 60 seconds
        for _ in range(60):
            if stop_event.is_set():
                break
            time.sleep(1)


def restore_locking() -> int:
    """Put back lock settings a crashed ``prot`` left switched off."""
    guard = LockGuard.from_state(str(LOCK_STATE_FILE))
    if not guard.changed:
        print("[protect] Nothing to restore: no lock setting was left changed.")
        return 0
    for what in guard.restore():
        print(f"[protect] Restored {what}.")
    return 0


def start_protection(*, force: bool = False, allow_lock: bool = False) -> int:
    """Blank the screen and wait for the wake word. Returns an exit code.

    Refuses rather than pretending when the screen cannot be blanked.
    Going into raw mode with the display still on is the worst of both:
    the machine looks unlocked and the keyboard looks broken, and the
    old code did exactly that on every Wayland session, every Mac and
    every desktop with DPMS disabled. *force* is for somebody who wants
    the wake-word prompt anyway and knows the screen will stay on.
    """
    if not HAS_TERMIOS:
        print("Error: protect module requires a POSIX terminal (termios/tty).")
        return 1

    cfg = load_config()
    wake_word = cfg.get("wake_word", DEFAULT_WAKE_WORD)
    do_health = cfg.get("health_checks", True)

    session = detect_session()
    blanker = choose_blanker(session)
    if blanker is None and not force:
        refusal = _no_method(session)
        print(f"[protect] {refusal.reason}")
        if refusal.hint:
            print(f"[protect] {refusal.hint}")
        print("[protect] Not entering raw mode: the screen would stay on and "
              "the keyboard would look dead. Pass --force to do it anyway.")
        return 1

    if session == "darwin" and macos_locks_on_sleep() and not allow_lock:
        print("[protect] This Mac locks as soon as the display sleeps (System Settings > "
              "Lock Screen), so blanking it would lock it. Change that setting, or pass "
              "--allow-lock to blank it anyway.")
        return 1

    guard = LockGuard(state_file=str(LOCK_STATE_FILE))
    held = guard.disable()
    if held:
        print(f"[protect] Holding the session unlocked ({', '.join(held)}); put back on exit.")

    print("[protect] Entering protection mode.")
    print(f"[protect] Type '{wake_word}' to wake up.")
    time.sleep(1.5)

    # Start health monitor
    stop_event = threading.Event()
    health_thread = None
    if do_health:
        health_thread = threading.Thread(target=_health_monitor_thread, args=(stop_event,), daemon=True)
        health_thread.start()

    # Read before enabling DPMS, so the "on" call can put a session that
    # had it deliberately disabled back the way it was.
    was_enabled = dpms_enabled()
    blanked = set_monitor_state("off", blanker=blanker)
    if not blanked.ok:
        stop_event.set()
        if health_thread:
            health_thread.join(timeout=1.0)
        print(f"[protect] {blanked.reason}")
        if blanked.hint:
            print(f"[protect] {blanked.hint}")
        if not force:
            guard.restore()
            print("[protect] Not entering raw mode. Pass --force to do it anyway.")
            return 1

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # A closed terminal (SIGHUP) or a kill (SIGTERM) must still run the
    # finally below: it is what turns the screen on and puts the lock
    # settings back.
    import signal

    def _stop(signum, _frame):  # noqa: ANN001 - signal handler signature
        raise SystemExit(128 + signum)

    previous = {}
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            previous[sig] = signal.signal(sig, _stop)
    
    buffer = ""
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            char = sys.stdin.read(1)
            # Ctrl+C is ASCII 3
            if char == '\x03':
                break
            
            buffer += char
            # Keep buffer size manageable
            if len(buffer) > len(wake_word) * 2:
                buffer = buffer[-len(wake_word)*2:]
                
            if buffer.endswith(wake_word):
                break
    finally:
        # The terminal comes back first. If turning the screen on fails,
        # somebody needs to be able to type -- and the message about it
        # needs somewhere to land.
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        woken = set_monitor_state("on", blanker=blanker, restore_dpms=was_enabled)
        guard.restore()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        stop_event.set()
        if health_thread:
            health_thread.join(timeout=1.0)

    print("\n[protect] Waking up. Protection mode ended.")
    if not woken.ok:
        # Said out loud rather than swallowed: a screen that did not come
        # back looks like a crashed machine, and the remedy is one line.
        print(f"[protect] The monitor did not come back on: {woken.reason}")
        print("[protect] Move the mouse, or run: xset dpms force on")
        return 1
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return 0
    force = "--force" in args
    allow_lock = "--allow-lock" in args
    args = [a for a in args if a not in ("--force", "--allow-lock")]

    if not args or args[0] == "start":
        return start_protection(force=force, allow_lock=allow_lock)
    if args[0] == "restore":
        return restore_locking()
        
    if args[0] == "bind":
        if len(args) < 2:
            print("Usage: hnx prot bind [set <word> | reset]")
            return 1
        
        if args[1] == "reset":
            reset_wake_word()
            return 0
            
        if args[1] == "set":
            if len(args) < 3:
                print("Error: Missing wake word.")
                return 1
            set_wake_word(args[2])
            return 0

    print("Usage:\n  hnx prot [start] [--force] [--allow-lock]\n  hnx prot restore\n"
          "  hnx prot bind [set <word> | reset]")
    return 1

if __name__ == "__main__":
    cli_main()
