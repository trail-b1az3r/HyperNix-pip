"""hypernix.system.pids: process liveness without Ctrl+C on Windows."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from hypernix.system import pids

SRC = Path(__file__).resolve().parents[2] / "src" / "hypernix"


def test_this_process_is_alive():
    # On Windows the old probe, os.kill(pid, 0), sent CTRL_C_EVENT: this
    # test ended the whole run with a KeyboardInterrupt.
    assert pids.alive(os.getpid())


def test_a_child_is_alive_until_it_exits():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pids.alive(child.pid)
    finally:
        child.kill()
        child.wait()
    assert not pids.alive(child.pid)


def test_nonsense_pids_are_not_alive():
    assert not pids.alive(0)
    assert not pids.alive(-1)
    assert not pids.alive(4194303)  # above the default pid_max


def test_nothing_probes_with_signal_zero():
    """os.kill(pid, 0) is Ctrl+C to the console on Windows; use pids.alive."""
    probe = re.compile(r"os\.kill\([^)]*,\s*0\s*\)")
    offenders = [
        f"{path.relative_to(SRC)}:{n}"
        for path in SRC.rglob("*.py")
        if path.name != "pids.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if probe.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, offenders
