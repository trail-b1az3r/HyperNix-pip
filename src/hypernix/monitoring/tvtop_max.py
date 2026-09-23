"""tvtop-max — tvtop-pro on OpenTUI, with the run itself on screen.

    tvtop-max                      # find the busiest Python run, its script and log
    tvtop-max -s                   # phone layout: one column, 40-56 columns wide
    tvtop-max -l train.log -S train.py
    tvtop-max --hide procs,modules

tvtop-pro shows how the machine is doing. tvtop-max shows that and what
the run *is*, in ten panels, each toggled with its number key:

  1 cpu      2 mem      3 gpu      4 training  5 pressure cooker
  6 model    7 logs     8 warnings 9 modules & libraries   0 processes

The screen is an OpenTUI app on Bun, like hyped-pro, installed on first
run the same way (:func:`hypernix.interfaces.hyped_pro_otui.prepare_app`).
Everything it shows comes from :mod:`hypernix.monitoring.tvtop_max_bridge`,
which reads the machine the way tvtop-pro does and reads the script and
log with :mod:`hypernix.monitoring.run_inspect`, never importing or
running either.

tvtop-pro, the Rich version, stays installed and needs no Bun.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = ["APP_DIR", "cli_main"]

APP_DIR = Path(__file__).resolve().parent / "tvtop_max_app"

INSTALL_BUN_HINT = (
    "tvtop-max runs on Bun (OpenTUI's renderer needs it).\n"
    "  Install it:   curl -fsSL https://bun.sh/install | bash\n"
    "  or point HYPED_PRO_BUN at an existing bun binary.\n"
    "tvtop-pro shows the machine without Bun: run `tvtop-pro`."
)


def cli_main(argv: list[str] | None = None) -> int:
    from hypernix.interfaces.hyped_pro_otui import LaunchError, command_for, find_bun, prepare_app

    argv = list(sys.argv[1:] if argv is None else argv)
    debug = bool(os.environ.get("TVTOP_MAX_DEBUG")) or "--debug" in argv
    forward = [a for a in argv if a != "--debug"]

    try:
        bun = find_bun(debug=debug)
    except LaunchError as exc:
        # find_bun's own advice names hyped-pro; this program has its own.
        too_old = "or later" in str(exc)
        message = str(exc).replace("hyped-pro", "tvtop-max").replace(
            "The previous tvtop-max is still here: run `hyped-plus`.",
            "tvtop-pro shows the machine without Bun: run `tvtop-pro`.",
        ) if too_old else INSTALL_BUN_HINT
        print(message, file=sys.stderr)
        return exc.exit_code
    try:
        app = prepare_app(APP_DIR, bun=bun, debug=debug, name="tvtop-max", fallback="tvtop-pro")
    except LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    env = os.environ.copy()
    # The interpreter that has hypernix, for the bridge: the one running now.
    env["TVTOP_MAX_PYTHON"] = sys.executable
    command = command_for(bun, app, forward)
    if debug:
        print(f"[tvtop-max] exec {command}", file=sys.stderr)
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(cli_main())
