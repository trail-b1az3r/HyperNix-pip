"""``hypernix-t1 launch-script`` — run something that outlives the SSH session.

    hypernix-t1 launch-script ./train.py --name training-job --detach
    hypernix-t1 launch-script --status training-job
    hypernix-t1 launch-script --logs training-job
    hypernix-t1 launch-script --stop training-job

Closing the terminal is the normal end of a remote working session, and
it should not be the end of a training run. ``&`` does not achieve that:
a backgrounded process is still in the shell's process group and still
holds the tty, so the SIGHUP that follows a dropped connection reaches
it. :mod:`hypernix.system.launcher` puts the job in its own session
under a real supervisor instead.

Authentication
--------------
The command requires it, and takes it three ways: a key already
configured for this machine, ``-k``/``--key``, or the server's admin
credential. None of them reaches the job's command line — a key in
``argv`` is readable by every user on the box through ``ps``, which is
the one place a credential must not be.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..system.launcher import (
    JobStore,
    LaunchError,
    available_supervisor,
    launch,
    read_logs,
    refresh,
    stop,
)

__all__ = ["main", "cli_main"]

_EPILOG = """\
The job keeps running after you disconnect. It is put in its own session
under systemd (when a user bus is available) or a detached wrapper that
records the exit status, so neither closing the terminal nor dropping
the connection reaches it.

Everything about a job is on disk, so --status and --logs work from a
different SSH session, after a reboot, and whether or not the T1 server
is running.

Authentication is required. A key configured for this machine is used
automatically; -k takes one explicitly; --admin-password authenticates
with the server's admin credential. None of them is passed to the job.
"""


class AuthError(RuntimeError):
    """The caller could not be authenticated."""


def _configured_key() -> str:
    """A key this machine already has, if any.

    Environment first, then the T1 config directory's .env — which is
    what install-t1.sh writes and what "already authenticated on this
    machine" means in practice.
    """
    for name in ("T1_API_KEY", "HYPERNIX_T1_KEY", "T1_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    config = os.environ.get("T1_CONFIG_DIR", "")
    env_file = (Path(config) if config else Path.home() / ".hypernix" / "t1api") / ".env"
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in ("T1_API_KEY", "T1_BOOTSTRAP_KEY") and value.strip():
                return value.strip()
    except OSError:
        pass
    return ""


def _admin_password_matches(supplied: str) -> bool:
    import hmac

    expected = os.environ.get("T1_ADMIN_PASSWORD", "")
    if not expected:
        config = os.environ.get("T1_CONFIG_DIR", "")
        env_file = (
            Path(config) if config else Path.home() / ".hypernix" / "t1api"
        ) / ".env"
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() in ("T1_ADMIN_PASSWORD", "T1_T2_ADMIN_PASSWORD"):
                    expected = value.strip()
                    break
        except OSError:
            return False
    if not expected:
        return False
    # Constant time: this is a password check, and the timing of a
    # string comparison is a side channel that leaks its prefix.
    return hmac.compare_digest(supplied, expected)


def authenticate(args) -> str:
    """Return a description of who the caller is, or raise.

    Deliberately returns a *description* rather than the credential:
    everything downstream wants to record who launched a job, and
    nothing downstream should be handling the key.
    """
    from ..security.keymaster import validate_t1_key

    if getattr(args, "admin_password", ""):
        if _admin_password_matches(args.admin_password):
            return "admin-password"
        raise AuthError("That is not this server's admin password.")

    key = (getattr(args, "key", "") or "").strip() or _configured_key()
    if not key:
        raise AuthError(
            "No credential. This command needs one of:\n"
            "  -k 'T1_...'                      a key\n"
            "  --admin-password '...'           the server's admin password\n"
            "  T1_API_KEY in the environment    or in the T1 .env\n"
            "Run `hypernix-t1 key create` if you do not have one."
        )
    if key.startswith("T2_") or validate_t1_key(key):
        # A T2 key converts to a valid T1 key, so both are accepted;
        # the server remains the authority on scope and revocation.
        return f"key:{key[:8]}…"
    raise AuthError(
        "That key is not a valid T1 or T2 key. Check it was copied whole — "
        "they are long and a truncated one fails exactly like this."
    )


def _parse_env(pairs: list[str]) -> dict[str, str]:
    env = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            raise LaunchError(f"--env expects NAME=VALUE, got {pair!r}")
        env[name.strip()] = value
    return env


def _describe(job, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(job.to_dict(), indent=2))
        return
    print(f"{job.name}  ({job.job_id})")
    print(f"  status     : {job.status}"
          + (f" (exit {job.exit_status})" if job.exit_status is not None else ""))
    print(f"  supervisor : {job.supervisor}"
          + (f" unit {job.unit}" if job.unit else f" pid {job.pid}"))
    print(f"  command    : {' '.join(job.command)}")
    print(f"  cwd        : {job.cwd}")
    print(f"  log        : {job.log_path}")
    if job.env_keys:
        # Names only. The values are the caller's environment.
        print(f"  env        : {', '.join(job.env_keys)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypernix-t1 launch-script",
        description="Run a script so it survives an SSH disconnect.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("script", nargs="?", default="",
                        help="The script to run. Omit it when using "
                             "--status/--logs/--stop/--restart/--list.")
    # Not REMAINDER: that swallows everything after the script path,
    # so `launch-script ./train.py --name job` silently named the job
    # "train" and handed --name to the script. The documented form puts
    # the flags after the path, so the flags have to win; arguments for
    # the script go after a `--`, which is the usual convention and the
    # only unambiguous one.
    parser.add_argument("args", nargs="*", default=[],
                        help="Ignored. Put the script's own arguments after "
                             "a `--`: launch-script ./t.py --name j -- --lr 3e-4")

    auth = parser.add_argument_group("authentication")
    auth.add_argument("-k", "--key", default="",
                      help="A T1 or T2 key. Prefer the environment: a key "
                           "here is visible in `ps` to every user on the box.")
    auth.add_argument("--admin-password", default="",
                      help="Authenticate with the server's admin credential.")

    run = parser.add_argument_group("running")
    run.add_argument("--name", default="", help="A name to refer to the job by.")
    run.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                     help="Set a variable for the job. Repeatable.")
    run.add_argument("--cwd", default="", help="Working directory for the job.")
    run.add_argument("--detach", action="store_true",
                     help="Return immediately. This is the default, and the "
                          "flag is accepted so scripts can say what they mean.")
    run.add_argument("--timeout", type=int, default=0,
                     help="Seconds before the job is stopped. 0 means none.")
    run.add_argument("--log-file", default="", help="Where to write the job's output.")
    run.add_argument("--priority", type=int, default=0,
                     help="nice value; positive is lower priority.")
    run.add_argument("--gpu", default="",
                     help="GPU index or list for the job. Sets both "
                          "CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES.")
    run.add_argument("--cpu", default="", help="CPU affinity, e.g. 0-3.")

    manage = parser.add_argument_group("managing")
    manage.add_argument("--status", metavar="JOB", nargs="?", const="*", default=None,
                        help="Show a job, or all of them.")
    manage.add_argument("--logs", metavar="JOB", default="", help="Show a job's output.")
    manage.add_argument("--tail", type=int, default=200,
                        help="Lines of log to show. 0 for all.")
    manage.add_argument("--stop", metavar="JOB", default="", help="Stop a job.")
    manage.add_argument("--restart", metavar="JOB", default="",
                        help="Stop a job and start it again the same way.")
    manage.add_argument("--list", action="store_true", help="List every job.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Machine-readable output.")

    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        cut = raw.index("--")
        raw, script_args = raw[:cut], raw[cut + 1:]
    else:
        script_args = []

    args = parser.parse_args(raw)
    args.args = script_args
    store = JobStore()

    try:
        who = authenticate(args)
    except AuthError as exc:
        print(f"hypernix-t1 launch-script: {exc}", file=sys.stderr)
        return 1

    try:
        return _dispatch(args, store, who, parser)
    except LaunchError as exc:
        print(f"hypernix-t1 launch-script: {exc}", file=sys.stderr)
        return 1


def _dispatch(args, store: JobStore, who: str, parser) -> int:
    if args.list or args.status == "*":
        jobs = [refresh(j, store) for j in store.list()]
        if args.as_json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2))
        elif not jobs:
            print("No jobs. Start one with `hypernix-t1 launch-script ./script.py`.")
        else:
            for job in jobs:
                exit_note = "" if job.exit_status is None else f" exit {job.exit_status}"
                print(f"  {job.status:10} {job.name:24} {job.job_id}{exit_note}")
        return 0

    for flag in ("status", "logs", "stop", "restart"):
        needle = getattr(args, flag) or ""
        if not needle or needle == "*":
            continue
        job = store.find(needle)
        if job is None:
            print(f"hypernix-t1 launch-script: no job called {needle!r}. "
                  f"`--list` shows what there is.", file=sys.stderr)
            return 1
        if flag == "status":
            _describe(refresh(job, store), as_json=args.as_json)
            return 0
        if flag == "logs":
            print(read_logs(job, tail=args.tail))
            return 0
        if flag == "stop":
            stop(job, store)
            print(f"stopped {job.name} ({job.job_id})")
            return 0
        if flag == "restart":
            stop(job, store)
            fresh = launch(
                job.command[-1] if len(job.command) > 1 else job.command[0],
                args=[], name=job.name, cwd=job.cwd, timeout=job.timeout,
                priority=job.priority, gpu=job.gpu, cpu=job.cpu, store=store,
            )
            print(f"restarted {fresh.name} ({fresh.job_id})")
            return 0

    if not args.script:
        parser.print_help()
        return 0

    job = launch(
        args.script,
        args=list(args.args),
        name=args.name,
        cwd=args.cwd or None,
        env=_parse_env(args.env),
        timeout=args.timeout,
        log_file=args.log_file or None,
        priority=args.priority,
        gpu=args.gpu,
        cpu=args.cpu,
        store=store,
    )
    if args.as_json:
        print(json.dumps({**job.to_dict(), "launched_by": who}, indent=2))
        return 0
    print(f"started {job.name} ({job.job_id}) under {job.supervisor}")
    print(f"  logs   : hypernix-t1 launch-script --logs {job.name}")
    print(f"  status : hypernix-t1 launch-script --status {job.name}")
    print(f"  stop   : hypernix-t1 launch-script --stop {job.name}")
    print()
    print("  It keeps running after you disconnect.")
    if available_supervisor().value == "setsid":
        print("  (No systemd user bus here, so it runs detached rather than")
        print("   supervised. It survives the disconnect either way.)")
    return 0


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
