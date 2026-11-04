"""The part of ``hnx sandbox`` that runs inside the new namespaces (Linux).

Started by :mod:`hypernix.interfaces.sandbox` as::

    unshare -r -m [-n] --propagation private -- python -I sandbox_launch.py PLAN

where PLAN is base64 JSON. ``unshare -r`` makes this process root in a new
user namespace, which is what lets it mount. It then:

1. keeps a handle on the directory (``O_PATH``), so it can be put back
   after something is mounted over the place it lives;
2. with ``write=dir``, bind-mounts the directory onto itself, so it is a
   mount of its own that can stay writable;
3. with ``write`` other than ``all``, remounts every other mount
   read-only, and checks that ``/`` and the home folder really are;
4. with ``tmp=on``, mounts a private tmpfs on ``/tmp``;
5. with ``home=hide``, mounts an empty tmpfs over the home folder;
6. puts the directory back where it was if 4 or 5 covered it;
7. enters one more user namespace that maps the caller's own uid and gid
   back, so the command does not run as (namespace) root;
8. changes into the directory, applies the resource limits and execs the
   command.

Standard library only, and no ``hypernix`` import: it runs with ``-I``
and has to start quickly.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import sys

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_NOATIME = 1024
MS_NODIRATIME = 2048
MS_RELATIME = 1 << 21
MS_STRICTATIME = 1 << 24
CLONE_NEWUSER = 0x10000000

_libc = ctypes.CDLL(None, use_errno=True)


class LaunchError(RuntimeError):
    pass


def _mount(source: str | None, target: str, fstype: str | None, flags: int,
           data: str | None = None) -> None:
    rc = _libc.mount(source.encode() if source else None, target.encode(),
                     fstype.encode() if fstype else None, flags,
                     data.encode() if data else None)
    if rc != 0:
        err = ctypes.get_errno()
        raise LaunchError(f"mount {target}: {os.strerror(err)}")


def _locked_flags(options: str) -> int:
    """The flags a remount inside a user namespace must keep."""
    opts = set(options.split(","))
    flags = 0
    for name, bit in (("nosuid", MS_NOSUID), ("nodev", MS_NODEV), ("noexec", MS_NOEXEC),
                      ("noatime", MS_NOATIME), ("nodiratime", MS_NODIRATIME),
                      ("relatime", MS_RELATIME), ("strictatime", MS_STRICTATIME)):
        if name in opts:
            flags |= bit
    return flags


def _mounts() -> list[tuple[str, str]]:
    out = []
    with open("/proc/self/mountinfo") as handle:
        for line in handle:
            fields = line.split()
            point = fields[4].encode().decode("unicode_escape")
            out.append((point, fields[5]))
    return out


def _read_only_everywhere(keep: str | None) -> None:
    for point, options in _mounts():
        if (keep and point == keep) or point == "/proc" or point.startswith("/proc/"):
            continue  # /proc is not files; the uid map is written through it
        try:
            _mount(None, point, None, MS_REMOUNT | MS_BIND | MS_RDONLY | _locked_flags(options))
        except LaunchError:
            pass  # /proc and friends; what matters is checked below


def _same(path: str, fd: int) -> bool:
    try:
        a, b = os.stat(path), os.fstat(fd)
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _map_back(uid: int, gid: int) -> None:
    if _libc.unshare(CLONE_NEWUSER) != 0:
        raise LaunchError(f"user namespace: {os.strerror(ctypes.get_errno())}")
    with open("/proc/self/setgroups", "w") as handle:
        handle.write("deny")
    with open("/proc/self/uid_map", "w") as handle:
        handle.write(f"{uid} 0 1")
    with open("/proc/self/gid_map", "w") as handle:
        handle.write(f"{gid} 0 1")


def _limits(plan: dict) -> None:
    import resource

    for key, which, scale in (("memory", "RLIMIT_DATA", 1024 * 1024),
                              ("cpu", "RLIMIT_CPU", 1),
                              ("filesize", "RLIMIT_FSIZE", 1024 * 1024)):
        value = int(plan.get(key) or 0)
        if value > 0 and hasattr(resource, which):
            resource.setrlimit(getattr(resource, which), (value * scale, value * scale))


def launch(plan: dict) -> None:
    directory = plan["dir"]
    write, home = plan["write"], plan.get("home_path") or ""
    if write == "dir":
        _mount(directory, directory, None, MS_BIND)
    # Opened after that bind, so the handle is on the writable mount.
    fd = os.open(directory, os.O_PATH | os.O_DIRECTORY)
    if write != "all":
        _read_only_everywhere(directory if write == "dir" else None)
        for probe in ("/", home):
            if probe and os.path.isdir(probe) and os.access(probe, os.W_OK):
                raise LaunchError(f"{probe} could not be made read-only")
    if plan.get("tmp"):
        _mount("tmpfs", "/tmp", "tmpfs", MS_NOSUID | MS_NODEV, "size=1g,mode=1777")
    if plan.get("hide_home") and home and os.path.isdir(home):
        _mount("tmpfs", home, "tmpfs", MS_NOSUID | MS_NODEV, "size=64m,mode=0755")
    if not _same(directory, fd):
        os.makedirs(directory, exist_ok=True)
        _mount(f"/proc/self/fd/{fd}", directory, None, MS_BIND)
        if write == "none":
            _mount(None, directory, None, MS_REMOUNT | MS_BIND | MS_RDONLY)
    if plan.get("hide_home") and home and write == "none" and os.path.isdir(home):
        try:
            _mount(None, home, None, MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV)
        except LaunchError:
            pass
    os.close(fd)

    _map_back(int(plan["uid"]), int(plan["gid"]))
    os.chdir(directory)
    _limits(plan)
    argv, env = plan["argv"], plan["env"]
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as exc:
        print(f"hnx sandbox: {argv[0]}: {exc.strerror}", file=sys.stderr)
        sys.exit(127)


def main() -> None:
    try:
        launch(json.loads(base64.b64decode(sys.argv[1])))
    except (LaunchError, OSError) as exc:
        print(f"hnx sandbox: could not set the sandbox up: {exc}", file=sys.stderr)
        sys.exit(126)


if __name__ == "__main__":
    main()
