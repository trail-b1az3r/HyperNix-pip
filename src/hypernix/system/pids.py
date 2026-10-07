"""Whether a process id names a running process, on every platform.

``os.kill(pid, 0)`` is the POSIX way to ask, and it is a trap on
Windows: there ``0`` is ``signal.CTRL_C_EVENT``, so the "harmless" probe
calls ``GenerateConsoleCtrlEvent`` and delivers Ctrl+C to every process
sharing the console -- the caller included. A test run on Windows ended
in a ``KeyboardInterrupt`` from exactly that. Windows asks the kernel
instead: open the process for a query and read its exit code.
"""
from __future__ import annotations

import os

#: GetExitCodeProcess's answer for a process that has not exited.
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5


def alive(pid: int) -> bool:
    """True when ``pid`` is a running process (one we may not signal counts)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # It exists but belongs to someone we may not query.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED  # type: ignore[attr-defined]
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)
