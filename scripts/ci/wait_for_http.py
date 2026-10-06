#!/usr/bin/env python3
"""Wait for an HTTP endpoint, and say why when it never arrives.

Every integration job here starts two background servers and then talks
to them. Whether a job waits for each one was, until this existed, a
per-job decision written out by hand — and the ubuntu jobs waited for the
fake model while the macOS ones did not. The macOS runner is slower to
start a Python process, so the probe reached the bridge first and the
build failed with

    MODEL_UNAVAILABLE ... No LM Studio server answering at 127.0.0.1:1234

which reads as a broken bridge and is actually a race. One helper, used
by every job, is what stops the two from drifting again.

On timeout it prints the server's own log before exiting, because the
interesting line is always in there and a job that fails without it costs
a second run to find out.

    python3 scripts/ci/wait_for_http.py http://127.0.0.1:1234/v1/models \
        --timeout 60 --log fake-model.log --name "the fake model"
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Local servers only, so the proxy variables a runner may set must not
#: be consulted — going through a proxy to reach 127.0.0.1 fails in a way
#: that looks exactly like the server being down.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    """One attempt. Returns (ok, reason)."""
    try:
        with _DIRECT.open(url, timeout=timeout) as response:
            code = response.getcode()
            if 200 <= code < 400:
                return True, f"HTTP {code}"
            return False, f"HTTP {code}"
    except urllib.error.HTTPError as exc:
        # A 4xx means something is listening and answering, which is the
        # question being asked. /v1/models on a server that wants auth is
        # up, and waiting longer will not change its mind.
        return True, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - any failure means "not yet"
        return False, type(exc).__name__


def wait_for(
    url: str,
    *,
    timeout: float,
    name: str,
    log: Path | None = None,
    interval: float = 1.0,
) -> int:
    deadline = time.monotonic() + timeout
    last = ""
    attempts = 0
    while time.monotonic() < deadline:
        ok, reason = probe(url)
        attempts += 1
        if ok:
            waited = timeout - (deadline - time.monotonic())
            print(f"  ✓ {name} is up after {waited:.1f}s ({reason})")
            return 0
        last = reason
        time.sleep(interval)

    print(f"  ✗ {name} never answered {url}", file=sys.stderr)
    print(f"    {attempts} attempts over {timeout:.0f}s, last: {last}", file=sys.stderr)
    if log is not None and log.exists():
        text = log.read_text(errors="replace", encoding="utf-8").strip()
        print(f"    --- {log} ---", file=sys.stderr)
        print(text or "    (empty — the process wrote nothing)", file=sys.stderr)
    elif log is not None:
        print(f"    {log} does not exist — the process never started",
              file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--name", default=None, help="What to call it in output.")
    parser.add_argument("--log", default=None, help="Print this on failure.")
    args = parser.parse_args(argv)

    return wait_for(
        args.url,
        timeout=args.timeout,
        interval=args.interval,
        name=args.name or args.url,
        log=Path(args.log) if args.log else None,
    )


if __name__ == "__main__":
    sys.exit(main())
