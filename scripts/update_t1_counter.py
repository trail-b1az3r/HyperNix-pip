"""Refresh the README's count of public T1 servers (run hourly by CI).

Reads the counter's summary — ``T1_COUNTER_SUMMARY_URL``, else the
``summary`` in ``docs/public/v1/t1-counter-endpoint.json`` — and writes:

* ``docs/public/v1/t1-counter.json``: the summary, plus when it was checked;
* ``docs/public/v1/t1-counter-badge.json``: a shields.io endpoint badge.

With no counter configured the badge says so instead of showing a number
nobody measured; a counter that cannot be reached keeps the last count
and says it is stale.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "public" / "v1"


def _summary_url() -> str:
    url = os.environ.get("T1_COUNTER_SUMMARY_URL", "").strip()
    if url:
        return url
    try:
        return str(json.loads((DATA / "t1-counter-endpoint.json").read_text(encoding="utf-8")).get("summary") or "")
    except (OSError, ValueError):
        return ""


def _fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "hypernix-counter-badge"})
    with urllib.request.urlopen(request, timeout=30) as reply:  # nosec B310 - the project's own counter
        data = json.loads(reply.read(65536))
    if not isinstance(data, dict) or not isinstance(data.get("up"), int):
        raise ValueError("the counter's answer has no integer 'up'")
    return data


def badge(up: int | None, state: str) -> dict:
    if state == "unconfigured":
        return {"schemaVersion": 1, "label": "public T1 servers", "message": "counter not hosted yet",
                "color": "lightgrey"}
    message = f"{up} up" + (" (stale)" if state == "stale" else "")
    color = "lightgrey" if state == "stale" else ("brightgreen" if up else "orange")
    return {"schemaVersion": 1, "label": "public T1 servers", "message": message, "color": color}


def main() -> int:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    previous = {}
    try:
        previous = json.loads((DATA / "t1-counter.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    url = _summary_url()
    if not url:
        record = {"up": None, "state": "unconfigured", "checked": now,
                  "note": "No counter is hosted yet; see docs/public/v1/t1-counter-endpoint.json."}
    else:
        try:
            summary = _fetch(url)
            record = {**summary, "state": "live", "checked": now}
        except Exception as exc:  # noqa: BLE001 - reported in the file, not a failed run
            record = {"up": previous.get("up"), "t1_versions": previous.get("t1_versions", {}),
                      "state": "stale", "checked": now, "error": f"{type(exc).__name__}: {exc}"[:300]}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "t1-counter.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (DATA / "t1-counter-badge.json").write_text(
        json.dumps(badge(record.get("up"), record["state"]), indent=2) + "\n", encoding="utf-8")
    print(f"public T1 servers: {record.get('up')} ({record['state']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
