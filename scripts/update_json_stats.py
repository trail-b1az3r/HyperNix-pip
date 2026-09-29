#!/usr/bin/env python3
"""
update_json_stats.py

Fetches current PyPI download stats for the `hypernix` package and writes
them to docs/public/v1/json. The running daily history is split into three windows:
  - recent (last 30 days, day-by-day)
  - three_month (31-90 days ago, collapsed into a summary)
  - older (90+ days ago, collapsed into a summary)
so the docs site can show a "recent" vs "3 months" vs "older" breakdown
instead of just dropping anything past 30 days. Also fetches Python
version and OS breakdowns server-side (pypistats.org doesn't allow direct
browser/CORS access to these, so the docs site reads them from this file
instead of calling pypistats.org itself).

Data sources:
  - https://pypistats.org/api/packages/{package}/recent       -> last_day / last_week / last_month
  - https://pypistats.org/api/packages/{package}/overall       -> full daily download history
  - https://pypistats.org/api/packages/{package}/python_minor  -> downloads by Python version
  - https://pypistats.org/api/packages/{package}/system        -> downloads by OS
  - PyPI JSON API                                               -> current version string

Environment variables (set by the workflow):
  PYPI_PACKAGE        package name on PyPI (default: hypernix)
  JSON_PATH           path to the JSON file to update (default: docs/public/v1/json)
  MANUAL_RUN_REASON   optional note when triggered via workflow_dispatch
"""

import json
import os
import sys
from datetime import UTC, datetime, timedelta

import requests

PACKAGE = os.environ.get("PYPI_PACKAGE", "hypernix")
JSON_PATH = os.environ.get("JSON_PATH", "docs/public/v1/json")
MANUAL_RUN_REASON = os.environ.get("MANUAL_RUN_REASON", "").strip()

PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE}/json"
PYPISTATS_RECENT_URL = f"https://pypistats.org/api/packages/{PACKAGE}/recent"
PYPISTATS_OVERALL_URL = f"https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=false"
PYPISTATS_PYTHON_MINOR_URL = f"https://pypistats.org/api/packages/{PACKAGE}/python_minor"
PYPISTATS_SYSTEM_URL = f"https://pypistats.org/api/packages/{PACKAGE}/system"

REQUEST_TIMEOUT = 20
OLD_THRESHOLD_DAYS = 30
THREE_MONTH_THRESHOLD_DAYS = 90


def fetch_json(url):
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "hypernix-stats-bot/1.0"})
    resp.raise_for_status()
    return resp.json()


def get_current_version():
    try:
        data = fetch_json(PYPI_JSON_URL)
        return data.get("info", {}).get("version", "unknown")
    except Exception as exc:  # noqa: BLE001 - log and degrade gracefully
        print(f"warning: could not fetch version from PyPI JSON API: {exc}", file=sys.stderr)
        return "unknown"


def get_recent_counts():
    """last_day / last_week / last_month from pypistats' /recent endpoint."""
    try:
        data = fetch_json(PYPISTATS_RECENT_URL)
        d = data.get("data", {})
        return {
            "last_day": int(d.get("last_day", 0)),
            "last_week": int(d.get("last_week", 0)),
            "last_month": int(d.get("last_month", 0)),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not fetch recent stats: {exc}", file=sys.stderr)
        return {"last_day": 0, "last_week": 0, "last_month": 0}


def get_daily_history():
    """
    Full daily download history from pypistats' /overall endpoint.
    Returns a list of {"date": "YYYY-MM-DD", "downloads": int} sorted oldest -> newest.
    """
    try:
        data = fetch_json(PYPISTATS_OVERALL_URL)
        rows = data.get("data", [])
        history = {}
        for row in rows:
            date = row.get("date")
            downloads = int(row.get("downloads", 0))
            if not date:
                continue
            # pypistats can report "with_mirrors"/"without_mirrors" as separate
            # rows for the same date; sum them so each date appears once.
            history[date] = history.get(date, 0) + downloads
        return [{"date": d, "downloads": n} for d, n in sorted(history.items())]
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not fetch daily history: {exc}", file=sys.stderr)
        return []


def get_category_breakdown(url, category_key, label_key):
    """
    Shared helper for the python_minor / system endpoints, which both return
    {"data": [{"category": <label>, "downloads": int, ...}, ...]} with one row
    per (category, date). Aggregates total downloads per category, drops the
    "null" bucket pypistats uses for unknown/unreported values, and returns a
    list sorted by downloads descending in the shape the docs site expects:
    [{label_key: <label>, "downloads": <total>}, ...]
    """
    try:
        data = fetch_json(url)
        rows = data.get("data", [])
        totals = {}
        for row in rows:
            category = row.get("category")
            if not category or category == "null":
                continue
            downloads = int(row.get("downloads", 0))
            totals[category] = totals.get(category, 0) + downloads
        breakdown = [{label_key: k, "downloads": v} for k, v in totals.items()]
        breakdown.sort(key=lambda x: x["downloads"], reverse=True)
        return breakdown
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not fetch {category_key} breakdown: {exc}", file=sys.stderr)
        return []


def get_python_version_breakdown():
    return get_category_breakdown(PYPISTATS_PYTHON_MINOR_URL, "python_minor", "version")


def get_system_breakdown():
    return get_category_breakdown(PYPISTATS_SYSTEM_URL, "system", "os")


def split_history_buckets(history, recent_days=OLD_THRESHOLD_DAYS, three_month_days=THREE_MONTH_THRESHOLD_DAYS):
    """
    Split a daily history list into three buckets:
      - recent:      entries from the last `recent_days` days (daily granularity, kept as-is)
      - three_month: entries older than `recent_days` but within `three_month_days`, collapsed into a summary
      - older:       everything before that, collapsed into a summary
    """
    if not history:
        empty = {"days": 0, "total_downloads": 0, "from": None, "to": None}
        return [], dict(empty), dict(empty)

    today = datetime.now(UTC).date()
    recent_cutoff = today - timedelta(days=recent_days)
    three_month_cutoff = today - timedelta(days=three_month_days)

    recent = []
    three_month_entries = []
    older_entries = []
    for entry in history:
        try:
            entry_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if entry_date >= recent_cutoff:
            recent.append(entry)
        elif entry_date >= three_month_cutoff:
            three_month_entries.append(entry)
        else:
            older_entries.append(entry)

    def summarize(entries):
        return {
            "days": len(entries),
            "total_downloads": sum(e["downloads"] for e in entries),
            "from": entries[0]["date"] if entries else None,
            "to": entries[-1]["date"] if entries else None,
        }

    return recent, summarize(three_month_entries), summarize(older_entries)


def load_existing(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not parse existing {path}, starting fresh: {exc}", file=sys.stderr)
    return {}


def main():
    existing = load_existing(JSON_PATH)

    version = get_current_version()
    recent_counts = get_recent_counts()
    daily_history = get_daily_history()
    recent_days, three_month_summary, older_summary = split_history_buckets(
        daily_history, OLD_THRESHOLD_DAYS, THREE_MONTH_THRESHOLD_DAYS
    )

    python_versions = get_python_version_breakdown()
    operating_systems = get_system_breakdown()

    total_downloads = sum(e["downloads"] for e in daily_history) if daily_history else existing.get(
        "total_downloads", 0
    )

    # If a fetch fails for either breakdown this run, keep whatever was
    # committed previously rather than wiping the section out for a transient
    # pypistats.org error.
    if not python_versions:
        python_versions = existing.get("python_versions", [])
    if not operating_systems:
        operating_systems = existing.get("operating_systems", [])

    output = {
        "version": version,
        "total_downloads": total_downloads,
        "last_day": recent_counts["last_day"],
        "last_week": recent_counts["last_week"],
        "last_month": recent_counts["last_month"],
        "downloads": {
            "recent": {
                "window_days": OLD_THRESHOLD_DAYS,
                "daily": recent_days,
            },
            "three_month": three_month_summary,
            "older": older_summary,
        },
        "python_versions": python_versions,
        "operating_systems": operating_systems,
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }

    if MANUAL_RUN_REASON:
        output["last_manual_run_reason"] = MANUAL_RUN_REASON

    os.makedirs(os.path.dirname(JSON_PATH) or ".", exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {JSON_PATH}:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
