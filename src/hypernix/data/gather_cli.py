"""``hnx gather`` — the command line over :mod:`hypernix.data.gather`.

    hnx gather crawl -W https://example.com -Q 2 -T 4 -f jsonl -o ./corpus
    hnx gather crawl -L "https://a.dev,https://b.dev" -f parquet -C --xz
    hnx gather probe -W https://example.com
    hnx gather formats

Usable in a script, which was a requirement and shapes three things:

* Every subcommand takes ``--json`` and prints one machine-readable
  object on stdout, with progress and warnings on stderr. A pipeline can
  read stdout without filtering anything out of it.
* The exit status distinguishes outcomes: 0 wrote something, 1 could not
  start, 2 finished with nothing fetched, 3 finished but some pages
  failed. ``if hnx gather ...; then`` means what it looks like.
* ``main()`` takes ``argv`` and returns an int, so it can be called
  in-process without ``subprocess``.

Defaults chosen to be safe rather than fast
-------------------------------------------
``-Q 1`` (the seed and its links), ``-T 1``, ``-p 1.0``, robots.txt
respected, and same-host only. A tool whose defaults download the
internet at full speed is a tool that gets its user blocked before they
have read the help.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .gather import (
    COMPRESSORS,
    DEFAULT_DELAY,
    FORMATS,
    MAX_PAGES,
    CrawlPlan,
    GatherError,
    crawl,
    probe_rate_limit,
    write_output,
)

__all__ = ["main", "cli_main", "default_output_dir"]

_EPILOG = """\
formats (-f)
  html               one file per page, as fetched
  html-full          every page merged into one document (not with -L)
  html-full-wimages  the same, with images inlined as data URIs
  text               tags stripped, one file per page
  jsonl              one JSON object per page — the training default
  parquet            columnar, for a datasets pipeline (needs pyarrow)
  js                 the JavaScript a page references, saved as files

compression (-C, needs one of)
  --xz  --7z  --zip  --gz

exit status
  0  wrote output
  1  could not start (bad flags, nothing to crawl)
  2  finished, fetched nothing
  3  finished, wrote output, some pages failed

Nothing downloaded is ever executed. -f js *saves* JavaScript files;
there is no interpreter in this tool. A single-page app will therefore
come back nearly empty, and that is reported rather than hidden.
"""


def default_output_dir() -> Path:
    """Where output goes with no ``-o``.

    The user's home, not the working directory: a crawl writes many files
    and dropping them into whatever directory the shell happened to be in
    is how a repository ends up with four hundred stray .html files. On
    Windows this lands under the profile directory, which is the
    equivalent of the documented ``/C:`` intent -- writing to a drive
    root needs administrator rights on any modern Windows and would fail.
    """
    return Path(os.path.expanduser("~")) / "hypernix-gather"


def _plan_from(args: argparse.Namespace) -> CrawlPlan:
    sites: list[str] = []
    if args.site:
        sites.append(args.site.strip())
    if args.list:
        sites.extend(part.strip() for part in args.list.split(",") if part.strip())
    # Deduplicated while keeping order, so -W x -L x,y crawls x once.
    seen: set[str] = set()
    ordered = [s for s in sites if not (s in seen or seen.add(s))]

    # A bare host is what people type. Defaulting to https rather than
    # http: an http default silently downgrades a site that supports
    # both, and every fetch after the redirect is still plaintext.
    ordered = [
        s if "://" in s else f"https://{s}"
        for s in ordered
    ]

    compress = ""
    for candidate in COMPRESSORS:
        if getattr(args, candidate.replace("7z", "sevenz"), False):
            compress = candidate
            break
    if args.compress and not compress:
        raise GatherError(
            "-C needs one of --xz, --7z, --zip or --gz so the archive format "
            "is explicit. There is no sensible default: xz is smallest, zip "
            "opens anywhere, and picking one for you would be a surprise in "
            "a script."
        )
    if compress and not args.compress:
        raise GatherError(
            f"--{compress} was given without -C. Add -C to compress, or drop "
            f"--{compress} to leave the output as files."
        )

    return CrawlPlan(
        sites=ordered,
        depth=args.depth,
        threads=args.threads,
        delay=args.pause,
        fmt=args.format,
        output=Path(args.output).expanduser() if args.output else default_output_dir(),
        header=args.header or "",
        compress=compress,
        respect_robots=not args.no_robots,
        same_host=not args.any_host,
        max_pages=args.max_pages,
        include=args.include or "",
        exclude=args.exclude or "",
        timeout=args.timeout,
    )


def _add_crawl_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-W", "--site", default="",
                        help="One site to crawl.")
    parser.add_argument("-L", "--list", default="",
                        help="Several sites, comma separated.")
    parser.add_argument("-T", "--threads", type=int, default=1,
                        help="Concurrent fetches (default 1).")
    parser.add_argument("-Q", "--depth", type=int, default=1,
                        help="Link depth from each seed (default 1: the seed "
                             "and the pages it links to).")
    parser.add_argument("-p", "--pause", type=float, default=DEFAULT_DELAY,
                        help=f"Seconds between requests to one host "
                             f"(default {DEFAULT_DELAY}). Per host, so more "
                             f"threads is faster without being ruder.")
    parser.add_argument("-f", "--format", default="jsonl", choices=FORMATS,
                        help="Output format (default jsonl).")
    parser.add_argument("-o", "--output", default="",
                        help=f"Where to write (default {default_output_dir()}).")
    parser.add_argument("-O", "--header", default="",
                        help="A provenance line written into the output. With "
                             "-C this names the archive instead.")
    parser.add_argument("-C", "--compress", action="store_true",
                        help="Compress the output. Needs --xz, --7z, --zip "
                             "or --gz.")
    parser.add_argument("--xz", action="store_true", help="Compress with xz.")
    parser.add_argument("--7z", dest="sevenz", action="store_true",
                        help="Compress with 7z (needs py7zr).")
    parser.add_argument("--zip", action="store_true", help="Compress with zip.")
    parser.add_argument("--gz", action="store_true", help="Compress with gzip.")
    parser.add_argument("-U", "--upload", default="",
                        help="Upload the result to a GitHub or Hugging Face "
                             "repo (owner/name). Asks first.")
    parser.add_argument("--yes", action="store_true",
                        help="Answer the upload confirmation. For scripts.")
    parser.add_argument("--no-robots", action="store_true",
                        help="Ignore robots.txt. Logged loudly every run.")
    parser.add_argument("--any-host", action="store_true",
                        help="Follow links off the seed's host. Off by "
                             "default; on, a crawl can wander far.")
    parser.add_argument("--include", default="",
                        help="Only fetch URLs matching this regex.")
    parser.add_argument("--exclude", default="",
                        help="Never fetch URLs matching this regex.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES,
                        help=f"Hard ceiling on pages (default {MAX_PAGES}).")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Per-request timeout in seconds.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="One machine-readable object on stdout.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="No progress on stderr.")


def _cmd_crawl(args: argparse.Namespace) -> int:
    plan = _plan_from(args)
    plan.validate()

    if not args.quiet and not args.as_json:
        print(
            f"  {len(plan.sites)} site(s), depth {plan.depth}, "
            f"{plan.threads} thread(s), {plan.delay}s apart"
            + ("" if plan.respect_robots else "   [robots.txt IGNORED]"),
            file=sys.stderr,
        )

    seen = 0

    def progress(page) -> None:
        nonlocal seen
        seen += 1
        if args.quiet or args.as_json:
            return
        mark = "ok  " if page.ok else "fail"
        detail = "" if page.ok else f"  ({page.error})"
        print(f"  {mark} [{seen}] {page.url}{detail}", file=sys.stderr)

    result = crawl(plan, on_page=progress)

    if result.fetched == 0:
        # Nothing written, and the reason matters: robots.txt refusing is
        # a different problem from a dead host, and both look like "it
        # did not work".
        detail = "nothing was fetched."
        if result.robots_denied:
            detail = (
                f"every URL was refused by robots.txt "
                f"({len(result.robots_denied)} of them)."
            )
        elif result.pages:
            first = next((p.error for p in result.pages if p.error), "")
            detail = f"no page fetched successfully. First error: {first}"
        if args.as_json:
            print(json.dumps({**result.to_dict(), "error": detail}, indent=2))
        else:
            print(f"  {detail}", file=sys.stderr)
        return 2

    result.outputs = write_output(plan, result)

    uploaded: dict[str, object] = {}
    if args.upload:
        uploaded = _upload(args.upload, result.outputs, confirmed=args.yes)

    payload = {**result.to_dict(), "format": plan.fmt}
    if uploaded:
        payload["upload"] = uploaded

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n  fetched {result.fetched} page(s) in "
              f"{result.duration:.1f}s", file=sys.stderr)
        for path in result.outputs:
            print(f"  {path}")
        if result.retained and not args.quiet:
            # -C compresses; it does not delete. Said out loud, because
            # the point of -C is usually disk and finding the originals
            # still there later is a surprise.
            print(f"  ({len(result.retained)} uncompressed file(s) kept "
                  f"beside the archive)", file=sys.stderr)
        if result.skipped and not args.quiet:
            print(f"  skipped {len(result.skipped)}", file=sys.stderr)

    failed = len(result.pages) - result.fetched
    return 3 if failed else 0


def _upload(target: str, outputs: list[Path], *, confirmed: bool) -> dict[str, object]:
    """Push the result to GitHub or Hugging Face.

    Confirmation is required and ``--yes`` is the only way to skip it.
    An upload is the one thing in this tool that cannot be undone: a
    scraped corpus published to a public repo is published, and "it also
    uploaded it" is not something to discover afterwards. Licensing is
    the user's to judge and the prompt says so, because a crawler cannot
    tell whether a site's content may be redistributed.
    """
    if not confirmed:
        print(
            f"\n  About to upload {len(outputs)} file(s) to {target}.\n"
            f"  This publishes scraped content. Whether that content may be "
            f"redistributed is not something this tool can judge -- check the "
            f"source's licence and terms first.\n"
            f"  Re-run with --yes to upload.",
            file=sys.stderr,
        )
        return {"uploaded": False, "reason": "not confirmed"}

    if "/" not in target:
        return {"uploaded": False, "reason": f"{target!r} is not owner/name"}

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {
            "uploaded": False,
            "reason": "needs huggingface_hub (pip install huggingface_hub)",
        }

    try:
        api = HfApi()
        for path in outputs:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=target,
                repo_type="dataset",
            )
        return {"uploaded": True, "repo": target, "files": len(outputs)}
    except Exception as exc:  # noqa: BLE001 - the crawl is already on disk
        return {"uploaded": False, "reason": str(exc)}


def _cmd_probe(args: argparse.Namespace) -> int:
    sites = [args.site] if args.site else []
    if args.list:
        sites.extend(part.strip() for part in args.list.split(",") if part.strip())
    if not sites:
        print("gather probe: give -W <url> or -L <url,url>", file=sys.stderr)
        return 1

    reports = [
        probe_rate_limit(
            site if "://" in site else f"https://{site}",
            requests=args.requests, delay=args.pause, timeout=args.timeout,
        )
        for site in sites
    ]

    if args.as_json:
        print(json.dumps(reports, indent=2))
        return 0

    for report in reports:
        print(f"\n  {report['url']}")
        print(f"      robots.txt allows   {report['robots_allows']}")
        if report["robots_crawl_delay"] is not None:
            print(f"      robots.txt asks for {report['robots_crawl_delay']}s")
        print(f"      requests made       {report['requests_made']}  "
              f"{report['statuses']}")
        print(f"      rate limited        {report['limited']}")
        print(f"      suggested -p        {report['suggested_delay']}")
        for note in report["notes"]:
            print(f"      {note}")
    print()
    return 0


def _cmd_formats(args: argparse.Namespace) -> int:
    if args.as_json:
        print(json.dumps({"formats": list(FORMATS),
                          "compressors": list(COMPRESSORS)}, indent=2))
        return 0
    print(_EPILOG)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hnx gather",
        description="Crawl sites into a training corpus, politely.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command")

    crawl_parser = subs.add_parser(
        "crawl", help="Fetch pages and write a corpus.",
        epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_crawl_flags(crawl_parser)
    crawl_parser.set_defaults(handler=_cmd_crawl)

    probe_parser = subs.add_parser(
        "probe", help="Find out what a host tolerates, before a long crawl.",
    )
    probe_parser.add_argument("-W", "--site", default="")
    probe_parser.add_argument("-L", "--list", default="")
    probe_parser.add_argument("-u", "--requests", type=int, default=5,
                              help="Requests to make (default 5). Stops at "
                                   "the first refusal.")
    probe_parser.add_argument("-p", "--pause", type=float, default=0.5)
    probe_parser.add_argument("--timeout", type=float, default=15.0)
    probe_parser.add_argument("--json", dest="as_json", action="store_true")
    probe_parser.set_defaults(handler=_cmd_probe)

    formats_parser = subs.add_parser("formats", help="List formats and exit.")
    formats_parser.add_argument("--json", dest="as_json", action="store_true")
    formats_parser.set_defaults(handler=_cmd_formats)

    # `hnx gather -W site` with no subcommand means crawl. The
    # subcommand exists for probe and formats; requiring it for the
    # common case would be ceremony.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"crawl", "probe", "formats", "-h", "--help"}:
        argv = ["crawl", *argv]

    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.WARNING,
        format="  %(levelname)s  %(message)s",
        stream=sys.stderr,
    )
    try:
        return int(args.handler(args))
    except GatherError as exc:
        if getattr(args, "as_json", False):
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"gather: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ngather: stopped.", file=sys.stderr)
        return 1


def cli_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli_main()
