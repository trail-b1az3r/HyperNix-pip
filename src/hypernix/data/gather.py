"""gather — crawl a site into a dataset.

    hnx gather crawl -W https://example.com -Q 2 -T 4 -f parquet -o ./out

The job is training data: point it at documentation, a wiki, a blog, and
get files you can feed to :mod:`hypernix.training`. Everything here is
built around three properties that a scraper written in an afternoon
does not have.

**It asks permission and it waits.** ``robots.txt`` is honoured by
default and there is a delay between requests by default. Both are
overridable and neither is silent about being overridden. A crawler that
ignores robots.txt out of the box is a crawler that gets its user's IP
banned and their name in someone's abuse log, and "I did not know it did
that" is not a defence anyone accepts.

**It never runs what it downloads.** The ``js`` format *saves*
JavaScript files; nothing here evaluates one, and there is no HTML
renderer, no headless browser and no ``eval``. That is a deliberate
limitation with a real cost -- a single-page app renders to an empty
document and gather says so rather than pretending -- and it is not
negotiable. Fetched bytes are data.

**It cannot write outside where you sent it.** A URL path becomes a file
path, and a URL path is attacker-controlled: ``/../../.ssh/authorized_keys``
is a legal path segment. Every output path is resolved and required to be
under the output directory. See :func:`safe_output_path`.

Concurrency, and why the rate limit is per host
----------------------------------------------
``-T`` threads fetch in parallel and ``-p`` is the delay between
requests *to one host*. Those are not in conflict: crawling six sites
with four threads should be six times faster than crawling one, and
should not be four times ruder to any of them. So the limiter is keyed
on host, and a thread that would breach a host's delay waits rather than
picking a different host -- which would work, and would make the crawl
order depend on timing, which makes a crawl unreproducible.
"""
from __future__ import annotations

import gzip
import hashlib
import html
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GatherError",
    "Page",
    "CrawlPlan",
    "CrawlResult",
    "RateLimiter",
    "RobotsCache",
    "FORMATS",
    "COMPRESSORS",
    "crawl",
    "fetch",
    "extract_links",
    "extract_text",
    "safe_output_path",
    "probe_rate_limit",
    "write_output",
]


class GatherError(RuntimeError):
    """A crawl could not be started, or an output could not be written."""


#: What ``-f`` accepts.
FORMATS = (
    "html",              # one file per page, as fetched
    "html-full",         # every page merged into one document
    "html-full-wimages",  # the same, with images inlined as data URIs
    "text",              # tags stripped, one file per page
    "jsonl",             # one JSON object per page — the training default
    "parquet",           # columnar, for a datasets pipeline
    "js",                # the JavaScript files a page references, saved
)

#: What ``-C`` accepts, and the flag each needs.
COMPRESSORS = ("xz", "7z", "zip", "gz")

#: Per-page ceiling. A crawl that hits a 4 GB video should skip it, not
#: buy it: this is training text, and nothing useful here is bigger.
MAX_PAGE_BYTES = 8 * 1024 * 1024

#: Hard ceiling on a crawl regardless of depth. Depth 3 on a site with a
#: calendar widget is unbounded in practice, and an unbounded default is
#: how a crawl runs all night.
MAX_PAGES = 5000

#: Default politeness. 1 second is what most robots.txt asks for when it
#: asks, and it is the value that keeps a crawl off an abuse list.
DEFAULT_DELAY = 1.0

_USER_AGENT = (
    "HyperNixGather/1.0 (+https://github.com/minerofthesoal/hypernix-pip) "
    "hnx-gather"
)


@dataclass
class Page:
    """One fetched page."""

    url: str
    status: int = 0
    title: str = ""
    html: str = ""
    text: str = ""
    links: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    content_type: str = ""
    bytes: int = 0
    depth: int = 0
    fetched_at: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "status": self.status,
            "title": self.title,
            "text": self.text,
            "links": self.links,
            "content_type": self.content_type,
            "bytes": self.bytes,
            "depth": self.depth,
            "fetched_at": self.fetched_at,
            "error": self.error,
        }


@dataclass
class CrawlPlan:
    """Everything the CLI's flags resolve to.

    A dataclass rather than a dozen arguments so a caller in a script can
    build one, adjust a field and pass it -- which is the "gather needs
    to be usable in scripts" requirement, and it is much easier to keep
    honest than a function with fourteen keyword arguments.
    """

    sites: list[str] = field(default_factory=list)
    depth: int = 1
    threads: int = 1
    delay: float = DEFAULT_DELAY
    fmt: str = "jsonl"
    output: Path = field(default_factory=Path)
    header: str = ""
    compress: str = ""
    #: Off means robots.txt is ignored. Never the default, and the
    #: caller is told when it is off.
    respect_robots: bool = True
    #: Stay on the host each seed named. A crawler that follows every
    #: outbound link is a crawler that downloads the internet.
    same_host: bool = True
    max_pages: int = MAX_PAGES
    include: str = ""       # regex a URL must match
    exclude: str = ""       # regex a URL must not match
    timeout: float = 20.0

    def validate(self) -> None:
        if not self.sites:
            raise GatherError("No site given. Use -W <url> or -L <url,url,...>.")
        if self.fmt not in FORMATS:
            raise GatherError(
                f"Unknown format {self.fmt!r}. Choose from: {', '.join(FORMATS)}"
            )
        if self.compress and self.compress not in COMPRESSORS:
            raise GatherError(
                f"Unknown compressor {self.compress!r}. "
                f"Choose from: {', '.join(COMPRESSORS)}"
            )
        # -f html-full merges every page into one document, which has no
        # meaning across sites: you would get one file whose contents are
        # two unrelated websites interleaved by crawl order.
        if self.fmt.startswith("html-full") and len(self.sites) > 1:
            raise GatherError(
                "-f html-full merges every page into one document, so it "
                "cannot be combined with a list of sites -- the result would "
                "be several unrelated sites interleaved by crawl order. "
                "Crawl them one at a time, or use -f html."
            )
        if self.depth < 0:
            raise GatherError("-Q depth cannot be negative.")
        if self.threads < 1:
            raise GatherError("-T threads must be at least 1.")
        if self.delay < 0:
            raise GatherError("-p delay cannot be negative.")


@dataclass
class CrawlResult:
    pages: list[Page] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    #: Hosts whose robots.txt refused, and what it refused.
    robots_denied: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    outputs: list[Path] = field(default_factory=list)
    #: With ``-C``, the uncompressed files left beside the archive.
    #: ``outputs`` is the archive alone -- that is what gets uploaded --
    #: so without this the report would say one file was written while
    #: a hundred sat next to it on disk.
    retained: list[Path] = field(default_factory=list)

    @property
    def fetched(self) -> int:
        return sum(1 for page in self.pages if page.ok)

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched,
            "pages": len(self.pages),
            "skipped": self.skipped,
            "robots_denied": self.robots_denied,
            "duration_seconds": round(self.duration, 2),
            "outputs": [str(p) for p in self.outputs],
            **({"retained": [str(p) for p in self.retained]}
               if self.retained else {}),
        }


# ---------------------------------------------------------------------------
# Politeness
# ---------------------------------------------------------------------------


class RateLimiter:
    """One delay per host, enforced across threads.

    Per host and not global, because crawling six sites with four threads
    should be six times faster than crawling one and should not be four
    times ruder to any of them.

    A thread that would breach a host's delay *waits* rather than
    stealing another host's turn. Work-stealing would be faster and would
    make the crawl order depend on timing, and a crawl whose order
    depends on timing is a crawl that cannot be reproduced.
    """

    def __init__(self, delay: float = DEFAULT_DELAY) -> None:
        self.delay = max(0.0, float(delay))
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> float:
        """Block until *host* may be contacted. Returns seconds waited."""
        if self.delay <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            earliest = self._next.get(host, 0.0)
            # The slot is claimed inside the lock, so two threads cannot
            # both decide they may go now.
            start = max(now, earliest)
            self._next[host] = start + self.delay
        pause = start - time.monotonic()
        if pause > 0:
            time.sleep(pause)
            return pause
        return 0.0

    def note_retry_after(self, host: str, seconds: float) -> None:
        """Honour a server's own ``Retry-After``.

        A 429 with a Retry-After is a host telling you its rate limit
        exactly. Ignoring it in favour of ``-p`` is how a crawl gets an
        IP banned after being told politely.
        """
        if seconds <= 0:
            return
        with self._lock:
            self._next[host] = max(
                self._next.get(host, 0.0), time.monotonic() + seconds
            )


class RobotsCache:
    """``robots.txt`` per host, fetched once.

    A failure to fetch robots.txt is treated as *allowed*, which is the
    convention every crawler follows and the RFC's own guidance: a site
    with no robots.txt has not refused anything. A 5xx is the one
    exception worth arguing about and this follows the majority
    behaviour rather than inventing its own.
    """

    def __init__(self, user_agent: str = _USER_AGENT, timeout: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _parser(self, base: str):
        with self._lock:
            cached = self._parsers.get(base)
            if cached is not None:
                return cached
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(f"{base}/robots.txt")
        try:
            request = urllib.request.Request(  # noqa: S310 - scheme checked
                f"{base}/robots.txt", headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                parser.parse(
                    response.read(512 * 1024).decode("utf-8", errors="replace")
                    .splitlines()
                )
        except Exception:  # noqa: BLE001 - absence is permission
            logger.debug("gather: no usable robots.txt at %s", base, exc_info=True)
            parser.parse([])
        with self._lock:
            self._parsers[base] = parser
        return parser

    def allows(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        try:
            return bool(self._parser(base).can_fetch(self.user_agent, url))
        except Exception:  # noqa: BLE001
            return True

    def crawl_delay(self, url: str) -> float | None:
        """What the host asked for, if it asked."""
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        try:
            value = self._parser(base).crawl_delay(self.user_agent)
        except Exception:  # noqa: BLE001
            return None
        return float(value) if value is not None else None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _is_http(url: str) -> bool:
    """Only http and https.

    Checked before every fetch, because ``urlopen`` also speaks
    ``file://`` and ``ftp://`` -- and a page that links to
    ``file:///etc/passwd`` would otherwise have it read and written into
    the dataset.
    """
    try:
        return urllib.parse.urlsplit(url).scheme in ("http", "https")
    except ValueError:
        return False


def fetch(
    url: str,
    *,
    timeout: float = 20.0,
    max_bytes: int = MAX_PAGE_BYTES,
    limiter: RateLimiter | None = None,
    user_agent: str = _USER_AGENT,
) -> Page:
    """One page. Never raises.

    Returns a :class:`Page` whose ``error`` says what went wrong, because
    a crawl of two hundred pages should not stop at the first 404 -- and
    a caller that wants to know which failed needs the failures in the
    result rather than in a log.
    """
    page = Page(url=url, fetched_at=time.time())
    if not _is_http(url):
        page.error = "not an http(s) URL"
        return page

    host = urllib.parse.urlsplit(url).netloc
    if limiter is not None:
        limiter.wait(host)

    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
            # Asked for explicitly and decoded below: without it some
            # hosts send gzip anyway and the body is binary.
            "Accept-Encoding": "gzip, identity",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            page.status = response.status
            page.content_type = response.headers.get("Content-Type", "")
            raw = response.read(max_bytes + 1)
            if response.headers.get("Content-Encoding", "") == "gzip":
                try:
                    raw = gzip.decompress(raw)
                except (OSError, EOFError):
                    # A truncated gzip body -- which is what reading
                    # max_bytes+1 of a large one produces. Reported
                    # rather than half-decoded.
                    page.error = "gzip body was truncated or malformed"
                    return page
            if len(raw) > max_bytes:
                page.error = (
                    f"larger than {max_bytes // 1024 // 1024} MB; skipped"
                )
                page.bytes = len(raw)
                return page
            page.bytes = len(raw)
    except urllib.error.HTTPError as exc:
        page.status = exc.code
        page.error = f"HTTP {exc.code}"
        if exc.code == 429 and limiter is not None:
            retry = exc.headers.get("Retry-After", "") if exc.headers else ""
            try:
                limiter.note_retry_after(host, float(retry))
            except (TypeError, ValueError):
                # A Retry-After can be an HTTP date rather than seconds.
                # Backing off a fixed minute beats parsing dates wrong.
                limiter.note_retry_after(host, 60.0)
        return page
    except Exception as exc:  # noqa: BLE001 - one page must not stop a crawl
        page.error = str(exc) or exc.__class__.__name__
        return page

    # Only markup and text are decoded. A PDF or an image decoded as
    # UTF-8 is a page of replacement characters that then looks like real
    # training data.
    lowered = page.content_type.lower()
    if lowered and not (
        lowered.startswith("text/")
        or "html" in lowered
        or "xml" in lowered
        or "json" in lowered
        or "javascript" in lowered
    ):
        page.error = f"not text ({page.content_type})"
        return page

    page.html = raw.decode(_charset(page.content_type), errors="replace")
    page.title = extract_title(page.html)
    page.text = extract_text(page.html)
    page.links = extract_links(page.html, url)
    page.scripts = extract_assets(page.html, url, "script", "src")
    page.images = extract_assets(page.html, url, "img", "src")
    return page


def _charset(content_type: str) -> str:
    match = re.search(r"charset=([\w-]+)", content_type or "", re.IGNORECASE)
    if not match:
        return "utf-8"
    encoding = match.group(1).lower()
    # A charset the running Python does not know would raise inside
    # decode(); utf-8 with errors="replace" is the safer wrong answer.
    try:
        "".encode(encoding)
    except LookupError:
        return "utf-8"
    return encoding


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"'>]+)["']""", re.IGNORECASE)
_SCRIPT_BLOCK = re.compile(r"<script\b.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_BLOCK = re.compile(r"<style\b.*?</style\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def extract_title(markup: str) -> str:
    match = _TITLE.search(markup or "")
    return html.unescape(match.group(1)).strip() if match else ""


def extract_text(markup: str) -> str:
    """Tags stripped, whitespace collapsed.

    Regex rather than a parser, and that is a real limitation stated
    rather than hidden: it handles the markup real sites emit and it will
    mangle pathological nesting. The alternative is a dependency, and
    this module is meant to work on a training box with torch and
    nothing else.

    Script and style bodies go first. Leaving them in puts minified
    JavaScript in the middle of the training text, which is worse than
    losing a paragraph.
    """
    text = _COMMENT.sub(" ", markup or "")
    text = _SCRIPT_BLOCK.sub(" ", text)
    text = _STYLE_BLOCK.sub(" ", text)
    text = _TAG.sub(" ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_links(markup: str, base_url: str) -> list[str]:
    """Absolute http(s) links, deduplicated, fragments dropped.

    Fragments dropped because ``page#a`` and ``page#b`` are one page, and
    a crawler that treats them as two fetches a documentation site's
    table of contents once per heading.
    """
    out: list[str] = []
    seen: set[str] = set()
    for href in _HREF.findall(markup or ""):
        candidate = html.unescape(href).strip()
        if not candidate or candidate.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            continue
        try:
            absolute = urllib.parse.urljoin(base_url, candidate)
        except ValueError:
            continue
        absolute, _fragment = urllib.parse.urldefrag(absolute)
        if not _is_http(absolute) or absolute in seen:
            continue
        seen.add(absolute)
        out.append(absolute)
    return out


def extract_assets(markup: str, base_url: str, tag: str, attribute: str) -> list[str]:
    pattern = re.compile(
        rf"""<{tag}\b[^>]*?\b{attribute}\s*=\s*["']([^"'>]+)["']""", re.IGNORECASE
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in pattern.findall(markup or ""):
        candidate = html.unescape(value).strip()
        if not candidate or candidate.startswith("data:"):
            continue
        try:
            absolute = urllib.parse.urljoin(base_url, candidate)
        except ValueError:
            continue
        if _is_http(absolute) and absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def safe_output_path(root: Path, url: str, suffix: str) -> Path:
    """A file under *root* named after *url*, and never outside it.

    A URL path is attacker-controlled and becomes a file path here.
    ``https://x/../../../root/.ssh/authorized_keys`` has legal segments,
    and ``urljoin`` will happily produce one. So the path is built,
    resolved, and required to be under the root -- and when it is not,
    the name falls back to a hash rather than raising, because one hostile
    link should not end a crawl of two hundred pages.

    Long names are hashed too: most filesystems cap a component at 255
    bytes and a query string blows through that easily.
    """
    root = root.resolve()
    parts = urllib.parse.urlsplit(url)
    raw = f"{parts.netloc}{parts.path}"
    if parts.query:
        # The query is part of the identity of the page, and it is full
        # of characters that are not safe in a filename.
        raw += "_" + hashlib.sha256(parts.query.encode()).hexdigest()[:12]
    if raw.endswith("/") or not parts.path or parts.path == "/":
        raw = raw.rstrip("/") + "/index"

    segments: list[str] = []
    for segment in raw.split("/"):
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", segment).strip("._")
        # "." and ".." never survive: they are what the traversal needs.
        if not cleaned or cleaned in (".", ".."):
            continue
        segments.append(cleaned[:120])
    if not segments:
        segments = [hashlib.sha256(url.encode()).hexdigest()[:16]]

    candidate = (root / Path(*segments)).with_suffix(suffix)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        logger.warning(
            "gather: %s resolved outside the output directory; naming it by "
            "hash instead", url,
        )
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        resolved = (root / digest).with_suffix(suffix)
    return resolved


# ---------------------------------------------------------------------------
# The crawl
# ---------------------------------------------------------------------------


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc.lower()


def crawl(plan: CrawlPlan, *, on_page=None) -> CrawlResult:
    """Breadth-first, bounded, and polite.

    Breadth-first rather than depth-first because ``-Q 1`` should mean
    "the seed and everything it links to", which is what a person
    expects, and depth-first on a site with deep navigation reaches one
    leaf and calls it a crawl.

    ``on_page`` is called with each finished :class:`Page`, which is how
    a script gets progress without this module knowing what a progress
    bar is.
    """
    plan.validate()
    result = CrawlResult(started_at=time.time())

    limiter = RateLimiter(plan.delay)
    robots = RobotsCache() if plan.respect_robots else None
    if not plan.respect_robots:
        # Logged at warning, every run. Ignoring robots.txt is a choice
        # with someone else's server on the other end of it, and it
        # should never be possible to make it without noticing.
        logger.warning(
            "gather: --no-robots is set. robots.txt is being ignored for "
            "every host in this crawl."
        )

    include = re.compile(plan.include) if plan.include else None
    exclude = re.compile(plan.exclude) if plan.exclude else None
    seed_hosts = {_host(site) for site in plan.sites}

    # A host that asked for a longer delay than -p gets it. -p is the
    # floor on politeness, not the ceiling: a site asking for 10 seconds
    # and being given 1 is being ignored while robots.txt is nominally
    # "respected".
    if robots is not None:
        for site in plan.sites:
            asked = robots.crawl_delay(site)
            if asked is not None and asked > limiter.delay:
                logger.info(
                    "gather: %s asks for %.1fs between requests; using that "
                    "rather than %.1fs.", _host(site), asked, limiter.delay,
                )
                limiter.delay = asked

    queue: deque[tuple[str, int]] = deque()
    seen: set[str] = set()
    for site in plan.sites:
        normalised, _fragment = urllib.parse.urldefrag(site.strip())
        if normalised and normalised not in seen:
            seen.add(normalised)
            queue.append((normalised, 0))

    lock = threading.Lock()

    def admissible(url: str, depth: int) -> str:
        """"" if the URL should be fetched, else why not."""
        if depth > plan.depth:
            return "deeper than -Q"
        if plan.same_host and _host(url) not in seed_hosts:
            return "different host"
        if include is not None and not include.search(url):
            return "did not match --include"
        if exclude is not None and exclude.search(url):
            return "matched --exclude"
        if robots is not None and not robots.allows(url):
            return "robots.txt"
        return ""

    def work(url: str, depth: int) -> Page:
        page = fetch(
            url, timeout=plan.timeout, limiter=limiter, max_bytes=MAX_PAGE_BYTES
        )
        page.depth = depth
        return page

    # Level by level, so -Q means what it says and so each level's link
    # discovery is complete before the next starts.
    depth = 0
    while queue and len(result.pages) < plan.max_pages:
        level: list[tuple[str, int]] = []
        while queue and queue[0][1] == depth:
            level.append(queue.popleft())
        if not level:
            depth = queue[0][1] if queue else depth + 1
            continue

        batch: list[tuple[str, int]] = []
        for url, url_depth in level:
            reason = admissible(url, url_depth)
            if reason:
                result.skipped[url] = reason
                if reason == "robots.txt":
                    result.robots_denied.append(url)
                continue
            batch.append((url, url_depth))
            if len(result.pages) + len(batch) >= plan.max_pages:
                break

        if not batch:
            depth += 1
            continue

        if plan.threads > 1:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(plan.threads, len(batch))
            ) as pool:
                pages = list(pool.map(lambda item: work(*item), batch))
        else:
            pages = [work(url, url_depth) for url, url_depth in batch]

        for page in pages:
            with lock:
                result.pages.append(page)
            if on_page is not None:
                try:
                    on_page(page)
                except Exception:  # noqa: BLE001 - a callback must not stop a crawl
                    logger.debug("gather: on_page callback raised", exc_info=True)
            if not page.ok:
                continue
            for link in page.links:
                if link not in seen:
                    seen.add(link)
                    queue.append((link, page.depth + 1))

        depth += 1

    if len(result.pages) >= plan.max_pages and queue:
        logger.warning(
            "gather: stopped at the %d-page ceiling with %d URLs still "
            "queued. Raise --max-pages, or narrow the crawl with --include.",
            plan.max_pages, len(queue),
        )
    result.finished_at = time.time()
    return result


# ---------------------------------------------------------------------------
# Rate-limit probing (-u)
# ---------------------------------------------------------------------------


def probe_rate_limit(
    url: str, *, requests: int = 5, delay: float = 0.5, timeout: float = 15.0
) -> dict[str, Any]:
    """Find out what a host will tolerate, before a long crawl does.

    Deliberately gentle and deliberately short: it makes a handful of
    requests to the *one* URL given, stops the moment it sees a 429 or a
    403, and reports what it found. A "rate limit tester" that hammers a
    host to find the breaking point is an attack with a friendly name.

    Reports robots.txt's own answer too, which is the actual answer most
    of the time and costs one request to get.
    """
    report: dict[str, Any] = {
        "url": url,
        "requests_made": 0,
        "statuses": [],
        "limited": False,
        "limited_after": None,
        "retry_after": None,
        "robots_crawl_delay": None,
        "robots_allows": True,
        "suggested_delay": DEFAULT_DELAY,
        "notes": [],
    }
    if not _is_http(url):
        report["notes"].append("not an http(s) URL")
        return report

    robots = RobotsCache()
    report["robots_allows"] = robots.allows(url)
    asked = robots.crawl_delay(url)
    report["robots_crawl_delay"] = asked
    if not report["robots_allows"]:
        report["notes"].append(
            "robots.txt disallows this URL. Nothing was fetched; the crawl "
            "would skip it too."
        )
        return report

    limiter = RateLimiter(delay)
    for attempt in range(1, max(1, requests) + 1):
        page = fetch(url, timeout=timeout, limiter=limiter)
        report["requests_made"] = attempt
        report["statuses"].append(page.status or page.error)
        if page.status in (429, 403):
            report["limited"] = True
            report["limited_after"] = attempt
            report["notes"].append(
                f"HTTP {page.status} after {attempt} request(s) at {delay}s "
                f"apart. Stopped."
            )
            break

    # robots.txt wins when it asked, because it is the host's own
    # statement rather than an inference from five requests.
    if asked is not None:
        report["suggested_delay"] = asked
        report["notes"].append(f"robots.txt asks for {asked}s; use that.")
    elif report["limited"]:
        report["suggested_delay"] = max(5.0, delay * 10)
        report["notes"].append(
            "Rate limited during the probe. The suggestion is a guess -- the "
            "probe stops at the first refusal rather than searching for the "
            "threshold, so it knows the limit is tighter than what it tried "
            "and not how much tighter."
        )
    else:
        report["suggested_delay"] = max(DEFAULT_DELAY, delay)
        report["notes"].append(
            f"No limit hit in {report['requests_made']} requests at {delay}s "
            f"apart. That is not proof there is none -- it is a handful of "
            f"requests, and a limit measured per hour would not show."
        )
    return report


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

#: The extension each format's per-page files get.
_SUFFIX = {
    "html": ".html",
    "text": ".txt",
    "js": ".js",
}


def _header_block(plan: CrawlPlan, result: CrawlResult) -> str:
    """Provenance, written into every output.

    ``-O`` sets the human part; the rest is added regardless. A scraped
    corpus with no record of where it came from is a licensing problem
    six months later, and the moment to write it down is the moment the
    file is made.
    """
    lines = [
        plan.header or f"gathered by hnx gather from {', '.join(plan.sites)}",
        f"gathered_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"sites: {', '.join(plan.sites)}",
        f"depth: {plan.depth}  pages: {result.fetched}",
        f"robots_respected: {str(plan.respect_robots).lower()}",
    ]
    return "\n".join(lines)


def write_output(plan: CrawlPlan, result: CrawlResult) -> list[Path]:
    """Write *result* in ``plan.fmt``. Returns what it wrote."""
    root = Path(plan.output).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pages = [page for page in result.pages if page.ok]

    if plan.fmt == "jsonl":
        target = root / f"{_stem(plan)}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            # A metadata line first, marked so a reader can skip it. One
            # file that carries its own provenance beats a sidecar that
            # gets separated from it.
            handle.write(json.dumps({"_gather": _header_block(plan, result).split("\n")}) + "\n")
            for page in pages:
                handle.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")
        written.append(target)

    elif plan.fmt == "parquet":
        written.extend(_write_parquet(root, plan, result, pages))

    elif plan.fmt in ("html-full", "html-full-wimages"):
        written.append(_write_merged(root, plan, result, pages))

    elif plan.fmt in ("html", "text", "js"):
        written.extend(_write_per_page(root, plan, pages))

    else:  # pragma: no cover - validate() rejects anything else
        raise GatherError(f"Unhandled format {plan.fmt!r}")

    if plan.compress:
        # The originals stay on disk (see _compress). Recorded here so
        # the run's report says so, rather than naming the archive and
        # letting the caller assume it is the only thing written.
        result.retained = list(written)
        written = [_compress(written, root, plan)]
    return written


def _stem(plan: CrawlPlan) -> str:
    """The output file's name.

    ``-O`` names the file when compressing and is a header otherwise,
    which is the documented behaviour and reads oddly until you see why:
    a compressed archive has one name and no room for a header, and an
    uncompressed corpus has a header and gets its name from the site.
    """
    if plan.compress and plan.header:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", plan.header).strip("._") or "gather"
    first = plan.sites[0] if plan.sites else "gather"
    host = _host(first) or "gather"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", host).strip("._") or "gather"


def _write_per_page(root: Path, plan: CrawlPlan, pages: list[Page]) -> list[Path]:
    written: list[Path] = []
    suffix = _SUFFIX.get(plan.fmt, ".txt")

    if plan.fmt == "js":
        # The scripts a page references, saved as files. Downloaded, not
        # executed -- there is no interpreter anywhere in this module,
        # and there will not be one.
        limiter = RateLimiter(plan.delay)
        seen: set[str] = set()
        for page in pages:
            for src in page.scripts:
                if src in seen:
                    continue
                seen.add(src)
                asset = fetch(src, timeout=plan.timeout, limiter=limiter)
                if not asset.ok:
                    continue
                target = safe_output_path(root, src, ".js")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(asset.html, encoding="utf-8")
                written.append(target)
        return written

    for page in pages:
        target = safe_output_path(root, page.url, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = page.html if plan.fmt == "html" else page.text
        if plan.fmt == "text" and plan.header:
            body = f"{plan.header}\n\n{body}"
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written


def _write_merged(
    root: Path, plan: CrawlPlan, result: CrawlResult, pages: list[Page]
) -> Path:
    """Every page in one document, in crawl order."""
    target = root / f"{_stem(plan)}.html"
    inline_images = plan.fmt == "html-full-wimages"
    limiter = RateLimiter(plan.delay) if inline_images else None

    parts = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8">',
        f"<title>{html.escape(plan.header or _stem(plan))}</title>",
        f"<!--\n{html.escape(_header_block(plan, result))}\n-->",
        "</head><body>",
    ]
    for page in pages:
        parts.append(
            f'<article data-source="{html.escape(page.url, quote=True)}">'
        )
        parts.append(f"<h1>{html.escape(page.title or page.url)}</h1>")
        body = page.html
        if inline_images:
            body = _inline_images(body, page, limiter, plan.timeout)
        parts.append(body)
        parts.append("</article>\n<hr>")
    parts.append("</body></html>")
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


#: Per-image ceiling when inlining. A data URI is base64, so a 2 MB
#: image becomes 2.7 MB of document -- and a merged corpus with a hundred
#: of those is a file no editor will open.
MAX_INLINE_IMAGE_BYTES = 512 * 1024


def _inline_images(
    markup: str, page: Page, limiter: RateLimiter | None, timeout: float
) -> str:
    """Replace image sources with data URIs.

    Fetched with the same limiter as the pages, because a hundred images
    from one host is a hundred requests and the politeness setting was
    not meant to exclude them.
    """
    import base64

    replacements: dict[str, str] = {}
    for src in page.images:
        if not _is_http(src):
            continue
        try:
            request = urllib.request.Request(  # noqa: S310 - scheme checked
                src, headers={"User-Agent": _USER_AGENT}
            )
            if limiter is not None:
                limiter.wait(urllib.parse.urlsplit(src).netloc)
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                kind = response.headers.get("Content-Type", "image/png")
                raw = response.read(MAX_INLINE_IMAGE_BYTES + 1)
        except Exception:  # noqa: BLE001 - a missing image is not a failure
            continue
        if len(raw) > MAX_INLINE_IMAGE_BYTES or not kind.startswith("image/"):
            continue
        encoded = base64.b64encode(raw).decode("ascii")
        replacements[src] = f"data:{kind};base64,{encoded}"

    if not replacements:
        return markup

    def swap(match: re.Match[str]) -> str:
        raw = html.unescape(match.group(1)).strip()
        absolute = urllib.parse.urljoin(page.url, raw)
        data = replacements.get(absolute)
        return match.group(0).replace(match.group(1), data) if data else match.group(0)

    return re.sub(
        r"""<img\b[^>]*?\bsrc\s*=\s*["']([^"'>]+)["']""",
        swap, markup, flags=re.IGNORECASE,
    )


def _write_parquet(
    root: Path, plan: CrawlPlan, result: CrawlResult, pages: list[Page]
) -> list[Path]:
    """Columnar output, via pyarrow.

    Falls back to JSONL with a clear message rather than failing: the
    crawl has already happened by the time this runs, and throwing away
    ten minutes of polite fetching because an optional dependency is
    missing would be the wrong trade.
    """
    target = root / f"{_stem(plan)}.parquet"
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet as pq
        from pyarrow import Table
    except ImportError:
        logger.warning(
            "gather: -f parquet needs pyarrow (pip install pyarrow). Writing "
            "JSONL instead so the crawl is not lost; convert it later."
        )
        fallback = CrawlPlan(**{**plan.__dict__, "fmt": "jsonl", "compress": ""})
        return write_output(fallback, result)

    rows = [page.to_dict() for page in pages]
    columns = {
        key: [row.get(key) for row in rows]
        for key in ("url", "status", "title", "text", "content_type", "bytes", "depth", "fetched_at")
    }
    # Links are a list per row; parquet handles that natively and it is
    # worth keeping -- the link graph is half of why you crawled.
    columns["links"] = [row.get("links") or [] for row in rows]
    table = Table.from_pydict(columns)
    table = table.replace_schema_metadata(
        {"gather": _header_block(plan, result)}
    )
    pq.write_table(table, target)
    return [target]


def _compress(files: list[Path], root: Path, plan: CrawlPlan) -> Path:
    """One archive from *files*, in ``plan.compress``.

    The archive is written beside the files and the files are left in
    place. Deleting the input after compressing would be the obvious
    convenience and the wrong default: a failed archive plus deleted
    originals is a lost crawl.
    """
    stem = _stem(plan)

    if plan.compress == "zip":
        import zipfile

        target = root / f"{stem}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=_arcname(path, root))
        return target

    if plan.compress == "gz":
        return _tarball(files, root, f"{stem}.tar.gz", "w:gz")
    if plan.compress == "xz":
        return _tarball(files, root, f"{stem}.tar.xz", "w:xz")

    if plan.compress == "7z":
        try:
            import py7zr
        except ImportError as exc:
            raise GatherError(
                "-C --7z needs py7zr (pip install py7zr). The crawl is on "
                f"disk uncompressed at {root}; nothing was lost."
            ) from exc
        target = root / f"{stem}.7z"
        with py7zr.SevenZipFile(target, "w") as archive:
            for path in files:
                archive.write(path, arcname=_arcname(path, root))
        return target

    raise GatherError(f"Unknown compressor {plan.compress!r}")


def _arcname(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _tarball(files: list[Path], root: Path, name: str, mode: str) -> Path:
    import tarfile

    target = root / name
    with tarfile.open(target, mode) as archive:
        for path in files:
            archive.add(path, arcname=_arcname(path, root))
    return target
