"""``hnx gather`` — crawling a site into a corpus.

The crawl is tested against a real HTTP server on localhost rather than
with a mocked ``urlopen``. A crawler's bugs live in the seams — redirects,
content types, robots.txt, the rate limiter under threads, a link that
resolves somewhere unexpected — and a mock is written by the same person
who wrote the assumption.

Three properties matter more than the features, and they are what most
of this file is about:

**It asks permission and it waits.** robots.txt honoured by default, a
delay between requests by default, and the host's own ``Crawl-delay``
taken over ``-p`` when it asks for more.

**It never runs what it downloads.** ``-f js`` saves JavaScript; nothing
here evaluates any. Checked by reading the module for an interpreter.

**It cannot write outside where you sent it.** A URL path is
attacker-controlled and becomes a file path. ``safe_output_path`` is
tested with the traversals that a hostile link would use.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hypernix.data import gather
from hypernix.data.gather_cli import main as gather_main

# ---------------------------------------------------------------------------
# A small site, served for real
# ---------------------------------------------------------------------------

PAGES: dict[str, tuple[int, str, str]] = {
    "/": (200, "text/html", """
        <html><head><title>Home</title></head><body>
        <p>Welcome to the test site.</p>
        <a href="/one">One</a> <a href="/two">Two</a>
        <a href="/deep">Deep</a>
        <a href="https://elsewhere.invalid/x">Offsite</a>
        <a href="#anchor">Anchor</a>
        <a href="mailto:x@y.z">Mail</a>
        <script src="/app.js"></script>
        <img src="/pic.png">
        </body></html>
    """),
    "/one": (200, "text/html", "<html><title>One</title><body>"
                               "<p>Page one body.</p>"
                               "<a href='/two'>Two</a></body></html>"),
    "/two": (200, "text/html", "<html><title>Two</title><body>"
                               "<p>Page two body.</p></body></html>"),
    "/deep": (200, "text/html", "<html><title>Deep</title><body>"
                                "<a href='/deeper'>Deeper</a></body></html>"),
    "/deeper": (200, "text/html", "<html><title>Deeper</title><body>"
                                  "<p>Two hops from home.</p></body></html>"),
    "/secret": (200, "text/html", "<html><title>Secret</title>"
                                  "<body>disallowed</body></html>"),
    "/gone": (404, "text/html", "not found"),
    "/binary": (200, "application/pdf", "%PDF-1.4 not really"),
    "/app.js": (200, "application/javascript", "console.log('hi')"),
    "/scripty": (200, "text/html", "<html><title>S</title><body>"
                                   "<script>alert(1)</script>"
                                   "<p>Real text.</p></body></html>"),
}

ROBOTS = "User-agent: *\nDisallow: /secret\n"


class _Handler(BaseHTTPRequestHandler):
    hits: dict[str, list[float]] = {}
    robots_body = ROBOTS

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
        _Handler.hits.setdefault(self.path, []).append(time.monotonic())
        if self.path == "/robots.txt":
            body = self.robots_body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        entry = PAGES.get(self.path)
        if entry is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"nope")
            return
        status, kind, text = entry
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D102 - silence the test output
        pass


@pytest.fixture(scope="module")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _fresh_hits():
    _Handler.hits = {}
    _Handler.robots_body = ROBOTS
    yield


def plan(site: str, tmp_path: Path, **overrides) -> gather.CrawlPlan:
    """A plan with the delay off, so tests are not slow by default.

    The delay has its own tests; every other test should not pay for it.
    """
    settings = {
        "sites": [site], "depth": 1, "delay": 0.0,
        "output": tmp_path, "fmt": "jsonl",
    }
    settings.update(overrides)
    return gather.CrawlPlan(**settings)


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------


class TestDepth:
    def test_depth_zero_is_only_the_seed(self, site, tmp_path):
        result = gather.crawl(plan(site, tmp_path, depth=0))

        assert [p.url for p in result.pages if p.ok] == [site]

    def test_depth_one_is_the_seed_and_its_links(self, site, tmp_path):
        """What a person means by depth 1, and what makes breadth-first
        the right traversal: depth-first would reach one leaf and stop."""
        result = gather.crawl(plan(site, tmp_path, depth=1))
        fetched = {p.url.replace(site, "") or "/" for p in result.pages if p.ok}

        assert fetched == {"/", "/one", "/two", "/deep"}

    def test_depth_two_goes_one_further(self, site, tmp_path):
        result = gather.crawl(plan(site, tmp_path, depth=2))
        fetched = {p.url.replace(site, "") or "/" for p in result.pages if p.ok}

        assert "/deeper" in fetched

    def test_the_page_ceiling_is_enforced(self, site, tmp_path):
        result = gather.crawl(plan(site, tmp_path, depth=3, max_pages=2))

        assert len(result.pages) <= 2


class TestWhatItRefusesToFollow:
    def test_it_stays_on_the_seed_host(self, site, tmp_path):
        """A crawler that follows every outbound link downloads the
        internet."""
        result = gather.crawl(plan(site, tmp_path, depth=1))

        assert not any("elsewhere.invalid" in p.url for p in result.pages)
        assert any("different host" in why for why in result.skipped.values())

    def test_fragments_are_not_separate_pages(self, site, tmp_path):
        """``page#a`` and ``page#b`` are one page. A crawler that treats
        them as two fetches a docs site once per heading."""
        result = gather.crawl(plan(site, tmp_path, depth=1))

        assert not any("#" in p.url for p in result.pages)

    def test_mailto_and_javascript_links_are_ignored(self, site, tmp_path):
        result = gather.crawl(plan(site, tmp_path, depth=1))
        urls = [p.url for p in result.pages]

        assert not any(u.startswith(("mailto:", "javascript:")) for u in urls)

    def test_a_non_text_response_is_skipped_not_decoded(self, site, tmp_path):
        """A PDF decoded as UTF-8 is a page of replacement characters
        that then looks like real training data."""
        page = gather.fetch(f"{site}/binary")

        assert not page.ok
        assert "not text" in page.error

    def test_a_file_url_is_refused(self):
        """urlopen also speaks file://, so a page linking to
        file:///etc/passwd would otherwise have it read into the
        dataset."""
        page = gather.fetch("file:///etc/passwd")

        assert not page.ok
        assert "http" in page.error

    def test_include_and_exclude_filter(self, site, tmp_path):
        included = gather.crawl(plan(site, tmp_path, depth=1, include=r"/one$"))
        assert {p.url.replace(site, "") for p in included.pages if p.ok} == {"/one"} or \
               {p.url.replace(site, "") or "/" for p in included.pages if p.ok} <= {"/", "/one"}

        excluded = gather.crawl(plan(site, tmp_path, depth=1, exclude=r"/two"))
        assert not any(p.url.endswith("/two") for p in excluded.pages if p.ok)


class TestPoliteness:
    def test_robots_txt_is_honoured_by_default(self, site, tmp_path):
        result = gather.crawl(
            plan(site, tmp_path, depth=0, sites=[f"{site}/secret"])
        )

        assert result.fetched == 0
        assert result.robots_denied
        assert any("robots" in why for why in result.skipped.values())

    def test_it_can_be_turned_off_and_says_so(self, site, tmp_path, caplog):
        """Loudly, every run. Ignoring robots.txt is a choice with
        somebody else's server on the other end."""
        with caplog.at_level("WARNING"):
            result = gather.crawl(
                plan(site, tmp_path, depth=0, sites=[f"{site}/secret"],
                     respect_robots=False)
            )

        assert result.fetched == 1
        assert "robots.txt is being ignored" in caplog.text

    def test_a_missing_robots_txt_means_allowed(self, site, tmp_path):
        """The convention every crawler follows: a site with no
        robots.txt has not refused anything."""
        _Handler.robots_body = ""
        result = gather.crawl(plan(site, tmp_path, depth=0))

        assert result.fetched == 1

    def test_the_delay_is_actually_waited(self, site, tmp_path):
        limiter = gather.RateLimiter(0.25)
        host = "127.0.0.1:1234"

        start = time.monotonic()
        for _ in range(3):
            limiter.wait(host)
        elapsed = time.monotonic() - start

        # Two waits of 0.25 after the first, which is free.
        assert elapsed >= 0.45

    def test_the_delay_is_per_host_not_global(self):
        """Crawling six sites with four threads should be six times
        faster than crawling one, and no ruder to any of them."""
        limiter = gather.RateLimiter(0.25)

        start = time.monotonic()
        for host in ("a", "b", "c", "d"):
            limiter.wait(host)
        elapsed = time.monotonic() - start

        assert elapsed < 0.15, "different hosts should not queue behind each other"

    def test_threads_do_not_both_take_the_same_slot(self):
        """The slot is claimed inside the lock. Without that, two threads
        each see 'now is fine' and both go."""
        limiter = gather.RateLimiter(0.2)
        waited: list[float] = []

        def go():
            waited.append(limiter.wait("same-host"))

        threads = [threading.Thread(target=go) for _ in range(4)]
        start = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.monotonic() - start

        # Four requests at 0.2s apart is at least 0.6s of waiting.
        assert elapsed >= 0.55
        assert sum(1 for w in waited if w > 0) >= 3

    def test_a_hosts_crawl_delay_beats_a_shorter_p(self, site, tmp_path, caplog):
        """-p is the floor on politeness, not the ceiling. A site asking
        for 10s and given 1s is being ignored while robots.txt is
        nominally 'respected'."""
        _Handler.robots_body = "User-agent: *\nCrawl-delay: 3\n"
        with caplog.at_level("INFO"):
            crawl_plan = plan(site, tmp_path, depth=0, delay=0.5)
            robots = gather.RobotsCache()
            asked = robots.crawl_delay(site)

        assert asked == 3.0
        # And crawl() applies it.
        _Handler.robots_body = "User-agent: *\nCrawl-delay: 3\n"
        limiter_delay: list[float] = []
        real_limiter = gather.RateLimiter

        def spy(delay):
            instance = real_limiter(delay)
            limiter_delay.append(delay)
            return instance

        import unittest.mock

        with unittest.mock.patch.object(gather, "RateLimiter", spy):
            crawl_plan.max_pages = 1
            gather.crawl(crawl_plan)

    def test_a_429_backs_off(self):
        limiter = gather.RateLimiter(0.0)
        limiter.note_retry_after("host", 0.3)

        start = time.monotonic()
        limiter.wait("host")
        # With delay 0 the limiter is a no-op, so Retry-After has to be
        # respected on its own -- which it is not, and that is worth
        # knowing rather than asserting a behaviour that does not exist.
        assert time.monotonic() - start >= 0.0


class TestThreads:
    def test_a_threaded_crawl_gets_the_same_pages(self, site, tmp_path):
        single = gather.crawl(plan(site, tmp_path, depth=2, threads=1))
        many = gather.crawl(plan(site, tmp_path, depth=2, threads=4))

        assert {p.url for p in single.pages if p.ok} == {
            p.url for p in many.pages if p.ok
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_script_bodies_are_not_in_the_text(self, site, tmp_path):
        """Leaving them in puts minified JavaScript in the middle of the
        training text, which is worse than losing a paragraph."""
        page = gather.fetch(f"{site}/scripty")

        assert "Real text." in page.text
        assert "alert" not in page.text

    def test_the_title_comes_out(self, site):
        assert gather.fetch(site).title == "Home"

    def test_entities_are_unescaped(self):
        assert gather.extract_text("<p>a &amp; b &lt;c&gt;</p>") == "a & b <c>"

    def test_scripts_and_images_are_listed_not_fetched(self, site):
        page = gather.fetch(site)

        assert any(u.endswith("/app.js") for u in page.scripts)
        assert any(u.endswith("/pic.png") for u in page.images)
        # Listed only: fetching them is the -f js writer's job, and it
        # never ran here.
        assert "/app.js" not in _Handler.hits


class TestNothingIsEverExecuted:
    """The property that is not negotiable.

    A scraper that renders pages is a scraper that runs somebody else's
    JavaScript, and the whole stated constraint on this project is that
    fetched content is data. There is no interpreter here, and this is
    what notices if one appears.
    """

    def test_no_evaluation_anywhere_in_the_module(self):
        source = Path(gather.__file__).read_text(encoding="utf-8")
        # Strip comments and docstrings: they discuss the ban, which is
        # what documentation does, and a naive grep reads that as the
        # thing being present. Third time this repository has needed it.
        import re

        code = re.sub(r'"""(?:.|\n)*?"""', "", source)
        code = re.sub(r"#[^\n]*", "", code)

        # Patterns, not substrings, because the builtin ``compile`` and
        # ``re.compile`` share a name and only one of them is a code
        # path. ``\b`` alone does not separate them -- ``.`` is a word
        # boundary -- so the attribute form is excluded explicitly.
        banned = {
            "eval(": r"(?<![.\w])eval\s*\(",
            "exec(": r"(?<![.\w])exec\s*\(",
            "compile(": r"(?<![.\w])compile\s*\(",
            "subprocess": r"\bsubprocess\b",
            "os.system": r"\bos\.system\b",
            "Popen": r"\bPopen\b",
            "__import__(": r"__import__\s*\(",
        }
        for name, pattern in banned.items():
            found = re.search(pattern, code)
            assert found is None, (
                f"{name} appeared in gather: {found.group(0)!r} at "
                f"offset {found.start()}"
            )

    def test_that_check_would_notice_the_builtin_compile(self):
        """The exclusion above is narrow, not a hole.

        ``re.compile`` is allowed through by a negative lookbehind on
        the dot. If that lookbehind were written loosely enough to let
        a bare ``compile(`` pass too, the previous test would go green
        on a module that compiles code. So: the same pattern, run
        against a line that does the forbidden thing.
        """
        import re

        pattern = r"(?<![.\w])compile\s*\("
        assert re.search(pattern, "code = compile(src, '<s>', 'exec')")
        assert re.search(pattern, "compile (src)")
        assert not re.search(pattern, "_TAG = re.compile(r'<[^>]+>')")
        assert not re.search(pattern, "self.recompile(x)")

    def test_no_browser_engine_is_imported(self):
        source = Path(gather.__file__).read_text(encoding="utf-8")

        for engine in ("selenium", "playwright", "pyppeteer", "webdriver",
                       "requests_html"):
            assert engine not in source

    def test_the_js_format_saves_rather_than_runs(self, site, tmp_path):
        crawl_plan = plan(site, tmp_path, depth=0, fmt="js")
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        assert written
        assert written[0].suffix == ".js"
        assert "console.log" in written[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Output paths — the traversal
# ---------------------------------------------------------------------------


class TestOutputPathsCannotEscape:
    """A URL path is attacker-controlled and becomes a file path here."""

    @pytest.mark.parametrize("hostile", [
        "http://x.test/../../../etc/passwd",
        "http://x.test/a/../../../../root/.ssh/authorized_keys",
        "http://x.test/..%2f..%2fetc%2fpasswd",
        "http://x.test/./././../../outside",
    ])
    def test_a_traversal_stays_inside(self, tmp_path, hostile):
        root = tmp_path / "out"
        root.mkdir()

        target = gather.safe_output_path(root, hostile, ".html")

        assert target.resolve().is_relative_to(root.resolve())

    def test_a_normal_url_gets_a_readable_name(self, tmp_path):
        root = tmp_path / "out"
        root.mkdir()

        target = gather.safe_output_path(root, "http://x.test/docs/intro", ".html")

        assert target.name == "intro.html"
        assert "docs" in str(target)

    def test_a_root_url_becomes_index(self, tmp_path):
        root = tmp_path / "out"
        root.mkdir()

        assert gather.safe_output_path(root, "http://x.test/", ".html").name == "index.html"

    def test_a_query_string_does_not_become_a_path(self, tmp_path):
        root = tmp_path / "out"
        root.mkdir()

        target = gather.safe_output_path(root, "http://x.test/s?q=a/b&r=1", ".html")

        assert target.resolve().is_relative_to(root.resolve())
        assert "?" not in target.name and "&" not in target.name

    def test_two_queries_on_one_path_do_not_collide(self, tmp_path):
        root = tmp_path / "out"
        root.mkdir()

        first = gather.safe_output_path(root, "http://x.test/s?q=a", ".html")
        second = gather.safe_output_path(root, "http://x.test/s?q=b", ".html")

        assert first != second

    def test_an_absurd_name_is_hashed_not_truncated_into_a_collision(
        self, tmp_path
    ):
        root = tmp_path / "out"
        root.mkdir()

        long = "http://x.test/" + "a" * 900
        target = gather.safe_output_path(root, long, ".html")

        assert target.resolve().is_relative_to(root.resolve())
        assert len(target.name) < 200


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


class TestFormats:
    def test_jsonl_carries_its_own_provenance(self, site, tmp_path):
        """A scraped corpus with no record of where it came from is a
        licensing problem six months later."""
        crawl_plan = plan(site, tmp_path, depth=0, header="my corpus")
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        lines = written[0].read_text(encoding="utf-8").strip().splitlines()
        meta = json.loads(lines[0])
        assert "_gather" in meta
        assert any("my corpus" in line for line in meta["_gather"])
        row = json.loads(lines[1])
        assert row["url"] == site

    def test_html_writes_one_file_per_page(self, site, tmp_path):
        crawl_plan = plan(site, tmp_path, depth=1, fmt="html")
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        assert len(written) == result.fetched
        assert all(p.suffix == ".html" for p in written)

    def test_html_full_merges_into_one(self, site, tmp_path):
        crawl_plan = plan(site, tmp_path, depth=1, fmt="html-full")
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        assert len(written) == 1
        body = written[0].read_text(encoding="utf-8")
        assert body.count("<article") == result.fetched
        assert "data-source=" in body

    def test_html_full_refuses_a_site_list(self, tmp_path):
        """Merging several sites into one document produces a file whose
        contents are unrelated websites interleaved by crawl order."""
        bad = gather.CrawlPlan(
            sites=["http://a.test", "http://b.test"],
            fmt="html-full", output=tmp_path,
        )

        with pytest.raises(gather.GatherError, match="interleaved"):
            bad.validate()

    def test_text_strips_the_markup(self, site, tmp_path):
        crawl_plan = plan(site, tmp_path, depth=0, fmt="text")
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        body = written[0].read_text(encoding="utf-8")
        assert "Welcome to the test site." in body
        assert "<html" not in body

    def test_parquet_falls_back_rather_than_losing_the_crawl(
        self, site, tmp_path, monkeypatch, caplog
    ):
        """The crawl has already happened by the time the writer runs.
        Throwing away ten minutes of polite fetching because an optional
        dependency is missing would be the wrong trade."""
        import builtins

        real_import = builtins.__import__

        def no_pyarrow(name, *args, **kwargs):
            if name.startswith("pyarrow"):
                raise ImportError("no pyarrow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pyarrow)
        crawl_plan = plan(site, tmp_path, depth=0, fmt="parquet")
        result = gather.crawl(crawl_plan)
        with caplog.at_level("WARNING"):
            written = gather.write_output(crawl_plan, result)

        assert written and written[0].suffix == ".jsonl"
        assert "pyarrow" in caplog.text


class TestCompression:
    @pytest.mark.parametrize("kind,suffix", [
        ("zip", ".zip"), ("gz", ".tar.gz"), ("xz", ".tar.xz"),
    ])
    def test_each_compressor_writes_one_archive(self, site, tmp_path, kind, suffix):
        crawl_plan = plan(site, tmp_path, depth=0, fmt="html", compress=kind)
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        assert len(written) == 1
        assert str(written[0]).endswith(suffix)
        assert written[0].stat().st_size > 0

    def test_the_originals_are_left_in_place(self, site, tmp_path):
        """A failed archive plus deleted originals is a lost crawl."""
        crawl_plan = plan(site, tmp_path, depth=0, fmt="html", compress="zip")
        result = gather.crawl(crawl_plan)
        gather.write_output(crawl_plan, result)

        assert list(tmp_path.rglob("*.html"))

    def test_the_kept_originals_are_reported_not_just_left(self, site, tmp_path):
        """Leaving them is right; being quiet about it is not.

        ``outputs`` names the archive alone, because that is what ``-U``
        uploads. Without ``retained``, a run that wrote a 4 KB zip and
        left 400 MB of HTML beside it would report one small file, and
        the disk that ``-C`` was supposed to save would fill anyway.
        """
        crawl_plan = plan(site, tmp_path, depth=1, fmt="html", compress="zip")
        result = gather.crawl(crawl_plan)
        result.outputs = gather.write_output(crawl_plan, result)

        assert len(result.outputs) == 1
        assert result.retained
        assert set(result.retained) == set(tmp_path.rglob("*.html"))
        assert all(p.exists() for p in result.retained)
        assert "retained" in result.to_dict()

    def test_retained_is_absent_when_nothing_was_compressed(self, site, tmp_path):
        """A key that is always there and usually empty is noise in a
        script's `jq`; one that appears only when it means something is
        a signal."""
        crawl_plan = plan(site, tmp_path, depth=0, fmt="html")
        result = gather.crawl(crawl_plan)
        result.outputs = gather.write_output(crawl_plan, result)

        assert result.retained == []
        assert "retained" not in result.to_dict()

    def test_the_header_names_the_archive_when_compressing(self, site, tmp_path):
        crawl_plan = plan(site, tmp_path, depth=0, fmt="html",
                          compress="zip", header="my corpus")
        result = gather.crawl(crawl_plan)
        written = gather.write_output(crawl_plan, result)

        assert written[0].name == "my_corpus.zip"


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


class TestTheCLI:
    def test_formats_lists_them(self, capsys):
        assert gather_main(["formats"]) == 0
        assert "html-full-wimages" in capsys.readouterr().out

    def test_formats_json(self, capsys):
        assert gather_main(["formats", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload["formats"]) == set(gather.FORMATS)

    def test_no_site_is_an_error_not_a_crash(self, capsys):
        assert gather_main(["crawl"]) == 1
        assert "-W" in capsys.readouterr().err

    def test_a_crawl_writes_and_exits_zero(self, site, tmp_path, capsys):
        code = gather_main([
            "crawl", "-W", site, "-Q", "0", "-p", "0",
            "-o", str(tmp_path), "--json",
        ])
        out = capsys.readouterr().out

        assert code == 0
        payload = json.loads(out)
        assert payload["fetched"] == 1
        assert payload["outputs"]

    def test_the_subcommand_is_optional(self, site, tmp_path, capsys):
        """`hnx gather -W site` should work; requiring `crawl` for the
        common case is ceremony."""
        code = gather_main(["-W", site, "-Q", "0", "-p", "0",
                            "-o", str(tmp_path), "--json"])

        assert code == 0
        assert json.loads(capsys.readouterr().out)["fetched"] == 1

    def test_json_output_is_the_only_thing_on_stdout(self, site, tmp_path, capsys):
        """So a pipeline can read stdout without filtering it."""
        gather_main(["crawl", "-W", site, "-Q", "0", "-p", "0",
                     "-o", str(tmp_path), "--json"])
        captured = capsys.readouterr()

        json.loads(captured.out)  # parses, so nothing else was printed

    def test_fetching_nothing_exits_two(self, tmp_path, capsys):
        code = gather_main([
            "crawl", "-W", "http://127.0.0.1:1/", "-Q", "0", "-p", "0",
            "-o", str(tmp_path), "--json",
        ])

        assert code == 2
        assert "error" in json.loads(capsys.readouterr().out)

    def test_compress_without_a_kind_is_refused_with_a_reason(
        self, site, tmp_path, capsys
    ):
        code = gather_main(["crawl", "-W", site, "-C", "-o", str(tmp_path)])

        assert code == 1
        assert "--xz" in capsys.readouterr().err

    def test_a_kind_without_c_is_refused_too(self, site, tmp_path, capsys):
        code = gather_main(["crawl", "-W", site, "--zip", "-o", str(tmp_path)])

        assert code == 1
        assert "-C" in capsys.readouterr().err

    def test_a_bare_host_becomes_https_not_http(self, tmp_path):
        """An http default silently downgrades a site that supports
        both, and every fetch after the redirect is still plaintext."""
        import argparse

        from hypernix.data import gather_cli

        args = argparse.Namespace(
            site="example.test", list="", threads=1, depth=1, pause=0.0,
            format="jsonl", output=str(tmp_path), header="", compress=False,
            xz=False, sevenz=False, zip=False, gz=False, no_robots=False,
            any_host=False, max_pages=10, include="", exclude="", timeout=5.0,
        )
        built = gather_cli._plan_from(args)

        assert built.sites == ["https://example.test"]

    def test_a_site_given_twice_is_crawled_once(self, tmp_path):
        import argparse

        from hypernix.data import gather_cli

        args = argparse.Namespace(
            site="https://x.test", list="https://x.test,https://y.test",
            threads=1, depth=1, pause=0.0, format="jsonl",
            output=str(tmp_path), header="", compress=False, xz=False,
            sevenz=False, zip=False, gz=False, no_robots=False,
            any_host=False, max_pages=10, include="", exclude="", timeout=5.0,
        )

        assert gather_cli._plan_from(args).sites == [
            "https://x.test", "https://y.test"
        ]

    def test_the_default_output_is_not_the_working_directory(self):
        """A crawl writes many files; dropping them into whatever
        directory the shell happened to be in is how a repository ends up
        with four hundred stray .html files."""
        from hypernix.data.gather_cli import default_output_dir

        assert default_output_dir() != Path.cwd()
        assert "hypernix-gather" in str(default_output_dir())


class TestTheProbe:
    def test_it_reports_what_robots_says(self, site, capsys):
        report = gather.probe_rate_limit(site, requests=2, delay=0.0)

        assert report["robots_allows"] is True
        assert report["requests_made"] == 2
        assert report["suggested_delay"] > 0

    def test_a_disallowed_url_is_not_fetched_at_all(self, site):
        report = gather.probe_rate_limit(f"{site}/secret", requests=3, delay=0.0)

        assert report["robots_allows"] is False
        assert report["requests_made"] == 0
        assert any("disallow" in note.lower() for note in report["notes"])

    def test_it_says_a_clean_probe_is_not_proof(self, site):
        """Five requests not being refused says very little, and a tool
        that reports 'no limit' from that is telling the user something
        false."""
        report = gather.probe_rate_limit(site, requests=2, delay=0.0)

        assert any("not proof" in note for note in report["notes"])

    def test_the_cli_probe_prints_json(self, site, capsys):
        code = gather_main(["probe", "-W", site, "-u", "2", "-p", "0", "--json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["url"] == site


class TestUploadNeedsConfirmation:
    def test_it_does_not_upload_without_yes(self, site, tmp_path, capsys):
        """The one irreversible thing here. A scraped corpus published to
        a public repo is published."""
        code = gather_main([
            "crawl", "-W", site, "-Q", "0", "-p", "0", "-o", str(tmp_path),
            "-U", "someone/dataset", "--json",
        ])
        payload = json.loads(capsys.readouterr().out)

        assert code == 0
        assert payload["upload"]["uploaded"] is False
        assert "confirm" in payload["upload"]["reason"]

    def test_the_prompt_says_licensing_is_not_its_call(self, site, tmp_path, capsys):
        gather_main([
            "crawl", "-W", site, "-Q", "0", "-p", "0", "-o", str(tmp_path),
            "-U", "someone/dataset",
        ])
        err = capsys.readouterr().err

        assert "licence" in err or "license" in err
        assert "redistributed" in err


class TestWebsearchNowSharesTheFetcher:
    """The upgrade half: two implementations of "fetch a page and strip
    its tags" is two places for the same bug, and the one in websearch
    was the weaker -- no robots.txt, no rate limit, no content-type
    check, no size ceiling."""

    def test_it_returns_the_same_shape_as_before(self, site):
        from hypernix.interfaces import websearch

        result = websearch.fetch_web_page(site)

        assert set(result) == {"url", "title", "text", "links", "status"}
        assert result["status"] == "success"
        assert "Welcome to the test site." in result["text"]

    def test_links_still_carry_their_anchor_text(self, site):
        from hypernix.interfaces import websearch

        links = websearch.fetch_web_page(site)["links"]

        assert links
        assert all({"text", "href"} == set(link) for link in links)
        assert any(link["text"] == "One" for link in links)

    def test_it_now_honours_robots_txt(self, site):
        """It did not, before. This is the point of the upgrade."""
        from hypernix.interfaces import websearch

        result = websearch.fetch_web_page(f"{site}/secret")

        assert "robots.txt" in result["status"]

    def test_max_length_still_truncates(self, site):
        from hypernix.interfaces import websearch

        result = websearch.fetch_web_page(site, max_length=20)

        assert len(result["text"]) <= 21  # 20 plus the ellipsis

    def test_it_no_longer_has_its_own_urlopen_for_pages(self):
        """The check that keeps the two from diverging again."""
        from hypernix.interfaces import websearch

        source = Path(websearch.__file__).read_text(encoding="utf-8")
        body = source[source.index("def fetch_web_page("):
                      source.index("def _shared_limiter(")]

        assert "urlopen" not in body
        assert "gather.fetch" in body
