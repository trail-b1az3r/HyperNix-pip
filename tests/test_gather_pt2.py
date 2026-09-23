"""gather — where it writes, what it writes, and when it stops.

The stopping half is the interesting one. `gather` looked like it never
finished, and the crawl loop was correct: it stops when the queue
empties or the page ceiling is hit. What it had no answer for is a site
where the queue *never* empties — a calendar, a session id, a faceted
search — because those generate unique URLs faster than a crawl
consumes them. So it ran to MAX_PAGES, which at the default one-second
politeness is eighty-three minutes, long after every page worth having
had been fetched.

A page ceiling is not a stopping condition anybody can feel. A clock
is. These tests build real traps and assert the clock wins.
"""
from __future__ import annotations

import http.server
import itertools
import socketserver
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hypernix.data.gather import (
    BARREN_LEVELS,
    CLOCK_CHECK_EVERY,
    FORMATS,
    MAX_SECONDS,
    CrawlPlan,
    crawl,
    write_output,
)
from hypernix.data.gather_cli import (
    _site_slug,
    default_output_dir,
    session_name,
)


def serve(handler_body) -> str:
    """A throwaway local site. Returns its base URL."""

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            body = handler_body(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/"


# ---------------------------------------------------------------------------
# Where it writes
# ---------------------------------------------------------------------------


class TestTheSessionDirectory:
    def test_it_is_under_hypernix_data_by_site(self):
        path = default_output_dir("https://docs.example.com/guide")
        assert path.parts[-3] == "data"
        assert path.parts[-2] == "docs.example.com"

    def test_two_runs_of_the_same_site_do_not_collide(self):
        """The real problem with one flat directory: the second crawl
        silently replaced the first, and keeping both meant remembering
        -o before starting."""
        first = default_output_dir("https://x.example.com", when=1_000_000)
        second = default_output_dir("https://x.example.com", when=1_000_060)
        assert first != second
        assert first.parent == second.parent

    def test_a_session_name_sorts(self):
        """Because what people do with these is `ls` and take the last."""
        names = [session_name(t) for t in (1_000_000, 2_000_000, 1_500_000)]
        assert sorted(names) == [names[0], names[2], names[1]]

    def test_no_site_still_lands_somewhere_sensible(self):
        assert default_output_dir("").parts[-2] == "site"

    @pytest.mark.parametrize(
        "site,expected",
        [
            ("https://docs.example.com/a/b", "docs.example.com"),
            ("http://127.0.0.1:8080/x", "127.0.0.1"),
            ("example.com", "example.com"),
            ("weird site", "weird_site"),
        ],
    )
    def test_a_host_becomes_a_directory_name(self, site, expected):
        assert _site_slug(site) == expected

    def test_it_is_never_the_working_directory(self):
        """How a repository ends up with four hundred stray .html files."""
        assert Path.cwd() not in default_output_dir("x.com").parents


# ---------------------------------------------------------------------------
# When it stops
# ---------------------------------------------------------------------------


class TestStopping:
    def test_an_ordinary_site_finishes_and_says_so(self):
        pages = {
            "/": b'<html><a href="/a">a</a></html>',
            "/a": b'<html><a href="/">home</a></html>',
        }
        base = serve(lambda path: pages.get(path, b"<html></html>"))
        result = crawl(CrawlPlan(sites=[base], depth=2, delay=0.0,
                                 respect_robots=False, max_pages=50))
        assert result.stopped_because == "done"
        assert result.complete is True
        assert result.left_queued == 0

    @pytest.mark.slow
    def test_a_faceted_trap_is_stopped_by_the_clock(self):
        """Thirty unique links per page. The queue never empties, so the
        page ceiling is the only other brake and it is an hour away."""
        counter = itertools.count()

        def body(_path: str) -> bytes:
            links = "".join(
                f'<a href="/f/{next(counter)}">x</a>' for _ in range(30)
            )
            return f"<html>{links}</html>".encode()

        base = serve(body)
        started = time.time()
        result = crawl(CrawlPlan(
            sites=[base], depth=8, delay=0.0, respect_robots=False,
            max_pages=100_000, max_seconds=3,
        ))
        took = time.time() - started

        assert result.stopped_because == "max-seconds"
        assert result.complete is False
        assert result.left_queued > 0
        # Generous, but far below the 100,000-page ceiling it would
        # otherwise have run to.
        assert took < 25, f"overshot the 3s budget by {took:.0f}s"

    @pytest.mark.slow
    def test_the_budget_is_honoured_inside_one_level_too(self):
        """Checking only between levels overshot a 4-second budget by 33
        seconds, because one level of a faceted trap held 27,931 URLs.
        A level is not a unit of time."""
        counter = itertools.count()

        def body(_path: str) -> bytes:
            links = "".join(
                f'<a href="/f/{next(counter)}">x</a>' for _ in range(40)
            )
            return f"<html>{links}</html>".encode()

        base = serve(body)
        started = time.time()
        crawl(CrawlPlan(sites=[base], depth=8, delay=0.0,
                        respect_robots=False, max_pages=100_000,
                        max_seconds=3))
        assert time.time() - started < 20

    def test_zero_removes_the_budget(self):
        pages = {"/": b"<html></html>"}
        base = serve(lambda path: pages.get(path, b"<html></html>"))
        result = crawl(CrawlPlan(sites=[base], depth=1, delay=0.0,
                                 respect_robots=False, max_seconds=0))
        assert result.stopped_because == "done"

    def test_the_page_ceiling_still_reports_itself(self):
        counter = itertools.count()

        def body(_path: str) -> bytes:
            links = "".join(
                f'<a href="/f/{next(counter)}">x</a>' for _ in range(10)
            )
            return f"<html>{links}</html>".encode()

        base = serve(body)
        result = crawl(CrawlPlan(sites=[base], depth=5, delay=0.0,
                                 respect_robots=False, max_pages=12,
                                 max_seconds=60))
        assert result.stopped_because == "max-pages"
        assert result.complete is False

    def test_the_reason_is_in_the_report(self):
        """"It finished" and "it hit a ceiling with 40,000 URLs queued"
        look identical from outside and mean opposite things about
        whether the data is complete."""
        base = serve(lambda path: b"<html></html>")
        payload = crawl(CrawlPlan(sites=[base], depth=1, delay=0.0,
                                  respect_robots=False)).to_dict()
        assert payload["stopped_because"] == "done"
        assert payload["complete"] is True

    def test_the_defaults_are_a_clock_not_just_a_ceiling(self):
        assert MAX_SECONDS > 0
        assert BARREN_LEVELS >= 1
        assert CLOCK_CHECK_EVERY >= 1
        assert CrawlPlan(sites=["http://x"]).max_seconds == MAX_SECONDS


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class TestSQLite:
    def test_it_is_an_offered_format(self):
        assert "sqlite" in FORMATS

    @pytest.fixture
    def written(self, tmp_path):
        pages = {
            "/": b'<html><title>Home</title><a href="/a">a</a></html>',
            "/a": b'<html><title>A</title><a href="/">home</a></html>',
        }
        base = serve(lambda path: pages.get(path, b"<html></html>"))
        plan = CrawlPlan(sites=[base], depth=2, delay=0.0,
                         respect_robots=False, fmt="sqlite",
                         output=tmp_path)
        result = crawl(plan)
        outputs = write_output(plan, result)
        return outputs[0]

    def test_it_writes_one_database(self, written):
        assert written.suffix == ".sqlite3"
        assert written.is_file()

    def test_no_part_file_is_left_behind(self, written):
        """An interrupted crawl must not leave a database that opens and
        is missing half the pages — that fails much later and looks like
        a data problem."""
        assert not list(written.parent.glob("*.part"))

    def test_every_page_is_a_row(self, written):
        connection = sqlite3.connect(written)
        assert connection.execute("select count(*) from pages").fetchone()[0] == 2

    def test_the_text_is_in_the_database_not_a_path(self, written):
        """So the database is the corpus and moving it moves everything."""
        connection = sqlite3.connect(written)
        texts = [r[0] for r in connection.execute("select text from pages")]
        assert any(t for t in texts)

    def test_links_are_queryable(self, written):
        connection = sqlite3.connect(written)
        assert connection.execute("select count(*) from links").fetchone()[0] > 0

    def test_the_provenance_travels_with_it(self, written):
        connection = sqlite3.connect(written)
        meta = dict(connection.execute("select key, value from crawl"))
        assert meta["sites"]
        assert meta["stopped_because"] == "done"
        assert meta["gathered_at"]

    def test_the_thing_people_do_next_is_a_query(self, written):
        """"Every page under /a that mentions X" is a query here and a
        script with jsonl."""
        connection = sqlite3.connect(written)
        rows = connection.execute(
            "select url from pages where depth = 1 order by url"
        ).fetchall()
        assert rows
