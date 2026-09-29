"""Every fetch in HyperNix is http or https, never file:, ftp: or data:."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from hypernix.security import safeurl
from hypernix.security.safeurl import UnsafeURLError, check_url

SRC = Path(__file__).resolve().parents[2] / "src" / "hypernix"


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/health", "https://huggingface.co/x",
                                 "HTTPS://example.com"])
def test_http_and_https_pass(url):
    assert check_url(url) == url


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "data:text/plain,hi",
                                 "/etc/passwd", "localhost:8000/health", "gopher://x"])
def test_everything_else_is_refused(url):
    with pytest.raises(UnsafeURLError):
        check_url(url)


def test_a_file_url_never_reaches_urllib(monkeypatch, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("T1_TOKEN_SECRET=x")
    called = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: called.append(a))
    with pytest.raises(UnsafeURLError):
        safeurl.urlopen(secret.as_uri())
    with pytest.raises(UnsafeURLError):
        safeurl.urlopen(urllib.request.Request(secret.as_uri()))
    assert called == []


def test_an_allowed_request_goes_to_urllib_as_given(monkeypatch):
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda target, *a, **k: seen.append((target, k)) or "r")
    req = urllib.request.Request("https://example.com/a")
    assert safeurl.urlopen(req, timeout=3) == "r"
    assert seen == [(req, {"timeout": 3})]


def test_the_package_calls_urlopen_only_through_the_guard():
    """A new direct call would reopen file:// reads; it has to go through safeurl."""
    allowed = {"security/safeurl.py", "t1api/transport.py"}
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in allowed:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "urllib.request.urlopen(" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{number}")
    assert offenders == []
