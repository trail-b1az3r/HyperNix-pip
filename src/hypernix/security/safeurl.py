"""``urlopen`` that only speaks HTTP(S).

``urllib.request.urlopen`` opens whatever scheme it is handed:
``file:///home/you/.hypernix/t1api/.env`` reads a local file, and
``ftp://`` and ``data:`` work too. Plenty of URLs in HyperNix come from
somewhere else -- a web page's links, a model card, a search result, a
chat message a model asked to read -- so every fetch goes through this,
which refuses anything but ``http`` and ``https`` before a byte is read.

Redirects are already safe: urllib's redirect handler follows only
http, https and ftp, and refuses ``file:``.

It calls ``urllib.request.urlopen`` at call time, so a test that patches
that still intercepts every fetch.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any

__all__ = ["ALLOWED_SCHEMES", "UnsafeURLError", "check_url", "urlopen"]

ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeURLError(ValueError):
    """A URL with a scheme HyperNix does not fetch."""


def check_url(url: str) -> str:
    """*url*, if its scheme is http or https; otherwise :class:`UnsafeURLError`."""
    scheme = urllib.parse.urlsplit(str(url)).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"Only http and https URLs are fetched, not {scheme or 'a URL with no scheme'}: {url!r}"
        )
    return str(url)


def urlopen(target: Any, *args: Any, **kwargs: Any) -> Any:
    """``urllib.request.urlopen``, after :func:`check_url` on its URL."""
    url = target.full_url if isinstance(target, urllib.request.Request) else str(target)
    check_url(url)
    return urllib.request.urlopen(target, *args, **kwargs)  # nosec B310 - scheme checked above
