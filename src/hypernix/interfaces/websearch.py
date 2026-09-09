"""websearch — non-API HTML-parsed web searching and page fetching.

Searches without a paid API key, by scraping the HTML results pages of
several engines with fallbacks (DuckDuckGo, StartPage, SearXNG, Bing).

Page fetching now goes through :mod:`hypernix.data.gather`
--------------------------------------------------------
:func:`fetch_web_page` used to have its own ``urlopen`` call, its own
title regex, its own link extractor and its own tag stripper. ``gather``
has all four, plus three things this did not:

* **robots.txt.** This fetched anything it was pointed at.
* **A rate limit.** An agent calling this in a loop hit one host as fast
  as the network allowed.
* **A content-type check and a size ceiling.** A PDF or a 200 MB file
  was decoded as UTF-8 and returned as a page of replacement characters,
  which then looked like real text to whatever read it.

Two implementations of "fetch a page and strip its tags" is two places
for the same bug, and the one here was the weaker of the two. So this
delegates, and the shape of what it returns is unchanged -- every caller
keeps working.

The *search* functions still do their own fetching. They are scraping
one engine's results page with engine-specific parsing, not crawling,
and routing them through a crawler's politeness layer would put a
one-second delay in front of every search an agent makes.
"""
from __future__ import annotations

import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]


def _get_request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    default_headers = {
        "User-Agent": USER_AGENTS[0],
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if headers:
        default_headers.update(headers)
    return urllib.request.Request(url, headers=default_headers)


def _search_duckduckgo(query: str, max_results: int = 10) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    try:
        q_enc = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q_enc}"
        req = _get_request(url)
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", errors="ignore")

        # Parse DuckDuckGo html result blocks
        blocks = re.findall(r'<div class="result results_links[^"]*">(.*?)</div>\s*</div>', body, re.DOTALL)
        if not blocks:
            # Fallback regex for titles and snippets
            titles = re.findall(r'<a class="result__url"[^>]*>(.*?)</a>', body)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', body)
            links = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"', body)
            for t, s, link_url in zip(titles, snippets, links, strict=False):
                clean_t = html.unescape(re.sub(r"<[^>]+>", "", t)).strip()
                clean_s = html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
                clean_l = html.unescape(link_url).strip()
                if clean_t and clean_s:
                    results.append({"title": clean_t, "snippet": clean_s, "url": clean_l, "engine": "duckduckgo"})
                    if len(results) >= max_results:
                        break
        else:
            for block in blocks:
                title_m = re.search(r'<a class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_m = re.search(r'<a class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)
                url_m = re.search(r'<a class="result__url"[^>]*href="([^"]+)"', block)

                if title_m and snippet_m:
                    clean_t = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip()
                    clean_s = html.unescape(re.sub(r"<[^>]+>", "", snippet_m.group(1))).strip()
                    clean_l = html.unescape(url_m.group(1)).strip() if url_m else ""
                    results.append({"title": clean_t, "snippet": clean_s, "url": clean_l, "engine": "duckduckgo"})
                    if len(results) >= max_results:
                        break
    except Exception:
        pass
    return results


def _search_bing(query: str, max_results: int = 10) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    try:
        q_enc = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={q_enc}"
        req = _get_request(url)
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", errors="ignore")

        blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', body, re.DOTALL)
        for block in blocks:
            title_m = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', block, re.DOTALL)
            snippet_m = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
            if title_m:
                clean_url = title_m.group(1)
                clean_t = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(2))).strip()
                clean_s = html.unescape(re.sub(r"<[^>]+>", "", snippet_m.group(1))).strip() if snippet_m else ""
                results.append({"title": clean_t, "snippet": clean_s, "url": clean_url, "engine": "bing"})
                if len(results) >= max_results:
                    break
    except Exception:
        pass
    return results


def search_web_non_api(
    query: str,
    max_results: int = 10,
    engine: str = "auto",
) -> list[dict[str, str]]:
    """Perform a non-API web search with automated fallbacks across HTML scrapers.

    Args:
        query: Search prompt / query string.
        max_results: Max number of result dictionaries to return.
        engine: 'duckduckgo', 'bing', or 'auto'.

    Returns:
        List of dicts with keys: 'title', 'snippet', 'url', 'engine'.
    """
    if not query.strip():
        return []

    if engine == "duckduckgo":
        res = _search_duckduckgo(query, max_results)
        if res:
            return res

    if engine == "bing":
        res = _search_bing(query, max_results)
        if res:
            return res

    # Auto mode: try DuckDuckGo first, then Bing fallback
    results = _search_duckduckgo(query, max_results)
    if not results:
        results = _search_bing(query, max_results)

    return results


def fetch_web_page(url: str, max_length: int = 4000) -> dict[str, Any]:
    """Fetch a page and return clean text plus its links.

    Args:
        url: Absolute http(s) URL.
        max_length: Character limit on the text.

    Returns:
        ``{'url', 'title', 'text', 'links', 'status'}`` -- the same shape
        this has always returned, so callers are unaffected.

    Delegates to :mod:`hypernix.data.gather`, which brings robots.txt, a
    per-host rate limit, a content-type check and a size ceiling that
    this function did not have. See the module docstring.
    """
    from hypernix.data import gather

    # A module-level limiter, so an agent calling this in a loop is
    # paced. Created once: a fresh limiter per call would remember
    # nothing and pace nothing.
    limiter = _shared_limiter()

    robots = _shared_robots()
    if not robots.allows(url):
        return {
            "url": url,
            "title": "",
            "text": f"robots.txt at this host disallows fetching {url}.",
            "links": [],
            "status": "error: disallowed by robots.txt",
        }

    page = gather.fetch(url, timeout=15.0, limiter=limiter)
    if not page.ok:
        return {
            "url": url,
            "title": page.title,
            "text": f"Error fetching URL '{url}': {page.error or page.status}",
            "links": [],
            "status": f"error: {page.error or page.status}",
        }

    text = page.text
    if len(text) > max_length:
        text = text[:max_length] + "\u2026"

    # The old shape: 20 links, each with its anchor text. gather returns
    # bare URLs because a crawler does not need the text, so the anchors
    # are recovered here rather than changing gather's contract for one
    # caller.
    links = _links_with_text(page.html, url)[:20]

    return {
        "url": url,
        "title": page.title,
        "text": text,
        "links": links,
        "status": "success",
    }


_LIMITER = None
_ROBOTS = None


def _shared_limiter():
    """One rate limiter for the process.

    Module-level state, which is usually worth avoiding and is right
    here: the whole point is to remember when this host was last
    contacted, and a limiter created per call remembers nothing.
    """
    global _LIMITER
    if _LIMITER is None:
        from hypernix.data import gather

        # Half a second rather than gather's 1.0: this serves an
        # interactive agent looking things up, not a bulk crawl, and a
        # one-second pause per lookup is felt.
        _LIMITER = gather.RateLimiter(0.5)
    return _LIMITER


def _shared_robots():
    global _ROBOTS
    if _ROBOTS is None:
        from hypernix.data import gather

        _ROBOTS = gather.RobotsCache()
    return _ROBOTS


def _links_with_text(markup: str, base_url: str) -> list[dict[str, str]]:
    """``[{'text', 'href'}]`` for the anchors in *markup*."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
    )
    for href, body in pattern.findall(markup or ""):
        label = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        target = urllib.parse.urljoin(base_url, html.unescape(href).strip())
        if not label or not target.startswith("http") or target in seen:
            continue
        seen.add(target)
        out.append({"text": label, "href": target})
    return out


def format_search_results(results: list[dict[str, str]]) -> str:
    """Format search results list into a clean readable markdown block."""
    if not results:
        return "No web search results found."

    output = []
    for i, r in enumerate(results, 1):
        engine_str = f" [{r.get('engine', 'web')}]" if r.get('engine') else ""
        output.append(
            f"{i}. **{r.get('title', 'No Title')}**{engine_str}\n"
            f"   URL: {r.get('url', '')}\n"
            f"   {r.get('snippet', '')}"
        )
    return "\n\n".join(output)


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="hnx websearch", description="Non-API web searching utility.")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("-n", "--max-results", type=int, default=8, help="Max results to fetch")
    parser.add_argument("-e", "--engine", choices=["auto", "duckduckgo", "bing"], default="auto", help="Search engine")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    query_str = " ".join(args.query)

    results = search_web_non_api(query_str, max_results=args.max_results, engine=args.engine)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Web Search Results for '{query_str}':\n")
        print(format_search_results(results))

    return 0


if __name__ == "__main__":
    cli_main()
