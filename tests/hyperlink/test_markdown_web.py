"""HyperLink on the web renders CommonMark, GFM and Markdown Extra (0.72.8a3).

``web/markdown.js`` run under node, one case per feature, and the
security cases: nothing a model writes becomes markup the renderer did
not make.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LIBRARY = Path(__file__).resolve().parents[2] / "src" / "hypernix" / "hyperlink" / "web" / "markdown.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

CASES = {
    # CommonMark
    "atx": "# One\n### Three ###",
    "setext": "Title\n=====\n\nSub\n---",
    "emphasis": "*a* _b_ **c** __d__ ***e*** snake_case_name",
    "code_span": "`` a ` b `` and `<x>`",
    "fenced": "~~~js\nlet a = '<b>';\n~~~",
    "indented": "    code here\n    more",
    "quote": "> one\n>> two",
    "bullets": "- a\n- b\n  - c",
    "ordered": "3. three\n4. four",
    "loose": "- a\n\n- b",
    "hr": "a\n\n* * *\n\nb",
    "link_ref": "[x][ref] and [ref]\n\n[ref]: https://example.org \"T\"",
    "image": "![alt *text*](https://e.org/a.png \"cap\")",
    "autolink": "<https://example.org> and <me@example.org>",
    "escapes": "\\*not em\\* and \\# not heading",
    "entities": "&copy; &#169; &#xA9; & bare",
    "hard_break": "a  \nb\\\nc",
    # GFM
    "table": "| L | C | R |\n|:--|:-:|--:|\n| 1 | 2 | 3 |",
    "tasks": "- [x] done\n- [ ] todo",
    "strike": "~~gone~~ ~also~",
    "autolink_literal": "see www.example.org/a, https://example.org/b) and x@y.io.",
    "footnote": "A[^n] B[^m].\n\n[^n]: First.\n[^m]: Second\n    continued.",
    # Markdown Extra
    "heading_id": "## Setup {#setup .big}",
    "definitions": "Apple\n: A fruit.\n: A company.\n\nPear\n: Another fruit.",
    "abbr": "*[T1]: The HyperNix server API\nT1 runs here; T10 does not.",
    "fence_attrs": "```{.python}\nx = 1\n```",
    # Safety
    "script": "<script>alert(1)</script>",
    "img_onerror": "<img src=x onerror=alert(1)>",
    "js_link": "[x](javascript:alert(1)) [y](JAVASCRIPT:alert(1)) [z](data:text/html,hi)",
    "js_image": "![a](javascript:alert(1)) ![b](data:image/svg+xml;base64,PHN2Zz4=)",
    "attr_breakout": "[x](https://e.org/\"onmouseover=\"alert(1))",
    "allowed_inline": "<kbd>K</kbd> <sup>2</sup> <b>no</b> <span style=x>no</span>",
    "raw_block": "<div onclick=x>hi</div>",
    "unfinished": "**bold and `code",
}


@pytest.fixture(scope="module")
def html() -> dict[str, str]:
    script = LIBRARY.read_text(encoding="utf-8") + f"""
const cases = {json.dumps(CASES)};
const out = {{}};
for (const [k, v] of Object.entries(cases)) out[k] = globalThis.HyperMarkdown.render(v);
console.log(JSON.stringify(out));
"""
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


class TestCommonMark:
    def test_headings(self, html):
        assert html["atx"] == '<h1 id="user-content-one">One</h1><h3 id="user-content-three">Three</h3>'
        assert '<h1 id="user-content-title">Title</h1>' in html["setext"] and "<h2" in html["setext"]

    def test_emphasis(self, html):
        out = html["emphasis"]
        for part in ("<em>a</em>", "<em>b</em>", "<strong>c</strong>", "<strong>d</strong>",
                     "<em><strong>e</strong></em>", "snake_case_name"):
            assert part in out, out

    def test_code(self, html):
        assert "<code>a ` b</code>" in html["code_span"] and "<code>&lt;x&gt;</code>" in html["code_span"]
        assert html["fenced"] == '<pre><code class="language-js">let a = &#39;&lt;b&gt;&#39;;\n</code></pre>'
        assert html["indented"] == "<pre><code>code here\nmore\n</code></pre>"

    def test_blocks(self, html):
        assert html["quote"] == "<blockquote><p>one</p><blockquote><p>two</p></blockquote></blockquote>"
        assert html["bullets"] == "<ul><li>a</li><li>b<ul><li>c</li></ul></li></ul>"
        assert html["ordered"] == '<ol start="3"><li>three</li><li>four</li></ol>'
        assert html["loose"] == "<ul><li><p>a</p></li><li><p>b</p></li></ul>"
        assert html["hr"] == "<p>a</p><hr><p>b</p>"

    def test_links_and_images(self, html):
        assert html["link_ref"].count('<a href="https://example.org" title="T"') == 2
        assert '<img src="https://e.org/a.png" alt="alt text" title="cap" loading="lazy">' in html["image"]
        assert '<a href="https://example.org"' in html["autolink"]
        assert '<a href="mailto:me@example.org">' in html["autolink"]

    def test_escapes_entities_breaks(self, html):
        assert html["escapes"] == "<p>*not em* and # not heading</p>"
        assert html["entities"] == "<p>&copy; &#169; &#xA9; &amp; bare</p>"
        assert html["hard_break"] == "<p>a<br>\nb<br>\nc</p>"


class TestGFM:
    def test_table(self, html):
        out = html["table"]
        assert '<th style="text-align:left">L</th>' in out and '<th style="text-align:center">C</th>' in out
        assert '<td style="text-align:right">3</td>' in out

    def test_tasks(self, html):
        assert '<input type="checkbox" disabled checked> done' in html["tasks"]
        assert '<input type="checkbox" disabled> todo' in html["tasks"]

    def test_strikethrough(self, html):
        assert html["strike"] == "<p><del>gone</del> <del>also</del></p>"

    def test_autolink_literals_stop_at_punctuation(self, html):
        out = html["autolink_literal"]
        assert '<a href="http://www.example.org/a"' in out and '>www.example.org/a</a>,' in out
        assert '<a href="https://example.org/b"' in out and "</a>)" in out
        assert '<a href="mailto:x@y.io">x@y.io</a>.' in out

    def test_footnotes(self, html):
        out = html["footnote"]
        assert '<a href="#fn-n" id="fnref-n">1</a>' in out and '<a href="#fn-m" id="fnref-m">2</a>' in out
        assert '<section class="footnotes">' in out and "Second<br>\ncontinued." in out


class TestMarkdownExtra:
    def test_heading_attributes(self, html):
        assert html["heading_id"] == '<h2 id="user-content-setup" class="big">Setup</h2>'

    def test_definition_lists(self, html):
        assert html["definitions"] == ("<dl><dt>Apple</dt><dd>A fruit.</dd><dd>A company.</dd>"
                                       "<dt>Pear</dt><dd>Another fruit.</dd></dl>")

    def test_abbreviations_whole_words_only(self, html):
        out = html["abbr"]
        assert '<abbr title="The HyperNix server API">T1</abbr> runs here; T10 does not.' in out

    def test_fenced_code_attributes(self, html):
        assert '<code class="language-python">' in html["fence_attrs"]


class TestNothingRunsThatWasNotMadeHere:
    def test_scripts_and_handlers_are_text(self, html):
        assert "<script" not in html["script"] and "&lt;script&gt;" in html["script"]
        assert "<img" not in html["img_onerror"]
        assert "<div" not in html["raw_block"] and "onclick" in html["raw_block"]   # shown, escaped

    def test_unsafe_addresses_are_dropped(self, html):
        assert "href" not in html["js_link"] and "javascript" not in html["js_link"].lower().replace(
            "javascript:alert(1)", "")
        assert "<img" not in html["js_image"]

    def test_an_address_cannot_break_out_of_its_attribute(self, html):
        assert 'onmouseover="' not in html["attr_breakout"]

    def test_only_listed_inline_tags_and_none_with_attributes(self, html):
        out = html["allowed_inline"]
        assert "<kbd>K</kbd>" in out and "<sup>2</sup>" in out
        assert "&lt;b&gt;no" in out and "&lt;span style=x&gt;" in out

    def test_half_written_markup_is_text(self, html):
        assert html["unfinished"] == "<p>**bold and `code</p>"
