"""browser/firefox: the HyperNix Browser Agent extension.

The manifest and the files it names are checked everywhere. The content
script -- the part that reads pages and clicks and types in them -- is run
in a real browser page when Playwright and a Chromium are available.
The pure modules have their own tests (``bun test`` in browser/firefox).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "browser" / "firefox"
MANIFEST = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))


class TestManifest:
    def test_it_is_manifest_v3_for_firefox(self):
        assert MANIFEST["manifest_version"] == 3
        gecko = MANIFEST["browser_specific_settings"]["gecko"]
        assert gecko["id"].endswith("@hypernix")
        assert re.fullmatch(r"\d+(\.\d+){1,3}", MANIFEST["version"])

    def test_the_version_follows_the_package(self):
        from hypernix import __version__

        assert __version__.startswith(MANIFEST["version"])

    def test_every_file_it_names_exists(self):
        named = [*MANIFEST["background"]["scripts"], MANIFEST["sidebar_action"]["default_panel"],
                 MANIFEST["options_ui"]["page"], *MANIFEST["icons"].values()]
        for path in named:
            assert (EXT / path).is_file(), path

    def test_pages_load_only_local_scripts(self):
        for html in EXT.rglob("*.html"):
            for src in re.findall(r'<script[^>]*src="([^"]+)"', html.read_text(encoding="utf-8")):
                assert not src.startswith(("http:", "https:", "//")), (html, src)
                assert (html.parent / src).resolve().is_file(), (html, src)

    def test_permissions_are_what_it_uses(self):
        assert set(MANIFEST["permissions"]) == {"tabs", "scripting", "storage"}

    def test_no_model_ids_are_built_in(self):
        """The person picks the model; nothing defaults to a provider's model."""
        for path in [*EXT.glob("*.js"), *EXT.glob("lib/*.js"), *EXT.glob("options/*.js")]:
            assert not re.search(r"['\"](gpt-[\w.-]+|claude-[\w.-]+)['\"]", path.read_text(encoding="utf-8")), path


@pytest.mark.skipif(shutil.which("bun") is None and not (Path.home() / ".bun/bin/bun").exists(), reason="needs bun")
def test_the_pure_modules_pass_their_own_tests():
    bun = shutil.which("bun") or str(Path.home() / ".bun/bin/bun")
    done = subprocess.run([bun, "test"], cwd=EXT, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


# -- the content script, in a real page -------------------------------------------

PAGE = """<!doctype html><html><head><title>Test shop</title></head><body>
<h1>Welcome</h1><p>Prices from 3 pounds.</p>
<a href="/docs">Docs</a>
<form id="f" onsubmit="event.preventDefault(); document.body.dataset.submitted = document.getElementById('q').value">
  <label for="q">Search</label><input id="q" name="q">
  <input id="pw" type="password" value="secret">
  <select id="size"><option value="s">Small</option><option value="l">Large</option></select>
  <button type="submit">Go</button>
</form>
<div contenteditable="true" id="ed"></div>
<button style="display:none">Hidden</button>
<div style="height: 3000px"></div>
</body></html>"""


@pytest.fixture(scope="module")
def page():
    sync_api = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("CHROMIUM_PATH") or next(
        (str(p) for p in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome") if p.is_file()), None)
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=executable) if executable else pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser on this machine
            pytest.skip(f"no Chromium to run the content script in: {exc}")
        tab = browser.new_page()
        tab.set_content(PAGE)
        tab.add_script_tag(path=str(EXT / "content.js"))
        yield tab
        browser.close()


def agent(page, cmd, **args):
    return page.evaluate("([cmd, args]) => window.__hnxAgent[cmd](args)", [cmd, args])


class TestContentScript:
    def test_a_snapshot_numbers_what_can_be_used(self, page):
        snap = agent(page, "snapshot")
        assert snap["title"] == "Test shop"
        assert "Prices from 3 pounds." in snap["text"]
        labels = [(e["tag"], e["label"]) for e in snap["elements"]]
        assert ("a", "Docs") in labels and ("input", "Search") in labels and ("button", "Go") in labels
        assert not any(e["label"] == "Hidden" for e in snap["elements"])
        password = next(e for e in snap["elements"] if e.get("type") == "password")
        assert password["password"] is True and "value" not in password

    def test_typing_and_submitting(self, page):
        snap = agent(page, "snapshot")
        box = next(e["id"] for e in snap["elements"] if e["label"] == "Search")
        agent(page, "type", element=box, text="red shoes", submit=True)
        assert page.evaluate("document.getElementById('q').value") == "red shoes"
        assert page.evaluate("document.body.dataset.submitted") == "red shoes"

    def test_password_fields_are_refused_by_default(self, page):
        snap = agent(page, "snapshot")
        pw = next(e["id"] for e in snap["elements"] if e.get("type") == "password")
        with pytest.raises(Exception, match="password fields"):
            agent(page, "type", element=pw, text="hunter2")
        assert page.evaluate("document.getElementById('pw').value") == "secret"

    def test_choosing_an_option_by_its_text(self, page):
        snap = agent(page, "snapshot")
        size = next(e["id"] for e in snap["elements"] if e["tag"] == "select")
        assert "Large" in agent(page, "select", element=size, value="large")
        assert page.evaluate("document.getElementById('size').value") == "l"

    def test_contenteditable(self, page):
        snap = agent(page, "snapshot")
        ed = next(e["id"] for e in snap["elements"] if e["tag"] == "div")
        agent(page, "type", element=ed, text="hello")
        assert page.evaluate("document.getElementById('ed').textContent") == "hello"

    def test_click_and_stale_numbers(self, page):
        page.evaluate("document.querySelector('button[type=submit]').addEventListener('click', () => document.body.dataset.clicked = 'yes')")
        snap = agent(page, "snapshot")
        go = next(e["id"] for e in snap["elements"] if e["label"] == "Go")
        agent(page, "click", element=go)
        assert page.evaluate("document.body.dataset.clicked") == "yes"
        with pytest.raises(Exception, match="read the page again"):
            agent(page, "click", element=999)

    def test_scroll_and_find(self, page):
        assert "now at" in agent(page, "scroll", direction="down")
        assert page.evaluate("scrollY") > 0
        agent(page, "scroll", direction="top")
        assert page.evaluate("scrollY") == 0
        found = agent(page, "find", text="pounds")
        assert found["lines"] == ["Prices from 3 pounds."]

    def test_it_installs_once(self, page):
        page.add_script_tag(path=str(EXT / "content.js"))
        assert agent(page, "snapshot")["title"] == "Test shop"
