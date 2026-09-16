"""What the HyperLink app gained in 0.72.5 pt3, checked without a Mac.

There is no Swift toolchain here and there never will be — these are
structural checks on the source, the same trick as
`test_hyperlink_ios_wiring.py`. They are worth having because every bug
in this list produced *a build that succeeded*: the app compiled, ran,
and did the wrong thing quietly.

The four:

* **Stop did not stop.** The button cancelled the phone's read task and
  nothing told the server, so the model finished the whole answer into a
  socket nobody was reading. Server-side that landed in pt2; the app has
  to make the call.
* **The model list was LM Studio's.** A machine with forty GGUFs in
  `~/.hypernix/models` showed an empty list under a message telling the
  user to go and open LM Studio.
* **Markdown arrived as asterisks.** Models write markdown; the prose
  half of a message was rendered with plain `Text`, so a numbered list
  came out as one wrapped paragraph with the numbers buried in it.
* **One server, and only one.** Pairing with a laptop overwrote the
  desktop's record, and there was one keychain account for both.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IOS = ROOT / "ios"
SOURCES = IOS / "HyperLink" / "Sources"
TESTS = IOS / "Tests" / "HyperLinkTests"

pytestmark = pytest.mark.skipif(
    not IOS.is_dir(), reason="the iOS app is not in this checkout"
)


def read(*parts: str) -> str:
    return (SOURCES.joinpath(*parts)).read_text(encoding="utf-8")


def code(*parts: str) -> str:
    """Source with comment lines dropped.

    Every one of these files explains the bug it fixes in a comment, in
    the old code's own words. A check that read the comments would find
    exactly the thing it is supposed to be looking for the absence of.
    """
    return "\n".join(
        line for line in read(*parts).splitlines()
        if not line.strip().startswith("//")
    )


class TestStopReachesTheServer:
    """Half the job was already done, and it was the visible half."""

    def test_the_client_can_ask_the_server_to_stop(self):
        assert "func stopGeneration(" in code("Networking", "HyperLinkClient.swift")

    def test_it_calls_the_endpoint_that_exists(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "/chat/stop" in body

    def test_cancelling_calls_it(self):
        """The one that matters. Cancelling the task is instant and
        cancelling the server is the point, and only one of them was
        happening."""
        assert "stopGeneration(" in code("Store", "AppState.swift")

    def test_the_local_cancel_still_happens_first(self):
        """Order is a UX decision: the button must respond now, not
        after a round trip to a machine that may be asleep."""
        body = code("Store", "AppState.swift")
        start = body.index("func cancelStreaming()")
        block = body[start:start + 1600]
        assert block.index("streamTask?.cancel()") < block.index("stopGeneration(")

    def test_the_generation_id_is_carried_from_the_stream(self):
        """Two devices open on one conversation: "stop whatever is
        running here" would stop the other phone's answer."""
        assert "generationID" in code("Networking", "SSEStream.swift")
        assert "streamingGenerationID" in code("Store", "AppState.swift")

    def test_the_id_is_sent_as_a_query_parameter(self):
        """The endpoint takes `generation_id` in the query string and no
        body. A JSON body would be read as an empty query and stop every
        generation in the session instead of the named one."""
        body = code("Networking", "HyperLinkClient.swift")
        assert "generation_id=" in body

    def test_a_failure_to_stop_is_not_shown(self):
        """A Stop that reached a server too old to have the endpoint
        still stopped the thing the person was looking at."""
        body = code("Store", "AppState.swift")
        assert "try? await client.stopGeneration(" in body


class TestTheModelListIsNotJustLMStudio:
    def test_the_client_reads_the_catalogue(self):
        assert "func modelCatalogue(" in code("Networking", "HyperLinkClient.swift")
        assert '"/hyperlink/models"' in code("Networking", "HyperLinkClient.swift")

    def test_the_store_keeps_it(self):
        assert "var catalogue" in code("Store", "AppState.swift")

    def test_the_refresh_fills_it(self):
        body = code("Store", "AppState.swift")
        start = body.index("func refreshModels()")
        assert "modelCatalogue()" in body[start:start + 900]

    @pytest.mark.parametrize("view", ["ModelsView", "ModelPickerSheet"])
    def test_the_screens_show_the_catalogue(self, view):
        body = code("Views", "ModelsView.swift")
        start = body.index(f"struct {view}")
        block = body[start:body.find("\nstruct ", start + 1)]
        assert "catalogue" in block, f"{view} is still showing only the bridge"

    def test_nothing_still_tells_people_to_open_lm_studio_to_see_a_model(self):
        """The message shown on a machine that had forty models on
        disk. The words are fine in a footer *about* LM Studio; what
        cannot come back is them being the whole explanation for an
        empty list."""
        body = code("Views", "ModelsView.swift")
        assert "Load one in LM Studio and pull to refresh" not in body
        assert "Open LM Studio on your PC, load a model" not in body

    def test_an_empty_list_says_which_source_failed(self):
        """An empty list meant either "no models" or "LM Studio is not
        running", and one blank screen was shown for both."""
        assert "unavailable" in code("Views", "ModelsView.swift")

    def test_the_catalogue_type_carries_the_source_reports(self):
        body = code("Models", "APITypes.swift")
        assert "struct CatalogueSource" in body
        assert "var unavailable" in body

    def test_every_catalogue_field_is_optional_when_decoded(self):
        """This list is merged from three sources of differing richness
        and a `local` entry carries little more than a path. Throwing on
        a missing key would drop exactly the models this exists to
        surface."""
        body = read("Models", "APITypes.swift")
        start = body.index("struct CatalogueModel")
        block = body[start:body.index("struct CatalogueSource")]
        decoder = block[block.index("init(from decoder"):]
        assert "decodeIfPresent" in decoder
        assert re.search(r"try c\.decode\(", decoder) is None, (
            "a required key in the catalogue decoder drops the whole model"
        )


class TestMarkdownIsRendered:
    def test_there_is_a_markdown_renderer(self):
        assert (SOURCES / "Views" / "Markdown.swift").is_file()

    def test_the_bubble_uses_it(self):
        assert "MarkdownText(" in code("Views", "MessageBubble.swift")

    def test_the_bubble_no_longer_renders_prose_as_plain_text(self):
        body = code("Views", "MessageBubble.swift")
        start = body.index("case let .text(body):")
        block = body[start:start + 400]
        assert "Text(body)" not in block

    def test_it_does_not_use_text_as_a_localization_key(self):
        """`Text("**hi**")` parses a little markdown by treating the
        string as a *localization key*: model output goes through the
        app's string catalogue and `%@` in a reply becomes a format
        specifier. Wrong tool, right shape."""
        body = code("Views", "Markdown.swift")
        assert "AttributedString(" in body

    def test_block_structure_is_split_before_inline_parsing(self):
        """`.full` collapses newlines into one run, which is the
        opposite of what is wanted: the blocks are what needs laying
        out."""
        body = code("Views", "Markdown.swift")
        assert "inlineOnlyPreservingWhitespace" in body
        assert ".full" not in body

    def test_a_parse_failure_still_renders_the_text(self):
        """Half-written markup arrives on every single stream. Text that
        vanishes while the model finishes a token looks like a bug."""
        body = code("Views", "Markdown.swift")
        assert "returnPartiallyParsedIfPossible" in body
        assert "?? AttributedString(text)" in body

    @pytest.mark.parametrize(
        "kind", ["heading", "paragraph", "listItem", "quote", "rule"]
    )
    def test_every_block_kind_has_a_view(self, kind):
        body = code("Views", "Markdown.swift")
        assert f"case {kind}" in body or f"case .{kind}" in body
        assert f"case let .{kind}" in body or f"case .{kind}:" in body

    def test_it_is_tested_on_the_mac_too(self):
        assert (TESTS / "MarkdownTests.swift").is_file()


class TestMoreThanOneServer:
    def test_there_is_a_list(self):
        assert (SOURCES / "Networking" / "SavedServers.swift").is_file()

    def test_the_limit_is_thirty_two(self):
        body = code("Networking", "SavedServers.swift")
        assert re.search(r"maxServers\s*=\s*32", body), "the request said 32"

    def test_each_server_has_its_own_keychain_account(self):
        """One account for all of them means forgetting one machine
        signs you out of every machine."""
        token = code("Networking", "TokenStore.swift")
        assert "account: String = legacyAccount" in token
        assert "var tokenAccount" in code("Networking", "SavedServers.swift")

    def test_saving_one_token_does_not_delete_another(self):
        """`save` is delete-then-add, and the delete used to take no
        account: saving the laptop's token would clear the desktop's."""
        token = code("Networking", "TokenStore.swift")
        start = token.index("static func save(")
        block = token[start:token.index("static func load(")]
        assert "delete(account: account)" in block
        assert re.search(r"\bdelete\(\)", block) is None

    def test_the_existing_pairing_is_migrated(self):
        """Every install has a record in the old single-pairing shape.
        An update that did not carry it across would come up unpaired on
        every device at once — the worst possible way to ship a feature
        about keeping connections."""
        body = code("Networking", "SavedServers.swift")
        assert "migrateIfNeeded" in body
        assert "PairingStore.connectionKey" in body

    def test_the_migration_runs_before_every_read(self):
        """Migrating from one call site at launch breaks the moment an
        App Intent or the CarPlay scene reads the list in a process
        where that call site never ran — which is the situation
        `PairingStore` exists for."""
        body = code("Networking", "SavedServers.swift")
        start = body.index("static func all(")
        assert "migrateIfNeeded" in body[start:start + 300]

    def test_the_migrated_record_keeps_the_legacy_token(self):
        body = code("Networking", "SavedServers.swift")
        assert "usesLegacyToken: true" in body

    def test_pairing_store_reads_through_to_the_list(self):
        """Three readers — the app, the intents, the CarPlay scene — and
        none of them should have to know there is a list."""
        body = code("Networking", "StoredPairing.swift")
        assert "SavedServers.selected(" in body
        assert "SavedServers.remember(" in body

    def test_the_store_can_switch_and_forget(self):
        body = code("Store", "AppState.swift")
        assert "func switchTo(serverID:" in body
        assert "func forget(serverID:" in body

    def test_switching_clears_the_previous_machines_content(self):
        """Sessions, messages and the model list all belong to the
        server they came from. A half-swapped view showing one machine's
        chats under another machine's name is worse than an empty one."""
        body = code("Store", "AppState.swift")
        start = body.index("func switchTo(serverID:")
        block = body[start:body.index("func forget(serverID:")]
        for cleared in ("sessions = []", "messages = []", "catalogue = .empty"):
            assert cleared in block, f"switching leaves {cleared.split()[0]} behind"

    def test_there_is_a_screen_for_it(self):
        assert (SOURCES / "Views" / "ServersView.swift").is_file()
        assert "ServersView()" in code("Views", "SettingsView.swift")

    def test_it_is_tested_on_the_mac_too(self):
        assert (TESTS / "SavedServersTests.swift").is_file()


class TestUptimeIsOnScreen:
    """A conversation that lost its context, or a pairing that stopped
    working, is usually a PC that rebooted — and nothing in the app said
    so."""

    def test_the_client_asks(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "func uptime(" in body
        assert '"/hyperlink/uptime"' in body

    def test_the_store_keeps_it(self):
        assert "var uptime: ServerUptime?" in code("Store", "AppState.swift")

    def test_it_is_refreshed_with_everything_else(self):
        body = code("Store", "AppState.swift")
        start = body.index("func refreshAll()")
        assert "refreshUptime()" in body[start:start + 500]

    def test_settings_shows_it(self):
        assert "uptime.serverDescription" in code("Views", "SettingsView.swift")

    def test_a_server_too_old_to_answer_is_not_an_error(self):
        """It is an ornament next to the server name. A red line about a
        404 for a feature nobody asked for is worse than no line."""
        body = code("Store", "AppState.swift")
        start = body.index("func refreshUptime()")
        assert "try?" in body[start:start + 200]


class TestTheVersionTheRequestAskedFor:
    def test_the_app_ships_1_0_26_9_2_3(self):
        """Derived from `T1_VERSION` rather than restated, so the app
        and the server quote the same string in a support question."""
        import subprocess
        import sys

        printed = subprocess.run(
            [sys.executable, str(IOS / "scripts" / "app_version.py")],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert printed == "1.0.26.9.2.3"
