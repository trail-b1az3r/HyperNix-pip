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


class TestTheRunnerReachesTheApp:
    """`/runner/*` landed as an API and nothing in HyperLink could drive
    it, so "switch model" still meant walking over to the PC — which is
    the thing the runner was built to end."""

    @pytest.mark.parametrize(
        "call", ["runnerStatus", "runnerPlan", "runnerLoad", "runnerUnload"]
    )
    def test_the_client_has_it(self, call):
        assert f"func {call}(" in code("Networking", "HyperLinkClient.swift")

    def test_the_store_exposes_load_and_unload(self):
        body = code("Store", "AppState.swift")
        assert "func loadModel(" in body
        assert "func unloadModel(" in body

    def test_planning_is_separate_from_loading(self):
        """Loading evicts whatever people are currently talking to.
        Seeing the consequence first is not a nicety."""
        assert "func planLoad(" in code("Store", "AppState.swift")
        assert "/runner/plan" in code("Networking", "HyperLinkClient.swift")

    def test_there_is_a_screen(self):
        assert (SOURCES / "Views" / "RunnerView.swift").is_file()

    def test_the_menu_reaches_it(self):
        assert "RunnerView()" in code("Views", "SettingsView.swift")

    def test_the_backends_come_from_the_server(self):
        """A machine without CUDA must not be offered CUDA, and only the
        server knows which build it has."""
        body = code("Views", "RunnerView.swift")
        assert "state.runner.backends" in body

    def test_a_server_without_a_runner_offers_nothing(self):
        """A 404 from `/runner/status` is a normal state — an older
        server — not an error to put on screen."""
        body = code("Views", "RunnerView.swift")
        assert "runnerAvailable" in body

    def test_loading_refreshes_the_model_list(self):
        """The catalogue's `loaded` flags are stale the moment a load
        succeeds, and a picker still showing the old green dot is how
        somebody talks to the wrong model."""
        body = code("Store", "AppState.swift")
        start = body.index("func loadModel(")
        assert "refreshModels()" in body[start:body.index("func unloadModel(")]

    def test_a_refusal_is_shown_rather_than_swallowed(self):
        """Unlike the read paths: somebody just asked for a specific
        thing to happen to a shared machine, and "there is no built
        llama.cpp" is something they can act on."""
        assert "runnerError" in code("Store", "AppState.swift")
        assert "runnerError" in code("Views", "RunnerView.swift")


class TestTheHardwarePage:
    def test_the_client_asks(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "func hardware(" in body
        assert '"/hyperlink/hardware"' in body

    def test_there_is_a_screen_the_menu_reaches(self):
        assert (SOURCES / "Views" / "HardwareView.swift").is_file()
        assert "HardwareView()" in code("Views", "SettingsView.swift")

    def test_a_refusal_is_explained(self):
        """Admin or partial admin server-side. An ordinary phone gets a
        403, and an empty dashboard is the wrong way to say that."""
        assert "refusal" in code("Views", "HardwareView.swift")

    def test_nothing_that_could_not_be_read_is_drawn_as_zero(self):
        """A panel rendering a missing GPU temperature as 0°C is a
        confident wrong answer about hardware nobody can see."""
        assert "unavailable" in code("Views", "HardwareView.swift")

    def test_the_field_names_match_the_sampler(self):
        """A CodingKey that does not match is a silent nil, which renders
        as "—" and looks exactly like a sensor that could not be read —
        so a typo here is invisible rather than loud."""
        from hypernix.system.hardware import CPUReading, DiskReading, GPUReading

        body = read("Models", "APITypes.swift")
        import dataclasses

        for dataclass in (CPUReading, DiskReading, GPUReading):
            for field in dataclasses.fields(dataclass):
                if field.name in ("frequency_mhz", "power_w", "power_limit_w",
                                  "used_bytes", "model", "index", "vendor"):
                    continue  # not surfaced on the phone
                assert f'"{field.name}"' in body or field.name in body, (
                    f"{dataclass.__name__}.{field.name} has no Swift CodingKey"
                )


class TestUpdatingTheServer:
    """The report was: the T1 installed thinks it is running an older
    version."""

    def test_the_endpoint_exists(self):
        from hypernix.t1api import upgrade

        assert hasattr(upgrade, "plan")

    def test_the_client_asks_for_it(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "func upgradeAdvice(" in body
        assert '"/hyperlink/upgrade"' in body

    def test_there_is_a_copy_area(self):
        assert (SOURCES / "Views" / "ServerUpdateView.swift").is_file()
        assert "ServerUpdateView()" in code("Views", "SettingsView.swift")

    def test_it_actually_copies(self):
        assert "UIPasteboard.general.string" in code("Views", "ServerUpdateView.swift")

    def test_it_shows_the_interpreter(self):
        """The one field that makes the commands correct rather than
        plausible."""
        assert "installation.executable" in code("Views", "ServerUpdateView.swift")

    def test_it_does_not_offer_to_run_them(self):
        """Updating the package under a running server is a decision
        with a restart attached. A phone button that did it silently
        takes a machine down mid-conversation."""
        body = code("Views", "ServerUpdateView.swift")
        assert "runnerLoad" not in body
        assert "POST" not in body


class TestEditMode:
    """Changing what you asked, and re-asking it."""

    def test_the_client_can_edit_and_delete(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "func editMessage(" in body
        assert "func deleteMessage(" in body

    def test_the_store_exposes_it(self):
        body = code("Store", "AppState.swift")
        assert "func editMessage(" in body
        assert "func deleteMessage(" in body

    def test_there_is_a_sheet(self):
        assert (SOURCES / "Views" / "EditMessageSheet.swift").is_file()

    def test_the_chat_view_offers_it(self):
        body = code("Views", "ChatView.swift")
        assert "EditMessageSheet(" in body
        assert "contextMenu" in body

    def test_only_your_own_messages_can_be_edited(self):
        """The server refuses an assistant edit, and the menu must not
        offer what the server will refuse — an Edit button that always
        errors is worse than no button."""
        body = code("Views", "ChatView.swift")
        start = body.index("contextMenu")
        block = body[start:start + 700]
        assert "message.isUser" in block

    def test_the_cost_is_shown_before_it_is_paid(self):
        """"Replacing this removes 11 messages" is a decision. Finding
        eleven messages gone afterwards is a bug report."""
        assert "editWouldRemove" in code("Store", "AppState.swift")
        assert "wouldRemove" in code("Views", "EditMessageSheet.swift")

    def test_the_history_is_reloaded_rather_than_patched(self):
        """The server just deleted an unknown number of rows.
        Reconstructing that locally is how a phone ends up showing a
        conversation the server does not have."""
        body = code("Store", "AppState.swift")
        start = body.index("func editMessage(")
        assert "client.messages(in: sessionID)" in body[start:start + 1400]

    def test_deleting_does_not_truncate(self):
        """Deleting is usually about removing something that should not
        be stored. Taking the thread with it makes people keep the
        secret instead."""
        body = code("Store", "AppState.swift")
        start = body.index("func deleteMessage(")
        block = body[start:start + 700]
        assert "messages.removeAll" in block

    def test_an_empty_edit_cannot_be_saved(self):
        body = code("Views", "EditMessageSheet.swift")
        assert "empty" in body
        assert "Delete it instead" in read("Views", "EditMessageSheet.swift")


class TestTheSettingsScreen:
    """Who you are, and how you want to be answered — as opposed to
    `SettingsView`, which is about the machine."""

    def test_the_client_reads_and_writes_them(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "func preferences(" in body
        assert "func savePreferences(" in body

    def test_it_is_a_patch_not_a_put(self):
        """A build that knows about six settings must not blank the four
        it has never heard of by sending them as defaults."""
        body = code("Networking", "HyperLinkClient.swift")
        start = body.index("func savePreferences(")
        assert 'method: "PATCH"' in body[start:start + 400]

    def test_every_patch_field_is_optional(self):
        """nil has to mean "leave it alone" for the PATCH to be a patch."""
        body = read("Models", "APITypes.swift")
        start = body.index("struct PreferencesPatch")
        block = body[start:body.index("}", start)]
        declarations = [
            line for line in block.splitlines() if line.strip().startswith("var ")
        ]
        assert declarations
        for line in declarations:
            stripped = line.strip()
            # The type is optional *and* carries `= nil`. Both matter:
            # the optional is what lets a field mean "leave it alone",
            # and the default is what lets a call site name one field
            # instead of all ten — a `var x: T?` with no initial value
            # gets no default in the synthesised memberwise init.
            assert "?" in stripped, f"not optional: {stripped}"
            assert "= nil" in stripped, f"no memberwise default: {stripped}"

    def test_there_is_a_screen_and_a_tab(self):
        assert (SOURCES / "Views" / "MySettingsView.swift").is_file()
        assert "MySettingsView()" in code("Views", "RootView.swift")

    def test_it_is_not_buried_under_the_server_tab(self):
        """A bio and a system prompt under a screen titled "Server" is
        how nobody finds them."""
        body = code("Views", "RootView.swift")
        assert 'Label("You"' in body

    def test_the_effort_levels_come_from_the_server(self):
        """An effort level the phone offers and the server rejects is a
        settings screen that cannot save, and the phone has no way to
        know which levels a given build has."""
        assert "state.settings.effortLevels" in code("Views", "MySettingsView.swift")

    def test_the_context_bounds_come_from_the_server_too(self):
        body = code("Views", "MySettingsView.swift")
        assert "contextFloor" in body
        assert "contextCeiling" in body

    def test_the_clamp_notes_are_shown(self):
        """Silently storing something other than what somebody typed is
        how a settings screen becomes untrustworthy."""
        body = code("Views", "MySettingsView.swift")
        assert "notes" in body
        assert "adjusted" in read("Views", "MySettingsView.swift")

    def test_the_system_prompt_has_its_own_screen(self):
        assert "SystemPromptView" in code("Views", "MySettingsView.swift")

    def test_every_setting_the_request_named_is_there(self):
        body = read("Views", "MySettingsView.swift")
        for wanted in ("Name", "bio", "System prompt", "Effort", "Context",
                       "Backup model", "tools", "Remember"):
            assert wanted.lower() in body.lower(), f"no sign of {wanted}"


class TestMemoryInTheApp:
    def test_the_client_can_read_and_change_them(self):
        body = code("Networking", "HyperLinkClient.swift")
        for call in ("func memories(", "func rememberFact(", "func forgetMemory("):
            assert call in body

    def test_there_is_a_screen(self):
        assert (SOURCES / "Views" / "MemoryView.swift").is_file()
        assert "MemoryView()" in code("Views", "MySettingsView.swift")

    def test_automatic_memories_are_marked(self):
        """The first question about a fact you did not write is where it
        came from."""
        body = code("Views", "MemoryView.swift")
        assert "isAutomatic" in body

    def test_they_can_be_deleted(self):
        """A model that remembers things about you and gives you no way
        to see them is a model you cannot correct."""
        assert "state.forget(" in code("Views", "MemoryView.swift")


class TestToolCallingReachesTheModel:
    def test_the_loop_exists(self):
        from hypernix.hyperlink.toolloop import run_tool_loop

        assert run_tool_loop

    def test_it_is_off_until_switched_on(self):
        """Letting a model write files on somebody's machine is not a
        default."""
        from hypernix.hyperlink.preferences import Preferences

        assert Preferences().tools_enabled is False

    def test_the_app_can_switch_it_on(self):
        assert "tools_enabled" in code("Views", "MySettingsView.swift")

    def test_what_it_did_is_recorded_on_the_message(self):
        """A thread that shows "wrote three files" is very different to
        read tomorrow than one that shows only the summary."""
        body = (ROOT / "src" / "hypernix" / "t1api" / "routers" / "hyperlink.py").read_text(
            encoding="utf-8"
        )
        assert "tool_rounds" in body


class TestACustomRunnerNotJustTheBridge:
    """`/runner/load` used to start a model that nothing could talk to:
    the chat path went straight to LM Studio and refused when it was
    off."""

    def test_the_chat_path_chooses(self):
        body = (ROOT / "src" / "hypernix" / "t1api" / "routers" / "hyperlink.py").read_text(
            encoding="utf-8"
        )
        assert "_chat_backend" in body

    def test_nothing_in_the_chat_path_hard_codes_lmstudio(self):
        """`backend: "lmstudio"` on a machine with no LM Studio is the
        kind of small lie that costs somebody an afternoon."""
        import ast

        source = (
            ROOT / "src" / "hypernix" / "t1api" / "routers" / "hyperlink.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name not in ("chat_turn", "chat_turn_stream"):
                continue
            body = ast.unparse(node)
            assert "'lmstudio'" not in body, f"{node.name} still hard-codes the backend"

    def test_the_app_shows_what_would_answer(self):
        body = code("Views", "SettingsView.swift")
        assert "answeringWith" in body

    def test_the_client_can_ask(self):
        body = code("Networking", "HyperLinkClient.swift")
        assert "func backends(" in body
        assert '"/hyperlink/backends"' in body


class TestAnimations:
    def test_there_is_one_vocabulary(self):
        """Views that each pick their own spring produce an app where
        two things doing the same thing move differently."""
        assert (SOURCES / "Theme" / "Motion.swift").is_file()

    def test_the_chat_uses_it(self):
        body = code("Views", "ChatView.swift")
        assert "Motion." in body
        assert "messageArrival()" in body

    def test_reduce_motion_is_honoured(self):
        """Somebody who turns it on gets vestibular symptoms from
        parallax and scale. It is not a suggestion."""
        body = code("Theme", "Motion.swift")
        assert "accessibilityReduceMotion" in body

    def test_reduced_motion_still_shows_the_change(self):
        """"Reduce motion" means no movement, not no feedback — a change
        that simply appears with no transition is one people miss."""
        body = code("Theme", "Motion.swift")
        assert ".opacity" in body

    def test_no_view_invents_its_own_spring(self):
        """The check that keeps the vocabulary a vocabulary."""
        import re

        offenders = []
        for path in sorted((SOURCES / "Views").glob("*.swift")):
            text = "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("//")
            )
            for match in re.finditer(r"Animation\.spring\(|\.spring\(response:", text):
                offenders.append(f"{path.name}:{text[:match.start()].count(chr(10)) + 1}")
        assert not offenders, (
            "these declare their own spring instead of using Motion: "
            + ", ".join(offenders)
        )


class TestTheBackgroundWindow:
    """The request asked for 2.5 hours. iOS grants about 30 seconds, and
    an app that claims otherwise is one that gets terminated and does not
    notice."""

    def test_there_is_a_background_session(self):
        assert (SOURCES / "Store" / "BackgroundSession.swift").is_file()

    def test_the_window_is_two_and_a_half_hours(self):
        body = code("Store", "BackgroundSession.swift")
        assert "2.5 * 60 * 60" in body

    def test_it_does_not_pretend_ios_grants_that(self):
        """The honest part: the promise is kept because the *server*
        keeps generating and persists as it goes, so the phone does not
        need to stay connected at all."""
        body = read("Store", "BackgroundSession.swift")
        assert "systemGrant" in body
        assert "30 seconds" in body

    def test_coming_back_reloads_the_conversation(self):
        assert "state.reload(" in code("HyperLinkApp.swift")

    def test_the_store_can_reload_one_thread(self):
        assert "func reload(sessionID:" in code("Store", "AppState.swift")


class TestEverySettingCanActuallyBeSet:
    """A setting the app can never send is indistinguishable from one
    that does not save.

    Three were in that state at once and the report was simply "the
    settings don't save":

    * `bio` was bound to a TextField, loaded on appear, and sent
      nowhere. There was no `save` call naming it anywhere in the app.
    * `display_name` was sent only from `.onSubmit`, which is the Return
      key. Typing a name and tapping Back lost it.
    * `backend` had a field on the patch type and an accepting endpoint,
      and no screen offered it at all.

    So the check is against the patch type rather than against a list
    written by hand: every field it declares has to be sent from
    somewhere, or it is a setting nobody can change.

    Matching is deliberately strict. A loose `\\bbackend\\s*:` matched
    `let backend: String` on an unrelated struct and passed `backend` as
    "sent" while it was the broken one -- which is the same lesson as
    every other guard in this repo: a check that matches the wrong thing
    reads as coverage.
    """

    @staticmethod
    def _sources() -> str:
        return "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((IOS / "HyperLink" / "Sources").rglob("*.swift"))
        )

    @staticmethod
    def _patch_fields(source: str) -> list[str]:
        import re

        body = re.search(r"struct PreferencesPatch[^{]*\{(.*?)\n\}", source, re.S)
        assert body, "PreferencesPatch not found"
        fields = re.findall(r"var (\w+):", body.group(1))
        assert fields, "PreferencesPatch has no fields"
        return fields

    @staticmethod
    def _labels_in_patch_literals(source: str) -> set[str]:
        """Argument labels of every `.init(...)` / `PreferencesPatch(...)`.

        Balanced parens rather than a character class. The first version
        used `\\([^)]*\\b<field>\\s*:` and reported `context_maximum`
        as never sent -- it is sent, from a call spanning four lines
        whose *first* argument is `Int(contextMinimum) ?? 0`, so the
        class stopped at that inner `)` before reaching the second
        label. A matcher that cannot read the code it guards produces
        exactly the false alarm that teaches people to delete it.
        """
        import re

        labels: set[str] = set()
        for opening in re.finditer(r"(?:\.init|PreferencesPatch)\s*\(", source):
            i = opening.end()
            depth = 1
            while i < len(source) and depth:
                if source[i] == "(":
                    depth += 1
                elif source[i] == ")":
                    depth -= 1
                i += 1
            args = source[opening.end():i - 1]
            # Only top-level labels: `Int(x) ?? 0` must not contribute.
            depth = 0
            current = []
            pieces = []
            for ch in args:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if ch == "," and depth == 0:
                    pieces.append("".join(current))
                    current = []
                else:
                    current.append(ch)
            pieces.append("".join(current))
            for piece in pieces:
                found = re.match(r"\s*(\w+)\s*:", piece)
                if found:
                    labels.add(found.group(1))
        return labels

    def test_every_patch_field_is_sent_from_somewhere(self):
        import re

        source = self._sources()
        labels = self._labels_in_patch_literals(source)
        missing = []
        for field in self._patch_fields(source):
            assigned = re.search(rf"\w*[Pp]atch\.{field}\s*=", source)
            if field not in labels and not assigned:
                missing.append(field)
        assert not missing, (
            "these settings exist on PreferencesPatch and nothing in the app "
            "ever sends them, so they cannot be changed from the phone: "
            + ", ".join(missing)
        )

    def test_the_matcher_reads_a_multi_line_call(self):
        """The false alarm that nearly deleted this test, pinned.

        A call whose first argument contains its own parentheses must
        not hide the labels after it.
        """
        sample = (
            "await save(.init(\n"
            "    context_minimum: Int(a) ?? 0,\n"
            "    context_maximum: Int(b) ?? 0\n"
            "))\n"
        )
        labels = self._labels_in_patch_literals(sample)
        assert labels == {"context_minimum", "context_maximum"}, labels

    def test_the_strict_matcher_rejects_a_bare_declaration(self):
        """The false pass that hid `backend`, pinned.

        `let backend: String` is a declaration, not a send. If this
        stops being rejected, the test above is matching type
        declarations again and will pass with the UI missing.
        """
        import re

        decoy = "struct Thing {\n    let backend: String\n}\n"
        literal = re.search(
            r"(?:\.init|PreferencesPatch)\s*\([^)]*\bbackend\s*:", decoy, re.S
        )
        assigned = re.search(r"\w*[Pp]atch\.backend\s*=", decoy)
        assert not literal and not assigned

    def test_the_free_text_fields_commit_when_they_lose_focus(self):
        """Everything else on that screen commits on change -- a toggle,
        a picker, a button. Text fields have no such moment, so leaving
        one has to be it."""
        view = (
            IOS / "HyperLink" / "Sources" / "Views" / "MySettingsView.swift"
        ).read_text(encoding="utf-8")
        assert "@FocusState" in view, "nothing tracks which field is being edited"
        assert "onChange(of: focused)" in view, (
            "leaving a field does not commit it"
        )
        assert "onDisappear" in view, (
            "leaving the screen with the keyboard up does not commit"
        )


class TestAddingAndEditingAServerFromTheList:
    """The Servers screen could switch and forget, and nothing else.

    Adding a machine meant unpairing the current one first — the pairing
    screen was only reachable when nothing was paired at all — and a
    server that had moved to a different port could not be followed
    without pairing again, which means finding a key again and throwing
    away the pinned fingerprint that says it is the same machine.
    """

    def test_there_is_a_button_to_add_one(self):
        body = code("Views", "ServersView.swift")
        assert ".toolbar {" in body
        assert "Add a server" in body

    def test_the_button_opens_the_pairing_screen(self):
        """Not a second pairing flow. `PairingView` goes through
        `SavedServers.remember`, which is what keeps the others."""
        body = code("Views", "ServersView.swift")
        assert "PairingView {" in body
        assert ".sheet(isPresented: $addingServer)" in body

    def test_the_sheet_closes_when_the_pairing_lands(self):
        """`isPaired` is already true here, so the root swap that
        dismisses the first-run pairing screen never happens — the form
        would sit on top of the app it had just added a machine to."""
        pairing = code("Views", "PairingView.swift")
        assert "var onDone: (() -> Void)?" in pairing
        assert "if paired { onDone?() }" in pairing
        assert "addingServer = false" in code("Views", "ServersView.swift")

    def test_a_failed_pairing_keeps_the_form_up(self):
        """Dismissing unconditionally would take `connectionError` off
        screen with it — the one thing that says what went wrong."""
        pairing = code("Views", "PairingView.swift")
        start = pairing.index("let paired: Bool")
        block = pairing[start:start + 900]
        assert "if paired { onDone?() }" in block
        assert "onDone?()\n" not in block.replace("if paired { onDone?() }", "")

    def test_there_is_only_one_navigation_stack(self):
        """PairingView brings its own. Wrapping the sheet in another
        gives two navigation bars."""
        body = code("Views", "ServersView.swift")
        start = body.index(".sheet(isPresented: $addingServer)")
        assert "NavigationStack" not in body[start:start + 300]

    def test_the_sheet_has_a_way_out(self):
        pairing = code("Views", "PairingView.swift")
        assert 'Button("Cancel") { onDone?() }' in pairing
        assert "if onDone != nil {" in pairing

    def test_the_button_stops_at_the_limit(self):
        """Past 32, pairing silently drops the least recently used
        machine. Offering the button anyway makes that a surprise."""
        body = code("Views", "ServersView.swift")
        assert "state.savedServers.count >= SavedServers.maxServers" in body
        start = body.index("Add a server")
        assert ".disabled(" in body[start:start + 200]

    def test_swiping_right_edits_the_port(self):
        body = code("Views", "ServersView.swift")
        assert ".swipeActions(edge: .leading)" in body
        assert "editingPort = server" in body

    def test_forgetting_is_still_the_other_way(self):
        """Destructive on the leading edge, one thumb away from the
        edit, is how you forget a machine you meant to re-port."""
        body = code("Views", "ServersView.swift")
        trailing = body.index(".swipeActions(edge: .trailing)")
        leading = body.index(".swipeActions(edge: .leading)")
        assert "role: .destructive" in body[trailing:leading]
        assert "role: .destructive" not in body[leading:leading + 400]

    def test_the_editor_previews_every_endpoint(self):
        """A pairing carries a LAN address and a tailnet name. "Change
        the port" has to mean all of them, and showing what they become
        is how somebody sees that it did."""
        body = code("Views", "ServersView.swift")
        assert "struct PortEditor" in body
        assert "ForEach(server.connection.endpoints" in body
        assert "SavedServers.replacingPort(in: endpoint, with: wanted)" in body

    def test_the_store_applies_it(self):
        body = code("Store", "AppState.swift")
        assert "func setPort(serverID: String, port: Int) async -> Bool" in body

    def test_changing_the_current_machines_port_reconnects(self):
        """Editing the machine you are talking to and then still talking
        to the old port is the one outcome that makes this useless."""
        body = code("Store", "AppState.swift")
        start = body.index("func setPort(serverID:")
        block = body[start:body.index("func forget(serverID:")]
        assert "client.configure(" in block
        assert "refreshAll()" in block

    def test_the_rewrite_is_in_the_model_not_the_view(self):
        """So the CarPlay scene and the intents read the same endpoints
        the phone does."""
        body = code("Networking", "SavedServers.swift")
        assert "static func setPort(" in body
        assert "static func replacingPort(" in body
        assert "static func commonPort(" in body

    def test_an_impossible_port_is_refused_before_anything_is_written(self):
        body = code("Networking", "SavedServers.swift")
        start = body.index("static func setPort(")
        block = body[start:start + 700]
        assert "(1...65535).contains(port)" in block
        assert block.index("guard (1...65535)") < block.index("write(")

    def test_it_is_tested_on_the_mac_too(self):
        body = (TESTS / "SavedServersTests.swift").read_text(encoding="utf-8")
        assert "func testTheNewPortReachesEveryEndpoint()" in body
        assert "func testTheMachineIsStillTheSameMachine()" in body
