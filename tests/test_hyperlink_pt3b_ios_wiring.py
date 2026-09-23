"""HyperLink 0.72.6 pt3, checked without a Mac.

Structural checks on the Swift source, as in the other *_ios_wiring
tests, and for the same reason: each bug below produced a build that
*succeeded*.

* **The whole on-device layer was unreachable.** Search, fit, download
  and the runner — about 1,800 lines — and no screen created any of
  them. Models could not be downloaded, installed or run on the phone
  at all, because there was no way to get to any of it.
* **An HTTP error page was installed as a model.** URLSession hands a
  401 or 404 body over as a finished download; its size matched its own
  Content-Length, so a gated repo's "access restricted" page landed in
  the installed list and failed to load like a broken GGUF.
* **A download finished in the background was never installed** — the
  background session was only created on the first download.
* **`.swipeActions` on a ScrollView does nothing.** It works on List rows
  only, compiles anywhere, and the chat is not a List.
"""
from __future__ import annotations

from pathlib import Path

import pytest

IOS = Path(__file__).resolve().parents[1] / "ios" / "HyperLink" / "Sources"
if not IOS.is_dir():  # pragma: no cover
    pytest.skip("the iOS sources are not here", allow_module_level=True)


def src(*parts: str) -> str:
    return IOS.joinpath(*parts).read_text(encoding="utf-8")


def all_swift_outside(folder: str) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in IOS.rglob("*.swift")
                     if folder not in p.parts)


class TestOnDeviceIsReachable:
    @pytest.mark.parametrize("piece,created,used", [
        ("store", "ModelStore = .shared", "ModelStore.shared"),
        ("inference", "LocalInference(settings:", "hub.inference."),
        ("search", "HuggingFaceSearch()", "hub.search."),
        ("settings", "OnDeviceSettings()", "hub.settings."),
    ])
    def test_every_on_device_piece_is_reached_by_a_screen(self, piece, created, used):
        """The bug itself: all of this existed and nothing reached it.

        Followed as a chain rather than by type name — the app creates
        the hub, the hub creates each piece, a screen uses it through the
        hub — because a type name can appear in a view that never runs.
        """
        assert created in src("OnDevice", "OnDeviceHub.swift"), piece
        assert used in all_swift_outside("OnDevice"), piece

    def test_fit_is_shown_before_download(self):
        """Nobody should spend twenty minutes of cellular data to learn a
        model cannot load."""
        assert "ModelFit.plan(" in src("Views", "OnDeviceModelsView.swift")

    def test_a_view_actually_creates_the_screen(self):
        text = all_swift_outside("OnDevice")
        assert "OnDeviceModelsView()" in text

    def test_it_is_reachable_without_a_server(self):
        """Models on the phone are for having no PC. The pairing screen
        was the whole app for somebody without one."""
        assert "OnDeviceModelsView()" in src("Views", "PairingView.swift")

    def test_and_from_the_main_tabs(self):
        assert "OnDeviceModelsView()" in src("Views", "RootView.swift")

    def test_the_hub_is_created_once_at_launch(self):
        app = src("HyperLinkApp.swift")
        assert "@StateObject private var onDevice = OnDeviceHub()" in app
        assert ".environmentObject(onDevice)" in app

    def test_the_hub_forwards_what_it_holds(self):
        """A view observing the hub is not observing `inference.loaded`;
        without forwarding, "Load" stays "Load" after the model loads."""
        hub = src("OnDevice", "OnDeviceHub.swift")
        assert "inference.objectWillChange" in hub
        assert "self?.objectWillChange.send()" in hub

    def test_one_token_for_search_and_download(self):
        hub = src("OnDevice", "OnDeviceHub.swift")
        assert "store.setToken(token)" in hub and "search.setToken(token)" in hub

    def test_there_is_somewhere_to_put_the_token(self):
        assert "huggingFaceToken" in src("Views", "OnDeviceSettingsView.swift")


class TestDownloadsInstallOnlyModels:
    def test_the_http_status_is_checked(self):
        store = src("OnDevice", "ModelStore.swift")
        assert "HTTPURLResponse" in store
        assert "200..<300" in store

    def test_gated_repos_say_why(self):
        assert "accept its licence" in src("OnDevice", "ModelStore.swift")

    def test_the_gguf_magic_is_checked(self):
        """What the status cannot catch: a CDN answering 200 with HTML."""
        assert 'Data("GGUF".utf8)' in src("OnDevice", "ModelStore.swift")

    def test_the_check_runs_before_the_temp_file_is_gone(self):
        """URLSession deletes `location` when the delegate returns, so the
        check has to read it synchronously, before the hop to MainActor."""
        store = src("OnDevice", "ModelStore.swift")
        delegate = store.split("didFinishDownloadingTo location: URL")[1]
        assert delegate.index("Self.rejection(") < delegate.index("Task { @MainActor")

    def test_a_rejected_download_is_not_installed(self):
        finish = src("OnDevice", "ModelStore.swift").split("fileprivate func finish(")[1]
        assert finish.index("if let rejection") < finish.index("installed.append")


class TestBackgroundDownloads:
    def test_the_session_is_recreated_at_launch(self):
        hub = src("OnDevice", "OnDeviceHub.swift")
        assert "store.reconnect()" in hub
        assert "_ = session" in src("OnDevice", "ModelStore.swift")

    def test_the_app_delegate_takes_the_completion_handler(self):
        app = src("HyperLinkApp.swift")
        hub = src("OnDevice", "OnDeviceHub.swift")
        assert "@UIApplicationDelegateAdaptor(HyperLinkAppDelegate.self)" in app
        assert "handleEventsForBackgroundURLSession" in hub
        assert "urlSessionDidFinishEvents" in src("OnDevice", "ModelStore.swift")

    def test_one_session_identifier(self):
        """Two stores would be two sessions fighting over one identifier."""
        store = src("OnDevice", "ModelStore.swift")
        assert "static let shared = ModelStore()" in store
        assert store.count('"com.hypernix.hyperlink.models"') == 1


class TestSwipe:
    def test_the_chat_does_not_use_list_only_swipe_actions(self):
        """`.swipeActions` compiles on a ScrollView and does nothing."""
        chat = src("Views", "ChatView.swift")
        assert ".swipeActions(" not in chat
        assert ".swipeToReveal(" in chat

    def test_it_does_not_steal_the_scroll(self):
        swipe = src("Views", "SwipeActions.swift")
        assert ".simultaneousGesture(" in swipe
        assert "abs(dx) > abs(dy)" in swipe

    def test_voiceover_can_reach_the_same_actions(self):
        assert ".accessibilityActions" in src("Views", "SwipeActions.swift")

    def test_yours_offer_edit_and_resend(self):
        chat = src("Views", "ChatView.swift")
        actions = chat.split("private func swipeActions(for message")[1]
        assert '"Edit"' in actions and '"Resend"' in actions and '"Retry"' in actions

    def test_resend_confirms_when_it_removes_more_than_the_old_reply(self):
        chat = src("Views", "ChatView.swift")
        assert "editWouldRemove(message.messageID) > 1" in chat


class TestEditingResends:
    def test_an_edit_asks_again(self):
        edit = src("Store", "AppState.swift").split("func editMessage(")[1].split("\n    }\n")[0]
        assert "regenerateLast()" in edit

    def test_regenerate_is_sent_as_a_flag_not_the_text_again(self):
        """Sending the text again would put the question in twice."""
        client = src("Networking", "HyperLinkClient.swift")
        assert "let regenerate: Bool" in client
        assert "regenerate: true" in src("Store", "AppState.swift")

    def test_resend_is_an_unchanged_edit(self):
        state = src("Store", "AppState.swift")
        resend = state.split("func resendMessage(")[1].split("\n    }\n")[0]
        assert "editMessage(message.messageID, to: message.content)" in resend


class TestMemoriesRefresh:
    def test_after_a_turn_when_auto_memory_is_on(self):
        state = src("Store", "AppState.swift")
        done = state.split("case .done:")[1].split("case let .failed")[0]
        assert "refreshMemories()" in done
        assert "autoMemory" in done
