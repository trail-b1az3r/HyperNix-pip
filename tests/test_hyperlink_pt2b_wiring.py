"""HyperLink's 0.72.6 pt2 changes, checked without a Mac.

Structural checks on the Swift source, the same trick as
`test_hyperlink_ios_wiring.py` and for the same reason: every bug below
produced a build that *succeeded* and then did the wrong thing quietly.

* **One photo at a time.** The picker was single-selection, so sending
  four photos meant four trips through the menu.
* **The version shown was the wrong version.** A server row read
  `v1.1.26.9.0.0` — the T1 API generation, labelled as if it were the
  machine's hyperNix, which is the number somebody upgrading actually
  watches and the one that does not move when the API generation does.
* **"Updated" meant "changed".** Renaming an old chat sent it to the
  top of the list reading "updated 2m ago".

The decoder checks matter most. A Swift `Codable` type with an explicit
`init(from:)` is the only way a new key can be added without signing
everybody out — the synthesised decoder throws on a missing key however
the property is defaulted, and the throw is swallowed by the `try?`
around `restore()`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

IOS = Path(__file__).resolve().parents[1] / "ios" / "HyperLink" / "Sources"

if not IOS.is_dir():  # pragma: no cover - the app is not in every checkout
    pytest.skip("the iOS sources are not here", allow_module_level=True)


def source(*parts: str) -> str:
    return (IOS.joinpath(*parts)).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Several photos at once
# ---------------------------------------------------------------------------


class TestPhotoPicker:
    def test_the_picker_takes_a_list(self):
        text = source("Views", "ChatView.swift")
        assert "photoItems: [PhotosPickerItem]" in text
        assert "selection: $photoItems" in text

    def test_there_is_no_single_item_binding_left(self):
        """Both bindings present compiles and means whichever the
        `.photosPicker` modifier names wins — silently."""
        text = source("Views", "ChatView.swift")
        assert "photoItem:" not in text.replace("photoItems:", "")

    def test_the_selection_is_capped(self):
        """The picker will hand back a whole camera roll, and every
        item is read into memory and pushed over the link to the PC."""
        text = source("Views", "ChatView.swift")
        assert "maxSelectionCount:" in text
        assert "maxPhotosAtOnce" in text

    def test_the_selection_is_cleared_after_a_batch(self):
        """SwiftUI compares the new selection with the old one, so
        picking the *same* photos twice would do nothing the second
        time unless the binding is emptied."""
        text = source("Views", "ChatView.swift")
        assert "photoItems = []" in text

    def test_a_partial_failure_keeps_the_successes(self):
        """Losing seven good uploads because the eighth could not be
        read is not a better outcome than seven attachments and a note."""
        text = source("Views", "ChatView.swift")
        assert "failed += 1" in text
        assert "continue" in text
        assert "could not be attached; the rest are ready" in text

    def test_progress_is_shown_for_a_batch(self):
        text = source("Views", "ChatView.swift")
        assert "uploadProgress" in text
        assert "Uploading" in text


# ---------------------------------------------------------------------------
# The server's actual hyperNix version
# ---------------------------------------------------------------------------


class TestServerVersion:
    def test_the_connection_stores_it(self):
        text = source("Networking", "HyperLinkClient.swift")
        assert "var hypernixVersion: String" in text
        assert "var hypernixStale: Bool" in text

    def test_the_decoder_defaults_the_new_keys(self):
        """Every record on disk predates them. A synthesised decoder
        throws on the missing key, `restore()`'s `try?` swallows it, and
        the update signs everybody out with nothing in any log."""
        text = source("Networking", "HyperLinkClient.swift")
        assert 'forKey: .hypernixVersion\n        ) ?? ""' in text
        assert "forKey: .hypernixStale\n        ) ?? false" in text

    def test_the_label_prefers_the_hypernix_version(self):
        text = source("Networking", "HyperLinkClient.swift")
        assert "var versionLabel: String" in text
        assert "hypernixVersion.isEmpty" in text

    def test_the_label_falls_back_rather_than_going_blank(self):
        """An unknown version is better shown as the older fact than as
        nothing — a row that loses its version entirely reads as a
        server that stopped answering."""
        text = source("Networking", "HyperLinkClient.swift")
        assert 'return t1Version.isEmpty ? "" : "T1 \\(t1Version)"' in text

    def test_a_pending_restart_is_visible(self):
        """`/version` reports the running process and the installed
        distribution separately, and they disagree exactly when the pip
        upgrade landed and nobody restarted the server."""
        text = source("Networking", "HyperLinkClient.swift")
        assert "restart pending" in text

    def test_the_list_shows_the_label_not_the_t1_version(self):
        text = source("Views", "ServersView.swift")
        assert "connection.versionLabel" in text
        assert 'Text("v\\(server.connection.t1Version)")' not in text

    def test_the_running_version_is_what_is_stored(self):
        """Not the installed one. A server upgraded and not restarted is
        still answering with the old code, and it is the old code the
        app has to be compatible with."""
        text = source("Store", "AppState.swift")
        assert "let running = found.hypernix" in text
        assert "connection.hypernixVersion = running" in text

    def test_it_is_persisted(self):
        """The servers list is opened while not connected to most of
        the machines on it."""
        text = source("Store", "AppState.swift")
        # The last occurrence: `refreshVersion` is also called from
        # elsewhere, and the definition is the one to look inside.
        body = text.split("func refreshVersion")[-1]
        assert "SavedServers.remember(" in body
        assert "savedServers = SavedServers.all()" in body


# ---------------------------------------------------------------------------
# "Updated" means the conversation said something
# ---------------------------------------------------------------------------


class TestUpdatedAt:
    def test_the_session_carries_both_clocks(self):
        text = source("Models", "APITypes.swift")
        assert "let touchedAt: Double" in text
        assert 'case touchedAt = "touched_at"' in text

    def test_the_decoder_is_written_by_hand(self):
        """A server too old to send `touched_at` must still decode. A
        throw here empties the chat list."""
        text = source("Models", "APITypes.swift")
        chat = text.split("struct ChatSession")[1].split("struct SessionResponse")[0]
        assert "init(from decoder: Decoder) throws" in chat
        assert "forKey: .touchedAt)" in chat

    def test_it_falls_back_to_updated_at(self):
        text = source("Models", "APITypes.swift")
        chat = text.split("struct ChatSession")[1].split("struct SessionResponse")[0]
        assert "?? updatedAt" in chat

    def test_the_list_still_shows_updated_at(self):
        """`touchedAt` exists for a client that wants it; what the row
        says "updated" about is conversation activity."""
        text = source("Views", "ChatListView.swift")
        assert "relativeTime(session.updatedAt)" in text
        assert "session.touchedAt" not in text
