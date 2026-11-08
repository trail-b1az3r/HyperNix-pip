"""0.72.8.post1: "New chat" in HyperLink for iOS did nothing.

Since 0.72.8 a chat belongs to an account or a key, and the server
refuses one to a keyless connection -- the usual way a phone connects
over Tailscale -- with 403 AUTH_INSUFFICIENT_SCOPE and a message saying
how to get an owner. The app put that message in `lastError`, which only
an open conversation shows, so on the Chats screen the tap seemed to be
ignored. The web app had the same hole: `ensureSession` ran outside the
`try` in `send`, so a refused chat was an unhandled rejection.

Structural checks on the Swift (no toolchain here; the iOS CI job
compiles it), the browser code, and the server answer they rely on.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "ios" / "HyperLink" / "Sources"
WEB_APP = ROOT / "src" / "hypernix" / "hyperlink" / "web" / "app.js"


def _swift(*parts: str) -> str:
    return (SOURCES.joinpath(*parts)).read_text(encoding="utf-8")


def _func(body: str, signature: str) -> str:
    start = body.index(signature)
    end = body.find("\n    func ", start + len(signature))
    return body[start:end if end != -1 else len(body)]


@pytest.mark.skipif(not SOURCES.is_dir(), reason="the iOS app is not in this checkout")
class TestTheApp:
    def test_a_failed_new_chat_is_kept_for_the_chats_screen(self):
        state = _swift("Store", "AppState.swift")
        assert "private(set) var chatListError: String?" in state
        new_session = _func(state, "func newSession(")
        assert "noteChatListFailure(error)" in new_session
        assert "noteChatListFailure(error)" in _func(state, "func refreshSessions(")

    def test_a_keyless_refusal_is_recognised(self):
        note = _func(_swift("Store", "AppState.swift"), "private func noteChatListFailure(")
        assert "isKeyless" in note
        assert 'code == "AUTH_INSUFFICIENT_SCOPE"' in note and "status == 403" in note
        # A cancelled request (a reconnect) is still nothing to report.
        assert "CancellationError" in note

    def test_the_chats_screen_shows_it_and_offers_a_way_out(self):
        view = _swift("Views", "ChatListView.swift")
        assert "state.chatListError" in view
        assert "problemSection(" in view
        section = view[view.index("private func problemSection("):]
        assert "state.keylessChatsRefused" in section and "state.unpair()" in section
        # No "New chat" button in the empty state that can only fail again.
        assert "!state.keylessChatsRefused" in view

    def test_signing_out_clears_it(self):
        unpair = _func(_swift("Store", "AppState.swift"), "func unpair(")
        assert "chatListError = nil" in unpair and "keylessChatsRefused = false" in unpair


class TestTheWebApp:
    def test_a_refused_chat_is_shown_not_dropped(self):
        code = WEB_APP.read_text(encoding="utf-8")
        send = code[code.index("async function send(text)"):]
        send = send[: send.index("\n}\n")]
        guarded = re.search(r"try \{\s*sessionId = await ensureSession\(", send)
        assert guarded, "ensureSession must run inside a try in send()"
        assert "Could not start a chat" in send
        # The message and the attachments come back for another try.
        assert "state.pending.unshift(...attached)" in send and '$("message").value = text' in send


def test_the_server_refuses_a_keyless_chat_with_the_403_the_app_reads(monkeypatch):
    from fastapi import HTTPException  # noqa: F401 - skip cleanly without the extra
    from starlette.requests import Request

    from hypernix.t1api import deps
    from hypernix.t1api.errors import T1APIError

    monkeypatch.delenv("T1_HYPERLINK_SHARED_KEYLESS_CHATS", raising=False)
    from starlette.applications import Starlette

    request = Request({"type": "http", "method": "POST", "path": "/hyperlink/sessions",
                       "headers": [], "query_string": b"", "client": ("192.168.1.9", 5000), "app": Starlette()})
    with pytest.raises(T1APIError) as refused:
        deps._refuse_keyless_chats(request)
    assert refused.value.http_status == 403
    assert refused.value.code.value == "AUTH_INSUFFICIENT_SCOPE"
