"""HyperLink's Swift for titles, image capability, private chats, the
shell, compression, memory organisation and bubble tails.

There is no Swift toolchain in the Python CI, so these check wiring the
compiler cannot: that each route the app calls exists on the server,
that each new screen is reachable, and that the permission a framework
needs is declared. The iOS job compiles it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "ios" / "HyperLink" / "Sources"


def swift(*parts: str) -> str:
    return (SOURCES.joinpath(*parts)).read_text(encoding="utf-8")


def all_swift() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in SOURCES.rglob("*.swift"))


# -- the routes the app calls exist ------------------------------------------------


@pytest.fixture(scope="module")
def routes(tmp_path_factory) -> set[tuple[str, str]]:
    pytest.importorskip("fastapi")
    from conftest import clear_t1_config

    from hypernix.t1api.app import create_app

    # Isolated like every other app a test builds: create_app opens its
    # database, and the default is the real one in ~/.hypernix.
    home = tmp_path_factory.mktemp("routes")
    with pytest.MonkeyPatch.context() as monkeypatch:
        clear_t1_config(monkeypatch)
        monkeypatch.setenv("T1_DB_PATH", str(home / "t.sqlite3"))
        monkeypatch.setenv("T1_CONFIG_DIR", str(home))
        app = create_app()
    # From the schema rather than app.routes: included routers are nested
    # (and lazily expanded) in some FastAPI versions.
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/memory/categories"),
    ("POST", "/memory/categories/rename"),
    ("POST", "/memory/organise"),
    ("POST", "/hyperlink/sessions/{session_id}/title"),
    ("GET", "/hyperlink/shell"),
    ("POST", "/hyperlink/shell"),
    ("POST", "/chat/compact/dynamic"),
])
def test_every_route_the_app_calls_exists(routes, method, path):
    assert (method, path) in routes


def test_the_client_calls_those_routes():
    client = swift("Networking", "HyperLinkClient.swift")
    for path in ["/memory/categories", "/memory/categories/rename", "/memory/organise",
                 '/hyperlink/sessions/\\(sessionID)/title', "/hyperlink/shell", "/chat/compact/dynamic"]:
        assert path in client, path


def test_preference_fields_match_the_server():
    pytest.importorskip("fastapi")
    from hypernix.t1api.schemas import PreferencesRequest

    patch = swift("Models", "APITypes.swift")
    body = patch[patch.index("struct PreferencesPatch"):]
    body = body[: body.index("\n}\n")]
    sent = set(re.findall(r"var (\w+): \w+\? = nil", body))
    assert {"auto_compact", "model_titles"} <= sent
    assert sent <= set(PreferencesRequest.model_fields)


# -- titles and compression reach the screen -------------------------------------------


def test_the_stream_understands_title_and_compacted_frames():
    sse = swift("Networking", "SSEStream.swift")
    assert 'case "title":' in sse and "return .title(title)" in sse
    assert 'case "compacted":' in sse
    state = swift("Store", "AppState.swift")
    assert "case .title:" in state and "case .compacted:" in state


def test_a_compaction_summary_is_a_marker_not_hidden():
    types = swift("Models", "APITypes.swift")
    assert "compaction_summary" in types and "isCompactionSummary" in types
    chat = swift("Views", "ChatView.swift")
    assert "!$0.isSystem || $0.isCompactionSummary" in chat
    assert "CompactionMarker(message: message)" in chat


def test_compress_and_rename_with_ai_are_offered():
    chat = swift("Views", "ChatView.swift")
    assert "state.compress(sessionID)" in chat and "state.retitle(sessionID)" in chat
    assert "state.retitle(session.sessionID)" in swift("Views", "ChatListView.swift")


def test_the_settings_have_both_toggles():
    settings = swift("Views", "MySettingsView.swift")
    assert "model_titles: on" in settings and "auto_compact: on" in settings
    assert "conversationSection" in settings.split("var body")[1][:3000]


# -- images ---------------------------------------------------------------------------


def test_the_photo_options_follow_the_models_capability():
    types = swift("Models", "APITypes.swift")
    assert 'case supportsImages = "supports_images"' in types
    chat = swift("Views", "ChatView.swift")
    assert "imagesSupported != false" in chat
    assert "state.modelSupportsImages(" in chat


# -- private chats ------------------------------------------------------------------------


def test_face_id_is_declared_where_it_is_used():
    uses = "LAContext" in all_swift()
    declared = "NSFaceIDUsageDescription" in (ROOT / "ios" / "project.yml").read_text(encoding="utf-8")
    assert uses and declared


def test_private_chats_lock_when_the_app_leaves_the_foreground():
    listing = swift("Views", "ChatListView.swift")
    assert "scenePhase" in listing and "privateChats.lock()" in listing
    # Hidden chats are not in the main list.
    assert "!hiddenIDs.contains($0.sessionID)" in listing


def test_unlocking_never_bypasses_a_phone_without_a_passcode():
    private = swift("Store", "PrivateChats.swift")
    assert "canEvaluatePolicy(.deviceOwnerAuthentication" in private
    guard = private[private.index("guard context.canEvaluatePolicy"):]
    assert "return false" in guard[: guard.index("}") + 1] or "return false" in guard[:400]


# -- shell ------------------------------------------------------------------------------


def test_the_shell_screen_is_reachable_and_says_when_it_is_off():
    assert "ShellView()" in swift("Views", "SettingsView.swift")
    view = swift("Views", "ShellView.swift")
    assert "status.enabled" in view and "howToEnable" in view


# -- memory ------------------------------------------------------------------------------


def test_memory_is_grouped_and_organisable():
    view = swift("Views", "MemoryView.swift")
    assert "Dictionary(grouping:" in view
    assert "state.organiseMemories()" in view
    assert "state.renameMemoryCategory(" in view and "state.moveMemory(" in view


# -- tails ---------------------------------------------------------------------------------


def test_bubbles_have_tails_only_at_the_end_of_a_run():
    bubble = swift("Views", "MessageBubble.swift")
    assert "struct BubbleShape: Shape" in bubble
    assert "BubbleShape(isUser: message.isUser, showsTail: showsTail)" in bubble
    chat = swift("Views", "ChatView.swift")
    assert "showsTail: tailAfter(index)" in chat
    assert "next.role != message.role" in chat


def test_new_animations_use_the_shared_motion():
    """The house rule the pt3 wiring test enforces for springs."""
    bubble = swift("Views", "MessageBubble.swift")
    assert "Motion.respecting(reduceMotion" in bubble
    assert "Animation.spring(" not in bubble
