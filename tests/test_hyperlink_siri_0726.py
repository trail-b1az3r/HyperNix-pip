"""Siri on App Intents entities (0.72.6).

Siri can only recognise a value inside a spoken phrase when the
parameter is an AppEntity or AppEnum. With only String parameters,
"read me the chat called Groceries" was never one sentence. These pin
the pieces that make it one, and the call that tells Siri the names.
"""
from __future__ import annotations

import re
from pathlib import Path

IOS = Path(__file__).resolve().parent.parent / "ios" / "HyperLink" / "Sources"
INTENTS = (IOS / "Intents" / "HyperLinkIntents.swift").read_text(encoding="utf-8")
APP_STATE = (IOS / "Store" / "AppState.swift").read_text(encoding="utf-8")


def test_chats_and_models_are_entities_with_queries():
    for entity, query in (("ChatEntity", "ChatEntityQuery"), ("ModelEntity", "ModelEntityQuery")):
        assert f"struct {entity}: AppEntity" in INTENTS
        assert f"struct {query}: EntityStringQuery" in INTENTS
        assert f"static let defaultQuery = {query}()" in INTENTS
        body = INTENTS[INTENTS.index(f"struct {query}"):]
        for method in ("entities(for identifiers:", "entities(matching string:", "suggestedEntities()"):
            assert method in body.split("\n}\n", 1)[0], (query, method)


def test_the_parameters_siri_should_hear_are_entities():
    assert re.search(r"var model: ModelEntity\b", INTENTS)
    assert INTENTS.count("var chat: ChatEntity?") == 2
    assert "var chat: String" not in INTENTS and "var model: String" not in INTENTS


def test_phrases_name_the_entities():
    phrases = re.findall(r'"([^"]*\\\(\\\.\$(?:chat|model)\)[^"]*)"', INTENTS)
    assert any("$model" in p for p in phrases)
    assert any("$chat" in p for p in phrases)
    for phrase in phrases:
        assert ".applicationName" in phrase, phrase


def test_the_app_tells_siri_when_the_names_change():
    assert "import AppIntents" in APP_STATE
    assert APP_STATE.count("HyperLinkShortcuts.updateAppShortcutParameters()") == 2
    sessions = APP_STATE[APP_STATE.index("func refreshSessions()"):APP_STATE.index("func refreshModels()")]
    assert "updateAppShortcutParameters" in sessions


def test_intents_still_never_open_the_app():
    assert "openAppWhenRun = true" not in INTENTS
