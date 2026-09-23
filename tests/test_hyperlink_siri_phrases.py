"""Every Siri sentence the docs promise is one Siri can match (0.72.6).

Reported from a real iPhone: Siri answered "Sorry, HyperLink hasn't
added support for that with Siri." The README and the intents file both
told people to say "Siri, ask HyperLink what's the weather" and "load
the model Gemma 4 E2B on blazeindustries in HyperLink", and no
registered phrase matched either. A phrase can name only an entity or an
enum, never free text, so the question can't be said in the same breath.

So each advertised sentence is checked against the phrases the app
registers, read from the AppShortcutsProvider itself. An entity slot
matches only an entity's title, the way Siri matches it. The docs' example
names stand in for the titles the server sends at run time, so
"the model Gemma 4 E2B on blazeindustries" does not fit a model slot.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INTENTS_DIR = ROOT / "ios" / "HyperLink" / "Sources" / "Intents"
INTENTS = (INTENTS_DIR / "HyperLinkIntents.swift").read_text(encoding="utf-8")
SPOKEN = (INTENTS_DIR / "SpokenName.swift").read_text(encoding="utf-8")

#: Pages that are history keep the sentences they shipped with.
HISTORY = {"Changelog.md", "Release-Timeline.md"}

#: The entity titles the docs use as examples. An advertised sentence
#: names one of these where a phrase has a slot.
EXAMPLE_TITLES = ("Gemma 4 E2B", "Gemma 4", "Groceries")


def _phrases() -> list[str]:
    provider = INTENTS[INTENTS.index("struct HyperLinkShortcuts: AppShortcutsProvider"):]
    return re.findall(r'^\s*"([^"]*\\\(\.applicationName\)[^"]*)",?\s*$', provider, re.M)


def _pattern(phrase: str) -> re.Pattern[str]:
    parts = re.split(r"(\\\(\.applicationName\)|\\\(\\\.\$\w+\))", phrase)
    regex = ""
    for part in parts:
        if part == "\\(.applicationName)":
            regex += "HyperLink"
        elif part.startswith("\\(\\.$"):
            regex += "(?:" + "|".join(re.escape(title) for title in EXAMPLE_TITLES) + ")"
        else:
            regex += re.escape(part)
    return re.compile(rf"{regex}", re.I)


PATTERNS = [_pattern(p) for p in _phrases()]


def _heard(sentence: str) -> str:
    """The words after "Siri," as Siri would get them."""
    sentence = re.sub(r"^\s*(hey\s+)?siri,?\s*", "", sentence, flags=re.I)
    sentence = re.sub(r"‹[^›]*›", "Gemma 4", sentence)
    return sentence.strip().rstrip(".?!").strip()


def _advertised() -> list[tuple[str, str]]:
    pages = [ROOT / "README.md", ROOT / "ios" / "README.md", INTENTS_DIR / "HyperLinkIntents.swift",
             *sorted((ROOT / "wiki").glob("*.md"))]
    found: list[tuple[str, str]] = []
    for page in pages:
        if page.name in HISTORY:
            continue
        text = page.read_text(encoding="utf-8")
        # Markdown wraps lines, so a quoted sentence can span two.
        flat = re.sub(r"\s*\n\s*(?:///|//)?\s*", " ", text)
        for sentence in re.findall(r'"((?:hey )?siri, [^"]+)"', flat, re.I):
            found.append((page.name, sentence))
    # The iOS README's "Say" column names phrases without "Siri,".
    table = (ROOT / "ios" / "README.md").read_text(encoding="utf-8").split("## Siri", 1)[1]
    for row in re.findall(r"^\| (\"[^|]*\") \|", table, re.M):
        for sentence in re.findall(r'"([^"]+)"', row):
            found.append(("ios/README.md table", sentence))
    return found


ADVERTISED = _advertised()


def test_the_phrases_were_read():
    assert len(PATTERNS) >= 15, "the provider's phrases could not be read"
    assert len(ADVERTISED) >= 8, "the advertised sentences could not be found"


@pytest.mark.parametrize("where, sentence", ADVERTISED, ids=[s for _, s in ADVERTISED])
def test_every_advertised_sentence_is_a_registered_phrase(where, sentence):
    heard = _heard(sentence)
    assert any(p.fullmatch(heard) for p in PATTERNS), (
        f"{where} tells people to say {sentence!r}, and no registered phrase "
        f"matches it, so Siri answers that HyperLink hasn't added support for it"
    )


@pytest.mark.parametrize("sentence", [
    "Siri, ask HyperLink what's the weather",
    "Siri, load the model Gemma 4 E2B on blazeindustries in HyperLink",
])
def test_the_sentences_that_failed_are_still_refused(sentence):
    """The check has to be able to fail: these are the two the docs
    used to promise, and Siri refused them on a real phone."""
    assert not any(p.fullmatch(_heard(sentence)) for p in PATTERNS)


def test_the_question_is_never_a_phrase_slot():
    """Free text cannot be named in a phrase. Only entities can."""
    for phrase in _phrases():
        for slot in re.findall(r"\\\(\\\.\$(\w+)\)", phrase):
            assert slot in {"model", "chat"}, f"{phrase!r} names {slot}, which is not an entity"


def test_models_are_titled_the_way_they_are_said():
    """Siri matches against the entity's title. The file name is not
    something anybody says."""
    entity = INTENTS[INTENTS.index("struct ModelEntity: AppEntity"):]
    entity = entity[:entity.index("\n}\n")]
    assert "SpokenName.model(id)" in entity
    assert "synonyms: SpokenName.synonyms(id)" in entity
    assert "title: \"\\(shortModelName(id))\"" not in entity
    # The query accepts what it offered.
    assert "SpokenName.model($0)" in INTENTS[INTENTS.index("struct ModelEntityQuery"):]


def test_the_spoken_name_drops_what_describes_the_file():
    for marker in ("i?q[0-9]", "mxfp4", "bf16", "f16", "gguf"):
        assert marker in SPOKEN, f"SpokenName no longer strips {marker}"
    assert '["it", "instruct", "chat"]' in SPOKEN
