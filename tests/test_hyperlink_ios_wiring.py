"""The HyperLink iOS app's wiring, checked without Xcode.

None of this compiles Swift. It cannot — there is no macOS here — and a
test that pretended otherwise would be worse than none. What it checks is
the class of mistake that does not need a compiler to find and that a
compiler would not catch anyway:

* A Swift file that exists but is in a directory the project does not
  build. Xcode compiles it, ships it, and the feature is simply absent.
* A scene delegate the Info.plist never names. CarPlay then shows
  nothing, with no error anywhere, because the scene never connects.
* A permission the code needs and the plist does not declare. That is a
  crash on first use, not a refusal — `SFSpeechRecognizer` terminates the
  app rather than returning an error when its usage string is missing.
* An entitlement file referenced by the project and not present, or
  present and not referenced.
* A Siri phrase that names an intent struct nobody wrote.

Every one of those produces a build that succeeds. That is what makes
them worth a test here rather than leaving them to the build.
"""
from __future__ import annotations

import plistlib
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
IOS = ROOT / "ios"
SOURCES = IOS / "HyperLink" / "Sources"

pytestmark = pytest.mark.skipif(
    not IOS.is_dir(), reason="the iOS app is not in this checkout"
)


@pytest.fixture(scope="module")
def project() -> dict:
    return yaml.safe_load((IOS / "project.yml").read_text())


@pytest.fixture(scope="module")
def properties(project) -> dict:
    return project["targets"]["HyperLink"]["info"]["properties"]


def swift(name: str) -> str:
    """One Swift file's text, by filename anywhere under Sources."""
    matches = list(SOURCES.rglob(name))
    assert matches, f"{name} is not in {SOURCES}"
    return matches[0].read_text()


def all_swift() -> str:
    return "\n".join(path.read_text() for path in SOURCES.rglob("*.swift"))


class TestEverySourceFileIsBuilt:
    """A file Xcode does not compile is a feature that silently is not
    there."""

    def test_the_target_builds_the_whole_sources_tree(self, project):
        paths = {
            entry["path"] if isinstance(entry, dict) else entry
            for entry in project["targets"]["HyperLink"]["sources"]
        }
        # One recursive entry, so a new directory is picked up without
        # anyone remembering to add it. The four added in this release
        # (Theme, Intents, CarPlay, plus Views) rely on exactly that.
        assert "HyperLink/Sources" in paths

    @pytest.mark.parametrize("directory", ["Theme", "Intents", "CarPlay", "Views"])
    def test_the_new_directories_have_swift_in_them(self, directory):
        found = list((SOURCES / directory).glob("*.swift"))
        assert found, f"Sources/{directory} has no Swift files"


class TestCarPlayIsReachable:
    def test_the_scene_delegate_exists(self):
        assert "class CarPlaySceneDelegate" in swift("CarPlaySceneDelegate.swift")

    def test_the_plist_names_that_exact_class(self, properties):
        """The one that fails silently. A typo here is a car that shows
        nothing and a log with nothing in it."""
        manifest = properties["UIApplicationSceneManifest"]
        configs = manifest["UISceneConfigurations"]
        carplay = configs["CPTemplateApplicationSceneSessionRoleApplication"]
        assert len(carplay) == 1
        delegate = carplay[0]["UISceneDelegateClassName"]
        assert delegate.endswith(".CarPlaySceneDelegate")
        assert carplay[0]["UISceneClassName"] == "CPTemplateApplicationScene"

    def test_multiple_scenes_are_declared_supported(self, properties):
        """Without this the CarPlay scene cannot be connected alongside
        the phone's own window."""
        manifest = properties["UIApplicationSceneManifest"]
        assert manifest["UIApplicationSupportsMultipleScenes"] is True

    def test_the_window_role_is_left_to_swiftui(self, properties):
        """A SwiftUI-lifecycle app declares only the CarPlay role.
        Listing UIWindowSceneSessionRoleApplication would mean
        hand-writing a UIWindowSceneDelegate this app does not have, and
        the WindowGroup would stop being used."""
        configs = properties["UIApplicationSceneManifest"]["UISceneConfigurations"]
        assert "UIWindowSceneSessionRoleApplication" not in configs

    def test_the_entitlement_file_exists_and_is_referenced(self, project):
        setting = project["targets"]["HyperLink"]["settings"]["base"]
        path = setting.get("CODE_SIGN_ENTITLEMENTS")
        assert path, "no CODE_SIGN_ENTITLEMENTS; CarPlay cannot be granted"
        assert (IOS / path).is_file(), f"{path} is referenced and missing"

    def test_it_asks_for_the_communication_category(self, project):
        """The category has to match what the app does. Asking for the
        wrong one is the usual reason Apple refuses the request."""
        path = project["targets"]["HyperLink"]["settings"]["base"][
            "CODE_SIGN_ENTITLEMENTS"
        ]
        entitlements = plistlib.loads((IOS / path).read_bytes())
        assert entitlements.get("com.apple.developer.carplay-communication") is True

    def test_the_carplay_code_is_conditionally_compiled(self):
        """So a checkout, a simulator build, or a platform without
        CarPlay still compiles."""
        for name in ("CarPlayController.swift", "CarPlaySceneDelegate.swift",
                     "Speech.swift"):
            assert "#if canImport(" in swift(name), name

    def test_the_keyboard_is_gated_on_the_car_being_stopped(self):
        source = swift("CarPlayController.swift")
        assert "keyboardAvailable" in source
        # And the gate is read per use rather than stored, because the
        # answer changes while the app is running.
        assert "private var keyboardAvailable: Bool" in source
        assert "limitedUserInterfaces.contains(.keyboard)" in source

    def test_typing_uses_the_one_template_that_has_a_keyboard(self):
        """CarPlay has no general-purpose text-entry template.
        `CPSearchTemplate` is the only public one that puts a keyboard on
        screen, which is why typing is built on search."""
        source = swift("CarPlayController.swift")
        assert "CPSearchTemplate()" in source
        assert "CPSearchTemplateDelegate" in source


class TestPermissionsAreDeclared:
    """A missing usage string is a crash on first use, not a refusal."""

    @pytest.mark.parametrize("key,used_by", [
        ("NSSpeechRecognitionUsageDescription", "SFSpeechRecognizer"),
        ("NSMicrophoneUsageDescription", "AVAudioApplication.requestRecordPermission"),
        ("NSCameraUsageDescription", "UIImagePickerController"),
        ("NSPhotoLibraryUsageDescription", "PhotosPicker"),
    ])
    def test_what_the_code_uses_the_plist_declares(self, properties, key, used_by):
        source = all_swift()
        if used_by not in source:
            pytest.skip(f"nothing uses {used_by}")
        assert key in properties, f"{used_by} is used and {key} is missing"
        assert properties[key].strip(), f"{key} is empty"


class TestSiriIntents:
    """A phrase naming an intent that does not exist is a shortcut that
    fails when spoken, and nothing at build time."""

    def test_the_shortcuts_provider_exists(self):
        source = swift("HyperLinkIntents.swift")
        assert "AppShortcutsProvider" in source

    @pytest.mark.parametrize("intent", [
        "AskHyperLinkIntent", "LoadModelIntent",
        "ReadChatIntent", "SendMessageIntent",
    ])
    def test_every_advertised_intent_is_defined(self, intent):
        source = swift("HyperLinkIntents.swift")
        assert f"struct {intent}: AppIntent" in source
        assert f"intent: {intent}()" in source, f"{intent} has no phrases"

    def test_every_phrase_names_the_app(self):
        """Apple requires it, and it is also what stops "read me the
        chat" competing with every other app on the phone."""
        source = swift("HyperLinkIntents.swift")
        phrases = re.findall(r'"((?:Ask|Load|Read|Send|Switch|Tell|What)[^"]*)"', source)
        spoken = [p for p in phrases if "\\(" in p]
        assert spoken, "no phrases found; the regex or the file changed"
        for phrase in spoken:
            assert ".applicationName" in phrase, phrase

    def test_no_intent_opens_the_app(self):
        """The point of asking from a car dock or a watch is that the
        phone stays where it is."""
        source = swift("HyperLinkIntents.swift")
        assert "openAppWhenRun = true" not in source
        assert source.count("openAppWhenRun = false") >= 4

    def test_the_plist_lists_the_activity_types(self, properties):
        declared = set(properties.get("NSUserActivityTypes", []))
        for intent in ("AskHyperLinkIntent", "LoadModelIntent",
                       "ReadChatIntent", "SendMessageIntent"):
            assert intent in declared, intent

    def test_replies_are_trimmed_before_being_spoken(self):
        """A reply is written to be read: code fences, markdown, six
        paragraphs. Read aloud verbatim that is two minutes of talking
        and "asterisk asterisk important asterisk asterisk"."""
        source = swift("HyperLinkIntents.swift")
        assert "enum Speech" in source
        assert "Speech.trim(" in source


class TestThemes:
    def test_every_theme_is_complete(self):
        """A theme missing a colour is a crash at the use site, and the
        use site is a chat transcript."""
        source = swift("Theme.swift")
        entries = re.findall(r'id: "([a-z0-9]+)",', source)
        assert len(entries) >= 8, entries
        for field in ("accent:", "userBubble:", "assistantBubble:",
                      "good:", "bad:", "note:"):
            assert source.count(field) >= len(entries), field

    def test_theme_ids_are_unique(self):
        source = swift("Theme.swift")
        ids = re.findall(r'id: "([a-z0-9]+)",', source)
        assert len(ids) == len(set(ids)), ids

    def test_bubble_text_is_computed_rather_than_stored(self):
        """A stored field is a field somebody forgets to set when they
        add a theme, and the result is black on black."""
        source = swift("Theme.swift")
        assert "var userBubbleText: Color { Self.readableText(" in source
        assert "static func readableText(" in source

    def test_the_bubbles_read_the_theme(self):
        """The whole point. Before this they were one accent at 18% and
        the system grey at 12%, which in several appearances left only
        the side of the screen distinguishing the speakers."""
        source = swift("MessageBubble.swift")
        assert "theme.userBubble" in source
        assert "theme.assistantBubble" in source
        assert "Color.accentColor.opacity" not in source

    def test_the_app_applies_a_theme(self):
        source = swift("HyperLinkApp.swift")
        assert ".hyperLinkTheme(" in source
        assert "ThemeStore()" in source

    def test_settings_can_reach_the_picker(self):
        assert "ThemePickerView()" in swift("SettingsView.swift")


class TestChatRename:
    """The server has taken a title on PATCH since HyperLink shipped.
    Nothing on the phone ever sent one."""

    def test_the_client_can_send_a_title(self):
        source = swift("HyperLinkClient.swift")
        assert "func rename(" in source
        assert 'method: "PATCH"' in source

    def test_the_store_exposes_it(self):
        assert "func rename(" in swift("AppState.swift")

    def test_the_title_is_mutable_for_the_optimistic_update(self):
        """`let title` would not compile against the optimistic path,
        and waiting for a round trip feels broken on a phone four hops
        and a relay away from the PC."""
        source = swift("APITypes.swift")
        assert re.search(r"struct ChatSession[^}]*?var title: String", source, re.S)

    def test_both_the_list_and_the_chat_can_rename(self):
        assert "RenameChatSheet" in swift("ChatListView.swift")
        assert "RenameChatSheet" in swift("ChatView.swift")

    def test_the_sheet_exists(self):
        assert "struct RenameChatSheet" in swift("RenameChatSheet.swift")


class TestTheAttachmentMenu:
    def test_it_offers_all_four_ways_in(self):
        """The old menu had one item beside a bare photo button, so two
        of the four things people attach were reachable by the server
        and by nothing on screen."""
        source = swift("ChatView.swift")
        for label in ("Photo or video", "Take a photo", "File", "Code or text"):
            assert f'Label("{label}"' in source, label

    def test_the_camera_path_exists(self):
        assert "struct CameraCapture" in swift("CameraCapture.swift")
        assert "attachCameraImage" in swift("ChatView.swift")

    def test_the_code_importer_is_narrowed(self):
        """A "code or text" picker showing every file on the phone is
        the "File" row again with a different name."""
        source = swift("ChatView.swift")
        assert ".sourceCode" in source


# ---------------------------------------------------------------------------
# Symbol drift
#
# The tests above check wiring between Swift and the project/plist. These
# check Swift against Swift, and they exist because the first draft of the
# CarPlay and Siri code called four things that were not there:
#
#     PairingStore.load()                 - no such type
#     HyperLinkError.unreachable          - no such case
#     client.chat(sessionID:text:)        - the label is `content:`
#     client.configure(...)  unawaited    - HyperLinkClient is an actor
#
# Every one is a compiler error, so none of them could ship. But the
# compiler is on a Mac and this checkout is not, and "it will be caught at
# build time" is only true for whoever runs the build - which, for an iOS
# target in a Python repo, can be days later and somebody else.
#
# This is deliberately a *superset* check: the member lists are extracted
# with a regex and err towards including too much, so a name that is not
# in one is definitely not declared. It can miss drift; it cannot invent
# it.
# ---------------------------------------------------------------------------


def _strip_noise(text: str) -> str:
    """Swift with comments and string literals blanked out.

    Both would otherwise be scanned for references: this file is full of
    prose naming `PairingStore.load()`, and a user-facing string
    containing a full stop would parse as a member access.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if text.startswith("//", i):
            newline = text.find("\n", i)
            i = n if newline < 0 else newline
        elif text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = n if close < 0 else close + 2
        elif char == '"':
            if text.startswith('"""', i):
                close = text.find('"""', i + 3)
                i = n if close < 0 else close + 3
            else:
                j = i + 1
                while j < n and text[j] != '"':
                    j += 2 if text[j] == "\\" else 1
                i = j + 1
            out.append('""')
        else:
            out.append(char)
            i += 1
    return "".join(out)


_DECLARATION = re.compile(
    r"(?:^|\n)\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:public |private |internal |fileprivate |open )?(?:final )?"
    r"(?:enum|struct|class|actor|protocol|extension) +([A-Za-z_]\w*)"
)
_MEMBER = re.compile(r"\b(?:func|var|let|case)\s+([A-Za-z_]\w*)")


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    """Every Swift file, comments and strings removed, keyed by name."""
    return {
        path.name: _strip_noise(path.read_text())
        for path in sorted(SOURCES.rglob("*.swift"))
    }


def _bodies(text: str, name: str) -> list[str]:
    """Each `enum/struct/class/actor/extension <name> { … }` body."""
    found = []
    for match in _DECLARATION.finditer(text):
        if match.group(1) != name:
            continue
        start = text.find("{", match.end())
        if start < 0:
            continue
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        found.append(text[start:i])
    return found


def _outermost(body: str) -> str:
    """A body with every nested brace pair removed.

    Without this a `let trimmed` inside a method reads as a member of the
    type, and a check that every reference resolves becomes one that
    passes on anything.
    """
    kept: list[str] = []
    depth = 0
    for char in body[1:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0:
            kept.append(char)
    return "".join(kept)


def members_of(name: str, sources: dict[str, str]) -> set[str]:
    """What `name` declares, across its declaration and its extensions."""
    found: set[str] = set()
    for text in sources.values():
        for body in _bodies(text, name):
            outer = _outermost(body)
            found |= set(_MEMBER.findall(outer))
            # `case a, b` and `case a(X), b(Y)` declare two names on one
            # line; the regex above only sees the first.
            for run in re.findall(r"\bcase\s+([^\n]*)", outer):
                for piece in run.split(","):
                    first = re.match(r"\s*([A-Za-z_]\w*)", piece)
                    if first:
                        found.add(first.group(1))
    return found


def _labels(parameters: str) -> set[str]:
    """The *external* labels of a Swift parameter list.

    Not simply every identifier before a colon: `messages(in sessionID:)`
    writes the label and the internal name side by side, and taking the
    second is how a check of `chat(sessionID:text:)` could pass while the
    code did not compile.
    """
    found: set[str] = set()
    depth, current = 0, ""
    for char in parameters + ",":
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth -= 1
        if char == "," and depth == 0:
            head = current.split(":", 1)[0].split()
            if head:
                found.add(head[0])
            current = ""
        else:
            current += char
    return found


class TestTheSwiftReferencesResolve:
    #: Types reached by name from the new code. `PairingStore` is first
    #: because its absence is what started this.
    REACHED = [
        "PairingStore", "IntentBridge", "Speech", "Dictation",
        "TokenStore", "AdminCredentialStore",
        "ModelMatch", "ChatMatch",
        "HyperLinkTheme", "ThemeStore",
        "HyperLinkClient", "HyperLinkError",
    ]

    @pytest.mark.parametrize("name", REACHED)
    def test_the_type_exists(self, name, sources):
        assert members_of(name, sources), f"{name} is referenced and not declared"

    @pytest.mark.parametrize("name", REACHED)
    def test_every_member_reached_through_it_exists(self, name, sources):
        known = members_of(name, sources)
        missing = set()
        for filename, text in sources.items():
            for use in re.finditer(rf"(?<![A-Za-z0-9_.]){name}\.([A-Za-z_]\w*)", text):
                if use.group(1) not in known and use.group(1) != "self":
                    missing.add(f"{filename}: {name}.{use.group(1)}")
        assert not missing, sorted(missing)

    def test_every_error_case_switched_on_exists(self, sources):
        """`HyperLinkError.unreachable` is the one that was invented.

        Only switches over something bound by `as? HyperLinkError` are
        checked — a `switch` over an SSE event has its own cases and
        nothing to do with this.
        """
        known = members_of("HyperLinkError", sources)
        missing = set()
        for filename, text in sources.items():
            for bind in re.finditer(
                r"let ([A-Za-z_]\w*) = \w+ as\? HyperLinkError", text
            ):
                head = text.find(f"switch {bind.group(1)} ", bind.end())
                if head < 0:
                    continue
                block = text[head:text.find("\n        }\n", head)]
                for case in re.finditer(r"case (?:let )?\.([A-Za-z_]\w*)", block):
                    if case.group(1) not in known:
                        missing.add(f"{filename}: .{case.group(1)}")
        assert not missing, sorted(missing)

    def test_every_client_call_names_a_real_method(self, sources):
        known = members_of("HyperLinkClient", sources)
        missing = set()
        for filename, text in sources.items():
            for use in re.finditer(r"\bclient\??\.([A-Za-z_]\w*)", text):
                if use.group(1) not in known:
                    missing.add(f"{filename}: client.{use.group(1)}")
        assert not missing, sorted(missing)

    def test_every_client_call_is_awaited(self, sources):
        """`HyperLinkClient` is an actor. Every call across that boundary
        is `await`, and forgetting it is the mistake that made the first
        draft of the intents look synchronous."""
        missing = set()
        for filename, text in sources.items():
            for use in re.finditer(r"\bclient\??\.([A-Za-z_]\w*)\s*\(", text):
                preceding = text[max(0, use.start() - 60):use.start()]
                if "await" not in preceding:
                    missing.add(f"{filename}: client.{use.group(1)}()")
        assert not missing, sorted(missing)

    @pytest.mark.parametrize("method", ["chat", "createSession", "rename", "messages"])
    def test_call_sites_use_the_declared_argument_labels(self, method, sources):
        """`chat(sessionID:text:)` compiled in nobody's head and in no
        compiler. The labels are the signature in Swift."""
        client = sources["HyperLinkClient.swift"]
        declaration = re.search(
            rf"func {method}\((.*?)\)\s*(?:async|->|\{{)", client, re.S
        )
        assert declaration, f"no declaration for {method}"
        declared = _labels(declaration.group(1))

        for filename, text in sources.items():
            if filename == "HyperLinkClient.swift":
                continue
            for use in re.finditer(rf"\bclient\??\.{method}\(", text):
                depth, i = 0, use.end() - 1
                while i < len(text):
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                arguments = text[use.end():i]
                used = set(re.findall(r"(?:^|,)\s*([A-Za-z_]\w*)\s*:", arguments))
                unknown = used - declared
                assert not unknown, f"{filename}: {method}({sorted(unknown)})"


# ---------------------------------------------------------------------------
# Colour
#
# `Theme.swift` says every theme stays readable. That sentence was there
# before anything checked it, and two of the eight did not:
#
#   * white on the default HyperNix green measured 3.37:1, under the
#     4.5:1 WCAG asks for, on the single most-seen colour pair in the app
#   * Paper's two bubbles sat 1.13:1 apart, which in a transcript is two
#     bubbles of one colour with the side of the screen doing all the work
#
# Both are arithmetic on constants in a Swift file, which is exactly the
# kind of claim that can be checked from here.
# ---------------------------------------------------------------------------


def _luminance(rgb: int) -> float:
    """WCAG relative luminance of an `0xrrggbb` literal.

    The same formula as `Color.relativeLuminance` in Theme.swift, which
    is why the numbers here are the numbers the app gets.
    """
    def channel(value: int) -> float:
        part = value / 255
        return part / 12.92 if part <= 0.03928 else ((part + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(rgb >> 16 & 0xFF)
        + 0.7152 * channel(rgb >> 8 & 0xFF)
        + 0.0722 * channel(rgb & 0xFF)
    )


def _ratio(one: float, other: float) -> float:
    high, low = max(one, other) + 0.05, min(one, other) + 0.05
    return high / low


def _themes() -> list[dict]:
    source = swift("Theme.swift")
    table = source[source.index("static let all:"):]
    fields = ("accent", "userBubble", "assistantBubble", "good", "bad")
    found = []
    for entry in re.finditer(
        r'id: "(\w+)",\s*\n\s*name: "([^"]+)",(.*?)\n\s*\),', table, re.S
    ):
        colours = {}
        for field in fields:
            match = re.search(rf"{field}: Color\(hex: 0x([0-9A-Fa-f]{{6}})\)", entry.group(3))
            assert match, f"{entry.group(1)} has no {field}"
            colours[field] = int(match.group(1), 16)
        found.append({"id": entry.group(1), "name": entry.group(2), **colours})
    return found


@pytest.fixture(scope="module")
def themes() -> list[dict]:
    parsed = _themes()
    assert len(parsed) >= 8, [theme["id"] for theme in parsed]
    return parsed


class TestEveryThemeIsReadable:
    #: WCAG AA for normal-size text. Bubble text is body size.
    AA = 4.5
    #: How far apart the two bubbles have to be. Not a WCAG number —
    #: they are shapes, not text on each other, and side and alignment
    #: also separate them — but 1.13 was not enough and 1.5 is a
    #: difference the eye lands on.
    BUBBLES = 1.5

    @pytest.mark.parametrize("bubble", ["userBubble", "assistantBubble"])
    def test_the_text_on_each_bubble_meets_aa(self, themes, bubble):
        """`readableText` picks black or white; whichever it picks has to
        clear AA, because there is no third option to fall back on."""
        failures = []
        for theme in themes:
            background = _luminance(theme[bubble])
            best = max(_ratio(background, 0.0), _ratio(background, 1.0))
            if best < self.AA:
                failures.append(f"{theme['id']}.{bubble}: {best:.2f}:1")
        assert not failures, failures

    def test_readable_text_picks_the_better_of_the_two(self, themes):
        """And that `readableText` actually picks it. A threshold on
        luminance is what got white onto HyperNix green; comparing the
        two ratios is what this asserts the code still does."""
        source = swift("Theme.swift")
        body = re.search(
            r"static func readableText\(on background: Color\) -> Color \{(.*?)\n    \}",
            source, re.S,
        )
        assert body, "readableText is not where this test expects it"
        assert "contrastRatio(against: .black)" in body.group(1)
        assert "contrastRatio(against: .white)" in body.group(1)
        assert "relativeLuminance >" not in body.group(1), (
            "back to a luminance threshold; see the comment above readableText"
        )

    def test_the_two_bubbles_are_told_apart(self, themes):
        failures = []
        for theme in themes:
            apart = _ratio(
                _luminance(theme["userBubble"]), _luminance(theme["assistantBubble"])
            )
            if apart < self.BUBBLES:
                failures.append(f"{theme['id']}: {apart:.2f}:1")
        assert not failures, failures

    #: How far apart "connected" and "failed" have to be in luminance.
    #: They are usually green and red, and hue is exactly the cue a
    #: red-green colourblind reader does not have — so the separation has
    #: to survive the colour being taken away. Mono has no colour at all
    #: and passes on luminance alone, which is the standard being set.
    STATUS = 1.45

    def test_good_and_bad_are_told_apart_without_colour(self, themes):
        failures = []
        for theme in themes:
            assert theme["good"] != theme["bad"], theme["id"]
            apart = _ratio(_luminance(theme["good"]), _luminance(theme["bad"]))
            if apart < self.STATUS:
                failures.append(f"{theme['id']}: {apart:.2f}:1")
        assert not failures, failures

    def test_nothing_exports_a_colour_to_carplay(self):
        """CarPlay templates have no tint an app can set. A property
        claiming otherwise is a promise the picker then repeats to the
        user."""
        code = "\n".join(
            _strip_noise(path.read_text()) for path in SOURCES.rglob("*.swift")
        )
        assert "carPlayTint" not in code


# ---------------------------------------------------------------------------
# Apple's symbols
#
# TestTheSwiftReferencesResolve above checks the app's own symbols against
# the app's own source. It cannot check Apple's, and that is the gap that
# shipped a build failure: `CPTextInputTemplate` does not exist. CarPlay
# has no general-purpose text-entry template — the only public template
# with a keyboard is `CPSearchTemplate` — and nothing here caught it,
# because from Python `CPTextInputTemplate` looks exactly like
# `CPListTemplate`.
#
# CI caught it, on the one job that has an SDK, at EmitSwiftModule. That
# is the right place for it to be caught and a slow place to find out. So
# the CarPlay surface is small enough to write down, and this asserts the
# code stays inside it.
#
# An allowlist is a maintenance cost, and it is worth it here for one
# reason: CarPlay's template set is closed by design. Apple adds to it
# about once a year. A type not on this list is far more likely to be
# invented than new, and when it is genuinely new, adding a line is the
# whole cost.
# ---------------------------------------------------------------------------

#: Every CarPlay symbol this app is allowed to name. Public API as of the
#: iOS 26 SDK; add to it deliberately.
CARPLAY_API = {
    # Templates. This is the closed set — there is no text-input one.
    "CPTemplate", "CPListTemplate", "CPGridTemplate", "CPAlertTemplate",
    "CPActionSheetTemplate", "CPSearchTemplate", "CPVoiceControlTemplate",
    "CPInformationTemplate", "CPPointOfInterestTemplate", "CPTabBarTemplate",
    "CPMapTemplate", "CPNowPlayingTemplate", "CPContactTemplate",
    # Their contents.
    "CPListItem", "CPListSection", "CPListImageRowItem", "CPMessageListItem",
    "CPAlertAction", "CPGridButton", "CPBarButton", "CPTextButton",
    "CPVoiceControlState", "CPInformationItem", "CPPointOfInterest",
    "CPImageSet", "CPNowPlayingButton",
    # The scene, the controller, and what they hand you.
    "CPTemplateApplicationScene", "CPTemplateApplicationSceneDelegate",
    "CPInterfaceController", "CPInterfaceControllerDelegate",
    "CPSessionConfiguration", "CPSessionConfigurationDelegate",
    "CPLimitableUserInterface", "CPContentStyle",
    # Delegates for the templates above.
    "CPListTemplateDelegate", "CPSearchTemplateDelegate",
    "CPMapTemplateDelegate", "CPTabBarTemplateDelegate",
    "CPNowPlayingTemplateObserver",
    # Errors.
    "CPError",
}


class TestOnlyRealCarPlayTypes:
    def test_every_cp_symbol_is_one_apple_ships(self):
        """The check that would have caught `CPTextInputTemplate` here
        rather than in CI."""
        used: set[str] = set()
        for path in (SOURCES / "CarPlay").glob("*.swift"):
            text = _strip_noise(path.read_text())
            used |= set(re.findall(r"\b(CP[A-Z][A-Za-z0-9]*)\b", text))
        invented = used - CARPLAY_API
        assert not invented, (
            f"{sorted(invented)} are not CarPlay types this app knows to "
            "exist. If Apple has added one, add it to CARPLAY_API; if it "
            "was invented, it will not compile."
        )

    def test_the_allowlist_has_not_been_emptied(self):
        """A guard that passes because its list is empty is not a guard,
        and emptying it is the easiest way to make this test stop
        complaining."""
        assert len(CARPLAY_API) > 25

    def test_the_type_that_started_this_is_not_on_the_list(self):
        assert "CPTextInputTemplate" not in CARPLAY_API
        assert "CPTextInputTemplateDelegate" not in CARPLAY_API
