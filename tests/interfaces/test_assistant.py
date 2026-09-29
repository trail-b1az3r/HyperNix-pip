"""The assistant does real work, or says why it can't.

These tests exist because the module used to do neither: it crashed on
import without ``rich``, and every command that "succeeded" printed a
canned string instead of doing the thing — a chat reply of "Response
generated.", a TTS "Saved to output.wav" with no file, a web UI that
announced a server it never started.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from hypernix.interfaces import assistant

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def _unwrapped(text: str) -> str:
    """Console output with rich's soft wrapping undone.

    ``console.print`` wraps at the terminal width — 80 columns when
    stdout isn't a tty, as under pytest — so a long path is emitted with
    a newline inserted mid-string. That is presentation, not content, but
    it breaks a naive ``in`` check: on Linux CI a tmpdir path
    (``/tmp/pytest-of-runner/...``) fits in 80 columns and on macOS the
    same path (``/private/var/folders/df/.../T/pytest-of-runner/...``)
    does not, so the assertion passed on one runner and failed on the
    other for reasons that had nothing to do with the code under test.

    Stripping whitespace and box-drawing characters from both sides of
    the comparison keeps the assertion meaningful — the full path still
    has to be present, in order — without pinning it to a particular
    terminal width.
    """
    return "".join(ch for ch in text if not ch.isspace() and ch not in "│|")


class TestRichlessFallback:
    def test_module_imports_without_rich(self) -> None:
        """`console = Console()` at module scope used to raise NameError."""
        script = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("no rich")
        return None

sys.meta_path.insert(0, Blocker())
for mod in [m for m in sys.modules if m.startswith("rich")]:
    del sys.modules[mod]

from hypernix.interfaces import assistant
assert assistant.RICH_AVAILABLE is False
assistant.console.print("[bold red]hi[/]")
assistant.InteractiveCLI()
print("OK")
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin:/usr/local/bin"},
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().splitlines()[-1] == "OK"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[bold red]hi[/]", "hi"),
            ("[green]a[/green] b", "a b"),
            ("  [1] Models", "  [1] Models"),          # menu labels survive
            ("array[0] and list[i]", "array[0] and list[i]"),
        ],
    )
    def test_markup_stripping(self, raw: str, expected: str) -> None:
        assert assistant._MARKUP_RE.sub("", raw) == expected

    def test_plain_prompt_uses_default_on_empty_input(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt="": "")
        assert assistant._PlainPrompt.ask("Choice", default="7") == "7"

    def test_plain_prompt_rejects_values_outside_choices(self, monkeypatch) -> None:
        answers = iter(["nope", "b"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
        assert assistant._PlainPrompt.ask("Pick", choices=["a", "b"]) == "b"


class TestModelSelection:
    def test_returns_none_when_nothing_is_configured(self, monkeypatch) -> None:
        """No default and no API key must not silently pick a 27B download."""
        import hypernix.system.config as config

        monkeypatch.setattr(config, "get_default_model", lambda: None)
        monkeypatch.setattr(config, "get_provider_key", lambda _vendor: None)
        assert assistant._default_model_short() is None

    def test_prefers_the_configured_default(self, monkeypatch) -> None:
        import hypernix.interfaces.hyped_pro_core as core
        import hypernix.system.config as config

        pick = core.MODELS[-1].short
        monkeypatch.setattr(config, "get_default_model", lambda: pick)
        monkeypatch.setattr(config, "get_provider_key", lambda _vendor: "key")
        assert assistant._default_model_short() == pick

    def test_ignores_a_configured_model_it_does_not_know(self, monkeypatch) -> None:
        import hypernix.system.config as config

        monkeypatch.setattr(config, "get_default_model", lambda: "not-a-model")
        monkeypatch.setattr(config, "get_provider_key", lambda _vendor: None)
        assert assistant._default_model_short() is None

    def test_falls_back_to_a_cloud_model_with_a_key(self, monkeypatch) -> None:
        import hypernix.interfaces.hyped_pro_core as core
        import hypernix.system.config as config

        cloud = next(m for m in core.MODELS if m.kind == "cloud")
        monkeypatch.setattr(config, "get_default_model", lambda: None)
        monkeypatch.setattr(
            config, "get_provider_key", lambda vendor: "k" if vendor == cloud.vendor else None,
        )
        assert assistant._default_model_short() == cloud.short


class TestChatLLM:
    def test_chat_dispatches_through_the_real_provider_registry(self, monkeypatch) -> None:
        import hypernix.interfaces.hyped_pro_core as core

        seen: dict[str, object] = {}

        def fake_send(model_short, messages, system=None, **kwargs):
            seen.update(model=model_short, messages=messages, system=system, kwargs=kwargs)
            return {"content": "real reply", "thinking": None}

        monkeypatch.setattr(core, "send_chat_message", fake_send)
        llm = assistant.ChatLLM("claude-haiku-4.5", system="be terse")
        out = llm.chat([{"role": "user", "content": "hi"}], max_new_tokens=32)

        assert out == "real reply"
        assert seen["model"] == "claude-haiku-4.5"
        assert seen["system"] == "be terse"
        assert seen["kwargs"]["max_tokens"] == 32

    def test_generate_wraps_a_single_user_turn(self, monkeypatch) -> None:
        import hypernix.interfaces.hyped_pro_core as core

        captured: list[list[dict[str, str]]] = []

        def fake_send(_model, messages, **_kwargs):
            captured.append(messages)
            return {"content": "answer", "thinking": None}

        monkeypatch.setattr(core, "send_chat_message", fake_send)
        assert assistant.ChatLLM("m").generate("what is 2+2?") == "answer"
        assert captured == [[{"role": "user", "content": "what is 2+2?"}]]


class TestRealOutputs:
    def test_write_wav_produces_a_playable_file(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")

        out = tmp_path / "speech.wav"
        written = assistant._write_wav(out, torch.zeros(2048), 22050)

        assert out.is_file()
        assert written == out.stat().st_size > 44  # header + frames
        with wave.open(str(out), "rb") as wav:
            assert wav.getframerate() == 22050
            assert wav.getnchannels() == 1
            assert wav.getnframes() == 2048

    def test_tts_command_writes_the_file_it_reports(self, tmp_path, monkeypatch, capsys) -> None:
        torch = pytest.importorskip("torch")
        import hypernix.models.workshop as workshop

        out = tmp_path / "spoken.wav"

        class FakeEngine:
            config = workshop.TTSConfig(sample_rate=16000)

            def initialize(self) -> None:
                pass

            def synthesize(self, _text: str, speaker_id: int = 0):
                return torch.zeros(1600)

        monkeypatch.setattr(workshop, "TTSEngine", FakeEngine)
        answers = iter(["hello there", str(out), ""])
        monkeypatch.setattr(assistant._prompt, "ask", staticmethod(lambda *a, **k: next(answers)))

        assistant.InteractiveCLI().cmd_tts()

        assert out.is_file(), "cmd_tts reported a file it never wrote"
        assert _unwrapped(str(out.resolve())) in _unwrapped(capsys.readouterr().out)

    def test_tts_command_reports_the_real_error(self, monkeypatch, capsys) -> None:
        import hypernix.models.workshop as workshop

        class ExplodingEngine:
            config = workshop.TTSConfig()

            def initialize(self) -> None:
                raise RuntimeError("TTS requires transformers")

        monkeypatch.setattr(workshop, "TTSEngine", ExplodingEngine)
        answers = iter(["hello", "out.wav", ""])
        monkeypatch.setattr(assistant._prompt, "ask", staticmethod(lambda *a, **k: next(answers)))

        assistant.InteractiveCLI().cmd_tts()

        out = capsys.readouterr().out
        assert "TTS requires transformers" in out
        assert "Wrote" not in out

    def test_shell_command_reports_output_and_exit_status(self, capsys) -> None:
        assistant.InteractiveCLI()._run_shell("echo marker-line; exit 3")
        out = capsys.readouterr().out
        assert "marker-line" in out
        assert "exit status 3" in out


class TestChatLoop:
    def _run(self, monkeypatch, inputs: list[str], reply: str = "a real answer"):
        import hypernix.interfaces.hyped_pro_core as core

        sent: list[list[dict[str, str]]] = []

        def fake_send(_model, messages, system=None, **_kwargs):
            sent.append([dict(m) for m in messages])
            return {"content": reply, "thinking": None}

        monkeypatch.setattr(core, "send_chat_message", fake_send)
        answers = iter([*inputs, "/quit"])
        monkeypatch.setattr(assistant._prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
        cli = assistant.InteractiveCLI(model=core.MODELS[0].short)
        cli.cmd_assistant()
        return sent

    def test_a_turn_returns_the_model_reply(self, monkeypatch, capsys) -> None:
        sent = self._run(monkeypatch, ["what is up"])
        out = capsys.readouterr().out
        assert sent == [[{"role": "user", "content": "what is up"}]]
        assert "a real answer" in out
        assert "Response generated" not in out

    def test_history_accumulates_across_turns(self, monkeypatch) -> None:
        sent = self._run(monkeypatch, ["one", "two"])
        assert [m["content"] for m in sent[-1]] == ["one", "a real answer", "two"]

    def test_reset_clears_history(self, monkeypatch) -> None:
        sent = self._run(monkeypatch, ["one", "/reset", "two"])
        assert sent[-1] == [{"role": "user", "content": "two"}]

    def test_save_writes_the_transcript(self, monkeypatch, tmp_path) -> None:
        target = tmp_path / "t.json"
        self._run(monkeypatch, ["hello", f"/save {target}"])
        assert json.loads(target.read_text(encoding="utf-8")) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "a real answer"},
        ]

    def test_failed_turn_is_reported_and_not_recorded(self, monkeypatch, capsys) -> None:
        import hypernix.interfaces.hyped_pro_core as core

        calls = {"n": 0}

        def flaky(_model, messages, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise core.HypedProError("HPC-CLOUD-001", "no API key configured")
            return {"content": "second time lucky", "thinking": None}

        monkeypatch.setattr(core, "send_chat_message", flaky)
        answers = iter(["first", "second", "/quit"])
        monkeypatch.setattr(assistant._prompt, "ask", staticmethod(lambda *a, **k: next(answers)))
        assistant.InteractiveCLI(model=core.MODELS[0].short).cmd_assistant()

        out = capsys.readouterr().out
        assert "no API key configured" in out
        assert "second time lucky" in out

    def test_no_model_configured_says_so_instead_of_guessing(
        self, monkeypatch, capsys,
    ) -> None:
        import hypernix.system.config as config

        monkeypatch.setattr(config, "get_default_model", lambda: None)
        monkeypatch.setattr(config, "get_provider_key", lambda _vendor: None)
        monkeypatch.setattr(assistant._prompt, "ask", staticmethod(lambda *a, **k: "/quit"))

        assistant.InteractiveCLI().cmd_assistant()

        assert "No model selected" in capsys.readouterr().out


def _emittable_strings(path: Path) -> list[str]:
    """Every string literal in ``path`` that isn't a docstring.

    Docstrings are excluded so this module's own explanation of what it
    used to fake doesn't trip the guard below.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_canned_success_strings_remain() -> None:
    """Regression guard for the specific fabrications that used to ship."""
    banned = [
        "Response generated",
        "All services operational",
        "Server started on",
        "Tailscale integration active",
        "demo assistant",
        "Training module ready",
        "Voice mode enabled. Speak now",
    ]
    for name in ("assistant.py", "cli.py"):
        literals = _emittable_strings(SRC / "hypernix" / "interfaces" / name)
        for phrase in banned:
            offenders = [s for s in literals if phrase in s]
            assert offenders == [], f"{name} still claims: {offenders!r}"
