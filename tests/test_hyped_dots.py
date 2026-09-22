"""hyped, and the dots that configure it.

Two promises that normally contradict each other:

* it works with **no** configuration, and
* **everything** about it can be changed.

They resolve because the defaults live in code and a dot overrides
them, so an empty dot and no dot do the same thing. Most of these tests
are one or other half of that.

The security-shaped one is `dot_paths`: a project-local `.hyped.py`
would be the obvious feature and it is the one that turns cloning a
repository into running its code. It is deliberately absent, and there
is a test that it stays absent.
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pytest

from hypernix.interfaces.dots import (
    Config,
    DotError,
    dot_paths,
    load,
    load_dot,
)
from hypernix.interfaces.hyped_basic import (
    Backend,
    Paint,
    Session,
    _hex_to_ansi,
    models_dir,
)


def dot(tmp_path: Path, body: str, name: str = "hyped.py") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Nothing required
# ---------------------------------------------------------------------------


class TestItWorksWithNothing:
    def test_a_bare_config_is_usable(self):
        config = Config()
        assert config.temperature > 0
        assert config.max_tokens > 0
        assert config.theme.accent

    def test_no_dots_anywhere_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv("HYPED_DOT", raising=False)
        config = load()
        assert config.sources == []
        assert isinstance(config, Config)

    def test_an_empty_dot_and_no_dot_agree(self, tmp_path, monkeypatch):
        """The sentence the whole design rests on."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv("HYPED_DOT", raising=False)
        without = load().to_dict()

        dot(tmp_path / "hypernix", "")
        with_empty = load().to_dict()

        without.pop("sources", None)
        with_empty.pop("sources", None)
        assert without == with_empty

    def test_a_session_with_nowhere_to_talk_says_what_to_do(self):
        from hypernix.interfaces.hyped_basic import discover_backend

        # Port 1 is nothing, and the models directory is empty in CI,
        # so this is the genuine "there is nowhere to talk" message.
        message = discover_backend(Config(server="http://127.0.0.1:1",
                                          models_dir="/nonexistent"),
                                   timeout=0.05).detail
        assert "hypernix-t1 start" in message
        assert "/search" in message


# ---------------------------------------------------------------------------
# Everything configurable
# ---------------------------------------------------------------------------


class TestDots:
    def test_a_dot_sets_a_value(self, tmp_path):
        config = Config()
        load_dot(dot(tmp_path, 'config.model = "qwen3-8b"'), config)
        assert config.model == "qwen3-8b"

    def test_a_dot_can_reach_into_the_theme(self, tmp_path):
        config = Config()
        load_dot(dot(tmp_path, 'config.theme.accent = "#00ff00"'), config)
        assert config.theme.accent == "#00ff00"

    def test_a_dot_can_add_a_command(self, tmp_path):
        config = Config()
        load_dot(dot(tmp_path, '''
            @config.command("hi")
            def hi(argument):
                return "hello " + argument
        '''), config)
        assert config.commands["hi"]("world") == "hello world"

    def test_a_command_name_is_cleaned(self, tmp_path):
        config = Config()
        load_dot(dot(tmp_path, '''
            @config.command("/spaced ")
            def f(a):
                return a
        '''), config)
        assert "spaced" in config.commands

    def test_a_nameless_command_is_refused(self):
        with pytest.raises(DotError):
            Config().command("  ")

    def test_a_hook_changes_one_turn_only(self, tmp_path):
        """A hook that switches to a bigger model for one long prompt
        must not switch the session. The first design mutated the
        config and produced a session that drifted onto a bigger model
        and stayed there, with nothing saying it had."""
        config = Config(model="small")
        load_dot(dot(tmp_path, '''
            @config.on_prompt
            def bigger(text, settings):
                if len(text) > 10:
                    settings.model = "big"
        '''), config)

        assert config.settings_for("hi").model == "small"
        assert config.settings_for("x" * 40).model == "big"
        assert config.model == "small", "the session drifted"

    def test_a_broken_hook_does_not_end_the_session(self, tmp_path):
        config = Config(model="small")
        load_dot(dot(tmp_path, '''
            @config.on_prompt
            def broken(text, settings):
                raise RuntimeError("oops")
        '''), config)
        assert config.settings_for("hi").model == "small"

    def test_a_reply_hook_runs(self, tmp_path):
        config = Config()
        load_dot(dot(tmp_path, '''
            seen = []
            @config.on_reply
            def remember(text):
                seen.append(text)
            config.commands["seen"] = lambda a: len(seen)
        '''), config)
        config.announce_reply("hello")
        assert config.commands["seen"]("") == 1

    def test_a_broken_dot_names_the_file(self, tmp_path):
        """Rather than a traceback through this module that looks like
        hyped is broken."""
        path = dot(tmp_path, "this is not python")
        with pytest.raises(DotError, match=str(path.name)):
            load_dot(path, Config())

    def test_a_dot_that_raises_names_the_exception(self, tmp_path):
        path = dot(tmp_path, 'raise ValueError("deliberate")')
        with pytest.raises(DotError, match="deliberate"):
            load_dot(path, Config())

    def test_an_explicit_missing_dot_is_an_error(self, tmp_path):
        """Asked for by name and absent is a typo, not a default."""
        with pytest.raises(DotError, match="no such file"):
            load(tmp_path / "absent.py")

    def test_a_failing_dot_is_reported_not_swallowed(self, tmp_path, monkeypatch, caplog):
        """Silently falling back to defaults is how somebody spends an
        afternoon wondering why their theme does nothing."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv("HYPED_DOT", raising=False)
        dot(tmp_path / "hypernix", "raise ValueError('bad')")
        config = load()
        assert any("failed" in source for source in config.sources)


class TestWhereDotsComeFrom:
    def test_the_working_directory_is_never_searched(self, tmp_path, monkeypatch):
        """The feature that turns `git clone` into code execution."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv("HYPED_DOT", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".hyped.py").write_text("config.model = 'evil'")
        (tmp_path / "hyped.py").write_text("config.model = 'evil'")

        paths = dot_paths()
        assert all(tmp_path not in p.parents or "hypernix" in p.parts
                   for p in paths), paths
        assert load().model != "evil"

    def test_conf_d_loads_in_filename_order(self, tmp_path, monkeypatch):
        """So ordering is a filename rather than a setting somewhere
        else."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv("HYPED_DOT", raising=False)
        base = tmp_path / "hypernix" / "hyped.d"
        dot(base, 'config.model = "first"', "10-a.py")
        dot(base, 'config.model = "second"', "90-z.py")
        assert load().model == "second"

    def test_an_env_var_can_name_one(self, tmp_path, monkeypatch):
        path = dot(tmp_path, 'config.model = "from-env"')
        monkeypatch.setenv("HYPED_DOT", str(path))
        assert load().model == "from-env"

    def test_explicit_beats_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPED_DOT", str(dot(tmp_path, 'config.model="env"')))
        chosen = dot(tmp_path, 'config.model = "explicit"', "other.py")
        assert load(chosen).model == "explicit"


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class TestSession:
    @pytest.fixture
    def session(self):
        return Session(Config(), backend=Backend("none", detail="nothing here"),
                       out=io.StringIO())

    def test_a_plain_line_is_not_a_command(self, session):
        assert session.handle_command("hello there") is False

    def test_an_unknown_command_says_so(self, session):
        assert session.handle_command("/nonsense") is True
        assert "/help" in session.out.getvalue()

    def test_quit_stops_the_loop(self, session):
        session.handle_command("/quit")
        assert session.running is False

    def test_a_dot_command_cannot_shadow_quit(self):
        """Sounds paranoid until somebody writes a command called
        `exit` that does something else."""
        config = Config()
        config.commands["quit"] = lambda a: "not today"
        session = Session(config, backend=Backend("none"), out=io.StringIO())
        session.handle_command("/quit")
        assert session.running is False

    def test_a_dot_command_runs(self):
        config = Config()
        config.commands["double"] = lambda a: a * 2
        session = Session(config, backend=Backend("none"), out=io.StringIO())
        session.handle_command("/double ab")
        assert "abab" in session.out.getvalue()

    def test_a_broken_dot_command_does_not_end_the_session(self):
        config = Config()

        def broken(_argument):
            raise RuntimeError("oops")

        config.commands["broken"] = broken
        session = Session(config, backend=Backend("none"), out=io.StringIO())
        assert session.handle_command("/broken") is True
        assert session.running is True
        assert "oops" in session.out.getvalue()

    def test_config_never_prints_a_token(self):
        """`/config` is what people run while screen-sharing to ask why
        something is not working."""
        config = Config(token="secret-token", hf_token="hf_secret")
        session = Session(config, backend=Backend("none"), out=io.StringIO())
        session.cmd_config()
        printed = session.out.getvalue()
        assert "secret-token" not in printed
        assert "hf_secret" not in printed
        assert "(set)" in printed

    def test_config_says_when_nothing_is_configured(self, session):
        session.cmd_config()
        assert "no dots loaded" in session.out.getvalue()

    def test_a_failed_turn_does_not_leave_the_message_on_the_stack(self):
        """So retrying does not send it twice."""
        session = Session(Config(), backend=Backend("t1", base_url="http://127.0.0.1:1"),
                          out=io.StringIO())
        session.send("hello")
        assert session.messages == []

    def test_help_lists_the_dot_commands_too(self):
        config = Config()
        config.commands["weather"] = lambda a: "sunny"
        session = Session(config, backend=Backend("none"), out=io.StringIO())
        session.cmd_help()
        assert "/weather" in session.out.getvalue()


class TestColour:
    def test_a_hex_becomes_truecolor(self):
        assert _hex_to_ansi("#c8192e") == "\033[38;2;200;25;46m"

    def test_nonsense_is_left_alone(self):
        assert _hex_to_ansi("not a colour") == ""
        assert _hex_to_ansi("#zzzzzz") == ""

    def test_colour_is_off_when_not_a_terminal(self):
        """Escape codes in a pipe are noise in somebody's log."""
        assert Paint(Config(), io.StringIO()).enabled is False

    def test_no_color_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")

        class Terminal(io.StringIO):
            def isatty(self):
                return True

        assert Paint(Config(), Terminal()).enabled is False


class TestModelsDirectory:
    def test_it_defaults_under_the_home_directory(self):
        assert models_dir(Config()).parts[-2:] == (".hypernix", "models")

    def test_a_dot_can_move_it(self, tmp_path):
        assert models_dir(Config(models_dir=str(tmp_path))) == tmp_path


class TestTheEntryPoints:
    def test_hyped_points_at_the_simple_one(self):
        import tomllib

        root = Path(__file__).resolve().parent.parent
        scripts = tomllib.loads(
            (root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["scripts"]
        assert scripts["hyped"].startswith("hypernix.interfaces.hyped_basic")
        # And the agentic one is still reachable rather than deleted.
        assert scripts["hyped-agent"].startswith("hypernix.interfaces.hyped:")
