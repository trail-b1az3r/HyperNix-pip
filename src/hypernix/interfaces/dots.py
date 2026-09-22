"""interfaces.dots — configuration as Python, with nothing required.

The rule this exists to enforce: **hyped works with no configuration at
all**, and everything about it can be changed. Those pull against each
other in the usual way — a program with a hundred settings has a
hundred defaults somebody has to be told about — and the resolution
here is that the defaults live in code and a dot file *overrides* them,
so an empty config and no config are the same thing.

Why Python and not TOML
-----------------------
TOML was the obvious choice and it is wrong for this. Half of what
people want to configure here is behaviour, not values: which model to
pick for a given prompt, what the prompt should say today, which
commands the assistant may run in this directory. A data format
expresses that as a string somebody has to parse, which is a worse
programming language than the one already present.

So a dot is a Python file that gets a `config` object and mutates it.
It is not imported as a module and it does not get the package's
namespace — see :func:`load_dot`.

    # ~/.config/hypernix/hyped.py
    config.model = "qwen3-8b"
    config.theme.accent = "#c8192e"

    @config.on_prompt
    def bigger_model_for_long_questions(text, settings):
        if len(text) > 400:
            settings.model = "qwen3-32b"

What a dot cannot do
--------------------
A dot is code the user wrote, running as the user, so this is not a
sandbox and does not pretend to be one — it can do anything the user
can. What it *cannot* do is arrive by accident: dots are read from a
fixed list of paths under the user's own config directory, never from
the working directory, and never from anything a model or a download
put there. A project-local `.hyped.py` would be the obvious feature and
it is the one that turns cloning a repository into running its code.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Theme",
    "Settings",
    "Config",
    "dot_paths",
    "load_dot",
    "load",
    "DotError",
]


class DotError(Exception):
    """A dot file failed. Carries the path so the message can name it."""


@dataclass
class Theme:
    """Colours, as ANSI or hex. Everything has a default that works."""

    accent: str = "#c8192e"
    text: str = "#f2f2f0"
    dim: str = "#9aa0a6"
    ok: str = "#3fa46a"
    warn: str = "#e0a030"
    #: Printed before the input line. Kept short on purpose.
    prompt: str = "› "
    #: False strips every colour. Also forced off when stdout is not a
    #: terminal, because escape codes in a pipe are noise in a log.
    colour: bool = True


@dataclass
class Settings:
    """What a single turn runs with.

    Separate from :class:`Config` because a hook gets a *copy* of this
    to modify, so a hook that changes the model for one long prompt
    does not change it for the rest of the session. That was the first
    design and it produced a session that silently drifted onto a
    bigger model and stayed there.
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    system: str = ""
    #: Where to talk to. Empty means "the local runner, or ask".
    server: str = ""


@dataclass
class Config:
    """Everything hyped reads. Every field has a working default."""

    # -- what it talks to --------------------------------------------
    model: str = ""
    server: str = ""
    token: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    system: str = ""

    # -- Hugging Face ------------------------------------------------
    #: Read from the environment when unset, because that is where it
    #: already is for anybody who has used huggingface-cli.
    hf_token: str = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    )
    hf_search_limit: int = 20
    models_dir: str = ""

    # -- appearance ---------------------------------------------------
    theme: Theme = field(default_factory=Theme)
    banner: bool = True
    #: Lines of history kept on screen. Not the context window.
    scrollback: int = 2000

    # -- behaviour ----------------------------------------------------
    stream: bool = True
    save_history: bool = True
    confirm_downloads: bool = True

    # -- hooks --------------------------------------------------------
    #: Called with (text, settings) before each turn. A hook mutates
    #: the settings copy it is given; the return value is ignored, so
    #: a hook that forgets to return does not silently do nothing.
    _prompt_hooks: list[Callable[[str, Settings], None]] = field(
        default_factory=list, repr=False
    )
    #: Called with (text) after each reply.
    _reply_hooks: list[Callable[[str], None]] = field(
        default_factory=list, repr=False
    )
    #: Extra slash commands: name -> callable(argument) -> str | None.
    commands: dict[str, Callable[[str], Any]] = field(default_factory=dict)

    #: Filled by :func:`load` with the dots that were read, so `/config`
    #: can say where a surprising setting came from.
    sources: list[str] = field(default_factory=list, repr=False)

    # -- the decorators a dot uses -----------------------------------

    def on_prompt(self, function: Callable[[str, Settings], None]):
        """Register a hook that runs before each turn."""
        self._prompt_hooks.append(function)
        return function

    def on_reply(self, function: Callable[[str], None]):
        self._reply_hooks.append(function)
        return function

    def command(self, name: str):
        """Register a slash command. ``@config.command("weather")``."""
        clean = name.strip().lstrip("/")
        if not clean:
            raise DotError("a command needs a name")

        def register(function: Callable[[str], Any]):
            self.commands[clean] = function
            return function

        return register

    # -- what a turn actually runs with ------------------------------

    def settings_for(self, text: str) -> Settings:
        """The settings for one turn, after the hooks have had a look.

        A *copy*, so a hook that switches to a bigger model for one long
        prompt does not switch the session. The first version mutated
        the config and produced a session that drifted onto a bigger
        model and stayed there, with nothing saying it had.
        """
        settings = Settings(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=self.system,
            server=self.server,
        )
        for hook in self._prompt_hooks:
            try:
                hook(text, settings)
            except Exception as exc:  # noqa: BLE001
                # A broken hook must not end the session. It is the
                # user's own code and they are sitting right there, so
                # it is reported rather than swallowed.
                logger.warning("dots: prompt hook %s failed: %s",
                               getattr(hook, "__name__", "?"), exc)
        return settings

    def announce_reply(self, text: str) -> None:
        for hook in self._reply_hooks:
            try:
                hook(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dots: reply hook %s failed: %s",
                               getattr(hook, "__name__", "?"), exc)

    def to_dict(self) -> dict[str, Any]:
        """The plain values, for `/config` and for tests.

        Hooks and commands are summarised rather than serialised: a
        function has no useful representation here and printing
        `<function foo at 0x...>` tells the reader nothing about
        whether their dot loaded.
        """
        out: dict[str, Any] = {}
        for entry in dataclasses.fields(self):
            if entry.name.startswith("_"):
                continue
            value = getattr(self, entry.name)
            if dataclasses.is_dataclass(value):
                out[entry.name] = dataclasses.asdict(value)
            elif entry.name == "commands":
                out[entry.name] = sorted(value)
            else:
                out[entry.name] = value
        out["prompt_hooks"] = [
            getattr(h, "__name__", "?") for h in self._prompt_hooks
        ]
        out["reply_hooks"] = [
            getattr(h, "__name__", "?") for h in self._reply_hooks
        ]
        # Never print the tokens. `/config` is the command people run
        # when screen-sharing to ask why something is not working.
        for secret in ("token", "hf_token"):
            if out.get(secret):
                out[secret] = "(set)"
        return out


# ---------------------------------------------------------------------------
# Finding and running dots
# ---------------------------------------------------------------------------


def dot_paths(explicit: str | Path | None = None) -> list[Path]:
    """Where dots are looked for, in load order.

    Deliberately *not* the working directory. A project-local
    `.hyped.py` is the obvious feature and it is the one that turns
    cloning a repository into running its code — so it is not here, and
    `--dot` is the way to load one on purpose.

    ``HYPED_DOT`` names one file and wins over the directory scan, for
    a script that wants a known configuration without touching the
    user's own.
    """
    if explicit:
        return [Path(explicit).expanduser()]

    from_env = os.environ.get("HYPED_DOT", "").strip()
    if from_env:
        return [Path(from_env).expanduser()]

    base = Path(
        os.environ.get("XDG_CONFIG_HOME", "") or (Path.home() / ".config")
    ).expanduser() / "hypernix"

    found: list[Path] = [base / "hyped.py"]
    # Then everything in conf.d, sorted, so `10-theme.py` lands before
    # `90-overrides.py` and the ordering is a filename rather than a
    # setting somewhere else.
    directory = base / "hyped.d"
    if directory.is_dir():
        found.extend(sorted(p for p in directory.glob("*.py") if p.is_file()))
    return found


def load_dot(path: Path, config: Config) -> None:
    """Run one dot against *config*.

    Executed with :func:`exec` in a namespace holding exactly `config`,
    `Settings`, `Theme` and `__file__` — not imported as a module.
    Importing would put it on `sys.modules` under a name that collides
    with anything, cache it so a second load is a no-op, and give it
    the package's `__name__`; none of which a config file wants.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DotError(f"{path}: cannot read it — {exc}") from exc

    namespace: dict[str, Any] = {
        "config": config,
        "Settings": Settings,
        "Theme": Theme,
        "__file__": str(path),
        "__name__": "hyped_dot",
    }
    try:
        exec(compile(source, str(path), "exec"), namespace)   # noqa: S102
    except Exception as exc:  # noqa: BLE001
        # Named with the file and the line, because the alternative is a
        # traceback through this module that looks like hyped is broken.
        raise DotError(f"{path}: {type(exc).__name__}: {exc}") from exc


def load(
    explicit: str | Path | None = None, *, strict: bool = False
) -> Config:
    """The config, after every dot that exists has had a go.

    A missing dot is not an error — that is the whole "works with no
    configuration" promise. A dot that exists and *fails* is reported:
    silently falling back to defaults when somebody's config has a typo
    is how a person spends an afternoon wondering why their theme does
    nothing.
    """
    config = Config()
    for path in dot_paths(explicit):
        if not path.is_file():
            if explicit:
                raise DotError(f"{path}: no such file")
            continue
        try:
            load_dot(path, config)
        except DotError:
            if strict:
                raise
            logger.error("%s", _last_dot_error(path))
            config.sources.append(f"{path} (failed)")
            continue
        config.sources.append(str(path))
    return config


def _last_dot_error(path: Path) -> str:
    return f"dots: {path} failed to load; its settings were not applied"
