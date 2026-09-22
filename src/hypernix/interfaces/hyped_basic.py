"""hyped — a local AI you can talk to, and nothing you have to set up.

Two promises that usually contradict each other:

**It works with no configuration.** Run `hyped`. It finds a model — the
T1 server on this machine, or a GGUF in the models directory — and you
type at it. There is no wizard, no first-run questionnaire, no file to
create. If there is nothing to talk to it says which of the two things
is missing and how to get one.

**Everything about it can be changed.** Not through a settings menu,
through :mod:`hypernix.interfaces.dots` — a Python file that gets the
config object and modifies it. Colours, the model, the system prompt,
the commands, and the behaviour: which model to use for which kind of
question is a function, not a string, because that is what it actually
is.

Those resolve because the defaults live in code. An empty dot file and
no dot file do the same thing, so "works with nothing" and "everything
is configurable" are the same statement read from either end.

What this is not
----------------
Not an agent. It does not read your files, run commands, or edit code.
That is `hyped-pro`, which is a different program with a different
threat model, and the split is the point: a tool that can only talk is
a tool you can run against an untrusted model without thinking about
it.

What it does have beyond chat is finding models: `/search` asks Hugging
Face, `/download` fetches a GGUF, `/models` lists what is here. That is
the one thing a local-model chat window genuinely cannot do without,
because the first question is always "what can I run".
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dots import Config, DotError, Settings, load

__all__ = ["main", "cli_main", "Session", "Backend", "search_huggingface"]

HF_API = "https://huggingface.co/api/models"


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def _hex_to_ansi(value: str) -> str:
    """A #rrggbb into a 24-bit SGR prefix, or '' for anything else.

    Truecolor rather than a 256-colour approximation: every terminal
    worth configuring a theme in has had it for years, and the
    approximation makes a carefully-chosen accent into a different
    colour.
    """
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return ""
    try:
        red, green, blue = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return ""
    return f"\033[38;2;{red};{green};{blue}m"


class Paint:
    """Colour, or not.

    Off when the config says so, when ``NO_COLOR`` is set (the
    convention), or when stdout is not a terminal — escape codes in a
    pipe are noise in somebody's log file.
    """

    RESET = "\033[0m"

    def __init__(self, config: Config, stream=None):
        stream = stream or sys.stdout
        self.enabled = (
            config.theme.colour
            and not os.environ.get("NO_COLOR")
            and hasattr(stream, "isatty")
            and stream.isatty()
        )
        self._theme = config.theme

    def __call__(self, text: str, colour: str) -> str:
        if not self.enabled:
            return text
        prefix = _hex_to_ansi(colour)
        return f"{prefix}{text}{self.RESET}" if prefix else text

    def accent(self, text: str) -> str:
        return self(text, self._theme.accent)

    def dim(self, text: str) -> str:
        return self(text, self._theme.dim)

    def ok(self, text: str) -> str:
        return self(text, self._theme.ok)

    def warn(self, text: str) -> str:
        return self(text, self._theme.warn)


# ---------------------------------------------------------------------------
# Talking to something
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """Where the words come from, and what to say if there is nowhere."""

    kind: str          # "t1" | "local" | "none"
    base_url: str = ""
    token: str = ""
    model: str = ""
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.kind != "none"


def discover_backend(config: Config, *, timeout: float = 2.0) -> Backend:
    """A T1 server on this machine, or a local GGUF, or neither.

    The server first, deliberately: if one is running it already has a
    model loaded and the GPU, and starting a second llama.cpp beside it
    takes the VRAM the first one is using.
    """
    url = (config.server or os.environ.get("T1_URL", "")
           or "http://127.0.0.1:8000").rstrip("/")
    try:
        request = urllib.request.Request(f"{url}/health", method="GET")
        if config.token:
            request.add_header("Authorization", f"Bearer {config.token}")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status < 400:
                return Backend("t1", base_url=url, token=config.token,
                               model=config.model,
                               detail=f"T1 server at {url}")
    except (urllib.error.URLError, OSError, ValueError):
        pass          # not running, or not ours. Fall through quietly.

    local = _first_local_model(config)
    if local is not None:
        return Backend("local", model=str(local),
                       detail=f"local model {local.name}")

    return Backend(
        "none",
        detail=(
            "Nothing to talk to yet. Either:\n"
            "  start a server   hypernix-t1 start\n"
            "  or get a model   /search qwen3   then   /download <repo> <file>"
        ),
    )


def models_dir(config: Config) -> Path:
    if config.models_dir:
        return Path(config.models_dir).expanduser()
    return Path.home() / ".hypernix" / "models"


def _first_local_model(config: Config) -> Path | None:
    directory = models_dir(config)
    if not directory.is_dir():
        return None
    found = sorted(directory.rglob("*.gguf"))
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


def search_huggingface(
    query: str, *, token: str = "", limit: int = 20, timeout: float = 15.0
) -> list[dict[str, Any]]:
    """GGUF repositories matching *query*, most downloaded first.

    Filtered to `gguf` at the API rather than after: the unfiltered
    search returns mostly things this cannot run, and a list where four
    of twenty entries are usable is a list people stop reading.
    """
    parameters = urllib.parse.urlencode({
        "search": query,
        "filter": "gguf",
        "sort": "downloads",
        "direction": -1,
        "limit": max(1, min(limit, 100)),
    })
    request = urllib.request.Request(f"{HF_API}?{parameters}")
    request.add_header("Accept", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read() or b"[]")
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError(
                "Hugging Face refused the search. Set HF_TOKEN, or "
                "`config.hf_token` in your dot, if you are looking for "
                "gated repositories."
            ) from error
        raise RuntimeError(f"Hugging Face returned {error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(f"Could not reach Hugging Face: {error}") from error

    return [
        {
            "id": item.get("modelId") or item.get("id") or "?",
            "downloads": item.get("downloads") or 0,
            "likes": item.get("likes") or 0,
        }
        for item in payload
        if isinstance(item, dict)
    ]


def list_gguf_files(
    repo: str, *, token: str = "", timeout: float = 15.0
) -> list[str]:
    """The .gguf files in a repository, smallest name first."""
    request = urllib.request.Request(f"{HF_API}/{urllib.parse.quote(repo)}")
    request.add_header("Accept", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise RuntimeError(f"Could not read {repo}: {error}") from error

    return sorted(
        entry.get("rfilename", "")
        for entry in payload.get("siblings") or []
        if str(entry.get("rfilename", "")).endswith(".gguf")
    )


def download_gguf(
    repo: str,
    filename: str,
    destination: Path,
    *,
    token: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch one file. Returns where it landed.

    Written to a `.part` beside the target and renamed at the end, so an
    interrupted download cannot leave something that looks like a model
    and is half a model — which fails at load with a message about the
    GGUF magic number rather than about the download.
    """
    url = (f"https://huggingface.co/{urllib.parse.quote(repo)}/resolve/main/"
           f"{urllib.parse.quote(filename)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with partial.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)
    partial.replace(destination)
    return destination


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class Session:
    """One conversation, and the commands that act on it."""

    def __init__(self, config: Config, backend: Backend | None = None,
                 out=None):
        self.config = config
        self.out = out or sys.stdout
        self.paint = Paint(config, self.out)
        self.backend = backend if backend is not None else discover_backend(config)
        self.messages: list[dict[str, str]] = []
        self.running = True

    # -- output ------------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def warn(self, text: str) -> None:
        print(self.paint.warn(text), file=self.out)

    # -- commands ----------------------------------------------------

    def builtin_commands(self) -> dict[str, Callable[[str], None]]:
        return {
            "help": self.cmd_help,
            "quit": self.cmd_quit,
            "exit": self.cmd_quit,
            "clear": self.cmd_clear,
            "config": self.cmd_config,
            "models": self.cmd_models,
            "search": self.cmd_search,
            "download": self.cmd_download,
            "model": self.cmd_model,
            "where": self.cmd_where,
        }

    def handle_command(self, line: str) -> bool:
        """True if *line* was a command. False means it is a message."""
        if not line.startswith("/"):
            return False
        name, _, argument = line[1:].partition(" ")
        name = name.strip().lower()

        builtin = self.builtin_commands()
        if name in builtin:
            builtin[name](argument.strip())
            return True

        # A dot's command. Looked up after the built-ins so a dot cannot
        # shadow /quit -- which sounds paranoid until somebody writes a
        # command called `exit` that does something else.
        custom = self.config.commands.get(name)
        if custom is not None:
            try:
                result = custom(argument.strip())
            except Exception as exc:  # noqa: BLE001
                self.warn(f"/{name} failed: {type(exc).__name__}: {exc}")
                return True
            if result is not None:
                self.say(str(result))
            return True

        self.warn(f"No command /{name}. Try /help.")
        return True

    def cmd_help(self, _argument: str = "") -> None:
        self.say(self.paint.accent("hyped") + " — talk to a local model")
        self.say()
        for name, description in (
            ("/models", "what is on this machine"),
            ("/search <text>", "find GGUF models on Hugging Face"),
            ("/download <repo> [file]", "fetch one"),
            ("/model <name>", "use a different model"),
            ("/where", "what this is talking to"),
            ("/config", "every setting, and which dot set it"),
            ("/clear", "forget the conversation"),
            ("/quit", "leave"),
        ):
            self.say(f"  {self.paint.accent(name):<34} {description}")
        if self.config.commands:
            self.say()
            self.say("  from your dots:")
            for name in sorted(self.config.commands):
                self.say(f"    {self.paint.accent('/' + name)}")

    def cmd_quit(self, _argument: str = "") -> None:
        self.running = False

    def cmd_clear(self, _argument: str = "") -> None:
        self.messages.clear()
        self.say(self.paint.dim("conversation cleared"))

    def cmd_where(self, _argument: str = "") -> None:
        self.say(self.backend.detail or self.backend.kind)

    def cmd_config(self, _argument: str = "") -> None:
        self.say(json.dumps(self.config.to_dict(), indent=2, default=str))
        if self.config.sources:
            self.say()
            self.say(self.paint.dim("dots: " + ", ".join(self.config.sources)))
        else:
            self.say(self.paint.dim(
                "no dots loaded — this is all defaults. See "
                "`hypernix.interfaces.dots` for how to change any of it."
            ))

    def cmd_model(self, argument: str) -> None:
        if not argument:
            self.say(self.config.model or "(whatever the server has loaded)")
            return
        self.config.model = argument
        self.say(self.paint.ok(f"model is now {argument}"))

    def cmd_models(self, _argument: str = "") -> None:
        directory = models_dir(self.config)
        found = sorted(directory.rglob("*.gguf")) if directory.is_dir() else []
        if not found:
            self.say(self.paint.dim(f"nothing in {directory}"))
            self.say("Try /search qwen3 to find one.")
            return
        for path in found:
            size = path.stat().st_size / 1e9
            self.say(f"  {path.name}  {self.paint.dim(f'{size:.1f} GB')}")

    def cmd_search(self, argument: str) -> None:
        if not argument:
            self.warn("/search needs something to search for")
            return
        try:
            results = search_huggingface(
                argument,
                token=self.config.hf_token,
                limit=self.config.hf_search_limit,
            )
        except RuntimeError as exc:
            self.warn(str(exc))
            return
        if not results:
            self.say(self.paint.dim(f"nothing on Hugging Face for {argument!r}"))
            return
        for item in results:
            self.say(
                f"  {self.paint.accent(item['id'])}  "
                + self.paint.dim(f"{item['downloads']:,} downloads")
            )
        self.say()
        self.say(self.paint.dim("/download <repo> to see its files"))

    def cmd_download(self, argument: str) -> None:
        parts = argument.split()
        if not parts:
            self.warn("/download needs a repository, e.g. /download Qwen/Qwen3-8B-GGUF")
            return
        repo = parts[0]
        try:
            files = list_gguf_files(repo, token=self.config.hf_token)
        except RuntimeError as exc:
            self.warn(str(exc))
            return
        if not files:
            self.warn(f"{repo} has no .gguf files")
            return

        if len(parts) == 1:
            self.say(f"{repo} has:")
            for name in files:
                self.say(f"  {name}")
            self.say()
            self.say(self.paint.dim(f"/download {repo} <file> to fetch one"))
            return

        wanted = parts[1]
        if wanted not in files:
            self.warn(f"{repo} has no {wanted}. It has: {', '.join(files[:8])}")
            return

        target = models_dir(self.config) / repo.replace("/", "_") / wanted
        if self.config.confirm_downloads:
            self.say(f"Fetching {wanted} into {target.parent}")
        try:
            download_gguf(
                repo, wanted, target,
                token=self.config.hf_token,
                on_progress=self._progress,
            )
        except Exception as exc:  # noqa: BLE001
            self.warn(f"download failed: {exc}")
            return
        self.say()
        self.say(self.paint.ok(f"saved {target}"))

    def _progress(self, done: int, total: int) -> None:
        if not total:
            return
        share = done / total
        width = 28
        filled = int(width * share)
        bar = "█" * filled + "·" * (width - filled)
        print(f"\r  {bar} {share:5.1%}", end="", file=self.out, flush=True)

    # -- a turn ------------------------------------------------------

    def send(self, text: str) -> str:
        """One exchange. Returns the reply."""
        if not self.backend.available:
            return self.backend.detail

        settings = self.config.settings_for(text)
        self.messages.append({"role": "user", "content": text})
        try:
            reply = self._ask(settings)
        except Exception as exc:  # noqa: BLE001
            # The conversation is not poisoned by a failed turn: the
            # user's message comes back off the stack so retrying does
            # not send it twice.
            self.messages.pop()
            return self.paint.warn(f"could not answer: {exc}")

        self.messages.append({"role": "assistant", "content": reply})
        self.config.announce_reply(reply)
        return reply

    def _ask(self, settings: Settings) -> str:
        if self.backend.kind != "t1":
            raise RuntimeError(
                "a local GGUF is present but this build cannot run it "
                "directly — start a server with `hypernix-t1 start` and it "
                "will be used"
            )
        body: dict[str, Any] = {
            "messages": (
                ([{"role": "system", "content": settings.system}]
                 if settings.system else [])
                + self.messages
            ),
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }
        if settings.model:
            body["model"] = settings.model

        request = urllib.request.Request(
            f"{self.backend.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        if self.backend.token:
            request.add_header("Authorization", f"Bearer {self.backend.token}")
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read() or b"{}")
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("the server returned no reply")
        return str(choices[0].get("message", {}).get("content", ""))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def banner(session: Session) -> Iterator[str]:
    paint = session.paint
    yield paint.accent("hyped") + paint.dim("  ·  local, and yours")
    yield paint.dim(session.backend.detail)
    if not session.config.sources:
        yield paint.dim("no dots loaded — /config shows what you can change")
    yield ""


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="hyped",
        description="Talk to a local model. No setup; everything configurable.",
    )
    parser.add_argument("--dot", default="",
                        help="a specific config file to load instead of the usual ones")
    parser.add_argument("--no-dots", action="store_true",
                        help="ignore every dot — the bare defaults")
    parser.add_argument("--model", default="")
    parser.add_argument("--server", default="")
    parser.add_argument("-c", "--command", default="",
                        help="run one message and exit")
    args = parser.parse_args(argv)

    try:
        config = Config() if args.no_dots else load(args.dot or None)
    except DotError as exc:
        print(f"hyped: {exc}", file=sys.stderr)
        return 1

    # The command line wins over a dot, which wins over the defaults.
    # Anything else makes `--model` mysteriously not work for the one
    # person who has a dot that sets it.
    if args.model:
        config.model = args.model
    if args.server:
        config.server = args.server

    session = Session(config)

    if args.command:
        print(session.send(args.command))
        return 0 if session.backend.available else 1

    if config.banner:
        for line in banner(session):
            session.say(line)

    try:
        import readline  # noqa: F401
    except ImportError:
        # No line editing. Everything still works; arrow keys print
        # escape codes, which is annoying and not worth failing over.
        pass

    while session.running:
        try:
            line = input(session.paint.accent(config.theme.prompt))
        except (EOFError, KeyboardInterrupt):
            session.say()
            break
        line = line.strip()
        if not line:
            continue
        if session.handle_command(line):
            continue
        session.say(session.send(line))
        session.say()
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
