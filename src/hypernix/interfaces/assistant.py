"""assistant — the interactive HyperNix menu (``hypernix assistant``).

A thin front end over the modules that do the actual work: ``download``
for the model registry, ``instant_pot`` / ``recipe_book`` for training,
``workshop`` for ASR/TTS, ``hyped_pro_core`` for chat dispatch, ``utils``
for diagnostics and ``t1api`` for the HTTP server.

Two rules this module holds to, because it used to break both:

* **Nothing is announced that didn't happen.** Every command either
  performs the real operation and reports what it produced, or reports
  the exact reason it couldn't — the missing package, the unset API key,
  the file that isn't there. There is no "Response generated.", no
  "Saved to output.wav" without a file, no "Server started" without a
  socket.
* **It works without ``rich``.** ``rich`` is an optional dependency, so
  the console falls back to plain ``print`` (markup stripped) rather
  than raising ``NameError`` at import time.
"""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on a rich-less install
    RICH_AVAILABLE = False

# Markup stripping for the rich-less fallback. Matching "anything in square
# brackets" would eat real text — menu labels like "[1] Models", indexes like
# `list[i]` — so only these style words count as a tag.
_STYLE_WORDS = frozenset(
    """
    bold dim italic underline strike reverse blink not
    black red green yellow blue magenta cyan white
    bright_black bright_red bright_green bright_yellow
    bright_blue bright_magenta bright_cyan bright_white
    on
    """.split()
)
_STYLE_ALT = "|".join(sorted(_STYLE_WORDS))
_MARKUP_RE = re.compile(rf"\[/?(?:(?:{_STYLE_ALT})(?:\s+(?:{_STYLE_ALT}))*)?\]")


class _PlainConsole:
    """The subset of ``rich.console.Console`` this module uses."""

    def print(self, *objects: Any, **_kwargs: Any) -> None:
        if not objects:
            print()
            return
        for obj in objects:
            print(_MARKUP_RE.sub("", str(obj)))

    def rule(self, title: str = "", **_kwargs: Any) -> None:
        title = _MARKUP_RE.sub("", title)
        print(f"-- {title} " .ljust(60, "-") if title else "-" * 60)

    @contextlib.contextmanager
    def status(self, message: str = "", **_kwargs: Any):
        print(_MARKUP_RE.sub("", message))
        yield


class _PlainPrompt:
    """The subset of ``rich.prompt.Prompt`` this module uses."""

    @staticmethod
    def ask(
        prompt: str = "",
        *,
        choices: list[str] | None = None,
        default: str | None = None,
        **_kwargs: Any,
    ) -> str:
        label = _MARKUP_RE.sub("", prompt)
        if choices:
            label += f" [{'/'.join(choices)}]"
        if default is not None:
            label += f" ({default})"
        while True:
            answer = input(f"{label}: ").strip()
            if not answer and default is not None:
                return default
            if choices and answer not in choices:
                print(f"Choose one of: {', '.join(choices)}")
                continue
            return answer


console: Any = Console() if RICH_AVAILABLE else _PlainConsole()
_prompt: Any = Prompt if RICH_AVAILABLE else _PlainPrompt


def check_rich() -> bool:
    """True when ``rich`` is installed; warns (once, honestly) when not."""
    if not RICH_AVAILABLE:
        console.print("rich is not installed — using the plain-text menu.")
        console.print("Install it for the full TUI: pip install 'rich>=13'")
        return False
    return True


def _panel(body: str, **kwargs: Any) -> Any:
    """A ``rich`` panel when available, otherwise the bare text."""
    return Panel(body, **kwargs) if RICH_AVAILABLE else body


def _report(exc: BaseException, what: str) -> None:
    """Print a real failure. Never swallowed, never dressed up as success."""
    console.print(f"[red]{what} failed:[/] {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------

class ChatLLM:
    """Adapter exposing ``hyped_pro_core``'s dispatch as a pipeline LLM.

    :class:`hypernix.models.workshop.ASRToLLMToTTS` accepts any object
    with a ``chat(history, max_new_tokens=...)`` method, so this hands it
    the same real cloud/local dispatch the ``hyped`` CLI uses instead of
    the echo stub that used to sit here.
    """

    def __init__(self, model_short: str, system: str | None = None) -> None:
        self.model_short = model_short
        self.system = system

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int | None = None) -> str:
        from hypernix.interfaces import hyped_pro_core as core

        reply = core.send_chat_message(
            self.model_short,
            [{"role": m["role"], "content": m["content"]} for m in messages],
            system=self.system,
            max_tokens=max_new_tokens,
        )
        return reply["content"] or ""

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        return self.chat([{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens)


def _default_model_short() -> str | None:
    """Pick a model that will actually answer, or None if nothing is set up.

    Order: the model the user configured, then any cloud model whose API
    key is already present. Local models are never auto-selected — that
    would kick off a multi-gigabyte download nobody asked for — so with
    neither configured, the caller is told to choose one.
    """
    from hypernix.interfaces import hyped_pro_core as core
    from hypernix.system.config import get_default_model, get_provider_key

    known = {m.short for m in core.MODELS}
    configured = get_default_model()
    if configured and configured in known:
        return configured

    for model in core.MODELS:
        if model.kind != "cloud":
            continue
        try:
            if get_provider_key(model.vendor):
                return model.short
        except Exception:  # noqa: BLE001 - a broken keyring shouldn't decide this
            continue
    return None


def _write_wav(path: Path, audio: Any, sample_rate: int) -> int:
    """Write a 1-D waveform tensor to ``path``. Returns bytes written."""
    import torch

    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    samples = (audio.detach().flatten().clamp(-1, 1) * 32767).short().cpu().numpy()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return path.stat().st_size


def _record_microphone(destination: Path, sample_rate: int = 16000) -> Path | None:
    """Record until Ctrl-C. Returns the file, or None if nothing was captured."""
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        console.print(
            "[red]Recording needs sounddevice + soundfile:[/] "
            "pip install sounddevice soundfile"
        )
        console.print(f"[dim]({exc})[/]")
        return None

    console.print("[yellow]Recording… press Ctrl-C to stop.[/]")
    chunks: list[Any] = []

    def callback(indata, _frames, _time, _status):  # noqa: ANN001 - sounddevice API
        chunks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                pass
    except Exception as exc:  # noqa: BLE001 - device errors are reported, not hidden
        _report(exc, "Recording")
        return None

    if not chunks:
        console.print("[red]No audio captured — the input device returned nothing.[/]")
        return None

    sf.write(str(destination), np.concatenate(chunks, axis=0), sample_rate)
    console.print(f"[green]Recorded {destination} ({destination.stat().st_size} bytes)[/]")
    return destination


class InteractiveCLI:
    """Interactive menu for HyperNix.

    ``model`` pins the chat model (default: whatever ``hypernix config``
    has set, else the first local model in the catalog). ``voice`` starts
    the assistant with spoken replies enabled.
    """

    def __init__(self, model: str | None = None, voice: bool = False) -> None:
        self.running = True
        self.model_short = model
        self.speak_replies = voice
        self._tts: Any = None

    # -- entry points -------------------------------------------------------

    def run(self) -> None:
        """Main loop."""
        if not check_rich():
            self.run_fallback()
            return

        console.print(_panel("[bold blue]HyperNix Interactive CLI[/]", subtitle="q to quit"))
        while self.running:
            self.show_main_menu()

    def run_fallback(self) -> None:
        """Plain-text loop for installs without ``rich``."""
        commands = {
            "models": self.cmd_models,
            "train": self.cmd_train,
            "asr": self.cmd_asr,
            "tts": self.cmd_tts,
            "pipeline": self.cmd_pipeline,
            "assistant": self.cmd_assistant,
            "serve": self.cmd_serve,
        }
        print("\n=== HyperNix CLI ===")
        for name, fn in commands.items():
            print(f"  {name:<10} - {(fn.__doc__ or '').splitlines()[0]}")
        print("  quit       - Exit")

        while True:
            try:
                cmd = input("\nhypernix> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if cmd in ("quit", "exit", "q"):
                return
            handler = commands.get(cmd)
            if handler is None:
                print(f"Unknown command: {cmd}")
                continue
            try:
                handler()
            except KeyboardInterrupt:
                print("\n(interrupted)")

    def show_main_menu(self) -> None:
        """Display the main menu and dispatch one choice."""
        menu_text = Text()
        menu_text.append("\nSelect an option:\n\n", style="bold")
        menu_text.append("  [1] Models\n", style="cyan")
        menu_text.append("  [2] Train\n", style="green")
        menu_text.append("  [3] ASR (Speech→Text)\n", style="yellow")
        menu_text.append("  [4] TTS (Text→Speech)\n", style="magenta")
        menu_text.append("  [5] Pipeline (ASR→LLM→TTS)\n", style="blue")
        menu_text.append("  [6] Local AI Assistant\n", style="red")
        menu_text.append("  [7] Serve the T1 API\n", style="bright_blue")
        menu_text.append("  [q] Quit\n", style="dim")

        console.print(_panel(menu_text, title="[bold]HyperNix Menu[/]"))
        choice = _prompt.ask(
            "Choice", choices=["1", "2", "3", "4", "5", "6", "7", "q"], default="1",
        )

        handlers = {
            "1": self.cmd_models,
            "2": self.cmd_train,
            "3": self.cmd_asr,
            "4": self.cmd_tts,
            "5": self.cmd_pipeline,
            "6": self.cmd_assistant,
            "7": self.cmd_serve,
        }
        if choice == "q":
            self.running = False
            return
        try:
            handlers[choice]()
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")

    # -- commands -----------------------------------------------------------

    def cmd_models(self) -> None:
        """List the models HyperNix knows how to fetch."""
        console.print(_panel("[bold]Model Registry[/]"))
        try:
            from hypernix.models.download import KNOWN_MODELS
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the model registry")
            return

        if RICH_AVAILABLE:
            table = Table(title=f"Known models ({len(KNOWN_MODELS)})")
            table.add_column("Short name", style="cyan", no_wrap=True)
            table.add_column("Repo", style="white")
            table.add_column("Arch", style="magenta")
            table.add_column("Family", style="green")
            for name, info in sorted(KNOWN_MODELS.items()):
                table.add_row(name, info.repo_id, info.arch, info.family or "-")
            console.print(table)
        else:
            for name, info in sorted(KNOWN_MODELS.items()):
                print(f"  {name:<24} {info.repo_id:<44} {info.arch}")

        console.print(
            "[dim]Fetch one with: hypernix download <short name>[/]"
        )
        _prompt.ask("\nPress Enter", default="")

    def cmd_train(self) -> None:
        """Run a training recipe end to end."""
        console.print(_panel("[bold]Training[/]"))
        try:
            from hypernix.training import instant_pot
            from hypernix.training.recipe_book import RecipeBook
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the training modules")
            return

        book = RecipeBook.from_builtins()
        console.print(f"Built-in recipes: [cyan]{', '.join(book.names()) or '(none)'}[/]")
        answer = _prompt.ask(
            "Recipe name or path to a recipe .json (blank to cancel)", default="",
        )
        if not answer:
            console.print("[dim]Cancelled — nothing was run.[/]")
            return

        if answer in book:
            recipe = book.get(answer)
        else:
            path = Path(answer).expanduser()
            if not path.is_file():
                console.print(f"[red]No recipe named {answer!r} and no file at {path}.[/]")
                return
            try:
                recipe = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _report(exc, f"Reading {path}")
                return

        console.print(f"[yellow]Running recipe:[/] {json.dumps(recipe, indent=2)[:800]}")
        try:
            out_dir = instant_pot.brew(recipe)
        except Exception as exc:  # noqa: BLE001 - training failures are reported as-is
            _report(exc, "Training")
            return
        console.print(f"[green]Training finished. Snapshot written to:[/] {out_dir}")
        _prompt.ask("Press Enter", default="")

    def cmd_asr(self) -> None:
        """Transcribe an audio file or a microphone recording."""
        console.print(_panel("[bold]Automatic Speech Recognition[/]"))
        try:
            import torch

            from hypernix.models.workshop import ASREngine
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the ASR engine")
            return

        answer = _prompt.ask("Audio file path (blank to record from the microphone)", default="")
        if answer:
            audio_file = Path(answer).expanduser()
            if not audio_file.is_file():
                console.print(f"[red]No such file:[/] {audio_file}")
                return
        else:
            import tempfile

            tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            recorded = _record_microphone(tmp)
            if recorded is None:
                tmp.unlink(missing_ok=True)
                return
            audio_file = recorded

        engine = ASREngine()
        try:
            with console.status("[bold green]Loading the ASR model…"):
                engine.initialize()
            with console.status("[bold green]Transcribing…"):
                import torchaudio

                audio, sample_rate = torchaudio.load(str(audio_file))
                if sample_rate != engine.config.sample_rate:
                    ratio = engine.config.sample_rate / sample_rate
                    audio = torch.nn.functional.interpolate(
                        audio.unsqueeze(0),
                        size=int(audio.shape[1] * ratio),
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                text = engine.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Transcription")
            return

        console.print(f"[green]Transcript:[/] {text}")
        _prompt.ask("Press Enter", default="")

    def cmd_tts(self) -> None:
        """Synthesize speech and write it to a .wav file."""
        console.print(_panel("[bold]Text-to-Speech[/]"))
        try:
            from hypernix.models.workshop import TTSEngine
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the TTS engine")
            return

        text = _prompt.ask("Text to synthesize", default="")
        if not text:
            console.print("[dim]Nothing to synthesize.[/]")
            return
        out_path = Path(_prompt.ask("Output audio file", default="output.wav")).expanduser()

        engine = TTSEngine()
        try:
            with console.status("[bold green]Loading the TTS model…"):
                engine.initialize()
            with console.status("[bold green]Synthesizing…"):
                waveform = engine.synthesize(text)
                written = _write_wav(out_path, waveform, engine.config.sample_rate)
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Synthesis")
            return

        console.print(
            f"[green]Wrote[/] {out_path.resolve()} "
            f"({written} bytes, {engine.config.sample_rate} Hz mono)"
        )
        _prompt.ask("Press Enter", default="")

    def cmd_pipeline(self) -> None:
        """Run the full ASR → LLM → TTS pipeline on one utterance."""
        console.print(_panel("[bold]ASR → LLM → TTS[/]"))
        try:
            from hypernix.models.workshop import ASREngine, ASRToLLMToTTS, TTSEngine
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the pipeline")
            return

        answer = _prompt.ask("Input audio file (blank to record from the microphone)", default="")
        if answer:
            audio_file = Path(answer).expanduser()
            if not audio_file.is_file():
                console.print(f"[red]No such file:[/] {audio_file}")
                return
        else:
            import tempfile

            tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            recorded = _record_microphone(tmp)
            if recorded is None:
                tmp.unlink(missing_ok=True)
                return
            audio_file = recorded

        default_model = self.model_short or _default_model_short()
        model_short = _prompt.ask("Model", default=default_model or "")
        if not model_short:
            console.print("[red]The LLM stage needs a model — pass one or set a default.[/]")
            return

        pipeline = ASRToLLMToTTS(ASREngine(), ChatLLM(model_short), TTSEngine())
        try:
            with console.status("[bold green]Running ASR → LLM → TTS…"):
                response_text, audio_bytes = pipeline.process(str(audio_file))
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Pipeline")
            return

        out_path = Path("pipeline_output.wav")
        out_path.write_bytes(audio_bytes)
        console.print(f"[green]Response:[/] {response_text}")
        console.print(f"[green]Wrote[/] {out_path.resolve()} ({len(audio_bytes)} bytes)")
        _prompt.ask("Press Enter", default="")

    def cmd_assistant(self) -> None:
        """Chat with a model, with real dispatch and real diagnostics."""
        console.print(_panel("[bold]HyperNix Assistant[/]"))
        try:
            from hypernix.chat.menu import Menu
            from hypernix.interfaces import hyped_pro_core as core
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the chat backend")
            return

        model_short = self.model_short or _default_model_short()
        if model_short is None:
            console.print(
                "[red]No model selected.[/] Pick one with "
                "[cyan]hypernix assistant --model <name>[/], set a default with "
                "[cyan]hypernix config dmodel <name>[/], or add an API key with "
                "[cyan]hypernix config[/]."
            )
            console.print("Registered models:")
            for model in core.MODELS:
                console.print(f"  [cyan]{model.short:<22}[/] {model.kind}")
            return
        try:
            model_short = core.get_model(model_short).short
        except Exception as exc:  # noqa: BLE001
            _report(exc, f"Selecting model {model_short!r}")
            return

        menu = Menu.from_builtins()
        persona = "default" if "default" in menu else (menu.names() or [None])[0]
        system = menu.get(persona) if persona else None
        history: list[dict[str, str]] = []

        console.print(f"Model: [cyan]{model_short}[/]   Persona: [cyan]{persona}[/]")
        console.print(
            "[dim]/help  /model <name>  /persona <name>  /models  /personas  "
            "/system [cmd]  /voice  /listen  /reset  /save <path>  /quit[/]"
        )
        if self.speak_replies and not self._ensure_tts():
            self.speak_replies = False

        while True:
            try:
                user_input = _prompt.ask("You", default="")
            except (EOFError, KeyboardInterrupt):
                console.print()
                return
            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in ("/quit", "/exit", "quit", "exit"):
                return
            if lowered == "/help":
                console.print(
                    "[cyan]/model <name>[/]     switch model\n"
                    "[cyan]/persona <name>[/]   switch system prompt\n"
                    "[cyan]/models[/]           list registered models\n"
                    "[cyan]/personas[/]         list system-prompt presets\n"
                    "[cyan]/system[/]           environment diagnostics\n"
                    "[cyan]/system <command>[/] run a shell command and show its output\n"
                    "[cyan]/voice[/]            toggle spoken replies (TTS)\n"
                    "[cyan]/listen[/]           record from the mic and use the transcript\n"
                    "[cyan]/reset[/]            clear the conversation\n"
                    "[cyan]/save <path>[/]      write the transcript to JSON\n"
                    "[cyan]/quit[/]             back to the menu"
                )
                continue
            if lowered == "/models":
                for model in core.MODELS:
                    console.print(
                        f"  [cyan]{model.short:<22}[/] {model.kind:<6} "
                        f"{model.context_window:>7} ctx  {model.notes}"
                    )
                continue
            if lowered == "/personas":
                console.print("  " + ", ".join(menu.names()))
                continue
            if lowered.startswith("/model "):
                candidate = user_input.split(maxsplit=1)[1].strip()
                try:
                    model_short = core.get_model(candidate).short
                except Exception as exc:  # noqa: BLE001
                    _report(exc, f"Selecting model {candidate!r}")
                    continue
                console.print(f"[green]Model is now {model_short}.[/]")
                continue
            if lowered.startswith("/persona "):
                candidate = user_input.split(maxsplit=1)[1].strip()
                matched = menu.find(candidate)
                if matched is None:
                    console.print(f"[red]No persona matching {candidate!r}.[/] Try /personas.")
                    continue
                persona, system = matched, menu.get(matched)
                console.print(f"[green]Persona is now {persona}.[/]")
                continue
            if lowered.startswith("/system"):
                command = user_input[len("/system"):].strip()
                if command:
                    self._run_shell(command)
                else:
                    self._print_diagnostics()
                continue
            if lowered == "/voice":
                if self.speak_replies:
                    self.speak_replies = False
                    console.print("[green]Spoken replies off.[/]")
                else:
                    self.speak_replies = self._ensure_tts()
                    if self.speak_replies:
                        console.print("[green]Spoken replies on.[/]")
                continue
            if lowered == "/listen":
                spoken = self._transcribe_one_turn()
                if spoken is None:
                    continue
                console.print(f"[dim]Heard:[/] {spoken}")
                user_input = spoken
            elif lowered == "/reset":
                history.clear()
                console.print("[green]Conversation cleared.[/]")
                continue
            elif lowered.startswith("/save"):
                parts = user_input.split(maxsplit=1)
                target = Path(parts[1].strip() if len(parts) > 1 else "transcript.json")
                try:
                    target.write_text(json.dumps(history, indent=2), encoding="utf-8")
                except OSError as exc:
                    _report(exc, f"Writing {target}")
                    continue
                console.print(f"[green]Wrote {target.resolve()} ({len(history)} messages).[/]")
                continue
            elif user_input.startswith("/"):
                console.print(f"[red]Unknown command:[/] {user_input}. Try /help.")
                continue

            history.append({"role": "user", "content": user_input})
            try:
                with console.status("[bold green]Thinking…"):
                    reply = core.send_chat_message(model_short, history, system=system)
            except Exception as exc:  # noqa: BLE001 - a failed turn is reported, not faked
                history.pop()
                _report(exc, "Chat")
                continue

            content = reply["content"] or ""
            history.append({"role": "assistant", "content": content})
            console.print(f"[bold green]{model_short}:[/] {content}")
            if self.speak_replies and content:
                self._speak(content)

    def cmd_serve(self) -> None:
        """Serve the T1 API over HTTP (and report the Tailscale address)."""
        console.print(_panel("[bold]Serve the T1 API[/]"))
        try:
            import uvicorn

            from hypernix.t1api import create_app
        except Exception as exc:  # noqa: BLE001
            console.print(
                "[red]The HTTP server needs the t1api extra:[/] "
                "pip install 'hypernix[t1api]'"
            )
            console.print(f"[dim]({type(exc).__name__}: {exc})[/]")
            return

        host = _prompt.ask("Bind host", default="127.0.0.1")
        port_text = _prompt.ask("Bind port", default="8080")
        try:
            port = int(port_text)
        except ValueError:
            console.print(f"[red]{port_text!r} is not a port number.[/]")
            return

        self._print_tailscale_status(port)

        try:
            app = create_app()
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Building the T1 API app")
            return

        console.print(f"[green]Serving on[/] http://{host}:{port}  [dim](Ctrl-C to stop)[/]")
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except KeyboardInterrupt:
            console.print("\n[dim]Server stopped.[/]")
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Serving")

    # -- helpers ------------------------------------------------------------

    def _print_diagnostics(self) -> None:
        """Print the real environment report, not a canned 'all systems go'."""
        try:
            from hypernix.system.utils import diagnostic_info
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Collecting diagnostics")
            return

        info = diagnostic_info()
        if RICH_AVAILABLE:
            table = Table(title="Environment")
            table.add_column("Key", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")
            for key, value in info.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in sorted(value.items()):
                        table.add_row(f"{key}.{sub_key}", str(sub_value))
                else:
                    table.add_row(key, str(value))
            console.print(table)
        else:
            console.print(json.dumps(info, indent=2, default=str))

    def _run_shell(self, command: str) -> None:
        """Run a shell command and show its real output and exit status."""
        try:
            result = subprocess.run(
                # nosec B602 - `/system <cmd>` is an operator-typed shell
                # escape, the same as `!` in a REPL. No model output reaches
                # here; shell=True is the feature being asked for.
                # nosemgrep: python.lang.security.audit.subprocess-shell-true.subprocess-shell-true
                command, shell=True, capture_output=True, text=True, timeout=60, check=False,  # noqa: S602
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _report(exc, f"Running {command!r}")
            return
        if result.stdout:
            console.print(result.stdout.rstrip())
        if result.stderr:
            console.print(f"[red]{result.stderr.rstrip()}[/]")
        if result.returncode != 0:
            console.print(f"[red]exit status {result.returncode}[/]")

    def _ensure_tts(self) -> bool:
        """Load the TTS engine. False (with the reason) when it can't be."""
        if self._tts is not None:
            return True
        try:
            from hypernix.models.workshop import TTSConfig, TTSEngine

            engine = TTSEngine(TTSConfig(sample_rate=22050))
            with console.status("[bold green]Loading the TTS model…"):
                engine.initialize()
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Enabling spoken replies")
            return False
        self._tts = engine
        return True

    def _speak(self, text: str) -> None:
        """Synthesize ``text`` and play it, or say where the audio landed."""
        if not self._ensure_tts():
            return
        import tempfile

        path = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            waveform = self._tts.synthesize(text)
            _write_wav(path, waveform, self._tts.config.sample_rate)
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Speech synthesis")
            path.unlink(missing_ok=True)
            return

        player = next(
            (p for p in ("paplay", "aplay", "afplay", "ffplay") if shutil.which(p)), None,
        )
        if player is None:
            # No player on this box — keep the file rather than pretending
            # something was heard.
            console.print(f"[dim]No audio player found; reply audio saved to {path}[/]")
            return
        args = [player, str(path)]
        if player == "ffplay":
            args = [player, "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        try:
            subprocess.run(args, check=False, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            console.print(f"[dim]Playback via {player} failed ({exc}); audio is at {path}[/]")
            return
        path.unlink(missing_ok=True)

    def _print_tailscale_status(self, port: int) -> None:
        """Report the machine's real Tailscale name, or that there isn't one."""
        if shutil.which("tailscale") is None:
            console.print("[dim]tailscale is not installed — the server will be local only.[/]")
            return
        try:
            raw = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout
            name = json.loads(raw).get("Self", {}).get("DNSName", "").rstrip(".")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            console.print(f"[dim]Could not read tailscale status: {exc}[/]")
            return
        if name:
            console.print(f"[green]Reachable on your tailnet at[/] http://{name}:{port}")
        else:
            console.print("[dim]tailscale is installed but this node has no DNS name yet.[/]")

    def _transcribe_one_turn(self) -> str | None:
        """Record from the microphone and return the transcript, or None."""
        import tempfile

        try:
            import torchaudio

            from hypernix.models.workshop import ASREngine
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Loading the ASR engine")
            return None

        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            if _record_microphone(tmp) is None:
                return None
            engine = ASREngine()
            with console.status("[bold green]Transcribing…"):
                engine.initialize()
                audio, _sample_rate = torchaudio.load(str(tmp))
                return engine.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            _report(exc, "Transcription")
            return None
        finally:
            tmp.unlink(missing_ok=True)


def cli_main() -> int:
    """Entry point for the ``assistant`` menu."""
    try:
        InteractiveCLI().run()
    except KeyboardInterrupt:
        console.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
