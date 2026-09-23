"""carbon — quality of life, and elements you write yourself.

Element 6, because everything organic is built on it.

Two jobs
--------
**Quality of life**, as an oven addon: tidy the output a model returns
(trailing whitespace, runs of blank lines, stray CRLFs) and expand
``::snippet`` shortcuts in prompts. The tidying never touches the inside
of a fenced code block — whitespace there can be the program, and a
formatter that "cleans up" a Makefile's tabs or a Python string's
trailing spaces has changed what the code does.

**User elements.** :func:`scaffold` writes a starting file into
``~/.hypernix/elements/``, and :func:`load_user_elements` reads that
directory into a registry. Each file is executed in isolation, so one
that raises is recorded and skipped rather than taking the rest with it.
A user file is always registered as a user element whatever its spec
says, which is what stops a file from claiming to be a built-in and
displacing HyperNix's own code.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Any

from ..system import errorcatalogue as codes
from ..system.errorcodes import HyperNixError
from .hydrogen import (
    Element,
    ElementSpec,
    Registry,
    atomic_number,
    is_experimental,
    name_of,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Carbon",
    "tidy",
    "expand_snippets",
    "user_elements_dir",
    "scaffold",
    "load_user_elements",
]

_FENCE = re.compile(r"^(\s*)(```|~~~)")
_SNIPPET = re.compile(r"(?<![\w:])::([A-Za-z][\w-]*)")


def tidy(text: str) -> str:
    """Trailing whitespace, CRLFs and blank-line runs — outside code only."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    fence: str | None = None
    blank_run = 0
    for line in lines:
        marker = _FENCE.match(line)
        if fence is None:
            if marker:
                fence = marker.group(2)
                blank_run = 0
                out.append(line.rstrip())
                continue
            stripped = line.rstrip()
            if not stripped:
                blank_run += 1
                if blank_run > 1:
                    continue
            else:
                blank_run = 0
            out.append(stripped)
        else:
            # Inside a fence: untouched, byte for byte, until it closes.
            out.append(line)
            if marker and marker.group(2) == fence:
                fence = None
    joined = "\n".join(out).lstrip("\n")
    # Trailing blank lines go only when the text ended outside a fence.
    # A model cut off mid code block leaves the fence open, and what is
    # in it is code, down to the last newline.
    return joined if fence is not None else joined.rstrip("\n")


def expand_snippets(prompt: str, snippets: dict[str, str]) -> str:
    """``::name`` -> its text. An unknown name is left as typed.

    Left rather than removed or refused: ``::`` turns up in C++ and Rust
    and a prompt about ``std::vector`` must not lose half of itself.
    The lookbehind already skips ``std::vector``; leaving unknowns alone
    covers the rest.
    """
    if not snippets:
        return prompt
    return _SNIPPET.sub(lambda m: snippets.get(m.group(1), m.group(0)), prompt)


class Carbon(Element):
    spec = ElementSpec(
        symbol="C",
        summary="Quality of life: tidy output, ::snippets in prompts, and "
                "the loader for elements you write yourself.",
        version="1.0.0",
        permissions=frozenset({"oven"}),
    )

    def transform_prompt(self, prompt: str) -> str:
        return expand_snippets(prompt, self.context.config.get("snippets", {}))

    def transform_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        snippets = self.context.config.get("snippets", {})
        return [
            {**m, "content": expand_snippets(m.get("content", ""), snippets)}
            if m.get("role") == "user" else m
            for m in messages
        ]

    def transform_output(self, text: str) -> str:
        if self.context.config.get("tidy", True) is False:
            return text
        return tidy(text)


# ---------------------------------------------------------------------------
# User elements
# ---------------------------------------------------------------------------


def user_elements_dir() -> Path:
    return Path.home() / ".hypernix" / "elements"


_TEMPLATE = '''"""{name} — a user element.

Loaded by `hypernix elements` from ~/.hypernix/elements/. Override only
the hooks you need; every one has a no-op default.
"""
from hypernix.elements.hydrogen import Element, ElementSpec


class {cls}(Element):
    spec = ElementSpec(
        symbol="{symbol}",
        summary="What {name} does, in one line.",
        # Ask only for what you use. Undeclared permissions are refused
        # at the moment of use: processes, network, filesystem, oven, shell.
        permissions=frozenset({{"oven"}}),
        user=True,
    )

    def transform_prompt(self, prompt: str) -> str:
        return prompt

    def transform_output(self, text: str) -> str:
        return text
'''


def scaffold(symbol: str, directory: Path | None = None, *,
             registry: Registry | None = None) -> Path:
    """Write a starting file for a user element. Refuses to overwrite."""
    atomic_number(symbol)
    if registry is not None:
        try:
            held = registry.lookup(symbol)
        except HyperNixError:
            held = None
        if held is not None and not held.spec.user:
            raise HyperNixError(
                codes.ELEMENT_CONFLICT,
                f"{symbol} is the built-in {held.spec.name}; pick another symbol",
            )
    name = name_of(symbol)
    target = (directory or user_elements_dir()) / f"{name}.py"
    if target.exists():
        raise HyperNixError(codes.ELEMENT_CONFLICT,
                            f"{target} already exists; not overwriting it")
    target.parent.mkdir(parents=True, exist_ok=True)
    cls = "".join(part.capitalize() for part in re.split(r"[^A-Za-z]", name) if part)
    target.write_text(_TEMPLATE.format(name=name, cls=cls or symbol,
                                       symbol=symbol), encoding="utf-8")
    if is_experimental(atomic_number(symbol)):
        logger.info("carbon: %s is in period 6 or 7 and will need "
                    "--experimental to run", symbol)
    return target


def load_user_elements(registry: Registry, directory: Path | None = None) -> list[str]:
    """Register every element in *directory*. Returns the symbols loaded.

    Each file runs in its own namespace — executed, not imported, so a
    file called ``json.py`` cannot shadow the standard library for the
    rest of the process.
    """
    root = directory or user_elements_dir()
    loaded: list[str] = []
    if not root.is_dir():
        return loaded
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        namespace: dict[str, Any] = {"__name__": f"hypernix_user_element_{path.stem}",
                                     "__file__": str(path)}
        try:
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
            exec(code, namespace)  # noqa: S102 - the user's own file, by design
        except Exception as exc:  # noqa: BLE001 - one bad file, not the loader
            registry.failures[path.stem] = HyperNixError(
                codes.ELEMENT_LOAD_FAILED, f"{path.name}: {exc}").args[0]
            continue
        for value in list(namespace.values()):
            if not (isinstance(value, type) and issubclass(value, Element)
                    and isinstance(getattr(value, "spec", None), ElementSpec)):
                continue
            if value.__module__ != namespace["__name__"]:
                # Imported into the file, not defined in it — `Element`
                # itself, or a built-in like `Carbon` the user subclassed.
                # Touching its spec would rewrite a built-in for the whole
                # process.
                continue
            if not value.spec.user:
                # Forced, whatever the file says: a user file claiming to
                # be a built-in is exactly how one would displace ours.
                value.spec = dataclasses.replace(value.spec, user=True)
            try:
                registry.register(value)
                loaded.append(value.spec.symbol)
            except HyperNixError as exc:
                registry.failures[value.spec.symbol] = str(exc)
    return loaded
