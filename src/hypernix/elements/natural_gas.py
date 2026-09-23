"""natural_gas — elements as Neo Oven addons.

Methane: one carbon, four hydrogens. It is what the oven burns.

What it does
------------
:func:`attach` takes an oven and some elements and wraps the oven's
``complete`` and ``chat`` so each element's ``transform_*`` hooks run
around every generation — prompt in, output out. :func:`detach` puts
the originals back exactly.

Per instance, not per class. Patching ``NeoOven`` itself would attach an
addon to every oven in the process, including the one somebody else
loaded for a different job, and the only symptom would be output that is
subtly not what that job asked for.

Order is by atomic number, lightest first, on the way in, and the
reverse on the way out — so an element wraps the ones lighter than it
the same way a function wraps the functions it calls, and "carbon
tidies what magnesium produced" means the same thing every run.

An addon that raises is skipped for that call, logged, and counted — the
generation still happens. An oven that stops answering because a
formatting addon threw is a worse outcome than an unformatted answer.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any

from ..system import errorcatalogue as codes
from ..system.errorcodes import HyperNixError
from .hydrogen import Element, Registry, default_registry

logger = logging.getLogger(__name__)

__all__ = ["Attachment", "attach", "detach", "attached"]

_ATTR = "_natural_gas"
_WRAPPED = ("complete", "chat")


@dataclass
class Attachment:
    """What natural gas did to one oven, so it can be undone."""

    elements: list[Element]
    originals: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        return [e.spec.symbol for e in self.elements]

    def _fail(self, element: Element, hook: str, exc: Exception) -> None:
        key = f"{element.spec.symbol}.{hook}"
        self.failures[key] = self.failures.get(key, 0) + 1
        logger.warning("natural_gas: %s raised in %s: %s",
                       element.spec.symbol, hook, exc)

    def prompt(self, text: str) -> str:
        for element in self.elements:
            try:
                text = element.transform_prompt(text)
            except Exception as exc:  # noqa: BLE001 - one addon, not the oven
                self._fail(element, "transform_prompt", exc)
        return text

    def messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        # A copy, so an element that edits in place cannot reach back
        # into the caller's conversation history.
        current = [dict(m) for m in messages]
        for element in self.elements:
            try:
                current = element.transform_messages(current)
            except Exception as exc:  # noqa: BLE001
                self._fail(element, "transform_messages", exc)
        return current

    def output(self, text: str) -> str:
        for element in reversed(self.elements):
            try:
                text = element.transform_output(text)
            except Exception as exc:  # noqa: BLE001
                self._fail(element, "transform_output", exc)
        return text


def attached(oven: Any) -> Attachment | None:
    return oven.__dict__.get(_ATTR) if hasattr(oven, "__dict__") else None


def attach(
    oven: Any,
    elements: list[str | Element],
    *,
    registry: Registry | None = None,
) -> Attachment:
    """Wrap *oven* so *elements* run around each generation.

    Attaching again replaces the previous set rather than stacking a
    second wrapper on the first — stacking is how an addon ends up
    running twice per call with nothing on screen to say so.
    """
    registry = registry or default_registry()
    if attached(oven) is not None:
        detach(oven)

    # Check every element before starting any. The framework asks on each
    # element's behalf: one that did not declare `oven` does not get to
    # touch one — and must be refused before it is activated, or asking
    # for magnesium here would renice the machine and only then say no.
    live: list[Element] = []
    for item in elements:
        element = item if isinstance(item, Element) else registry.instance(item)
        element.context.require("oven")
        live.append(element)
    live.sort(key=lambda e: e.spec.number)

    started: list[Element] = []
    try:
        for element in live:
            if not element.active:
                element.activate()
                element.active = True
                started.append(element)
    except BaseException:
        # All or nothing: an attach that fails part-way leaves nothing it
        # started still running.
        for element in reversed(started):
            try:
                element.deactivate()
            finally:
                element.active = False
        raise

    attachment = Attachment(elements=live)
    for name in _WRAPPED:
        original = getattr(oven, name, None)
        if original is None:
            continue
        attachment.originals[name] = original
        setattr(oven, name, _wrap(name, original, attachment))
    oven.__dict__[_ATTR] = attachment
    return attachment


def detach(oven: Any) -> list[str]:
    """Put the oven's own methods back. Returns what was removed."""
    attachment = attached(oven)
    if attachment is None:
        return []
    for name in attachment.originals:
        # Deleting the instance attribute, not assigning the saved bound
        # method back: that would leave an instance attribute shadowing
        # the class method, and a later patch to the class would never
        # reach this oven.
        oven.__dict__.pop(name, None)
    oven.__dict__.pop(_ATTR, None)
    return attachment.symbols


def _wrap(name: str, original, attachment: Attachment):
    if name == "chat":
        @functools.wraps(original)
        def chat(messages, *args, **kwargs):
            return attachment.output(original(attachment.messages(messages),
                                              *args, **kwargs))
        return chat

    @functools.wraps(original)
    def complete(prompt, *args, **kwargs):
        return attachment.output(original(attachment.prompt(prompt),
                                          *args, **kwargs))
    return complete


def require_attached(oven: Any) -> Attachment:  # pragma: no cover - convenience
    found = attached(oven)
    if found is None:
        raise HyperNixError(codes.ELEMENT_NOT_FOUND, "no addons on this oven")
    return found
