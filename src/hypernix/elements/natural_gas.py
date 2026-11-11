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

Dependencies are resolved by Hydrogen before activation. The dependency
graph is topologically ordered (with atomic number as a stable tie-breaker),
and dependencies are released in reverse activation order.

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
    roots: list[Element] = field(default_factory=list)
    started: list[Element] = field(default_factory=list)
    registry: Registry | None = None

    @property
    def symbols(self) -> list[str]:
        return [e.spec.symbol for e in (self.roots or self.elements)]

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
    """Wrap one oven with requested elements and their transitive dependencies."""
    registry = registry or default_registry()
    if attached(oven) is not None:
        detach(oven)

    roots: list[str] = []
    for item in elements:
        if isinstance(item, Element):
            try:
                registry.lookup(item.spec.symbol)
            except HyperNixError:
                registry.register(type(item))
                registry._live[item.spec.symbol] = item
            symbol = item.spec.symbol
        else:
            symbol = registry.lookup(item).spec.symbol
        if symbol not in roots:
            roots.append(symbol)

    order = registry.dependency_order(roots)
    live = [registry.instance(symbol) for symbol in order]
    root_set = set(roots)

    for element in live:
        unexpected = element.spec.permissions - {"oven"}
        if element.spec.symbol not in root_set and unexpected:
            raise HyperNixError(
                codes.ELEMENT_PERMISSION,
                f"{element.spec.symbol} dependency requests non-chat permissions "
                f"{sorted(unexpected)}; refusing automatic chat activation",
            )
        has_hooks = any(
            getattr(type(element), hook) is not getattr(Element, hook)
            for hook in ("transform_prompt", "transform_messages", "transform_output")
        )
        if element.spec.symbol in root_set or has_hooks:
            element.context.require("oven")

    started: list[Element] = []
    try:
        for symbol in order:
            before = {element.spec.symbol for element in registry.active()}
            registry.activate(symbol)
            after = {element.spec.symbol for element in registry.active()}
            known_started = {element.spec.symbol for element in started}
            started.extend(
                registry.instance(item) for item in order
                if item in after - before and item not in known_started
            )
    except BaseException:
        for element in reversed(started):
            try:
                registry.deactivate(element.spec.symbol)
            except Exception as exc:  # noqa: BLE001 - preserve activation failure
                logger.warning("natural_gas: rollback of %s failed: %s",
                               element.spec.symbol, exc)
        raise

    attachment = Attachment(
        elements=live,
        roots=[registry.instance(symbol) for symbol in roots],
        started=started,
        registry=registry,
    )
    try:
        for name in _WRAPPED:
            original = getattr(oven, name, None)
            if original is None:
                continue
            attachment.originals[name] = original
            setattr(oven, name, _wrap(name, original, attachment))
        oven.__dict__[_ATTR] = attachment
        registry.retain_attachment(
            [element.spec.symbol for element in live],
            [element.spec.symbol for element in started],
        )
    except BaseException:
        oven.__dict__.pop(_ATTR, None)
        for name in attachment.originals:
            oven.__dict__.pop(name, None)
        for element in reversed(started):
            try:
                registry.deactivate(element.spec.symbol)
            except Exception:
                pass
        raise
    return attachment


def detach(oven: Any) -> list[str]:
    """Restore the oven's methods and release its element/dependency leases."""
    attachment = attached(oven)
    if attachment is None:
        return []
    for name in attachment.originals:
        oven.__dict__.pop(name, None)
    oven.__dict__.pop(_ATTR, None)
    if attachment.registry is not None:
        attachment.registry.release_attachment(
            element.spec.symbol for element in attachment.elements
        )
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
