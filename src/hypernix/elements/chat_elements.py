"""Resolve elements selected by hypernix chat --elements."""
from __future__ import annotations

from pathlib import Path

from ..system import errorcatalogue as codes
from ..system.errorcodes import HyperNixError
from . import carbon
from .builtins import register_all
from .hydrogen import Registry


def resolve_chat_elements(
    selection: str | None = None,
    group: str | None = None,
    *,
    allow_experimental: bool = False,
) -> tuple[Registry, list[str]]:
    """Resolve file/directory/all selection and return chat-addon roots."""
    registry = Registry(allow_experimental=allow_experimental)
    register_all(registry)

    selected = selection or "all"
    explicit_path = selected != "all"
    if explicit_path:
        path = Path(selected).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"element path does not exist: {path}")
        if path.is_dir():
            root, requested_file = path, None
        elif path.is_file() and path.suffix.lower() in {".py", ".hnx"}:
            root, requested_file = path.parent, path.resolve()
        else:
            raise ValueError("element path must be a .py/.hnx file or a directory")
    else:
        root, requested_file = carbon.user_elements_dir(), None

    loaded = carbon.load_user_elements(registry, root)
    if not explicit_path:
        candidates = [spec.symbol for spec in registry.specs()]
    elif requested_file is not None:
        candidates = list(registry.sources.get(str(requested_file), []))
        if not candidates:
            reason = registry.failures.get(requested_file.stem)
            raise HyperNixError(
                codes.ELEMENT_LOAD_FAILED,
                f"could not load {requested_file.name}" + (f": {reason}" if reason else ""),
            )
    else:
        candidates = loaded

    if group:
        wanted = group.strip().casefold()
        candidates = [
            symbol for symbol in candidates
            if any(name.casefold() == wanted for name in registry.lookup(symbol).spec.groups)
        ]
        if not candidates:
            raise HyperNixError(
                codes.ELEMENT_NOT_FOUND,
                f"no loaded elements belong to group {group!r}",
            )

    roots: list[str] = []
    for symbol in candidates:
        spec = registry.lookup(symbol).spec
        if "oven" not in spec.permissions:
            if requested_file is not None:
                raise HyperNixError(
                    codes.ELEMENT_PERMISSION,
                    f"{symbol} does not declare 'oven' permission and cannot attach to chat",
                )
            continue
        if spec.experimental and not allow_experimental:
            if requested_file is not None:
                raise HyperNixError(
                    codes.ELEMENT_EXPERIMENTAL,
                    f"{symbol} is experimental; pass --experimental-elements to load it",
                )
            continue
        roots.append(symbol)

    if not roots and (explicit_path or group):
        raise HyperNixError(codes.ELEMENT_NOT_FOUND, "no chat-compatible elements were selected")
    return registry, roots
