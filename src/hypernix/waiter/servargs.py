"""waiter.servargs — how ``waiter serv`` reads its letters (0.72.6).

Two rules, applied before argparse sees anything:

**Clusters.** Single letters can be run together, in any order and any
combination: ``-ArEK T2C_… -I 100.64.0.7`` is ``-A -r -E -K T2C_… -I
100.64.0.7``. A letter that takes a value must come last in its cluster
— its value is the next word — or stand on its own (``-K key -A`` is as
good as ``-AK key``). ``-AKE key`` is refused rather than guessed at: the
``K`` would swallow nothing.

Inside a cluster, ``r`` means *refresh* (``-R``). On its own, ``-r``
keeps the meaning it has always had, ``-r SUBJECT=LIMIT`` (force a
limit), because a lone ``-r`` has room for its value and a clustered one
does not. ``Rf`` in a cluster is the full refresh (``-Rf``) and ``ud`` is
update-to-exactly-the-server's-version (``-ud``).

**-b: bare strings are worked out.** Without ``-b`` a word with no flag
in front of it is an error, as it always was. With ``-b`` each one is
given to the flag it looks like it belongs to:

===============================  =========
Looks like                       Goes to
===============================  =========
``T1_…`` ``T2_…`` ``T2C_…`` etc  ``-K``
``http(s)://…``, an IP, a host   ``-I``
a whole number 1-65535           ``-P``
``key:…=N/Ns`` or ``N/Ns``       ``-r``
``KEY=VALUE``                    ``-C``
a ``.zip`` or a folder w/ kit    ``-k``
an existing file                 ``-F``
===============================  =========

A CIDR range is refused: it could be ``-B``, ``-W`` or ``-a``, and
guessing between blocking and allowing an address is not a guess worth
making. Two strings that both look like, say, a server are refused too.
"""
from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field

__all__ = ["ServArgsError", "expand", "VALUE_LETTERS", "FLAG_LETTERS"]


class ServArgsError(ValueError):
    """The command line cannot be read. The message says why and how."""


#: Letters whose option takes a value, and the long option each means.
VALUE_LETTERS: dict[str, str] = {
    "I": "--server",
    "K": "--key",
    "F": "--config-file",
    "P": "--port",
    "H": "--home",
    "B": "--blacklist",
    "W": "--whitelist",
    "r": "--force-limit",
    "a": "--appeal",
    "C": "--config",
    "k": "--kit-install",
}

#: Letters that are switches.
FLAG_LETTERS: dict[str, str] = {
    "A": "--auto",
    "E": "--encrypt",
    "s": "--save",
    "L": "--local-only",
    "G": "--gui",
    "g": "--cli",
    "R": "--refresh",
    "y": "--sync",
    "u": "--update",
    "c": "--conceal",
    "T": "--control",
    "Y": "--info",
    "S": "--security-check",
    "e": "--lock",
    "b": "--bundle",
}

#: Two-letter sequences with a meaning of their own inside a cluster.
_PAIRS = {"Rf": "--force-refresh", "ud": "--update-exact"}

#: Whole tokens that are options as they stand.
_EXACT = {"-Rf": "--force-refresh", "-ud": "--update-exact"}

_KEY_PREFIXES = ("T1_", "T2_", "T2S_", "T2P_", "T2C_", "T2CK_")
_LIMIT = re.compile(r"^(?:[a-z]+:[^=\s]+=)?\d+/\d+s?$", re.IGNORECASE)
_KV = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*=.*$")
_HOST = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*(:\d{1,5})?$")


@dataclass
class Expanded:
    argv: list[str]
    #: What -b decided, as (string, option) pairs, for the "I read that
    #: as ..." line waiter prints so a guess is never silent.
    assigned: list[tuple[str, str]] = field(default_factory=list)


def _is_option(token: str) -> bool:
    return token.startswith("-") and token != "-" and not token[1:].replace(".", "", 1).isdigit()


def _expand_cluster(token: str) -> list[str]:
    """``-ArEK`` -> ``['--auto', '--refresh', '--encrypt', '--key']``."""
    letters = token[1:]
    out: list[str] = []
    i = 0
    while i < len(letters):
        pair = letters[i:i + 2]
        if pair in _PAIRS:
            out.append(_PAIRS[pair])
            i += 2
            continue
        letter = letters[i]
        last = i == len(letters) - 1
        if letter in VALUE_LETTERS:
            if last:
                out.append(VALUE_LETTERS[letter])
            elif letter == "r":
                out.append("--refresh")
            else:
                raise ServArgsError(
                    f"-{letter} takes a value, so it has to end its group: in {token!r} it is "
                    f"followed by {letters[i + 1:]!r}. Put it last (-{letters.replace(letter, '')}{letter} "
                    f"<value>) or on its own (-{letter} <value>)."
                )
        elif letter in FLAG_LETTERS:
            out.append(FLAG_LETTERS[letter])
        else:
            raise ServArgsError(f"-{letter} (in {token!r}) is not a waiter serv option.")
        i += 1
    return out


def _takes_value(option: str) -> bool:
    return option in VALUE_LETTERS.values() or option in {
        f"-{letter}" for letter in VALUE_LETTERS
    }


def _classify(text: str) -> str | None:
    """The option a bare string belongs to, or None if it is unclear."""
    if text.startswith(_KEY_PREFIXES):
        return "--key"
    if _LIMIT.match(text):
        return "--force-limit"
    if text.startswith(("http://", "https://")):
        return "--server"
    if "/" in text:
        try:
            ipaddress.ip_network(text, strict=False)
            return "cidr"
        except ValueError:
            pass
    if _KV.match(text):
        return "--config"
    if os.path.isdir(text) and os.path.isfile(os.path.join(text, "kit.json")):
        return "--kit-install"
    if text.lower().endswith(".zip") and os.path.isfile(text):
        return "--kit-install"
    if os.path.isfile(text):
        return "--config-file"
    if text.isdigit() and 1 <= int(text) <= 65535:
        return "--port"
    try:
        ipaddress.ip_address(text.strip("[]"))
        return "--server"
    except ValueError:
        pass
    if text == "localhost" or ("." in text and _HOST.match(text)):
        return "--server"
    return None


_REPEATABLE = {"--config", "--force-limit"}


def expand(argv: list[str]) -> Expanded:
    """Rewrite ``waiter serv`` arguments into what argparse can read."""
    tokens: list[str] = []
    loose: list[str] = []
    expecting_value = False
    for token in argv:
        if expecting_value:
            tokens.append(token)
            expecting_value = False
            continue
        if token in _EXACT:
            tokens.append(_EXACT[token])
            continue
        if token.startswith("--") or not _is_option(token):
            if token.startswith("--"):
                tokens.append(token)
                expecting_value = "=" not in token and _takes_value(token.split("=", 1)[0])
            else:
                loose.append(token)
            continue
        if len(token) == 2:
            tokens.append(token)
            expecting_value = token[1] in VALUE_LETTERS
            continue
        expanded = _expand_cluster(token)
        tokens.extend(expanded)
        expecting_value = _takes_value(expanded[-1])

    if not loose:
        return Expanded(tokens)
    if "--bundle" not in tokens and "-b" not in tokens:
        raise ServArgsError(
            f"{loose[0]!r} has no option in front of it. Say which it is (-I, -K, -P, ...), "
            "or add -b and waiter will work it out."
        )
    assigned: list[tuple[str, str]] = []
    taken: dict[str, str] = {}
    for text in loose:
        option = _classify(text)
        if option == "cidr":
            raise ServArgsError(
                f"{text!r} is an address range: it could be -B (block), -W (allow) or -a "
                "(appeal). Say which."
            )
        if option is None:
            raise ServArgsError(f"-b could not tell what {text!r} is for. Put the option in front of it.")
        if option not in _REPEATABLE:
            if option in taken:
                raise ServArgsError(
                    f"{taken[option]!r} and {text!r} both look like {option}. Put the option "
                    "in front of the one you mean."
                )
            explicit = {"--key": "-K", "--server": "-I", "--port": "-P", "--config-file": "-F",
                        "--kit-install": "-k"}.get(option)
            if option in tokens or (explicit and explicit in tokens):
                raise ServArgsError(f"{text!r} looks like {option}, which is already given.")
            taken[option] = text
        tokens.extend([option, text])
        assigned.append((text, option))
    return Expanded(tokens, assigned)
