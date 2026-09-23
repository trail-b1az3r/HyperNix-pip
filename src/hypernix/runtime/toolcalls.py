"""toolcalls — reading a tool call out of whatever a model wrote.

Why this exists
---------------
A hosted model returns tool calls as structured ``tool_calls``. A local
GGUF usually does not. llama.cpp produces structured calls only when the
model's chat template supports it, and most models were trained on one
of half a dozen *text* conventions instead::

    <tool_call>{"name": "x", "arguments": {...}}</tool_call>   Hermes, Qwen
    <|python_tag|>{"name": "x", "parameters": {...}}            Llama 3.1+
    [TOOL_CALLS][{"name": "x", "arguments": {...}}]             Mistral
    ```json {"name": "x", "arguments": {...}} ```               most others

When nothing reads those, the call is shown to the person as if it were
the answer and nothing runs. That — not the model's reasoning — is most
of why local models look bad at calling API endpoints. This module reads
all of them.

Then it repairs what models get wrong in JSON (trailing commas, Python's
``True``/``None``, single quotes, a missing closing brace) and validates
the arguments against the tool's schema, with an error that says exactly
which field is wrong — because the model is shown it, and a model told
"``limit`` must be an integer, got the string 'ten'" fixes itself where
one told "invalid arguments" guesses.

One thing it will not do: run a tool whose name it had to guess. A
misspelt name comes back as an error listing the nearest real ones. Auto-
correcting ``delete_model`` to ``delete_models`` is how a typo becomes an
action nobody asked for.
"""
from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..system import errorcatalogue as codes

__all__ = [
    "ToolCall",
    "Extraction",
    "extract_calls",
    "repair_json",
    "validate_arguments",
    "tool_prompt",
    "strip_call_markup",
]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])
    #: Which convention it arrived in: "structured", "hermes", "llama",
    #: "mistral", "fenced", "bare".
    source: str = "structured"
    #: Set when the arguments had to be repaired to parse.
    repaired: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name,
                         "arguments": json.dumps(self.arguments)},
        }


@dataclass
class Extraction:
    calls: list[ToolCall] = field(default_factory=list)
    #: Calls found but unreadable, as ``(raw text, why)``. Reported back
    #: to the model rather than dropped, so it can try again.
    failures: list[tuple[str, str]] = field(default_factory=list)
    #: The reply with the call markup removed — what the person sees.
    text: str = ""


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}


def repair_json(raw: str) -> tuple[Any, bool]:
    """Parse *raw*, repairing the mistakes models make. ``(value, repaired)``.

    Raises ValueError when it cannot. Each repair is one a model
    genuinely produces; this is not a general-purpose JSON5 parser, and
    the more it guesses the likelier it is to "repair" something into a
    different meaning.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty")
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    fixed = text
    # Python literals outside strings.
    fixed = _outside_strings(fixed, lambda seg: re.sub(
        r"\b(True|False|None)\b", lambda m: _PY_LITERALS[m.group(1)], seg))
    # Trailing commas before a closer.
    fixed = _outside_strings(fixed, lambda seg: re.sub(r",\s*([}\]])", r"\1", seg))
    # Single-quoted strings, only when there are no double quotes at all —
    # mixing the two is ambiguous, and guessing turns an apostrophe in a
    # value into a string boundary.
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    # Unbalanced closers at the end: a model that stopped one brace short.
    opens = fixed.count("{") - fixed.count("}")
    brackets = fixed.count("[") - fixed.count("]")
    if 0 < opens <= 3 and brackets >= 0:
        fixed = fixed + "]" * brackets + "}" * opens
    try:
        return json.loads(fixed), True
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON even after repair: {exc.msg} "
                         f"at character {exc.pos}") from None


def _outside_strings(text: str, transform) -> str:
    """Apply *transform* only to the parts of *text* outside "strings"."""
    out, buf, in_str, escaped = [], [], False, False
    for ch in text:
        if in_str:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                out.append("".join(buf))
                buf, in_str = [], False
        elif ch == '"':
            out.append(transform("".join(buf)))
            buf, in_str = ['"'], True
        else:
            buf.append(ch)
    tail = "".join(buf)
    out.append(tail if in_str else transform(tail))
    return "".join(out)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_HERMES = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_LLAMA = re.compile(r"<\|python_tag\|>\s*(.*?)\s*(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.S)
_MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.S)
_FENCED = re.compile(r"```(?:json|tool_call|tool)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def _as_calls(value: Any, source: str, repaired: bool,
              tools: set[str] | None) -> list[ToolCall]:
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        # A bare JSON object is only a call if it names a tool we offer.
        # Otherwise any JSON a model writes as its *answer* — a config
        # file it was asked for — would be executed.
        if source in ("fenced", "bare") and tools is not None and name not in tools:
            continue
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args, fixed = repair_json(args)
                repaired = repaired or fixed
            except ValueError:
                args = {"_unparsed": args}
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append(ToolCall(name=name, arguments=args, source=source,
                              repaired=repaired))
    return calls


def extract_calls(
    message: dict[str, Any] | str,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> Extraction:
    """Every tool call in *message*, however it was written.

    *message* is an OpenAI-shaped assistant message or just its text.
    *tools* (the OpenAI tool list offered to the model) makes the
    fenced and bare-JSON forms safe: without it, those are not read.
    """
    names = {t["function"]["name"] for t in tools or []
             if isinstance(t, dict) and isinstance(t.get("function"), dict)}
    offered = names if tools is not None else None
    result = Extraction()

    if isinstance(message, dict):
        structured = message.get("tool_calls") or []
        for raw in structured:
            if not isinstance(raw, dict):
                continue
            calls = _as_calls(raw, "structured", False, offered)
            for call in calls:
                call.id = raw.get("id") or call.id
            result.calls += calls
        text = message.get("content") or ""
        if result.calls:
            result.text = text
            return result
    else:
        text = message or ""

    spans: list[tuple[int, int]] = []

    def take(pattern, source):
        for match in pattern.finditer(text):
            if any(a <= match.start() < b for a, b in spans):
                continue
            try:
                value, repaired = repair_json(match.group(1))
            except ValueError as exc:
                result.failures.append((match.group(1)[:200], str(exc)))
                spans.append(match.span())
                continue
            found = _as_calls(value, source, repaired, offered)
            if found:
                result.calls += found
                spans.append(match.span())

    take(_HERMES, "hermes")
    take(_LLAMA, "llama")
    take(_MISTRAL, "mistral")
    if offered is not None:
        take(_FENCED, "fenced")
        if not result.calls and not result.failures:
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    value, repaired = repair_json(stripped)
                    found = _as_calls(value, "bare", repaired, offered)
                    if found:
                        result.calls += found
                        spans.append((0, len(text)))
                except ValueError:
                    pass

    result.text = strip_call_markup(text, spans)
    return result


def strip_call_markup(text: str, spans: list[tuple[int, int]]) -> str:
    """*text* without the call markup — what the person should see."""
    if not spans:
        return text.strip()
    keep, cursor = [], 0
    for start, end in sorted(spans):
        keep.append(text[cursor:start])
        cursor = max(cursor, end)
    keep.append(text[cursor:])
    return re.sub(r"\n{3,}", "\n\n", "".join(keep)).strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TYPES = {"string": str, "integer": int, "number": (int, float),
          "boolean": bool, "object": dict, "array": list}


def validate_arguments(
    call: ToolCall, tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], str | None]:
    """``(arguments, None)`` if usable, else ``(arguments, error for the model)``.

    Coerces only what is unambiguous — ``"10"`` to 10 for an integer
    field, ``"true"`` to True for a boolean. Everything else is an error
    naming the field, the expected type, and what arrived.
    """
    by_name = {t["function"]["name"]: t["function"] for t in tools
               if isinstance(t.get("function"), dict)}
    spec = by_name.get(call.name)
    if spec is None:
        near = difflib.get_close_matches(call.name, list(by_name), n=3, cutoff=0.5)
        hint = f" Did you mean {' or '.join(near)}?" if near else ""
        return call.arguments, (
            f"{codes.TOOLCALL_UNKNOWN.code}: there is no tool called "
            f"{call.name!r}.{hint} Available: {', '.join(sorted(by_name))}."
        )

    schema = spec.get("parameters") or {}
    props = schema.get("properties") or {}
    args = dict(call.arguments)
    if "_unparsed" in args:
        return args, (f"{codes.TOOLCALL_BAD_ARGUMENTS.code}: the arguments were "
                      f"not valid JSON. Send them as a JSON object.")

    problems = []
    for required in schema.get("required") or []:
        if required not in args:
            problems.append(f"`{required}` is required")
    for key, value in list(args.items()):
        if key not in props:
            if schema.get("additionalProperties") is False:
                problems.append(f"`{key}` is not a parameter "
                                f"(known: {', '.join(sorted(props)) or 'none'})")
            continue
        want = props[key].get("type")
        coerced, ok = _coerce(value, want)
        if ok:
            args[key] = coerced
        else:
            problems.append(f"`{key}` must be {want}, got "
                            f"{type(value).__name__} {value!r}"[:160])
            continue
        allowed = props[key].get("enum")
        if allowed and args[key] not in allowed:
            problems.append(f"`{key}` must be one of {allowed}, got {args[key]!r}")
    if problems:
        return args, (f"{codes.TOOLCALL_BAD_ARGUMENTS.code}: {call.name}: "
                      + "; ".join(problems) + ".")
    return args, None


def _coerce(value: Any, want: str | None) -> tuple[Any, bool]:
    if want is None:
        return value, True
    kind = _TYPES.get(want)
    if kind is None:
        return value, True
    # bool is an int in Python; a JSON true is not a valid integer.
    if want in ("integer", "number") and isinstance(value, bool):
        return value, False
    if isinstance(value, kind):
        return value, True
    if isinstance(value, str):
        text = value.strip()
        if want == "integer" and re.fullmatch(r"-?\d+", text):
            return int(text), True
        if want == "number" and re.fullmatch(r"-?\d+(\.\d+)?", text):
            return float(text), True
        if want == "boolean" and text.lower() in ("true", "false"):
            return text.lower() == "true", True
    if want == "number" and isinstance(value, int):
        return value, True
    if want == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value), True
    return value, False


# ---------------------------------------------------------------------------
# Teaching a model the format
# ---------------------------------------------------------------------------


def tool_prompt(tools: list[dict[str, Any]]) -> str:
    """A system prompt that teaches a model without native tool support.

    One format, stated once, with one worked example built from a real
    tool — a model copies an example far more reliably than it follows a
    description, and an example using a tool that exists cannot teach it
    to call one that does not.
    """
    if not tools:
        return ""
    lines = ["You can call these tools:"]
    for tool in tools:
        fn = tool.get("function") or {}
        params = (fn.get("parameters") or {}).get("properties") or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        sig = ", ".join(
            f"{k}{'' if k in required else '?'}: {v.get('type', 'any')}"
            for k, v in params.items()
        )
        lines.append(f"- {fn.get('name')}({sig}) — {fn.get('description', '').strip()}")
    first = tools[0]["function"]
    example_args = {
        k: _example_value(v) for k, v in
        ((first.get("parameters") or {}).get("properties") or {}).items()
        if k in set((first.get("parameters") or {}).get("required") or [])
    }
    example = json.dumps({"name": first["name"], "arguments": example_args})
    lines += [
        "",
        "To call one, reply with exactly this and nothing else on those lines:",
        f"<tool_call>{example}</tool_call>",
        "",
        "Arguments are a JSON object with double-quoted keys. You will get "
        "the result back, then continue. Only call tools listed above. If "
        "you do not need a tool, just answer.",
    ]
    return "\n".join(lines)


def _example_value(schema: dict[str, Any]) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    return {"string": "…", "integer": 1, "number": 1.0, "boolean": True,
            "array": [], "object": {}}.get(schema.get("type"), "…")
