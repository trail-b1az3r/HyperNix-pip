"""toolcalls — reading a tool call out of whatever a model wrote.

Why this exists
---------------
A hosted model returns tool calls as structured ``tool_calls``. A local
GGUF usually does not. llama.cpp produces structured calls only when the
model's chat template supports it, and most models were trained on one
of half a dozen *text* conventions instead::

    <tool_call>{"name": "x", "arguments": {...}}</tool_call>   Hermes, Qwen
    <|python_tag|>{"name": "x", "parameters": {...}}            Llama 3.1+
    <function=x>{...}</function>                                Llama 3.x
    [TOOL_CALLS][{"name": "x", "arguments": {...}}]             Mistral
    [TOOL_REQUEST]{"name": "x", "arguments": {...}}[END_TOOL_REQUEST]
                                                                LM Studio's own
    <start_function_call>call:x{k:<escape>v<escape>}<end_function_call>
    <|tool_call>call:x{...}<tool_call|>                         Gemma
    ```tool_code  x(k="v")  ```                                 Gemma 3
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>x {...}<｜tool▁call▁end｜>
                                                                DeepSeek
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

import ast
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
    #: "llama-function", "mistral", "lmstudio", "gemma", "tool_code",
    #: "python", "deepseek", "fenced", "bare".
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
# LM Studio's default convention for a model with no native tool
# template. LM Studio parses it itself when it can, and passes it
# through as text when it cannot -- which is when it reached here unread.
_LMSTUDIO = re.compile(r"\[TOOL_REQUEST\]\s*(.*?)\s*(?:\[END_TOOL_REQUEST\]|$)", re.S)
_FUNCTION_TAG = re.compile(r"<function=([A-Za-z0-9_.\-]+)>\s*(.*?)\s*(?:</function>|$)", re.S)
_GEMMA = re.compile(
    r"(?:<start_function_call>|<\|tool_call>)\s*call:([A-Za-z0-9_.\-]+)\s*(\{.*?\})\s*"
    r"(?:<end_function_call>|<tool_call\|>|$)", re.S)
_TOOL_CODE = re.compile(r"```tool_code\s*(.*?)\s*```", re.S)
_DEEPSEEK = re.compile(
    r"<｜tool▁call▁begin｜>\s*(?:function)?\s*<｜tool▁sep｜>\s*([A-Za-z0-9_.\-]+)\s*"
    r"(?:```(?:json)?\s*)?(\{.*?\})\s*(?:```)?\s*<｜tool▁call▁end｜>", re.S)
_PY_CALL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*\s*\(.*\)$", re.S)
_ESCAPED = re.compile(r"<escape>(.*?)<escape>", re.S)
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _gemma_arguments(raw: str) -> tuple[Any, bool]:
    """``{location:<escape>London<escape>,days:3}`` as a dict.

    Gemma marks strings with ``<escape>`` and leaves keys bare. The
    strings become JSON strings first, so a colon or comma inside one is
    not mistaken for syntax, and then the keys outside them are quoted.
    """
    text = _ESCAPED.sub(lambda m: json.dumps(m.group(1)), raw)
    text = _outside_strings(text, lambda seg: _BARE_KEY.sub(r'\1"\2":', seg))
    value, _ = repair_json(text)
    return value, True


def _python_calls(source: str, schemas: dict[str, list[str]]) -> list[tuple[str, dict[str, Any]]]:
    """``name(a=1, b="x")`` statements as ``(name, arguments)``.

    Read with ``ast`` and ``literal_eval`` only: nothing is executed, and
    an argument that is not a literal is refused rather than guessed at.
    Positional arguments take the tool's parameters in schema order.
    """
    try:
        tree = ast.parse(source.strip())
    except SyntaxError:
        return []
    calls = []
    for statement in tree.body:
        node = statement.value if isinstance(statement, ast.Expr) else None
        # `print(tool(...))` is how Gemma's own examples write it.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print" and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)):
            node = node.args[0]
        if not isinstance(node, ast.Call):
            return []
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if name not in schemas:
            return []
        try:
            arguments = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
            for key, value in zip(schemas[name], node.args, strict=False):
                arguments[key] = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            return []
        calls.append((name, arguments))
    return calls


def _embedded_json_calls(text: str, offered: set[str]) -> list[tuple[int, int, dict[str, Any]]]:
    """``{"name": ..., "arguments": ...}`` objects written inside prose.

    Only an object naming an offered tool, with arguments, counts:
    anything else a model writes in JSON is its answer.
    """
    decoder = json.JSONDecoder()
    found = []
    start = text.find("{")
    looked = 0
    while start >= 0 and looked < 50:
        looked += 1
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if (isinstance(value, dict) and value.get("name") in offered
                and ("arguments" in value or "parameters" in value)):
            found.append((start, end, value))
            start = text.find("{", end)
        else:
            start = text.find("{", start + 1)
    return found


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
    schemas = {
        t["function"]["name"]: list((t["function"].get("parameters") or {}).get("properties") or {})
        for t in tools or []
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
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

    def take_named(pattern, source, parse=repair_json):
        """Patterns whose first group is the name and second the arguments."""
        for match in pattern.finditer(text):
            if any(a <= match.start() < b for a, b in spans):
                continue
            spans.append(match.span())
            try:
                args, repaired = parse(match.group(2)) if match.group(2).strip() else ({}, False)
            except ValueError as exc:
                result.failures.append((match.group(0)[:200], str(exc)))
                continue
            result.calls += _as_calls({"name": match.group(1), "arguments": args},
                                      source, repaired, offered)

    take(_HERMES, "hermes")
    take(_LLAMA, "llama")
    take(_MISTRAL, "mistral")
    take(_LMSTUDIO, "lmstudio")
    take_named(_FUNCTION_TAG, "llama-function")
    take_named(_GEMMA, "gemma", parse=_gemma_arguments)
    take_named(_DEEPSEEK, "deepseek")
    if offered is not None:
        take(_FENCED, "fenced")
        for match in _TOOL_CODE.finditer(text):
            if any(a <= match.start() < b for a, b in spans):
                continue
            for name, args in _python_calls(match.group(1), schemas):
                result.calls.append(ToolCall(name=name, arguments=args, source="tool_code"))
                spans.append(match.span())
        if not result.calls and not result.failures:
            stripped = text.strip()
            # A reply that is only a call, written as Python: what Gemma
            # sends when it leaves the fence off.
            if _PY_CALL.match(stripped) and "\n" not in stripped:
                for name, args in _python_calls(stripped, schemas):
                    result.calls.append(ToolCall(name=name, arguments=args, source="python"))
                    spans.append((0, len(text)))
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
        if not result.calls and not result.failures:
            # A call after a sentence: "Let me check. {"name": ...}".
            for start, end, value in _embedded_json_calls(text, names):
                result.calls += _as_calls(value, "bare", False, offered)
                spans.append((start, end))

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

    It also says the tools are real and when to reach for one, and shows
    the whole exchange, call, result and answer (0.72.6.post1). With only
    the call format, a small model treated the tools as optional
    decoration: it said it had no access to things a tool could reach,
    or stopped after calling one as though the call were the answer.
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
        "These tools are real. Calling one runs it on this server and gives you "
        "its actual result. When a question is about something a tool can "
        "reach, call the tool: do not say you cannot access it, and do not "
        "guess what it would say.",
        "",
        f"To call a tool, write the call on its own and stop. This one calls "
        f"{first['name']}:",
        f"<tool_call>{example}</tool_call>",
        "",
        "The result comes back to you, as a tool message or as "
        "<tool_response>…</tool_response>, and you then answer from it. "
        "A whole exchange looks like this:",
        f"User: (a question {first['name']} can answer)",
        f"Assistant: <tool_call>{example}</tool_call>",
        f'<tool_response name="{first["name"]}">(the real result)</tool_response>',
        "Assistant: (the answer, using what the result said)",
        "",
        "Arguments are a JSON object with double-quoted keys and the "
        "parameter names exactly as listed. Call one tool at a time and wait "
        "for its result; never write a <tool_response> yourself. Only call "
        "tools listed above. If no tool fits, just answer.",
    ]
    return "\n".join(lines)


def _example_value(schema: dict[str, Any]) -> Any:
    if schema.get("enum"):
        return schema["enum"][0]
    return {"string": "…", "integer": 1, "number": 1.0, "boolean": True,
            "array": [], "object": {}}.get(schema.get("type"), "…")
