"""noodle.providers — one interface over every model Noodle can drive.

Nine backends, three shapes. Most of the industry settled on OpenAI's
chat-completions wire format, so the majority of these are one base URL
and one header apart. Anthropic and Google are not, and pretending they
were is how a "universal" client ends up silently dropping system
prompts or tool calls.

===============  =================================  ================
Provider         Wire format                        Credential
===============  =================================  ================
OpenAI           OpenAI                             ``OPENAI_API_KEY``
Anthropic        Anthropic Messages                 ``ANTHROPIC_API_KEY``
Kimi (Moonshot)  OpenAI                             ``MOONSHOT_API_KEY``
Google Gemini    Gemini generateContent             ``GEMINI_API_KEY``
Qwen (DashScope) OpenAI                             ``DASHSCOPE_API_KEY``
xAI Grok         OpenAI                             ``XAI_API_KEY``
HyperNix T1      HyperNix bridge                    a T1/T2 key
Ollama           OpenAI (``/v1``)                   none
vLLM             OpenAI                             optional
===============  =================================  ================

Standard library only. An agent runner has to work on the machine with
the GPU, which is frequently a machine with nothing else installed.

What this module refuses to do
------------------------------
It does not silently fall back to a different provider when one fails.
A swarm that quietly re-routes a task from a local 7B to a paid frontier
model is a swarm that produces a surprising invoice, and one that
re-routes the other way produces surprising output. Failures surface;
:class:`~hypernix.interfaces.noodle.swarm.Swarm` decides what to do
about them, visibly.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

logger = logging.getLogger(__name__)

__all__ = [
    "Provider",
    "ProviderSpec",
    "PROVIDERS",
    "ModelClient",
    "ProviderError",
    "ChatResult",
    "ToolCall",
    "build_client",
    "available_providers",
]


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    KIMI = "kimi"
    GEMINI = "gemini"
    QWEN = "qwen"
    GROK = "grok"
    HYPERNIX = "hypernix"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ProviderError(RuntimeError):
    """A provider call failed. ``code`` is stable for branching.

    ``retryable`` is the field the swarm reads: a 429 or a 503 is worth
    trying again, a 401 never is, and burning a retry budget on an
    authentication failure just makes the eventual error slower.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "error",
        provider: str = "",
        status: int = 0,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.provider = provider
        self.status = status
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "provider": self.provider,
            "status": self.status,
            "retryable": self.retryable,
            "message": str(self),
        }


@dataclass(frozen=True)
class ProviderSpec:
    """Everything that differs between one provider and another."""

    provider: Provider
    label: str
    wire: str                       # openai | anthropic | gemini
    default_base_url: str
    env_keys: tuple[str, ...]
    default_model: str
    #: Does this provider bill per token? Drives the swarm's cost
    #: accounting and the "are you sure" on a large fan-out.
    paid: bool = True
    supports_tools: bool = True
    notes: str = ""

    def resolve_key(self, explicit: str = "") -> str:
        if explicit:
            return explicit
        for name in self.env_keys:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "label": self.label,
            "wire": self.wire,
            "base_url": self.default_base_url,
            "env_keys": list(self.env_keys),
            "default_model": self.default_model,
            "paid": self.paid,
            "supports_tools": self.supports_tools,
            "notes": self.notes,
        }


PROVIDERS: dict[Provider, ProviderSpec] = {
    spec.provider: spec
    for spec in (
        ProviderSpec(
            Provider.OPENAI, "OpenAI", "openai", "https://api.openai.com/v1",
            ("OPENAI_API_KEY",), "gpt-4o-mini",
        ),
        ProviderSpec(
            Provider.ANTHROPIC, "Anthropic Claude", "anthropic",
            "https://api.anthropic.com/v1", ("ANTHROPIC_API_KEY",), "claude-sonnet-4-5",
            notes="Messages API: system is a top-level field, not a message.",
        ),
        ProviderSpec(
            Provider.KIMI, "Moonshot Kimi", "openai", "https://api.moonshot.cn/v1",
            ("MOONSHOT_API_KEY", "KIMI_API_KEY"), "moonshot-v1-32k",
            notes="Set MOONSHOT_BASE_URL to https://api.moonshot.ai/v1 outside mainland China.",
        ),
        ProviderSpec(
            Provider.GEMINI, "Google Gemini", "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "gemini-2.0-flash",
            notes="generateContent: 'contents' with 'parts', and roles are user/model.",
        ),
        ProviderSpec(
            Provider.QWEN, "Qwen (DashScope)", "openai",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ("DASHSCOPE_API_KEY", "QWEN_API_KEY"), "qwen-plus",
        ),
        ProviderSpec(
            Provider.GROK, "xAI Grok", "openai", "https://api.x.ai/v1",
            ("XAI_API_KEY", "GROK_API_KEY"), "grok-2-latest",
        ),
        ProviderSpec(
            Provider.HYPERNIX, "HyperNix T1", "openai", "http://localhost:8000",
            ("HYPERNIX_T1_KEY", "T1_KEY"), "",
            paid=False,
            notes="Routed through the T1 API's LM Studio bridge; the model is whatever is loaded.",
        ),
        ProviderSpec(
            Provider.OLLAMA, "Ollama", "openai", "http://localhost:11434/v1",
            (), "llama3.2", paid=False,
            notes="No key. The /v1 path is Ollama's OpenAI-compatible surface.",
        ),
        ProviderSpec(
            Provider.VLLM, "vLLM", "openai", "http://localhost:8000/v1",
            ("VLLM_API_KEY",), "", paid=False,
            notes="Model name must match what vLLM was started with.",
        ),
    )
}


def available_providers(*, include_local: bool = True) -> list[ProviderSpec]:
    """Providers this machine could actually use right now.

    Local providers are included without a key because they do not need
    one; whether they are *running* is a separate question and one that
    only a probe can answer.
    """
    out: list[ProviderSpec] = []
    for spec in PROVIDERS.values():
        if not spec.paid:
            # A local backend's key is optional (vLLM) or absent
            # (Ollama). Excluding it for want of a key it does not need
            # would hide the only provider on an offline machine.
            if include_local:
                out.append(spec)
            continue
        if spec.resolve_key():
            out.append(spec)
    return out


@dataclass
class ToolCall:
    """A model asking for a tool to be run."""

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"call_id": self.call_id, "name": self.name, "arguments": dict(self.arguments)}


@dataclass
class ChatResult:
    """One model reply, normalised across every wire format."""

    content: str
    model: str
    provider: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "seconds": round(self.seconds, 2),
        }


class ModelClient:
    """Talks to one provider. Thread-safe; the swarm runs many at once."""

    def __init__(
        self,
        provider: Provider | str,
        *,
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        timeout: float = 180.0,
    ) -> None:
        self.spec = PROVIDERS[Provider(provider)]
        self.model = model or self.spec.default_model
        self.api_key = self.spec.resolve_key(api_key)
        self.base_url = (
            base_url
            or os.environ.get(f"{self.spec.provider.value.upper()}_BASE_URL", "")
            or self.spec.default_base_url
        ).rstrip("/")
        self.timeout = float(timeout)

        if self.spec.env_keys and not self.api_key and self.spec.paid:
            raise ProviderError(
                f"{self.spec.label} needs a key. Set one of: "
                f"{', '.join(self.spec.env_keys)}",
                code="missing_key",
                provider=self.spec.provider.value,
            )

    # -- transport ----------------------------------------------------

    def _post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
            method="POST",
        )
        try:
            with safe_urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace") or "{}")
        except urllib.error.HTTPError as exc:
            detail = self._error_detail(exc)
            # Retryable is the field the swarm actually reads: burning a
            # retry budget on a 401 only makes the error arrive slower.
            retryable = exc.code in (408, 409, 425, 429, 500, 502, 503, 504)
            raise ProviderError(
                f"{self.spec.label} returned HTTP {exc.code}: {detail}",
                code="rate_limited" if exc.code == 429 else "http_error",
                provider=self.spec.provider.value,
                status=exc.code,
                retryable=retryable,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                f"Could not reach {self.spec.label} at {self.base_url}: {exc}",
                code="unreachable",
                provider=self.spec.provider.value,
                retryable=True,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.spec.label} returned a non-JSON body",
                code="bad_response",
                provider=self.spec.provider.value,
            ) from exc

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return exc.reason or ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw[:300]
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or raw[:300])
        if isinstance(error, str):
            return error
        return raw[:300]

    # -- chat ---------------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system: str = "",
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """One completion, in whichever wire format this provider speaks."""
        started = time.monotonic()
        if self.spec.wire == "anthropic":
            result = self._chat_anthropic(messages, system, tools, temperature, max_tokens)
        elif self.spec.wire == "gemini":
            result = self._chat_gemini(messages, system, tools, temperature, max_tokens)
        else:
            result = self._chat_openai(messages, system, tools, temperature, max_tokens)
        result.seconds = time.monotonic() - started
        return result

    def _chat_openai(self, messages, system, tools, temperature, max_tokens) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if system:
            body["messages"] = [{"role": "system", "content": system}, *messages]
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._post("/chat/completions", body, headers)

        choices = payload.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        calls = [
            ToolCall(
                call_id=str(c.get("id") or ""),
                name=str((c.get("function") or {}).get("name") or ""),
                arguments=_loads((c.get("function") or {}).get("arguments") or "{}"),
            )
            for c in (message.get("tool_calls") or [])
            if isinstance(c, dict)
        ]
        usage = payload.get("usage") or {}
        return ChatResult(
            content=str(message.get("content") or ""),
            model=str(payload.get("model") or self.model),
            provider=self.spec.provider.value,
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=str(choices[0].get("finish_reason") or "") if choices else "",
            raw=payload,
        )

    def _chat_anthropic(self, messages, system, tools, temperature, max_tokens) -> ChatResult:
        # The Messages API takes system as a top-level field. Passing it
        # as a message role is accepted and then ignored, which is the
        # kind of silent behaviour difference that makes a "universal"
        # client worse than no client.
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m for m in messages if m.get("role") != "system"],
            "max_tokens": max_tokens or 4096,
        }
        inline_system = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        combined = " ".join(x for x in (system, inline_system) if x).strip()
        if combined:
            body["system"] = combined
        if tools:
            body["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
                }
                for t in tools
            ]
        if temperature is not None:
            body["temperature"] = temperature

        payload = self._post(
            "/messages", body,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        text_parts, calls = [], []
        for block in payload.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        call_id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=dict(block.get("input") or {}),
                    )
                )
        usage = payload.get("usage") or {}
        return ChatResult(
            content="".join(text_parts),
            model=str(payload.get("model") or self.model),
            provider=self.spec.provider.value,
            tool_calls=calls,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            finish_reason=str(payload.get("stop_reason") or ""),
            raw=payload,
        )

    def _chat_gemini(self, messages, system, tools, temperature, max_tokens) -> ChatResult:
        contents = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            contents.append(
                {
                    # Gemini calls the assistant "model". Sending
                    # "assistant" is silently dropped.
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": str(message.get("content") or "")}],
                }
            )
        body: dict[str, Any] = {"contents": contents}
        inline_system = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "system"
        )
        combined = " ".join(x for x in (system, inline_system) if x).strip()
        if combined:
            body["systemInstruction"] = {"parts": [{"text": combined}]}
        generation: dict[str, Any] = {}
        if temperature is not None:
            generation["temperature"] = temperature
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        if generation:
            body["generationConfig"] = generation
        if tools:
            body["tools"] = [{"functionDeclarations": list(tools)}]

        payload = self._post(
            f"/models/{self.model}:generateContent?key={self.api_key}", body, {}
        )
        candidates = payload.get("candidates") or []
        parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
        text = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
        calls = [
            ToolCall(
                call_id=str(p["functionCall"].get("name") or ""),
                name=str(p["functionCall"].get("name") or ""),
                arguments=dict(p["functionCall"].get("args") or {}),
            )
            for p in parts
            if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)
        ]
        usage = payload.get("usageMetadata") or {}
        return ChatResult(
            content=text,
            model=self.model,
            provider=self.spec.provider.value,
            tool_calls=calls,
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
            finish_reason=str(candidates[0].get("finishReason") or "") if candidates else "",
            raw=payload,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ModelClient({self.spec.provider.value}, model={self.model!r})"


def _loads(text: str) -> dict[str, Any]:
    """Parse tool arguments, tolerating a model that emitted nothing.

    A model that returns ``""`` or malformed JSON for a no-argument tool
    is common enough that failing the whole turn over it would make the
    swarm markedly less reliable than the models it drives.
    """
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_client(spec: str, **kwargs: Any) -> ModelClient:
    """Build a client from ``"provider"`` or ``"provider:model"``.

    The colon form is what the CLI and the swarm roster take, because
    "openai:gpt-4o-mini" is one token to type and unambiguous to parse.
    """
    provider, _, model = spec.partition(":")
    return ModelClient(provider.strip().lower(), model=model.strip(), **kwargs)
