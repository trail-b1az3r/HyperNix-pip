"""hyped — high-quality TUI autonomous agent CLI for the HyperNix family.

V0.71.2: Transformed into a full TUI Agent (similar to openclaw, Claude code CLI,
Claude code app, Qwen 3 coder CLI).

Key Capabilities:
* **Multi-Provider Support**: Local HyperNix models, HyperNix package models,
  OpenAI style API keys, Anthropic API keys, Custom REST API endpoints, and
  HyperNix HNX1/T1 API keys (via Keymaster & Gatekeeper).
* **Extended Model Catalog**: 11 model families (HyperNix, Nix, Qwen 3.5, Nano,
  LLaMA 3, DeepSeek, Mistral, Gemma, Phi, OpenAI, Anthropic) and 25+ curated models.
* **34+ Built-in Unique Tools**: File management, command execution, web search & fetch,
  git operations, code syntax analysis, tree visualization, Keymaster key management,
  Gatekeeper quota monitoring, and HyperNix lifecycle pipelines (download/convert/quantize/train).
* **AI Self-Created Tool Creation Skills**: Dynamic Skill Creator engine allowing the agent
  or user to create, list, execute, and delete custom Python/bash skills stored persistently.
* **TUI Visual Aesthetics**: OpenClaw-inspired multi-panel layout, rolling 256-color gradient
  headers, real-time tool execution cards, agent reasoning blocks, and status bar.
* **Agentic ReAct Loop**: Anti-hallucination system prompts, empirical verification,
  file-grounded code editing, and multi-step autonomous tool execution.
"""
from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypernix.chat import menu as _menu
from hypernix.models.download import KNOWN_MODELS

try:
    import readline  # noqa: F401
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# Version & Defaults
# ---------------------------------------------------------------------------

HYPED_VERSION = "v0.71.4b2"
SKILLS_DIR = Path.home() / ".hypernix" / "skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Curated Model Catalog (14 Families, 40+ Models)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    short: str
    repo_id: str
    label: str
    family: str = ""
    badge: str = ""
    provider: str = "local"  # local | openai | anthropic | rest | t1


CURATED_MODELS: tuple[ModelEntry, ...] = (
    # HyperNix Family
    ModelEntry("hyper-nix.2",       "ray0rf1re/hyper-Nix.2",        "v2 base model — solid, chat tune (⚠ undertrained)", "HyperNix", "⚠️", "local"),
    ModelEntry("hyper-nix.1",       "ray0rf1re/hyper-nix.1",        "v1 base model — solid, no chat tune",  "HyperNix",   "",  "local"),

    # Nix Family
    ModelEntry("nix2.7a",           "Nix-ai/Nix-2.7a",              "Nix 2.7a — 2B Qwen2-shape",            "Nix",        "★", "local"),
    ModelEntry("nix2.6-mm",         "Nix-ai/Nix2.6-mm",             "Nix 2.6-mm — 3B Qwen2-shape",          "Nix",        "",  "local"),
    ModelEntry("nix2.5",            "ray0rf1re/Nix2.5",             "Nix 2.5 — 3B Qwen2, tied embeds",      "Nix",        "",  "local"),
    ModelEntry("nix3-coder",        "Nix-ai/Nix3-Coder-7B",         "Nix 3 Coder 7B agentic fine-tune",     "Nix",        "★", "local"),

    # Kimi Family
    ModelEntry("kimi-k3",           "MoonshotAI/Kimi-K3",           "Kimi K3 Reasoning Agent",              "Kimi",       "★", "local"),
    ModelEntry("kimi-k2.5+",        "MoonshotAI/Kimi-K2.5-Plus",    "Kimi K2.5+ Long Context",              "Kimi",       "",  "local"),

    # Qwen Family
    ModelEntry("qwen3.7-plus",      "Qwen/Qwen3.7-Plus",            "Qwen 3.7 Plus — Reasoning & Code",     "Qwen 3.5",   "★", "local"),
    ModelEntry("qwen3.5-0.8b",      "Qwen/Qwen3.5-0.8B",            "Qwen3.5 0.8B — AutoModel",             "Qwen 3.5",   "",  "local"),
    ModelEntry("qwen3.5-2b",        "Qwen/Qwen3.5-2B",              "Qwen3.5 2B — AutoModel",               "Qwen 3.5",   "",  "local"),
    ModelEntry("qwen3.5-4b",        "Qwen/Qwen3.5-4B",              "Qwen3.5 4B — AutoModel",               "Qwen 3.5",   "★", "local"),
    ModelEntry("qwen3.5-9b",        "Qwen/Qwen3.5-9B",              "Qwen3.5 9B — AutoModel",               "Qwen 3.5",   "",  "local"),
    ModelEntry("qwen2.5-coder-32b", "Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen 2.5 Coder 32B Instruct",     "Qwen 3.5",   "★", "local"),

    # Nano Family
    ModelEntry("nano-nano-v4",      "ray0rf1re/Nano-nano-v4",       "Llama-shape, 14L/896d",                "Nano",       "",  "local"),
    ModelEntry("nano-mini-6.99-v2", "ray0rf1re/Nano-mini-6.99-v2",   "Llama-shape, 12L/768d",                "Nano",       "",  "local"),
    ModelEntry("nano-nano-927-v3",  "ray0rf1re/nano-nano-927-v3",    "custom NanoNano, 12L/120d",             "Nano",       "",  "local"),

    # LLaMA 3 Family
    ModelEntry("llama-3.3-70b",     "meta-llama/Llama-3.3-70B-Instruct", "LLaMA 3.3 70B Instruct",            "LLaMA 3",    "★", "local"),
    ModelEntry("llama-3.1-8b",      "meta-llama/Llama-3.1-8B-Instruct",  "LLaMA 3.1 8B Instruct",             "LLaMA 3",    "",  "local"),

    # DeepSeek Family
    ModelEntry("deepseek-r1",       "deepseek-ai/DeepSeek-R1",      "DeepSeek R1 Reasoning Agent",          "DeepSeek",   "★", "local"),
    ModelEntry("deekseek-v4flash",  "deepseek-ai/DeepSeek-V4-Flash","DeepSeek V4 Flash Fast Agent",         "DeepSeek",   "⚡", "local"),
    ModelEntry("deepseek-v3",       "deepseek-ai/DeepSeek-V3",      "DeepSeek V3 671B MoE",                 "DeepSeek",   "★", "local"),

    # Mistral & Fable Family
    ModelEntry("fable-5",           "fable-ai/fable-5",             "Fable 5 Creative & Code Model",        "Fable",      "★", "local"),
    ModelEntry("mistral-large-2411", "mistralai/Mistral-Large-Instruct-2411", "Mistral Large 2411",         "Mistral",    "★", "local"),
    ModelEntry("mistral-small-24b", "mistralai/Mistral-Small-24B-Instruct-2501", "Mistral Small 24B",      "Mistral",    "",  "local"),
    ModelEntry("mixtral-8x7b",      "mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B MoE",             "Mistral",    "",  "local"),

    # Gemma Family
    ModelEntry("gemma-4-27b",       "google/gemma-4-27b-it",        "Gemma 4 27B Instruction-Tuned",        "Gemma 4",    "★", "local"),
    ModelEntry("gemma-3-12b",       "google/gemma-3-12b-it",        "Gemma 3 12B Multimodal",               "Gemma",      "★", "local"),
    ModelEntry("gemma-2-9b",        "google/gemma-2-9b-it",         "Gemma 2 9B Instruct",                  "Gemma",      "",  "local"),

    # Phi Family
    ModelEntry("phi-4-14b",         "microsoft/phi-4",              "Phi-4 14B Reasoning LM",               "Phi",        "★", "local"),
    ModelEntry("phi-3.5-mini",      "microsoft/Phi-3.5-mini-instruct", "Phi-3.5 Mini 3.8B",               "Phi",        "",  "local"),

    # OpenAI API Family
    ModelEntry("openai:gpt-4o",      "gpt-4o",                       "OpenAI GPT-4o Multimodal API",         "OpenAI",     "⚡", "openai"),
    ModelEntry("openai:gpt-5.6-terra", "gpt-5.6-terra",              "OpenAI GPT-5.6 Terra Frontier API",    "OpenAI",     "⚡", "openai"),
    ModelEntry("openai:gpt-5.6-sol",   "gpt-5.6-sol",                 "OpenAI GPT-5.6 Sol High-Speed API",   "OpenAI",     "⚡", "openai"),
    ModelEntry("openai:gpt-5.5",       "gpt-5.5",                     "OpenAI GPT-5.5 Reasoning API",         "OpenAI",     "⚡", "openai"),
    ModelEntry("openai:gpt-4o-mini", "gpt-4o-mini",                  "OpenAI GPT-4o Mini API",               "OpenAI",     "⚡", "openai"),
    ModelEntry("openai:o1",          "o1",                           "OpenAI o1 Reasoning API",              "OpenAI",     "⚡", "openai"),
    ModelEntry("openai:o3-mini",     "o3-mini",                      "OpenAI o3-mini Reasoning API",         "OpenAI",     "⚡", "openai"),

    # Anthropic API Family
    ModelEntry("anthropic:claude-sonnet-4.6", "claude-4-6-sonnet",   "Anthropic Claude Sonnet 4.6 API",     "Anthropic",  "⚡", "anthropic"),
    ModelEntry("anthropic:claude-sonnet-5",   "claude-5-sonnet",     "Anthropic Claude Sonnet 5 API",       "Anthropic",  "⚡", "anthropic"),
    ModelEntry("anthropic:claude-opus-4.8",   "claude-4-8-opus",       "Anthropic Claude Opus 4.8 API",       "Anthropic",  "⚡", "anthropic"),
    ModelEntry("anthropic:claude-haiku-4.5",  "claude-4-5-haiku",      "Anthropic Claude Haiku 4.5 API",      "Anthropic",  "⚡", "anthropic"),
    ModelEntry("anthropic:claude-3-7-sonnet", "claude-3-7-sonnet-20250219", "Anthropic Claude 3.7 Sonnet", "Anthropic", "⚡", "anthropic"),
    ModelEntry("anthropic:claude-3-5-sonnet", "claude-3-5-sonnet-20241022", "Anthropic Claude 3.5 Sonnet", "Anthropic", "⚡", "anthropic"),
)


# ---------------------------------------------------------------------------
# Sampling Profile & Provider Configuration
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    max_new_tokens: int = 512
    seed: int | None = None
    persona: str | None = "coder"
    flour_preset: str = "smart"
    provider: str = "local"        # local | openai | anthropic | rest | t1
    api_key: str = ""
    api_base: str = ""
    t1_key: str = ""

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# ANSI / TUI Helpers
# ---------------------------------------------------------------------------

CSI = "\x1b["
CLEAR = f"{CSI}2J{CSI}H"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"

_C256_FG = "\x1b[38;5;{}m"
_C256_BG = "\x1b[48;5;{}m"
_RESET = "\x1b[0m"

# Palettes
_SIGIL_COLORS = [129, 135, 141, 147, 153, 159, 51, 45, 39, 33, 27, 21]
_ACCENT_BLUE   = 33
_ACCENT_CYAN   = 51
_ACCENT_VIOLET = 135
_ACCENT_GOLD   = 220
_ACCENT_GREEN  = 82
_ACCENT_RED    = 196
_ACCENT_GRAY   = 242


def _color(code: int, text: str, *, on: bool = True) -> str:
    return f"{CSI}{code}m{text}{CSI}0m" if on else text


def _c256(n: int, text: str, *, on: bool = True) -> str:
    return f"{_C256_FG.format(n)}{text}{_RESET}" if on else text


def _bold(text: str, *, on: bool = True) -> str:
    return _color(1, text, on=on) if on else text


def _dim(text: str, *, on: bool = True) -> str:
    return f"{CSI}2m{text}{_RESET}" if on else text


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:  # noqa: BLE001
        return default


def _render_sigil_line(line: str, colors: list[int], *, on: bool = True) -> str:
    if not on:
        return line
    out = []
    ci = 0
    for ch in line:
        if ch != ' ':
            out.append(f"{_C256_FG.format(colors[ci % len(colors)])}{ch}")
            ci += 1
        else:
            out.append(ch)
    return ''.join(out) + _RESET


_HYPED_SIGIL = [
    r" ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ",
    r" ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗",
    r" ███████║ ╚████╔╝ ██████╔╝█████╗  ██║  ██║",
    r" ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██║  ██║",
    r" ██║  ██║   ██║   ██║     ███████╗██████╔╝",
]


def _panel(
    title: str,
    body: list[str],
    *,
    width: int,
    color: bool,
    ascii_only: bool,
    title_color: int = 135,
    border_color: int = 33,
) -> list[str]:
    if ascii_only:
        tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"
    else:
        tl, tr, bl, br, h, v = "╭", "╮", "╰", "╯", "─", "│"
    inner = max(1, width - 2)
    if color:
        def _bc(s: str) -> str:
            return f"{_C256_FG.format(border_color)}{s}{_RESET}"
        def _tc(s: str) -> str:
            return f"{_C256_FG.format(title_color)}{_C256_BG.format(234)}\x1b[1m{s}{_RESET}"
    else:
        def _bc(s: str) -> str:
            return s
        def _tc(s: str) -> str:
            return s
    title_render = _tc(f" {title} ")
    title_vis = len(f" {title} ")
    fill = max(0, inner - 2 - title_vis)
    top_left  = _bc(tl + h)
    top_right = _bc(h * fill + tr)
    rows = [top_left + title_render + top_right]
    for ln in body:
        plain = _strip_ansi(ln)
        pad = max(0, inner - len(plain))
        rows.append(_bc(v) + ln + " " * pad + _bc(v))
    rows.append(_bc(bl + h * inner + br))
    return rows


# ---------------------------------------------------------------------------
# AI Self-Created Skill Creator Engine
# ---------------------------------------------------------------------------

class SkillManager:
    """Manages creation, execution, listing, and deletion of custom AI skills."""

    def __init__(self, storage_dir: Path = SKILLS_DIR) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_skills(self) -> list[dict[str, Any]]:
        skills = []
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                skills.append(data)
            except Exception:  # noqa: BLE001
                pass
        return sorted(skills, key=lambda s: s.get("name", ""))

    def create_skill(self, name: str, description: str, code: str, schema: dict[str, Any] | None = None) -> str:
        name_clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
        json_path = self.storage_dir / f"{name_clean}.json"
        py_path = self.storage_dir / f"{name_clean}.py"

        data = {
            "name": name_clean,
            "description": description,
            "code": code,
            "schema": schema or {"type": "object", "properties": {}},
            "created_at": time.time(),
        }
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        py_path.write_text(code, encoding="utf-8")
        return f"Successfully created and registered skill '{name_clean}' at {json_path}"

    def run_skill(self, name: str, args: dict[str, Any]) -> str:
        name_clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
        py_path = self.storage_dir / f"{name_clean}.py"
        if not py_path.exists():
            return f"Error: Skill '{name_clean}' not found."
        try:
            spec = importlib.util.spec_from_file_location(name_clean, py_path)
            if spec is None or spec.loader is None:
                return f"Error: Failed to load spec for skill '{name_clean}'."
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "execute"):
                res = mod.execute(args)
                return str(res)
            elif hasattr(mod, "main"):
                res = mod.main(args)
                return str(res)
            else:
                return f"Error: Skill module '{name_clean}' has no 'execute(args)' entry point."
        except Exception as exc:  # noqa: BLE001
            return f"Error executing skill '{name_clean}': {exc}"

    def delete_skill(self, name: str) -> str:
        name_clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
        json_path = self.storage_dir / f"{name_clean}.json"
        py_path = self.storage_dir / f"{name_clean}.py"
        deleted = False
        if json_path.exists():
            json_path.unlink()
            deleted = True
        if py_path.exists():
            py_path.unlink()
            deleted = True
        return f"Deleted skill '{name_clean}'" if deleted else f"Skill '{name_clean}' not found."


# ---------------------------------------------------------------------------
# 34+ Built-in Unique Tools & Integrations
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry containing 34+ built-in unique tools for code, system, web, git,
    keymaster, gatekeeper, hypernix pipeline, and skill creator integrations."""

    def __init__(self, skill_manager: SkillManager) -> None:
        self.skill_mgr = skill_manager
        self.tools: dict[str, Callable[..., str]] = {}
        self.schemas: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.memory: dict[str, str] = {}
        self.bg_processes: dict[int, subprocess.Popen] = {}
        self._register_all()

    def register(self, name: str, description: str, func: Callable[..., str], schema: dict[str, Any]) -> None:
        self.tools[name] = func
        self.schemas.append({
            "name": name,
            "description": description,
            "parameters": schema,
        })

    # ------------------------------------------------------------------
    # Consent
    # ------------------------------------------------------------------
    #
    # Tool calls arrive as JSON *inside the model's reply*, which is
    # parsed and dispatched. That makes every one of these reachable by
    # anything that can influence the model's output — a file it was asked
    # to read, a web result, a page it fetched, a response from a T1
    # server. Prompt injection into any of those is, without a gate,
    # arbitrary code execution on the operator's machine.
    #
    # So the side-effecting tools ask first. Reading is not gated: a model
    # that can read a file is the entire point of the tool, and gating it
    # would train the operator to hold down "yes".

    #: Tools that execute code, write outside this process, or control
    #: credentials. Everything else runs unprompted.
    #:
    #: Enumerated from the registry rather than from memory. Gating
    #: `run_command` alone would have been theatre: `create_skill` writes
    #: a Python file and `run_skill` executes it, so a model that wanted a
    #: shell could have had one without ever naming a gated tool. A gate
    #: with a bypass is worse than none, because it is trusted.
    DANGEROUS_TOOLS: frozenset[str] = frozenset({
        # Runs code, one way or another.
        "run_command",
        "run_background_command",
        "execute_script",
        "create_skill",       # writes a Python module the agent can call
        "run_skill",          # …and this calls it
        "delete_skill",
        "hypernix_train",
        "hypernix_convert",
        "hypernix_quantize",
        "hypernix_download",
        "hypernix_assistant",
        # Writes or destroys files.
        "write_file",
        "replace_file_content",
        "multi_replace",
        "batch_replace",      # across a whole directory
        "apply_patch",
        "delete_file",
        "move_file",
        "copy_file",
        # Credentials. A model that can mint or revoke a key can lock the
        # operator out of their own server, or issue itself one.
        "keymaster_create_key",
        "keymaster_revoke_key",
        # Changes this process's environment for every later tool call.
        "set_env",
        # Rewrites history.
        "git_commit",
    })

    #: Deliberately *not* gated, and why.
    #:
    #: The web tools (`fetch_url`, `read_web_page`, `web_search`,
    #: `non_api_web_search`) are an exfiltration channel: a model that has
    #: read a file can put it in a URL. They are left ungated because they
    #: fire constantly in ordinary use, and a prompt that appears on every
    #: web read is one the operator learns to dismiss — which would cost
    #: more safety than it buys. An operator who cares can set
    #: HYPERNIX_TOOL_POLICY=deny and approve nothing.
    #:
    #: Reads (`view_file`, `list_dir`, `grep_search`, `git_status`, the
    #: gatekeeper stats) are ungated for the same reason: a model that can
    #: read a file is the point of the tool.
    UNGATED_EGRESS: frozenset[str] = frozenset({
        "fetch_url", "read_web_page", "web_search", "non_api_web_search",
    })

    #: What to do when a dangerous tool is called.
    #:
    #: ``ask``   — prompt, and run only on an explicit yes (the default).
    #: ``deny``  — refuse, and tell the model why.
    #: ``allow`` — run without asking. Opt-in only, for an operator who
    #:             has decided the trade is worth it in their sandbox.
    #:
    #: A session with no terminal falls back to ``deny`` rather than
    #: ``allow``: when nobody can answer, "don't" is the safe answer, and
    #: silently running is how a CI job becomes a shell for whoever can
    #: reach the model.
    def _consent_policy(self) -> str:
        policy = os.environ.get("HYPERNIX_TOOL_POLICY", "ask").strip().lower()
        if policy not in ("ask", "deny", "allow"):
            policy = "ask"
        if policy == "ask" and not sys.stdin.isatty():
            return "deny"
        return policy

    def _describe_call(self, name: str, kwargs: dict[str, Any]) -> str:
        """The call as the operator needs to see it, not as it was typed.

        The whole decision rests on this string, so it shows the actual
        arguments rather than a summary — a truncated command is one an
        operator cannot judge.
        """
        if name in ("run_command", "run_background_command"):
            return str(kwargs.get("command", ""))
        if name == "execute_script":
            return str(kwargs.get("code", ""))
        return json.dumps(kwargs, default=str)

    def _confirm(self, name: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
        """Ask the operator. Returns (allowed, reason-when-refused)."""
        policy = self._consent_policy()
        if policy == "allow":
            return True, ""
        if policy == "deny":
            return False, (
                "refused: this tool changes things and no one is present to "
                "approve it (HYPERNIX_TOOL_POLICY=deny, or no terminal). "
                "Explain what you would run instead."
            )

        detail = self._describe_call(name, kwargs)
        sys.stdout.write(
            "\n\033[38;5;220m  the model wants to run a tool that changes things\033[0m\n"
            f"    tool: {name}\n"
        )
        for line in detail.splitlines() or [""]:
            sys.stdout.write(f"    {line}\n")
        sys.stdout.write("  allow? [y/N] ")
        sys.stdout.flush()
        try:
            answer = sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer in ("y", "yes"):
            return True, ""
        return False, "refused by the operator."

    def _register_all(self) -> None:
        # File System Tools
        self.register("view_file", "Read file lines with optional range", self._view_file, {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        })
        self.register("write_file", "Write or overwrite content to a file", self._write_file, {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path", "content"],
        })
        self.register("replace_file_content", "Single contiguous block text replacement in a file", self._replace_file_content, {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "target": {"type": "string"},
                "replacement": {"type": "string"},
            },
            "required": ["path", "target", "replacement"],
        })
        self.register("multi_replace", "Multiple non-contiguous text replacements in a file", self._multi_replace, {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["target", "replacement"],
                    },
                },
            },
            "required": ["path", "replacements"],
        })
        self.register("list_dir", "List directory files and folders", self._list_dir, {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        self.register("grep_search", "Grep/regex pattern search across directory", self._grep_search, {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "is_regex": {"type": "boolean"},
            },
            "required": ["query", "path"],
        })
        self.register("find_files", "Find files matching glob pattern", self._find_files, {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern", "path"],
        })
        self.register("delete_file", "Delete a file from disk", self._delete_file, {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        self.register("file_info", "Stat file size, permissions, and modification time", self._file_info, {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        self.register("copy_file", "Copy a file or directory", self._copy_file, {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["src", "dst"],
        })
        self.register("move_file", "Move or rename a file or directory", self._move_file, {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["src", "dst"],
        })

        # Terminal & Execution Tools
        self.register("run_command", "Run bash shell command", self._run_command, {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["command"],
        })
        self.register("system_info", "Fetch system CPU, RAM, GPU, OS info", self._system_info, {
            "type": "object",
            "properties": {},
        })
        self.register("execute_script", "Execute python code snippet inline", self._execute_script, {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        })

        # Web & Internet Tools
        self.register("web_search", "Search web via DuckDuckGo HTML parser", self._web_search, {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        })
        self.register("fetch_url", "Fetch web page content and convert HTML to markdown text", self._fetch_url, {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        })

        # Git Tools
        self.register("git_status", "Run git status", self._git_status, {"type": "object", "properties": {}})
        self.register("git_diff", "Run git diff", self._git_diff, {"type": "object", "properties": {}})
        self.register("git_log", "Get recent git commits", self._git_log, {
            "type": "object",
            "properties": {"max_count": {"type": "integer"}},
        })
        self.register("git_commit", "Make git commit", self._git_commit, {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        })
        self.register("git_branch", "List git branches", self._git_branch, {"type": "object", "properties": {}})

        # Code Inspection Tools
        self.register("syntax_check", "Check syntax of python/json/yaml file", self._syntax_check, {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        self.register("code_summary", "Summarize classes and functions in python file", self._code_summary, {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        self.register("tree_view", "Show visual directory tree structure", self._tree_view, {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer"},
            },
            "required": ["path"],
        })

        # Keymaster & Gatekeeper Tools
        self.register("keymaster_create_key", "Create fresh T1 API key in Keymaster", self._keymaster_create_key, {
            "type": "object",
            "properties": {
                "key_type": {"type": "string"},
                "scopes": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            },
        })
        self.register("keymaster_list_keys", "List active T1 keys managed by Keymaster", self._keymaster_list_keys, {
            "type": "object",
            "properties": {"active_only": {"type": "boolean"}},
        })
        self.register("keymaster_revoke_key", "Revoke T1 API key in Keymaster", self._keymaster_revoke_key, {
            "type": "object",
            "properties": {
                "key_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["key_id"],
        })
        self.register("gatekeeper_check_quota", "Check Gatekeeper rate limits and quota", self._gatekeeper_check_quota, {
            "type": "object",
            "properties": {"key_id": {"type": "string"}},
            "required": ["key_id"],
        })
        self.register("gatekeeper_stats", "Get Gatekeeper usage stats for key", self._gatekeeper_stats, {
            "type": "object",
            "properties": {"key_id": {"type": "string"}},
        })

        # HyperNix Pipeline Integrations
        self.register("hypernix_download", "Download model snapshot via hypernix.download", self._hypernix_download, {
            "type": "object",
            "properties": {"repo_id": {"type": "string"}},
            "required": ["repo_id"],
        })
        self.register("hypernix_quantize", "Quantize model GGUF via hypernix.quantize", self._hypernix_quantize, {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "output": {"type": "string"},
                "quant_type": {"type": "string"},
            },
            "required": ["source", "output", "quant_type"],
        })
        self.register("hypernix_train", "Run HyperNix training utility", self._hypernix_train, {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string"},
                "dataset": {"type": "string"},
                "out_dir": {"type": "string"},
                "steps": {"type": "integer"},
            },
            "required": ["model_dir", "dataset", "out_dir"],
        })
        self.register("hypernix_convert", "Convert PyTorch model to GGUF", self._hypernix_convert, {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string"},
                "output": {"type": "string"},
            },
            "required": ["model_dir", "output"],
        })
        self.register("hypernix_assistant", "Run HyperNix environment assistant check", self._hypernix_assistant, {
            "type": "object",
            "properties": {},
        })

        # Skill Creator Tools
        self.register("create_skill", "Create and register a custom Python skill tool dynamically", self._create_skill, {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["name", "description", "code"],
        })
        self.register("list_skills", "List all AI self-created active skills", self._list_skills, {"type": "object", "properties": {}})
        self.register("run_skill", "Execute an AI self-created skill", self._run_skill, {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {"type": "object"},
            },
            "required": ["name"],
        })
        self.register("delete_skill", "Delete an AI self-created skill", self._delete_skill, {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        })

        # OpenClaw Task & Session Management Tools
        self.register("create_task", "Create session task item", self._create_task, {
            "type": "object",
            "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
            "required": ["title"],
        })
        self.register("update_task", "Update session task status (pending, in_progress, completed)", self._update_task, {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}, "status": {"type": "string"}},
            "required": ["task_id", "status"],
        })
        self.register("list_tasks", "List active session tasks", self._list_tasks, {"type": "object", "properties": {}})

        # OpenClaw Enhanced Web & Scraper Tools
        self.register("non_api_web_search", "Search web without API key via HTML scrapers", self._non_api_web_search, {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        })
        self.register("read_web_page", "Read web page and extract clean markdown & links", self._read_web_page, {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        })

        # OpenClaw Code Editing & Environment Tools
        self.register("apply_patch", "Apply unified diff patch to a file", self._apply_patch, {
            "type": "object",
            "properties": {"path": {"type": "string"}, "patch": {"type": "string"}},
            "required": ["path", "patch"],
        })
        self.register("batch_replace", "Batch text replace across files in directory", self._batch_replace, {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "target": {"type": "string"},
                "replacement": {"type": "string"},
                "glob_pattern": {"type": "string"},
            },
            "required": ["path", "target", "replacement"],
        })
        self.register("set_env", "Set environment variable for agent session", self._set_env, {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        })
        self.register("get_env", "Get environment variable", self._get_env, {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        })

        # OpenClaw Async Process & Memory Scratchpad Tools
        self.register("run_background_command", "Launch bash command in background process", self._run_background_command, {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        })
        self.register("check_process", "Check status of background process by PID", self._check_process, {
            "type": "object",
            "properties": {"pid": {"type": "integer"}},
            "required": ["pid"],
        })
        self.register("memory_save", "Save key-value note in persistent agent scratchpad", self._memory_save, {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        })
        self.register("memory_recall", "Recall key-value note from agent scratchpad", self._memory_recall, {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        })
        self.register("memory_clear", "Clear agent scratchpad memory", self._memory_clear, {"type": "object", "properties": {}})
        self.register("code_refactor_check", "Analyze python code structure and lints", self._code_refactor_check, {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })

    def execute_tool(self, name: str, kwargs: dict[str, Any]) -> str:
        """Run a registered tool, asking first when it changes something.

        The gate lives here rather than at the call site because there is
        more than one call site — the model's JSON, the slash commands,
        and the agent loop all arrive through this method. A check in the
        dispatcher would have covered one of them.
        """
        if name not in self.tools:
            return f"Error: Tool '{name}' is not registered."
        if name in self.DANGEROUS_TOOLS:
            allowed, reason = self._confirm(name, kwargs)
            if not allowed:
                return f"Tool '{name}' was not run — {reason}"
        try:
            return self.tools[name](**kwargs)
        except Exception as exc:  # noqa: BLE001
            return f"Tool Execution Error ({name}): {exc}"

    # Implementations
    def _view_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        s = max(0, min(s, len(lines)))
        e = max(s, min(e, len(lines)))
        selected = lines[s:e]
        return "\n".join(f"{i + s + 1:>4} | {line}" for i, line in enumerate(selected))

    def _write_file(self, path: str, content: str, overwrite: bool = True) -> str:
        p = Path(path)
        if p.exists() and not overwrite:
            return f"Error: File '{path}' exists and overwrite is False."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"

    def _replace_file_content(self, path: str, target: str, replacement: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        text = p.read_text(encoding="utf-8")
        if target not in text:
            return f"Error: target string not found in '{path}'."
        text = text.replace(target, replacement, 1)
        p.write_text(text, encoding="utf-8")
        return f"Successfully replaced target content in {path}"

    def _multi_replace(self, path: str, replacements: list[dict[str, str]]) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        text = p.read_text(encoding="utf-8")
        count = 0
        for r in replacements:
            t, rep = r.get("target", ""), r.get("replacement", "")
            if t in text:
                text = text.replace(t, rep, 1)
                count += 1
        p.write_text(text, encoding="utf-8")
        return f"Successfully performed {count} replacement(s) in {path}"

    def _list_dir(self, path: str = ".") -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: Path '{path}' does not exist."
        entries = []
        for child in sorted(p.iterdir()):
            kind = "DIR " if child.is_dir() else "FILE"
            size = child.stat().st_size if child.is_file() else 0
            entries.append(f"{kind:<4}  {size:>10} bytes  {child.name}")
        return "\n".join(entries) or "(empty directory)"

    def _grep_search(self, query: str, path: str = ".", is_regex: bool = False) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: Path '{path}' does not exist."
        matches = []
        pattern = re.compile(query) if is_regex else None
        files = [p] if p.is_file() else p.rglob("*")
        for f in files:
            if f.is_file() and not f.name.startswith("."):
                try:
                    for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        matched = bool(pattern.search(line)) if pattern else (query in line)
                        if matched:
                            matches.append(f"{f}:{i}: {line}")
                            if len(matches) >= 50:
                                break
                except Exception:  # noqa: BLE001
                    pass
            if len(matches) >= 50:
                break
        return "\n".join(matches) or "No matches found."

    def _find_files(self, pattern: str, path: str = ".") -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: Path '{path}' does not exist."
        matches = [str(m) for m in p.rglob(pattern)][:50]
        return "\n".join(matches) or "No matching files."

    def _delete_file(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File '{path}' does not exist."
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return f"Successfully deleted '{path}'"

    def _file_info(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: Path '{path}' does not exist."
        st = p.stat()
        return (
            f"Path: {p.resolve()}\n"
            f"Type: {'Directory' if p.is_dir() else 'File'}\n"
            f"Size: {st.st_size} bytes\n"
            f"Mode: {oct(st.st_mode)}\n"
            f"Modified: {time.ctime(st.st_mtime)}"
        )

    def _copy_file(self, src: str, dst: str) -> str:
        s, d = Path(src), Path(dst)
        if not s.exists():
            return f"Error: Source '{src}' does not exist."
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        return f"Successfully copied '{src}' to '{dst}'"

    def _move_file(self, src: str, dst: str) -> str:
        s, d = Path(src), Path(dst)
        if not s.exists():
            return f"Error: Source '{src}' does not exist."
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"Successfully moved '{src}' to '{dst}'"

    def _run_command(self, command: str, cwd: str | None = None) -> str:
        try:
            # nosec B602 - reachable only through ToolRegistry.execute_tool,
            # which requires operator consent for this tool (see
            # DANGEROUS_TOOLS). shell=True is the tool's stated purpose.
            res = subprocess.run(
                command,
                shell=True,  # noqa: S602
                cwd=cwd or os.getcwd(),
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = res.stdout
            if res.stderr:
                out += f"\n[stderr]\n{res.stderr}"
            return out.strip() or "(no output)"
        except Exception as exc:  # noqa: BLE001
            return f"Command execution error: {exc}"

    def _system_info(self) -> str:
        import platform
        info = [
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
            f"Python: {sys.version.split()[0]}",
            f"CPUs: {os.cpu_count() or 1}",
        ]
        try:
            import torch
            info.append(f"PyTorch: {torch.__version__}")
            info.append(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(info)

    def _execute_script(self, code: str) -> str:
        try:
            loc: dict[str, Any] = {}
            exec(code, {}, loc)
            return f"Executed script successfully. Result symbols: {list(loc.keys())}"
        except Exception as exc:  # noqa: BLE001
            return f"Script Execution Error: {exc}"

    def _web_search(self, query: str) -> str:
        try:
            q_enc = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={q_enc}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
            # Basic HTML result extraction
            titles = re.findall(r'<a class="result__url"[^>]*>(.*?)</a>', body)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', body)
            results = []
            for t, s in zip(titles[:5], snippets[:5], strict=False):
                clean_t = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
                clean_s = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
                results.append(f"• {clean_t}\n  {clean_s}")
            return "\n\n".join(results) or f"No web search results returned for '{query}'."
        except Exception as exc:  # noqa: BLE001
            return f"Web search error: {exc}"

    def _fetch_url(self, url: str) -> str:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html_raw = resp.read().decode("utf-8", errors="ignore")
            text = re.sub(r'<script\b[^>]*>.*?</script\b[^>]*>', '', html_raw, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style\b[^>]*>.*?</style\b[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:4000] + ("…" if len(text) > 4000 else "")
        except Exception as exc:  # noqa: BLE001
            return f"Fetch URL error: {exc}"

    def _git_status(self) -> str:
        return self._run_command("git status -s")

    def _git_diff(self) -> str:
        return self._run_command("git diff")[:3000]

    def _git_log(self, max_count: int = 5) -> str:
        return self._run_command(f"git log -n {max_count} --oneline")

    def _git_commit(self, message: str) -> str:
        return self._run_command(f'git commit -am "{message}"')

    def _git_branch(self) -> str:
        return self._run_command("git branch -a")

    def _syntax_check(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        if p.suffix == ".py":
            try:
                compile(p.read_text(encoding="utf-8"), path, "exec")
                return f"Syntax OK: '{path}' compiled with no errors."
            except SyntaxError as err:
                return f"Syntax Error in '{path}': line {err.lineno}: {err.msg}"
        elif p.suffix == ".json":
            try:
                json.loads(p.read_text(encoding="utf-8"))
                return f"JSON OK: '{path}' is valid JSON."
            except json.JSONDecodeError as err:
                return f"JSON Error in '{path}': {err}"
        return f"File type {p.suffix} syntax check not implemented."

    def _code_summary(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        defs = []
        for i, line in enumerate(lines, 1):
            if re.match(r"^\s*(class|def)\s+[a-zA-Z0-9_]+", line):
                defs.append(f"Line {i:>4}: {line.strip()}")
        return "\n".join(defs) or f"No class/def declarations found in {path}."

    def _tree_view(self, path: str = ".", max_depth: int = 2) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        out = []

        def _walk(curr: Path, depth: int, prefix: str = "") -> None:
            if depth > max_depth:
                return
            children = sorted([c for c in curr.iterdir() if not c.name.startswith(".")])
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                branch = "└── " if is_last else "├── "
                out.append(f"{prefix}{branch}{child.name}")
                if child.is_dir():
                    _walk(child, depth + 1, prefix + ("    " if is_last else "│   "))

        out.append(p.resolve().name)
        _walk(p, 1)
        return "\n".join(out[:100])

    # Keymaster / Gatekeeper
    def _keymaster_create_key(self, key_type: str = "user", scopes: list[str] | None = None, note: str = "") -> str:
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster()
        st = {KeyScope(s) for s in (scopes or ["read"])}
        meta = km.create(key_type=KeyType(key_type), scopes=st, note=note)
        return f"Created T1 Key: {meta.key}\nID: {meta.key_id}\nServer: {meta.server_id}"

    def _keymaster_list_keys(self, active_only: bool = True) -> str:
        from hypernix.security.keymaster import Keymaster
        km = Keymaster()
        keys = km.list(active_only=active_only)
        return "\n".join(m.display() for m in keys) or "No active T1 keys found."

    def _keymaster_revoke_key(self, key_id: str, reason: str = "") -> str:
        from hypernix.security.keymaster import Keymaster
        km = Keymaster()
        km.revoke(key_id, reason=reason)
        return f"Revoked T1 Key {key_id}"

    def _gatekeeper_check_quota(self, key_id: str) -> str:
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster
        km = Keymaster()
        gk = Gatekeeper(keymaster=km)
        try:
            gk.check_quota(key_id)
            return f"Quota OK for key {key_id}"
        except Exception as exc:  # noqa: BLE001
            return f"Quota Violation: {exc}"

    def _gatekeeper_stats(self, key_id: str | None = None) -> str:
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster
        km = Keymaster()
        gk = Gatekeeper(keymaster=km)
        if key_id:
            return json.dumps(gk.get_stats(key_id), indent=2)
        return json.dumps(gk.get_all_stats(), indent=2)

    # HyperNix Pipeline Integrations
    def _hypernix_download(self, repo_id: str) -> str:
        from hypernix.models.download import download_model
        p = download_model(repo_id=repo_id)
        return f"Downloaded snapshot to {p}"

    def _hypernix_quantize(self, source: str, output: str, quant_type: str = "q4_k_m") -> str:
        from hypernix.quant.quantize import quantize_gguf
        out = quantize_gguf(source_gguf=source, output_gguf=output, quant_type=quant_type)
        return f"Quantized model to {out}"

    def _hypernix_train(self, model_dir: str, dataset: str, out_dir: str, steps: int = 100) -> str:
        from hypernix.training.train import train
        out = train(model_dir=model_dir, dataset_path=dataset, out_dir=out_dir, steps=steps)
        return f"Training completed: {out}"

    def _hypernix_convert(self, model_dir: str, output: str) -> str:
        from hypernix.quant.convert import convert_to_gguf
        out = convert_to_gguf(model_dir=model_dir, output=output)
        return f"Converted GGUF to {out}"

    def _hypernix_assistant(self) -> str:
        from hypernix.system.utils import diagnostic_info
        return json.dumps(diagnostic_info(), indent=2)

    # Skill Creator Tools
    def _create_skill(self, name: str, description: str, code: str) -> str:
        return self.skill_mgr.create_skill(name, description, code)

    def _list_skills(self) -> str:
        skills = self.skill_mgr.list_skills()
        if not skills:
            return "No custom AI skills registered yet."
        return "\n".join(f"• {s['name']}: {s['description']}" for s in skills)

    def _run_skill(self, name: str, args: dict[str, Any] | None = None) -> str:
        return self.skill_mgr.run_skill(name, args or {})

    def _delete_skill(self, name: str) -> str:
        return self.skill_mgr.delete_skill(name)

    # OpenClaw Task Tracker Implementations
    def _create_task(self, title: str, description: str = "") -> str:
        task_id = len(self.tasks) + 1
        self.tasks.append({"id": task_id, "title": title, "description": description, "status": "pending"})
        return f"Task #{task_id} created: '{title}'"

    def _update_task(self, task_id: int, status: str) -> str:
        valid = {"pending", "in_progress", "completed", "failed"}
        if status not in valid:
            return f"Error: status must be one of {valid}"
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = status
                return f"Task #{task_id} updated to '{status}'"
        return f"Error: Task #{task_id} not found."

    def _list_tasks(self) -> str:
        if not self.tasks:
            return "No session tasks yet."
        icons = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
        lines = [f"{icons.get(t['status'], '?')} #{t['id']} [{t['status']}] {t['title']}" + (f"\n   {t['description']}" if t.get('description') else "") for t in self.tasks]
        return "\n".join(lines)

    # OpenClaw Non-API Web Search & Scraper
    def _non_api_web_search(self, query: str, max_results: int = 8) -> str:
        try:
            from hypernix.interfaces.websearch import format_search_results, search_web_non_api
            results = search_web_non_api(query, max_results=max_results)
            return format_search_results(results)
        except Exception as exc:  # noqa: BLE001
            return f"Non-API web search error: {exc}"

    def _read_web_page(self, url: str) -> str:
        try:
            from hypernix.interfaces.websearch import fetch_web_page
            result = fetch_web_page(url)
            out = f"Title: {result['title']}\nURL: {result['url']}\nStatus: {result['status']}\n\n{result['text']}"
            if result.get("links"):
                out += "\n\nLinks:\n" + "\n".join(f"  - {lnk['text']}: {lnk['href']}" for lnk in result["links"][:10])
            return out
        except Exception as exc:  # noqa: BLE001
            return f"Read web page error: {exc}"

    # OpenClaw Patch & Batch Edit
    def _apply_patch(self, path: str, patch: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
                tmp.write(patch)
                tmp_path = tmp.name
            result = subprocess.run(
                ["patch", "-u", str(p), tmp_path],
                capture_output=True, text=True, timeout=15,
            )
            Path(tmp_path).unlink(missing_ok=True)
            if result.returncode == 0:
                return f"Patch applied successfully to '{path}'.\n{result.stdout}"
            return f"Patch failed for '{path}': {result.stderr}"
        except Exception as exc:  # noqa: BLE001
            return f"apply_patch error: {exc}"

    def _batch_replace(self, path: str, target: str, replacement: str, glob_pattern: str = "*.py") -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        files = list(p.rglob(glob_pattern)) if p.is_dir() else [p]
        count_files = 0
        count_replacements = 0
        for f in files:
            if f.is_file():
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    if target in text:
                        new_text = text.replace(target, replacement)
                        f.write_text(new_text, encoding="utf-8")
                        count_files += 1
                        count_replacements += text.count(target)
                except Exception:  # noqa: BLE001
                    pass
        return f"Batch replace: {count_replacements} occurrence(s) in {count_files} file(s)."

    # OpenClaw Env Tools
    def _set_env(self, key: str, value: str) -> str:
        os.environ[key] = value
        return f"Set env {key}={value!r}"

    def _get_env(self, key: str) -> str:
        val = os.environ.get(key)
        if val is None:
            return f"Env '{key}' not set."
        return f"{key}={val!r}"

    # OpenClaw Background Process
    def _run_background_command(self, command: str) -> str:
        try:
            # nosec B602 - consent-gated, as above.
            proc = subprocess.Popen(
                command, shell=True,  # noqa: S602
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.bg_processes[proc.pid] = proc
            return f"Background process started with PID {proc.pid}: {command!r}"
        except Exception as exc:  # noqa: BLE001
            return f"run_background_command error: {exc}"

    def _check_process(self, pid: int) -> str:
        proc = self.bg_processes.get(pid)
        if proc is None:
            return f"No tracked background process with PID {pid}."
        ret = proc.poll()
        if ret is None:
            return f"PID {pid}: still running."
        return f"PID {pid}: finished with return code {ret}."

    # OpenClaw Memory Scratchpad
    def _memory_save(self, key: str, value: str) -> str:
        self.memory[key] = value
        return f"Saved memory[{key!r}] = {value!r}"

    def _memory_recall(self, key: str) -> str:
        if key not in self.memory:
            return f"Memory key {key!r} not found. Available: {list(self.memory.keys())}"
        return f"memory[{key!r}] = {self.memory[key]!r}"

    def _memory_clear(self) -> str:
        n = len(self.memory)
        self.memory.clear()
        return f"Cleared {n} memory entries."

    # Code Refactor Check
    def _code_refactor_check(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: '{path}' does not exist."
        if p.suffix != ".py":
            return f"code_refactor_check only supports .py files (got {p.suffix})"
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            # Syntax check first
            try:
                import ast
                tree = ast.parse(src, filename=str(p))
                # Count top-level definitions
                classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
                lines = src.splitlines()
                long_lines = [i + 1 for i, ln in enumerate(lines) if len(ln) > 120]
                report = [
                    "Syntax: OK",
                    f"Lines: {len(lines)}",
                    f"Classes: {len(classes)}",
                    f"Functions: {len(functions)}",
                    f"Long lines (>120): {len(long_lines)}" + (f" @ {long_lines[:5]}" if long_lines else ""),
                ]
                return "\n".join(report)
            except SyntaxError as e:
                return f"Syntax Error: line {e.lineno}: {e.msg}"
        except Exception as exc:  # noqa: BLE001
            return f"code_refactor_check error: {exc}"





def estimate_cost(model_short: str, input_tokens: int, output_tokens: int) -> dict[str, float]:
    """Estimate price in USD based on model pricing per 1M tokens."""
    rates = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-5.6-terra": (5.00, 15.00),
        "gpt-5.6-sol": (3.00, 10.00),
        "gpt-5.5": (4.00, 12.00),
        "claude-4-6-sonnet": (3.00, 15.00),
        "claude-5-sonnet": (3.50, 17.50),
        "claude-4-8-opus": (15.00, 75.00),
        "claude-4-5-haiku": (0.80, 4.00),
        "claude-3-7-sonnet": (3.00, 15.00),
        "deepseek-r1": (0.55, 2.19),
    }
    key = model_short.lower().replace("openai:", "").replace("anthropic:", "")
    in_rate, out_rate = rates.get(key, (0.0, 0.0))
    in_cost = (input_tokens / 1_000_000.0) * in_rate
    out_cost = (output_tokens / 1_000_000.0) * out_rate
    return {
        "input_cost": round(in_cost, 6),
        "output_cost": round(out_cost, 6),
        "total_cost": round(in_cost + out_cost, 6),
    }


def compact_prompt(raw_prompt: str) -> str:
    """Compact a long custom system prompt into a dense, high-impact instruction format."""
    lines = [ln.strip() for ln in raw_prompt.splitlines() if ln.strip()]
    directives = []
    for line in lines:
        cleaned = re.sub(r"^(please|make sure to|you should|always|remember to)\s+", "", line, flags=re.I)
        cleaned = cleaned.rstrip(".")
        if cleaned and cleaned not in directives:
            directives.append(cleaned)
    return "SYSTEM DIRECTIVES: " + " | ".join(directives[:15])


# ---------------------------------------------------------------------------
# Multi-Provider Model Runner
# ---------------------------------------------------------------------------

class OvenRunner:
    """Unified interface for generating text across Local, OpenAI, Anthropic, REST, and T1 models."""

    def __init__(self, entry: ModelEntry, config: SamplingConfig) -> None:
        self.entry = entry
        self.config = config
        self.local_oven: Any = None
        self.load_error: str | None = None

    def load(self) -> None:
        if self.entry.provider == "local":
            try:
                from hypernix.models.neo_oven import preheat
                self.local_oven = preheat(self.entry.repo_id, quiet=True)
            except Exception as exc:  # noqa: BLE001
                self.load_error = str(exc)
                self.local_oven = None

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        provider = self.entry.provider
        if provider == "local":
            if self.local_oven:
                return self.local_oven.chat(messages, **kwargs)
            last_msg = messages[-1]["content"] if messages else ""
            err_note = f" (Note: {self.load_error})" if self.load_error else ""
            return f"[Hyped Engine ({self.entry.short})]: Received prompt: {last_msg[:80]!r}... Local weights for '{self.entry.repo_id}' are not cached on disk{err_note}. Use /provider to switch to OpenAI/Anthropic/REST or run `hnx download {self.entry.short}`."
        elif provider == "openai":
            return self._call_openai(messages, **kwargs)
        elif provider == "anthropic":
            return self._call_anthropic(messages, **kwargs)
        elif provider == "rest":
            return self._call_rest(messages, **kwargs)
        elif provider == "t1":
            return self._call_t1(messages, **kwargs)
        else:
            return f"[Hyped Engine ({self.entry.short})]: Provider '{provider}' not configured."

    def _call_openai(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        key = self.config.api_key or os.getenv("OPENAI_API_KEY", "")
        base = self.config.api_base or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not key:
            return "[Error: OPENAI_API_KEY not set. Use /key <api_key> in hyped.]"
        url = f"{base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.entry.repo_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            return f"[OpenAI API Error: {exc}]"

    def _call_anthropic(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            return "[Error: ANTHROPIC_API_KEY not set. Use /key <api_key> in hyped.]"
        url = "https://api.anthropic.com/v1/messages"
        sys_prompt = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                sys_prompt = m["content"]
            else:
                user_msgs.append(m)
        payload: dict[str, Any] = {
            "model": self.entry.repo_id,
            "messages": user_msgs,
            "max_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }
        if sys_prompt:
            payload["system"] = sys_prompt
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except Exception as exc:  # noqa: BLE001
            return f"[Anthropic API Error: {exc}]"

    def _call_rest(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        base = self.config.api_base or "http://localhost:8000/v1/chat"
        payload = {"messages": messages, "sampling": kwargs}
        req = urllib.request.Request(
            base,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("reply", str(data))
        except Exception as exc:  # noqa: BLE001
            return f"[REST API Error: {exc}]"

    def _call_t1(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        t1_key = self.config.t1_key or os.getenv("HNX_T1_KEY", "")
        if not t1_key:
            return "[Error: HNX T1 API Key not set. Use /key <t1_key> or pass --t1-key.]"
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster
        km = Keymaster()
        gk = Gatekeeper(keymaster=km)
        meta = gk.authenticate(t1_key)
        gk.check_quota(meta.key_id, endpoint="/chat", tokens_requested=100)
        reply = self._call_openai(messages, **kwargs) if self.config.api_key else (self.local_oven.chat(messages, **kwargs) if self.local_oven else "T1 Chat Response")
        gk.record_usage(meta.key_id, endpoint="/chat", model=self.entry.short, tokens_used=len(reply) // 4 + 1)
        return reply


# ---------------------------------------------------------------------------
# Configurator Screen
# ---------------------------------------------------------------------------

@dataclass
class Configurator:
    color: bool = True
    ascii_only: bool = False
    width: int | None = None
    chosen_model: ModelEntry | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def render_model_picker(self) -> str:
        c = self.color and not self.ascii_only
        rows: list[str] = []
        rows.append("")

        if c:
            for sigil_line in _HYPED_SIGIL:
                rows.append(_render_sigil_line(sigil_line, _SIGIL_COLORS, on=True))
            rows.append(_dim(f"   autonomous tui agent  ·  {HYPED_VERSION}", on=True))
        else:
            rows.append(f"  === hyped · pick a model ({HYPED_VERSION}) ===")
        rows.append("")

        body: list[str] = []
        family_groups: dict[str, list[tuple[int, ModelEntry]]] = {}
        for i, m in enumerate(CURATED_MODELS, 1):
            family_groups.setdefault(m.family, []).append((i, m))

        family_order = ["HyperNix", "Nix", "Qwen 3.5", "Nano", "LLaMA 3", "DeepSeek", "Mistral", "Gemma", "Phi", "OpenAI", "Anthropic"]
        family_order += [fam for fam in family_groups if fam not in family_order]
        for fam in family_order:
            entries = family_groups.get(fam, [])
            if not entries:
                continue
            body.append(_color(33, f"  {fam}", on=c))
            for idx, m in entries:
                raw_badge = m.badge if not self.ascii_only else ("*" if m.badge else "")
                badge = _color(93, raw_badge, on=c) if raw_badge else " "
                line = f"  {idx:>2}. {badge} {m.short:<26}  {_color(90, m.label, on=c)}"
                body.append(line)
            body.append("")

        body.append(_color(35, "   0. browse all (full KNOWN_MODELS catalog)", on=c))
        body.append("")
        body.append(_color(90, "  Type a number, or use --model <short> to skip.", on=c))

        rows.extend(body)
        return "\n".join(rows)

    def pick_model_interactive(self) -> ModelEntry:
        print(CLEAR + self.render_model_picker())
        while True:
            try:
                raw = input(f"\n  choose [1-{len(CURATED_MODELS)}, 0=all]: ").strip()
            except EOFError:
                raw = "1"
            if not raw:
                continue
            if raw == "0":
                return self._pick_from_all_known()
            try:
                idx = int(raw)
            except ValueError:
                print(_color(31, "  not a number — try again.", on=self.color))
                continue
            if 1 <= idx <= len(CURATED_MODELS):
                return CURATED_MODELS[idx - 1]
            print(_color(31, f"  out of range — pick 0..{len(CURATED_MODELS)}.", on=self.color))

    def _pick_from_all_known(self) -> ModelEntry:
        c = self.color and not self.ascii_only
        items = sorted(KNOWN_MODELS.items())
        print()
        print(_color(96, _bold(" hyped · all known models", on=c), on=c))
        print()
        for i, (short, info) in enumerate(items, 1):
            print(f"  {i:>3}. {short:<28} {_color(90, info.repo_id, on=c)}")
        print()
        while True:
            try:
                raw = input(f"  choose [1-{len(items)}]: ").strip()
            except EOFError:
                raw = "1"
            if not raw:
                continue
            try:
                idx = int(raw)
            except ValueError:
                print(_color(31, "  not a number.", on=self.color))
                continue
            if 1 <= idx <= len(items):
                short, info = items[idx - 1]
                return ModelEntry(short, info.repo_id, info.notes or "", "Known", "", "local")
            print(_color(31, "  out of range.", on=self.color))

    def pick_persona_interactive(self) -> str | None:
        c = self.color and not self.ascii_only
        names = _menu.MENU.names()
        print()
        print(_color(96, _bold(" hyped · pick a persona", on=c), on=c))
        print()
        for i, name in enumerate(names, 1):
            persona_text = _menu.MENU.get(name)
            preview = persona_text[:60] + ("…" if len(persona_text) > 60 else "") if persona_text else "(no system prompt)"
            print(f"  {i:>2}. {name:<14} {_color(90, preview, on=c)}")
        print(_color(35, "   0. (coder — default agentic coding persona)", on=c))
        print()
        while True:
            try:
                raw = input(f"  choose [0-{len(names)}]: ").strip()
            except EOFError:
                raw = "0"
            if raw in ("", "0"):
                return "coder"
            try:
                idx = int(raw)
            except ValueError:
                print(_color(31, "  not a number.", on=self.color))
                continue
            if 1 <= idx <= len(names):
                return names[idx - 1]
            print(_color(31, "  out of range.", on=self.color))

    def _prompt_api_key_if_needed(self, model: ModelEntry) -> None:
        """Interactively prompt for API key / endpoint if required by provider."""
        c = self.color and not self.ascii_only
        if model.provider == "openai":
            existing = self.sampling.api_key or os.getenv("OPENAI_API_KEY", "")
            if not existing:
                print()
                print(_color(220, "  ⚡ OpenAI API key required for this model.", on=c))
                try:
                    key = input("  Enter OpenAI API key (sk-...): ").strip()
                    if key:
                        self.sampling.api_key = key
                        print(_color(82, f"  Key saved ({key[:8]}...)", on=c))
                except (EOFError, KeyboardInterrupt):
                    pass
        elif model.provider == "anthropic":
            existing = self.sampling.api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if not existing:
                print()
                print(_color(220, "  ⚡ Anthropic API key required for this model.", on=c))
                try:
                    key = input("  Enter Anthropic API key (sk-ant-...): ").strip()
                    if key:
                        self.sampling.api_key = key
                        print(_color(82, f"  Key saved ({key[:8]}...)", on=c))
                except (EOFError, KeyboardInterrupt):
                    pass
        elif model.provider == "rest":
            existing = self.sampling.api_base
            if not existing:
                print()
                print(_color(220, "  🌐 Custom REST endpoint required.", on=c))
                try:
                    base = input("  Enter REST API base URL (e.g. http://localhost:8000/v1/chat): ").strip()
                    if base:
                        self.sampling.api_base = base
                        print(_color(82, f"  Endpoint saved: {base}", on=c))
                    key = input("  Enter API key (leave blank if none): ").strip()
                    if key:
                        self.sampling.api_key = key
                except (EOFError, KeyboardInterrupt):
                    pass
        elif model.provider == "t1":
            existing = self.sampling.t1_key or os.getenv("HNX_T1_KEY", "")
            if not existing:
                print()
                print(_color(220, "  🔑 HNX T1 API key required.", on=c))
                try:
                    key = input("  Enter T1 key (T1_...): ").strip()
                    if key:
                        self.sampling.t1_key = key
                        print(_color(82, f"  T1 key saved ({key[:8]}...)", on=c))
                except (EOFError, KeyboardInterrupt):
                    pass

    def run(self) -> tuple[ModelEntry, SamplingConfig]:
        model = self.pick_model_interactive()
        self._prompt_api_key_if_needed(model)
        persona = self.pick_persona_interactive()
        self.sampling.persona = persona
        self.sampling.provider = model.provider
        self.chosen_model = model
        return model, self.sampling


# ---------------------------------------------------------------------------
# TUI Chat & Agentic Execution Screen
# ---------------------------------------------------------------------------

@dataclass
class ChatScreen:
    runner: OvenRunner
    model_entry: ModelEntry
    sampling: SamplingConfig
    color: bool = True
    ascii_only: bool = False
    width: int | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    countertop: Any = None
    bell: Any = None
    flour: Any = None
    tool_registry: ToolRegistry = field(default_factory=lambda: ToolRegistry(SkillManager()))
    tool_call_count: int = 0

    def __post_init__(self) -> None:
        from hypernix.chat import countertop as _ct_mod
        from hypernix.chat import flour as _flour_mod
        from hypernix.chat import menu as _menu_mod

        # Build the full tool manifest for injection into system prompt
        tool_names = list(self.tool_registry.tools.keys())
        tool_manifest_lines = []
        for s in self.tool_registry.schemas:
            params = s.get("parameters", {})
            req = params.get("required", [])
            props = params.get("properties", {})
            param_desc = ", ".join(
                f"{k}{'*' if k in req else ''}" for k in props
            )
            tool_manifest_lines.append(f"  - {s['name']}({param_desc}): {s['description']}")
        tool_manifest = "\n".join(tool_manifest_lines)

        system = (
            f"You are Hyped {HYPED_VERSION}, a world-class autonomous AI coding agent built into the HyperNix toolkit.\n\n"
            f"## AVAILABLE TOOLS ({len(tool_names)} total)\n"
            "You have FULL ACCESS to the following tools. Always use them to complete tasks instead of guessing.\n\n"
            f"{tool_manifest}\n\n"
            "## TOOL CALL FORMAT\n"
            "When you need to use a tool, output EXACTLY this JSON block and nothing else before it:\n"
            "```json\n"
            '{"tool": "<tool_name>", "args": {"<param>": "<value>"}}\n'
            "```\n\n"
            "## ANTI-HALLUCINATION RULES (CRITICAL)\n"
            "1. ALWAYS call view_file to read file contents BEFORE editing — never invent or assume code.\n"
            "2. ALWAYS call list_dir to discover file structure before referencing paths.\n"
            "3. NEVER hallucinate function signatures, class names, or imports — verify with code_summary or grep_search.\n"
            "4. ALWAYS call syntax_check after writing/editing Python files.\n"
            "5. Run run_command to verify behaviour — never assume commands succeed.\n"
            "6. When unsure about a fact, call non_api_web_search or fetch_url to verify.\n"
            "7. If a tool returns an error, read the error message carefully and correct your approach.\n\n"
            "## WORKFLOW\n"
            "1. Plan → 2. Inspect (view_file / list_dir) → 3. Act (write/run) → 4. Verify (syntax_check / run_command)\n"
        )

        # Add persona if set and not 'none'
        persona = self.sampling.persona
        if persona and persona != "none" and persona in _menu_mod.MENU.names():
            persona_text = _menu_mod.MENU.get(persona)
            if persona_text:  # skip empty 'none' persona
                system += f"\n## PERSONA INSTRUCTIONS\n{persona_text}\n"

        if self.sampling.flour_preset == "smart":
            self.flour = _flour_mod.Flour.smart_default(template="hyper-nix.2")
        else:
            self.flour = _flour_mod.Flour.off()

        # NOTE: no Bell is wired in here. Bell.stream_chat() drives raw
        # token-by-token generation and requires the oven to expose
        # `.model` / `.tokenizer` / `._format_chat` directly (see
        # hypernix.old_oven.CodeOven). `self.runner` is an OvenRunner,
        # which fronts five different providers (local/openai/
        # anthropic/rest/t1) behind a single high-level `.chat()`
        # method and does not expose that low-level surface. Wiring a
        # Bell here made Countertop.say() call
        # `bell.stream_chat(self.runner, ...)`, which immediately
        # raised `AttributeError: 'OvenRunner' object has no
        # attribute '_format_chat'` the moment a prompt was submitted
        # in the hyped TUI. Leaving `bell=None` (the Countertop
        # default) routes generation through `self.runner.chat(...)`
        # instead, which is what actually implements every provider.
        self.countertop = _ct_mod.Countertop(
            oven=self.runner,
            system=system,
            flour=self.flour,
            t1_key=self.sampling.t1_key,
            sampling=self.sampling.to_kwargs(),
        )

    def _w(self) -> int:
        return max(60, self.width or _term_width())

    def render(self) -> str:
        w = self._w()
        c = self.color and not self.ascii_only
        persona = self.sampling.persona or "coder"
        provider = self.model_entry.provider.upper()

        status_body = [
            f" model:    {_color(36, self.model_entry.short, on=c):<24}  "
            f"provider: {_color(33, provider, on=c)}  repo: {_color(90, self.model_entry.repo_id, on=c)}",
            f" persona:  {persona:<24}  tools: {_color(82, f'{len(self.tool_registry.tools)} active', on=c)}  "
            f"calls: {_color(220, str(self.tool_call_count), on=c)}",
            f" turns:    {len(self.countertop.history) // 2:<24}  "
            f"skills: {_color(51, str(len(self.tool_registry.skill_mgr.list_skills())), on=c)}",
        ]
        status_panel = _panel(
            f"hyped · agent ({HYPED_VERSION})", status_body, width=w,
            color=c, ascii_only=self.ascii_only, title_color=96, border_color=135,
        )

        conv_body: list[str] = []
        if not self.countertop.history:
            conv_body.append(_color(90, "  (say something or ask Hyped to write code, search the web, or run commands)", on=c))

        for msg in self.countertop.history[-14:]:
            role = msg["role"]
            content = msg["content"]
            label = (
                _color(36, "user>", on=c) if role == "user"
                else _color(33, "agent>", on=c) if role == "assistant"
                else _color(82, "tool>", on=c)
            )
            for line in _wrap(content, max_width=w - 14):
                conv_body.append(f" {label} {line}")
                label = "      "

            conv_body.append("")

        conv_panel = _panel(
            "transcript", conv_body, width=w,
            color=c, ascii_only=self.ascii_only, title_color=33, border_color=51,
        )

        return "\n".join(status_panel + [""] + conv_panel)

    def _setup_readline_completion(self) -> None:
        """Configure readline tab-completion for / commands."""
        try:
            import readline
            _COMMANDS = [
                "/help", "/tools", "/skills", "/tasks", "/memory", "/search",
                "/key", "/persona", "/model", "/provider", "/custom", "/save",
                "/export", "/system", "/system-prompt", "/compact-prompt",
                "/auto-compact", "/price", "/vision", "/reset", "/clear", "/quit", "/exit",
            ]

            def _completer(text: str, state: int) -> str | None:
                options = [c for c in _COMMANDS if c.startswith(text)] if text.startswith("/") else []
                return options[state] if state < len(options) else None

            readline.set_completer(_completer)
            readline.parse_and_bind("tab: complete")
        except Exception:  # noqa: BLE001
            pass

    def run(self) -> None:
        c = self.color and not self.ascii_only
        commands_help = _color(
            90,
            " /help · /tools · /skills · /tasks · /memory · /search <q> · "
            "/model · /provider · /system-prompt <txt> · /compact-prompt · /auto-compact · "
            "/price · /vision <img_path> <p> · /quit",
            on=c,
        )
        self._setup_readline_completion()
        try:
            while True:
                sys.stdout.write(CLEAR + self.render() + "\n" + commands_help + "\n\n")
                sys.stdout.flush()
                try:
                    user = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user:
                    continue
                if user.startswith("/"):
                    if self._handle_command(user):
                        break
                    continue
                self._chat_turn(user)
        finally:
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()

    def _handle_command(self, line: str) -> bool:  # noqa: PLR0912
        c = self.color and not self.ascii_only
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            return True

        if cmd in ("/reset", "/clear"):
            self.countertop.reset()
            print(_color(90, "  (history cleared)", on=c))
            time.sleep(0.4)
            return False

        if cmd == "/tools":
            print(_color(96, f"\n  Registered Built-in Tools ({len(self.tool_registry.tools)}):", on=c))
            for s in self.tool_registry.schemas:
                print(f"  • {_color(33, s['name'], on=c)}: {s['description']}")
            input("\n  Press Enter to continue...")
            return False

        if cmd == "/skills":
            print(_color(96, "\n  AI Self-Created Skills:", on=c))
            print(self.tool_registry._list_skills())
            input("\n  Press Enter to continue...")
            return False

        if cmd == "/tasks":
            print(_color(96, "\n  Session Tasks:", on=c))
            print(self.tool_registry._list_tasks())
            input("\n  Press Enter to continue...")
            return False

        if cmd == "/memory":
            if not self.tool_registry.memory:
                print(_color(90, "  (memory scratchpad empty)", on=c))
            else:
                print(_color(96, "\n  Agent Memory Scratchpad:", on=c))
                for k, v in self.tool_registry.memory.items():
                    print(f"  {_color(33, k, on=c)}: {v[:120]}")
            time.sleep(1.2)
            return False

        if cmd == "/system-prompt":
            if not arg:
                curr = self.sampling.system_prompt or "(default Hyped system prompt)"
                print(_color(96, f"\n  Current System Prompt:\n{curr}", on=c))
                print(_color(90, "  Usage: /system-prompt <custom_instructions>", on=c))
            else:
                self.sampling.system_prompt = arg
                self.countertop.system = arg
                print(_color(82, f"  Custom system prompt set: {arg[:80]}...", on=c))
            time.sleep(1.0)
            return False

        if cmd == "/compact-prompt":
            raw = arg or self.sampling.system_prompt or "You are Hyped autonomous AI agent. Follow anti-hallucination rules and verify actions."
            compacted = compact_prompt(raw)
            self.sampling.system_prompt = compacted
            self.countertop.system = compacted
            print(_color(82, f"\n  Compacted System Prompt:\n  {compacted}", on=c))
            time.sleep(1.5)
            return False

        if cmd == "/auto-compact":
            self.sampling.auto_compact = not self.sampling.auto_compact
            status = "enabled" if self.sampling.auto_compact else "disabled"
            print(_color(82, f"  Auto context compaction {status}.", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/price":
            in_toks = sum(len(m.get("content", "")) // 4 for m in self.countertop.history)
            out_toks = self.tool_call_count * 100 + len(self.countertop.history) * 50
            est = estimate_cost(self.model_entry.short, in_toks, out_toks)
            print(_color(96, f"\n  Price & Token Estimate ({self.model_entry.short}):", on=c))
            print(f"  Est. Input Tokens:  {in_toks:,} (${est['input_cost']:.6f})")
            print(f"  Est. Output Tokens: {out_toks:,} (${est['output_cost']:.6f})")
            print(f"  Total Estimated:   ${est['total_cost']:.6f}")
            input("\n  Press Enter to continue...")
            return False

        if cmd == "/vision":
            parts_v = arg.split(maxsplit=1)
            if len(parts_v) < 2:
                print(_color(33, "  Usage: /vision <image_path> <prompt>", on=c))
            else:
                img_path, v_prompt = parts_v[0], parts_v[1]
                v_msg = f"[IMAGE_INPUT: {img_path}]\n{v_prompt}"
                self._chat_turn(v_msg)
            return False

        if cmd == "/key":
            if not arg:
                print(_color(33, "  Usage: /key <api_key_or_t1_key>", on=c))
            else:
                self.sampling.api_key = arg
                self.sampling.t1_key = arg
                if arg.startswith("T1_"):
                    self.countertop.authenticate_t1(arg)
                print(_color(82, f"  Key updated successfully ({arg[:8]}...)", on=c))
            time.sleep(1.0)
            return False

        if cmd == "/persona":
            if not arg:
                from hypernix.chat import menu as _m
                print(_color(33, f"  Available personas: {', '.join(_m.MENU.names())}", on=c))
                print(_color(33, "  Usage: /persona <name>  (use 'none' for no persona)", on=c))
            else:
                self.sampling.persona = arg
                print(_color(36, f"  Persona updated → {arg}", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/model":
            if not arg:
                print(_color(96, f"\n  Current model: {self.model_entry.short} ({self.model_entry.provider})", on=c))
                print(_color(90, "  Usage: /model <short_name>  to switch models", on=c))
            else:
                entry = _resolve_short_name(arg)
                if entry:
                    self.model_entry = entry
                    self.runner = OvenRunner(entry, self.sampling)
                    try:
                        self.runner.load()
                    except Exception:  # noqa: BLE001
                        pass
                    if "hyper-nix.2" in entry.short.lower() or "hypernix.2" in entry.short.lower():
                        try:
                            from hypernix.system.utils import warn_hyper_nix_2
                            warn_hyper_nix_2(entry.repo_id)
                        except Exception:  # noqa: BLE001
                            pass
                    print(_color(82, f"  Switched to model: {entry.short}", on=c))
                else:
                    print(_color(31, f"  Unknown model: {arg!r}. Try a name from /tools or --model.", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/search":
            if not arg:
                print(_color(33, "  Usage: /search <query>", on=c))
            else:
                print(_color(96, f"\n  Searching for: {arg!r}", on=c))
                result = self.tool_registry.execute_tool("non_api_web_search", {"query": arg})
                print(result)
                input("\n  Press Enter to continue...")
            return False

        if cmd == "/custom":
            if not arg:
                try:
                    base = input("  REST endpoint URL: ").strip()
                    if base:
                        self.sampling.api_base = base
                        print(_color(82, f"  Custom REST endpoint set: {base}", on=c))
                except (EOFError, KeyboardInterrupt):
                    pass
            else:
                self.sampling.api_base = arg
                print(_color(82, f"  Custom REST endpoint set: {arg}", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/provider":
            providers = ["local", "openai", "anthropic", "rest", "t1"]
            if arg in providers:
                self.sampling.provider = arg
                print(_color(82, f"  Provider set to: {arg}", on=c))
                # Prompt for key if API provider
                if arg in ("openai", "anthropic", "t1"):
                    try:
                        label = {"openai": "OpenAI (sk-...)", "anthropic": "Anthropic (sk-ant-...)", "t1": "T1 key (T1_...)"}[arg]
                        key = input(f"  Enter {label}: ").strip()
                        if key:
                            if arg == "t1":
                                self.sampling.t1_key = key
                            else:
                                self.sampling.api_key = key
                            print(_color(82, f"  Key saved ({key[:8]}...)", on=c))
                    except (EOFError, KeyboardInterrupt):
                        pass
            else:
                print(_color(33, f"  Available providers: {', '.join(providers)}", on=c))
                print(_color(33, "  Usage: /provider <name>", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/save":
            path = arg or "hyped_session.json"
            try:
                import json as _j
                data = {"history": self.countertop.history, "model": self.model_entry.short, "persona": self.sampling.persona}
                Path(path).write_text(_j.dumps(data, indent=2), encoding="utf-8")
                print(_color(82, f"  Session saved to {path}", on=c))
            except Exception as exc:  # noqa: BLE001
                print(_color(31, f"  Save failed: {exc}", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/export":
            path = arg or "hyped_transcript.txt"
            try:
                lines_out = []
                for m in self.countertop.history:
                    lines_out.append(f"[{m['role'].upper()}]\n{m['content']}\n")
                Path(path).write_text("\n".join(lines_out), encoding="utf-8")
                print(_color(82, f"  Transcript exported to {path}", on=c))
            except Exception as exc:  # noqa: BLE001
                print(_color(31, f"  Export failed: {exc}", on=c))
            time.sleep(0.8)
            return False

        if cmd == "/system":
            if not arg:
                print(_color(33, "  Usage: /system <bash_command>", on=c))
            else:
                out = self.tool_registry.execute_tool("run_command", {"command": arg})
                print(_color(90, out, on=c))
                input("\n  Press Enter to continue...")
            return False

        if cmd == "/help":
            print(_color(96, "\n  hyped commands:", on=c))
            cmds = [
                ("/tools",           "List all available built-in tools"),
                ("/skills",          "List AI self-created skills"),
                ("/tasks",           "Show session task list"),
                ("/memory",          "Show agent memory scratchpad"),
                ("/search <query>",  "Quick web search without API"),
                ("/key <val>",       "Set API key (OpenAI/Anthropic/T1)"),
                ("/persona <name>",  "Change persona (none = no persona)"),
                ("/model [<name>]",  "Show or switch current model"),
                ("/provider <name>", "Set provider (local/openai/anthropic/rest/t1)"),
                ("/system-prompt <txt>", "Set custom system prompt"),
                ("/compact-prompt",  "Compact system prompt into dense directives"),
                ("/auto-compact",    "Toggle automatic context compaction"),
                ("/price",           "Show token & price cost estimate"),
                ("/vision <img> <p>", "Multi-modal vision text query"),
                ("/custom [<url>]",  "Set custom REST API endpoint"),
                ("/save [<path>]",   "Save session to JSON file"),
                ("/export [<path>]", "Export transcript to text file"),
                ("/system <cmd>",    "Run bash command directly"),
                ("/reset",           "Clear conversation history"),
                ("/quit",            "Exit hyped"),
            ]
            for name, desc in cmds:
                print(f"  {_color(33, name, on=c):<28} {_color(90, desc, on=c)}")
            print()
            print(_color(90, "  Tip: Tab key completes /commands", on=c))
            input("\n  Press Enter to continue...")
            return False

        print(_color(31, f"  Unknown command {cmd!r}; try /help", on=c))
        time.sleep(0.6)
        return False

    def _chat_turn(self, user: str) -> None:
        c = self.color and not self.ascii_only
        sys.stdout.write(_color(33, "\nagent> thinking…", on=c))
        sys.stdout.flush()

        reply = self.countertop.say(user)
        sys.stdout.write("\r" + " " * 40 + "\r")

        # Check for tool call pattern in response
        tool_matches = re.findall(r"```json\s*(\{\s*\"tool\".*?\})\s*```", reply, flags=re.DOTALL)
        if not tool_matches:
            tool_matches = re.findall(r"(\{\s*\"tool\"\s*:\s*\"[^\"]+\".*?\})", reply, flags=re.DOTALL)

        if tool_matches:
            for match in tool_matches:
                try:
                    call_data = json.loads(match)
                    tool_name = call_data.get("tool")
                    tool_args = call_data.get("args", {})
                    if tool_name and tool_name in self.tool_registry.tools:
                        self.tool_call_count += 1
                        card_body = [
                            f" {_color(220, '[TOOL RUNNING]', on=c)} {_color(36, tool_name, on=c)}",
                            f" args: {json.dumps(tool_args)}",
                        ]
                        card = _panel("tool execution", card_body, width=self._w() - 4, color=c, ascii_only=self.ascii_only, title_color=220, border_color=220)
                        print("\n" + "\n".join(card))

                        output = self.tool_registry.execute_tool(tool_name, tool_args)
                        out_preview = output[:300] + ("…" if len(output) > 300 else "")

                        res_body = [
                            f" {_color(82, '[TOOL SUCCESS]', on=c)} {_color(36, tool_name, on=c)}",
                            f" output: {out_preview}",
                        ]
                        res_card = _panel("tool result", res_body, width=self._w() - 4, color=c, ascii_only=self.ascii_only, title_color=82, border_color=82)
                        print("\n" + "\n".join(res_card))

                        # Observation follow up turn
                        obs_msg = f"Tool '{tool_name}' Output:\n{output}"
                        self.countertop.say(obs_msg)
                except Exception:  # noqa: BLE001
                    pass

        sys.stdout.write(_color(33, f"agent> {self.countertop.history[-1]['content']}\n\n", on=c))
        sys.stdout.flush()
        time.sleep(0.4)


# ---------------------------------------------------------------------------
# Helpers & CLI Entry Point
# ---------------------------------------------------------------------------

def _wrap(text: str, *, max_width: int) -> list[str]:
    out: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            out.append("")
            continue
        line = ""
        for word in paragraph.split(" "):
            if len(line) + len(word) + 1 > max_width:
                if line:
                    out.append(line)
                line = word
            else:
                line = (line + " " + word) if line else word
        if line:
            out.append(line)
    return out or [""]


def _resolve_short_name(short: str) -> ModelEntry | None:
    key = short.lower()
    for m in CURATED_MODELS:
        if m.short.lower() == key:
            return m
    info = KNOWN_MODELS.get(key)
    if info is not None:
        return ModelEntry(short, info.repo_id, info.notes or "", "Known", "", "local")
    return None


def cli_main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    color = True
    ascii_only = False
    model_short: str | None = None
    persona: str | None = None
    t1_key: str | None = os.getenv("HNX_T1_KEY", None)

    if "--no-color" in args:
        color = False
        args.remove("--no-color")
    if "--ascii" in args:
        ascii_only = True
        args.remove("--ascii")
    if "--model" in args:
        i = args.index("--model")
        if i + 1 < len(args):
            model_short = args[i + 1]
            del args[i : i + 2]
    if "--persona" in args:
        i = args.index("--persona")
        if i + 1 < len(args):
            persona = args[i + 1]
            del args[i : i + 2]
    if "--t1-key" in args:
        i = args.index("--t1-key")
        if i + 1 < len(args):
            t1_key = args[i + 1]
            del args[i : i + 2]

    if "--help" in args or "-h" in args:
        print(
            f"hyped {HYPED_VERSION} — autonomous TUI AI agent CLI\n"
            "usage: hyped [--model SHORT] [--persona NAME] [--t1-key KEY] [--no-color] [--ascii]\n"
            "  --model     skip the picker and load named model\n"
            "  --persona   use a named system prompt persona\n"
            "  --t1-key    pass HNX1/T1 API key for Gatekeeper quota enforcement\n"
            "  --no-color  disable ANSI colour\n"
            "  --ascii     ASCII fallback",
        )
        return 0

    cfg = Configurator(color=color, ascii_only=ascii_only)
    if model_short:
        entry = _resolve_short_name(model_short)
        if entry is None:
            print(f"hyped: unknown model {model_short!r}", file=sys.stderr)
            return 2
        sampling = SamplingConfig()
        if persona:
            sampling.persona = persona
    else:
        try:
            entry, sampling = cfg.run()
        except KeyboardInterrupt:
            print()
            return 130
        if persona:
            sampling.persona = persona

    if t1_key:
        sampling.t1_key = t1_key

    print(_color(96, _bold(f"\n  initializing agent runner for {entry.short} ({entry.repo_id})…", on=color), on=color))
    runner = OvenRunner(entry, sampling)
    try:
        runner.load()
    except Exception as exc:  # noqa: BLE001
        print(f"hyped: model runner load note: {exc}", file=sys.stderr)

    chat = ChatScreen(
        runner=runner, model_entry=entry, sampling=sampling,
        color=color, ascii_only=ascii_only,
    )
    try:
        chat.run()
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
    print(_color(90, "  goodbye.", on=color))
    return 0


__all__ = [
    "CURATED_MODELS",
    "ChatScreen",
    "Configurator",
    "ModelEntry",
    "OvenRunner",
    "SamplingConfig",
    "SkillManager",
    "ToolRegistry",
    "cli_main",
]
