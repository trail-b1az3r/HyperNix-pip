"""hypernix.interfaces.noodle — the autonomous executor inside Hyped Pro.

Noodle runs agents. One agent is a model, a sandboxed workspace, a set of
tools and a self-correction loop; a swarm is several of those across
several providers, working a task list in parallel.

    from hypernix.interfaces.noodle import Swarm, syntax_verifier

    swarm = Swarm(
        roster=["ollama:llama3.2", "anthropic:claude-sonnet-4-5"],
        root="/tmp/work",
        verifier=syntax_verifier(),
    )
    swarm.submit("Write fizzbuzz.py and make it pass `python3 fizzbuzz.py`")
    report = swarm.run()
    print(report.to_dict()["by_provider"])

Nine providers: OpenAI, Anthropic, Moonshot Kimi, Google Gemini, Qwen,
xAI Grok, HyperNix T1, Ollama and vLLM. Three wire formats between them
(:mod:`~hypernix.interfaces.noodle.providers`), normalised so a
transcript can move between models mid-swarm.

Ten tools (:mod:`~hypernix.interfaces.noodle.tools`), sandboxed to a
workspace root, with execution off by default and memory off unless the
server enabled it.

Two things Noodle deliberately does not do: it does not fail a task over
to a different provider on its own (surprising invoices in one direction,
surprising output in the other), and it does not execute a client
application a remote server asked it to run. Both are opt-in and both
say what they are doing.

Inside hyped-pro
----------------
The line above about being "the autonomous executor inside Hyped Pro" was
aspirational until 0.72.5: Noodle was a library and a console script, and
hyped-pro had no command, no bridge verb and no idea it existed.
:mod:`hypernix.interfaces.noodle.hyped` is the connection — sessions the
TUI and the GUI drive with ``/noodle`` and a Noodle tab respectively,
with a live event feed rather than a frozen screen for the length of a
run.
"""
from __future__ import annotations

from . import hyped
from .agent import Agent, AgentEvent, AgentResult
from .hpo import HPOResult, SearchSpace, Trial, random_search, successive_halving
from .providers import (
    PROVIDERS,
    ChatResult,
    ModelClient,
    Provider,
    ProviderError,
    ProviderSpec,
    ToolCall,
    available_providers,
    build_client,
)
from .swarm import Swarm, SwarmReport, SwarmTask
from .tools import TOOLS, Tool, ToolContext, ToolError, ToolResult, run_tool, tool_schemas
from .validate import combine, command_verifier, syntax_verifier

__noodle_version__ = "0.72.1"

__all__ = [
    "__noodle_version__",
    "Agent", "AgentEvent", "AgentResult",
    "Swarm", "SwarmReport", "SwarmTask",
    "ModelClient", "Provider", "ProviderSpec", "ProviderError", "PROVIDERS",
    "ChatResult", "ToolCall", "build_client", "available_providers",
    "Tool", "ToolContext", "ToolError", "ToolResult", "TOOLS", "run_tool", "tool_schemas",
    "syntax_verifier", "command_verifier", "combine",
    "SearchSpace", "Trial", "HPOResult", "random_search", "successive_halving",
    "hyped",
]
