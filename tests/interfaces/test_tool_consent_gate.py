"""The consent gate on hyped-pro's side-effecting tools.

Tool calls arrive as JSON *inside the model's reply*: the reply is
regex-matched, ``json.loads``-ed, and dispatched. That makes every tool
reachable by anything that can influence the model's output — a file it
was asked to read, a web result, a page it fetched, a response from a T1
server it is talking to.

Without a gate, ``{"tool": "run_command", "args": {"command": "…"}}``
appearing anywhere in that stream is arbitrary code execution on the
operator's machine, with no prompt and no record. It ran immediately; the
"[TOOL RUNNING]" banner printed after the decision, not before it.
"""
from __future__ import annotations

import pytest

hyped = pytest.importorskip(
    "hypernix.interfaces.hyped", reason="hyped-pro needs its optional deps"
)
ToolRegistry = hyped.ToolRegistry


@pytest.fixture
def registry():
    """A registry with the real tools, built without the skill manager."""
    reg = ToolRegistry.__new__(ToolRegistry)
    reg.tools = {}
    reg.schemas = []
    reg.tasks = []
    reg.memory = {}
    reg.bg_processes = {}
    reg.skill_mgr = None
    ToolRegistry._register_all(reg)
    return reg


class TestTheGateStopsTheInjectionPath:
    def test_a_command_is_not_run_without_consent(self, registry, tmp_path, monkeypatch):
        """The whole point, exercised with a real side effect.

        A file that does not appear is the only proof that matters —
        asserting on the returned string would pass against a gate that
        printed a refusal and ran the command anyway.
        """
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        marker = tmp_path / "executed"
        result = registry.execute_tool("run_command", {"command": f"touch {marker}"})
        assert not marker.exists(), "the command ran despite being refused"
        assert result.startswith("Tool 'run_command' was not run")

    def test_the_refusal_tells_the_model_why(self, registry, monkeypatch):
        """It goes back into the conversation, so it has to be useful.

        A bare "error" invites the model to retry the same call; a reason
        lets it do something else.
        """
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        result = registry.execute_tool("run_command", {"command": "true"})
        assert "no one is present to approve" in result

    def test_allow_is_opt_in_and_works(self, registry, tmp_path, monkeypatch):
        """An operator who wants the old behaviour can still have it."""
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "allow")
        marker = tmp_path / "executed"
        registry.execute_tool("run_command", {"command": f"touch {marker}"})
        assert marker.exists()

    def test_ask_without_a_terminal_denies(self, registry, tmp_path, monkeypatch):
        """The default, in the place it matters most.

        "ask" with nobody to ask must not mean "yes". A CI job, a daemon,
        a piped session — anywhere stdin is not a terminal — would
        otherwise become a shell for whoever can reach the model.
        """
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "ask")
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: False})())
        marker = tmp_path / "executed"
        registry.execute_tool("run_command", {"command": f"touch {marker}"})
        assert not marker.exists()

    def test_an_unset_policy_is_ask_not_allow(self, registry, monkeypatch):
        monkeypatch.delenv("HYPERNIX_TOOL_POLICY", raising=False)
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: False})())
        assert registry._consent_policy() == "deny"  # ask, with no tty

    def test_an_unrecognised_policy_falls_back_to_ask(self, registry, monkeypatch):
        """A typo in the variable must not read as "allow"."""
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "yolo")
        monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": lambda self: True})())
        assert registry._consent_policy() == "ask"


class TestWhatIsGated:
    @pytest.mark.parametrize(
        "name",
        ["run_command", "run_background_command", "execute_script", "write_file",
         "delete_file", "move_file", "set_env"],
    )
    def test_side_effecting_tools_are_listed(self, name):
        assert name in ToolRegistry.DANGEROUS_TOOLS

    @pytest.mark.parametrize("name", ["view_file", "list_dir", "git_status", "git_diff"])
    def test_reading_is_not_gated(self, registry, name, monkeypatch):
        """Gating reads would train the operator to hold down "yes".

        A model that can read a file is the entire point of the tool, and
        a prompt that fires on every one stops being read.
        """
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        if name not in registry.tools:
            pytest.skip(f"{name} is not registered in this build")
        result = registry.execute_tool(
            name, {"path": "src"} if "file" in name or "dir" in name else {}
        )
        # `startswith`, not `in`. A read tool returns arbitrary text — and
        # `git_diff` returns this repository's uncommitted diff, so an
        # edit to a test that mentions the refusal string puts that string
        # in the tool's own output and fails this assertion. Which is
        # exactly what happened. The refusal is a whole-result prefix
        # (`execute_tool` returns it and nothing else), so anchoring to
        # position 0 tests the gate instead of the working tree.
        assert not result.startswith(f"Tool '{name}' was not run")

    def test_every_gated_name_is_a_real_tool(self, registry):
        """A typo in the set silently un-gates the tool it meant to cover."""
        missing = sorted(ToolRegistry.DANGEROUS_TOOLS - set(registry.tools))
        assert not missing, f"gated names that no longer exist: {missing}"


class TestTheGateIsInTheRightPlace:
    def test_it_covers_every_dispatch_path(self, registry, tmp_path, monkeypatch):
        """Placed in execute_tool, not at the model's call site.

        The JSON dispatcher, the slash commands and the agent loop all
        arrive through execute_tool. A check in the dispatcher would have
        covered one of the three.
        """
        import inspect

        source = inspect.getsource(ToolRegistry.execute_tool)
        assert "DANGEROUS_TOOLS" in source
        assert "_confirm" in source

    def test_the_prompt_shows_the_actual_command(self, registry):
        """An operator cannot judge a call they cannot see in full."""
        shown = registry._describe_call("run_command", {"command": "rm -rf /tmp/x"})
        assert shown == "rm -rf /tmp/x"
        assert registry._describe_call("execute_script", {"code": "print(1)"}) == "print(1)"


class TestTheGateHasNoBypass:
    """Gating `run_command` alone would have been theatre.

    `create_skill` writes a Python module and `run_skill` executes it. A
    model that wanted a shell could have taken that route without ever
    naming a gated tool — and a gate with a bypass is worse than no gate,
    because it gets trusted.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "create_skill", "run_skill", "delete_skill",      # arbitrary code
            "apply_patch", "batch_replace", "multi_replace",  # writes
            "copy_file",
            "keymaster_create_key", "keymaster_revoke_key",   # credentials
            "hypernix_train", "hypernix_convert",
        ],
    )
    def test_the_other_routes_are_gated_too(self, registry, name, monkeypatch):
        if name not in registry.tools:
            pytest.skip(f"{name} is not registered in this build")
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        assert registry.execute_tool(name, {}).startswith(
            f"Tool '{name}' was not run"
        )

    def test_a_skill_cannot_be_written_and_run_without_consent(
        self, registry, tmp_path, monkeypatch
    ):
        """The bypass, end to end, with a real side effect."""
        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "deny")
        marker = tmp_path / "via_skill"
        registry.execute_tool(
            "create_skill",
            {
                "name": "pwn",
                "code": f"import os\ndef run():\n    os.system('touch {marker}')\n",
            },
        )
        registry.execute_tool("run_skill", {"name": "pwn"})
        assert not marker.exists(), "the create_skill/run_skill route still executes"

    def test_every_code_executing_tool_is_covered(self, registry):
        """A new tool that runs code must be added to the set.

        Checked by description rather than by name, so a tool added later
        with an un-obvious name is still caught.
        """
        runners = []
        for schema in registry.schemas:
            words = schema["description"].lower()
            if any(w in words for w in ("execute", "run ", "launch")):
                if schema["name"] in ToolRegistry.DANGEROUS_TOOLS:
                    continue
                if schema["name"] in ToolRegistry.UNGATED_EGRESS:
                    continue
                runners.append(schema["name"])
        # Anything left is a tool whose description says it runs something
        # and which nothing gates. Read-only inspectors are allowed here.
        allowed = {"check_process", "syntax_check", "code_refactor_check",
                   "git_status", "git_diff", "git_log", "git_branch",
                   "gatekeeper_check_quota", "gatekeeper_stats", "list_tasks",
                   "update_task", "create_task", "list_skills"}
        assert not (set(runners) - allowed), (
            f"tools that run something and are not gated: {sorted(set(runners) - allowed)}"
        )
