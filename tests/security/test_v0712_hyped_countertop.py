"""Unit tests for V0.71.2: hyped TUI Agent, SkillManager, ToolRegistry, multi-provider runners,
and Countertop T1 API key integration.
"""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from hypernix import __version__
from hypernix.countertop import countertop
from hypernix.gatekeeper import Gatekeeper
from hypernix.hyped import (
    CURATED_MODELS,
    HYPED_VERSION,
    ChatScreen,
    ModelEntry,
    OvenRunner,
    SamplingConfig,
    SkillManager,
    ToolRegistry,
)
from hypernix.keymaster import Keymaster, KeyScope, KeyType


def _numeric_prefix(version_str: str) -> tuple[int, ...]:
    """Leading dotted-int prefix, e.g. 'v0.71.4b1' -> (0, 71, 4)."""
    match = re.match(r"^v?(\d+(?:\.\d+)*)", version_str)
    assert match, f"no numeric version prefix in {version_str!r}"
    return tuple(int(p) for p in match.group(1).split("."))


def test_version_v0712():
    assert _numeric_prefix(__version__) > (0, 71)
    assert _numeric_prefix(HYPED_VERSION) > (0, 71)


def test_countertop_t1_integration():
    km = Keymaster()
    meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
    gk = Gatekeeper(keymaster=km)

    class MockOven:
        repo_id = "test-model-nix"
        def chat(self, messages, **kwargs):
            return "Mock reply from oven"

    ct = countertop(
        oven=MockOven(),
        t1_key=meta.key,
        keymaster=km,
        gatekeeper=gk,
    )

    assert ct.t1_meta is not None
    assert ct.t1_meta.key_id == meta.key_id

    reply = ct.say("Hello T1 test")
    assert reply == "Mock reply from oven"

    stats = ct.t1_stats()
    assert stats["total_requests"] == 1
    assert stats["lifetime_request_count"] == 1


def test_skill_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SkillManager(storage_dir=Path(tmpdir))
        res = sm.create_skill(
            name="multiply_skill",
            description="Multiplies x and y",
            code="def execute(args):\n    return str(args.get('x', 0) * args.get('y', 0))",
        )
        assert "multiply_skill" in res

        skills = sm.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "multiply_skill"

        out = sm.run_skill("multiply_skill", {"x": 6, "y": 7})
        assert out == "42"

        del_res = sm.delete_skill("multiply_skill")
        assert "Deleted" in del_res
        assert len(sm.list_skills()) == 0


def test_tool_registry(monkeypatch):
    """The file tools do what they say — with the consent gate opened.

    `write_file` and `replace_file_content` are side-effecting, so
    `execute_tool` gates them on HYPERNIX_TOOL_POLICY, which defaults to
    "ask" and degrades to **deny** with no terminal — which is exactly
    what a test run is. Without opting in here, the writes are refused
    and the failure surfaces two lines later as "file does not exist",
    naming neither the gate nor the tool that was actually stopped.

    The gate itself is covered in tests/interfaces/test_tool_consent_gate.py; this
    asserts the refusal once, so a test that opts in cannot also hide the
    gate disappearing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SkillManager(storage_dir=Path(tmpdir))
        tr = ToolRegistry(sm)
        assert len(tr.tools) >= 34

        file_path = os.path.join(tmpdir, "sample.txt")

        # Default policy, no terminal: refused, and it says so rather than
        # returning something that reads like success.
        monkeypatch.delenv("HYPERNIX_TOOL_POLICY", raising=False)
        refused = tr.execute_tool(
            "write_file", {"path": file_path, "content": "nope"}
        )
        assert refused.startswith("Tool 'write_file' was not run")
        assert not Path(file_path).exists()

        monkeypatch.setenv("HYPERNIX_TOOL_POLICY", "allow")

        # Test view_file & write_file
        wrote = tr.execute_tool("write_file", {"path": file_path, "content": "Line 1\nLine 2\nLine 3"})
        assert "Successfully wrote" in wrote
        view_out = tr.execute_tool("view_file", {"path": file_path, "start_line": 1, "end_line": 2})
        assert "Line 1" in view_out
        assert "Line 2" in view_out

        # Test replace_file_content
        replaced = tr.execute_tool("replace_file_content", {"path": file_path, "target": "Line 2", "replacement": "Line Two"})
        assert "Successfully replaced" in replaced
        view_after = tr.execute_tool("view_file", {"path": file_path, "start_line": 1, "end_line": 3})
        assert "Line Two" in view_after

        # Test list_dir
        dir_out = tr.execute_tool("list_dir", {"path": tmpdir})
        assert "sample.txt" in dir_out

        # Test file_info
        info_out = tr.execute_tool("file_info", {"path": file_path})
        assert "Size:" in info_out

        # Test system_info
        sys_out = tr.execute_tool("system_info", {})
        assert "OS:" in sys_out


def test_chat_screen_submit_prompt_does_not_crash():
    """Regression test: submitting a prompt in the hyped TUI used to
    raise ``AttributeError: 'OvenRunner' object has no attribute
    '_format_chat'`` on every turn, for every provider, because
    ChatScreen wired a low-level ``Bell`` streamer (which requires
    ``.model`` / ``.tokenizer`` / ``._format_chat`` on the oven it's
    given) around ``OvenRunner``, a multi-provider wrapper that only
    exposes the high-level ``.chat()`` API. ChatScreen no longer wires
    a Bell into its Countertop, so a submitted prompt should route
    straight through ``OvenRunner.chat()`` and return normally.
    """
    entry = ModelEntry("nix-mini", "hypernix/nix-mini", "test model", "HyperNix", "", "local")
    sampling = SamplingConfig()
    runner = OvenRunner(entry, sampling)
    runner.local_oven = MagicMock()
    runner.local_oven.chat.return_value = "Hello from the model!"

    screen = ChatScreen(
        runner=runner, model_entry=entry, sampling=sampling,
        color=False, ascii_only=True,
    )

    reply = screen.countertop.say("hi there")

    assert reply == "Hello from the model!"
    assert runner.local_oven.chat.called


def test_curated_models_catalog():
    families = {m.family for m in CURATED_MODELS}
    assert len(families) >= 11
    assert "HyperNix" in families
    assert "Nix" in families
    assert "Qwen 3.5" in families
    assert "Nano" in families
    assert "LLaMA 3" in families
    assert "DeepSeek" in families
    assert "Mistral" in families
    assert "Gemma" in families
    assert "Phi" in families
    assert "OpenAI" in families
    assert "Anthropic" in families
