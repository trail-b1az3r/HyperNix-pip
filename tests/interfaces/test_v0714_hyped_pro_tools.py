"""Tests for hypernix.hyped_pro_tools: the real file create/edit/read/search
tools hyped-pro's agentic chat loop calls, and their workspace safety scoping.
"""
from __future__ import annotations

import pytest

from hypernix import hyped_pro_tools as tools
from hypernix.hyped_pro_tools import ToolError


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPED_PRO_WORKSPACE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------

def test_create_file_writes_real_content(_workspace):
    tools.create_file("hello.txt", "world")
    assert (_workspace / "hello.txt").read_text() == "world"


def test_create_file_creates_parent_dirs(_workspace):
    tools.create_file("a/b/c.txt", "nested")
    assert (_workspace / "a" / "b" / "c.txt").read_text() == "nested"


def test_create_file_refuses_to_overwrite_existing(_workspace):
    (_workspace / "exists.txt").write_text("original")
    with pytest.raises(ToolError) as exc_info:
        tools.create_file("exists.txt", "clobber")
    assert exc_info.value.code == "TOOL-CREATE-001"
    assert (_workspace / "exists.txt").read_text() == "original"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

def test_edit_file_replaces_unique_match(_workspace):
    (_workspace / "f.py").write_text("def greet():\n    return 'hi'\n")
    tools.edit_file("f.py", "return 'hi'", "return 'hello'")
    assert "return 'hello'" in (_workspace / "f.py").read_text()


def test_edit_file_missing_old_str_raises(_workspace):
    (_workspace / "f.py").write_text("content")
    with pytest.raises(ToolError) as exc_info:
        tools.edit_file("f.py", "not present", "x")
    assert exc_info.value.code == "TOOL-EDIT-003"


def test_edit_file_ambiguous_match_raises(_workspace):
    (_workspace / "f.py").write_text("x = 1\nx = 1\n")
    with pytest.raises(ToolError) as exc_info:
        tools.edit_file("f.py", "x = 1", "x = 2")
    assert exc_info.value.code == "TOOL-EDIT-004"


def test_edit_file_nonexistent_file_raises(_workspace):
    with pytest.raises(ToolError) as exc_info:
        tools.edit_file("nope.py", "a", "b")
    assert exc_info.value.code == "TOOL-EDIT-001"


def test_edit_file_empty_new_str_deletes(_workspace):
    (_workspace / "f.txt").write_text("keep THIS remove")
    tools.edit_file("f.txt", " THIS remove", "")
    assert (_workspace / "f.txt").read_text() == "keep"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_returns_full_content(_workspace):
    (_workspace / "f.txt").write_text("line1\nline2\nline3")
    assert tools.read_file("f.txt") == "line1\nline2\nline3"


def test_read_file_line_range_is_numbered(_workspace):
    (_workspace / "f.txt").write_text("a\nb\nc\nd\n")
    result = tools.read_file("f.txt", start_line=2, end_line=3)
    assert result == "2\tb\n3\tc"


def test_read_file_nonexistent_raises(_workspace):
    with pytest.raises(ToolError) as exc_info:
        tools.read_file("nope.txt")
    assert exc_info.value.code == "TOOL-READ-001"


def test_read_file_on_directory_raises(_workspace):
    (_workspace / "adir").mkdir()
    with pytest.raises(ToolError) as exc_info:
        tools.read_file("adir")
    assert exc_info.value.code == "TOOL-READ-002"


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def test_list_directory_shows_files_and_dirs(_workspace):
    (_workspace / "file.txt").write_text("x")
    (_workspace / "subdir").mkdir()
    result = tools.list_directory(".")
    assert "file.txt" in result
    assert "subdir/" in result


def test_list_directory_skips_noise_dirs(_workspace):
    (_workspace / "__pycache__").mkdir()
    (_workspace / ".git").mkdir()
    (_workspace / "real.txt").write_text("x")
    result = tools.list_directory(".")
    assert "__pycache__" not in result
    assert "real.txt" in result


def test_list_directory_empty_says_so(_workspace):
    (_workspace / "empty").mkdir()
    assert tools.list_directory("empty") == "(empty directory)"


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

def test_search_files_content_mode_finds_match(_workspace):
    (_workspace / "a.py").write_text("def target_function():\n    pass\n")
    (_workspace / "b.py").write_text("def other():\n    pass\n")
    result = tools.search_files("target_function", ".", mode="content")
    assert "a.py" in result
    assert "b.py" not in result


def test_search_files_filename_mode_glob(_workspace):
    (_workspace / "keep.py").write_text("x")
    (_workspace / "skip.txt").write_text("x")
    result = tools.search_files("*.py", ".", mode="filename")
    assert "keep.py" in result
    assert "skip.txt" not in result


def test_search_files_no_matches_says_so(_workspace):
    (_workspace / "a.txt").write_text("nothing relevant")
    assert tools.search_files("zzz_no_such_thing", ".") == "(no matches)"


def test_search_files_invalid_regex_raises(_workspace):
    with pytest.raises(ToolError) as exc_info:
        tools.search_files("(unclosed[", ".")
    assert exc_info.value.code == "TOOL-SEARCH-003"


# ---------------------------------------------------------------------------
# Workspace safety scoping -- the whole point of the workspace_root design
# ---------------------------------------------------------------------------

def test_path_traversal_is_blocked(_workspace):
    with pytest.raises(ToolError) as exc_info:
        tools.create_file("../../etc/evil.txt", "x")
    assert exc_info.value.code == "TOOL-PATH-001"


def test_absolute_path_outside_workspace_is_blocked(_workspace, tmp_path_factory):
    other = tmp_path_factory.mktemp("elsewhere") / "evil.txt"
    with pytest.raises(ToolError) as exc_info:
        tools.create_file(str(other), "x")
    assert exc_info.value.code == "TOOL-PATH-001"
    assert not other.exists()


def test_absolute_path_inside_workspace_is_allowed(_workspace):
    target = _workspace / "ok.txt"
    tools.create_file(str(target), "fine")
    assert target.read_text() == "fine"


# ---------------------------------------------------------------------------
# execute_tool dispatch + schema conversion
# ---------------------------------------------------------------------------

def test_execute_tool_dispatches_by_name(_workspace):
    result = tools.execute_tool("create_file", {"path": "x.txt", "content": "y"})
    assert "Created" in result
    assert (_workspace / "x.txt").read_text() == "y"


def test_execute_tool_unknown_name_raises():
    with pytest.raises(ToolError) as exc_info:
        tools.execute_tool("not_a_real_tool", {})
    assert exc_info.value.code == "TOOL-CFG-001"


def test_execute_tool_bad_arguments_raises():
    with pytest.raises(ToolError) as exc_info:
        tools.execute_tool("create_file", {"wrong_kwarg": "x"})
    assert exc_info.value.code == "TOOL-CFG-002"


def test_anthropic_tools_schema_shape():
    schema = tools.anthropic_tools_schema()
    assert len(schema) == len(tools.TOOLS)
    for entry in schema:
        assert set(entry.keys()) == {"name", "description", "input_schema"}


def test_openai_tools_schema_shape():
    schema = tools.openai_tools_schema()
    assert len(schema) == len(tools.TOOLS)
    for entry in schema:
        assert entry["type"] == "function"
        assert set(entry["function"].keys()) == {"name", "description", "parameters"}


def test_all_tool_names_match_between_schemas():
    anthropic_names = {t["name"] for t in tools.anthropic_tools_schema()}
    openai_names = {t["function"]["name"] for t in tools.openai_tools_schema()}
    assert anthropic_names == openai_names == {t.name for t in tools.TOOLS}
