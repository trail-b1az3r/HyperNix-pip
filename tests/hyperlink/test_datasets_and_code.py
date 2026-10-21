"""HyperLink's datasets and code section (0.72.8a3).

* parts of dataset files (JSONL, JSON, Parquet, CSV, text, Markdown) as
  text a model reads, chosen by rows, columns, search or sample;
* git in a /code sandbox, with hooks and non-https transports off;
* the model's tools for a code session's sandbox, bound by the same
  permissions as a person;
* the routes: dataset preview, and sandbox git, search and import.
"""
from __future__ import annotations

import json
import shutil
import time

import pytest

from hypernix.hyperlink import datasets as ds
from hypernix.hyperlink.codetools import CodeTools
from hypernix.t1api import codebox as cb
from hypernix.t1api import codegit

ROWS = [{"prompt": f"q{i}", "response": f"answer {i}" + (" cats" if i % 7 == 0 else ""), "id": i}
        for i in range(100)]


@pytest.fixture
def files(tmp_path):
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps({"data": ROWS[:5]}), encoding="utf-8")
    (tmp_path / "d.csv").write_text("a,b\n1,x\n2,y\n3,z\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# A\nintro\n## Setup\nstep 1\nstep 2\n## Use\nrun it\n", encoding="utf-8")
    (tmp_path / "big.txt").write_text("\n".join(f"line {i}" for i in range(1000)), encoding="utf-8")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(ROWS), tmp_path / "d.parquet")
    except ImportError:
        pass
    return tmp_path


class TestExcerpts:
    def test_the_default_is_the_first_rows_and_says_so(self, files):
        part = ds.excerpt(files / "d.jsonl")
        assert part.shown == ds.DEFAULT_ROWS and part.text.startswith("--- d.jsonl (jsonl; rows 0-19;")

    def test_a_range_and_columns(self, files):
        part = ds.excerpt(files / "d.jsonl", {"rows": "10-12", "columns": ["prompt"]})
        assert part.columns == ["prompt"] and "| 10 | q10 |" in part.text and "| 11 | q11 |" in part.text
        assert "response" not in part.text.split("\n", 1)[1]

    def test_the_last_rows(self, files):
        part = ds.excerpt(files / "d.jsonl", {"rows": "-2", "style": "jsonl"})
        assert [json.loads(line)["id"] for line in part.text.splitlines()[1:]] == [98, 99]
        assert part.total == 100

    def test_a_search_and_a_regex(self, files):
        assert ds.excerpt(files / "d.jsonl", {"query": "cats", "rows": "0-3"}).shown == 3
        assert "(no rows match)" in ds.excerpt(files / "d.jsonl", {"query": "dogs"}).text
        assert ds.excerpt(files / "d.jsonl", {"query": "re:\\bq9\\d\\b", "rows": "0-"}).shown == 10
        with pytest.raises(ds.DatasetError, match="regular expression"):
            ds.excerpt(files / "d.jsonl", {"query": "re:("})

    def test_a_sample_is_repeatable(self, files):
        a = ds.excerpt(files / "d.jsonl", {"sample": 5, "seed": 3})
        b = ds.excerpt(files / "d.jsonl", {"sample": 5, "seed": 3})
        assert a.text == b.text and a.shown == 5 and "random sample of 5 rows" in a.text

    def test_parquet(self, files):
        pytest.importorskip("pyarrow")
        part = ds.excerpt(files / "d.parquet", {"rows": "50-52", "columns": ["id", "nope"]})
        assert part.total == 100 and "| 50 | 50 |" in part.text and "no column nope" in part.text

    def test_json_and_csv(self, files):
        assert ds.excerpt(files / "d.json").shown == 5
        assert ds.excerpt(files / "d.csv").columns == ["a", "b"]

    def test_text_lines_and_a_markdown_section(self, files):
        assert ds.excerpt(files / "big.txt", {"rows": "500-502"}).text.endswith("line 500\nline 501")
        section = ds.excerpt(files / "notes.md", {"query": "setup"}).text
        assert "step 2" in section and "run it" not in section

    def test_the_size_cap_says_it_cut(self, files):
        part = ds.excerpt(files / "d.jsonl", {"rows": "0-", "max_chars": 600})
        assert part.truncated and part.text.endswith("ask for fewer rows or columns]")

    def test_bad_selections(self, files):
        with pytest.raises(ds.DatasetError, match="rows is like"):
            ds.excerpt(files / "d.jsonl", {"rows": "x"})
        with pytest.raises(ds.DatasetError, match="no column"):
            ds.excerpt(files / "d.jsonl", {"columns": ["nope"]})
        with pytest.raises(ds.DatasetError, match="not a format"):
            ds.excerpt(files / "notes.docx")


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def box(tmp_path):
    store = cb.SandboxStore(tmp_path / "code")
    # Network on for runs that are not about the network, so they pass on a
    # machine without user namespaces (as in tests/t1api/test_code_sandbox.py).
    perms = cb.CodePerms(network=True, timeout=20)
    sandbox = store.create("me", perms, files={"main.py": "print('hi')\n"})
    return store, sandbox, perms


@needs_git
class TestGit:
    def test_init_add_commit_log_diff(self, box):
        store, sandbox, perms = box
        assert codegit.git(sandbox, perms, "init")["ok"]
        assert codegit.git(sandbox, perms, "add", paths=["main.py"])["ok"]
        assert codegit.git(sandbox, perms, "commit", message="first")["ok"]
        log = codegit.git(sandbox, perms, "log")["output"]
        assert "\tHyperLink\tfirst" in log
        store.write(sandbox, "main.py", "print('changed')\n", perms)
        diff = codegit.git(sandbox, perms, "diff")["output"]
        assert "-print('hi')" in diff and "+print('changed')" in diff
        status = codegit.git(sandbox, perms, "status")["output"]
        assert " M main.py" in status
        assert [f["path"] for f in sandbox.files()] == ["main.py"]     # .git's insides are not listed

    def test_hooks_do_not_run(self, box):
        store, sandbox, perms = box
        codegit.git(sandbox, perms, "init")
        hook = sandbox.root / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\ntouch HOOK_RAN\n")
        hook.chmod(0o755)
        codegit.git(sandbox, perms, "add", paths=["."])
        assert codegit.git(sandbox, perms, "commit", message="x")["ok"]
        assert not (sandbox.root / "HOOK_RAN").exists()

    def test_refusals(self, box):
        _store, sandbox, perms = box
        with pytest.raises(cb.SandboxError, match="not a git repository"):
            codegit.git(sandbox, perms, "status")
        codegit.git(sandbox, perms, "init")
        with pytest.raises(cb.SandboxError, match="inside the sandbox"):
            codegit.git(sandbox, perms, "add", paths=["../x"])
        with pytest.raises(cb.SandboxError, match="branch or commit"):
            codegit.git(sandbox, perms, "checkout", ref="--upload-pack=evil")
        with pytest.raises(cb.SandboxError, match="https"):
            codegit.git(sandbox, perms, "clone", url="file:///etc")
        with pytest.raises(cb.SandboxError, match="network"):
            codegit.git(sandbox, cb.CodePerms(network=False), "clone", url="https://github.com/x/y")
        with pytest.raises(cb.SandboxError, match="operation is one of"):
            codegit.git(sandbox, perms, "push")


class TestTheModelsTools:
    def test_a_writer_gets_everything(self, box):
        store, sandbox, perms = box
        tools = CodeTools(store, sandbox, perms, may_write=True)
        assert {"code_write_file", "code_edit_file", "code_run", "code_read_file", "code_git"} <= tools.names
        ok, text = tools.call("code_write_file", {"path": "lib/util.py", "content": "A = 1\nB = 2\n"})
        assert ok and "wrote lib/util.py" in text
        ok, text = tools.call("code_edit_file", {"path": "lib/util.py", "old_text": "B = 2", "new_text": "B = 3"})
        assert ok
        ok, text = tools.call("code_read_file", {"path": "lib/util.py", "start_line": 2, "end_line": 2})
        assert ok and text.strip() == "2  B = 3"
        ok, text = tools.call("code_search", {"query": "B =", "glob": "*.py"})
        assert "lib/util.py:2: B = 3" in text
        ok, text = tools.call("code_edit_file", {"path": "lib/util.py", "old_text": "nothing", "new_text": "x"})
        assert not ok and "occurs 0 times" in text

    def test_running(self, box):
        store, sandbox, perms = box
        ok, text = CodeTools(store, sandbox, perms, may_write=True).call("code_run", {"path": "main.py"})
        assert ok and json.loads(text)["stdout"].strip() == "hi"

    def test_a_reader_only_reads(self, box):
        store, sandbox, perms = box
        tools = CodeTools(store, sandbox, perms, may_write=False)
        assert not {"code_write_file", "code_edit_file", "code_run"} & tools.names
        ok, text = tools.call("code_write_file", {"path": "x", "content": "y"})
        assert not ok and "only read" in text
        ok, _ = tools.call("code_git", {"op": "commit", "message": "m"})
        assert not ok

    def test_escaping_the_sandbox_is_a_readable_refusal(self, box):
        store, sandbox, perms = box
        ok, text = CodeTools(store, sandbox, perms, may_write=True).call("code_read_file", {"path": "../../etc/passwd"})
        assert not ok and text

    def test_the_tool_loop_offers_and_runs_them(self, box):
        from hypernix.hyperlink.toolloop import T1Access, _prepare, _run_call
        from hypernix.interfaces.noodle.tools import ToolContext

        store, sandbox, perms = box
        access = T1Access("http://127.0.0.1:8000", "")
        access.code = CodeTools(store, sandbox, perms, may_write=True)
        context = ToolContext(root=sandbox.root, memory_path=sandbox.root / ".hnx" / "m")
        _m, tools, t1_names = _prepare([{"role": "user", "content": "hi"}], context, access, False, True)
        assert "code_list_files" in {t["function"]["name"] for t in tools}
        call = {"id": "c", "function": {"name": "code_list_files", "arguments": "{}"}}
        record = _run_call(call, context, tools, access, t1_names)
        assert record.ok and "main.py" in record.content


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from hypernix.security.gatekeeper import Gatekeeper
    from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
    from hypernix.t1api.app import create_app
    from hypernix.t1api.config import T1APIConfig

    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
    config = T1APIConfig(token_secret="test-secret-value-that-is-long-enough", db_path=str(tmp_path / "t1.sqlite3"),
                         module_storage_dir=str(tmp_path / "m"), hyperlink_files_dir=str(tmp_path / "files"),
                         default_plan="free", code_sandbox=True, code_sandbox_dir=str(tmp_path / "code"),
                         web_enabled=False)
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    app.state.t1_code_perms = cb.CodePerms(network=True, timeout=20)
    writer = {"Authorization": f"Bearer {km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key}"}
    reader = {"Authorization": f"Bearer {km.create(key_type=KeyType.USER, scopes={KeyScope.READ}).key}"}
    return TestClient(app, client=("127.0.0.1", 5000)), writer, reader


def _upload(client, headers, name, data: bytes) -> str:
    reply = client.post("/hyperlink/files", files={"file": (name, data, "application/octet-stream")}, headers=headers)
    assert reply.status_code == 200, reply.text
    return reply.json()["file"]["file_id"]


class TestRoutes:
    def test_dataset_preview(self, api):
        client, writer, _reader = api
        data = "\n".join(json.dumps(r) for r in ROWS).encode()
        file_id = _upload(client, writer, "rows.jsonl", data)
        reply = client.post("/hyperlink/datasets/preview",
                            json={"file_id": file_id, "rows": "3-5", "columns": ["prompt"]}, headers=writer)
        assert reply.status_code == 200, reply.text
        body = reply.json()
        assert body["rows_shown"] == 2 and body["columns"] == ["prompt"] and "| 3 | q3 |" in body["text"]
        bad = client.post("/hyperlink/datasets/preview", json={"file_id": file_id, "rows": "zz"}, headers=writer)
        assert bad.status_code == 400

    def test_a_dataset_reaches_the_model_as_an_excerpt(self, api):
        from hypernix.hyperlink.sessions import ChatMessage
        from hypernix.t1api.routers.hyperlink import _wire_messages

        client, writer, _reader = api
        many = [{**r, "id": i, "prompt": f"q{i}"} for i, r in enumerate(ROWS * 10)]      # over the "send it whole" size
        file_id = _upload(client, writer, "rows.jsonl", "\n".join(json.dumps(r) for r in many).encode())
        small = _upload(client, writer, "small.csv", b"a,b\n1,2\n")
        store = client.app.state.t1_attachment_store
        owner = client.get("/hyperlink/files", headers=writer).json()["files"][0]["owner"]
        attached = ChatMessage("m1", "s", 1, "user", "look", time.time(), attachment_ids=[file_id])
        picked = ChatMessage("m2", "s", 2, "user", "these", time.time(),
                             metadata={"datasets": [{"file_id": file_id, "rows": "40-42", "columns": ["id"]}]})
        whole = ChatMessage("m3", "s", 3, "user", "and this", time.time(), attachment_ids=[small])
        wire = _wire_messages([attached, picked, whole], store, owner)
        assert "--- rows.jsonl (jsonl; rows 0-19" in wire[0]["content"]          # an attachment: its first rows
        assert "| 1 | 2 |" in wire[2]["content"].replace("| 0 | 1 | 2 |", "| 1 | 2 |")  # a small one goes whole
        assert "| 40 | 40 |" in wire[1]["content"] and "q40" not in wire[1]["content"]

    @needs_git
    def test_sandbox_git_search_and_import(self, api):
        client, writer, reader = api
        box = client.post("/code/create/sandbox", json={"files": {"a.py": "x = 1\n"}}, headers=writer).json()["id"]
        assert client.post(f"/code/sandbox/{box}/git", json={"op": "init"}, headers=writer).json()["ok"]
        assert client.post(f"/code/sandbox/{box}/git", json={"op": "init"}, headers=reader).status_code == 403
        status = client.post(f"/code/sandbox/{box}/git", json={"op": "status"}, headers=writer).json()
        assert status["ok"] and "?? a.py" in status["output"]
        found = client.get(f"/code/sandbox/{box}/search", params={"q": "x ="}, headers=writer).json()
        assert found["matches"][0]["path"] == "a.py"
        file_id = _upload(client, writer, "data.csv", b"a,b\n1,2\n")
        imported = client.post(f"/code/sandbox/{box}/import", json={"file_id": file_id}, headers=writer).json()
        assert imported["path"] == "data.csv"
        read = client.get(f"/code/sandbox/{box}/files/data.csv", headers=writer).json()
        assert read["content"] == "a,b\n1,2\n"
