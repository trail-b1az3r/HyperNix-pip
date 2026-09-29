"""``hypernix-t1 index`` — a folder of GGUFs into a model registry.

The registry is the only place the T1 API looks up what a model can do,
and every route calls ``ModelRegistry.require`` rather than trusting a
client-supplied ``model_id``. That is the right design and it also means
a server whose registry is empty serves nothing.

The two ways to fill it were the installer's one-entry template --
placeholders, marked *"Edit before serving traffic"* -- and hand-writing
JSON. Both ask an operator to transcribe numbers that are already in the
files. A context limit mistyped there is not caught anywhere; it is
simply the number the server enforces.

So the tests here are mostly about the difference between *measured* and
*assumed*:

- the parameter count is summed from the tensor table, so it is the same
  for three quantisations of one model and it is never read off a
  filename;
- a value the file does not carry is defaulted **and reported as
  assumed**, rather than presented as though it had been read;
- policy -- pricing, plan, priority -- is never derived, because it
  cannot be;
- and an entry an operator has edited survives re-indexing, which is the
  property that decides whether this command is safe to run twice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_hyprslug_headers import _wide_model  # noqa: E402

from hypernix.quant.hyprslug import quantize_gguf  # noqa: E402
from hypernix.t1api.modelindex import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    IndexError_,
    build_entry,
    index_directory,
    inspect,
    model_id_for,
    write_registry,
)
from hypernix.t1api.modelindex_cli import main as index_main  # noqa: E402
from hypernix.t1api.registry import ModelRegistry  # noqa: E402


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> Path:
    """One model, three encodings, plus a file that is not a GGUF."""
    root = tmp_path_factory.mktemp("index") / "models"
    root.mkdir(parents=True)
    source = _wide_model(root / "toy-f32.gguf")
    quantize_gguf(source, root / "toy-IQ0.9_L.gguf", "IQ0.9_L",
                  quantize_embeddings=True, quantize_output=True)
    quantize_gguf(source, root / "toy-Q4_K_M.gguf", "Q4_K_M")
    (root / "broken.gguf").write_bytes(b"not a gguf at all")
    return root


class TestItReadsRatherThanGuesses:
    def test_the_parameter_count_comes_from_the_tensor_table(self, models):
        """Not from the filename, and not from the file size."""
        found = inspect(models / "toy-f32.gguf")

        assert found.parameters_b > 0
        assert found.architecture == "llama"

    def test_quantising_does_not_change_the_parameter_count(self, models):
        """Three encodings of one model are one model. A count derived
        from file size would say otherwise, by a factor of thirty."""
        counts = {
            inspect(models / name).parameters_b
            for name in ("toy-f32.gguf", "toy-IQ0.9_L.gguf", "toy-Q4_K_M.gguf")
        }

        assert len(counts) == 1, counts

    def test_the_files_do_differ_in_size(self, models):
        """So the test above is not vacuous."""
        sizes = {
            (models / name).stat().st_size
            for name in ("toy-f32.gguf", "toy-IQ0.9_L.gguf", "toy-Q4_K_M.gguf")
        }

        assert len(sizes) == 3

    def test_the_sub_bit_tier_is_recognised(self, models):
        found = inspect(models / "toy-IQ0.9_L.gguf")

        assert found.tier == "IQ0.9_L"
        assert found.is_extension is True
        assert found.bits_per_weight == pytest.approx(0.9375)

    def test_an_upstream_quant_is_not_marked_as_needing_hnxrun(self, models):
        found = inspect(models / "toy-Q4_K_M.gguf")

        assert found.is_extension is False

    def test_a_missing_value_is_defaulted_and_said_to_be_assumed(self, models):
        """The toy model carries no context_length. Defaulting is fine;
        defaulting silently is what turns a guess into a limit the
        server enforces without anyone knowing it was invented."""
        found = inspect(models / "toy-f32.gguf")

        assert found.context_limit == 8192
        assert "context_limit" in found.assumed

    def test_an_unreadable_file_is_reported_not_raised(self, models):
        found = inspect(models / "broken.gguf")

        assert not found.readable
        assert "GGUF magic" in found.error


class TestTheSlug:
    def test_it_keeps_the_quantisation(self):
        """Two quantisations of one model are two entries with different
        limits and different quality. One id could not say which is
        being served."""
        assert model_id_for("toy-IQ0.9_L.gguf") != model_id_for("toy-Q4_K_M.gguf")

    def test_it_is_a_slug(self):
        assert model_id_for("Qwen3.8-2B-IQ0.9_L.gguf") == "qwen3-8-2b-iq0-9-l"

    def test_it_never_ends_up_empty(self):
        assert model_id_for("---.gguf") == "model"


class TestWalkingTheDirectory:
    def test_it_finds_every_gguf_and_keeps_going_past_a_bad_one(self, models):
        rows = index_directory(models)

        assert len(rows) == 4
        assert sum(1 for r in rows if not r.readable) == 1

    def test_a_missing_directory_says_what_to_do(self, tmp_path):
        with pytest.raises(IndexError_, match="--dir"):
            index_directory(tmp_path / "absent")

    def test_the_default_is_the_documented_one(self):
        assert DEFAULT_MODELS_DIR == Path("./hypernix/models")


class TestPolicyIsNotDerived:
    def test_price_and_plan_come_from_the_caller(self, models):
        entry = build_entry(
            inspect(models / "toy-f32.gguf"),
            plan="pro", input_price=0.5, output_price=1.5, currency="GBP",
        )

        assert entry.minimum_plan == "pro"
        assert entry.pricing.input_price_per_1k == 0.5
        assert entry.pricing.currency == "GBP"
        assert entry.free_tier_available is False

    def test_a_zero_price_marks_the_free_tier(self, models):
        entry = build_entry(inspect(models / "toy-f32.gguf"))

        assert entry.free_tier_available is True

    def test_the_output_budget_cannot_exceed_the_window(self, models):
        """The prompt has to fit in the same context."""
        entry = build_entry(inspect(models / "toy-f32.gguf"))

        assert entry.output_token_limit <= entry.context_limit
        assert entry.input_token_limit == entry.context_limit

    def test_the_notes_say_the_tier_and_what_runs_it(self, models):
        entry = build_entry(inspect(models / "toy-IQ0.9_L.gguf"))

        assert "IQ0.9_L" in entry.notes
        assert "hnxrun" in entry.notes

    def test_the_notes_admit_an_assumed_value(self, models):
        entry = build_entry(inspect(models / "toy-f32.gguf"))

        assert "assumed" in entry.notes


class TestTheServerCanActuallyLoadWhatWasWritten:
    """The only claim that matters. A registry this writes and the
    server rejects is worse than no registry."""

    def test_the_real_loader_accepts_it(self, models, tmp_path):
        rows = [r for r in index_directory(models) if r.readable]
        target = tmp_path / "models.json"

        write_registry([build_entry(r) for r in rows], target)
        registry = ModelRegistry.load(target)

        assert len(registry) == 3

    def test_every_entry_is_routable(self, models, tmp_path):
        """An entry the router will not dispatch to is a model that was
        indexed and still cannot be reached."""
        rows = [r for r in index_directory(models) if r.readable]
        target = tmp_path / "models.json"
        write_registry([build_entry(r) for r in rows], target)

        registry = ModelRegistry.load(target)

        assert all(e.is_routable for e in registry.list())

    def test_require_finds_them_by_id(self, models, tmp_path):
        rows = [r for r in index_directory(models) if r.readable]
        target = tmp_path / "models.json"
        write_registry([build_entry(r) for r in rows], target)
        registry = ModelRegistry.load(target)

        assert registry.require("toy-iq0-9-l").architecture == "llama"


class TestReIndexingIsSafe:
    """The property that decides whether this can be run twice."""

    def _write_once(self, models, tmp_path) -> Path:
        rows = [r for r in index_directory(models) if r.readable]
        target = tmp_path / "models.json"
        write_registry([build_entry(r) for r in rows], target)
        return target

    def _edit(self, target: Path) -> None:
        data = json.loads(target.read_text(encoding="utf-8"))
        for entry in data:
            if entry["model_id"] == "toy-q4-k-m":
                entry["minimum_plan"] = "pro"
                entry["routing_priority"] = 1
                entry["notes"] = "hand-tuned"
                entry["pricing"] = {
                    "input_price_per_1k": 0.5,
                    "output_price_per_1k": 1.5,
                    "currency": "GBP",
                }
        # Trailing newline, as the tool writes it and as an editor would
        # leave it: otherwise the next run normalises the formatting and
        # a byte-comparison measures that instead of the data.
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def _entry(self, target: Path, model_id: str) -> dict:
        return next(
            e for e in json.loads(target.read_text(encoding="utf-8"))
            if e["model_id"] == model_id
        )

    def test_a_plain_rerun_changes_nothing(self, models, tmp_path):
        target = self._write_once(models, tmp_path)
        self._edit(target)
        before = target.read_text(encoding="utf-8")

        rows = [r for r in index_directory(models) if r.readable]
        result = write_registry([build_entry(r) for r in rows], target)

        assert target.read_text(encoding="utf-8") == before
        assert not result["added"] and not result["updated"]

    def test_an_unchanged_registry_is_not_even_rewritten(self, models, tmp_path):
        """Not just "the content is the same" -- the file is not touched.

        Re-indexing an unchanged folder is what running this twice does,
        and rewriting identical bytes moves the mtime, which is what a
        file watcher or a reload hook keys on.
        """
        target = self._write_once(models, tmp_path)
        before = target.stat().st_mtime_ns

        rows = [r for r in index_directory(models) if r.readable]
        result = write_registry([build_entry(r) for r in rows], target)

        assert result["written"] is False
        assert target.stat().st_mtime_ns == before

    def test_refresh_still_keeps_the_policy_fields(self, models, tmp_path):
        target = self._write_once(models, tmp_path)
        self._edit(target)

        rows = [r for r in index_directory(models) if r.readable]
        write_registry([build_entry(r) for r in rows], target, refresh=True)

        entry = self._entry(target, "toy-q4-k-m")
        assert entry["minimum_plan"] == "pro"
        assert entry["routing_priority"] == 1
        assert entry["notes"] == "hand-tuned"
        assert entry["pricing"]["currency"] == "GBP"

    def test_a_new_model_is_added_beside_the_edited_ones(self, models, tmp_path):
        target = self._write_once(models, tmp_path)
        self._edit(target)
        newcomer = models.parent / "extra"
        newcomer.mkdir(exist_ok=True)
        _wide_model(newcomer / "second-f32.gguf")

        rows = [r for r in index_directory(newcomer) if r.readable]
        result = write_registry([build_entry(r) for r in rows], target)

        assert "second-f32" in result["added"]
        assert self._entry(target, "toy-q4-k-m")["minimum_plan"] == "pro"

    def test_a_registry_that_is_not_json_is_refused_not_overwritten(
        self, models, tmp_path
    ):
        target = tmp_path / "models.json"
        target.write_text("{ this is not json", encoding="utf-8")

        with pytest.raises(IndexError_, match="not valid JSON"):
            write_registry([], target)

        assert target.read_text(encoding="utf-8") == "{ this is not json"


class TestTheCommandLine:
    def _run(self, *argv) -> tuple[int, str]:
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = index_main(list(argv))
        return code, out.getvalue()

    def test_a_dry_run_writes_nothing(self, models, tmp_path):
        target = tmp_path / "models.json"

        code, text = self._run("--dir", str(models), "-o", str(target), "--dry-run")

        assert not target.exists()
        assert "nothing written" in text
        assert code == 2  # the unreadable file

    def test_it_writes_and_reports_what_it_added(self, models, tmp_path):
        target = tmp_path / "models.json"

        _code, text = self._run("--dir", str(models), "-o", str(target))

        assert target.exists()
        assert "added" in text
        assert "toy-iq0-9-l" in text

    def test_json_output_parses(self, models, tmp_path):
        target = tmp_path / "models.json"

        _code, text = self._run(
            "--dir", str(models), "-o", str(target), "--json"
        )

        payload = json.loads(text)
        assert payload["total"] == 3
        assert len(payload["models"]) == 4

    def test_an_unreadable_file_makes_the_exit_non_zero(self, models, tmp_path):
        """The registry written is missing a model someone put there."""
        code, _text = self._run(
            "--dir", str(models), "-o", str(tmp_path / "m.json")
        )

        assert code == 2

    def test_a_clean_directory_exits_zero(self, models, tmp_path):
        clean = tmp_path / "clean"
        clean.mkdir()
        _wide_model(clean / "only-f32.gguf")

        code, _text = self._run(
            "--dir", str(clean), "-o", str(tmp_path / "m.json")
        )

        assert code == 0

    def test_a_missing_directory_is_an_error_not_a_traceback(self, tmp_path):
        code, text = self._run("--dir", str(tmp_path / "nope"))

        assert code == 1
        assert "No such directory" in text

    def test_the_refresh_hint_is_absent_when_refresh_was_passed(
        self, models, tmp_path
    ):
        """Telling someone to pass the flag they just passed."""
        target = tmp_path / "models.json"
        self._run("--dir", str(models), "-o", str(target))

        _code, text = self._run(
            "--dir", str(models), "-o", str(target), "--refresh"
        )

        assert "pass --refresh" not in text

    def test_the_hint_is_present_without_it(self, models, tmp_path):
        target = tmp_path / "models.json"
        self._run("--dir", str(models), "-o", str(target))

        _code, text = self._run("--dir", str(models), "-o", str(target))

        assert "pass --refresh" in text


class TestTheShellWrapper:
    SCRIPT = Path(__file__).resolve().parents[2] / "bin" / "hypernix-t1"

    def test_index_is_dispatched(self):
        source = self.SCRIPT.read_text(encoding="utf-8")

        assert "index)" in source
        assert "cmd_index()" in source

    def test_it_is_documented_in_the_usage(self):
        source = self.SCRIPT.read_text(encoding="utf-8")

        assert "index [--dir D]" in source

    def test_it_does_not_require_a_configured_server(self):
        """Someone who has just dropped models in a folder has not run
        `create` yet, and indexing needs neither the server nor the
        [t1api] extra."""
        source = self.SCRIPT.read_text(encoding="utf-8")
        body = source[source.index("cmd_index()"):source.index("cmd_autostart()")]
        # Comments stripped: the function explains *why* it does not call
        # require_installed, and a plain substring search finds that
        # explanation and reads it as the call it is denying.
        code = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("#")
        )

        assert "require_installed" not in code


class TestTheWaiterDiagnosesARefusedConnection:
    """`waiter serv -A` against a server that is not running said only

        ✗ Automatic setup failed: Could not reach
          http://192.168.1.95:8000/auth/t1/validate: [Errno 111] Connection
          refused

    which is accurate and answers none of the questions the reader has.
    The diagnosis machinery to answer them already existed in
    `waiter.diagnose` — it was wired into exactly one of a dozen
    T1ClientError handlers, and `serv -A`, the first command anyone runs
    after an install, was one of the eleven that got the bare errno.
    """

    REFUSED = (
        "Could not reach http://192.168.1.95:8000/auth/t1/validate: "
        "[Errno 111] Connection refused"
    )

    def _render(self, context: str = "") -> str:
        import contextlib
        import io

        from hypernix.waiter import cli

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cli._err_connection(RuntimeError(self.REFUSED), context)
        return err.getvalue()

    def test_it_says_nothing_is_listening(self):
        text = self._render("Automatic setup failed")

        assert "Nothing is listening on 192.168.1.95:8000" in text

    def test_it_names_the_command_that_starts_the_server(self):
        """Not a shell script the reader may not have — the CLI the
        installer leaves behind."""
        text = self._render("Automatic setup failed")

        assert "hypernix-t1 start" in text

    def test_it_keeps_the_context_of_what_failed(self):
        text = self._render("Automatic setup failed")

        assert "Automatic setup failed" in text

    def test_it_does_not_state_the_same_error_twice(self):
        text = self._render("Automatic setup failed")

        assert text.count("[Errno 111] Connection refused") == 1

    def test_a_non_connection_error_is_passed_through_unchanged(self):
        """Only unreachability earns the extra work."""
        import contextlib
        import io

        from hypernix.waiter import cli

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cli._err_connection(RuntimeError("401 Unauthorized"), "Sync failed")

        assert "401 Unauthorized" in err.getvalue()
        assert "Nothing is listening" not in err.getvalue()

    def test_every_client_error_handler_is_routed_through_it(self):
        """The gap was that eleven of twelve were not."""
        import re
        from pathlib import Path

        import hypernix.waiter.cli as cli_module

        source = Path(cli_module.__file__).read_text(encoding="utf-8")
        bare = re.findall(
            r"except T1ClientError as exc:\n(?:\s*#[^\n]*\n)*\s+_err\(", source
        )

        assert not bare, f"{len(bare)} T1ClientError handler(s) still bypass the diagnosis"


class TestServeAcceptsAModelDirectory:
    """`hyprslug-headers serve <dir>` was refused.

    LM Studio's layout is ``<root>/<publisher>/<name>/<name>.gguf`` — and
    that is precisely the layout ``install-model`` writes. So the
    directory is what tab-completion stops at, what gets pasted, and what
    these commands would not accept, having just created it.
    """

    @pytest.fixture
    def store(self, models, tmp_path) -> Path:
        root = tmp_path / "lmstudio"
        (root / "HyperNix" / "Qwen3.8-2B").mkdir(parents=True)
        (root / "HyperNix" / "Qwen3.8-2B" / "Qwen3.8-2B.gguf").write_bytes(
            (models / "toy-IQ0.9_L.gguf").read_bytes()
        )
        (root / "Two").mkdir()
        for name in ("a.gguf", "b.gguf"):
            (root / "Two" / name).write_bytes((models / "toy-f32.gguf").read_bytes())
        (root / "Empty").mkdir()
        return root

    def test_a_directory_with_one_model_resolves_to_it(self, store):
        from hypernix.quant.hyprslug_headers import resolve_model_path

        found = resolve_model_path(store / "HyperNix" / "Qwen3.8-2B")

        assert found.name == "Qwen3.8-2B.gguf"

    def test_a_file_is_still_returned_unchanged(self, models):
        from hypernix.quant.hyprslug_headers import resolve_model_path

        target = models / "toy-f32.gguf"

        assert resolve_model_path(target) == target

    def test_a_publisher_folder_resolves_when_it_holds_one_model(self, store):
        from hypernix.quant.hyprslug_headers import resolve_model_path

        assert resolve_model_path(store / "HyperNix").name == "Qwen3.8-2B.gguf"

    def test_read_header_accepts_the_directory(self, store):
        from hypernix.quant.hyprslug_headers import read_header

        assert read_header(store / "HyperNix" / "Qwen3.8-2B").tier == "IQ0.9_L"

    def test_two_models_is_refused_with_the_list(self, store):
        """Picking one would be picking which model they meant."""
        from hypernix.quant.hyprslug_headers import HeaderError, resolve_model_path

        with pytest.raises(HeaderError) as caught:
            resolve_model_path(store / "Two")

        message = str(caught.value)
        assert "a.gguf" in message and "b.gguf" in message

    def test_an_empty_directory_says_so(self, store):
        from hypernix.quant.hyprslug_headers import HeaderError, resolve_model_path

        with pytest.raises(HeaderError, match="no .gguf in it"):
            resolve_model_path(store / "Empty")

    def test_an_absent_path_still_says_no_such_model(self, tmp_path):
        from hypernix.quant.hyprslug_headers import HeaderError, resolve_model_path

        with pytest.raises(HeaderError, match="No such model"):
            resolve_model_path(tmp_path / "nope")

    def test_the_server_reports_the_ambiguity_rather_than_guessing(self, store):
        from hypernix.quant.hyprslug_server import HyprslugModel, ServerError

        with pytest.raises(ServerError, match="not clear which one"):
            HyprslugModel(store / "Two")
