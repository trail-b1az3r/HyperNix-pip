"""Running a model without LM Studio, and deciding where it goes.

"Switch model" meant walking over to the PC. The server could hold forty
GGUFs in ~/.hypernix/models and serve none of them, because the only
thing it could reach was an LM Studio somebody had started by hand.

Two halves here. `plan_placement` decides where the weights sit — the
part that decides whether a model runs at all on a given machine — and
`/runner/*` is load, unload and switch as operations rather than
instructions.

The placement tests pass VRAM and RAM explicitly rather than reading the
machine, because a test whose result depends on whether the runner has a
GPU tests the runner.
"""
from __future__ import annotations

import logging

import pytest

from hypernix.hyperlink.managed import (
    BACKENDS,
    ManagedError,
    ManagedRunner,
    plan_placement,
)

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class TestPlacementOnAGPU:
    def test_a_model_that_fits_goes_entirely_on_the_card(self):
        plan = plan_placement(
            file_bytes=int(4.9 * GB), total_layers=33,
            vram_free=24 * GB, ram_free=32 * GB,
        )
        assert plan.gpu_layers == 33
        assert plan.fully_offloaded

    def test_a_model_that_does_not_fit_is_split(self):
        """The case the request was specifically about: some layers on
        the GPU, the rest in RAM."""
        plan = plan_placement(
            file_bytes=int(40 * GB), total_layers=81,
            vram_free=24 * GB, ram_free=64 * GB,
        )
        assert 0 < plan.gpu_layers < 81
        assert plan.ram_bytes > 0
        assert not plan.fully_offloaded

    def test_the_split_leaves_headroom(self):
        """Guessing one layer too many does not degrade gracefully: CUDA
        returns out-of-memory at load and the model does not run at all.
        So the plan must use less than all of it."""
        plan = plan_placement(
            file_bytes=int(40 * GB), total_layers=80,
            vram_free=20 * GB, ram_free=64 * GB,
        )
        assert plan.vram_bytes < 20 * GB

    def test_context_length_eats_into_the_offload(self):
        """The KV cache lives on the card too. A plan that budgets only
        for weights loads and then runs out of memory on a long
        conversation."""
        short = plan_placement(
            file_bytes=int(20 * GB), total_layers=60,
            vram_free=24 * GB, ram_free=64 * GB, context_length=2048,
        )
        long = plan_placement(
            file_bytes=int(20 * GB), total_layers=60,
            vram_free=24 * GB, ram_free=64 * GB, context_length=131072,
        )
        assert long.gpu_layers <= short.gpu_layers

    def test_no_vram_means_everything_in_ram(self):
        plan = plan_placement(
            file_bytes=int(4.9 * GB), total_layers=33,
            vram_free=0, ram_free=32 * GB,
        )
        assert plan.gpu_layers == 0
        assert plan.ram_bytes == int(4.9 * GB)


class TestAnExplicitNumberIsHonoured:
    """Somebody who has tuned their own machine should not have their
    number second-guessed."""

    def test_the_asked_for_count_is_used(self):
        plan = plan_placement(
            file_bytes=int(40 * GB), total_layers=81, gpu_layers=20,
            vram_free=24 * GB, ram_free=64 * GB,
        )
        assert plan.gpu_layers == 20
        assert plan.explicit

    def test_it_is_capped_at_what_the_model_has(self):
        """Asking for 200 layers of an 81-layer model is a typo, and
        passing it through produces a confusing llama.cpp error."""
        plan = plan_placement(
            file_bytes=int(40 * GB), total_layers=81, gpu_layers=200,
            vram_free=80 * GB, ram_free=64 * GB,
        )
        assert plan.gpu_layers == 81

    def test_zero_is_a_real_answer(self):
        """"Run it on the CPU" is a thing people ask for deliberately,
        and must not be confused with "work it out"."""
        plan = plan_placement(
            file_bytes=int(4.9 * GB), total_layers=33, gpu_layers=0,
            vram_free=24 * GB, ram_free=32 * GB,
        )
        assert plan.gpu_layers == 0
        assert plan.explicit

    def test_an_automatic_plan_is_not_marked_explicit(self):
        plan = plan_placement(
            file_bytes=int(4.9 * GB), total_layers=33,
            vram_free=24 * GB, ram_free=32 * GB,
        )
        assert not plan.explicit


class TestBackends:
    @pytest.mark.parametrize("backend", ["cpu", "cuda", "vulkan", "hnx-cuda", "hnx-cpu"])
    def test_every_advertised_backend_is_accepted(self, backend):
        plan = plan_placement(
            file_bytes=GB, total_layers=10, backend=backend,
            vram_free=8 * GB, ram_free=16 * GB,
        )
        assert plan.backend == backend

    def test_cpu_puts_nothing_on_the_card_even_with_vram_free(self):
        plan = plan_placement(
            file_bytes=GB, total_layers=10, backend="cpu",
            vram_free=80 * GB, ram_free=16 * GB,
        )
        assert plan.gpu_layers == 0

    def test_an_unknown_backend_is_refused_by_name(self):
        with pytest.raises(ManagedError, match="Unknown backend"):
            plan_placement(file_bytes=GB, backend="metal-3000")

    def test_the_refusal_lists_the_real_ones(self):
        with pytest.raises(ManagedError) as refused:
            plan_placement(file_bytes=GB, backend="nonsense")
        for name in BACKENDS:
            assert name in str(refused.value)


class TestSwapIsReportedNotUsed:
    """A model paging through swap produces tokens at a rate that reads
    as a hang. Offering it as a placement would be offering something
    nobody wants."""

    def test_swap_is_never_a_target(self):
        plan = plan_placement(
            file_bytes=int(40 * GB), total_layers=81,
            vram_free=0, ram_free=4 * GB,
        )
        assert plan.to_dict()["ram_bytes"] == int(40 * GB)
        assert "swap_used_bytes" in plan.to_dict()

    def test_a_tight_fit_is_said_out_loud(self):
        """Not refused — somebody may know something the planner does
        not — but not silent either."""
        plan = plan_placement(
            file_bytes=int(40 * GB), total_layers=81,
            vram_free=0, ram_free=8 * GB,
        )
        assert "tight" in plan.reason


class TestTheRunnerStartsEmpty:
    def test_nothing_is_loaded(self):
        assert ManagedRunner(port=8799).current is None

    def test_unloading_nothing_is_a_success(self):
        assert ManagedRunner(port=8799).unload() is False

    def test_a_missing_file_is_refused_before_anything_starts(self, tmp_path):
        runner = ManagedRunner(port=8799)
        with pytest.raises(ManagedError, match="No such model"):
            runner.load(tmp_path / "nope.gguf")
        assert runner.current is None

    def test_the_base_url_is_where_clients_should_look(self):
        runner = ManagedRunner(port=8799, host="127.0.0.1")
        assert runner.base_url == "http://127.0.0.1:8799"


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def models_dir(tmp_path):
    import struct

    from hypernix.quant.gguf import GGMLType, GGUFWriter

    directory = tmp_path / "models"
    directory.mkdir()
    path = directory / "qwen3-8b-q4-k-m.gguf"
    writer = GGUFWriter(path)
    writer.set_metadata("general.architecture", "llama")
    writer.set_metadata("general.name", "Qwen3 8B")
    writer.set_metadata("llama.block_count", 1)
    writer.add_tensor("token_embd.weight", (256, 4), int(GGMLType.F32))
    writer.write(lambda t: struct.pack("<1024f", *([0.01] * 1024)))
    return directory


def client(models_dir, monkeypatch, **env) -> TestClient:
    from hypernix.t1api.app import create_app

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    monkeypatch.setenv("T1_HF_DOWNLOAD_DIR", str(models_dir))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return TestClient(create_app(), client=("192.168.1.50", 5432))


class TestStatusIsOpen:
    """Knowing which model is answering is not an administrative secret,
    and a client that cannot tell shows the wrong model name."""

    def test_any_caller_can_read_it(self, models_dir, monkeypatch):
        response = client(models_dir, monkeypatch).get("/runner/status")
        assert response.status_code == 200

    def test_it_starts_unloaded(self, models_dir, monkeypatch):
        body = client(models_dir, monkeypatch).get("/runner/status").json()
        assert body["loaded"] is False

    def test_it_says_what_backends_exist(self, models_dir, monkeypatch):
        body = client(models_dir, monkeypatch).get("/runner/status").json()
        assert "cuda" in body["backends"]
        assert "vulkan" in body["backends"]


class TestPlanningChangesNothing:
    def test_it_reports_a_placement(self, models_dir, monkeypatch):
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1")
        body = app.post(
            "/runner/plan", json={"model_id": "qwen3-8b-q4-k-m"}
        ).json()
        assert "placement" in body
        assert body["placement"]["reason"]

    def test_nothing_is_loaded_afterwards(self, models_dir, monkeypatch):
        """Loading evicts whatever people are talking to. Seeing the
        consequence first is not a nicety."""
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1")
        app.post("/runner/plan", json={"model_id": "qwen3-8b-q4-k-m"})
        assert app.get("/runner/status").json()["loaded"] is False

    def test_an_unknown_model_is_a_404_naming_what_exists(self, models_dir, monkeypatch):
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1")
        response = app.post("/runner/plan", json={"model_id": "not-here"})
        assert response.status_code == 404
        assert "qwen3-8b-q4-k-m" in response.text


class TestWhoMaySwitch:
    """Changing what a shared server runs affects everybody using it."""

    def test_a_read_only_caller_cannot_load(self, models_dir, monkeypatch):
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0")
        response = app.post("/runner/load", json={"model_id": "qwen3-8b-q4-k-m"})
        assert response.status_code == 403

    def test_a_read_only_caller_cannot_unload(self, models_dir, monkeypatch):
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0")
        assert app.post("/runner/unload").status_code == 403

    def test_the_refusal_names_every_way_in(self, models_dir, monkeypatch):
        """A 403 that does not say how to stop being a 403 sends somebody
        to widen a key's scopes, which is not what this checks."""
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0")
        details = app.post("/runner/unload").json()["error"]["details"]
        assert "T1_RUNNER_SWITCH_PERM" in details["remedy"]
        assert "PARTIAL_ADMIN" in details["remedy"]

    def test_partial_admin_may(self, models_dir, monkeypatch):
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1")
        assert app.post("/runner/unload").status_code == 200

    def test_switch_perm_is_off_unless_configured(self, models_dir, monkeypatch):
        """"access 6+ if servers enable it" — the enabling is the
        operator's, and the default is off."""
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0")
        details = app.post("/runner/unload").json()["error"]["details"]
        assert details["switch_perm_enabled"] is False

    def test_configuring_it_is_reported(self, models_dir, monkeypatch):
        app = client(
            models_dir, monkeypatch,
            T1_TRUSTED_NETWORK_PARTIAL_ADMIN="0", T1_RUNNER_SWITCH_PERM="6",
        )
        details = app.post("/runner/unload").json()["error"]["details"]
        assert details["switch_perm_enabled"] is True
        assert details["switch_perm_level"] == 6


class TestLoadingWithoutAnEngine:
    def test_it_says_there_is_no_llama_cpp_rather_than_500ing(
        self, models_dir, monkeypatch
    ):
        """The common case on a fresh machine, and something the operator
        can act on. A 500 hides the remedy."""
        app = client(models_dir, monkeypatch, T1_TRUSTED_NETWORK_PARTIAL_ADMIN="1")
        response = app.post("/runner/load", json={"model_id": "qwen3-8b-q4-k-m"})
        assert response.status_code in (400, 422), response.text
        assert "llama" in response.text.lower()
