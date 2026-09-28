"""hyperchat — several prompts at once, or a queue when there cannot be.

What these tests are actually for
---------------------------------
Two of the three things this module has to get right are invisible when
they are wrong. A pool that allocates every core still answers prompts —
right up until the server stops accepting the next request, under
exactly the load the pool was added for. And a "queue" built on a mutex
still serves everybody, in an order nobody can predict, with one
unlucky request waiting behind prompts sent minutes after it.

So the assertions are on the core budget arithmetic and on *ordering*,
not on "the answer came back".

Nothing is mocked at the threading level: the workers are real threads
and the generators are real callables that block. A test that replaced
the queue with a list would be testing the list.
"""
from __future__ import annotations

import threading
import time

import pytest

from hypernix.hyperlink.hyperchat import (
    CORES_PER_INSTANCE,
    MAX_INSTANCES,
    RESERVED_CORES,
    Hyperchat,
    QueueFull,
    plan_cores,
)


def echo(prompt: str, **_) -> str:
    return f"answer: {prompt}"


def echo_factory():
    return echo


# ---------------------------------------------------------------------------
# The core budget
# ---------------------------------------------------------------------------


class TestPlanCores:
    def test_a_core_is_always_held_back(self):
        """The one that matters.

        llama.cpp takes every core it is given. With all of them spoken
        for, the process that has to accept the next HTTP request or
        notice a child died does not get scheduled, and the symptom is a
        server that stops responding under the load this feature exists
        to handle.
        """
        for cores in range(1, 65):
            budget = plan_cores(cores)
            assert budget.cores_used <= cores - RESERVED_CORES or budget.instances == 1

    def test_the_arithmetic(self):
        assert plan_cores(8).instances == 3       # (8 - 1) // 2
        assert plan_cores(16).instances == 7
        assert plan_cores(5).instances == 2

    def test_a_machine_too_small_for_a_pool_gets_the_queue(self):
        """Not an error and not a refusal: one instance answering in
        order is a real answer, and the note says which you got."""
        for cores in (1, 2, 3, 4):
            budget = plan_cores(cores)
            assert budget.instances == 1
            assert not budget.pooling
            assert budget.to_dict()["mode"] == "queue"

    def test_a_small_machine_says_why(self):
        note = plan_cores(2).note
        assert "queue" in note and "2" in note

    def test_asking_for_more_than_the_cores_allow_lowers_it(self):
        """An operator asking for six instances on a four-core box is
        asking for the machine to stop responding. The honest reply is
        the number it can have."""
        budget = plan_cores(8, wanted=6)
        assert budget.instances == 3
        assert "6" in budget.note and "3" in budget.note

    def test_asking_for_fewer_is_honoured(self):
        assert plan_cores(64, wanted=2).instances == 2

    def test_there_is_a_ceiling_however_many_cores(self):
        """Past a point the instances compete for memory bandwidth
        rather than adding throughput."""
        assert plan_cores(256).instances == MAX_INSTANCES

    def test_the_default_reads_the_machine(self):
        assert plan_cores().total_cores >= 1

    def test_cores_free_accounts_for_the_reservation(self):
        budget = plan_cores(8)
        assert budget.cores_used == 6
        assert budget.cores_free == 2

    def test_cores_per_instance_is_two(self):
        assert CORES_PER_INSTANCE == 2
        assert plan_cores(9).cores_per_instance == 2


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


class TestQueueing:
    def test_one_instance_answers_in_the_order_prompts_arrived(self):
        """The other one that matters.

        A mutex hands the next prompt to whichever thread the scheduler
        happens to wake. Replace the queue with a lock and this fails —
        eventually, and only under load, which is the worst way for it
        to fail.
        """
        order: list[str] = []
        gate = threading.Event()

        def slow(prompt: str, **_) -> str:
            gate.wait(2.0)
            order.append(prompt)
            return prompt

        with Hyperchat(lambda: slow, budget=plan_cores(2)) as chat:
            tickets = [chat.submit(f"p{i}") for i in range(12)]
            gate.set()
            for ticket in tickets:
                ticket.wait(10)
        assert order == [f"p{i}" for i in range(12)]

    def test_a_ticket_knows_its_position(self):
        """A phone showing "third in line" is worth far more than one
        showing a spinner that does not move."""
        gate = threading.Event()

        def slow(prompt: str, **_) -> str:
            gate.wait(2.0)
            return prompt

        with Hyperchat(lambda: slow, budget=plan_cores(2)) as chat:
            first = chat.submit("a")
            time.sleep(0.05)
            second = chat.submit("b")
            third = chat.submit("c")
            assert chat.position(third) >= chat.position(second)
            assert chat.position(first) == 0     # running, or next
            gate.set()
            for ticket in (first, second, third):
                ticket.wait(10)

    def test_a_waiting_prompt_can_be_cancelled(self):
        gate = threading.Event()

        with Hyperchat(lambda: (lambda p, **_: gate.wait(2.0) or p),
                       budget=plan_cores(2)) as chat:
            chat.submit("running")
            time.sleep(0.05)
            waiting = chat.submit("waiting")
            assert waiting.cancel() is True
            assert waiting.state == "cancelled"
            gate.set()

    def test_a_cancelled_prompt_is_never_generated(self):
        """Cancelling has to actually stop the work, or it is only a UI
        change with the cost still paid."""
        seen: list[str] = []
        gate = threading.Event()

        def watch(prompt: str, **_) -> str:
            gate.wait(2.0)
            seen.append(prompt)
            return prompt

        with Hyperchat(lambda: watch, budget=plan_cores(2)) as chat:
            chat.submit("first")
            time.sleep(0.05)
            doomed = chat.submit("doomed")
            doomed.cancel()
            gate.set()
            time.sleep(0.3)
        assert "doomed" not in seen

    def test_a_running_prompt_cannot_be_cancelled(self):
        """The token loop is inside llama.cpp; stopping it half way
        leaves the instance in a state this module cannot reason about."""
        started = threading.Event()
        gate = threading.Event()

        def slow(prompt: str, **_) -> str:
            started.set()
            gate.wait(2.0)
            return prompt

        with Hyperchat(lambda: slow, budget=plan_cores(2)) as chat:
            ticket = chat.submit("a")
            assert started.wait(2.0)
            assert ticket.cancel() is False
            gate.set()
            ticket.wait(10)

    def test_the_queue_is_bounded(self):
        """An unbounded queue is a memory leak with a waiting list
        attached, and a prompt accepted now and answered in an hour is
        worse than one refused now."""
        gate = threading.Event()

        with Hyperchat(lambda: (lambda p, **_: gate.wait(2.0) or p),
                       budget=plan_cores(2), max_queued=3) as chat:
            for _ in range(3):
                chat.submit("x")
            with pytest.raises(QueueFull):
                for _ in range(5):
                    chat.submit("x")
            gate.set()

    def test_the_refusal_says_what_to_do(self):
        gate = threading.Event()
        with Hyperchat(lambda: (lambda p, **_: gate.wait(2.0) or p),
                       budget=plan_cores(2), max_queued=1) as chat:
            with pytest.raises(QueueFull) as caught:
                for _ in range(6):
                    chat.submit("x")
            assert "instance" in str(caught.value)
            gate.set()


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class TestPooling:
    def test_several_prompts_really_do_run_at_once(self):
        """Not "they all finished" — that is true of a queue too. Three
        generators have to be *inside* their call simultaneously."""
        inside = threading.Semaphore(0)
        release = threading.Event()
        peak = {"n": 0}
        live = {"n": 0}
        lock = threading.Lock()

        def slow(prompt: str, **_) -> str:
            with lock:
                live["n"] += 1
                peak["n"] = max(peak["n"], live["n"])
            inside.release()
            release.wait(3.0)
            with lock:
                live["n"] -= 1
            return prompt

        budget = plan_cores(8)                      # three instances
        with Hyperchat(lambda: slow, budget=budget) as chat:
            tickets = [chat.submit(f"p{i}") for i in range(3)]
            for _ in range(3):
                assert inside.acquire(timeout=3.0)
            release.set()
            for ticket in tickets:
                ticket.wait(10)
        assert peak["n"] == 3

    def test_each_instance_is_built_once_and_only_when_used(self):
        """N idle workers must not mean N copies of the model in memory
        — on a server that is usually idle, that is the difference
        between holding the model once and holding it three times."""
        built = {"n": 0}
        lock = threading.Lock()

        def factory():
            with lock:
                built["n"] += 1
            return echo

        chat = Hyperchat(factory, budget=plan_cores(8))
        assert built["n"] == 0                      # nothing asked yet
        chat.ask("one", timeout=5)
        assert built["n"] == 1                      # one worker woke up
        chat.close()

    def test_a_failing_prompt_does_not_take_the_pool_down(self):
        calls = {"n": 0}

        def sometimes(prompt: str, **_) -> str:
            calls["n"] += 1
            if prompt == "bad":
                raise RuntimeError("context overflow")
            return prompt

        with Hyperchat(lambda: sometimes, budget=plan_cores(2)) as chat:
            bad = chat.submit("bad")
            with pytest.raises(RuntimeError):
                bad.wait(5)
            assert bad.state == "failed"
            assert chat.ask("good", timeout=5) == "good"

    def test_a_failed_instance_is_rebuilt_rather_than_reused(self):
        """A generator that raised may have a half-consumed context in
        it, and the next prompt would inherit it."""
        built = {"n": 0}

        def factory():
            built["n"] += 1

            def generate(prompt: str, **_) -> str:
                if prompt == "bad":
                    raise RuntimeError("boom")
                return prompt

            return generate

        with Hyperchat(factory, budget=plan_cores(2)) as chat:
            with pytest.raises(RuntimeError):
                chat.submit("bad").wait(5)
            chat.ask("good", timeout=5)
        assert built["n"] == 2


# ---------------------------------------------------------------------------
# One shape either way
# ---------------------------------------------------------------------------


class TestOneShape:
    @pytest.mark.parametrize("cores", [2, 8])
    def test_the_api_is_the_same_whichever_mode(self, cores):
        """Callers must not branch. Anything else means HyperLink, the
        API and the CLI each implementing a fallback, and each of them
        getting it slightly different."""
        with Hyperchat(echo_factory, budget=plan_cores(cores)) as chat:
            ticket = chat.submit("hello")
            assert ticket.wait(5) == "answer: hello"
            assert chat.ask("again", timeout=5) == "answer: again"
            assert set(chat.stats()) >= {"mode", "instances", "waiting",
                                         "running", "served"}

    def test_stats_report_the_mode(self):
        with Hyperchat(echo_factory, budget=plan_cores(2)) as chat:
            assert chat.stats()["mode"] == "queue"
        with Hyperchat(echo_factory, budget=plan_cores(8)) as chat:
            assert chat.stats()["mode"] == "pool"

    def test_served_and_failed_are_counted(self):
        def sometimes(prompt: str, **_) -> str:
            if prompt == "bad":
                raise RuntimeError("no")
            return prompt

        with Hyperchat(lambda: sometimes, budget=plan_cores(2)) as chat:
            chat.ask("a", timeout=5)
            with pytest.raises(RuntimeError):
                chat.submit("bad").wait(5)
            time.sleep(0.05)
            stats = chat.stats()
        assert stats["served"] == 1
        assert stats["failed"] == 1

    def test_waiting_on_a_prompt_that_never_finishes_times_out(self):
        gate = threading.Event()
        with Hyperchat(lambda: (lambda p, **_: gate.wait(5.0) or p),
                       budget=plan_cores(2)) as chat:
            ticket = chat.submit("a")
            with pytest.raises(TimeoutError):
                ticket.wait(0.2)
            gate.set()

    def test_submitting_after_close_is_refused(self):
        chat = Hyperchat(echo_factory, budget=plan_cores(2))
        chat.close()
        with pytest.raises(RuntimeError):
            chat.submit("a")

    def test_closing_twice_is_fine(self):
        chat = Hyperchat(echo_factory, budget=plan_cores(2))
        chat.close()
        chat.close()


# ---------------------------------------------------------------------------
# Ports, for the pool of llama.cpp processes
# ---------------------------------------------------------------------------


class TestPorts:
    def test_ports_are_consecutive(self):
        from hypernix.hyperlink.managed import allocate_ports

        assert allocate_ports(8781, 3) == [8781, 8782, 8783]

    def test_a_port_to_avoid_is_skipped_not_fatal(self):
        """The port to avoid is normally the T1 server's own, and a
        collision there does not fail loudly — it takes the API down and
        leaves llama.cpp answering on it."""
        from hypernix.hyperlink.managed import allocate_ports

        assert allocate_ports(8000, 3, avoid=[8001]) == [8000, 8002, 8003]

    def test_a_pool_never_lands_on_a_reserved_port(self):
        from hypernix.hyperlink.managed import ManagedPool

        pool = ManagedPool(instances=4, port=8000, avoid_ports=[8001, 8003])
        assert 8001 not in pool.ports
        assert 8003 not in pool.ports
        assert len(set(pool.ports)) == 4

    def test_a_pool_needs_at_least_one_instance(self):
        from hypernix.hyperlink.managed import ManagedPool

        with pytest.raises(ValueError):
            ManagedPool(instances=0)

    def test_an_empty_pool_is_serving_nothing(self):
        from hypernix.hyperlink.managed import ManagedPool

        pool = ManagedPool(instances=2, port=8600)
        assert pool.current is None
        assert pool.live == []
        assert pool.to_dict()["live"] == 0

    def test_a_partial_load_is_rolled_back(self):
        """If instance three of four fails to start, the two that did
        are orphans: holding VRAM, answering nothing, invisible to the
        next load's placement arithmetic."""
        from hypernix.hyperlink.managed import ManagedError, ManagedPool

        pool = ManagedPool(instances=3, port=8700)
        calls = {"load": 0, "unload": 0}

        def fake_load(self, path, **kwargs):
            calls["load"] += 1
            if calls["load"] == 3:
                raise ManagedError("out of VRAM")
            return object()

        def fake_unload(self):
            calls["unload"] += 1
            return False

        for runner in pool._runners:
            runner.load = fake_load.__get__(runner)
            runner.unload = fake_unload.__get__(runner)

        with pytest.raises(ManagedError) as caught:
            pool.load("/tmp/model.gguf")
        assert "rolled back" in str(caught.value)
        # Once before the load, once on the unwind: every instance.
        assert calls["unload"] == 2 * len(pool._runners)


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


def _client(tmp_path, **overrides) -> tuple[TestClient, str]:
    km = Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False)
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        default_plan="free",
        **overrides,
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    key = km.create(key_type=KeyType.ADMIN,
                    scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key
    return TestClient(app, client=("127.0.0.1", 5000)), key


class TestTheHyperchatEndpoint:
    def test_it_reports_the_queue_when_multi_is_off(self, tmp_path):
        client, key = _client(tmp_path)
        got = client.get("/runner/hyperchat",
                         headers={"Authorization": f"Bearer {key}"})
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["enabled"] is False
        assert body["cores"]["mode"] == "queue"
        assert body["cores"]["instances"] == 1

    def test_off_is_distinguishable_from_a_small_machine(self, tmp_path):
        """An operator who set the flag and sees `instances: 1` has to
        be able to tell "disabled" from "this box has three cores"."""
        client, key = _client(tmp_path)
        note = client.get("/runner/hyperchat",
                          headers={"Authorization": f"Bearer {key}"}).json()
        assert "T1_HYPERCHAT_MULTI" in note["cores"]["note"]

    def test_it_reports_the_pool_when_multi_is_on(self, tmp_path):
        client, key = _client(tmp_path, hyperchat_multi=True,
                              hyperchat_instances=3)
        body = client.get("/runner/hyperchat",
                          headers={"Authorization": f"Bearer {key}"}).json()
        assert body["enabled"] is True
        assert body["cores"]["instances"] >= 1
        assert "T1_HYPERCHAT_MULTI" not in body["cores"]["note"]

    def test_the_instance_count_never_exceeds_what_the_cores_allow(self, tmp_path):
        """Asked for on the API, the answer is still what the machine
        can have — the endpoint must not echo the request back."""
        client, key = _client(tmp_path, hyperchat_multi=True,
                              hyperchat_instances=999)
        cores = client.get("/runner/hyperchat",
                           headers={"Authorization": f"Bearer {key}"}
                           ).json()["cores"]
        assert cores["instances"] <= max(1, (cores["total_cores"] - 1) // 2)
        assert cores["instances"] <= MAX_INSTANCES

    def test_it_needs_a_key(self, tmp_path):
        client, _ = _client(tmp_path)
        assert client.get("/runner/hyperchat").status_code in (401, 403)

    def test_the_settings_show_up_in_config(self, tmp_path):
        """An operator debugging "why are my prompts serialised" looks
        at /config first."""
        client, key = _client(tmp_path, hyperchat_multi=True)
        body = client.get("/config",
                          headers={"Authorization": f"Bearer {key}"}).json()
        assert body["config"]["hyperchat_multi"] is True
        assert "hyperchat_max_queued" in body["config"]
