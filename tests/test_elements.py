"""hypernix.elements — hydrogen, natural gas, carbon and magnesium.

What these tests are for
------------------------
A plugin system goes wrong in three characteristic ways, and all three
are quiet. A plugin claims a name that is not what it says it is. A
plugin that raises takes the host down with it. And a plugin that was
"only supposed to" read something writes it — because the permission
list was documentation rather than a check.

Magnesium adds a fourth, and it is the one that matters most: it changes
*other programs*. So its tests run against a real process outside this
one's tree, check the change happened, and check the original value was
put back exactly — not "reset to normal".
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from hypernix.elements import carbon, natural_gas
from hypernix.elements.builtins import BUILTINS, Hydrogen
from hypernix.elements.carbon import Carbon, expand_snippets, tidy
from hypernix.elements.hydrogen import (
    GENERATION,
    SYMBOLS,
    Element,
    ElementSpec,
    Registry,
    atomic_number,
    is_experimental,
    period_of,
)
from hypernix.elements.magnesium import Magnesium, is_protected_name
from hypernix.system import errorcatalogue as codes
from hypernix.system.errorcodes import HyperNixError


def registry(tmp_path, **kwargs) -> Registry:
    made = Registry(data_root=tmp_path / "data", **kwargs)
    for cls in BUILTINS:
        made.register(cls)
    return made


def raises(code):
    """`pytest.raises` that also checks *which* code — the point of codes."""
    class _Ctx:
        def __enter__(self):
            self._inner = pytest.raises(HyperNixError)
            self.info = self._inner.__enter__()
            return self.info

        def __exit__(self, *exc):
            result = self._inner.__exit__(*exc)
            assert self.info.value.code is code, self.info.value.code
            return result
    return _Ctx()


# ---------------------------------------------------------------------------
# The periodic table
# ---------------------------------------------------------------------------


class TestThePeriodicTable:
    def test_all_118(self):
        assert len(SYMBOLS) - 1 == 118
        assert len(set(SYMBOLS[1:])) == 118

    @pytest.mark.parametrize("symbol,number", [
        ("H", 1), ("C", 6), ("Mg", 12), ("Mn", 25), ("Fe", 26), ("Cs", 55),
        ("Au", 79), ("Rn", 86), ("Fr", 87), ("Og", 118),
    ])
    def test_symbols(self, symbol, number):
        assert atomic_number(symbol) == number

    def test_case_matters(self):
        """`Co` is cobalt; `CO` is carbon monoxide and not an element."""
        assert atomic_number("Co") == 27
        with raises(codes.ELEMENT_NOT_FOUND):
            atomic_number("CO")

    @pytest.mark.parametrize("number,period", [
        (1, 1), (2, 1), (3, 2), (10, 2), (11, 3), (18, 3), (19, 4),
        (36, 4), (37, 5), (54, 5), (55, 6), (86, 6), (87, 7), (118, 7),
    ])
    def test_periods_at_every_boundary(self, number, period):
        assert period_of(number) == period

    def test_rows_six_and_seven_are_experimental(self):
        """Decided by atomic number, so an element added there later is
        gated without anybody remembering to gate it."""
        assert [n for n in range(1, 119) if is_experimental(n)] == list(range(55, 119))

    def test_there_is_no_element_zero_or_119(self):
        for bad in (0, 119, -1):
            with raises(codes.ELEMENT_NOT_FOUND):
                period_of(bad)


# ---------------------------------------------------------------------------
# hydrogen: specs and the registry
# ---------------------------------------------------------------------------


class TestSpec:
    def test_the_symbol_must_be_real(self):
        with raises(codes.ELEMENT_NOT_FOUND):
            ElementSpec(symbol="Xx", summary="nope")

    def test_the_name_comes_from_the_symbol(self):
        """So Magnesium cannot be registered as Mn — manganese. A name
        that points somewhere other than its symbol sends whoever
        searches for it to the wrong element."""
        assert ElementSpec(symbol="Mg", summary="x").name == "magnesium"
        assert ElementSpec(symbol="Mn", summary="x").name != "magnesium"

    def test_an_unknown_permission_is_refused(self):
        """A typo in a permission list is how an element ends up quietly
        unable to do its job."""
        with raises(codes.ELEMENT_PERMISSION):
            ElementSpec(symbol="Na", summary="x", permissions=frozenset({"proceses"}))

    def test_to_dict(self):
        d = Magnesium.spec.to_dict()
        assert (d["symbol"], d["number"], d["period"], d["experimental"]) == ("Mg", 12, 3, False)


class TestRegistry:
    def test_the_builtins_register(self, tmp_path):
        assert [s.symbol for s in registry(tmp_path).specs()] == ["H", "C", "Mg"]

    def test_lookup_by_symbol_or_name(self, tmp_path):
        reg = registry(tmp_path)
        assert reg.lookup("Mg") is Magnesium
        assert reg.lookup("magnesium") is Magnesium
        assert reg.lookup("MAGNESIUM") is Magnesium

    def test_an_unknown_element_names_what_is_there(self, tmp_path):
        with raises(codes.ELEMENT_NOT_FOUND) as caught:
            registry(tmp_path).lookup("Zn")
        assert "Mg" in str(caught.value)

    def test_a_class_without_a_spec_is_refused(self, tmp_path):
        class Bare(Element):
            pass
        with raises(codes.ELEMENT_LOAD_FAILED):
            registry(tmp_path).register(Bare)

    def test_the_wrong_generation_is_refused_at_registration(self, tmp_path):
        """At registration, not at first call — a mismatch found on the
        first generation is found mid-answer."""
        class Old(Element):
            spec = ElementSpec(symbol="Li", summary="x", generation=GENERATION + 1)
        with raises(codes.ELEMENT_API_MISMATCH):
            registry(tmp_path).register(Old)

    def test_a_user_element_cannot_take_a_builtin_symbol(self, tmp_path):
        class Impostor(Element):
            spec = ElementSpec(symbol="Mg", summary="x", user=True)
        with raises(codes.ELEMENT_CONFLICT):
            registry(tmp_path).register(Impostor)

    def test_two_builtins_with_one_symbol_is_a_bug(self, tmp_path):
        class Second(Element):
            spec = ElementSpec(symbol="C", summary="x")
        with raises(codes.ELEMENT_CONFLICT):
            registry(tmp_path).register(Second)

    def test_registering_the_same_class_twice_is_fine(self, tmp_path):
        reg = registry(tmp_path)
        assert reg.register(Carbon) is Carbon

    def test_an_experimental_element_needs_permission(self, tmp_path):
        class Gold(Element):
            spec = ElementSpec(symbol="Au", summary="x")
        reg = registry(tmp_path)
        reg.register(Gold)
        with raises(codes.ELEMENT_EXPERIMENTAL):
            reg.activate("Au")

    def test_and_runs_when_allowed(self, tmp_path):
        class Gold(Element):
            spec = ElementSpec(symbol="Au", summary="x")
        reg = registry(tmp_path, allow_experimental=True)
        reg.register(Gold)
        assert reg.activate("Au").active

    def test_activation_is_idempotent(self, tmp_path):
        calls = []

        class Counting(Element):
            spec = ElementSpec(symbol="Na", summary="x")

            def activate(self):
                calls.append(1)
        reg = registry(tmp_path)
        reg.register(Counting)
        reg.activate("Na")
        reg.activate("Na")
        assert calls == [1]

    def test_a_failed_deactivate_still_marks_it_stopped(self, tmp_path):
        """Otherwise it is reported as running and nothing tries again."""
        class Stubborn(Element):
            spec = ElementSpec(symbol="K", summary="x")

            def deactivate(self):
                raise RuntimeError("no")
        reg = registry(tmp_path)
        reg.register(Stubborn)
        reg.activate("K")
        with pytest.raises(RuntimeError):
            reg.deactivate("K")
        assert reg.active() == []

    def test_deactivate_all_keeps_going_past_a_failure(self, tmp_path):
        order = []

        def make(symbol, fails=False):
            class E(Element):
                spec = ElementSpec(symbol=symbol, summary="x")

                def deactivate(self):
                    order.append(symbol)
                    if fails:
                        raise RuntimeError("no")
            return E
        reg = registry(tmp_path)
        for sym, fails in (("Li", False), ("Be", True), ("B", False)):
            reg.register(make(sym, fails))
            reg.activate(sym)
        reg.deactivate_all()
        # Heaviest first — the reverse of the order they build on each other.
        assert order == ["B", "Be", "Li"]

    def test_permissions_are_checked_not_declared(self, tmp_path):
        ctx = registry(tmp_path).instance("C").context
        ctx.require("oven")
        with raises(codes.ELEMENT_PERMISSION):
            ctx.require("shell")


# ---------------------------------------------------------------------------
# natural gas: elements on an oven
# ---------------------------------------------------------------------------


class FakeOven:
    """The two methods natural gas wraps, and a record of what they saw."""

    def __init__(self):
        self.seen = []

    def complete(self, prompt, **kwargs):
        self.seen.append(prompt)
        return f"out[{prompt}]"

    def chat(self, messages, **kwargs):
        self.seen.append(messages)
        return "reply  \n\n\n\nend"


def element(symbol, *, prompt=None, output=None, fails=None,
            permissions=frozenset({"oven"})):
    class E(Element):
        spec = ElementSpec(symbol=symbol, summary="x", permissions=permissions)

        def transform_prompt(self, text):
            if fails == "prompt":
                raise RuntimeError("boom")
            return prompt(text) if prompt else text

        def transform_output(self, text):
            if fails == "output":
                raise RuntimeError("boom")
            return output(text) if output else text
    return E


class TestNaturalGas:
    def test_hooks_run_around_the_generation(self, tmp_path):
        reg = registry(tmp_path)
        reg.register(element("Li", prompt=lambda t: t + "+Li",
                             output=lambda t: t + "-Li"))
        oven = FakeOven()
        natural_gas.attach(oven, ["Li"], registry=reg)
        assert oven.complete("p") == "out[p+Li]-Li"
        assert oven.seen == ["p+Li"]

    def test_order_is_by_atomic_number_in_and_reverse_out(self, tmp_path):
        """So an element wraps the lighter ones the way a function wraps
        what it calls, and the result is the same every run."""
        reg = registry(tmp_path)
        reg.register(element("B", prompt=lambda t: t + "B", output=lambda t: t + "B"))
        reg.register(element("Li", prompt=lambda t: t + "Li", output=lambda t: t + "Li"))
        oven = FakeOven()
        natural_gas.attach(oven, ["B", "Li"], registry=reg)  # given heavy-first
        assert oven.complete("") == "out[LiB]BLi"

    def test_it_is_per_instance(self, tmp_path):
        """Patching the class would put the addon on every oven in the
        process, including one loaded for somebody else's job."""
        reg = registry(tmp_path)
        reg.register(element("Li", output=lambda t: "CHANGED"))
        mine, theirs = FakeOven(), FakeOven()
        natural_gas.attach(mine, ["Li"], registry=reg)
        assert mine.complete("x") == "CHANGED"
        assert theirs.complete("x") == "out[x]"

    def test_detach_restores_the_class_method(self, tmp_path):
        """By deleting the instance attribute, not reassigning a saved
        bound method — which would leave a shadow that a later patch to
        the class never reaches."""
        reg = registry(tmp_path)
        reg.register(element("Li", output=lambda t: "CHANGED"))
        oven = FakeOven()
        natural_gas.attach(oven, ["Li"], registry=reg)
        assert natural_gas.detach(oven) == ["Li"]
        assert "complete" not in oven.__dict__
        assert oven.complete("x") == "out[x]"

    def test_attaching_twice_replaces_rather_than_stacks(self, tmp_path):
        """Stacking is how an addon ends up running twice per call with
        nothing on screen to say so."""
        reg = registry(tmp_path)
        reg.register(element("Li", output=lambda t: t + "!"))
        oven = FakeOven()
        natural_gas.attach(oven, ["Li"], registry=reg)
        natural_gas.attach(oven, ["Li"], registry=reg)
        assert oven.complete("x") == "out[x]!"

    def test_a_failing_addon_does_not_stop_the_oven(self, tmp_path):
        reg = registry(tmp_path)
        reg.register(element("Li", fails="prompt"))
        reg.register(element("Be", output=lambda t: t + "Be"))
        oven = FakeOven()
        attachment = natural_gas.attach(oven, ["Li", "Be"], registry=reg)
        assert oven.complete("x") == "out[x]Be"
        assert attachment.failures == {"Li.transform_prompt": 1}

    def test_an_element_without_the_oven_permission_cannot_attach(self, tmp_path):
        reg = registry(tmp_path)
        reg.register(element("Li", permissions=frozenset()))
        with raises(codes.ELEMENT_PERMISSION):
            natural_gas.attach(FakeOven(), ["Li"], registry=reg)

    def test_magnesium_is_not_an_oven_addon(self, tmp_path):
        """It asks for processes, not the oven — so it cannot be wired
        into a generation by accident."""
        with raises(codes.ELEMENT_PERMISSION):
            natural_gas.attach(FakeOven(), ["Mg"], registry=registry(tmp_path))

    def test_a_refused_element_is_never_activated(self, tmp_path):
        """The check comes first. Attaching magnesium used to activate it —
        renicing every process on the machine — and only then refuse."""
        started = []

        class Watched(Element):
            spec = ElementSpec(symbol="Li", summary="x", permissions=frozenset({"processes"}))

            def activate(self):
                started.append(self.spec.symbol)

        reg = registry(tmp_path)
        reg.register(Watched)
        with raises(codes.ELEMENT_PERMISSION):
            natural_gas.attach(FakeOven(), ["Li"], registry=reg)
        assert started == []
        assert not reg.instance("Li").active

    def test_attach_is_all_or_nothing(self, tmp_path):
        """One element failing to start stops the ones already started."""
        log = []

        def make(symbol, fail):
            class E(Element):
                spec = ElementSpec(symbol=symbol, summary="x", permissions=frozenset({"oven"}))

                def activate(self):
                    if fail:
                        raise RuntimeError("no")
                    log.append(("on", symbol))

                def deactivate(self):
                    log.append(("off", symbol))
            return E

        reg = registry(tmp_path)
        reg.register(make("Li", fail=False))
        reg.register(make("Be", fail=True))
        oven = FakeOven()
        with pytest.raises(RuntimeError):
            natural_gas.attach(oven, ["Be", "Li"], registry=reg)
        assert log == [("on", "Li"), ("off", "Li")]
        assert not reg.instance("Li").active
        assert natural_gas.attached(oven) is None
        assert oven.complete("x") == "out[x]"

    def test_chat_messages_are_copied_before_elements_see_them(self, tmp_path):
        """An element that edits in place must not reach back into the
        caller's conversation history."""
        class Mutating(Element):
            spec = ElementSpec(symbol="Li", summary="x", permissions=frozenset({"oven"}))

            def transform_messages(self, messages):
                messages[0]["content"] = "EDITED"
                return messages
        reg = registry(tmp_path)
        reg.register(Mutating)
        history = [{"role": "user", "content": "original"}]
        oven = FakeOven()
        natural_gas.attach(oven, ["Li"], registry=reg)
        oven.chat(history)
        assert history[0]["content"] == "original"
        assert oven.seen[0][0]["content"] == "EDITED"

    def test_carbon_tidies_a_real_chat_reply(self, tmp_path):
        oven = FakeOven()
        natural_gas.attach(oven, ["C"], registry=registry(tmp_path))
        assert oven.chat([{"role": "user", "content": "hi"}]) == "reply\n\nend"


# ---------------------------------------------------------------------------
# carbon
# ---------------------------------------------------------------------------


class TestTidy:
    def test_trailing_whitespace_and_blank_runs(self):
        assert tidy("a   \n\n\n\nb  ") == "a\n\nb"

    def test_crlf(self):
        assert tidy("a\r\nb\rc") == "a\nb\nc"

    def test_the_inside_of_a_code_fence_is_untouched(self):
        """Whitespace there can be the program: a Makefile's tabs, a
        Python string's trailing spaces, blank lines inside a heredoc."""
        body = "```make\nall:\n\techo hi   \n\n\n\n\techo bye\n```"
        assert tidy(body) == body

    def test_text_after_a_fence_is_tidied_again(self):
        assert tidy("```\nx  \n```\n\n\n\nafter   ") == "```\nx  \n```\n\nafter"

    def test_tilde_fences_count(self):
        assert tidy("~~~\nx   \n~~~") == "~~~\nx   \n~~~"

    def test_an_unclosed_fence_protects_the_rest(self):
        """A model cut off mid-code block: everything after the opening
        fence is code, so none of it is touched."""
        assert tidy("text  \n```\ncode   \n\n\n") == "text\n```\ncode   \n\n\n"


class TestSnippets:
    def test_expansion(self):
        assert expand_snippets("do ::fmt", {"fmt": "format it"}) == "do format it"

    def test_cpp_and_rust_paths_are_left_alone(self):
        """A prompt about `std::vector` must not lose half of itself."""
        text = "use std::vector and crate::io"
        assert expand_snippets(text, {"vector": "X", "io": "Y"}) == text

    def test_an_unknown_snippet_is_left_as_typed(self):
        assert expand_snippets("::nope", {"fmt": "x"}) == "::nope"

    def test_carbon_expands_only_user_messages(self, tmp_path):
        reg = registry(tmp_path)
        c = reg.instance("C", config={"snippets": {"s": "SNIP"}})
        out = c.transform_messages([
            {"role": "system", "content": "::s"},
            {"role": "user", "content": "::s"},
        ])
        assert [m["content"] for m in out] == ["::s", "SNIP"]


class TestUserElements:
    def test_scaffold_writes_a_loadable_element(self, tmp_path):
        target = carbon.scaffold("Na", tmp_path)
        assert target.name == "sodium.py"
        reg = registry(tmp_path)
        assert carbon.load_user_elements(reg, tmp_path) == ["Na"]
        assert reg.lookup("sodium").spec.user

    def test_scaffold_refuses_a_builtin_symbol(self, tmp_path):
        with raises(codes.ELEMENT_CONFLICT):
            carbon.scaffold("Mg", tmp_path, registry=registry(tmp_path))

    def test_scaffold_never_overwrites(self, tmp_path):
        carbon.scaffold("Na", tmp_path)
        with raises(codes.ELEMENT_CONFLICT):
            carbon.scaffold("Na", tmp_path)

    def test_a_broken_file_is_recorded_and_the_rest_load(self, tmp_path):
        carbon.scaffold("Na", tmp_path)
        (tmp_path / "broken.py").write_text("this is not python(")
        reg = registry(tmp_path)
        assert carbon.load_user_elements(reg, tmp_path) == ["Na"]
        assert "E2-00020.d3" in reg.failures["broken"]

    def test_a_file_claiming_to_be_builtin_is_forced_to_user(self, tmp_path):
        """What stops a file from displacing HyperNix's own code."""
        (tmp_path / "sneaky.py").write_text(
            "from hypernix.elements.hydrogen import Element, ElementSpec\n"
            "class Sneaky(Element):\n"
            "    spec = ElementSpec(symbol='Mg', summary='x', user=False)\n"
        )
        reg = registry(tmp_path)
        assert carbon.load_user_elements(reg, tmp_path) == []
        assert reg.lookup("Mg") is Magnesium
        assert "Mg" in reg.failures

    def test_a_user_file_is_always_marked_user(self, tmp_path):
        """With a symbol nobody holds, so the built-in collision is not
        what refuses it. Left unmarked, it would count as a built-in, and
        the next user file could not be told apart from HyperNix's own."""
        (tmp_path / "claims.py").write_text(
            "from hypernix.elements.hydrogen import Element, ElementSpec\n"
            "class Claims(Element):\n"
            "    spec = ElementSpec(symbol='Na', summary='x', user=False)\n"
        )
        reg = registry(tmp_path)
        assert carbon.load_user_elements(reg, tmp_path) == ["Na"]
        assert reg.lookup("Na").spec.user is True

    def test_importing_a_builtin_does_not_rewrite_it(self, tmp_path):
        """The loader walks the file's namespace. A built-in imported
        into it is in that namespace too, and forcing *its* spec to
        `user=True` would corrupt it for the whole process."""
        (tmp_path / "subclass.py").write_text(
            "from hypernix.elements.carbon import Carbon\n"
            "from hypernix.elements.hydrogen import ElementSpec\n"
            "class Diamond(Carbon):\n"
            "    spec = ElementSpec(symbol='Si', summary='x', user=True,\n"
            "                       permissions=frozenset({'oven'}))\n"
        )
        reg = registry(tmp_path)
        assert carbon.load_user_elements(reg, tmp_path) == ["Si"]
        assert Carbon.spec.user is False

    def test_files_are_executed_not_imported(self, tmp_path):
        """So a user file called `json.py` cannot shadow the standard
        library for the rest of the process."""
        carbon.scaffold("Na", tmp_path)
        (tmp_path / "sodium.py").rename(tmp_path / "json.py")
        carbon.load_user_elements(registry(tmp_path), tmp_path)
        import json
        assert hasattr(json, "dumps")

    def test_a_missing_directory_is_nothing_loaded(self, tmp_path):
        assert carbon.load_user_elements(registry(tmp_path), tmp_path / "no") == []


# ---------------------------------------------------------------------------
# magnesium
# ---------------------------------------------------------------------------


class FakeProc:
    def __init__(self, pid, name, username="me", cmdline=("x",), ppid=100):
        self.pid = pid
        self.info = {"pid": pid, "name": name, "username": username,
                     "cmdline": list(cmdline), "ppid": ppid}


#: The plan reads this process's family through psutil even when handed
#: fake processes. CI installs it (the dev extra); a bare checkout may not.
def _have_psutil() -> bool:
    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


needs_psutil = pytest.mark.skipif(not _have_psutil(), reason="magnesium needs psutil (pip install 'hypernix[elements]')")


class TestMagnesiumPlan:
    @pytest.mark.parametrize("name", [
        "bash", "zsh", "fish", "tmux", "kitty", "alacritty",
        "gnome-terminal-server", "WindowsTerminal.exe", "pwsh.exe", "cmd.exe",
        "python", "python3", "python3.12", "pypy3", "hypernix-t1",
        "llama-server", "gnome-shell", "Xorg", "pipewire-pulse", "systemd-journald",
    ])
    def test_the_never_touch_list(self, name):
        assert is_protected_name(name)

    def test_without_psutil_it_says_what_to_install(self, tmp_path, monkeypatch):
        """Not "the OS refused": nothing refused, a package is missing."""
        import builtins

        real_import = builtins.__import__

        def no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_psutil)
        with raises(codes.PACKAGE_MISSING) as caught:
            registry(tmp_path).instance("Mg").plan(processes=[])
        assert "hypernix[elements]" in str(caught.value)

    @pytest.mark.parametrize("name", ["firefox", "chrome", "Discord", "steam", "code"])
    def test_ordinary_apps_are_not_protected(self, name):
        assert not is_protected_name(name)

    def test_a_nameless_process_is_protected(self):
        assert is_protected_name("")

    @needs_psutil
    def test_the_plan_decides_each_one(self, tmp_path):
        mg = registry(tmp_path).instance("Mg")
        plan = mg.plan(processes=[
            FakeProc(1, "systemd"),
            FakeProc(5000, "firefox"),
            FakeProc(5001, "bash"),
            FakeProc(5002, "chrome", username="someone-else"),
        ])
        decisions = {t.name: t.decision for t in plan.targets}
        assert decisions["systemd"] == "protected"
        assert decisions["bash"] == "protected"
        assert decisions["firefox"] in ("limit", "not-ours")

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux signal")
    @needs_psutil
    def test_kernel_threads_are_protected_by_command_line(self, tmp_path):
        """Found by running the plan against a real machine: kworker,
        ksoftirqd and friends came out as "limit". There are hundreds of
        names and they change between kernels; the empty command line is
        the signal, and owner is no help when magnesium runs as root."""
        mg = registry(tmp_path).instance("Mg")
        plan = mg.plan(processes=[
            FakeProc(7000, "kworker/3:1H", cmdline=(), ppid=2),
            FakeProc(7001, "ksoftirqd/0", cmdline=(), ppid=2),
            FakeProc(7002, "irq/24-ACPI:Ged", cmdline=(), ppid=2),
        ])
        assert {t.decision for t in plan.targets} == {"protected"}

    @needs_psutil
    def test_this_process_and_its_parents_are_protected(self, tmp_path):
        mg = registry(tmp_path).instance("Mg")
        plan = mg.plan(processes=[FakeProc(os.getpid(), "firefox"),
                                  FakeProc(os.getppid(), "firefox")])
        assert {t.decision for t in plan.targets} == {"protected"}

    @needs_psutil
    def test_an_extra_protect_list_from_config(self, tmp_path):
        mg = registry(tmp_path).instance("Mg", config={"protect": ["OBS"]})
        plan = mg.plan(processes=[FakeProc(5000, "obs")])
        assert plan.targets[0].decision == "protected"

    @needs_psutil
    def test_the_plan_changes_nothing(self, tmp_path):
        mg = registry(tmp_path).instance("Mg")
        mg.plan()
        assert mg.changed == {}

    def test_root_can_always_restore(self, monkeypatch):
        from hypernix.elements import magnesium

        monkeypatch.setattr(magnesium.sys, "platform", "linux")
        monkeypatch.setattr(magnesium.os, "geteuid", lambda: 0, raising=False)
        assert magnesium.lowest_restorable_nice() is None
        assert magnesium.can_restore(-20)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_NICE is Linux")
    def test_linux_follows_rlimit_nice(self, monkeypatch):
        import resource

        from hypernix.elements import magnesium

        monkeypatch.setattr(magnesium.os, "geteuid", lambda: 1000)
        monkeypatch.setattr(resource, "getrlimit", lambda which: (0, 0))
        assert magnesium.lowest_restorable_nice() == 20
        assert not magnesium.can_restore(3)
        monkeypatch.setattr(resource, "getrlimit", lambda which: (20, 20))
        assert magnesium.lowest_restorable_nice() == 0
        assert magnesium.can_restore(3) and not magnesium.can_restore(-1)

    def test_other_unixes_need_root(self, monkeypatch):
        from hypernix.elements import magnesium

        monkeypatch.setattr(magnesium.sys, "platform", "darwin")
        monkeypatch.setattr(magnesium.os, "geteuid", lambda: 501, raising=False)
        assert not magnesium.can_restore(0)

    def test_it_needs_the_processes_permission(self):
        assert "processes" in Magnesium.spec.permissions


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX niceness")
class TestMagnesiumForReal:
    """Against a real process outside this one's tree — a direct child
    would be protected as family, which is the rule working."""

    @pytest.fixture
    def orphan(self):
        pytest.importorskip("psutil")
        out = subprocess.run(
            # Started at nice 3, not 0: restoring to a hardcoded 0 would
            # be indistinguishable from restoring the original otherwise.
            ["sh", "-c", "nice -n 3 sleep 60 >/dev/null 2>&1 & echo $!"],
            capture_output=True, text=True, check=True,
        )
        pid = int(out.stdout.strip())
        time.sleep(0.1)
        yield pid
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass

    def test_it_lowers_priority_and_puts_it_back_exactly(self, tmp_path, orphan):
        import psutil

        before = psutil.Process(orphan).nice()
        # Relative to wherever this runs: CI and containers often start
        # already nice'd, and `nice -n 3` adds to that.
        assert before == min(19, os.nice(0) + 3)
        target = before + 1
        if target > 15:
            pytest.skip(f"already at nice {before}; nothing lower to set")
        mg = registry(tmp_path).instance("Mg", config={"nice": target})
        mg.activate()
        if orphan not in mg.changed:
            # Unprivileged, lowering it back would be impossible, so it is
            # rightly left alone; the tests above cover that refusal.
            pytest.skip(f"not limitable here: "
                        f"{[t.decision for t in mg.last_plan.targets if t.pid == orphan]}")
        assert psutil.Process(orphan).nice() == target
        mg.deactivate()
        assert psutil.Process(orphan).nice() == before
        assert mg.changed == {}

    def test_it_will_not_make_a_change_it_cannot_undo(self, tmp_path, orphan, monkeypatch):
        """Without root, a raised niceness usually cannot be lowered again —
        so limiting would break the "put back exactly" promise. Found when
        CI first ran this as an ordinary user."""
        import psutil

        from hypernix.elements import magnesium

        monkeypatch.setattr(magnesium, "lowest_restorable_nice", lambda: 20)
        before = psutil.Process(orphan).nice()
        if before >= 15:
            pytest.skip(f"already at nice {before}")
        mg = registry(tmp_path).instance("Mg", config={"nice": 15})
        mg.activate()
        assert orphan not in mg.changed
        assert psutil.Process(orphan).nice() == before
        [target] = [t for t in mg.last_plan.targets if t.pid == orphan]
        assert target.decision == "irreversible"
        assert "allow_irreversible" in target.reason

    def test_allow_irreversible_is_an_explicit_choice(self, tmp_path, orphan, monkeypatch):
        import psutil

        from hypernix.elements import magnesium

        monkeypatch.setattr(magnesium, "lowest_restorable_nice", lambda: 20)
        before = psutil.Process(orphan).nice()
        if before >= 15:
            pytest.skip(f"already at nice {before}")
        mg = registry(tmp_path).instance("Mg", config={"nice": 15, "allow_irreversible": True})
        mg.activate()
        try:
            assert orphan in mg.changed
            assert psutil.Process(orphan).nice() == 15
        finally:
            mg.deactivate()

    def test_a_process_that_exits_meanwhile_is_not_an_error(self, tmp_path, orphan):
        mg = registry(tmp_path).instance("Mg")
        mg.activate()
        os.kill(orphan, 9)
        time.sleep(0.1)
        mg.deactivate()           # must not raise

    def test_deactivate_without_activate_is_safe(self, tmp_path):
        registry(tmp_path).instance("Mg").deactivate()


def test_hydrogen_is_listed_as_element_one():
    assert Hydrogen.spec.number == 1
