from __future__ import annotations

import json

import pytest

from hypernix.elements import carbon, natural_gas
from hypernix.elements.builtins import BUILTINS, register_all
from hypernix.elements.chat_elements import resolve_chat_elements
from hypernix.elements.hydrogen import Element, ElementSpec, Registry
from hypernix.elements.isotope_3h import ISOTOPE_3H, parse_hnx, scaffold_hnx
from hypernix.system import errorcatalogue as codes
from hypernix.system.errorcodes import HyperNixError

SCRIPT = """#!hnx 1.1.0
{"kind":"element","symbol":"Na","summary":"Prompt decorator","dependencies":["C"],"groups":["chat","prompt"],"permissions":["oven"]}
{"kind":"data","name":"prefix","value":"[Na]"}
fn transform_prompt(prompt: string) -> string {
    let cleaned: string = prompt.trim();
    if (cleaned.includes("!")) {
        return data.get("prefix") + " " + cleaned.replace("!", "?");
    }
    return data.get("prefix") + " " + cleaned;
}
"""


def make_registry(tmp_path, *, allow_experimental=False):
    registry = Registry(data_root=tmp_path / "data",
                        allow_experimental=allow_experimental)
    register_all(registry)
    return registry


def test_3h_is_a_runtime_isotope_not_a_119th_element():
    assert ISOTOPE_3H.symbol == "3H"
    assert ISOTOPE_3H.base_element == "H"
    assert ISOTOPE_3H.format_version == "1.1.0"
    assert len(BUILTINS) == 3


def test_hnx_is_editable_and_runs_the_safe_mixed_syntax(tmp_path):
    path = tmp_path / "sodium.hnx"
    path.write_text(SCRIPT, encoding="utf-8")
    spec, program = parse_hnx(path)
    assert spec.dependencies == ("C",)
    assert spec.groups == ("chat", "prompt")
    assert program.call("transform_prompt", "  hello!  ") == "[Na] hello?"

    registry = make_registry(tmp_path)
    assert carbon.load_user_elements(registry, tmp_path) == ["Na"]
    assert registry.instance("Na").transform_prompt("  hi  ") == "[Na] hi"


def test_cxx_and_typescript_function_signatures_are_supported(tmp_path):
    cxx = SCRIPT.replace(
        "fn transform_prompt(prompt: string) -> string {",
        "std::string transform_prompt(std::string prompt) {",
    )
    path = tmp_path / "cxx.hnx"
    path.write_text(cxx, encoding="utf-8")
    _spec, program = parse_hnx(path)
    assert program.call("transform_prompt", "hello") == "[Na] hello"

    ts = SCRIPT.replace(
        "fn transform_prompt(prompt: string) -> string {",
        "function transform_prompt(prompt: string): string {",
    )
    path.write_text(ts, encoding="utf-8")
    _spec, program = parse_hnx(path)
    assert program.call("transform_prompt", "hello") == "[Na] hello"


def test_hydrogen_activates_dependencies_before_the_requested_element(tmp_path):
    events = []

    class Lithium(Element):
        spec = ElementSpec(symbol="Li", summary="dependency")

        def activate(self):
            events.append("Li")

    class Sodium(Element):
        spec = ElementSpec(symbol="Na", summary="depends on lithium",
                           dependencies=("Li",))

        def activate(self):
            events.append("Na")

    registry = make_registry(tmp_path)
    registry.register(Lithium)
    registry.register(Sodium)
    assert registry.dependency_order(["Na"]) == ["Li", "Na"]
    assert registry.activate("Na").active
    assert events == ["Li", "Na"]
    assert [item.spec.symbol for item in registry.active()] == ["Li", "Na"]


def test_missing_dependency_is_rejected_before_activation(tmp_path):
    class Sodium(Element):
        spec = ElementSpec(symbol="Na", summary="missing", dependencies=("Li",))

    registry = make_registry(tmp_path)
    registry.register(Sodium)
    with pytest.raises(HyperNixError) as caught:
        registry.activate("Na")
    assert caught.value.code is codes.ELEMENT_NOT_FOUND
    assert not registry._live


def test_dependency_cycles_are_rejected(tmp_path):
    class Lithium(Element):
        spec = ElementSpec(symbol="Li", summary="cycle", dependencies=("Na",))

    class Sodium(Element):
        spec = ElementSpec(symbol="Na", summary="cycle", dependencies=("Li",))

    registry = make_registry(tmp_path)
    registry.register(Lithium)
    registry.register(Sodium)
    with pytest.raises(HyperNixError, match="dependency cycle"):
        registry.activate("Li")


def test_scaffold_and_group_selection_use_v110_metadata(tmp_path):
    path = scaffold_hnx("Na", tmp_path)
    assert path.suffix == ".hnx"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert record["dependencies"] == []
    registry, selected = resolve_chat_elements(str(tmp_path))
    assert selected == ["Na"]
    assert registry.lookup("Na").spec.format_version == "1.1.0"
    registry, selected = resolve_chat_elements(str(tmp_path), "CHAT")
    assert selected == ["Na"]
    assert registry.dependency_order(selected) == ["Na"]


def test_specific_file_selection_keeps_siblings_available_only_as_dependencies(tmp_path):
    first = tmp_path / "sodium.hnx"
    second = tmp_path / "lithium.hnx"
    first.write_text(SCRIPT, encoding="utf-8")
    second.write_text(SCRIPT.replace('"Na"', '"Li"'), encoding="utf-8")
    registry, selected = resolve_chat_elements(str(first))
    assert selected == ["Na"]
    assert registry.lookup("Li").spec.symbol == "Li"


def test_unsafe_hnx_code_is_rejected_not_executed(tmp_path):
    path = tmp_path / "bad.hnx"
    path.write_text(
        '#!hnx 1.1.0\n'
        '{"kind":"element","symbol":"Na","summary":"bad","dependencies":[],"permissions":["oven"]}\n'
        'import os\n'
        'fn transform_prompt(prompt: string) -> string {\n'
        ' return prompt;\n'
        '}\n',
        encoding="utf-8",
    )
    registry = make_registry(tmp_path)
    assert carbon.load_user_elements(registry, tmp_path) == []
    assert "bad" in registry.failures


def test_wrong_format_version_is_rejected(tmp_path):
    path = tmp_path / "old.hnx"
    path.write_text(
        '#!hnx 1.0.0\n'
        '{"kind":"element","symbol":"Na","summary":"old","dependencies":[]}\n'
        'fn transform_prompt(prompt: string) -> string {\n return prompt;\n}\n',
        encoding="utf-8",
    )
    registry = make_registry(tmp_path)
    carbon.load_user_elements(registry, tmp_path)
    assert "old" in registry.failures


def test_registry_group_lookup_is_case_insensitive(tmp_path):
    class Grouped(Element):
        spec = ElementSpec(symbol="Na", summary="x", groups=("Chat", "prompt"))

    registry = make_registry(tmp_path)
    registry.register(Grouped)
    assert registry.in_group("cHaT") == ["Na"]


def test_two_ovens_share_dependencies_until_the_last_detach(tmp_path):
    class FakeOven:
        def chat(self, messages, *args, **kwargs):
            return "reply"

    registry = make_registry(tmp_path)
    first, second = FakeOven(), FakeOven()
    natural_gas.attach(first, ["C"], registry=registry)
    natural_gas.attach(second, ["C"], registry=registry)
    assert registry.instance("C").active
    natural_gas.detach(first)
    assert registry.instance("C").active
    natural_gas.detach(second)
    assert not registry.instance("C").active
