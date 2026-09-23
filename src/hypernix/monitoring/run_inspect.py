"""hypernix.monitoring.run_inspect — what a training run is made of.

tvtop-pro shows how a machine is doing. tvtop-max also shows what the
run on it *is*: which script, which HyperNix modules and libraries it
pulls in, which model architecture it builds, which Pressure Cooker it
steps with, and what is wrong with it. This module works that out, and
it does so from the outside: it reads the script's source with :mod:`ast`
and the run's log, and never imports or executes either. A dashboard
that ran the thing it was watching would be one mistake away from
starting a second training run.

Every answer says where it came from (``source``), because "the model
is gemma3" read from a ``new_oven(arch="gemma3")`` call and read from a
checkpoint's ``config.json`` are claims of different strength.
"""
from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ImportInfo",
    "Finding",
    "ArchInfo",
    "CookerInfo",
    "ScriptReport",
    "script_from_command",
    "analyze_script",
    "findings_from_log",
    "arch_from_folder",
    "cooker_catalogue",
    "hypernix_module_info",
]

HYPERNIX_ROOT = Path(__file__).resolve().parents[1]

#: How far into a log findings are looked for: the tail, where the run is.
LOG_WINDOW_LINES = 2000


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass
class ImportInfo:
    module: str
    #: hypernix | library | local | stdlib | missing
    kind: str
    names: list[str] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    version: str = ""
    summary: str = ""
    #: What to use instead, when the module is deprecated.
    deprecated: str = ""


@dataclass
class Finding:
    #: error | warn | info
    level: str
    #: script | log
    source: str
    message: str
    line: int = 0
    count: int = 1


@dataclass
class ArchInfo:
    #: script | config.json | checkpoint | none
    source: str = "none"
    arch: str = ""
    #: A repo or path the script loads, for preheat()-style runs.
    repo: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    preset: dict[str, Any] = field(default_factory=dict)
    params: int | None = None
    line: int = 0
    note: str = ""


@dataclass
class CookerInfo:
    name: str
    generation: str = ""
    deprecated: str = ""
    summary: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    line: int = 0


@dataclass
class ScriptReport:
    path: str = ""
    error: str = ""
    imports: list[ImportInfo] = field(default_factory=list)
    arch: ArchInfo = field(default_factory=ArchInfo)
    cookers: list[CookerInfo] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Finding the script
# ---------------------------------------------------------------------------


#: Programs whose script comes after their own options.
_LAUNCHERS = {"torchrun", "accelerate", "deepspeed"}
_LAUNCHER_MODULES = {"torch.distributed.run", "torch.distributed.launch", "accelerate.commands.launch"}
#: Python options that take the next argument as their value.
_PYTHON_VALUE_OPTIONS = {"-W", "-X", "--check-hash-based-pycs"}
_PYTHON = re.compile(r"^(python|pypy)[\d.]*(\.exe)?$", re.IGNORECASE)


def script_from_command(command: str | list[str], cwd: str | Path | None = None) -> Path | None:
    """The ``.py`` file a command line runs, or ``None``.

    For ``python``, the script is the first argument that is not an
    option: anything after it belongs to the script. So ``python
    /usr/bin/tvtop-max -S train.py`` runs ``tvtop-max``, not
    ``train.py``. ``python -m pkg.mod`` is resolved to the module's file
    without importing it. ``torchrun``, ``accelerate launch``,
    ``deepspeed`` and ``python -m torch.distributed.run`` take their own
    options first, so for those it is the first ``.py`` argument. A
    relative path is taken from the process's working directory.
    """
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    base = Path(cwd) if cwd else Path.cwd()

    def found(arg: str) -> Path | None:
        if not arg.endswith(".py"):
            return None
        path = Path(arg)
        path = path if path.is_absolute() else base / path
        return path if path.is_file() else None

    def first_py(args: list[str]) -> Path | None:
        for arg in args:
            if not arg.startswith("-") and arg.endswith(".py"):
                return found(arg)
        return None

    for index, token in enumerate(argv):
        name = Path(token).name
        if name in _LAUNCHERS:
            return first_py(argv[index + 1:])
        if not _PYTHON.match(name):
            continue
        rest = argv[index + 1:]
        i = 0
        while i < len(rest):
            arg = rest[i]
            if arg == "-c":
                return None
            if arg == "-m":
                module = rest[i + 1] if i + 1 < len(rest) else ""
                if module in _LAUNCHER_MODULES:
                    return first_py(rest[i + 2:])
                try:
                    spec = importlib.util.find_spec(module) if module else None
                except (ImportError, ValueError):
                    spec = None
                if spec is not None and spec.origin and spec.origin.endswith(".py"):
                    return Path(spec.origin)
                return None
            if arg in _PYTHON_VALUE_OPTIONS:
                i += 2
                continue
            if arg.startswith("-"):
                i += 1
                continue
            return found(arg)
        return None
    # An executable script run directly: ./train.py
    return found(argv[0]) if argv else None


# ---------------------------------------------------------------------------
# HyperNix's own modules and optimizers, read from source
# ---------------------------------------------------------------------------


def _module_file(dotted: str) -> Path | None:
    """Where *dotted* (a ``hypernix.*`` name) lives in this install."""
    parts = dotted.split(".")[1:]
    base = HYPERNIX_ROOT.joinpath(*parts) if parts else HYPERNIX_ROOT
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    line = text.strip().splitlines()[0].strip()
    # "hypernix.x.y — what it is" -> "what it is"
    return re.sub(r"^[\w.]+\s+[—-]+\s+", "", line)


_MODULE_CACHE: dict[str, tuple[str, str]] = {}


def hypernix_module_info(dotted: str) -> tuple[str, str] | None:
    """``(summary, deprecated_instead)`` for a ``hypernix.*`` module, or
    ``None`` when there is no such file.

    Read from the file: importing a deprecated module is exactly what
    announces its deprecation, and the dashboard should not do that on
    the run's behalf.
    """
    if dotted in _MODULE_CACHE:
        return _MODULE_CACHE[dotted]
    path = _module_file(dotted)
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return ("", "")
    instead = ""
    for node in tree.body:
        call = node.value if isinstance(node, ast.Expr) else None
        if isinstance(call, ast.Call) and _call_name(call) == "deprecated_module":
            for keyword in call.keywords:
                if keyword.arg == "instead" and isinstance(keyword.value, ast.Constant):
                    instead = str(keyword.value.value)
            instead = instead or "a newer module"
    result = (_first_line(ast.get_docstring(tree)), instead)
    _MODULE_CACHE[dotted] = result
    return result


_GENERATION_OF_FILE = {
    "pressure_cooker": "v1",
    "pressure_cooker_v3": "v3",
    "pressure_cooker_v4": "v4",
    "pressure_cooker_v5": "v5",
    "pressure_cooker_v5s": "v5s",
    "pressure_cooker_v6": "v6",
    "pressure_cooker_v6v": "v6v",
}

_COOKERS: dict[str, tuple[str, str]] | None = None


def cooker_catalogue() -> dict[str, tuple[str, str]]:
    """Every Pressure Cooker class: ``name -> (generation, summary)``.

    From the optimizer sources, so no torch import: the dashboard may be
    watching a run from a machine, or a virtualenv, that has none.
    """
    global _COOKERS
    if _COOKERS is not None:
        return _COOKERS
    found: dict[str, tuple[str, str]] = {}
    folder = HYPERNIX_ROOT / "optimizers"
    for stem, generation in _GENERATION_OF_FILE.items():
        path = folder / f"{stem}.py"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and "Cooker" in node.name:
                found[node.name] = (generation, _first_line(ast.get_docstring(node)))
    found.setdefault("UniversalCooker", ("router", "picks a Pressure Cooker for the hardware"))
    _COOKERS = found
    return found


def _deprecated_generations() -> dict[str, str]:
    try:
        from hypernix.optimizers.deprecation import DEPRECATED_GENERATIONS
    except Exception:  # noqa: BLE001 - a missing optimizer package is not a crash here
        return {}
    return dict(DEPRECATED_GENERATIONS)


def _arch_presets() -> dict[str, dict[str, Any]]:
    """The oven's presets, when they can be had without surprises.

    Importing the oven imports torch, which takes a second or two and may
    not be installed; the architecture panel then shows the arch name and
    what the script passed, without the preset's defaults.
    """
    try:
        from hypernix.models.neo_oven import ARCH_PRESETS
    except Exception:  # noqa: BLE001
        return {}
    return {name: dict(values) for name, values in ARCH_PRESETS.items()}


# ---------------------------------------------------------------------------
# Reading a script
# ---------------------------------------------------------------------------


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ast.unparse(node) if hasattr(ast, "unparse") else "…"


def _distribution_version(top: str) -> str:
    try:
        distributions = importlib.metadata.packages_distributions().get(top) or [top]
    except Exception:  # noqa: BLE001
        distributions = [top]
    for name in distributions:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def _classify(module: str, script_dir: Path) -> ImportInfo:
    top = module.split(".")[0]
    if top == "hypernix":
        info = hypernix_module_info(module)
        if info is None:
            return ImportInfo(module=module, kind="missing",
                              summary="not a module in this HyperNix install")
        summary, deprecated = info
        # The running package's own version: an editable install's
        # metadata keeps whatever version it was installed at.
        from hypernix import __version__ as running

        return ImportInfo(module=module, kind="hypernix", summary=summary, deprecated=deprecated,
                          version=running)
    if top in sys.stdlib_module_names:
        return ImportInfo(module=module, kind="stdlib")
    if (script_dir / f"{top}.py").is_file() or (script_dir / top / "__init__.py").is_file():
        return ImportInfo(module=module, kind="local")
    version = _distribution_version(top)
    if version:
        return ImportInfo(module=module, kind="library", version=version)
    try:
        found = importlib.util.find_spec(top) is not None
    except (ImportError, ValueError):
        found = False
    return ImportInfo(module=module, kind="library" if found else "missing")


class _Reader(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: dict[str, tuple[set[str], set[int]]] = {}
        self.calls: list[ast.Call] = []
        self.names: list[ast.Name] = []

    def _add(self, module: str, name: str | None, line: int) -> None:
        names, lines = self.imports.setdefault(module, (set(), set()))
        if name:
            names.add(name)
        lines.add(line)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name, None, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module:
            return  # a relative import is the script's own package
        for alias in node.names:
            # `from hypernix import optimizers` is the module
            # hypernix.optimizers, not a name inside hypernix.
            sub = f"{node.module}.{alias.name}"
            if node.module.split(".")[0] == "hypernix" and _module_file(sub) is not None:
                self._add(sub, None, node.lineno)
            else:
                self._add(node.module, alias.name, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.names.append(node)


_ARCH_FIELDS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "dtype")


def _arch_from_calls(calls: list[ast.Call]) -> ArchInfo:
    for call in calls:
        name = _call_name(call)
        if name in ("new_oven", "new_brewed"):
            info = ArchInfo(source="script", arch="hypernix", line=call.lineno)
            for keyword in call.keywords:
                if keyword.arg == "arch":
                    info.arch = str(_literal(keyword.value))
                elif keyword.arg in _ARCH_FIELDS:
                    info.fields[keyword.arg] = _literal(keyword.value)
            info.preset = _arch_presets().get(info.arch, {})
            info.params = _estimate_params(info.fields)
            return info
        if name in ("preheat", "preheat_brewed", "from_pretrained"):
            repo = ""
            if call.args:
                repo = str(_literal(call.args[0]))
            for keyword in call.keywords:
                if keyword.arg in ("repo_id", "local_dir", "path"):
                    repo = str(_literal(keyword.value))
            return ArchInfo(source="script", repo=repo or "ray0rf1re/hyper-nix.1", line=call.lineno,
                            note="loaded from a snapshot: its config.json has the architecture")
    return ArchInfo()


def _estimate_params(fields: dict[str, Any]) -> int | None:
    """The dense-transformer count ``hnx map`` uses, from literal sizes."""
    try:
        vocab = int(fields.get("vocab_size", 32000))
        hidden = int(fields.get("hidden_size", 1024))
        inter = int(fields.get("intermediate_size", hidden * 4))
        layers = int(fields.get("num_hidden_layers", 16))
    except (TypeError, ValueError):
        return None
    per_layer = 4 * hidden * hidden + 3 * hidden * inter + 2 * hidden
    return vocab * hidden * 2 + layers * per_layer + hidden


def _cookers_from(calls: list[ast.Call], names: list[ast.Name]) -> list[CookerInfo]:
    catalogue = cooker_catalogue()
    deprecated = _deprecated_generations()
    found: dict[str, CookerInfo] = {}
    for call in calls:
        name = _call_name(call)
        if name in catalogue:
            generation, summary = catalogue[name]
            kwargs = {k.arg: _literal(k.value) for k in call.keywords if k.arg}
            found.setdefault(name, CookerInfo(
                name=name, generation=generation, summary=summary, kwargs=kwargs,
                deprecated=deprecated.get(generation, ""), line=call.lineno))
    # `optimizer_class=PressureCookerV5` names the class without calling it.
    for node in names:
        if node.id in catalogue and node.id not in found:
            generation, summary = catalogue[node.id]
            found[node.id] = CookerInfo(name=node.id, generation=generation, summary=summary,
                                        deprecated=deprecated.get(generation, ""), line=node.lineno)
    return sorted(found.values(), key=lambda c: c.line)


def _script_findings(tree: ast.Module, reader: _Reader, report: ScriptReport) -> list[Finding]:
    findings: list[Finding] = []
    for item in report.imports:
        line = item.lines[0] if item.lines else 0
        if item.kind == "missing":
            findings.append(Finding("error", "script", f"imports {item.module}, which is not "
                                    "installed in this environment", line))
        if item.deprecated:
            findings.append(Finding("warn", "script", f"{item.module} is deprecated: use "
                                    f"{item.deprecated}", line))
    for cooker in report.cookers:
        if cooker.deprecated:
            findings.append(Finding("warn", "script", f"{cooker.name} is Pressure Cooker "
                                    f"{cooker.generation.upper()}, which is deprecated: use "
                                    f"{cooker.deprecated}", cooker.line))
    call_names = {_call_name(call) for call in reader.calls}
    for call in reader.calls:
        if _call_name(call) == "load" and isinstance(call.func, ast.Attribute) \
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "torch":
            safe = any(k.arg == "weights_only" and _literal(k.value) is True for k in call.keywords)
            if not safe:
                findings.append(Finding("warn", "script", "torch.load without weights_only=True "
                                        "can run code hidden in the file", call.lineno))
    seeded = bool(call_names & {"manual_seed", "seed", "set_seed", "seed_everything"}) or any(
        k.arg == "seed" for call in reader.calls for k in call.keywords)
    if not seeded:
        findings.append(Finding("info", "script", "no seed is set, so two runs of this script "
                                "will not match"))
    saves = any("save" in name.lower() or "checkpoint" in name.lower() for name in call_names)
    trains = "train" in call_names or any("step" == n for n in call_names)
    if trains and not saves:
        findings.append(Finding("warn", "script", "nothing in the script saves a checkpoint, so "
                                "a crash loses the whole run"))
    return findings


def analyze_script(path: str | Path) -> ScriptReport:
    """Everything tvtop-max can tell about a training script, from its source."""
    path = Path(path)
    report = ScriptReport(path=str(path))
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.error = f"could not read {path}: {exc.strerror or exc}"
        return report
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        report.error = f"syntax error on line {exc.lineno}: {exc.msg}"
        report.findings.append(Finding("error", "script", f"the script does not parse: {exc.msg}",
                                       exc.lineno or 0))
        return report
    reader = _Reader()
    reader.visit(tree)
    for module, (names, lines) in sorted(reader.imports.items()):
        info = _classify(module, path.parent)
        info.names = sorted(names)
        info.lines = sorted(lines)
        report.imports.append(info)
    report.arch = _arch_from_calls(reader.calls)
    report.cookers = _cookers_from(reader.calls, reader.names)
    report.findings = _script_findings(tree, reader, report)
    return report


# ---------------------------------------------------------------------------
# The model, from what the run wrote
# ---------------------------------------------------------------------------


def arch_from_folder(folder: str | Path) -> ArchInfo:
    """The architecture from a ``config.json`` in *folder*, or nothing."""
    config = Path(folder) / "config.json"
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ArchInfo()
    if not isinstance(data, dict):
        return ArchInfo()
    fields = {key: data[key] for key in _ARCH_FIELDS if key in data}
    return ArchInfo(source="config.json", arch=str(data.get("model_type") or ""),
                    repo=str(folder), fields=fields, params=_estimate_params(fields))


# ---------------------------------------------------------------------------
# Reading a log
# ---------------------------------------------------------------------------

_LOG_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("error", re.compile(r"Traceback \(most recent call last\)"), "the run raised an exception"),
    ("error", re.compile(r"(CUDA out of memory|OutOfMemoryError)", re.I), "out of GPU memory"),
    ("error", re.compile(r"\bloss\s*[=:]\s*(nan|inf|-inf)\b", re.I), "the loss is {0}"),
    ("error", re.compile(r"^\s*Killed\s*$"), "the process was killed (often the OOM killer)"),
    ("error", re.compile(r"^(\w*(?:Error|Exception)): (.+)$"), "{0}: {1}"),
    ("warn", re.compile(r"\b(\w*Warning): (.+)$"), "{0}: {1}"),
    ("warn", re.compile(r"^\s*(?:WARNING|WARN|warning)[:\s]+(.+)$"), "{0}"),
)


def findings_from_log(lines: list[str]) -> list[Finding]:
    """Errors and warnings in a log's tail, each counted once per message.

    A warning printed every step is one warning seen four thousand times,
    and showing it four thousand times would push everything else off
    the panel.
    """
    seen: dict[tuple[str, str], Finding] = {}
    tail = lines[-LOG_WINDOW_LINES:]
    for number, raw in enumerate(tail, start=max(1, len(lines) - len(tail) + 1)):
        line = raw.rstrip()
        for level, pattern, template in _LOG_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            message = template.format(*[g.strip() for g in match.groups()]) if match.groups() else template
            message = message[:200]
            key = (level, message)
            if key in seen:
                seen[key].count += 1
                seen[key].line = number
            else:
                seen[key] = Finding(level, "log", message, number)
            break
    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(seen.values(), key=lambda f: (order.get(f.level, 3), -f.line))
