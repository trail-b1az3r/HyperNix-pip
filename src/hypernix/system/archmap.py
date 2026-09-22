"""hypernix.system.archmap — the architecture chart, generated from the tree.

A hand-drawn architecture diagram is wrong within a month and nobody
notices, because the only thing that would notice is somebody reading
it to learn the system — which is exactly the person least able to spot
that it is out of date. So this reads the repository instead: imports
become edges, packages become nodes, and the beta surface is whatever
is actually marked beta rather than whatever somebody remembered.

Two charts, because "what exists" and "what is landing" are different
questions
--------------------------------------------------------------------
The main chart is the shipped system: major dependencies, the modules,
and what talks to what. The second, smaller one below it is the beta
surface, and it uses line style to say what stage something is at:

* **solid** — shipped, in the main chart, working today.
* **dotted** — added in beta; present in the tree, not yet promised.
* **red** — declared but not built yet. Something that is planned
  loudly enough to be in the roadmap and honest enough to be drawn as
  missing.

The stage of each edge comes from the source, not from a list
maintained beside it: a module carrying ``__beta__ = True`` is beta, and
a ``__planned__`` tuple names edges that do not exist yet. A chart
whose legend is maintained by hand has the same decay problem as the
chart.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Edge",
    "Module",
    "Architecture",
    "read_architecture",
    "render_mermaid",
    "update_in_place",
    "CHART_START",
    "CHART_END",
]

#: The markers the chart lives between. Everything outside them in the
#: target file is left exactly as it was — this updates a section, it
#: does not own the document.
CHART_START = "<!-- ARCHMAP:START -->"
CHART_END = "<!-- ARCHMAP:END -->"

#: Third-party names worth drawing. The full dependency list is noise;
#: these are the ones whose presence changes what the system *is*.
MAJOR_DEPENDENCIES = {
    "torch": "PyTorch",
    "numpy": "NumPy",
    "fastapi": "FastAPI",
    "uvicorn": "uvicorn",
    "anthropic": "Claude API",
    "rich": "Rich",
    "textual": "Textual",
    "requests": "requests",
    "httpx": "httpx",
}


@dataclass(frozen=True)
class Edge:
    """One module depending on another, and how settled that is."""

    source: str
    target: str
    stage: str = "shipped"        # "shipped" | "beta" | "planned"

    def arrow(self) -> str:
        """Mermaid link syntax carrying the stage in the line style."""
        if self.stage == "beta":
            return "-.->"
        if self.stage == "planned":
            return "-->"
        return "-->"


@dataclass
class Module:
    name: str
    beta: bool = False
    summary: str = ""
    files: int = 0


@dataclass
class Architecture:
    modules: dict[str, Module] = field(default_factory=dict)
    edges: set[Edge] = field(default_factory=set)
    dependencies: set[tuple[str, str]] = field(default_factory=set)
    planned: set[Edge] = field(default_factory=set)

    @property
    def beta_modules(self) -> list[str]:
        return sorted(name for name, m in self.modules.items() if m.beta)


def _top_level(module: str, package: str = "hypernix") -> str:
    """`hypernix.t1api.routers.health` -> `t1api`.

    *package* is a parameter rather than the literal "hypernix" it
    started as: with the name hardcoded, every import in any other
    package resolved to that package's own root, so a tree called
    anything else produced one node and a pile of self-edges. It was
    invisible against this repository, which is called hypernix.
    """
    parts = module.split(".")
    if parts and parts[0] == package:
        parts = parts[1:]
    return parts[0] if parts else ""


def read_architecture(root: Path, package: str = "hypernix") -> Architecture:
    """Walk the package and work out what depends on what."""
    architecture = Architecture()
    source_root = root / "src" / package
    if not source_root.is_dir():
        source_root = root / package
    if not source_root.is_dir():
        return architecture

    for path in sorted(source_root.rglob("*.py")):
        if {"__pycache__", "node_modules"} & set(path.parts):
            continue
        relative = path.relative_to(source_root)
        owner = relative.parts[0] if len(relative.parts) > 1 else path.stem
        if owner in ("__init__", "__main__"):
            owner = package

        module = architecture.modules.setdefault(owner, Module(name=owner))
        module.files += 1

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue

        if not module.summary:
            doc = ast.get_docstring(tree) or ""
            first = doc.strip().splitlines()[0] if doc.strip() else ""
            # `name — what it is`: keep the half that says something.
            module.summary = first.split("—")[-1].strip()[:70]

        if re.search(r"^__beta__\s*=\s*True", source, re.M):
            module.beta = True

        for planned in re.findall(r"^__planned__\s*=\s*\((.*?)\)", source, re.M | re.S):
            for target in re.findall(r"[\"']([\w.]+)[\"']", planned):
                architecture.planned.add(
                    Edge(owner, _top_level(target, package), "planned")
                )

        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:          # relative import: same package
                    continue
                names = [node.module or ""]

            for name in names:
                head = name.split(".")[0]
                if head in MAJOR_DEPENDENCIES:
                    architecture.dependencies.add((owner, head))
                elif name.startswith(f"{package}."):
                    target = _top_level(name, package)
                    if target and target != owner:
                        architecture.edges.add(Edge(owner, target))

    # A module is beta if it says so; an edge into a beta module is a
    # beta edge, which is what makes the second chart draw itself.
    upgraded: set[Edge] = set()
    for edge in architecture.edges:
        target = architecture.modules.get(edge.target)
        stage = "beta" if target and target.beta else "shipped"
        upgraded.add(Edge(edge.source, edge.target, stage))
    architecture.edges = upgraded
    return architecture


def _identifier(name: str) -> str:
    return re.sub(r"\W", "_", name) or "unknown"


def render_mermaid(architecture: Architecture, *, max_modules: int = 28) -> str:
    """Both charts, as Mermaid, ready to drop into Markdown.

    Capped at *max_modules* by file count. A chart with ninety nodes
    communicates less than no chart at all, and the tail of any real
    package is single-file utilities nobody is looking for.
    """
    ranked = sorted(
        architecture.modules.values(), key=lambda m: (-m.files, m.name)
    )[:max_modules]
    keep = {module.name for module in ranked}

    lines: list[str] = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("    %% Shipped architecture — solid lines.")

    for dependency, label in sorted(MAJOR_DEPENDENCIES.items()):
        if any(dep == dependency for _owner, dep in architecture.dependencies):
            lines.append(f"    dep_{_identifier(dependency)}([{label}])")

    for module in ranked:
        shape = f'{_identifier(module.name)}["{module.name}"]'
        lines.append(f"    {shape}")

    seen: set[tuple[str, str]] = set()
    for edge in sorted(architecture.edges, key=lambda e: (e.source, e.target)):
        if edge.stage != "shipped":
            continue
        if edge.source not in keep or edge.target not in keep:
            continue
        if (edge.source, edge.target) in seen:
            continue
        seen.add((edge.source, edge.target))
        lines.append(
            f"    {_identifier(edge.source)} --> {_identifier(edge.target)}"
        )

    for owner, dependency in sorted(architecture.dependencies):
        if owner in keep:
            lines.append(
                f"    {_identifier(owner)} --> dep_{_identifier(dependency)}"
            )

    lines.append("```")

    # ---- the beta chart, smaller, below ----
    beta_edges = [e for e in architecture.edges if e.stage == "beta"]
    planned = sorted(architecture.planned, key=lambda e: (e.source, e.target))
    lines.append("")
    lines.append("### In beta, and not yet built")
    lines.append("")
    lines.append(
        "Solid is shipped and lives in the chart above. **Dotted** is a "
        "feature added in beta — present in the tree, not yet promised. "
        "**Red** is declared and not built yet."
    )
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")

    if not beta_edges and not planned:
        lines.append("    nothing[\"nothing in beta right now\"]")
    else:
        for edge in sorted(beta_edges, key=lambda e: (e.source, e.target)):
            lines.append(
                f"    {_identifier(edge.source)} -.-> {_identifier(edge.target)}"
            )
        for index, edge in enumerate(planned):
            lines.append(
                f"    {_identifier(edge.source)} --> {_identifier(edge.target)}"
            )
            lines.append(f"    linkStyle {len(beta_edges) + index} stroke:#d33,stroke-width:2px")

    lines.append("```")
    return "\n".join(lines)


def update_in_place(document: Path, chart: str) -> bool:
    """Replace the chart between the markers. ``True`` if it changed.

    Replaces rather than appends, which is the whole requirement: a
    release workflow that appends produces a document with eleven
    charts, ten of them wrong, and no way to tell which is current.
    """
    if not document.exists():
        raise FileNotFoundError(
            f"{document} does not exist. Create it with "
            f"{CHART_START} and {CHART_END} where the chart should go."
        )
    text = document.read_text(encoding="utf-8")
    if CHART_START not in text or CHART_END not in text:
        raise ValueError(
            f"{document} has no {CHART_START} / {CHART_END} markers, so there "
            f"is nowhere to put the chart without guessing. Add them."
        )
    before, _, rest = text.partition(CHART_START)
    _, _, after = rest.partition(CHART_END)
    rebuilt = f"{before}{CHART_START}\n{chart}\n{CHART_END}{after}"
    if rebuilt == text:
        return False
    document.write_text(rebuilt, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m hypernix.system.archmap [--write PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="hypernix.system.archmap",
        description="Generate the architecture chart from the source tree.",
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--package", default="hypernix")
    parser.add_argument(
        "--write", default="",
        help="Markdown file to update in place between the ARCHMAP markers.",
    )
    parser.add_argument("--max-modules", type=int, default=28)
    args = parser.parse_args(argv)

    architecture = read_architecture(Path(args.root), args.package)
    chart = render_mermaid(architecture, max_modules=args.max_modules)

    if not args.write:
        print(chart)
        return 0

    try:
        changed = update_in_place(Path(args.write), chart)
    except (FileNotFoundError, ValueError) as exc:
        print(f"archmap: {exc}")
        return 1
    print(
        f"archmap: {len(architecture.modules)} modules, "
        f"{len(architecture.edges)} edges -> {args.write} "
        f"({'updated' if changed else 'already current'})"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
