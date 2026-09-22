"""hypernix.system.archmap — the generated architecture chart.

The requirement that shapes this: the release workflow must *update*
one chart, not append a new one. A page with eleven charts, ten of them
wrong and no way to tell which is current, is worse than the hand-drawn
diagram it replaced — that one was only wrong once.

So the marker handling is tested harder than the graph building: losing
the surrounding prose, or duplicating the chart, are the two failures
that would make the page useless.
"""
from __future__ import annotations

import textwrap

import pytest

from hypernix.system.archmap import (
    CHART_END,
    CHART_START,
    Edge,
    read_architecture,
    render_mermaid,
    update_in_place,
)


@pytest.fixture
def tree(tmp_path):
    """A tiny package with a dependency, a beta module, and a plan."""
    package = tmp_path / "src" / "demo"
    (package / "alpha").mkdir(parents=True)
    (package / "beta").mkdir(parents=True)

    (package / "__init__.py").write_text('"""demo — the root."""\n')
    (package / "alpha" / "__init__.py").write_text(
        '"""alpha — does the work."""\n'
        "import torch\n"
        "from demo.beta import thing\n"
    )
    (package / "beta" / "__init__.py").write_text(
        '"""beta — not settled yet."""\n'
        "__beta__ = True\n"
        "__planned__ = ('demo.gamma',)\n"
    )
    return tmp_path


class TestReadingTheTree:
    def test_it_finds_the_modules(self, tree):
        architecture = read_architecture(tree, "demo")
        assert {"alpha", "beta"} <= set(architecture.modules)

    def test_it_finds_a_major_dependency(self, tree):
        architecture = read_architecture(tree, "demo")
        assert ("alpha", "torch") in architecture.dependencies

    def test_an_edge_into_a_beta_module_is_a_beta_edge(self, tree):
        """The second chart draws itself from this, rather than from a
        list maintained beside it."""
        architecture = read_architecture(tree, "demo")
        edge = next(e for e in architecture.edges if e.target == "beta")
        assert edge.stage == "beta"

    def test_a_planned_edge_is_read_from_the_source(self, tree):
        architecture = read_architecture(tree, "demo")
        assert Edge("beta", "gamma", "planned") in architecture.planned

    def test_beta_modules_are_listed(self, tree):
        assert read_architecture(tree, "demo").beta_modules == ["beta"]

    def test_a_missing_package_is_empty_not_an_error(self, tmp_path):
        assert read_architecture(tmp_path, "nope").modules == {}

    def test_a_broken_file_does_not_stop_the_walk(self, tree):
        (tree / "src" / "demo" / "bad.py").write_text("def f(:\n")
        assert read_architecture(tree, "demo").modules


class TestRendering:
    def test_both_charts_are_present(self, tree):
        chart = render_mermaid(read_architecture(tree, "demo"))
        assert chart.count("```mermaid") == 2

    def test_beta_edges_are_dotted(self, tree):
        assert "-.->" in render_mermaid(read_architecture(tree, "demo"))

    def test_planned_edges_are_red(self, tree):
        assert "stroke:#d33" in render_mermaid(read_architecture(tree, "demo"))

    def test_the_legend_explains_the_line_styles(self, tree):
        chart = render_mermaid(read_architecture(tree, "demo"))
        assert "Dotted" in chart and "Red" in chart

    def test_an_empty_beta_surface_says_so(self, tmp_path):
        """Rather than an empty chart, which reads as a rendering bug."""
        package = tmp_path / "src" / "demo"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""demo."""\n')
        assert "nothing in beta" in render_mermaid(
            read_architecture(tmp_path, "demo")
        )

    def test_it_caps_the_node_count(self, tmp_path):
        """A chart with ninety nodes communicates less than no chart."""
        package = tmp_path / "src" / "demo"
        for index in range(40):
            module = package / f"m{index:02d}"
            module.mkdir(parents=True)
            (module / "__init__.py").write_text(f'"""m{index} — a module."""\n')
        chart = render_mermaid(read_architecture(tmp_path, "demo"), max_modules=5)
        assert sum(1 for line in chart.splitlines() if '["m' in line) <= 5


class TestUpdatingInPlace:
    @staticmethod
    def _page(tmp_path, body: str = "chart goes here"):
        page = tmp_path / "Architecture.md"
        page.write_text(
            textwrap.dedent(f"""\
                # Architecture

                Prose above the chart.

                {CHART_START}
                {body}
                {CHART_END}

                Prose below the chart.
                """)
        )
        return page

    def test_it_replaces_rather_than_appends(self, tmp_path):
        """The whole requirement. Appending gives a page with eleven
        charts and no way to tell which is current."""
        page = self._page(tmp_path, "OLD CHART")
        update_in_place(page, "NEW CHART")
        text = page.read_text()
        assert "NEW CHART" in text
        assert "OLD CHART" not in text
        assert text.count(CHART_START) == 1

    def test_the_surrounding_prose_survives(self, tmp_path):
        page = self._page(tmp_path)
        update_in_place(page, "NEW")
        text = page.read_text()
        assert "Prose above the chart." in text
        assert "Prose below the chart." in text

    def test_running_it_twice_changes_nothing_the_second_time(self, tmp_path):
        """So a release that changes no code does not produce a commit."""
        page = self._page(tmp_path)
        assert update_in_place(page, "SAME") is True
        assert update_in_place(page, "SAME") is False

    def test_a_page_without_markers_is_refused_with_the_reason(self, tmp_path):
        """Rather than guessing where the chart goes and overwriting
        somebody's page."""
        page = tmp_path / "Plain.md"
        page.write_text("# Just a page\n")
        with pytest.raises(ValueError, match="markers"):
            update_in_place(page, "CHART")

    def test_a_missing_page_says_how_to_make_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ARCHMAP"):
            update_in_place(tmp_path / "absent.md", "CHART")


class TestAgainstThisRepository:
    def test_the_real_page_has_its_markers(self):
        from pathlib import Path

        page = Path(__file__).resolve().parent.parent / "wiki" / "Architecture.md"
        text = page.read_text(encoding="utf-8")
        assert CHART_START in text and CHART_END in text
        assert text.count(CHART_START) == 1, "more than one chart block"

    def test_the_real_tree_renders(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        chart = render_mermaid(read_architecture(root))
        assert "```mermaid" in chart
        assert "t1api" in chart
