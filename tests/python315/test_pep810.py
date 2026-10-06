"""PEP 810 -- explicit lazy imports -- as hyperNix-pip uses it.

Shared source cannot contain ``lazy import`` (3.12-3.14 cannot parse
it), so modules opt in with ``__lazy_modules__``: on 3.15 the listed
imports are deferred to first use, and before 3.15 the list is an
ordinary variable and nothing changes. These tests keep the use honest:
every listed module is really imported there, nothing security-related
is deferred, the deferral happens on 3.15, and the code still works
when the deferred module is finally touched.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hypernix._compat import is_py315_only

SRC = Path(__file__).resolve().parents[2] / "src"
needs_315 = pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports are 3.15")

#: Imports that must run when their module does: key handling, auth and
#: config initialisation fail loudly and early, never at first use.
MUST_STAY_EAGER = ("cryptography", "ssl", "hmac", "secrets", "hashlib",
                   "hypernix.security", "hypernix.t1api.config", "hypernix.t1api.auth")


def _declarations() -> dict[str, list[str]]:
    found = {}
    for path in sorted((SRC / "hypernix").rglob("*.py")):
        if is_py315_only(path) or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__lazy_modules__"):
                value = ast.literal_eval(node.value)
                dotted = ".".join(path.relative_to(SRC).with_suffix("").parts)
                found[dotted.removesuffix(".__init__")] = list(value)
    return found


DECLARED = _declarations()


def test_something_uses_it():
    assert DECLARED, "no module declares __lazy_modules__"


@pytest.mark.parametrize("module", sorted(DECLARED))
def test_each_declaration_is_well_formed(module):
    names = DECLARED[module]
    assert isinstance(names, list) and names and all(isinstance(n, str) for n in names)
    path = SRC / Path(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assign = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", "") == "__lazy_modules__")
    imported = set()
    for node in tree.body[assign + 1:]:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    for name in names:
        # Declared before the imports it affects, and naming one that is there.
        assert name in imported, f"{module} lists {name} but never imports it at top level"


@pytest.mark.parametrize("module", sorted(DECLARED))
def test_nothing_security_related_is_deferred(module):
    for name in DECLARED[module]:
        assert not name.startswith(MUST_STAY_EAGER), f"{module} defers {name}"


def test_the_lazy_keyword_stays_in_3_15_only_files():
    pattern = re.compile(r"^\s*lazy\s+(import|from)\s", re.MULTILINE)
    offenders = [str(p) for p in (SRC / "hypernix").rglob("*.py")
                 if not p.name.endswith("_py315.py")
                 and pattern.search(p.read_text(encoding="utf-8"))]
    assert not offenders


def _probe(module: str, names: list[str], *, lazy: bool = True) -> dict:
    """Import *module* in a fresh interpreter; report which *names* loaded."""
    off = "" if lazy else "sys.set_lazy_imports_filter(lambda *args: False)\n"
    code = (f"import sys, json\n{off}import {module}\n"
            f"print(json.dumps({{n: n in sys.modules for n in {names!r}}}))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_315
@pytest.mark.parametrize("module", sorted(DECLARED))
def test_on_3_15_the_import_is_deferred(module):
    loaded = _probe(module, DECLARED[module])
    assert not any(loaded.values()), f"importing {module} still loaded {loaded}"


@needs_315
@pytest.mark.parametrize("module", sorted(DECLARED))
def test_with_lazy_imports_filtered_off_they_load_as_before(module):
    """The baseline the deferral is measured against."""
    loaded = _probe(module, DECLARED[module], lazy=False)
    assert all(loaded.values()), loaded


@needs_315
def test_a_deferred_import_resolves_on_first_use():
    code = ("import sys\nimport hypernix.quant.llamaquants as lq\n"
            "assert 'numpy' not in sys.modules\n"
            "np = lq.np\nassert np.zeros(2).sum() == 0 and 'numpy' in sys.modules\n"
            "print('ok')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}, check=False)
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr


@needs_315
def test_the_keyword_itself_works_here():
    namespace: dict = {}
    exec("lazy import json\nresult = json.dumps([1])\n", namespace)
    assert namespace["result"] == "[1]"
