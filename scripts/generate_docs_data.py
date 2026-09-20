#!/usr/bin/env python3
"""Generate machine-readable API and codebase data for the HyperNix docs site.

The generated files are deliberately source-derived: the docs build can show
what is actually present in this checkout without maintaining a second hand-
written API inventory. When full git history is available, recent file history
is attached to each module and contributor line-change statistics are emitted.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "hypernix"
PUBLIC_ROOT = ROOT / "docs" / "public" / "v1"
API_DEEP_PATH = PUBLIC_ROOT / "api-deep.json"
T1_API_PATH = PUBLIC_ROOT / "t1-api.json"
CODE_STATS_PATH = PUBLIC_ROOT / "code-stats.json"
CHANGELOG_PATH = ROOT / "wiki" / "Changelog.md"
CHANGELOG_DATA_PATH = PUBLIC_ROOT / "changelog.json"

CODE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hh", ".hpp", ".m", ".mm", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".rs", ".go", ".java", ".kt", ".swift",
}
EXCLUDED_DIRS = {
    ".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__", ".pytest_cache",
}

# Keep this compiled once and close to the source-data configuration. The
# previous inline expression was accidentally double-escaped when generated,
# which made Python reject it as an unterminated regex on GitHub Actions.
API_CHANGE_RE = re.compile(
    r"(?:^|\s)(?:async\s+)?(?:def|class)\s+"
    r"|^@(?:router|app|[A-Za-z_][A-Za-z0-9_]*)\.(?:get|post|put|patch|delete|options|head)\("
    r"|add_api_route\("
    r"|deprecated_module\("
)

# Base and optional dependencies from pyproject.toml. This intentionally stays
# small and explicit because the docs need package-level install instructions,
# not every transitive module imported by a file.
DEPENDENCY_MAP = {
    "torch": ("base", "torch>=1.13,<3"),
    "numpy": ("base", "numpy>=1.26,<3"),
    "safetensors": ("base", "safetensors>=0.4.3"),
    "huggingface_hub": ("base", "huggingface-hub>=0.24"),
    "gguf": ("base", "gguf>=0.10.0"),
    "tqdm": ("base", "tqdm>=4.66"),
    "rich": ("base", "rich>14,<=15.0.0"),
    "sentencepiece": ("base", "sentencepiece>=0.2.1"),
    "llama_cpp": ("llama-cpp", "llama-cpp-python>=0.2.90"),
    "transformers": ("train", "transformers>=4.44"),
    "accelerate": ("train", "accelerate>=0.33"),
    "cryptography": ("security", "cryptography>=42"),
    "fastapi": ("t1api", "fastapi>=0.110,<1"),
    "uvicorn": ("t1api", "uvicorn[standard]>=0.29,<1"),
    "pydantic": ("t1api", "pydantic>=2,<3"),
    "dotenv": ("t1api", "python-dotenv>=1,<2"),
    "multipart": ("t1api", "python-multipart>=0.0.9,<1"),
    "psycopg": ("t1api-pg", "psycopg[binary,pool]>=3.1,<4"),
    "httpx": ("t1api-test", "httpx>=0.27,<1"),
    "PySide6": ("gui-qt", "PySide6>=6.6"),
    "gi": ("gui-gtk", "PyGObject>=3.48"),
}

CURATED_EXAMPLES = [
    {
        "title": "Download, convert, and quantize",
        "module": "hypernix",
        "description": "The public convenience API used by the bundled quickstart example.",
        "code": "from pathlib import Path\nfrom hypernix import convert_to_gguf, download_model, quantize_gguf\n\nmodel_dir = download_model(\"hyper-Nix.2\")\nout = Path(\"./hypernix-gguf\")\nfp16 = convert_to_gguf(model_dir, out / \"hypernix-fp16.gguf\", dtype=\"fp16\")\nquantize_gguf(fp16, out / \"hypernix-q4_k_m.gguf\", \"Q4_K_M\")",
        "source": "examples/quickstart.py",
    },
    {
        "title": "Use the multi-turn chat surface",
        "module": "hypernix.chat.countertop",
        "description": "Create a session around an oven, then persist it when the workflow finishes.",
        "code": "from hypernix.countertop import Countertop\nfrom hypernix.old_oven import preheat\n\noven = preheat(\"hyper-Nix.2\")\nchat = Countertop(oven, system=\"Answer with concise Python examples.\")\nprint(chat.say(\"Show me a dataclass with validation.\"))\nchat.save(\"session.json\")",
        "source": "src/hypernix/chat/countertop.py",
    },
    {
        "title": "Build a training framework",
        "module": "hypernix.models.workshop",
        "description": "Construct a framework, build its module, move it to a device, and save weights.",
        "code": "from hypernix.workshop import FrameworkConfig, WorkshopFramework\n\nframework = WorkshopFramework(FrameworkConfig(vocab_size=32000))\nmodel = framework.build().to(\"cuda\")\n# ... train model ...\nframework.save(\"./checkpoint\")",
        "source": "src/hypernix/models/workshop.py",
    },
    {
        "title": "Call a T1 API server",
        "module": "hypernix.t1sdk.client",
        "description": "The bundled SDK exposes typed helpers and a raw escape hatch for newer endpoints.",
        "code": "from hypernix.t1sdk import T1Client\n\nclient = T1Client(\"https://t1.example.com\", credential=\"YOUR_T1_KEY\")\nprint(client.health())\nprint(client.whoami())\nmodels = client.list_models()\nprint([m.model_id for m in models])\n# New endpoint not wrapped by this SDK release:\nprint(client.call(\"GET\", \"/status\"))",
        "source": "src/hypernix/t1sdk/client.py",
    },
]




def clean_changelog_text(text: str, limit: int = 360) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_~]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip() + "…"


def release_kind(version: str) -> str:
    """Classify releases so stable, post, alpha, beta, RC, and dev are distinct."""
    v = version.strip().lower().lstrip("v")
    base = re.match(r"^\d+\.\d+\.\d+", v)
    tail = v[base.end():] if base else v
    compact = re.sub(r"[._ -]+", "", tail)
    if v.startswith("postv") or re.search(r"(?:post\d+|post)$", compact):
        return "post"
    if re.search(r"(?:a\d+|alpha\d*|pt\d*a)$", compact):
        return "alpha"
    if re.search(r"(?:b\d+|beta\d*|pt\d*b)$", compact):
        return "beta"
    if re.search(r"(?:rc\d+|releasecandidate\d*)$", compact):
        return "rc"
    if re.search(r"(?:dev\d*|development)$", compact):
        return "dev"
    if re.search(r"pt\d*", compact) or re.search(r"[-_]\d+$", tail):
        return "patch"
    if base and not tail.strip():
        return "stable"
    return "other"


def normalize_release_label(value: str) -> str:
    v = value.strip().lstrip("vV")
    v = v.replace("–", "-").replace("—", "-")
    v = re.sub(r"[\(].*?\]?$", "", v).strip()
    v = re.sub(r"\s+", " ", v)
    return v.lower()


def changelog_aliases(version_label: str) -> set[str]:
    """Return exact-ish forms used by Git tags and human changelog headings."""
    v = normalize_release_label(version_label)
    aliases = {v, v.replace(" ", "-")}
    if " pt" in v:
        aliases.add(v.replace(" pt", "-pt"))
    if ".post" in v:
        aliases.add(v.replace(".post", "-post"))
    if re.search(r"[.]dev\d+$", v):
        aliases.add(v.replace(".dev", "dev"))
    return {a.lstrip("v") for a in aliases}


def parse_changelog() -> dict[str, Any]:
    if not CHANGELOG_PATH.exists():
        return {"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "entries": []}
    text = CHANGELOG_PATH.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    entries: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        raw_heading = match.group(1).strip()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():body_end].strip()
        # Headings such as "0.72.5 pt3 — ..." use the text before an em dash
        # as the actual release label. Parenthetical notes are retained only
        # when they are part of the label (e.g. pt 4).
        label = raw_heading.split("—", 1)[0].strip()
        label = label.split(" – ", 1)[0].strip()
        if not re.match(r"^v?\d+\.\d+\.\d+", label, re.I):
            continue
        title = ""
        if "—" in raw_heading:
            title = raw_heading.split("—", 1)[1].strip()
        elif " - " in raw_heading and re.match(r"^v?\d+\.\d+\.\d+[^ ]* - ", raw_heading, re.I):
            title = raw_heading.split(" - ", 1)[1].strip()
        summary = clean_changelog_text(title) if title else ""
        if not summary:
            for candidate in body.splitlines():
                candidate = candidate.strip()
                if not candidate or candidate.startswith("#") or candidate.startswith("```"):
                    continue
                summary = clean_changelog_text(candidate)
                if summary:
                    break
        entries.append({
            "version": label,
            "key": normalize_release_label(label),
            "aliases": sorted(changelog_aliases(label)),
            "kind": release_kind(label),
            "heading": raw_heading,
            "summary": summary or "Release notes in wiki/Changelog.md.",
        })
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "wiki/Changelog.md",
        "entries": entries,
    }

def run_git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip()


def git_available() -> bool:
    return bool(run_git("rev-parse", "--is-inside-work-tree"))


def current_commit() -> str | None:
    value = run_git("rev-parse", "HEAD")
    return value or None


def first_doc_paragraph(doc: str | None, limit: int = 520) -> str:
    if not doc:
        return ""
    clean = re.sub(r"\n{2,}", "\n\n", doc.strip())
    first = clean.split("\n\n", 1)[0]
    first = re.sub(r"\s+", " ", first).strip()
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def annotation_to_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def signature_for_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ast.unparse(node.args)
    ret = f" -> {annotation_to_text(node.returns)}" if node.returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}{args}{ret}"


def module_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT / "src").with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def import_names(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def required_modules(imports: set[str], *, t1: bool = False) -> list[dict[str, str]]:
    out = []
    seen = set()
    for name in sorted(imports):
        key = name if name in DEPENDENCY_MAP else name.split(".")[0]
        if key not in DEPENDENCY_MAP or key in seen:
            continue
        seen.add(key)
        extra, requirement = DEPENDENCY_MAP[key]
        out.append({"module": name, "extra": extra, "requirement": requirement})
    # T1 server pages always need the server extra for create_app even if a
    # particular router imports only stdlib objects.
    if t1 and not any(x["extra"] == "t1api" for x in out):
        out.append({
            "module": "fastapi / uvicorn / pydantic",
            "extra": "t1api",
            "requirement": "pip install 'hypernix[t1api]'",
        })
    return out


def extract_deprecations(tree: ast.Module) -> list[dict[str, str | None]]:
    items = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == "deprecated_module":
            vals: dict[str, str | None] = {"name": None, "instead": None, "since": None, "removed_in": None, "extra": None}
            if node.args and isinstance(node.args[0], ast.Constant):
                vals["name"] = str(node.args[0].value)
            for kw in node.keywords:
                if kw.arg in vals and isinstance(kw.value, ast.Constant):
                    vals[kw.arg] = str(kw.value.value) if kw.value.value is not None else None
            items.append(vals)
        elif name == "warn":
            msg = None
            category = None
            if node.args and isinstance(node.args[0], ast.Constant):
                msg = str(node.args[0].value)
            if len(node.args) > 1 and isinstance(node.args[1], ast.Name):
                category = node.args[1].id
            for kw in node.keywords:
                if kw.arg == "category" and isinstance(kw.value, ast.Name):
                    category = kw.value.id
            if msg or category:
                replacement = None
                if msg:
                    match = re.search(r"(?:use|consider)\s+[`'\"]?([A-Za-z_][A-Za-z0-9_.]+)[`'\"]?\s+instead", msg, re.I)
                    if match:
                        replacement = match.group(1)
                items.append({
                    "name": "warnings.warn",
                    "instead": replacement,
                    "since": None,
                    "removed_in": None,
                    "extra": (f"{category}: {msg}" if category and msg else msg or category),
                })
    return items


def error_names(tree: ast.Module) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (node.name.endswith("Error") or node.name.endswith("Exception")):
            found.add(node.name)
        elif isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name) and (exc.id.endswith("Error") or exc.id.endswith("Exception")):
                found.add(exc.id)
            elif isinstance(exc, ast.Attribute) and (exc.attr.endswith("Error") or exc.attr.endswith("Exception")):
                found.add(exc.attr)
    return sorted(found)


def public_api(tree: ast.Module) -> list[dict[str, Any]]:
    api: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            api.append({
                "kind": "function",
                "name": node.name,
                "signature": signature_for_function(node),
                "line": node.lineno,
                "doc": first_doc_paragraph(ast.get_docstring(node)),
            })
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    methods.append({
                        "name": child.name,
                        "signature": signature_for_function(child),
                        "line": child.lineno,
                        "doc": first_doc_paragraph(ast.get_docstring(child)),
                    })
            api.append({
                "kind": "class",
                "name": node.name,
                "signature": f"class {node.name}",
                "line": node.lineno,
                "doc": first_doc_paragraph(ast.get_docstring(node)),
                "methods": methods,
            })
    return api


def recent_history(path: Path, limit: int = 8) -> list[dict[str, Any]]:
    if not git_available():
        return []
    rel = str(path.relative_to(ROOT)).replace(os.sep, "/")
    raw = run_git("log", f"-n{limit}", "--date=short", "--format=%h%x09%ad%x09%an%x09%s", "--", rel)
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        sha, date, author, subject = parts
        diff = run_git("show", "--format=", "--unified=0", sha, "--", rel)
        changed: list[dict[str, str]] = []
        for diff_line in diff.splitlines():
            if diff_line.startswith(("+++", "---", "@@")):
                continue
            if not diff_line.startswith(("+", "-")):
                continue
            text = diff_line[1:].strip()
            if not API_CHANGE_RE.search(text):
                continue
            changed.append({"kind": "added" if diff_line[0] == "+" else "removed", "text": text[:300]})
        out.append({"sha": sha, "date": date, "author": author, "subject": subject, "api_changes": changed[:24]})
    return out


def build_module(path: Path, *, t1: bool = False) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return {
            "module": module_name_for(path),
            "source": str(path.relative_to(ROOT)),
            "line_count": len(source.splitlines()),
            "syntax_error": str(exc),
            "api": [],
            "errors": [],
            "deprecations": [],
            "required_modules": [],
            "changes": recent_history(path),
        }
    imports = import_names(tree)
    return {
        "module": module_name_for(path),
        "source": str(path.relative_to(ROOT)),
        "line_count": sum(1 for line in source.splitlines() if line.strip()),
        "summary": first_doc_paragraph(ast.get_docstring(tree)),
        "api": public_api(tree),
        "errors": error_names(tree),
        "deprecations": extract_deprecations(tree),
        "required_modules": required_modules(imports, t1=t1),
        "changes": recent_history(path),
    }


def source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if not any(part in EXCLUDED_DIRS for part in p.parts))


def git_code_contributions() -> tuple[list[dict[str, Any]], bool]:
    if not git_available():
        return [], False
    # Use numstat so merges and rename bookkeeping do not masquerade as huge
    # source contributions. Each commit contributes its actual added/deleted
    # line counts to its author; the docs UI sorts by additions first and
    # deletions second, as requested.
    raw = run_git("log", "--format=__COMMIT__%x09%H%x09%an", "--numstat", "--", "src", "scripts", "tests", "examples", "bin", "native", "docs/src")
    if not raw:
        return [], True
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"additions": 0, "deletions": 0, "commits": 0, "files": set()})
    current_author = None
    current_hash = None
    seen_commits: set[str] = set()
    for line in raw.splitlines():
        if line.startswith("__COMMIT__\t"):
            _, current_hash, current_author = line.split("\t", 2)
            if current_hash in seen_commits:
                current_author = None
                current_hash = None
                continue
            seen_commits.add(current_hash)
            totals[current_author]["commits"] += 1
        elif current_author:
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            add, delete, file_name = parts
            if add.isdigit():
                totals[current_author]["additions"] += int(add)
            if delete.isdigit():
                totals[current_author]["deletions"] += int(delete)
            totals[current_author]["files"].add(file_name)
    rows = []
    for author, data in totals.items():
        rows.append({
            "author": author,
            "additions": data["additions"],
            "deletions": data["deletions"],
            "net": data["additions"] - data["deletions"],
            "changes": data["additions"] + data["deletions"],
            "commits": data["commits"],
            "files": len(data["files"]),
        })
    rows.sort(key=lambda x: (-x["additions"], -x["deletions"], x["author"].lower()))
    return rows, True


def codebase_stats() -> dict[str, Any]:
    roots = [ROOT / name for name in ("src", "scripts", "tests", "examples", "bin", "native", "docs" / Path("src"))]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in CODE_EXTENSIONS and not any(part in EXCLUDED_DIRS for part in p.parts):
                files.append(p)
    files = sorted(set(files))
    line_counts = {}
    total = 0
    for p in files:
        try:
            count = sum(1 for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
        except OSError:
            continue
        line_counts[str(p.relative_to(ROOT)).replace(os.sep, "/")] = count
        total += count
    contributors, history_available = git_code_contributions()
    return {
        "definition": "non-blank source lines across repository code roots: src, scripts, tests, examples, bin, native, docs/src",
        "total_lines": total,
        "code_files": len(line_counts),
        "top_files": sorted(
            [{"path": p, "lines": n} for p, n in line_counts.items()],
            key=lambda x: (-x["lines"], x["path"]),
        )[:30],
        "contributors": contributors,
        "history_available": history_available,
        "commit": current_commit(),
    }


def t1_routes() -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    root = SRC_ROOT / "t1api" / "routers"
    route_re = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)\.(get|post|put|patch|delete|options|head)\(")
    prefix_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*APIRouter\((.*?)\)", re.S)
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        prefix_map: dict[str, tuple[str, str]] = {}
        for m in prefix_re.finditer(source):
            text = m.group(2)
            pm = re.search(r"prefix\s*=\s*[\"']([^\"']*)", text)
            tm = re.search(r"tags\s*=\s*\[[\"']([^\"']+)", text)
            prefix_map[m.group(1)] = (pm.group(1) if pm else "", tm.group(1) if tm else path.stem)
        lines = source.splitlines()
        for i, line in enumerate(lines, start=1):
            m = route_re.search(line)
            if m:
                router_var = m.group(1)
                decorator_lines = line
                j = i
                while ")" not in decorator_lines and j < len(lines):
                    j += 1
                    decorator_lines += lines[j - 1]
                pm = re.search(r"\(\s*[\"']([^\"']*)[\"']", decorator_lines)
                path_part = pm.group(1) if pm else ""
                response = re.search(r"response_model\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", decorator_lines)
                prefix, tag = prefix_map.get(router_var, ("", path.stem))
                full_path = (prefix.rstrip("/") + "/" + path_part.lstrip("/")).replace("//", "/") or "/"
                handler = None
                for k in range(j, min(j + 8, len(lines)) + 1):
                    hm = re.match(r"\s*(?:async\s+def|def)\s+(\w+)\s*\(", lines[k - 1])
                    if hm:
                        handler = hm.group(1)
                        break
                routes.append({
                    "method": m.group(2).upper(),
                    "router": router_var,
                    "path": full_path,
                    "handler": handler or "unknown",
                    "response_model": response.group(1) if response else None,
                    "source": str(path.relative_to(ROOT)).replace(os.sep, "/"),
                    "line": i,
                    "tag": tag,
                })
            # Some T1 compatibility routes are registered programmatically
            # with APIRouter.add_api_route rather than decorators. Record those
            # too so the generated table matches FastAPI's actual route set.
            add = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\.add_api_route\(\s*[\"']([^\"']+)[\"']\s*,\s*(\w+).*?methods\s*=\s*\[\s*[\"']([^\"']+)[\"']", line)
            if add:
                router_var, path_part, handler, method = add.groups()
                response = re.search(r"response_model\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", line)
                prefix, tag = prefix_map.get(router_var, ("", path.stem))
                full_path = (prefix.rstrip("/") + "/" + path_part.lstrip("/")).replace("//", "/") or "/"
                routes.append({
                    "method": method.upper(),
                    "router": router_var,
                    "path": full_path,
                    "handler": handler,
                    "response_model": response.group(1) if response else None,
                    "source": str(path.relative_to(ROOT)).replace(os.sep, "/"),
                    "line": i,
                    "tag": tag,
                })
    routes.sort(key=lambda x: (x["path"], x["method"], x["source"], x["line"]))
    return routes


def build_api_deep() -> dict[str, Any]:
    modules = [build_module(p) for p in source_files(SRC_ROOT)]
    deprecations = []
    for m in modules:
        for item in m.get("deprecations", []):
            entry = dict(item)
            entry["module"] = m["module"]
            entry["source"] = m["source"]
            deprecations.append(entry)
    return {
        "kind": "hypernix-api-deep",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": current_commit(),
        "history_available": git_available(),
        "source_root": "src/hypernix",
        "modules": modules,
        "deprecations": deprecations,
        "examples": CURATED_EXAMPLES,
    }


def build_t1_api() -> dict[str, Any]:
    t1_root = SRC_ROOT / "t1api"
    sdk_root = SRC_ROOT / "t1sdk"
    modules = [build_module(p, t1=True) for p in source_files(t1_root)] + [build_module(p, t1=True) for p in source_files(sdk_root)]
    errors: list[dict[str, str]] = []
    for path in sorted((t1_root, sdk_root)[0].glob("*.py")) if False else []:
        pass
    for m in modules:
        for err in m.get("errors", []):
            errors.append({"name": err, "module": m["module"]})
    # Pull the T1 error class hierarchy separately so the page can show the
    # stable error names without requiring server dependencies.
    error_mod = sdk_root / "errors.py"
    if error_mod.exists():
        try:
            tree = ast.parse(error_mod.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and (node.name.endswith("Error") or node.name.endswith("Exception")):
                    bases = [ast.unparse(b) for b in node.bases]
                    errors.append({"name": node.name, "module": "hypernix.t1sdk.errors", "base": ", ".join(bases)})
        except SyntaxError:
            pass
    return {
        "kind": "hypernix-t1-api",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": current_commit(),
        "history_available": git_available(),
        "server_extra": "pip install 'hypernix[t1api]'",
        "client_install": "pip install hypernix",
        "mount_prefix": "T1_MOUNT_PREFIX environment variable (default: empty); routes below are router-relative paths.",
        "modules": modules,
        "routes": t1_routes(),
        "errors": sorted({(e.get("name"), e.get("module"), e.get("base", "")): e for e in errors}.values(), key=lambda x: (x.get("name", ""), x.get("module", ""))),
        "examples": [
            {
                "title": "Use the Python SDK",
                "module": "hypernix.t1sdk.client",
                "description": "Typed helpers use the same transport, retries, TLS settings, and error mapping as raw calls.",
                "code": "from hypernix.t1sdk import T1Client\n\nclient = T1Client(\"https://t1.example.com\", credential=\"YOUR_T1_KEY\")\nprint(client.health())\nprint(client.whoami())\nprint(client.list_models())",
                "source": "src/hypernix/t1sdk/client.py",
            },
            {
                "title": "Run the T1 server",
                "module": "hypernix.t1api.app",
                "description": "The server entry point is a FastAPI application factory.",
                "code": "pip install 'hypernix[t1api]'\npython -m uvicorn hypernix.t1api.app:create_app --factory --host 127.0.0.1 --port 8000",
                "source": "examples/t1api/README.md",
            },
            {
                "title": "Raw endpoint escape hatch",
                "module": "hypernix.t1sdk.client",
                "description": "Use raw access when a server is newer than the SDK wrapper without waiting for a client release.",
                "code": "response = client.call(\"POST\", \"/usage/report\", body={\"events\": []}, auth=True)\nprint(response)",
                "source": "src/hypernix/t1sdk/client.py",
            },
        ],
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    api_deep = build_api_deep()
    t1_api = build_t1_api()
    stats = codebase_stats()
    # The three writes are deterministic except for generated_at. We preserve
    # the timestamp only when the substantive content changed so the hourly
    # workflow can skip commits on no-op runs.
    changelog = parse_changelog()
    existing_paths = [API_DEEP_PATH, T1_API_PATH, CODE_STATS_PATH, CHANGELOG_DATA_PATH]
    built = [api_deep, t1_api, stats, changelog]
    for path, data in zip(existing_paths, built):
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                old = None
            if isinstance(old, dict):
                data["generated_at"] = old.get("generated_at", data.get("generated_at"))
        write_json(path, data)
    print(json.dumps({
        "api_modules": len(api_deep["modules"]),
        "t1_modules": len(t1_api["modules"]),
        "t1_routes": len(t1_api["routes"]),
        "total_lines": stats["total_lines"],
        "code_files": stats["code_files"],
        "contributors": len(stats["contributors"]),
        "changelog_entries": len(changelog["entries"]),
        "history_available": stats["history_available"],
    }, indent=2))


if __name__ == "__main__":
    main()
