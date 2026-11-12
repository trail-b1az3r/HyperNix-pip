"""3H (tritium): the interpreted .hnx element scripting runtime.

The 3H isotope extends Hydrogen; it is not a second periodic-table element.
.hnx files are editable UTF-8 text: a format header, JSONL metadata records,
then a small brace-delimited language combining Python-like expressions,
TypeScript-like declarations and C++-style typed signatures.

The parser uses ast but never calls compile(), eval(), or exec(). Only a
small allow-list of statements, expressions and methods is interpreted.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..system import errorcatalogue as codes
from ..system.errorcodes import HyperNixError
from .hydrogen import ELEMENT_FORMAT_VERSION, Element, ElementSpec, Registry

__all__ = [
    "ISOTOPE_3H", "HNXProgram", "IsotopeSpec",
    "load_hnx_elements", "parse_hnx", "scaffold_hnx",
]

_HOOKS = frozenset({
    "activate", "deactivate", "transform_prompt",
    "transform_messages", "transform_output",
})
_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|===|!==|&&|\|\||!(?!=)|'
    r'\btrue\b|\bfalse\b|\bnull\b'
)
_TYPE_NAMES = {
    "string": "str", "str": "str", "boolean": "bool", "bool": "bool",
    "number": "float", "double": "float", "float": "float",
    "integer": "int", "int": "int", "json": "dict", "any": "object",
    "void": "None",
}
_SAFE_FUNCTIONS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "min": min, "max": max, "sum": sum, "sorted": sorted,
    "all": all, "any": any,
    "json_parse": json.loads,
    "json_stringify": lambda value: json.dumps(value, ensure_ascii=False),
}
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
_CMPOPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
_UNARYOPS = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}


@dataclass(frozen=True)
class IsotopeSpec:
    symbol: str
    base_element: str
    name: str
    purpose: str
    format_version: str = ELEMENT_FORMAT_VERSION

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "base_element": self.base_element,
            "name": self.name,
            "purpose": self.purpose,
            "format_version": self.format_version,
            "execution": "interpreted; no native compilation",
        }


ISOTOPE_3H = IsotopeSpec(
    symbol="3H",
    base_element="H",
    name="tritium",
    purpose="Interprets editable .hnx scripts and loads v1.1.0 elements.",
)


def _fail(message: str) -> HyperNixError:
    return HyperNixError(codes.ELEMENT_LOAD_FAILED, message)


def _rewrite_tokens(expression: str) -> str:
    """Translate JS/C++ operators without changing quoted literals."""
    replacements = {
        "===": "==", "!==": "!=", "&&": " and ", "||": " or ",
        "!": " not ", "true": "True", "false": "False", "null": "None",
    }

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token[:1] in ("'", '"'):
            return token
        return replacements.get(token, token)

    return _TOKEN.sub(replace, expression).strip()


def _python_types(source: str) -> str:
    for source_type, python_type in _TYPE_NAMES.items():
        source = re.sub(rf"\b{re.escape(source_type)}\b", python_type, source)
    source = re.sub(r"\blist\s*<\s*str\s*>", "list[str]", source)
    source = re.sub(r"\bvector\s*<\s*str\s*>", "list[str]", source)
    return source


def _normalise_script(source: str) -> str:
    """Convert .hnx braces/declarations into syntax the AST parser understands."""
    output: list[str] = []
    indent = 0

    for line_number, original in enumerate(source.splitlines(), start=1):
        line = original.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line.startswith("}"):
            indent -= 1
            if indent < 0:
                raise _fail(f".hnx line {line_number}: unmatched closing brace")
            line = line[1:].strip()
            if not line:
                continue

        opens_block = line.endswith("{")
        if opens_block:
            line = line[:-1].strip()
        if line.endswith(";"):
            line = line[:-1].rstrip()

        if line.startswith("fn "):
            signature = line[3:].strip()
            signature = re.sub(
                r"\b(?:std::)?(string|int|float|double|bool|boolean)\s+([A-Za-z_]\w*)",
                lambda match: f"{match.group(2)}: {_TYPE_NAMES[match.group(1)]}",
                signature,
            )
            line = "def " + _python_types(signature)
            if opens_block:
                line += ":"
        elif line.startswith("function "):
            signature = line[len("function "):].strip()
            signature = re.sub(r"\)\s*:\s*[A-Za-z_][\w<>\[\]]*$", ")", signature)
            signature = re.sub(
                r"\b(?:std::)?(string|int|float|double|bool|boolean)\s+([A-Za-z_]\w*)",
                lambda match: f"{match.group(2)}: {_TYPE_NAMES[match.group(1)]}",
                signature,
            )
            line = "def " + _python_types(signature)
            if opens_block:
                line += ":"
        elif re.match(
            r"^(?:std::)?(?:string|void|int|float|double|bool|auto)\s+[A-Za-z_]\w*\s*\(",
            line,
        ):
            match = re.match(
                r"^(?:std::)?(string|void|int|float|double|bool|auto)\s+"
                r"([A-Za-z_]\w*)\s*\((.*)\)$", line,
            )
            if not match:
                raise _fail(f".hnx line {line_number}: malformed C++-style hook signature")
            return_type, name, parameters = match.groups()
            parameters = re.sub(
                r"\b(?:std::)?(string|int|float|double|bool|boolean)\s+([A-Za-z_]\w*)",
                lambda item: f"{item.group(2)}: {_TYPE_NAMES[item.group(1)]}",
                parameters,
            )
            result_type = "" if return_type == "void" else f" -> {_TYPE_NAMES.get(return_type, 'object')}"
            line = f"def {name}({parameters}){result_type}"
            if opens_block:
                line += ":"
        elif line.startswith("def "):
            signature = re.sub(
                r"\b(?:std::)?(string|int|float|double|bool|boolean)\s+([A-Za-z_]\w*)",
                lambda match: f"{match.group(2)}: {_TYPE_NAMES[match.group(1)]}",
                line[4:].strip(),
            )
            line = "def " + _python_types(signature)
            if opens_block:
                line += ":"
        elif line.startswith("if "):
            condition = line[3:].strip()
            if condition.startswith("(") and condition.endswith(")"):
                condition = condition[1:-1].strip()
            line = "if " + _rewrite_tokens(condition)
            if opens_block:
                line += ":"
        elif line.startswith("else if "):
            condition = line[8:].strip()
            if condition.startswith("(") and condition.endswith(")"):
                condition = condition[1:-1].strip()
            line = "elif " + _rewrite_tokens(condition)
            if opens_block:
                line += ":"
        elif line == "else":
            line = "else:"
        else:
            declaration = re.match(
                r"^(?:let|const|var|auto)\s+([A-Za-z_]\w*)"
                r"(?:\s*:\s*[A-Za-z_][\w:<>\[\]]*)?\s*=\s*(.+)$",
                line,
            )
            if declaration:
                line = f"{declaration.group(1)} = {declaration.group(2)}"
            else:
                declaration = re.match(
                    r"^(?:std::)?(string|int|float|double|bool|boolean|auto)\s+"
                    r"([A-Za-z_]\w*)\s*=\s*(.+)$",
                    line,
                )
                if declaration:
                    line = f"{declaration.group(2)} = {declaration.group(3)}"
            line = _rewrite_tokens(line)

        if opens_block and not line.endswith(":"):
            raise _fail(f".hnx line {line_number}: unsupported block declaration")
        if line:
            output.append(("    " * indent) + _python_types(line))
        if opens_block:
            indent += 1

    if indent != 0:
        raise _fail("unterminated .hnx block; expected a closing brace")
    if not output:
        raise _fail(".hnx file contains no hook definitions")
    return "\n".join(output)


def _validate_expression(node: ast.expr) -> None:
    allowed = (
        ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice, ast.Call,
        ast.IfExp, ast.JoinedStr, ast.FormattedValue,
    )
    if not isinstance(node, allowed):
        raise _fail(f".hnx expression {type(node).__name__} is not supported")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise _fail("unsupported .hnx arithmetic operator")
        _validate_expression(node.left)
        _validate_expression(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise _fail("unsupported .hnx unary operator")
        _validate_expression(node.operand)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate_expression(value)
    elif isinstance(node, ast.Compare):
        if any(type(op) not in _CMPOPS for op in node.ops):
            raise _fail("unsupported .hnx comparison operator")
        _validate_expression(node.left)
        for value in node.comparators:
            _validate_expression(value)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for value in node.elts:
            _validate_expression(value)
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None:
                _validate_expression(key)
            _validate_expression(value)
    elif isinstance(node, ast.Subscript):
        _validate_expression(node.value)
        _validate_expression(node.slice)
    elif isinstance(node, ast.Slice):
        for value in (node.lower, node.upper, node.step):
            if value is not None:
                _validate_expression(value)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_FUNCTIONS:
                raise _fail(f".hnx function {node.func.id!r} is not allowed")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in {
                "strip", "trim", "lstrip", "rstrip", "lower", "upper",
                "toLowerCase", "toUpperCase", "casefold", "title", "replace",
                "startswith", "startsWith", "endswith", "endsWith", "includes",
                "contains", "split", "join", "count", "find", "get", "keys",
                "values", "items", "append", "extend", "index", "update",
            }:
                raise _fail(f".hnx method {node.func.attr!r} is not allowed")
            _validate_expression(node.func.value)
        else:
            raise _fail(".hnx calls must target a safe function or method")
        for value in node.args:
            _validate_expression(value)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise _fail(".hnx keyword expansion is not supported")
            _validate_expression(keyword.value)
    elif isinstance(node, ast.IfExp):
        _validate_expression(node.test)
        _validate_expression(node.body)
        _validate_expression(node.orelse)
    elif isinstance(node, ast.JoinedStr):
        for value in node.values:
            _validate_expression(value)
    elif isinstance(node, ast.FormattedValue):
        _validate_expression(node.value)


def _validate_statement(node: ast.stmt) -> None:
    if isinstance(node, (ast.Return, ast.Pass)):
        if isinstance(node, ast.Return) and node.value is not None:
            _validate_expression(node.value)
        return
    if isinstance(node, ast.Expr):
        _validate_expression(node.value)
        return
    if isinstance(node, ast.Assign):
        if not node.targets or any(not isinstance(target, ast.Name) for target in node.targets):
            raise _fail(".hnx assignments may target local names only")
        _validate_expression(node.value)
        return
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name) or node.value is None:
            raise _fail(".hnx typed assignments must assign a local name")
        _validate_expression(node.value)
        return
    if isinstance(node, ast.If):
        _validate_expression(node.test)
        for child in (*node.body, *node.orelse):
            _validate_statement(child)
        return
    raise _fail(f".hnx statement {type(node).__name__} is not supported")


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


def _method_call(target: Any, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    if name in ("trim", "strip") and isinstance(target, str):
        return target.strip(*args, **kwargs)
    if name == "lstrip" and isinstance(target, str):
        return target.lstrip(*args, **kwargs)
    if name == "rstrip" and isinstance(target, str):
        return target.rstrip(*args, **kwargs)
    if name in ("lower", "toLowerCase") and isinstance(target, str):
        return target.lower()
    if name in ("upper", "toUpperCase") and isinstance(target, str):
        return target.upper()
    if name == "casefold" and isinstance(target, str):
        return target.casefold()
    if name == "title" and isinstance(target, str):
        return target.title()
    if name in ("includes", "contains"):
        return args[0] in target
    if name in ("startswith", "startsWith") and isinstance(target, str):
        return target.startswith(*args)
    if name in ("endswith", "endsWith") and isinstance(target, str):
        return target.endswith(*args)
    if name == "replace" and isinstance(target, str):
        return target.replace(*args)
    if name == "split" and isinstance(target, str):
        return target.split(*args)
    if name == "join" and isinstance(target, str):
        return target.join(*args)
    if name == "count":
        return target.count(*args)
    if name == "find" and isinstance(target, str):
        return target.find(*args)
    if name == "get" and isinstance(target, dict):
        return target.get(*args, **kwargs)
    if name == "keys" and isinstance(target, dict):
        return list(target.keys())
    if name == "values" and isinstance(target, dict):
        return list(target.values())
    if name == "items" and isinstance(target, dict):
        return list(target.items())
    if name == "append" and isinstance(target, list):
        return target.append(*args)
    if name == "extend" and isinstance(target, list):
        return target.extend(*args)
    if name == "index" and isinstance(target, list):
        return target.index(*args)
    if name == "update" and isinstance(target, dict):
        return target.update(*args, **kwargs)
    raise _fail(f".hnx method {name!r} is not available for this value")


def _evaluate(node: ast.expr, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise _fail(f".hnx name {node.id!r} is not defined")
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_evaluate(node.left, env), _evaluate(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_evaluate(node.operand, env))
    if isinstance(node, ast.BoolOp):
        value = _evaluate(node.values[0], env)
        for child in node.values[1:]:
            if isinstance(node.op, ast.And) and not value:
                return value
            if isinstance(node.op, ast.Or) and value:
                return value
            value = _evaluate(child, env)
        return value
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, env)
        for op, expression in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(expression, env)
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.List):
        return [_evaluate(value, env) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(value, env) for value in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _evaluate(key, env): _evaluate(value, env)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Subscript):
        return _evaluate(node.value, env)[_evaluate(node.slice, env)]
    if isinstance(node, ast.Slice):
        return slice(
            _evaluate(node.lower, env) if node.lower else None,
            _evaluate(node.upper, env) if node.upper else None,
            _evaluate(node.step, env) if node.step else None,
        )
    if isinstance(node, ast.Call):
        args = [_evaluate(value, env) for value in node.args]
        kwargs = {keyword.arg: _evaluate(keyword.value, env) for keyword in node.keywords}
        if isinstance(node.func, ast.Name):
            return _SAFE_FUNCTIONS[node.func.id](*args, **kwargs)
        return _method_call(_evaluate(node.func.value, env), node.func.attr, args, kwargs)
    if isinstance(node, ast.IfExp):
        branch = node.body if _evaluate(node.test, env) else node.orelse
        return _evaluate(branch, env)
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            pieces.append(value.value if isinstance(value, ast.Constant)
                          else str(_evaluate(value, env)))
        return "".join(pieces)
    if isinstance(node, ast.FormattedValue):
        return str(_evaluate(node.value, env))
    raise _fail(f".hnx expression {type(node).__name__} cannot run")


def _execute(statements: list[ast.stmt], env: dict[str, Any]) -> None:
    for node in statements:
        if isinstance(node, ast.Return):
            raise _Return(_evaluate(node.value, env) if node.value else None)
        if isinstance(node, ast.Pass):
            continue
        if isinstance(node, ast.Assign):
            value = _evaluate(node.value, env)
            for target in node.targets:
                env[target.id] = value
            continue
        if isinstance(node, ast.AnnAssign):
            env[node.target.id] = _evaluate(node.value, env)
            continue
        if isinstance(node, ast.If):
            _execute(node.body if _evaluate(node.test, env) else node.orelse, env)
            continue
        if isinstance(node, ast.Expr):
            _evaluate(node.value, env)
            continue
        raise _fail(f".hnx statement {type(node).__name__} cannot run")


class HNXProgram:
    """Parsed .hnx hooks and JSONL data records."""

    def __init__(self, functions: dict[str, ast.FunctionDef], data: dict[str, Any]) -> None:
        self.functions = functions
        self.data = data

    @property
    def hooks(self) -> tuple[str, ...]:
        return tuple(self.functions)

    def call(self, hook: str, *args: Any) -> Any:
        function = self.functions.get(hook)
        if function is None:
            return None
        parameters = [*function.args.posonlyargs, *function.args.args]
        if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            raise _fail(f"{hook} uses unsupported variadic parameters")
        if len(parameters) != len(args):
            raise _fail(f"{hook} expects {len(parameters)} argument(s), received {len(args)}")
        env = {parameter.arg: value for parameter, value in zip(parameters, args, strict=True)}
        env["data"] = dict(self.data)
        try:
            _execute(function.body, env)
        except _Return as result:
            return result.value
        return None


def parse_hnx(path: Path | str) -> tuple[ElementSpec, HNXProgram]:
    """Read a v1.1.0 .hnx file, validating JSONL metadata and hook syntax."""
    source_path = Path(path)
    lines = source_path.read_text(encoding="utf-8").splitlines()
    meaningful = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    if not meaningful or not meaningful[0][1].startswith("#!hnx "):
        raise _fail(f"{source_path.name}: first non-empty line must be #!hnx 1.1.0")
    version = meaningful[0][1].split(maxsplit=1)[1]
    if version != ELEMENT_FORMAT_VERSION:
        raise _fail(
            f"{source_path.name}: format {version!r} is unsupported; expected {ELEMENT_FORMAT_VERSION}"
        )

    records: list[dict[str, Any]] = []
    code_start = meaningful[0][0] + 1
    for index in range(code_start, len(lines)):
        line = lines[index].strip()
        if not line or line.startswith("#") or line.startswith("//"):
            code_start = index + 1
            continue
        if not line.startswith("{"):
            code_start = index
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _fail(f"{source_path.name}:{index + 1}: invalid JSONL: {exc}") from exc
        if not isinstance(record, dict):
            raise _fail(f"{source_path.name}:{index + 1}: JSONL records must be objects")
        records.append(record)
        code_start = index + 1

    elements = [record for record in records if record.get("kind") == "element"]
    if len(elements) != 1:
        raise _fail(f"{source_path.name}: expected exactly one kind=element JSONL record")
    metadata = elements[0]
    if "dependencies" not in metadata:
        raise _fail(f"{source_path.name}: v1.1.0 records must declare dependencies (use [])")
    dependencies = metadata.get("dependencies")
    groups = metadata.get("groups", [])
    permissions = metadata.get("permissions", [])
    if not all(isinstance(value, list) for value in (dependencies, groups, permissions)):
        raise _fail(f"{source_path.name}: dependencies, groups, and permissions must be arrays")
    if not all(isinstance(value, str) for value in (*dependencies, *groups, *permissions)):
        raise _fail(f"{source_path.name}: dependency, group and permission names must be strings")

    data: dict[str, Any] = {}
    for record in records:
        if record is metadata:
            continue
        if record.get("kind") != "data" or not isinstance(record.get("name"), str):
            raise _fail(f"{source_path.name}: unknown JSONL record kind {record.get('kind')!r}")
        data[record["name"]] = record.get("value")

    python_source = _normalise_script("\n".join(lines[code_start:]))
    try:
        tree = ast.parse(python_source, filename=str(source_path), mode="exec")
    except SyntaxError as exc:
        raise _fail(f"{source_path.name}:{exc.lineno}: invalid .hnx syntax: {exc.msg}") from exc

    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            raise _fail(f"{source_path.name}: top-level code must define hook functions")
        if node.name not in _HOOKS:
            raise _fail(f"{source_path.name}: hook {node.name!r} is not supported")
        if node.name in functions:
            raise _fail(f"{source_path.name}: hook {node.name!r} is defined more than once")
        if node.decorator_list or node.args.defaults or node.args.kw_defaults:
            raise _fail(f"{source_path.name}: decorators/default parameters are not supported")
        for statement in node.body:
            _validate_statement(statement)
        functions[node.name] = node
    if not functions:
        raise _fail(f"{source_path.name}: define at least one supported hook")

    spec = ElementSpec(
        symbol=metadata.get("symbol", ""),
        summary=metadata.get("summary", ""),
        version=str(metadata.get("version", "0.1.0")),
        permissions=frozenset(permissions),
        user=True,
        format_version=version,
        dependencies=tuple(dependencies),
        groups=tuple(groups),
    )
    return spec, HNXProgram(functions, data)


def _element_class(spec: ElementSpec, program: HNXProgram) -> type[Element]:
    namespace: dict[str, Any] = {
        "spec": spec,
        "_hnx_program": program,
        "__module__": "hypernix.elements.user_hnx",
    }
    for hook in program.hooks:
        def bind(name: str):
            def run(self, *args: Any) -> Any:
                return self._hnx_program.call(name, *args)
            run.__name__ = name
            return run
        namespace[hook] = bind(hook)
    class_name = "HNX_" + spec.name.title().replace("_", "")
    return type(class_name, (Element,), namespace)


def load_hnx_elements(registry: Registry, directory: Path | str) -> list[str]:
    """Load .hnx files beneath a directory into a Hydrogen registry."""
    root = Path(directory)
    loaded: list[str] = []
    if not root.is_dir():
        return loaded
    for path in sorted(root.rglob("*.hnx")):
        if path.name.startswith("_") or any(
            part.startswith(".") for part in path.relative_to(root).parts
        ):
            continue
        source_key = str(path.resolve())
        if source_key in registry.sources:
            loaded.extend(registry.sources[source_key])
            continue
        try:
            spec, program = parse_hnx(path)
            registry.register(_element_class(spec, program))
            loaded.append(spec.symbol)
            registry.sources[source_key] = [spec.symbol]
        except Exception as exc:  # noqa: BLE001 - a bad script must not block its neighbours
            registry.failures[path.stem] = str(exc)
    return loaded


def scaffold_hnx(
    symbol: str,
    directory: Path | str | None = None,
    *,
    registry: Registry | None = None,
) -> Path:
    """Create an editable v1.1.0 .hnx file without overwriting existing work."""
    from .hydrogen import atomic_number, name_of

    atomic_number(symbol)
    if registry is not None:
        try:
            held = registry.lookup(symbol)
        except HyperNixError:
            held = None
        if held is not None and not held.spec.user:
            raise HyperNixError(
                codes.ELEMENT_CONFLICT,
                f"{symbol} is the built-in {held.spec.name}; pick another symbol",
            )
    root = Path(directory) if directory is not None else Path.home() / ".hypernix" / "elements"
    target = root / f"{name_of(symbol)}.hnx"
    if target.exists():
        raise HyperNixError(codes.ELEMENT_CONFLICT, f"{target} already exists; not overwriting")
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "element", "symbol": symbol,
        "summary": f"User-authored {name_of(symbol)} element.",
        "version": "0.1.0", "dependencies": [],
        "groups": ["chat"], "permissions": ["oven"],
    }
    target.write_text(
        "#!hnx 1.1.0\n"
        + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + "fn transform_prompt(prompt: string) -> string {\n"
        + "    let cleaned: string = prompt.trim();\n"
        + "    return cleaned;\n"
        + "}\n",
        encoding="utf-8",
    )
    return target
