#!/usr/bin/env python3
"""Lay the kernel out as kernel-builder would, without Nix.

For tests and local development only: releases are built and uploaded
by ``kernel-builder build-and-upload`` (see ``.github/workflows/
llama-essir-kernel.yml``). The output is the same shape kernel-builder
produces for a ``torch-noarch`` kernel -- ``build/torch-<backend>/``
with the package, a generated ``_ops.py``, a ``metadata.json`` and the
compatibility package -- so ``kernels.get_local_kernel(<dir>)`` and
``LOCAL_KERNELS=ray0rf1re/llama-essir=<dir>`` load it like the Hub copy.

    python native/llama-essir/local_build.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# kernel-builder's noarch _ops.py, with the template filled in. The unique
# id keeps two loaded versions apart; kernel-builder derives it from the
# git commit, a local build only needs it to differ from a Hub build's.
_OPS = '''import torch

def get_backend() -> str:
    """Detect the backend by inspecting torch."""
    import torch

    if hasattr(torch.backends, "tpu"):
        return "tpu"
    elif hasattr(torch, "neuron"):
        return "neuron"
    elif torch.version.cuda is not None:
        return "cuda"
    elif torch.version.hip is not None:
        return "rocm"
    elif torch.backends.mps.is_available():
        return "metal"
    elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
        return "xpu"
    else:
        return "cpu"


def _find_ops_name() -> str:
    kernel_name = "{kernel_name}"
    unique_id = "{unique_id}"
    backend = get_backend()
    return f"_{{kernel_name}}_{{backend}}_{{unique_id}}"


_OPS_NAME = _find_ops_name()

ops = getattr(torch.ops, _OPS_NAME)

def add_op_namespace_prefix(op_name: str) -> str:
    """
    Prefix op by namespace.
    """
    return f"{{_OPS_NAME}}::{{op_name}}"
'''


def build(out: Path, unique_id: str = "local") -> list[Path]:
    config = tomllib.loads((HERE / "build.toml").read_text(encoding="utf-8"))
    general = config["general"]
    name: str = general["name"]
    module = name.replace("-", "_")
    source = HERE / "torch-ext" / module

    variants = []
    for backend in sorted(general["backends"]):
        variant = out / "build" / f"torch-{backend}"
        if variant.exists():
            shutil.rmtree(variant)
        shutil.copytree(source, variant, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (variant / "_ops.py").write_text(
            _OPS.format(kernel_name=module, unique_id=unique_id), encoding="utf-8")
        metadata = {
            "name": name,
            "id": f"_{module}_{backend}_{unique_id}",
            "version": general["version"],
            "license": general["license"],
            "python-depends": [],
            "backend": {"type": backend},
        }
        (variant / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # The compatibility package older `kernels` releases import.
        compat = variant / module
        compat.mkdir()
        (compat / "__init__.py").write_text(
            "import importlib.util\nimport sys\nfrom pathlib import Path\n\n"
            "_parent = Path(__file__).parent.parent\n"
            "_spec = importlib.util.spec_from_file_location(\n"
            f"    '{module}_flattened', _parent / '__init__.py',\n"
            "    submodule_search_locations=[str(_parent)])\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "sys.modules[_spec.name] = _module\n"
            "_spec.loader.exec_module(_module)\n"
            "globals().update({k: getattr(_module, k) for k in _module.__all__})\n"
            "__all__ = list(_module.__all__)\n",
            encoding="utf-8")
        variants.append(variant)
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "result",
                        help="where to put build/ (default: native/llama-essir/result)")
    args = parser.parse_args(argv)
    for variant in build(args.out):
        print(variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
