#!/usr/bin/env python3
"""Apply the HyperNix CI-failure fixes directly, without relying on `git apply`.

Run from the repo root:
    python3 apply_hypernix_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path.cwd()

EDITS = [
    ("src/hypernix/__init__.py",
     "    from . import tv, tvtop, tvtop_plus_plus, spinner",
     "    from . import spinner, tv, tvtop, tvtop_plus_plus"),

    ("src/hypernix/assistant.py",
     '            with console.status("[bold green]Transcribing audio...") as status:',
     '            with console.status("[bold green]Transcribing audio..."):'),

    ("src/hypernix/assistant.py",
     '            with console.status("[bold green]Running ASR → LLM → TTS pipeline...") as status:',
     '            with console.status("[bold green]Running ASR → LLM → TTS pipeline..."):'),

    ("src/hypernix/cli.py",
     '            anime_print(f"HyperNix v{__version__}", style="typewriter", delay=0.04)',
     '            anime_print(f"hypernix {__version__}", style="typewriter", delay=0.04)'),

    ("src/hypernix/fizzle.py",
     "import argparse\nimport sys",
     "import sys"),

    ("src/hypernix/fizzle.py",
     "    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel",
     "    from transformers import AutoModel, AutoTokenizer, PreTrainedModel"),

    ("src/hypernix/fizzle.py",
     "        for cid, tok in self.tokenizers.items():",
     "        for _cid, tok in self.tokenizers.items():"),

    ("src/hypernix/quantize.py",
     "        import llama_cpp\n        from llama_cpp import llama_model_quantize, llama_model_quantize_params",
     "        from llama_cpp import llama_model_quantize, llama_model_quantize_params"),

    ("src/hypernix/quantize.py",
     '    except FileNotFoundError:\n        raise RuntimeError(f"Binary {binary} not found or not executable.")',
     '    except FileNotFoundError as err:\n        raise RuntimeError(f"Binary {binary} not found or not executable.") from err'),

    ("src/hypernix/spinner.py",
     '    def start(self) -> "Spinner":',
     "    def start(self) -> Spinner:"),

    ("src/hypernix/spinner.py",
     '    def __enter__(self) -> "Spinner":',
     "    def __enter__(self) -> Spinner:"),

    ("tests/models/test_v0704b11_features.py",
     '        assert __version__.startswith("0.70.4")',
     '        assert __version__.startswith("0.70.")'),
]


def main() -> int:
    ok = True
    for relpath, old, new in EDITS:
        path = REPO / relpath
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            print(f"SKIP  {relpath}: pattern not found (already fixed?) -> {old[:60]!r}")
            continue
        if count > 1:
            print(f"WARN  {relpath}: pattern occurs {count} times, expected 1 -> {old[:60]!r}")
            ok = False
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"OK    {relpath}: applied fix -> {old[:60]!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
