"""hypernix — end-to-end toolkit for HyperNix-family PyTorch models.

The package grew out of a one-shot ``ray0rf1re/hyper-nix.1`` → GGUF
converter (v1 is still fully supported) and now covers the whole
lifecycle for the chat-tuned ``ray0rf1re/hyper-Nix.2`` (current
default) and every related model — from a blank directory to a
trained, quantized, uploaded HuggingFace snapshot:

* :mod:`hypernix.download` / :func:`download_model` — pull snapshots
  from the Hub; :data:`KNOWN_MODELS` resolves short names like
  ``"nano-mini"``, ``"qwen3.5-4b"``, ``"gemma-4-e4b"``, ``"nix2.5"``.
* :mod:`hypernix.train` — :class:`HyperNixConfig`, :class:`HyperNixModel`,
  :func:`init_from_scratch`, :func:`expand_checkpoint`, :func:`train`.
  Non-HyperNix architectures (Gemma, Phi, DeepSeek, GLM, GPT-OSS,
  Qwen3/3.5/3.6, Gemma 4, …) route through a thin
  ``transformers.AutoModelForCausalLM`` wrapper.
* :mod:`hypernix.old_oven` and :mod:`hypernix.neo_oven` — :class:`CodeOven`, :class:`NeoOven`, :func:`preheat`,
  :func:`new_oven`, plus the :data:`ARCH_PRESETS` seed-init models
  across the Llama / Qwen / Gemma / Phi / GLM / DeepSeek / Nemotron /
  GPT-OSS / Nix families.
* :mod:`hypernix.old_fridge`, :mod:`hypernix.mediocre_fridge`,
  :mod:`hypernix.new_fridge` — memory housekeeping
  (freeze / unfreeze / parameter_stats), judge-training dataset
  generation, and training-curve plotting.
* :mod:`hypernix.new_range`, :mod:`hypernix.old_range`,
  :mod:`hypernix.industrial_range` — labeling rubrics that drop in as
  ``label_rule=...`` for :func:`mediocre_fridge.collect_responses_from`.
  ``new_range`` is a zero-dep first-fail rubric, ``old_range`` is a
  weighted-mean scored rubric with explainability, and
  ``industrial_range`` is the LLM-as-judge wrapper.
* :mod:`hypernix.freezer` — :class:`OldFreezer` (8-10 GB) /
  :class:`NewFreezer` (11 GB+) / :class:`FlashFreezer` (OOM-safe retry
  wrapper) + Pascal (sm_61 / CUDA 6.1) helpers:
  :func:`pascal_safe_dtype`, :func:`is_pascal`, :func:`pascal_mode_hints`.
* :mod:`hypernix.convert` / :mod:`hypernix.quantize` /
  :mod:`hypernix.upload` — the original GGUF pipeline, still intact.

See :doc:`/README.md` for the headline quickstart and
``wiki/`` in the source tree for deep-dive topic guides.

Import speed
------------
Nothing below is imported eagerly. ``import hypernix`` only runs this
file, which does no submodule work at all — it just registers where
each public name *would* come from. The first time you touch
``hypernix.<something>`` (attribute access), :func:`__getattr__`
below imports the one submodule that defines it and caches the result
on the package, so every later access is a plain attribute lookup with
no import overhead. Submodules never touched are never imported, so
running e.g. the ``tvtop`` dashboard or ``hyped`` chat CLI never pays
for importing torch/transformers/gguf/huggingface_hub just because
``hypernix/__init__.py`` happened to run first (every submodule import
runs the parent package's ``__init__.py`` before it runs the submodule
itself). This replaces the previous ``sys.argv[0]``-sniffing fast path,
which only covered a handful of hardcoded entry-point names and did
nothing for library use (``import hypernix`` from a REPL or a user's
own script).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import TYPE_CHECKING, Any

__version__ = "0.72.4.dev1"
DEFAULT_REPO_ID = "ray0rf1re/hyper-Nix.2"
DEFAULT_MODEL = "qwen3.5-4b"  # New default model

__all__ = [
    "ARCH_PRESETS",
    "CATEGORY_OF",
    "MODULE_CATEGORIES",
    "Abbicus",
    "AbbicusConfig",
    "Agedcookerv4",
    "Agedcookerv5",
    "CardboardBox",
    "CookerLite",
    "TurboAbbicus",
    "TurboAbbicusConfig",
    "QAProcessor",
    "STML",
    "calculate_vram_context",
    "CPU_PRESETS",
    "CPUPreset",
    "CodeOven",
    "ComputeArch",
    "ComputeFramework",
    "GPU_PRESETS",
    "GPUPreset",
    "HyperNixConfig",
    "HealthReport",
    "HyperNixModel",
    "HyperNixQuantizer",
    "KNOWN_MODELS",
    "LazySusan",
    "LazySusanConfig",
    "ModelInfo",
    "PressureCookerV3",
    "PressureCookerV3Plus",
    "PressureCookerV4",
    "PressureCookerV5",
    "PressureCookerV5Plus",
    "PressureCookerV5S",
    "V5SConfig",
    "PressureCookerV6",
    "PressureCookerV6V",
    "QUANT_CATALOG",
    "QUANT_TYPES",
    "QuantConfig",
    "QuantDtype",
    "QuantJob",
    "QuantSpec",
    "RoundPlan",
    "StovetopV3Cooker",
    "StovetopV3CookerPlus",
    "StovetopV4Cooker",
    "StovetopV4CookerPlus",
    "Tupperware",
    "TupperwareConfig",
    "ULTRAagedcookerv4",
    "ULTRAagedcookerv5",
    "Ultracookerv4",
    "abbicus",
    "apron",
    "cardboard_box",
    "qa",
    "assistant",
    "nano_nano",
    "NanoNanoModel",
    "neo_oven",
    "NeoOven",
    "net",
    "bake_code",
    "bell",
    "blender",
    "cake_pan",
    "coffee_maker",
    "compactor",
    "compute_framework",
    "cookbook",
    "countertop",
    "cutting_board",
    "deep_fryer",
    "diagnostic_info",
    "dishwasher",
    "ethanol",
    "convert_to_gguf",
    "cpu_preset",
    "download_model",
    "espresso_maker",
    "expand_checkpoint",
    "fetch_llama_quantize",
    "fill_middle",
    "flour",
    "food_processor",
    "freezer",
    "generate_text",
    "gpu_preset",
    "healthcheck",
    "hyped",
    "industrial_range",
    "init_from_scratch",
    "injection",
    "instant_pot",
    "lazy_suzan",
    "list_models",
    "load_pt",
    "load_snapshot",
    "lunchbox",
    "mediocre_fridge",
    "menu",
    "microwave",
    "new_fridge",
    "new_oven",
    "new_range",
    "old_fridge",
    "old_oven",
    "old_range",
    "optimizer_framework",
    "outage",
    "pans",
    "pepper_shaker",
    "plasma",
    "pressure_cooker",
    "pressure_cooker_v3",
    "pressure_cooker_v4",
    "pressure_cooker_v5",
    "pressure_cooker_v5s",
    "pressurecooker_v5s",
    "pressure_cooker_v6",
    "pressure_cooker_v6v",
    "preheat",
    "print_models",
    "quant_batch",
    "quant_by_category",
    "quant_estimate_size",
    "quant_for_size",
    "quant_list_types",
    "quant_recommend_profile",
    "quant_recommended",
    "quant_resolve_spec",
    "quantize_gguf",
    "recipe_book",
    "resolve_model_info",
    "resolve_repo_id",
    "salt_shaker",
    "save_snapshot",
    "session_dir",
    "sink",
    "smoke_alarm",
    "smoker",
    "stml",
    "strainer",
    "table",
    "thermometer",
    "timer",
    "toaster",
    "torch_compat",
    "train",
    "tupperware",
    "tv",
    "tvtop",
    "tvtop_plus_plus",
    "ups",
    "upload_gguf",
    "verify_snapshot",
    "vram",
    "wakeup",
    "whisk",
    "workshop",
    "fizzle",
    "spinner",
    "hyper_log",
    # v0.71.0
    "gatekeeper",
    "keymaster",
    "gkey_cli",
    "Gatekeeper",
    "Keymaster",
    "T1KeyGenerator",
    "KeyType",
    "KeyScope",
    "KeyMeta",
    "Quota",
    "QuotaViolation",
    "websearch",
    "search_web_non_api",
]

# ---------------------------------------------------------------------------
# Module layout
# ---------------------------------------------------------------------------
# Modules live in a category subpackage (``hypernix/timing/timer.py``) rather
# than in one flat directory of 100+ files. This table is the single source of
# truth for that layout: the lazy loader below, the back-compat alias finder,
# and the ``scripts/autofix-*`` tooling all read it instead of hardcoding
# paths, so moving a module between categories is a one-line change here.
MODULE_CATEGORIES: dict[str, tuple[str, ...]] = {
    # Chat templating, prompt presets and multi-turn session state.
    "chat": (
        "cookbook",
        "countertop",
        "flour",
        "injection",
        "menu",
    ),
    # Datasets: collection, cleaning, splitting, packing and augmentation.
    "data": (
        "blender",
        "cardboard_box",
        "cutting_board",
        "food_processor",
        "lunchbox",
        "mediocre_fridge",
        "pans",
        "pepper_shaker",
        "qa",
        "salt_shaker",
        "scavenger",
        "sink",
        "strainer",
        "toaster",
        "tupperware",
    ),
    # Scoring, rubric labelling, judging and module verification.
    "evaluation": (
        "espresso_maker",
        "industrial_range",
        "new_fridge",
        "new_range",
        "old_range",
        "vera",
    ),
    # Human-facing front ends: CLIs, TUIs, GUIs and launchers.
    "interfaces": (
        "assistant",
        "cli",
        "hyped",
        "hyped_pro",
        "hyped_pro_bridge",
        "hyped_pro_core",
        "hyped_pro_gui",
        "hyped_pro_tools",
        "version_launcher",
        "websearch",
        "wiki_cli",
    ),
    # Architectures, snapshot loading, generation and model utilities.
    "models": (
        "arch",
        "download",
        "generate",
        "microwave",
        "multilama",
        "nano_nano",
        "neo_oven",
        "old_oven",
        "stml",
        "whisk",
        "workshop",
    ),
    # Live dashboards, logging, telemetry and hardware sampling.
    "monitoring": (
        "cctvtop",
        "hyper_log",
        "map",
        "plasma",
        "table",
        "thermometer",
        "tv",
        "tvtop",
        "tvtop_plus_plus",
    ),
    # The Pressure Cooker optimizer family and optimizer plumbing.
    "optimizers": (
        "optimizer_framework",
        "pressure_cooker",
        "pressure_cooker_v3",
        "pressure_cooker_v4",
        "pressure_cooker_v5",
        "pressure_cooker_v5s",
        "pressure_cooker_v6",
        "pressure_cooker_v6v",
    ),
    # The GGUF pipeline: convert, quantize, fetch tooling and upload.
    "quant": (
        "convert",
        "fetcher",
        "quantize",
        "upload",
    ),
    # API keys, quotas and request gating.
    "security": (
        "gatekeeper",
        "gkey_cli",
        "keymaster",
    ),
    # Environment, dependencies, hardware and housekeeping.
    "system": (
        "compactor",
        "config",
        "deps",
        "dishwasher",
        "doctor",
        "ethanol",
        "freezer",
        "net",
        "old_fridge",
        "outage",
        "pathfix",
        "protect",
        "release",
        "torch_compat",
        "ups",
        "utils",
        "vram",
    ),
    # Timers, alarms, cadence control and progress animation.
    "timing": (
        "bell",
        "coffee_maker",
        "smoke_alarm",
        "spinner",
        "timer",
    ),
    # Training entry points, schedules and weight perturbation.
    "training": (
        "abbicus",
        "apron",
        "brewer",
        "cake_pan",
        "camouflage",
        "compute_framework",
        "deep_fryer",
        "fizzle",
        "instant_pot",
        "lazy_suzan",
        "mtp",
        "recipe_book",
        "smoker",
        "train",
    ),
}

#: Reverse of :data:`MODULE_CATEGORIES` — module name to category name.
CATEGORY_OF: dict[str, str] = {
    mod: cat for cat, mods in MODULE_CATEGORIES.items() for mod in mods
}

# Every public name hypernix exposes, mapped to the one submodule that
# actually defines it (and, if it's a re-exported member rather than the
# submodule itself, the attribute name to pull off of it once imported).
# Generated from the import statements this file used to run eagerly at
# module load time — see scripts/regen_lazy_attrs.py to rebuild it if the
# public API changes.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    'ARCH_PRESETS': ('models.old_oven', 'ARCH_PRESETS'),
    'Abbicus': ('training.abbicus', 'Abbicus'),
    'AbbicusConfig': ('training.abbicus', 'AbbicusConfig'),
    'Agedcookerv4': ('optimizers.pressure_cooker_v4', 'Agedcookerv4'),
    'Agedcookerv5': ('optimizers.pressure_cooker_v5', 'Agedcookerv5'),
    'CPUPreset': ('system.freezer', 'CPUPreset'),
    'CPU_PRESETS': ('system.freezer', 'CPU_PRESETS'),
    'CardboardBox': ('data.cardboard_box', 'CardboardBox'),
    'CodeOven': ('models.old_oven', 'CodeOven'),
    'NanoNanoModel': ('models.nano_nano', 'NanoNanoModel'),
    'NeoOven': ('models.neo_oven', 'NeoOven'),
    'ComputeArch': ('training.compute_framework', 'ComputeArch'),
    'ComputeFramework': ('training.compute_framework', 'ComputeFramework'),
    'CookerLite': ('optimizers.pressure_cooker_v4', 'CookerLite'),
    'GPUPreset': ('system.freezer', 'GPUPreset'),
    'GPU_PRESETS': ('system.freezer', 'GPU_PRESETS'),
    'HealthReport': ('system.utils', 'HealthReport'),
    'HyperNixConfig': ('training.train', 'HyperNixConfig'),
    'HyperNixModel': ('training.train', 'HyperNixModel'),
    'HyperNixQuantizer': ('quant.quantize', 'HyperNixQuantizer'),
    'KNOWN_MODELS': ('models.download', 'KNOWN_MODELS'),
    'LazySusan': ('training.lazy_suzan', 'LazySusan'),
    'LazySusanConfig': ('training.lazy_suzan', 'LazySusanConfig'),
    'ModelInfo': ('models.download', 'ModelInfo'),
    'PressureCookerV3': ('optimizers.pressure_cooker_v3', 'PressureCookerV3'),
    'PressureCookerV3Plus': ('optimizers.pressure_cooker_v3', 'PressureCookerV3Plus'),
    'PressureCookerV4': ('optimizers.pressure_cooker_v4', 'PressureCookerV4'),
    'PressureCookerV5': ('optimizers.pressure_cooker_v5', 'PressureCookerV5'),
    'PressureCookerV5Plus': ('optimizers.pressure_cooker_v5', 'PressureCookerV5Plus'),
    'PressureCookerV5S': ('optimizers.pressure_cooker_v5s', 'PressureCookerV5S'),
    'V5SConfig': ('optimizers.pressure_cooker_v5s', 'V5SConfig'),
    'PressureCookerV6': ('optimizers.pressure_cooker_v6', 'PressureCookerV6'),
    'PressureCookerV6V': ('optimizers.pressure_cooker_v6v', 'PressureCookerV6V'),
    'QAProcessor': ('data.qa', 'QAProcessor'),
    'QUANT_CATALOG': ('quant.quantize', 'CATALOG'),
    'QUANT_TYPES': ('quant.quantize', 'QUANT_TYPES'),
    'QuantConfig': ('optimizers.pressure_cooker_v3', 'QuantConfig'),
    'QuantDtype': ('optimizers.pressure_cooker_v3', 'QuantDtype'),
    'QuantJob': ('quant.quantize', 'QuantJob'),
    'QuantSpec': ('quant.quantize', 'QuantSpec'),
    'RoundPlan': ('data.tupperware', 'RoundPlan'),
    'STML': ('models.stml', 'STML'),
    'StovetopV3Cooker': ('optimizers.pressure_cooker_v3', 'StovetopV3Cooker'),
    'StovetopV3CookerPlus': ('optimizers.pressure_cooker_v3', 'StovetopV3CookerPlus'),
    'StovetopV4Cooker': ('optimizers.pressure_cooker_v4', 'StovetopV4Cooker'),
    'StovetopV4CookerPlus': ('optimizers.pressure_cooker_v4', 'StovetopV4CookerPlus'),
    'Tupperware': ('data.tupperware', 'Tupperware'),
    'TupperwareConfig': ('data.tupperware', 'TupperwareConfig'),
    'TurboAbbicus': ('training.abbicus', 'TurboAbbicus'),
    'TurboAbbicusConfig': ('training.abbicus', 'TurboAbbicusConfig'),
    'ULTRAagedcookerv4': ('optimizers.pressure_cooker_v4', 'ULTRAagedcookerv4'),
    'ULTRAagedcookerv5': ('optimizers.pressure_cooker_v5', 'ULTRAagedcookerv5'),
    'Ultracookerv4': ('optimizers.pressure_cooker_v4', 'Ultracookerv4'),
    'abbicus': ('training.abbicus', None),
    'apron': ('training.apron', None),
    'assistant': ('interfaces.assistant', None),
    'bake_code': ('models.old_oven', 'bake_code'),
    'bell': ('timing.bell', None),
    'blender': ('data.blender', None),
    'cake_pan': ('training.cake_pan', None),
    'calculate_vram_context': ('models.stml', 'calculate_vram_context'),
    'cardboard_box': ('data.cardboard_box', None),
    'coffee_maker': ('timing.coffee_maker', None),
    'compactor': ('system.compactor', None),
    'compute_framework': ('training.compute_framework', None),
    'convert': ('quant.convert', None),
    'convert_to_gguf': ('quant.convert', 'convert_to_gguf'),
    'cookbook': ('chat.cookbook', None),
    'countertop': ('chat.countertop', None),
    'cpu_preset': ('system.freezer', 'cpu_preset'),
    'cutting_board': ('data.cutting_board', None),
    'deep_fryer': ('training.deep_fryer', None),
    'diagnostic_info': ('system.utils', 'diagnostic_info'),
    'dishwasher': ('system.dishwasher', None),
    'download': ('models.download', None),
    'download_model': ('models.download', 'download_model'),
    'espresso_maker': ('evaluation.espresso_maker', None),
    'ethanol': ('system.ethanol', None),
    'expand_checkpoint': ('training.train', 'expand_checkpoint'),
    'fetch_llama_quantize': ('quant.fetcher', 'fetch_llama_quantize'),
    'fetcher': ('quant.fetcher', None),
    'fill_middle': ('models.old_oven', 'fill_middle'),
    'fizzle': ('training.fizzle', None),
    'flour': ('chat.flour', None),
    'food_processor': ('data.food_processor', None),
    'freezer': ('system.freezer', None),
    'generate': ('models.generate', None),
    'generate_text': ('models.generate', 'generate_text'),
    'gpu_preset': ('system.freezer', 'gpu_preset'),
    'healthcheck': ('system.utils', 'healthcheck'),
    'hyped': ('interfaces.hyped', None),
    'hyper_log': ('monitoring.hyper_log', None),
    'industrial_range': ('evaluation.industrial_range', None),
    'init_from_scratch': ('training.train', 'init_from_scratch'),
    'injection': ('chat.injection', None),
    'instant_pot': ('training.instant_pot', None),
    'lazy_suzan': ('training.lazy_suzan', None),
    'list_models': ('system.utils', 'list_models'),
    'load_pt': ('models.old_oven', 'load_pt'),
    'load_snapshot': ('training.train', 'load_snapshot'),
    'lunchbox': ('data.lunchbox', None),
    'mediocre_fridge': ('data.mediocre_fridge', None),
    'menu': ('chat.menu', None),
    'microwave': ('models.microwave', None),
    'nano_nano': ('models.nano_nano', None),
    'neo_oven': ('models.neo_oven', None),
    'net': ('system.net', None),
    'new_fridge': ('evaluation.new_fridge', None),
    'new_oven': ('models.old_oven', 'new_oven'),
    'new_range': ('evaluation.new_range', None),
    'old_fridge': ('system.old_fridge', None),
    'old_oven': ('models.old_oven', None),
    'old_range': ('evaluation.old_range', None),
    'optimizer_framework': ('optimizers.optimizer_framework', None),
    'outage': ('system.outage', None),
    'pans': ('data.pans', None),
    'pepper_shaker': ('data.pepper_shaker', None),
    'plasma': ('monitoring.plasma', None),
    'preheat': ('models.old_oven', 'preheat'),
    'pressure_cooker': ('optimizers.pressure_cooker', None),
    'pressure_cooker_v3': ('optimizers.pressure_cooker_v3', None),
    'pressure_cooker_v4': ('optimizers.pressure_cooker_v4', None),
    'pressure_cooker_v5': ('optimizers.pressure_cooker_v5', None),
    'pressure_cooker_v5s': ('optimizers.pressure_cooker_v5s', None),
    'pressurecooker_v5s': ('optimizers.pressure_cooker_v5s', None),
    'pressure_cooker_v6': ('optimizers.pressure_cooker_v6', None),
    'pressure_cooker_v6v': ('optimizers.pressure_cooker_v6v', None),
    'print_models': ('system.utils', 'print_models'),
    'qa': ('data.qa', None),
    'quant_batch': ('quant.quantize', 'batch_quantize'),
    'quant_by_category': ('quant.quantize', 'by_category'),
    'quant_estimate_size': ('quant.quantize', 'estimate_size'),
    'quant_for_size': ('quant.quantize', 'for_size'),
    'quant_list_types': ('quant.quantize', 'list_types'),
    'quant_recommend_profile': ('quant.quantize', 'recommend_profile'),
    'quant_recommended': ('quant.quantize', 'recommended'),
    'quant_resolve_spec': ('quant.quantize', 'resolve_spec'),
    'quantize': ('quant.quantize', None),
    'quantize_gguf': ('quant.quantize', 'quantize_gguf'),
    'recipe_book': ('training.recipe_book', None),
    'resolve_model_info': ('models.download', 'resolve_model_info'),
    'resolve_repo_id': ('models.download', 'resolve_repo_id'),
    'salt_shaker': ('data.salt_shaker', None),
    'save_snapshot': ('training.train', 'save_snapshot'),
    'session_dir': ('system.utils', 'session_dir'),
    'sink': ('data.sink', None),
    'smoke_alarm': ('timing.smoke_alarm', None),
    'smoker': ('training.smoker', None),
    'spinner': ('timing.spinner', None),
    'stml': ('models.stml', None),
    'strainer': ('data.strainer', None),
    'table': ('monitoring.table', None),
    'thermometer': ('monitoring.thermometer', None),
    'timer': ('timing.timer', None),
    'toaster': ('data.toaster', None),
    'torch_compat': ('system.torch_compat', None),
    'train': ('training.train', 'train'),
    'tupperware': ('data.tupperware', None),
    'tv': ('monitoring.tv', None),
    'tvtop': ('monitoring.tvtop', None),
    'tvtop_plus_plus': ('monitoring.tvtop_plus_plus', None),
    'upload': ('quant.upload', None),
    'upload_gguf': ('quant.upload', 'upload_gguf'),
    'ups': ('system.ups', None),
    'utils': ('system.utils', None),
    'verify_snapshot': ('models.download', 'verify_snapshot'),
    'vram': ('system.vram', None),
    'wakeup': ('audio.wakeup', None),
    'whisk': ('models.whisk', None),
    'workshop': ('models.workshop', None),
    # v0.71.0 — security & usage management
    'gatekeeper': ('security.gatekeeper', None),
    'keymaster': ('security.keymaster', None),
    'gkey_cli': ('security.gkey_cli', None),
    'Gatekeeper': ('security.gatekeeper', 'Gatekeeper'),
    'Keymaster': ('security.keymaster', 'Keymaster'),
    'T1KeyGenerator': ('security.keymaster', 'T1KeyGenerator'),
    'KeyType': ('security.keymaster', 'KeyType'),
    'KeyScope': ('security.keymaster', 'KeyScope'),
    'KeyMeta': ('security.keymaster', 'KeyMeta'),
    'Quota': ('security.gatekeeper', 'Quota'),
    'QuotaViolation': ('security.gatekeeper', 'QuotaViolation'),
    'websearch': ('interfaces.websearch', None),
    # v0.71.5post2 — the category packages themselves.
    'chat': ('chat', None),
    'data': ('data', None),
    'evaluation': ('evaluation', None),
    'interfaces': ('interfaces', None),
    'models': ('models', None),
    'monitoring': ('monitoring', None),
    'optimizers': ('optimizers', None),
    'quant': ('quant', None),
    'security': ('security', None),
    'system': ('system', None),
    'timing': ('timing', None),
    'training': ('training', None),
    'search_web_non_api': ('interfaces.websearch', 'search_web_non_api'),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loader.

    Only ever imports the single submodule that defines ``name``, the
    first time it's actually asked for, then caches the result directly
    on the package so subsequent lookups are a normal attribute access
    (no repeated importlib overhead, no re-running this function).
    """
    try:
        submodule, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f".{submodule}", __name__)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


# ---------------------------------------------------------------------------
# Back-compat: ``import hypernix.timer`` still works
# ---------------------------------------------------------------------------
# Modules moved into category packages in v0.71.5post2. Every one of them is
# still importable under the flat name it has always had — ``hypernix.timer``
# resolves to the very same module object as ``hypernix.timing.timer``, so
# ``is`` comparisons, ``isinstance`` checks and monkeypatching keep working
# across both spellings. The finder only ever claims names it knows about;
# anything else (``hypernix.t1api``, ``hypernix.cctvtop_ext``, a typo) falls
# through to the normal import machinery.


class _FlatAliasLoader:
    """Loader that hands back the module from its real, categorised home."""

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec: Any) -> Any:
        return importlib.import_module(self._real_name)

    def exec_module(self, module: Any) -> None:
        """No-op: :meth:`create_module` returned an already-executed module."""

    # ``runpy`` (``python -m hypernix.cli``), ``inspect.getsource`` and
    # ``linecache`` don't go through create_module — they ask the loader for
    # the code/source directly. Delegate to the real module's loader, passing
    # it the real dotted name: ``FileLoader`` methods are name-checked and
    # would reject the flat alias.
    def _real_loader(self) -> Any:
        module = importlib.import_module(self._real_name)
        return module.__spec__.loader

    def get_code(self, fullname: str | None = None) -> Any:
        return self._real_loader().get_code(self._real_name)

    def get_source(self, fullname: str | None = None) -> str | None:
        return self._real_loader().get_source(self._real_name)

    def get_filename(self, fullname: str | None = None) -> str:
        return self._real_loader().get_filename(self._real_name)

    def is_package(self, fullname: str | None = None) -> bool:
        return False  # only modules are aliased, never packages

    def __repr__(self) -> str:
        return f"<_FlatAliasLoader for {self._real_name!r}>"


class _FlatAliasFinder:
    """Meta-path finder for the pre-v0.71.5post2 flat module names."""

    _prefix = __name__ + "."

    @classmethod
    def find_spec(cls, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not fullname.startswith(cls._prefix):
            return None
        flat = fullname[len(cls._prefix):]
        category = CATEGORY_OF.get(flat)
        if category is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _FlatAliasLoader(f"{__name__}.{category}.{flat}"),
            origin=f"{__name__}.{category}.{flat}",
        )

    def __repr__(self) -> str:
        return "<_FlatAliasFinder>"


def _install_flat_alias_finder() -> None:
    """Register the finder once, even if this module is reloaded."""
    for existing in sys.meta_path:
        if getattr(existing, "__name__", None) == "_FlatAliasFinder":
            return
    sys.meta_path.append(_FlatAliasFinder)


_install_flat_alias_finder()

if TYPE_CHECKING:
    # Not executed at runtime (TYPE_CHECKING is always False), so this
    # costs nothing at import time — it exists purely so IDEs/type
    # checkers still see real symbols instead of "Any" for every
    # hypernix.* access. Keep this in sync with _LAZY_ATTRS above.
    # v0.71.0
    from .chat import (
        cookbook,
        countertop,
        flour,
        injection,
        menu,
    )
    from .data import (
        blender,
        cardboard_box,
        cutting_board,
        food_processor,
        lunchbox,
        mediocre_fridge,
        pans,
        pepper_shaker,
        qa,
        salt_shaker,
        sink,
        strainer,
        toaster,
        tupperware,
    )
    from .data.cardboard_box import CardboardBox
    from .data.qa import QAProcessor
    from .data.tupperware import RoundPlan, Tupperware, TupperwareConfig
    from .evaluation import (
        espresso_maker,
        industrial_range,
        new_fridge,
        new_range,
        old_range,
    )
    from .interfaces import (
        assistant,
        hyped,
        websearch,
    )
    from .interfaces.websearch import search_web_non_api
    from .models import (
        microwave,
        nano_nano,
        neo_oven,
        old_oven,
        stml,
        whisk,
        workshop,
    )
    from .models.download import (
        KNOWN_MODELS,
        ModelInfo,
        download_model,
        resolve_model_info,
        resolve_repo_id,
        verify_snapshot,
    )
    from .models.generate import generate_text
    from .models.nano_nano import NanoNanoModel
    from .models.neo_oven import NeoOven
    from .models.old_oven import (
        ARCH_PRESETS,
        CodeOven,
        bake_code,
        fill_middle,
        load_pt,
        new_oven,
        preheat,
    )
    from .models.stml import STML, calculate_vram_context
    from .monitoring import (
        hyper_log,
        plasma,
        table,
        thermometer,
        tv,
        tvtop,
        tvtop_plus_plus,
    )
    from .optimizers import (
        optimizer_framework,
        pressure_cooker,
        pressure_cooker_v3,
        pressure_cooker_v4,
        pressure_cooker_v5,
        pressure_cooker_v5s,
        pressure_cooker_v6,
        pressure_cooker_v6v,
    )

    # Historical misspelling kept as a public alias.
    from .optimizers import pressure_cooker_v5s as pressurecooker_v5s
    from .optimizers.pressure_cooker_v3 import (
        PressureCookerV3,
        PressureCookerV3Plus,
        QuantConfig,
        QuantDtype,
        StovetopV3Cooker,
        StovetopV3CookerPlus,
    )
    from .optimizers.pressure_cooker_v4 import (
        Agedcookerv4,
        CookerLite,
        PressureCookerV4,
        StovetopV4Cooker,
        StovetopV4CookerPlus,
        ULTRAagedcookerv4,
        Ultracookerv4,
    )
    from .optimizers.pressure_cooker_v5 import (
        Agedcookerv5,
        PressureCookerV5,
        PressureCookerV5Plus,
        ULTRAagedcookerv5,
    )
    from .optimizers.pressure_cooker_v5s import (
        PressureCookerV5S,
        V5SConfig,
    )
    from .optimizers.pressure_cooker_v6 import PressureCookerV6
    from .optimizers.pressure_cooker_v6v import PressureCookerV6V
    from .quant.convert import convert_to_gguf
    from .quant.fetcher import fetch_llama_quantize
    from .quant.quantize import CATALOG as QUANT_CATALOG  # noqa: I001
    from .quant.quantize import QUANT_TYPES, HyperNixQuantizer, QuantJob, QuantSpec, quantize_gguf
    from .quant.quantize import batch_quantize as quant_batch
    from .quant.quantize import by_category as quant_by_category
    from .quant.quantize import estimate_size as quant_estimate_size
    from .quant.quantize import for_size as quant_for_size
    from .quant.quantize import list_types as quant_list_types
    from .quant.quantize import recommend_profile as quant_recommend_profile
    from .quant.quantize import recommended as quant_recommended
    from .quant.quantize import resolve_spec as quant_resolve_spec
    from .quant.upload import upload_gguf
    from .security import (
        gatekeeper,
        gkey_cli,
        keymaster,
    )
    from .security.gatekeeper import Gatekeeper, Quota, QuotaViolation
    from .security.keymaster import Keymaster, KeyMeta, KeyScope, KeyType, T1KeyGenerator
    from .system import (
        compactor,
        dishwasher,
        ethanol,
        freezer,
        net,
        old_fridge,
        outage,
        torch_compat,
        ups,
    )
    from .system.freezer import (
        CPU_PRESETS,
        GPU_PRESETS,
        CPUPreset,
        GPUPreset,
        cpu_preset,
        gpu_preset,
    )
    from .system.utils import (
        HealthReport,
        diagnostic_info,
        healthcheck,
        list_models,
        print_models,
        session_dir,
    )
    from .timing import (
        bell,
        coffee_maker,
        smoke_alarm,
        spinner,
        timer,
    )
    from .training import (
        abbicus,
        apron,
        compute_framework,
        deep_fryer,
        fizzle,
        instant_pot,
        lazy_suzan,
        recipe_book,
        smoker,
    )
    from .training.abbicus import Abbicus, AbbicusConfig, TurboAbbicus, TurboAbbicusConfig
    from .training.compute_framework import ComputeArch, ComputeFramework
    from .training.lazy_suzan import LazySusan, LazySusanConfig
    from .training.train import (
        HyperNixConfig,
        HyperNixModel,
        expand_checkpoint,
        init_from_scratch,
        load_snapshot,
        save_snapshot,
        train,
    )
