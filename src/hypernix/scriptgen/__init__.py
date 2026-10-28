"""hypernix.scriptgen — build training scripts, in a GUI or from the CLI.

    hnx scriptgen                     open the GUI (the wizard, headless)
    hnx scriptgen --wizard            five questions in the terminal
    hnx scriptgen train.py            generate headless

Sixty-two parameters across seven groups, modelled on the nano-nano 5.1
trainer's surface: learning rates and warmup ratios, epoch and step
controls, batch and micro-batch sizing, gradient accumulation, loss
functions and optimisers — plus the HyperNix-specific parts (the
Pressure Cooker family, 6-bit momentum modes, and the Pascal
auto-tuner), and since 0.72.8rc1 LoRA adapters, data, device and
reporting options, and :mod:`~hypernix.scriptgen.recipes` to start from.
The generated script runs as written.

The modules, and the split matters:

* :mod:`~hypernix.scriptgen.params` — one definition per parameter, used
  to build the widget, validate the value, and emit the script. Defining
  them separately is how a GUI offers a learning rate the generated
  script rejects.
* :mod:`~hypernix.scriptgen.templates` — generate a whole script, inject
  a config block into an existing one, or emit just the values.
* :mod:`~hypernix.scriptgen.theme` — dark slate, charcoal, obsidian and
  HyperNix red, with the "no purple" rule and WCAG contrast enforced by
  :func:`~hypernix.scriptgen.theme.audit_palette` rather than by taste.
* :mod:`~hypernix.scriptgen.widgets` — the dual slider, log slider and
  toggle switch Tk does not have.

:class:`~hypernix.scriptgen.app.FormModel` holds the values, validation
and preview and imports no GUI toolkit, so all of that is testable on a
headless machine — which is also the machine the training usually runs
on.
"""
from __future__ import annotations

from .app import FormModel, launch
from .params import (
    ALL_PARAMS,
    GROUPS,
    Param,
    ParamGroup,
    ParamKind,
    defaults,
    search,
    suggest,
    validate_all,
)
from .recipes import RECIPES, Recipe, apply_recipe
from .templates import CONFIG_FORMATS, as_config, generate, inject
from .theme import FONTS, PALETTE, Palette, audit_palette, contrast_ratio

__scriptgen_version__ = "0.72.8rc1"

__all__ = [
    "__scriptgen_version__",
    "FormModel", "launch",
    "Param", "ParamGroup", "ParamKind", "GROUPS", "ALL_PARAMS",
    "defaults", "validate_all", "search", "suggest",
    "RECIPES", "Recipe", "apply_recipe",
    "generate", "inject", "as_config", "CONFIG_FORMATS",
    "PALETTE", "Palette", "FONTS", "audit_palette", "contrast_ratio",
]
