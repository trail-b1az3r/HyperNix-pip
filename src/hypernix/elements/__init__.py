"""hypernix.elements — addons named after the periodic table.

* :mod:`.hydrogen` — the framework every element is built on (H, 1)
* :mod:`.natural_gas` — attaches elements to a Neo Oven as addons
* :mod:`.carbon` — quality of life, and user-made elements (C, 6)
* :mod:`.magnesium` — keeps other apps out of the model's way (Mg, 12)
* :mod:`.isotope_3h` — the 3H (tritium) scripting runtime for .hnx elements

3H is an isotope/runtime extension of Hydrogen, not a 119th periodic-table
entry. Its editable v1.1.0 .hnx files combine JSONL metadata with a restricted
Python/TypeScript/C++-inspired language interpreted without native compilation.

Periods 6 and 7 — atomic numbers 55 to 118 — are experimental. That is
decided by the atomic number, not by a list, so an element added there
later is gated without anybody remembering to gate it.
"""
from __future__ import annotations
