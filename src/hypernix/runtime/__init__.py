"""hypernix.runtime — running models, and letting them use tools.

* :mod:`.hub_kernels` — Llama models on the llama-easy Hugging Face Hub kernel
* :mod:`.relay` — prompts longer than a model's context, read by small-context instances in series
* :mod:`.toolcalls` — reading a tool call out of whatever a model wrote
* :mod:`.t1tools` — the T1 API's endpoints, offered to a model as tools
"""
from __future__ import annotations
