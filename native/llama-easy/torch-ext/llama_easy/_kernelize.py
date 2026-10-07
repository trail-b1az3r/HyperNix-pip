"""``kernelize(model)``: the one-call way in, needing nothing but torch.

The ``kernels`` library's ``kernelize`` wants a mapping that names this
repository for every hook and device. This does the same swap for the
layers this kernel provides, straight from the loaded kernel:

    k = get_kernel("ray0rf1re/llama-easy", version=1, trust_remote_code=True)
    k.kernelize(model)

It replaces ``forward`` on the modules Transformers marks as hookable --
the ``RMSNorm`` layers and the ``rotary_pos_emb`` function -- exactly as
``kernels.kernelize`` would, and leaves everything else alone. The ops
run their Triton kernels on GPU tensors and their PyTorch definition
elsewhere, so a model that later moves device keeps working.
"""

from __future__ import annotations

import types

import torch.nn as nn

from . import layers


def _is_llama_rms_norm(module: nn.Module) -> bool:
    # Transformers' @use_kernel_forward_from_hub("RMSNorm") sets
    # kernel_layer_name on the class; requiring variance_epsilon as well
    # keeps out norms of that name with a different formula.
    return (
        getattr(type(module), "kernel_layer_name", None) == "RMSNorm"
        and hasattr(module, "variance_epsilon")
        and getattr(getattr(module, "weight", None), "dim", lambda: 0)() == 1
    )


def kernelize(model: nn.Module) -> nn.Module:
    """Swap *model*'s RMSNorm layers and rotary embedding for this
    kernel's, in place; returns the model.

    What was swapped is recorded in ``model.llama_easy_kernelized`` (a
    dict of counts), so ``{"RMSNorm": 0, ...}`` says plainly when a model
    had nothing this kernel recognises. Inference only: there is no
    backward pass.
    """
    swapped = {"RMSNorm": 0, "rotary_pos_emb": 0}
    rotary_done = set()
    for module in model.modules():
        if _is_llama_rms_norm(module):
            module.forward = types.MethodType(layers.RMSNorm.forward, module)
            swapped["RMSNorm"] += 1
        # use_kernelized_func(apply_rotary_pos_emb) puts the hookable
        # function, as a module, in the attention layer's _kernel_funcs.
        for func in (getattr(module, "_kernel_funcs", None) or {}).values():
            if getattr(func, "kernel_layer_name", None) == "rotary_pos_emb" and id(func) not in rotary_done:
                func.forward = types.MethodType(layers.ApplyRotary.forward, func)
                rotary_done.add(id(func))
                swapped["rotary_pos_emb"] += 1
    model.llama_easy_kernelized = swapped
    return model
