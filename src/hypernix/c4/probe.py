"""Running the model, so c4 can *test* a piece and not only look at it.

Measuring a stem's statistics needs only its weights. Telling whether a
layer is unneeded, or which weights would gain most from more training,
needs the model to run: a forward pass for the loss, a backward pass for
the gradients. :class:`Prober` is that, over a small calibration set.

The runnable module comes from ``transformers``: built from the
``config.json`` and handed the weights c4 already loaded (one copy, on
the GPU), or for a GGUF loaded through transformers' own GGUF support,
which covers the llama, mistral, qwen2/3, phi3, gemma and similar
families. A model transformers cannot build can still be exploded,
measured, reaped and sown -- only the tests that run it are unavailable,
and they say so.

Calibration text is tokenised with the model's own tokenizer when its
folder has one. ``Calibration.random_tokens`` makes synthetic batches
for structure checks; anything measured with them says ``synthetic``,
because a loss on random tokens says little about what a model knows.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .model import C4Error, C4Model

__all__ = ["Calibration", "Prober", "build_module"]


@dataclass
class Calibration:
    """Token batches to measure on: a list of ``(batch, seq)`` long tensors."""

    batches: list[Any]
    synthetic: bool = False
    source: str = ""

    @classmethod
    def from_ids(cls, ids: Any, *, seq_len: int = 128, batch_size: int = 4) -> Calibration:
        import torch

        flat = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        usable = (flat.numel() // seq_len) * seq_len
        if usable == 0:
            raise C4Error(f"calibration needs at least {seq_len} tokens")
        rows = flat[:usable].reshape(-1, seq_len)
        return cls([rows[i:i + batch_size] for i in range(0, rows.shape[0], batch_size)],
                   source="token ids")

    @classmethod
    def from_texts(cls, texts: Sequence[str], tokenizer: Any, *, seq_len: int = 128,
                   batch_size: int = 4) -> Calibration:
        ids: list[int] = []
        for text in texts:
            ids.extend(tokenizer(text)["input_ids"])
        made = cls.from_ids(ids, seq_len=seq_len, batch_size=batch_size)
        made.source = f"{len(texts)} texts"
        return made

    @classmethod
    def random_tokens(cls, vocab_size: int, *, batches: int = 2, batch_size: int = 2,
                      seq_len: int = 32, seed: int = 0) -> Calibration:
        import torch

        generator = torch.Generator().manual_seed(seed)
        made = [torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
                for _ in range(batches)]
        return cls(made, synthetic=True, source="random tokens")

    @property
    def tokens(self) -> int:
        return int(sum(b.numel() for b in self.batches))


def build_module(model: C4Model) -> Any:
    """A transformers causal-LM whose parameters are *model*'s tensors."""
    try:
        import transformers
    except ImportError as exc:
        raise C4Error("running a model needs transformers: pip install 'hypernix[train]'") from exc

    if model.format == "gguf":
        folder, name = model.path.parent, model.path.name
        try:
            module = transformers.AutoModelForCausalLM.from_pretrained(str(folder), gguf_file=name)
        except Exception as exc:  # noqa: BLE001 - transformers raises many things
            raise C4Error(f"transformers cannot run {name}: {exc}. Its stems can still be "
                          f"exploded, measured, reaped and sown.") from exc
        module.to(model.device)
        # Its parameters carry Hugging Face names; those are the names now.
        model.tensors = {n: p.data for n, p in module.named_parameters()}
        model.format = "safetensors"
        model.config = module.config.to_dict()
        model.module = module
        return module
    config_dict = {k: v for k, v in model.config.items() if not k.startswith("_")}
    try:
        config = transformers.AutoConfig.for_model(**config_dict)
    except Exception as exc:  # noqa: BLE001
        raise C4Error(f"transformers does not know this config ({config_dict.get('model_type')}): {exc}") from exc
    import torch

    with torch.device("meta"):
        module = transformers.AutoModelForCausalLM.from_config(config)
    state = {n: t for n, t in model.tensors.items()}
    missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
    tied = getattr(config, "tie_word_embeddings", False)
    missing = [m for m in missing if not (tied and m.endswith("lm_head.weight"))]
    if missing:
        raise C4Error(f"the weights do not fill the model: missing {missing[:5]}"
                      f"{' and more' if len(missing) > 5 else ''}")
    if tied and hasattr(module, "tie_weights"):
        module.tie_weights()
    _materialise_buffers(module, config, model.device)
    model.bind(module)
    return module


def _materialise_buffers(module: Any, config: Any, device: Any) -> None:
    """Compute the buffers a checkpoint does not carry (rotary frequencies).

    ``to_empty`` would leave them as whatever memory it was given: the
    loss came out NaN on some runs and not others. The submodule that
    owns them is built again on *device*; failing that, the whole model
    is built once with its weight initialisation skipped and its buffers
    copied across.
    """
    import inspect

    import torch

    owners = {}
    for name, buf in module.named_buffers():
        if buf.is_meta:
            owner, _, _ = name.rpartition(".")
            owners[owner] = module.get_submodule(owner) if owner else module
    if not owners:
        return
    leftover = []
    for owner, sub in owners.items():
        params = inspect.signature(type(sub)).parameters
        if "config" in params and owner:
            try:
                kwargs = {"device": device} if "device" in params else {}
                with torch.device(device):
                    fresh = type(sub)(config=getattr(sub, "config", config), **kwargs)
                parent, _, attr = owner.rpartition(".")
                setattr(module.get_submodule(parent) if parent else module, attr, fresh)
                continue
            except Exception:  # noqa: BLE001 - fall through to the whole-model build
                pass
        leftover.append(owner)
    if not leftover:
        return
    import transformers

    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4
        from transformers.modeling_utils import no_init_weights
    with torch.device(device), no_init_weights():
        full = transformers.AutoModelForCausalLM.from_config(config)
    for name, buf in full.named_buffers():
        owner, _, attr = name.rpartition(".")
        if owner in leftover:
            module.get_submodule(owner).register_buffer(attr, buf.to(device), persistent=False)
    del full


class Prober:
    """Loss and gradients of a model over a calibration set."""

    def __init__(self, model: C4Model, calibration: Calibration, *, module: Any = None) -> None:
        self.model = model
        self.calibration = calibration
        self.module = module or model.module or build_module(model)
        self.module.eval()

    @classmethod
    def for_model(cls, model: C4Model, *, texts: Iterable[str] | None = None, ids: Any = None,
                  seq_len: int = 128, batch_size: int = 4) -> Prober:
        """A prober with calibration from *texts* (the model's tokenizer),
        token *ids*, or, failing both, random tokens (marked synthetic)."""
        module = model.module or build_module(model)
        if ids is not None:
            calibration = Calibration.from_ids(ids, seq_len=seq_len, batch_size=batch_size)
        elif texts is not None:
            folder = model.path if model.path.is_dir() else model.path.parent
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
            except Exception as exc:  # noqa: BLE001
                raise C4Error(f"{folder} has no tokenizer to read calibration text with: {exc}") from exc
            calibration = Calibration.from_texts(list(texts), tokenizer, seq_len=seq_len,
                                                 batch_size=batch_size)
        else:
            vocab = int(getattr(module.config, "vocab_size", 0) or model.config.get("vocab_size") or 0)
            calibration = Calibration.random_tokens(vocab, seq_len=min(seq_len, 64))
        return cls(model, calibration, module=module)

    def _batches(self) -> Iterable[Any]:
        for batch in self.calibration.batches:
            yield batch.to(self.model.device)

    def loss(self) -> float:
        """Mean next-token loss over the calibration set."""
        import torch

        total, count = 0.0, 0
        with torch.no_grad():
            for batch in self._batches():
                out = self.module(input_ids=batch, labels=batch)
                total += float(out.loss) * batch.shape[0]
                count += batch.shape[0]
        return total / max(count, 1)

    def backward(self, *, zero: bool = True) -> float:
        """Gradients of the mean loss into every parameter's ``.grad``."""
        if zero:
            self.module.zero_grad(set_to_none=True)
        batches = list(self._batches())
        total = 0.0
        for batch in batches:
            out = self.module(input_ids=batch, labels=batch)
            (out.loss / len(batches)).backward()
            total += float(out.loss.detach())
        return total / max(len(batches), 1)

    def hidden_states(self) -> list[Any]:
        """Per layer boundary, the hidden states over the first calibration batch:
        ``states[i]`` enters layer *i*, ``states[-1]`` leaves the last."""
        import torch

        batch = next(iter(self._batches()))
        with torch.no_grad():
            out = self.module(input_ids=batch, output_hidden_states=True)
        return list(out.hidden_states)

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.module.named_parameters()]

    def parameter(self, name: str) -> Any:
        return dict(self.module.named_parameters())[name]

    def describe(self) -> dict[str, Any]:
        return {"tokens": self.calibration.tokens, "synthetic": self.calibration.synthetic,
                "source": self.calibration.source, "device": self.model.device}

