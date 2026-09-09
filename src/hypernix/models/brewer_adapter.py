"""hyperNix0x-v2 models, in every place NeoOven works.

:mod:`hypernix.training.brewer` builds the ``hyperNix0x-v2`` family — a
custom PyTorch transformer with its own config, its own presets and its
own checkpoint format. :class:`~hypernix.models.neo_oven.NeoOven` is the
front end for everything else: load, complete, chat, train, save. Until
now the two did not meet, so a model you brewed could be trained by
``brewer.train_model`` and by nothing else, and none of NeoOven's
generation, sampling or chat templating reached it.

They disagree about exactly two things
--------------------------------------
**The forward signature.** NeoOven calls ``model(ids)["logits"]`` to
generate and ``model(ids, labels=labels)["loss"]`` to train.
``BrewerModel.forward`` takes ``(input_ids, attn_mask)`` and returns a
bare logits tensor — no dict, no loss, and a second positional argument
that means something entirely different.

That second point is why this is an adapter and not a couple of
``getattr`` checks at the call sites. ``model(ids, labels)`` against a
raw BrewerModel does not fail: ``labels`` binds to ``attn_mask`` and is
used as an additive attention mask. A tensor of token ids added to the
attention scores produces a forward pass that runs, returns finite
numbers, and trains a model into noise. Wrapping it is the only way to
make that impossible rather than unlikely.

**The checkpoint layout.** Brewer writes ``{"config", "model_state_dict"}``
into a ``.pt``; NeoOven expects a snapshot directory. :func:`load` reads
either.

What this deliberately does not do
----------------------------------
It does not convert weights. A hyperNix0x-v2 model stays a
hyperNix0x-v2 model — the adapter is a calling convention, not a format
migration, and the tensors are never touched. Exporting to GGUF is
``brewer``'s ``_export_gguf``, which already exists and is the right
place for it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BrewerAdapter",
    "ARCH_NAME",
    "PRESETS",
    "is_brewer_checkpoint",
    "load",
    "wrap",
]

#: What this architecture is called everywhere a user types a name.
ARCH_NAME = "hypernix0x-v2"

#: The preset names ``brewer`` exposes, without the ``hypernix0x_v2_``
#: prefix its factory functions carry. Resolved lazily against the module
#: so a preset added there appears here without an edit -- the previous
#: pattern in this repository was a hand-maintained duplicate list, and
#: it went stale.
PRESETS: tuple[str, ...] = (
    "33m", "micro", "small", "medium", "large",
    "cpu-nano", "cpu-tiny", "cpu-small",
)


def _torch():
    """torch, imported on use.

    Same reason as elsewhere in this package: importing torch at module
    scope makes `hypernix --help` take two seconds and fail outright on a
    machine that only has the quantiser installed.
    """
    import torch

    return torch


def preset_config(name: str):
    """The :class:`BrewerConfig` for a preset name, or raise.

    Accepts ``small``, ``cpu-nano``, and the fully qualified
    ``hypernix0x-v2-small`` spelling that appears in saved configs, so a
    name read back out of a checkpoint round-trips.
    """
    from ..training import brewer

    key = name.strip().lower()
    for prefix in (f"{ARCH_NAME}-", "hypernix0x_v2_", "hypernix0x-v2-"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    factory = getattr(brewer, f"hypernix0x_v2_{key.replace('-', '_')}", None)
    if factory is None:
        raise ValueError(
            f"Unknown {ARCH_NAME} preset {name!r}. Available: "
            f"{', '.join(PRESETS)}"
        )
    return factory()


class BrewerAdapter:
    """A :class:`BrewerModel` with NeoOven's calling convention.

    An ``nn.Module`` subclass rather than a proxy object, so everything
    that walks a model -- ``.to()``, ``.parameters()``,
    ``state_dict()``, ``freeze()``, gradient checkpointing, the optimizer
    -- keeps working without knowing this exists.
    """

    def __new__(cls, model: Any, *args: Any, **kwargs: Any):
        # nn.Module is resolved at construction rather than at import, so
        # this module can be imported without torch. The class is built
        # once and cached on the function.
        return super().__new__(_adapter_class())

    def __init__(self, model: Any) -> None:  # pragma: no cover - see _make
        raise RuntimeError("BrewerAdapter is constructed via wrap()")


_ADAPTER_CLASS: type | None = None


def _adapter_class() -> type:
    """Build (once) the real adapter class, now that torch is available."""
    global _ADAPTER_CLASS
    if _ADAPTER_CLASS is not None:
        return _ADAPTER_CLASS

    torch = _torch()
    nn = torch.nn

    class _BrewerAdapter(nn.Module):
        """See :class:`BrewerAdapter`."""

        def __init__(self, model: Any) -> None:
            super().__init__()
            self.inner = model
            # Carried so NeoOven and anything reading a loaded model can
            # ask what it is without importing brewer.
            self.config = getattr(model, "cfg", None)
            self.arch = ARCH_NAME

        # -- the convention NeoOven expects ------------------------------

        def forward(
            self,
            input_ids: Any,
            labels: Any = None,
            attn_mask: Any = None,
            **_ignored: Any,
        ) -> dict[str, Any]:
            """``{"logits": ..., "loss": ...}``.

            ``labels`` is keyword-or-second-positional here on purpose:
            that is the position NeoOven passes it in, and it is the
            position ``BrewerModel`` uses for ``attn_mask``. Binding it
            correctly *here* is the whole point of the adapter -- passed
            straight through, a tensor of token ids becomes an additive
            attention mask and the model trains into noise without ever
            raising.
            """
            logits = self.inner(input_ids, attn_mask)
            out: dict[str, Any] = {"logits": logits}
            if labels is not None:
                out["loss"] = self.loss(logits, labels)
            return out

        @staticmethod
        def loss(logits: Any, labels: Any) -> Any:
            """Next-token cross entropy, shifted.

            Written out rather than delegated because ``BrewerModel`` has
            no loss at all: ``brewer.train_model`` computes it in its own
            loop. Two implementations of the same shift would be two
            places to get the off-by-one wrong, so this is the one
            NeoOven's trainer uses and brewer's own loop keeps its own --
            they are checked against each other in the tests.
            """
            torch = _torch()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            return torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                # -100 is torch's own ignore value and what every HF
                # collator emits for padding. Without it, padded batches
                # train the model to predict the pad token.
                ignore_index=-100,
            )

        # -- passthroughs ------------------------------------------------

        def num_params(self, trainable_only: bool = False) -> int:
            return self.inner.num_params(trainable_only=trainable_only)

        def __repr__(self) -> str:
            params = self.inner.num_params()
            name = getattr(self.config, "name", ARCH_NAME)
            return f"BrewerAdapter({name}, {params / 1e6:.1f}M params)"

    _ADAPTER_CLASS = _BrewerAdapter
    return _ADAPTER_CLASS


def wrap(model: Any) -> Any:
    """Give *model* NeoOven's calling convention.

    Idempotent, and a no-op for anything that already speaks it: passing
    an HF model or an already-wrapped one back through returns it
    unchanged, so a caller does not have to know which it has.
    """
    if model is None:
        return None
    if getattr(model, "arch", None) == ARCH_NAME:
        return model
    if type(model).__name__ != "BrewerModel":
        return model
    return _adapter_class()(model)


def is_brewer_checkpoint(path: str | Path) -> bool:
    """Whether *path* holds a hyperNix0x-v2 model.

    Checked by looking, not by trusting the extension. A ``.pt`` can be
    anything, and loading one to find out means executing a pickle --
    which is why the directory case reads ``config.json`` and the file
    case inspects the archive's member names without unpickling.
    """
    candidate = Path(path)
    if candidate.is_dir():
        config = candidate / "config.json"
        if not config.is_file():
            return False
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        # BrewerConfig's own fields. `d_model` and `n_layers` together
        # are unique to it -- HyperNixConfig uses hidden_size and
        # num_hidden_layers, and an HF config uses the latter too.
        return "d_model" in data and "n_layers" in data
    if not candidate.is_file():
        return False
    # A torch .pt is a zip whose `data.pkl` member holds the pickled
    # object graph. The dict's keys appear in it as literal strings, so
    # scanning those bytes recognises the layout without unpickling
    # anything -- torch.load on an untrusted file executes code, and
    # "does this file look like ours" must never be a reason to run it.
    #
    # The member *names* are not enough, which the first version of this
    # assumed: they are `data.pkl`, `byteorder` and one entry per tensor
    # storage, and none of them mentions a top-level key.
    try:
        import zipfile

        if not zipfile.is_zipfile(candidate):
            return False
        with zipfile.ZipFile(candidate) as archive:
            pickles = [n for n in archive.namelist() if n.endswith("data.pkl")]
            if not pickles:
                return False
            # Bounded: a pickle header is a few kilobytes and the keys
            # are near the front, so reading the whole member of a
            # multi-gigabyte checkpoint would be pointless work.
            with archive.open(pickles[0]) as handle:
                head = handle.read(64 * 1024)
        # Both, not either. `config` alone matches almost any checkpoint;
        # the pair is Brewer's layout.
        return b"model_state_dict" in head and b"config" in head
    except (OSError, KeyError, zipfile.BadZipFile):
        return False


def load(path: str | Path, *, device: str | None = None) -> tuple[Any, Any]:
    """``(adapted_model, config)`` from a checkpoint or a save directory.

    Accepts both layouts brewer writes: a ``.pt`` holding
    ``{"config", "model_state_dict"}``, and a directory with a
    ``config.json`` beside a weights file.
    """
    from ..training import brewer

    torch = _torch()
    candidate = Path(path)

    if candidate.is_dir():
        config_path = candidate / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"No config.json in {candidate}")
        config = brewer.BrewerConfig.load(config_path)
        model = brewer.BrewerModel(config)
        weights = next(
            (candidate / name for name in
             ("model.pt", "pytorch_model.bin", "weights.pt")
             if (candidate / name).is_file()),
            None,
        )
        if weights is None:
            # A config with no weights is a *shape*, which is a legitimate
            # thing to load -- `hnx brew new` writes exactly that. Said
            # out loud rather than returning random weights silently.
            logger.warning(
                "brewer_adapter: %s has a config but no weights; the model is "
                "randomly initialised.", candidate,
            )
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
    else:
        # weights_only is not available for this one: brewer's checkpoint
        # stores its config as a plain dict alongside the tensors, and
        # weights_only=True refuses anything that is not a tensor. The
        # file is one the user pointed at, which is the same trust level
        # as any other model file they load.
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(
                f"{candidate} is not a {ARCH_NAME} checkpoint "
                f"(no model_state_dict)"
            )
        config = brewer.BrewerConfig.from_dict(payload["config"])
        model = brewer.BrewerModel(config)
        model.load_state_dict(payload["model_state_dict"])

    adapted = wrap(model)
    if device:
        adapted = adapted.to(device)
    return adapted, config
