"""scriptgen.templates — turn a parameter dict into a training script.

Three modes, because "generate a script" means three different things
depending on what you already have:

* :func:`generate` writes a complete, runnable script from scratch.
* :func:`inject` puts a config block into an *existing* script, replacing
  the previous one if it finds it. This is the one people actually use
  after the first day — they have a training loop they like and want the
  GUI to drive its constants.
* :func:`as_config` emits just the values — as a Python block, JSON, YAML,
  TOML, environment variables, or command-line flags for a generated
  script — for a loop that reads its own config.

The generated script is deliberately readable
---------------------------------------------
It is not a wrapper that calls a HyperNix function with 40 keyword
arguments. It is a training loop someone can read, edit, and stop using
this GUI for. A generator whose output can only be regenerated is a
generator people abandon as soon as they need one thing it does not do.

Since 0.72.8rc1 it runs as written: it loads the model (a HyperNix model
through hypernix, anything else through transformers), reads the data
(a Hugging Face dataset or a local .jsonl, .json or .txt), and trains,
evaluates, saves, reports and stops early. It takes ``--name value`` for
any CONFIG key, so a second run with another learning rate needs no
regeneration. Earlier scripts stopped in ``main()`` with "bring your own
loader", and asked ``hypernix.optimizers`` for Pressure Cooker classes
that live in its submodules, with arguments they do not take; the
default optimiser could never be built. ``training_loop="skeleton"``
still gives the helpers alone.

Values are emitted through :func:`_literal`, never interpolated as text.
A dataset path with a quote in it is not an attack here — the person
typing it is the person running it — but it is a syntax error, and a
generator that produces one is broken.
"""
from __future__ import annotations

import json
import re
import textwrap
import time
from typing import Any

from .params import ALL_PARAMS, GROUPS

__all__ = ["generate", "inject", "as_config", "BEGIN_MARKER", "END_MARKER", "CONFIG_FORMATS"]

#: Injection markers. Distinctive enough not to collide with real code
#: and stable across versions — changing them would orphan every script
#: someone has already injected into.
BEGIN_MARKER = "# --- hypernix scriptgen: begin ---"
END_MARKER = "# --- hypernix scriptgen: end ---"

#: What :func:`as_config` can write.
CONFIG_FORMATS = ("python", "json", "yaml", "toml", "env", "args")

_BLOCK_RE = re.compile(
    re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
    re.DOTALL,
)


def _literal(value: Any) -> str:
    """Render a Python literal. Never string interpolation."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int,)):
        return repr(value)
    if isinstance(value, float):
        # repr(), not a format spec. Python's float repr is the shortest
        # string that round-trips exactly, so 3e-05 stays "3e-05" rather
        # than becoming "3.000000e-05" — and a formatted literal can
        # silently lose precision, which for a learning rate is a
        # different run than the one the GUI showed.
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_literal(v) for v in value) + ")"
    return repr(value)


def _config_block(values: dict[str, Any], *, include_comments: bool = True) -> str:
    """The CONFIG dict, grouped and commented like the form."""
    lines = [BEGIN_MARKER, "CONFIG = {"]
    for group in GROUPS:
        emitted = [p for p in group.params if p.name in values]
        if not emitted:
            continue
        if include_comments:
            lines.append(f"    # --- {group.title} ---")
        for param in emitted:
            value = values[param.name]
            rendered = f"    {param.script_name!r}: {_literal(value)},"
            if include_comments and param.hint:
                # Hints wrap rather than run off the edge; a generated
                # file with 200-character lines is a file people reformat
                # before reading, and then the markers move.
                for hint_line in textwrap.wrap(param.hint, width=76):
                    lines.append(f"    # {hint_line}")
            lines.append(rendered)
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.append("}")
    lines.append(END_MARKER)
    return "\n".join(lines)


def _scalar(value: Any) -> str:
    """One value for YAML or TOML: JSON's spelling of a scalar is valid in both."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(str(value))


def as_config(values: dict[str, Any], *, fmt: str = "python") -> str:
    """Just the values, in one of :data:`CONFIG_FORMATS`.

    ``args`` is the flags a generated script takes (``--learning_rate
    3e-05``), for running one script with another config.
    """
    payload = {
        ALL_PARAMS[name].script_name: value
        for name, value in values.items()
        if name in ALL_PARAMS
    }
    if fmt == "json":
        return json.dumps(payload, indent=2, default=str)
    if fmt == "env":
        lines = []
        for key, value in payload.items():
            if isinstance(value, bool):
                rendered = str(int(value))
            elif isinstance(value, (list, tuple)):
                rendered = ",".join(str(v) for v in value)
            else:
                rendered = str(value)
            if " " in rendered or not rendered:
                rendered = json.dumps(rendered)
            lines.append(f"HYPERNIX_{key.upper()}={rendered}")
        return "\n".join(lines)
    if fmt in ("yaml", "toml"):
        sep = ": " if fmt == "yaml" else " = "
        lines = ["# hnx scriptgen config"]
        for group in GROUPS:
            names = [p for p in group.params if p.name in values]
            if not names:
                continue
            lines.append("")
            lines.append(f"# {group.title}")
            for param in names:
                lines.append(f"{param.script_name}{sep}{_scalar(values[param.name])}")
        return "\n".join(lines)
    if fmt == "args":
        import shlex

        parts = []
        for key, value in payload.items():
            if isinstance(value, (list, tuple)):
                rendered = ",".join(repr(v) for v in value)
            elif isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = repr(value) if isinstance(value, float) else str(value)
            parts.append(f"--{key} {shlex.quote(rendered)}")
        return " \\\n  ".join(["python train.py", *parts])
    if fmt == "python":
        return _config_block(values)
    raise ValueError(f"fmt must be one of {', '.join(CONFIG_FORMATS)}; got {fmt!r}")


def generate(values: dict[str, Any], *, include_comments: bool = True) -> str:
    """A complete training script."""
    notes = str(values.get("notes") or "").strip()
    # The notes go into the docstring; a backslash or three quotes in them
    # would end it early.
    notes = notes.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    skeleton = str(values.get("training_loop", "full")) == "skeleton"
    header = [
        '"""Training script generated by `hnx scriptgen`.',
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.",
    ]
    if notes:
        header.append("")
        for paragraph in notes.splitlines():
            header.extend(textwrap.wrap(paragraph, width=76) or [""])
    if skeleton:
        usage = [
            "",
            "The helpers below (optimiser, scheduler, loss, LoRA, hardware tuning)",
            "are ready to use; main() is where your model and data loading go.",
        ]
    else:
        usage = [
            "",
            "Run it:",
            "",
            "    python train.py",
            "    python train.py --learning_rate 1e-4 --epochs 1   # any CONFIG key",
        ]
    header.extend(
        [
            *usage,
            "",
            "This is an ordinary training loop, not a wrapper. Edit it. The block",
            "between the scriptgen markers is the only part the GUI rewrites, so",
            "anything you change outside it survives a regeneration.",
            '"""',
        ]
    )
    block = _config_block(values, include_comments=include_comments)
    return "\n".join(header) + _IMPORTS + block + _HELPERS + (_SKELETON if skeleton else _FULL)


_IMPORTS = '''
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

'''

_HELPERS = '''


def parse_overrides(argv, config):
    """``--name value`` for any CONFIG key, typed like the value it replaces."""
    parser = argparse.ArgumentParser(description="Train with CONFIG; any key can be overridden.")
    for key, value in config.items():
        if isinstance(value, bool):
            kind = lambda text: text.strip().lower() in ("1", "true", "yes", "on")  # noqa: E731
        elif isinstance(value, (list, tuple)):
            kind = lambda text: tuple(float(v) for v in text.split(","))  # noqa: E731
        elif isinstance(value, (int, float)):
            kind = type(value)
        else:
            kind = str
        parser.add_argument(f"--{key}", type=kind, default=value, metavar=type(value).__name__.upper())
    return vars(parser.parse_args(argv))


#: Pressure Cooker classes and the hypernix module each lives in.
_COOKERS = {
    "PressureCookerV5": "pressure_cooker_v5",
    "PressureCookerV5Plus": "pressure_cooker_v5",
    "Agedcookerv5": "pressure_cooker_v5",
    "ULTRAagedcookerv5": "pressure_cooker_v5",
    "PressureCookerV5S": "pressure_cooker_v5s",
    "PressureCookerV6": "pressure_cooker_v6",
    "PressureCookerV6V": "pressure_cooker_v6v",
}


def build_optimizer(model, config):
    """Build the optimiser named in CONFIG.

    The Pressure Cooker family lives in hypernix.optimizers' submodules;
    everything else comes from torch. Only the arguments a class accepts
    are passed: the cookers take momentum_beta rather than betas, and
    clip inside the step only when asked (the loop clips already).
    """
    import importlib
    import inspect

    import torch

    name = config["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]

    if name in _COOKERS:
        module = importlib.import_module(f"hypernix.optimizers.{_COOKERS[name]}")
        cls = getattr(module, name)
        wanted = {
            "lr": config["learning_rate"], "weight_decay": config["weight_decay"],
            "eps": config["eps"], "momentum_beta": config["beta1"], "grad_clip": None,
            "six_bit_mode": config.get("six_bit_mode", "off"),
            "compile": config.get("compile_model", False),
        }
        accepted = set(inspect.signature(cls.__init__).parameters)
        for base in cls.__mro__[1:]:
            if "__init__" in vars(base):
                accepted |= set(inspect.signature(base.__init__).parameters)
        return cls(params, **{k: v for k, v in wanted.items() if k in accepted})

    common = dict(lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if name == "AdamW":
        return torch.optim.AdamW(
            params, betas=(config["beta1"], config["beta2"]), eps=config["eps"], **common
        )
    if name == "SGD":
        return torch.optim.SGD(params, momentum=config["beta1"], **common)
    if name == "Adafactor":
        from torch.optim import Adafactor  # torch >= 2.5

        return Adafactor(params, **common)
    if name == "Lion":
        try:
            from lion_pytorch import Lion
        except ImportError:
            raise SystemExit("Lion needs `pip install lion-pytorch`") from None
        return Lion(params, betas=(config["beta1"], config["beta2"]), **common)
    raise SystemExit(f"Unknown optimizer {name!r}")


def build_scheduler(optimizer, config, total_steps):
    """Warmup then decay, per CONFIG['lr_scheduler'].

    warmup_steps overrides warmup_ratio when non-zero, which is what a
    resumed run needs — the ratio is relative to a total the resumed run
    no longer has.
    """
    import torch

    warmup = config["warmup_steps"] or int(total_steps * config["warmup_ratio"])
    floor = config["min_lr_ratio"]
    kind = config["lr_scheduler"]

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if kind == "linear":
            return max(floor, 1.0 - progress)
        if kind == "constant_with_warmup":
            return 1.0
        if kind == "polynomial":
            return max(floor, (1.0 - progress) ** 2)
        if kind == "wsd":
            # Hold, then decay only over the last 20%: a run stopped
            # early is still at a usable learning rate.
            return 1.0 if progress < 0.8 else max(floor, 1.0 - (progress - 0.8) / 0.2)
        if kind == "cosine_with_restarts":
            cycle = (progress * 4) % 1.0
            return max(floor, 0.5 * (1.0 + math.cos(math.pi * cycle)))
        return max(floor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def compute_loss(logits, labels, config):
    """CONFIG['loss_function'], applied to next-token prediction."""
    import torch
    import torch.nn.functional as F

    kind = config["loss_function"]
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    if kind == "cross_entropy":
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    if kind == "label_smoothed_ce":
        return F.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100,
            label_smoothing=config["label_smoothing"],
        )
    if kind == "z_loss_ce":
        # Cross-entropy plus a penalty on logit magnitude. Cheap, and it
        # keeps long runs from drifting into numerically awkward logits.
        ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        keep = flat_labels != -100
        z = torch.logsumexp(flat_logits[keep], dim=-1).pow(2).mean()
        return ce + 1e-4 * z
    if kind == "focal":
        ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
        keep = flat_labels != -100
        return ((1 - torch.exp(-ce[keep])) ** 2 * ce[keep]).mean()
    if kind == "mse":
        keep = flat_labels != -100
        return F.mse_loss(flat_logits[keep], F.one_hot(
            flat_labels[keep], flat_logits.size(-1)).float())
    if kind == "kl_distill":
        raise SystemExit(
            "kl_distill needs a teacher's logits for every batch. Load the teacher in "
            "train() and replace this call with a KL between the two."
        )
    raise SystemExit(f"Unknown loss {kind!r}")


def add_lora(model, config):
    """Freeze *model* and put a low-rank adapter beside each target layer.

    Plain PyTorch, so it needs nothing installed and works on a HyperNix
    model as on a transformers one: both name their projections q_proj,
    k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
    """
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base, rank, alpha, dropout):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.drop = nn.Dropout(dropout)
            weight = base.weight
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=weight.device))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=weight.device))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, x):
            out = self.base(x)
            update = (self.drop(x).to(self.lora_a.dtype) @ self.lora_a.t()) @ self.lora_b.t()
            return out + (update * self.scale).to(out.dtype)

        def merged(self):
            with torch.no_grad():
                delta = (self.lora_b @ self.lora_a) * self.scale
                self.base.weight += delta.to(self.base.weight.dtype)
            return self.base

    targets = {t.strip() for t in str(config["lora_targets"]).split(",") if t.strip()}
    for param in model.parameters():
        param.requires_grad_(False)
    replaced = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(
                    child, config["lora_rank"], config["lora_alpha"], config["lora_dropout"]))
                replaced += 1
    if not replaced:
        raise SystemExit(f"LoRA: this model has no layer named {', '.join(sorted(targets))}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("LoRA: %d layers adapted; %s of %s parameters train (%.2f%%)",
             replaced, f"{trainable:,}", f"{total:,}", 100 * trainable / max(1, total))
    return model


def merge_lora(model):
    """Fold every adapter into its layer, leaving a plain model."""
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if hasattr(child, "merged") and hasattr(child, "lora_a"):
                setattr(module, child_name, child.merged())
    return model


def apply_hardware_tuning(config):
    """Let the Pascal tuner override dtype and batching on sm_61.

    Only when CONFIG asks for it. The tuner knows one thing this script
    cannot: that FP16 arithmetic is 1:64 on a GTX 1080 and 2:1 on a
    P100, so "use fp16" means opposite things on two cards that both
    report compute capability 6.x.
    """
    if not config.get("pascal_autotune") or config.get("device") in ("cpu", "mps"):
        return config

    try:
        from hypernix.system.pascal import autotune, detect
    except ImportError:
        log.warning("pascal auto-tune requested but hypernix.system.pascal is unavailable")
        return config

    info = detect()
    if not info.is_pascal:
        log.info("no Pascal GPU detected; leaving CONFIG as written")
        return config

    tuning = autotune(info, sequence_length=config["sequence_length"])
    log.info("Pascal detected (%s); applying tuned settings", info.name)
    for key, value in (
        ("precision", "fp16_fp32_compute" if tuning.compute_dtype == "fp32" else "fp16"),
        ("micro_batch", tuning.micro_batch),
        ("gradient_accumulation", tuning.gradient_accumulation),
        ("gradient_checkpointing", tuning.gradient_checkpointing),
    ):
        if config.get(key) != value:
            log.info("  %s: %r -> %r", key, config.get(key), value)
            config[key] = value
    for warning in tuning.warnings:
        log.warning("  %s", warning)
    return config
'''

_SKELETON = '''

def main() -> int:
    import torch

    config = parse_overrides(sys.argv[1:], dict(CONFIG))
    config = apply_hardware_tuning(config)

    torch.manual_seed(config["seed"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))

    effective_batch = config["micro_batch"] * config["gradient_accumulation"]
    log.info("effective batch %d (%d x %d)", effective_batch,
             config["micro_batch"], config["gradient_accumulation"])

    # ------------------------------------------------------------------
    # Replace this section with your own model and data loading. Nothing
    # below the marker block is regenerated, so edits here survive.
    # ------------------------------------------------------------------
    raise SystemExit(
        "This script is the skeleton: it has no model or dataset loader.\\n"
        "Fill in main() with your own loading code — build_optimizer, "
        "build_scheduler, compute_loss and add_lora above are ready to use. "
        "Generate with training_loop='full' for a complete loop."
    )


if __name__ == "__main__":
    sys.exit(main())
'''

_FULL = '''


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------

def pick_device(config):
    import torch

    wanted = config.get("device", "auto")
    if wanted == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if wanted == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device is 'cuda' but torch sees no CUDA GPU; use --device auto or cpu")
    return torch.device(wanted)


def autocast_dtype(config, device):
    """The dtype to autocast to, or None for FP32.

    fp16_fp32_compute keeps compute in FP32, which is what a GTX 1080
    wants (its FP16 arithmetic runs at 1/64 rate); the CPU trains in FP32.
    """
    import torch

    precision = config["precision"]
    if device.type == "cpu" or precision in ("fp32", "fp16_fp32_compute"):
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if device.type == "cuda":  # auto
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def fetch(source):
    """A local folder for *source*: itself, or a Hub download."""
    folder = Path(source).expanduser()
    if folder.exists():
        return folder
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(f"{source} is not a local folder, and downloading it needs "
                         "`pip install huggingface_hub`") from None
    log.info("downloading %s", source)
    return Path(snapshot_download(source))


class Tokens:
    """encode(text) -> ids, the end-of-text id, and how to save it, for
    either kind of model."""

    def __init__(self, encode, eos, save):
        self.encode, self.eos, self.save = encode, eos, save


def load_model(config):
    """``(model, tokens, context, kind)``. A HyperNix (Brewer) model loads
    through hypernix; anything else through transformers."""
    folder = fetch(config["model_id"])
    try:
        from hypernix.models import brewer_adapter
    except ImportError:
        brewer_adapter = None
    if brewer_adapter is not None and brewer_adapter.is_brewer_checkpoint(folder):
        model, cfg = brewer_adapter.load(folder)
        tokenizer_file = next((p for p in (folder / "tokenizer.json", folder / "tokenizer" / "tokenizer.json")
                               if p.is_file()), None)
        if tokenizer_file is not None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tokenizer_file))
            eos = next((tok.token_to_id(t) for t in ("<|endoftext|>", "</s>", "<eos>", "<|eos|>")
                        if tok.token_to_id(t) is not None), None)
            tokens = Tokens(lambda text: tok.encode(text).ids, eos,
                            lambda out: shutil.copy(tokenizer_file, out / "tokenizer.json"))
        else:
            log.warning("no tokenizer.json beside %s; training on raw bytes", folder)
            tokens = Tokens(lambda text: list(text.encode("utf-8")), 0, lambda out: None)
        log.info("loaded HyperNix model %s: %s parameters", folder.name,
                 f"{sum(p.numel() for p in model.parameters()):,}")
        return model, tokens, int(getattr(cfg, "max_seq_len", 0) or 0), "hypernix"

    try:
        import transformers
    except ImportError:
        raise SystemExit("this is not a HyperNix model, and loading it needs "
                         "`pip install transformers`") from None
    kwargs = {"trust_remote_code": bool(config.get("trust_remote_code"))}
    quant = config.get("quantization", "none")
    if quant in ("NF4", "INT8"):
        try:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=quant == "NF4", load_in_8bit=quant == "INT8",
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"{quant} weights need bitsandbytes: {exc}") from None
    elif quant != "none":
        log.warning("%s is an export format, not a training one; training full weights", quant)
    tok = transformers.AutoTokenizer.from_pretrained(folder, trust_remote_code=kwargs["trust_remote_code"])
    model = transformers.AutoModelForCausalLM.from_pretrained(folder, **kwargs)
    context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    tokens = Tokens(lambda text: tok(text, add_special_tokens=False)["input_ids"], tok.eos_token_id,
                    lambda out: tok.save_pretrained(out))
    return model, tokens, context, "transformers"


def as_text(row, field):
    if isinstance(row, str):
        return row
    value = row.get(field) if hasattr(row, "get") else None
    if value is None and hasattr(row, "get") and isinstance(row.get("messages"), list):
        # Chat data: one "role: content" line per message.
        value = "\\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in row["messages"])
    return "" if value is None else str(value)


def load_texts(config):
    """The training text: a Hugging Face dataset, or a local .jsonl, .json or .txt."""
    source = str(config["dataset"]).strip()
    if not source:
        raise SystemExit("CONFIG['dataset'] is empty: name a Hugging Face dataset, or a local "
                         ".jsonl, .json or .txt file (or pass --dataset PATH)")
    path = Path(source).expanduser()
    field = config["text_field"]
    if path.is_file():
        raw = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".txt":
            rows = [block for block in raw.split("\\n\\n") if block.strip()]
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw)
            rows = data if isinstance(data, list) else data.get("data") or data.get("rows") or [data]
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(f"{source} is not a local file, and loading a Hub dataset needs "
                             "`pip install datasets`") from None
        rows = load_dataset(source, split=config["dataset_split"])
    texts = [t for t in (as_text(row, field) for row in rows) if t.strip()]
    if config["max_samples"]:
        texts = texts[: config["max_samples"]]
    if not texts:
        raise SystemExit(f"no text in {source} (looked in the {field!r} field)")
    log.info("%s examples from %s", f"{len(texts):,}", source)
    return texts


def make_blocks(texts, tokens, length, packing):
    """Token blocks of at most *length*. Packed, examples are joined with
    the end-of-text token and cut into full blocks; otherwise each example
    is one block, truncated."""
    blocks = []
    if packing:
        stream = []
        for text in texts:
            stream.extend(tokens.encode(text))
            if tokens.eos is not None:
                stream.append(tokens.eos)
            while len(stream) >= length:
                blocks.append(stream[:length])
                del stream[:length]
        if len(stream) > 1 and not blocks:
            blocks.append(stream)
    else:
        for text in texts:
            ids = tokens.encode(text)[:length]
            if len(ids) > 1:
                blocks.append(ids)
    return blocks


def collate(blocks, device):
    """Input ids padded with 0, and labels padded with -100 so padding is
    never a target."""
    import torch

    width = max(len(b) for b in blocks)
    ids = torch.zeros((len(blocks), width), dtype=torch.long)
    labels = torch.full((len(blocks), width), -100, dtype=torch.long)
    for row, block in enumerate(blocks):
        ids[row, : len(block)] = torch.tensor(block)
        labels[row, : len(block)] = torch.tensor(block)
    return ids.to(device), labels.to(device)


def logits_of(output):
    return output["logits"] if isinstance(output, dict) else output.logits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class Reporter:
    """Metrics to metrics.jsonl, TensorBoard or W&B, and the live view."""

    def __init__(self, config, output):
        self.kind = config["report_to"]
        self.file = None
        self.writer = None
        self.run = None
        self.stream = None
        if self.kind == "jsonl":
            self.file = (output / "metrics.jsonl").open("a", encoding="utf-8")
        elif self.kind == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(output / "tensorboard"))
        elif self.kind == "wandb":
            import wandb

            self.run = wandb.init(project="hypernix", config=config, dir=str(output))
        if config.get("stream_metrics"):
            try:
                from hypernix.monitoring.livestream import LiveStreamServer

                self.stream = LiveStreamServer(port=config["stream_port"])
                port = self.stream.start()
                log.info("live view on http://127.0.0.1:%d", port)
            except Exception as exc:  # noqa: BLE001
                log.warning("live view unavailable: %s", exc)
                self.stream = None

    def __call__(self, step, total, **metrics):
        if self.file is not None:
            self.file.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\\n")
            self.file.flush()
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        if self.run is not None:
            self.run.log(metrics, step=step)
        if self.stream is not None:
            self.stream.progress("training", step / max(1, total), step=step, **metrics)

    def close(self):
        for thing, method in ((self.file, "close"), (self.writer, "close"), (self.run, "finish"),
                              (self.stream, "stop")):
            if thing is not None:
                getattr(thing, method)()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model, blocks, config, device, amp):
    import torch

    if not blocks:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(blocks), config["eval_batch"]):
            ids, labels = collate(blocks[start: start + config["eval_batch"]], device)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                loss = compute_loss(logits_of(model(ids)), labels, config)
            total += float(loss) * len(ids)
            count += len(ids)
    model.train()
    return total / max(1, count)


def save_checkpoint(model, optimizer, scheduler, step, config, output):
    import torch

    folder = output / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step}, folder / "state.pt")
    log.info("saved %s", folder)
    limit = config["save_total_limit"]
    if limit:
        saved = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for old in saved[:-limit]:
            shutil.rmtree(old, ignore_errors=True)


def resume_from(model, optimizer, scheduler, output):
    import torch

    saved = sorted(output.glob("checkpoint-*/state.pt"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not saved:
        log.info("resume: no checkpoint in %s; starting fresh", output)
        return 0
    state = torch.load(saved[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    log.info("resumed from %s (step %d)", saved[-1].parent.name, state["step"])
    return int(state["step"])


def save_final(model, tokens, kind, config, output):
    """The finished model, in a folder that loads like the one it came from."""
    folder = output / "final"
    folder.mkdir(parents=True, exist_ok=True)
    if config["use_lora"] and config["merge_lora"] and config["quantization"] == "none":
        model = merge_lora(model)
    model = getattr(model, "_orig_mod", model)  # torch.compile's wrapper
    if kind == "hypernix":
        from safetensors.torch import save_model

        inner = getattr(model, "inner", model)
        save_model(inner, str(folder / "model.safetensors"))
        inner.cfg.save(folder / "config.json")
    else:
        model.save_pretrained(folder)
    tokens.save(folder)
    log.info("final model in %s", folder)
    return folder


def train(config):
    """One run. Returns the best held-out loss (the last training loss
    when nothing is held out)."""
    import torch

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = pick_device(config)
    model, tokens, context, kind = load_model(config)
    length = config["sequence_length"]
    if context and length > context:
        log.info("sequence_length %d is longer than the model's context; using %d", length, context)
        length = context

    blocks = make_blocks(load_texts(config), tokens, length, config["packing"])
    if not blocks:
        raise SystemExit("the data made no training blocks; is it empty after tokenising?")
    random.Random(config["seed"]).shuffle(blocks)
    held = int(len(blocks) * config["val_split"])
    if config["val_split"] and len(blocks) > 1:
        held = max(1, held)
    val, train_blocks = blocks[:held], blocks[held:]
    if not train_blocks:
        raise SystemExit("everything was held out for evaluation; lower val_split or add data")
    log.info("%d training blocks of up to %d tokens, %d held out", len(train_blocks), length, len(val))

    if config["use_lora"]:
        model = add_lora(model, config)
    if config["gradient_checkpointing"]:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            from torch.utils.checkpoint import checkpoint

            for block in getattr(getattr(model, "inner", model), "blocks", []):
                block.forward = (lambda f: lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k))(block.forward)
    if not (getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)):
        model.to(device)  # bitsandbytes places quantised weights itself
    if config["compile_model"]:
        model = torch.compile(model)

    optimizer = build_optimizer(model, config)
    micro, accum = config["micro_batch"], config["gradient_accumulation"]
    batches = len(train_blocks) // micro if config["drop_last"] else math.ceil(len(train_blocks) / micro)
    batches = max(1, batches)
    per_epoch = max(1, math.ceil(batches / accum))
    total = config["max_steps"] or max(1, math.ceil(per_epoch * config["epochs"]))
    scheduler = build_scheduler(optimizer, config, total)
    amp = autocast_dtype(config, device)
    scaler = (torch.amp.GradScaler("cuda", init_scale=config["loss_scale"])
              if amp == torch.float16 and device.type == "cuda" else None)
    log.info("%s: %d optimiser steps, effective batch %d (%d x %d), %s on %s",
             config["model_id"], total, micro * accum, micro, accum, config["optimizer"], device)

    reporter = Reporter(config, output)
    step = resume_from(model, optimizer, scheduler, output) if config["resume"] else 0
    best, stale, last_loss = float("inf"), 0, float("nan")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started, seen_tokens, micro_step = time.time(), 0, 0
    try:
        while step < total:
            order = list(range(len(train_blocks)))
            if config["shuffle"]:
                random.shuffle(order)
            for start in range(0, len(order), micro):
                chunk = [train_blocks[i] for i in order[start: start + micro]]
                if config["drop_last"] and len(chunk) < micro and len(order) >= micro:
                    break
                ids, labels = collate(chunk, device)
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    loss = compute_loss(logits_of(model(ids)), labels, config) / accum
                (scaler.scale(loss) if scaler else loss).backward()
                seen_tokens += int((labels != -100).sum())
                micro_step += 1
                if micro_step % accum:
                    continue
                if scaler:
                    scaler.unscale_(optimizer)
                if config["max_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], config["max_grad_norm"])
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last_loss = float(loss) * accum

                if step % config["logging_steps"] == 0 or step == total:
                    rate = seen_tokens / max(1e-9, time.time() - started)
                    lr = scheduler.get_last_lr()[0]
                    log.info("step %d/%d  loss %.4f  lr %.2e  %s tok/s", step, total, last_loss, lr,
                             f"{rate:,.0f}")
                    reporter(step, total, loss=last_loss, lr=lr, tokens_per_second=rate)
                if val and config["eval_steps"] and step % config["eval_steps"] == 0:
                    held_loss = evaluate(model, val, config, device, amp)
                    log.info("eval  step %d  loss %.4f  perplexity %.2f", step, held_loss,
                             math.exp(min(held_loss, 50)))
                    reporter(step, total, eval_loss=held_loss)
                    if held_loss < best:
                        best, stale = held_loss, 0
                    else:
                        stale += 1
                        if config["early_stopping_patience"] and stale >= config["early_stopping_patience"]:
                            log.info("no better held-out loss in %d evaluations; stopping early", stale)
                            total = step
                if config["save_steps"] and step % config["save_steps"] == 0 and step < total:
                    save_checkpoint(model, optimizer, scheduler, step, config, output)
                if step >= total:
                    break
    finally:
        reporter.close()

    if val:
        held_loss = evaluate(model, val, config, device, amp)
        log.info("final held-out loss %.4f (perplexity %.2f)", held_loss, math.exp(min(held_loss, 50)))
        best = min(best, held_loss)
    if config.get("_sweep_trial"):
        return best if math.isfinite(best) else last_loss  # a trial's model is not kept
    final = save_final(model, tokens, kind, config, output)
    if config["push_to_hub"]:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(config["hub_repo"], private=True, exist_ok=True)
        api.upload_folder(folder_path=str(final), repo_id=config["hub_repo"])
        log.info("uploaded to https://huggingface.co/%s", config["hub_repo"])
    return best if math.isfinite(best) else last_loss


def sweep(config):
    """Short runs across lr_range; returns the learning rate that did best."""
    low, high = config["lr_range"]
    count = max(2, config["sweep_trials"])
    rates = [math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (count - 1))
             for i in range(count)]
    results = []
    for rate in rates:
        trial = dict(config, learning_rate=rate, max_steps=config["sweep_steps"], resume=False, _sweep_trial=True,
                     save_steps=0, push_to_hub=False, early_stopping_patience=0,
                     output_dir=str(Path(config["output_dir"]) / f"sweep-lr-{rate:.1e}"))
        log.info("sweep: learning rate %.2e", rate)
        results.append((train(trial), rate))
    for loss, rate in sorted(results):
        log.info("sweep: %.2e -> loss %.4f", rate, loss)
    return min(results)[1]


def main() -> int:
    config = parse_overrides(sys.argv[1:], dict(CONFIG))
    config = apply_hardware_tuning(config)
    if config["enable_sweep"]:
        config["learning_rate"] = sweep(config)
        log.info("training with the best learning rate, %.2e", config["learning_rate"])
    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def inject(existing: str, values: dict[str, Any], *, include_comments: bool = True) -> str:
    """Put the config block into an existing script.

    Replaces a previous block when the markers are present, so
    regenerating does not stack copies. When they are not, the block goes
    after the imports rather than at the top — a CONFIG dict above the
    imports is legal and reads as though someone pasted it in a hurry.
    """
    block = _config_block(values, include_comments=include_comments)
    if _BLOCK_RE.search(existing):
        return _BLOCK_RE.sub(lambda _: block, existing, count=1)

    lines = existing.splitlines()
    insert_at = 0
    in_docstring = False
    quote = ""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not in_docstring and stripped[:3] in ('"""', "'''"):
            quote = stripped[:3]
            # A single-line docstring opens and closes on the same line.
            if not (len(stripped) > 3 and stripped.endswith(quote)):
                in_docstring = True
            insert_at = index + 1
            continue
        if in_docstring:
            if quote in stripped:
                in_docstring = False
            insert_at = index + 1
            continue
        if stripped.startswith(("import ", "from ")) or not stripped:
            insert_at = index + 1
        elif stripped.startswith("#"):
            continue
        else:
            break

    return "\n".join([*lines[:insert_at], "", block, "", *lines[insert_at:]]) + "\n"
