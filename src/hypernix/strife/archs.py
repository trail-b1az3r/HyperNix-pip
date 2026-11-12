"""strife.archs — what each architecture is, layer by layer.

:func:`build_plan` turns a GGUF's metadata into a :class:`Plan`: for every
layer, its *mixer* (attention, LFM short convolution, or Qwen3.5's gated
delta net), its feed-forward (dense SwiGLU or a mixture of experts) and
its norm topology, plus every tensor that implies with the shape it must
have. Inspection checks those shapes before anything is loaded, and the
model runs from the same plan, so the two cannot disagree.

=============  ===============================================================
``llama``,     pre-norm attention (rope ``norm``; qwen: ``neox``), SwiGLU;
``mistral``,   qwen2 adds q/k/v biases, qwen3 per-head q/k RMSNorm
``qwen2``,
``qwen3``
``lfm3``       LFM3 (Liquid): a hybrid of gated short convolutions (width 7),
               full attention (half-rotary rope, base 1e6) and sliding-window
               attention (window 256, full rope, base 1e4); q/k norm and a
               sigmoid gate per head; sandwich norms around both blocks
``qwen35``,    Qwen3.5: gated delta-net linear attention, with full attention
``qwen35moe``  every few layers (q carries a sigmoid output gate, q/k norm,
               partial NeoX rope); dense SwiGLU, or (``qwen35moe``) softmax
               top-k experts plus a sigmoid-gated shared expert
``glm4moe``    GLM-4.5 / 4.5-Air: q/k/v biases, optional q/k norm, partial
               NeoX rope; leading dense layers, then sigmoid-routed experts
               with a selection bias, normalised and scaled, plus a shared
               expert
=============  ===============================================================

The GGUF conventions followed are llama.cpp's (``convert_hf_to_gguf.py``
and ``src/models/*.cpp``), so a file llama.cpp writes is a file Strife
reads: Qwen3.5's ``ssm_a`` is already ``-exp(A_log)``, its zero-centred
norms already have 1 added, and its value heads are in the tiled order
``ggml_repeat`` expects (value head *h* reads key head *h* mod *n_k*).
Multi-token-prediction (NextN) blocks are loaded by nobody and run by
nobody: they are for speculative drafting, not for the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ARCHITECTURES", "AttnSpec", "ConvSpec", "GDNSpec", "LayerSpec", "MoESpec", "Plan",
           "PlanError", "build_plan"]

ARCHITECTURES = ("llama", "mistral", "qwen2", "qwen3", "lfm3", "qwen35", "qwen35moe", "glm4moe")


class PlanError(ValueError):
    def __init__(self, code: str, message: str, remedy: str = "", tensor: str = "") -> None:
        super().__init__(message)
        self.code, self.remedy, self.tensor = code, remedy, tensor


@dataclass(frozen=True)
class AttnSpec:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope: str            # "norm": adjacent pairs; "neox": first half with second half
    n_rot: int           # dimensions rotated (from the front of each head)
    rope_base: float
    rope_scale: float = 1.0
    rope_factors: str = ""   # a tensor of per-frequency divisors (LFM3's zero tail)
    window: int = 0          # 0: full causal attention; else sliding window
    qkv_bias: bool = False
    qk_norm: bool = False
    gate: str = ""           # "": none; "head": sigmoid(x @ W) per head; "q": q carries it


@dataclass(frozen=True)
class ConvSpec:
    kernel: int


@dataclass(frozen=True)
class GDNSpec:
    n_k_heads: int
    n_v_heads: int
    head_k: int
    head_v: int
    kernel: int

    @property
    def key_dim(self) -> int:
        return self.n_k_heads * self.head_k

    @property
    def value_dim(self) -> int:
        return self.n_v_heads * self.head_v

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


@dataclass(frozen=True)
class MoESpec:
    n_expert: int
    n_used: int
    n_ff: int
    gating: str              # "softmax" | "sigmoid"
    norm: bool               # renormalise the chosen weights to sum to 1
    scale: float             # then multiply them by this
    probs_bias: bool         # a selection-only bias (GLM's e_score_correction_bias)
    fused_gate_up: bool
    shared_ff: int = 0       # 0: no shared expert
    shared_gate: bool = False


@dataclass(frozen=True)
class LayerSpec:
    index: int
    mixer: str               # "attn" | "conv" | "gdn"
    attn: AttnSpec | None = None
    conv: ConvSpec | None = None
    gdn: GDNSpec | None = None
    ffn: str = "dense"       # "dense" | "moe"
    n_ff: int = 0
    moe: MoESpec | None = None


@dataclass
class Plan:
    arch: str
    n_embd: int
    n_vocab: int
    eps: float
    norm_style: str          # "pre" | "sandwich" | "post_attn_ffn"
    layers: list[LayerSpec]
    tied: bool
    context_length: int
    #: Every tensor the plan reads, with its GGUF shape (ne order).
    tensors: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: Tensors the file may have and the plan uses if so.
    optional: dict[str, tuple[int, ...]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def recurrent(self) -> bool:
        """True when some layer carries state that cannot be rolled back."""
        return any(layer.mixer in ("conv", "gdn") for layer in self.layers)


# ---------------------------------------------------------------------------

def _get(md: dict[str, Any], key: str, default: Any = None) -> Any:
    value = md.get(key, default)
    return default if value is None else value


def _int(md: dict[str, Any], key: str, default: int = 0) -> int:
    value = _get(md, key, default)
    if isinstance(value, list):
        raise PlanError("strife.model.bad_metadata", f"{key} is a list where a number was expected")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise PlanError("strife.model.bad_metadata", f"{key} is not a number") from None


def _float(md: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(_get(md, key, default))
    except (TypeError, ValueError):
        raise PlanError("strife.model.bad_metadata", f"{key} is not a number") from None


def _per_layer(md: dict[str, Any], key: str, n: int, default: Any) -> list[Any]:
    value = md.get(key, default)
    if isinstance(value, list):
        if len(value) < n:
            raise PlanError("strife.model.bad_metadata", f"{key} has {len(value)} entries for {n} layers")
        return list(value[:n])
    return [value] * n


def _bound(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        raise PlanError("strife.model.bad_metadata", f"{name} = {value} is outside {low}..{high}")


class _Builder:
    def __init__(self, md: dict[str, Any], shapes: dict[str, tuple[int, ...]]) -> None:
        self.md, self.shapes = md, shapes
        self.arch = str(md.get("general.architecture") or "")
        self.tensors: dict[str, tuple[int, ...]] = {}
        self.optional: dict[str, tuple[int, ...]] = {}

    def k(self, key: str) -> str:
        return f"{self.arch}.{key}"

    def need(self, name: str, shape: tuple[int, ...]) -> None:
        self.tensors[name] = shape

    def maybe(self, name: str, shape: tuple[int, ...]) -> bool:
        if name in self.shapes:
            self.optional[name] = shape
            return True
        return False

    def has(self, name: str) -> bool:
        return name in self.shapes

    # -- pieces shared by several architectures ------------------------------

    def base(self) -> tuple[int, int, int, float]:
        n_embd = _int(self.md, self.k("embedding_length"))
        _bound("embedding_length", n_embd, 1, 1 << 17)
        emb = self.shapes.get("token_embd.weight")
        if emb is None or len(emb) != 2 or emb[0] != n_embd:
            raise PlanError("strife.model.missing_tensor",
                            "token_embd.weight is missing or the wrong shape", tensor="token_embd.weight")
        n_vocab = emb[1]
        eps = _float(self.md, self.k("attention.layer_norm_rms_epsilon"), 1e-5)
        context = _int(self.md, self.k("context_length"), 0)
        self.need("output_norm.weight", (n_embd,))
        return n_embd, n_vocab, context, eps

    def output(self, n_embd: int, n_vocab: int) -> bool:
        if self.has("output.weight"):
            self.need("output.weight", (n_embd, n_vocab))
            return False
        return True

    def attention(self, i: int, n_embd: int, a: AttnSpec) -> None:
        b = f"blk.{i}."
        q_out = a.n_heads * a.head_dim * (2 if a.gate == "q" else 1)
        kv_out = a.n_kv_heads * a.head_dim
        self.need(b + "attn_q.weight", (n_embd, q_out))
        self.need(b + "attn_k.weight", (n_embd, kv_out))
        self.need(b + "attn_v.weight", (n_embd, kv_out))
        self.need(b + "attn_output.weight", (a.n_heads * a.head_dim, n_embd))
        if a.qkv_bias:
            self.need(b + "attn_q.bias", (q_out,))
            self.need(b + "attn_k.bias", (kv_out,))
            self.need(b + "attn_v.bias", (kv_out,))
        if a.qk_norm:
            self.need(b + "attn_q_norm.weight", (a.head_dim,))
            self.need(b + "attn_k_norm.weight", (a.head_dim,))
        if a.gate == "head":
            self.need(b + "attn_gate.weight", (n_embd, a.n_heads))
        if a.rope_factors:
            self.need(a.rope_factors, (a.head_dim // 2,))

    def dense_ffn(self, i: int, n_embd: int, n_ff: int) -> None:
        b = f"blk.{i}."
        self.need(b + "ffn_gate.weight", (n_embd, n_ff))
        self.need(b + "ffn_up.weight", (n_embd, n_ff))
        self.need(b + "ffn_down.weight", (n_ff, n_embd))

    def moe_ffn(self, i: int, n_embd: int, m: MoESpec) -> None:
        b = f"blk.{i}."
        self.need(b + "ffn_gate_inp.weight", (n_embd, m.n_expert))
        if m.fused_gate_up:
            self.need(b + "ffn_gate_up_exps.weight", (n_embd, 2 * m.n_ff, m.n_expert))
        else:
            self.need(b + "ffn_gate_exps.weight", (n_embd, m.n_ff, m.n_expert))
            self.need(b + "ffn_up_exps.weight", (n_embd, m.n_ff, m.n_expert))
        self.need(b + "ffn_down_exps.weight", (m.n_ff, n_embd, m.n_expert))
        if m.probs_bias:
            self.need(b + "exp_probs_b.bias", (m.n_expert,))
        if m.shared_ff:
            self.need(b + "ffn_gate_shexp.weight", (n_embd, m.shared_ff))
            self.need(b + "ffn_up_shexp.weight", (n_embd, m.shared_ff))
            self.need(b + "ffn_down_shexp.weight", (m.shared_ff, n_embd))
            if m.shared_gate:
                self.need(b + "ffn_gate_inp_shexp.weight", (n_embd,))

    def rope_scale(self) -> float:
        kind = str(_get(self.md, self.k("rope.scaling.type"), "") or "")
        if kind in ("", "none"):
            return 1.0
        if kind == "linear":
            return _float(self.md, self.k("rope.scaling.factor"), 1.0) or 1.0
        raise PlanError("strife.arch.rope_scaling",
                        f"rope scaling {kind!r} is not implemented in Strife",
                        "Strife implements linear rope scaling; use the llama.cpp runner for "
                        f"{kind} scaling.")

    def finish(self, plan: Plan) -> Plan:
        plan.tensors, plan.optional = self.tensors, self.optional
        for name, shape in self.tensors.items():
            got = self.shapes.get(name)
            if got is None:
                raise PlanError("strife.model.missing_tensor", f"{name} is missing", tensor=name)
            if tuple(got) != tuple(shape):
                raise PlanError("strife.model.bad_shape", f"{name} is {tuple(got)}, expected {shape}",
                                tensor=name)
        return plan


# ---------------------------------------------------------------------------

def _llama_family(b: _Builder) -> Plan:
    md, arch = b.md, b.arch
    n_embd, n_vocab, context, eps = b.base()
    n_layers = _int(md, b.k("block_count"))
    _bound("block_count", n_layers, 1, 1024)
    n_heads = _int(md, b.k("attention.head_count"))
    n_kv = _int(md, b.k("attention.head_count_kv"), n_heads)
    n_ff = _int(md, b.k("feed_forward_length"))
    head_dim = _int(md, b.k("attention.key_length"), 0) or (n_embd // max(1, n_heads))
    if min(n_heads, n_kv, n_ff, head_dim) <= 0 or n_heads % n_kv or head_dim % 2:
        raise PlanError("strife.model.bad_metadata", "the attention metadata is inconsistent "
                        f"(heads={n_heads}, kv_heads={n_kv}, head_dim={head_dim}, ff={n_ff})")
    attn = AttnSpec(n_heads, n_kv, head_dim,
                    rope="neox" if arch in ("qwen2", "qwen3") else "norm",
                    n_rot=_int(md, b.k("rope.dimension_count"), head_dim) or head_dim,
                    rope_base=_float(md, b.k("rope.freq_base"), 10000.0), rope_scale=b.rope_scale(),
                    qkv_bias=arch == "qwen2", qk_norm=arch == "qwen3")
    layers = []
    for i in range(n_layers):
        b.need(f"blk.{i}.attn_norm.weight", (n_embd,))
        b.need(f"blk.{i}.ffn_norm.weight", (n_embd,))
        b.attention(i, n_embd, attn)
        b.dense_ffn(i, n_embd, n_ff)
        layers.append(LayerSpec(i, "attn", attn=attn, n_ff=n_ff))
    tied = b.output(n_embd, n_vocab)
    return b.finish(Plan(arch, n_embd, n_vocab, eps, "pre", layers, tied, context, summary={
        "heads": n_heads, "kv_heads": n_kv, "head_dim": head_dim, "ff": n_ff,
        "rope_base": attn.rope_base, "rope_scale": attn.rope_scale}))


def _lfm3(b: _Builder) -> Plan:
    md = b.md
    n_embd, n_vocab, context, eps = b.base()
    n_layers = _int(md, b.k("block_count"))
    _bound("block_count", n_layers, 1, 1024)
    if not bool(_get(md, b.k("attention.causal"), True)):
        raise PlanError("strife.arch.non_causal",
                        "this LFM3 model reads its prompt bidirectionally (a non-causal variant)",
                        "Strife runs causal language models; use the llama.cpp runner for this one.")
    n_heads = _int(md, b.k("attention.head_count"))
    kv = _per_layer(md, b.k("attention.head_count_kv"), n_layers, n_heads)
    swa = _per_layer(md, b.k("attention.sliding_window_pattern"), n_layers, False)
    head_dim = _int(md, b.k("attention.key_length"), 0) or (n_embd // max(1, n_heads))
    n_ff = _int(md, b.k("feed_forward_length"))
    window = _int(md, b.k("attention.sliding_window"), 0)
    kernel = _int(md, b.k("shortconv.l_cache"), 3)
    _bound("shortconv.l_cache", kernel, 1, 64)
    base = _float(md, b.k("rope.freq_base"), 1e6)
    base_swa = _float(md, b.k("rope.freq_base_swa"), base)
    n_rot = _int(md, b.k("rope.dimension_count"), head_dim) or head_dim
    factors = "rope_freqs.weight" if b.has("rope_freqs.weight") else ""
    sandwich = b.has("blk.0.post_attention_norm.weight")
    layers = []
    for i in range(n_layers):
        p = f"blk.{i}."
        b.need(p + "attn_norm.weight", (n_embd,))
        b.need(p + "ffn_norm.weight", (n_embd,))
        if sandwich:
            b.need(p + "post_attention_norm.weight", (n_embd,))
            b.need(p + "post_ffw_norm.weight", (n_embd,))
        b.dense_ffn(i, n_embd, n_ff)
        n_kv = int(kv[i] or 0)
        if n_kv == 0:
            b.need(p + "shortconv.conv.weight", (kernel, n_embd))
            b.need(p + "shortconv.in_proj.weight", (n_embd, 3 * n_embd))
            b.need(p + "shortconv.out_proj.weight", (n_embd, n_embd))
            layers.append(LayerSpec(i, "conv", conv=ConvSpec(kernel), n_ff=n_ff))
            continue
        if n_heads % n_kv:
            raise PlanError("strife.model.bad_metadata", f"layer {i}: {n_heads} heads over {n_kv} kv heads")
        sliding = bool(swa[i])
        if sliding and window <= 0:
            raise PlanError("strife.model.bad_metadata", "sliding layers without a sliding_window")
        attn = AttnSpec(n_heads, n_kv, head_dim, rope="neox", n_rot=n_rot,
                        rope_base=base_swa if sliding else base,
                        rope_factors="" if sliding else factors,
                        window=window if sliding else 0, qk_norm=True, gate="head")
        b.attention(i, n_embd, attn)
        layers.append(LayerSpec(i, "attn", attn=attn, n_ff=n_ff))
    tied = b.output(n_embd, n_vocab)
    kinds = [lay.mixer if lay.mixer != "attn" or not lay.attn or not lay.attn.window else "swa"
             for lay in layers]
    return b.finish(Plan("lfm3", n_embd, n_vocab, eps, "sandwich" if sandwich else "pre", layers,
                         tied, context, summary={
                             "heads": n_heads, "head_dim": head_dim, "ff": n_ff,
                             "layers": {k: kinds.count(k) for k in ("conv", "attn", "swa")},
                             "sliding_window": window, "rope_base": base, "rope_base_swa": base_swa}))


def _moe_spec(b: _Builder, i: int, *, gating_default: str, norm_default: bool,
              shared_gate: bool) -> MoESpec:
    md = b.md
    n_expert = _int(md, b.k("expert_count"))
    n_used = _int(md, b.k("expert_used_count"))
    n_ff = _int(md, b.k("expert_feed_forward_length"))
    if not (1 <= n_used <= n_expert <= 4096) or n_ff <= 0:
        raise PlanError("strife.model.bad_metadata",
                        f"expert metadata is inconsistent ({n_used} of {n_expert}, ff {n_ff})")
    groups = _int(md, b.k("expert_group_count"), 1) or 1
    used_groups = _int(md, b.k("expert_group_used_count"), groups) or groups
    if groups > 1 and used_groups < groups:
        raise PlanError("strife.arch.moe_groups",
                        f"group-limited expert routing ({used_groups} of {groups} groups) "
                        "is not implemented in Strife")
    func = _int(md, b.k("expert_gating_func"), 0)
    gating = {1: "softmax", 2: "sigmoid"}.get(func, gating_default)
    p = f"blk.{i}."
    shared_ff = 0
    if b.has(p + "ffn_up_shexp.weight"):
        shared_ff = _int(md, b.k("expert_shared_feed_forward_length"), 0)
        if not shared_ff:
            shared_ff = int(b.shapes[p + "ffn_up_shexp.weight"][1])
    return MoESpec(n_expert, n_used, n_ff, gating,
                   norm=bool(_get(md, b.k("expert_weights_norm"), norm_default)),
                   scale=_float(md, b.k("expert_weights_scale"), 1.0) or 1.0,
                   probs_bias=b.has(p + "exp_probs_b.bias"),
                   fused_gate_up=b.has(p + "ffn_gate_up_exps.weight"),
                   shared_ff=shared_ff,
                   shared_gate=shared_gate and b.has(p + "ffn_gate_inp_shexp.weight"))


def _qwen35(b: _Builder) -> Plan:
    md, arch = b.md, b.arch
    n_embd, n_vocab, context, eps = b.base()
    nextn = _int(md, b.k("nextn_predict_layers"), 0)
    n_layers = _int(md, b.k("block_count")) - nextn
    _bound("block_count", n_layers, 1, 1024)
    n_heads = _int(md, b.k("attention.head_count"))
    n_kv = _int(md, b.k("attention.head_count_kv"), n_heads)
    head_dim = _int(md, b.k("attention.key_length"), 0) or (n_embd // max(1, n_heads))
    n_rot = _int(md, b.k("rope.dimension_count"), head_dim // 4) or head_dim // 4
    base = _float(md, b.k("rope.freq_base"), 1e7)
    recurrent = md.get(b.k("attention.recurrent_layers"))
    if isinstance(recurrent, list) and len(recurrent) >= n_layers:
        is_recr = [bool(x) for x in recurrent[:n_layers]]
    else:
        interval = _int(md, b.k("full_attention_interval"), 4) or 4
        is_recr = [(i + 1) % interval != 0 for i in range(n_layers)]
    gdn = GDNSpec(n_k_heads=_int(md, b.k("ssm.group_count")),
                  n_v_heads=_int(md, b.k("ssm.time_step_rank")),
                  head_k=_int(md, b.k("ssm.state_size")),
                  head_v=0, kernel=_int(md, b.k("ssm.conv_kernel"), 4))
    inner = _int(md, b.k("ssm.inner_size"))
    if min(gdn.n_k_heads, gdn.n_v_heads, gdn.head_k, inner) <= 0 or gdn.n_v_heads % gdn.n_k_heads \
            or inner % gdn.n_v_heads:
        raise PlanError("strife.model.bad_metadata", "the linear-attention (ssm.*) metadata is inconsistent")
    gdn = GDNSpec(gdn.n_k_heads, gdn.n_v_heads, gdn.head_k, inner // gdn.n_v_heads, gdn.kernel)
    if n_heads % max(1, n_kv) or head_dim % 2 or n_rot > head_dim:
        raise PlanError("strife.model.bad_metadata", "the attention metadata is inconsistent")
    attn = AttnSpec(n_heads, n_kv, head_dim, rope="neox", n_rot=n_rot, rope_base=base,
                    rope_scale=b.rope_scale(), qk_norm=True, gate="q")
    moe = arch == "qwen35moe"
    n_ff = 0 if moe else _int(md, b.k("feed_forward_length"))
    layers = []
    for i in range(n_layers):
        p = f"blk.{i}."
        b.need(p + "attn_norm.weight", (n_embd,))
        b.need(p + "post_attention_norm.weight", (n_embd,))
        if is_recr[i]:
            b.need(p + "attn_qkv.weight", (n_embd, gdn.conv_dim))
            b.need(p + "attn_gate.weight", (n_embd, gdn.value_dim))
            b.need(p + "ssm_conv1d.weight", (gdn.kernel, gdn.conv_dim))
            b.need(p + "ssm_dt.bias", (gdn.n_v_heads,))
            b.need(p + "ssm_a", (gdn.n_v_heads,))
            b.need(p + "ssm_beta.weight", (n_embd, gdn.n_v_heads))
            b.need(p + "ssm_alpha.weight", (n_embd, gdn.n_v_heads))
            b.need(p + "ssm_norm.weight", (gdn.head_v,))
            b.need(p + "ssm_out.weight", (gdn.value_dim, n_embd))
            spec = LayerSpec(i, "gdn", gdn=gdn)
        else:
            b.attention(i, n_embd, attn)
            spec = LayerSpec(i, "attn", attn=attn)
        if moe:
            m = _moe_spec(b, i, gating_default="softmax", norm_default=True, shared_gate=True)
            b.moe_ffn(i, n_embd, m)
            spec = LayerSpec(spec.index, spec.mixer, spec.attn, spec.conv, spec.gdn, "moe", 0, m)
        else:
            b.dense_ffn(i, n_embd, n_ff)
            spec = LayerSpec(spec.index, spec.mixer, spec.attn, spec.conv, spec.gdn, "dense", n_ff)
        layers.append(spec)
    tied = b.output(n_embd, n_vocab)
    return b.finish(Plan(arch, n_embd, n_vocab, eps, "post_attn_ffn", layers, tied, context, summary={
        "heads": n_heads, "kv_heads": n_kv, "head_dim": head_dim, "rope_dims": n_rot,
        "rope_base": base, "linear_layers": sum(is_recr), "attention_layers": n_layers - sum(is_recr),
        "linear_heads": f"{gdn.n_k_heads} k / {gdn.n_v_heads} v of {gdn.head_k}",
        "mtp_layers_skipped": nextn}))


def _glm4moe(b: _Builder) -> Plan:
    md = b.md
    n_embd, n_vocab, context, eps = b.base()
    nextn = _int(md, b.k("nextn_predict_layers"), 0)
    n_layers = _int(md, b.k("block_count")) - nextn
    _bound("block_count", n_layers, 1, 1024)
    n_heads = _int(md, b.k("attention.head_count"))
    n_kv = _int(md, b.k("attention.head_count_kv"), n_heads)
    head_dim = _int(md, b.k("attention.key_length"), 0) or (n_embd // max(1, n_heads))
    if n_heads % max(1, n_kv) or head_dim % 2:
        raise PlanError("strife.model.bad_metadata", "the attention metadata is inconsistent")
    n_rot = _int(md, b.k("rope.dimension_count"), head_dim // 2) or head_dim // 2
    lead = _int(md, b.k("leading_dense_block_count"), 0)
    n_ff = _int(md, b.k("feed_forward_length"), 0)
    attn = AttnSpec(n_heads, n_kv, head_dim, rope="neox", n_rot=n_rot,
                    rope_base=_float(md, b.k("rope.freq_base"), 1e6), rope_scale=b.rope_scale(),
                    qkv_bias=b.has("blk.0.attn_q.bias"), qk_norm=b.has("blk.0.attn_q_norm.weight"))
    layers = []
    for i in range(n_layers):
        p = f"blk.{i}."
        b.need(p + "attn_norm.weight", (n_embd,))
        b.need(p + "post_attention_norm.weight", (n_embd,))
        b.attention(i, n_embd, attn)
        if i < lead:
            b.dense_ffn(i, n_embd, n_ff)
            layers.append(LayerSpec(i, "attn", attn=attn, ffn="dense", n_ff=n_ff))
        else:
            m = _moe_spec(b, i, gating_default="sigmoid", norm_default=False, shared_gate=False)
            b.moe_ffn(i, n_embd, m)
            layers.append(LayerSpec(i, "attn", attn=attn, ffn="moe", moe=m))
    tied = b.output(n_embd, n_vocab)
    first_moe = next((lay.moe for lay in layers if lay.moe), None)
    return b.finish(Plan("glm4moe", n_embd, n_vocab, eps, "post_attn_ffn", layers, tied, context,
                         summary={
                             "heads": n_heads, "kv_heads": n_kv, "head_dim": head_dim, "rope_dims": n_rot,
                             "dense_layers": min(lead, n_layers),
                             "experts": f"{first_moe.n_used} of {first_moe.n_expert}" if first_moe else "",
                             "mtp_layers_skipped": nextn}))


_BUILDERS = {"llama": _llama_family, "mistral": _llama_family, "qwen2": _llama_family,
             "qwen3": _llama_family, "lfm3": _lfm3, "qwen35": _qwen35, "qwen35moe": _qwen35,
             "glm4moe": _glm4moe}


def build_plan(metadata: dict[str, Any], shapes: dict[str, tuple[int, ...]]) -> Plan:
    """The plan for a model, or :class:`PlanError` saying why there is none."""
    arch = str(metadata.get("general.architecture") or "")
    builder = _BUILDERS.get(arch)
    if builder is None:
        raise PlanError("strife.arch.unsupported",
                        f"architecture {arch or '(none)'!r} is not one Strife runs",
                        "Strife runs: " + ", ".join(ARCHITECTURES)
                        + ". Use the llama.cpp runner (hnx-t1 runner start) for others.")
    return builder(_Builder(metadata, shapes))
