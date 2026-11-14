#!/usr/bin/env python3
"""Distil Compactbot/hypernix-3.1-mini (V2) into native HyperNix0xV3.

Compatible V2 weights are transferred explicitly. V3-only Tidepool and Keel
parameters keep their native initialization and learn through teacher-logit
KL distillation plus next-token cross-entropy. CUDA is mandatory; no CPU
fallback is permitted.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.optim import AdamW

TEACHER_ID = "Compactbot/hypernix-3.1-mini"
DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
VOCAB = 32_000
WIDTH = 512
LAYERS = 8
FF_WIDTH = 2203
TEACHER_CONTEXT = 512


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default=TEACHER_ID)
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--dataset-config", default=DATASET_CONFIG)
    p.add_argument("--split", default="train")
    p.add_argument("--eos-token-id", type=int, default=None,
                   help="Explicit document separator token ID; defaults to tokenizer EOS, or no separator if absent.")
    p.add_argument("--output", type=Path, default=Path("outputs/hypernix0xv3-distilled"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--student-layers", type=int, default=8)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--slot-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kd-weight", type=float, default=0.7)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--amp", choices=("fp16", "off"), default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build. CPU fallback is disabled."
        )
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    free, _ = torch.cuda.mem_get_info(device)
    print(f"[gpu] {props.name}; {props.total_memory / 2**30:.2f} GiB total; "
          f"{free / 2**30:.2f} GiB free")
    if (props.major, props.minor) < (6, 1):
        raise RuntimeError(f"Unsupported CUDA compute capability {props.major}.{props.minor}; need >= 6.1.")
    return device


def acquire_lock(output: Path):
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Training lock currently requires Linux/Unix fcntl.") from exc
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / ".train.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another trainer owns {output / '.train.lock'}") from exc
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def download_teacher(repo_id: str):
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub and transformers.") from exc
    folder = Path(snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "vocab.json", "merges.txt"],
    ))
    if not (folder / "model.safetensors").is_file():
        raise FileNotFoundError(f"Expected model.safetensors in {folder}")
    tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True, trust_remote_code=False)
    return folder, tokenizer


def build_teacher(folder: Path, device: torch.device):
    from safetensors.torch import load_file
    from hypernix.training.brewer import BrewerConfig, BrewerModel

    cfg = BrewerConfig(
        vocab_size=VOCAB, n_layers=LAYERS, n_heads=8, n_kv_heads=2,
        d_model=WIDTH, d_ff=FF_WIDTH, max_seq_len=TEACHER_CONTEXT,
        rope_theta=100_000.0, norm_eps=1e-5, tie_embeddings=True,
        use_sliding_window=False, attention_type="gqa",
        name="hypernix.3.1-mini",
    )
    model = BrewerModel(cfg)
    state = load_file(str(folder / "model.safetensors"), device="cpu")
    result = model.load_state_dict(state, strict=False)
    missing = set(result.missing_keys)
    if missing - {"lm_head.weight"} or result.unexpected_keys:
        raise RuntimeError(
            f"Teacher checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.lm_head.weight = model.embed.embed.weight
    model.to(device).eval().requires_grad_(False)
    return model


def transfer_v2_to_v3(teacher: torch.nn.Module, student: torch.nn.Module) -> dict[str, Any]:
    """Copy shared representations; leave genuinely new V3 mechanisms trainable."""
    src, dst = teacher.state_dict(), student.state_dict()
    copied: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    def put(target: str, source: str):
        if source not in src or target not in dst:
            skipped.append(f"{source} -> {target}: missing")
        elif src[source].shape != dst[target].shape:
            skipped.append(f"{source} -> {target}: shape mismatch")
        else:
            copied[target] = src[source].detach().clone()

    put("embed.weight", "embed.embed.weight")
    put("norm_out.weight", "norm_out.weight")
    n_student, n_teacher = len(student.blocks), len(teacher.blocks)
    for i in range(n_student):
        j = round(i * (n_teacher - 1) / max(1, n_student - 1))
        put(f"blocks.{i}.norm1.weight", f"blocks.{j}.norm1.weight")
        put(f"blocks.{i}.norm3.weight", f"blocks.{j}.norm2.weight")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            put(f"blocks.{i}.tidepool.{name}.weight", f"blocks.{j}.attn.{name}.weight")
        for name in ("gate_proj", "up_proj", "down_proj"):
            put(f"blocks.{i}.ffn.{name}.weight", f"blocks.{j}.ffn.{name}.weight")

    result = student.load_state_dict(copied, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected transferred keys: {result.unexpected_keys}")
    student.tie_weights()
    return {
        "copied_tensors": len(copied),
        "copied_parameters": sum(t.numel() for t in copied.values()),
        "new_v3_tensors": sorted(set(result.missing_keys)),
        "skipped": skipped,
    }


def row_to_text(row: dict[str, Any]) -> str:
    for key in ("text", "content", "document", "article", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages") or row.get("conversation")
    if isinstance(messages, list):
        parts = []
        for item in messages:
            if isinstance(item, dict):
                role = item.get("role", item.get("from", "user"))
                body = item.get("content", item.get("value", ""))
                if isinstance(body, str) and body.strip():
                    parts.append(f"{role}: {body.strip()}")
        return "\n".join(parts)
    return ""


def resolve_eos_token_id(tokenizer, override: int | None = None, vocab_size: int = VOCAB) -> int | None:
    """Return a verified document separator, or None when the tokenizer has no EOS."""
    eos = override if override is not None else getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return None
    if not isinstance(eos, int) or not 0 <= eos < vocab_size:
        raise ValueError(f"EOS token ID {eos!r} is outside the model vocabulary (0..{vocab_size - 1}).")
    return eos


def make_blocks(args, tokenizer, validation: bool) -> Iterator[torch.Tensor]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets package.") from exc
    stream = load_dataset(args.dataset, args.dataset_config or None,
                          split=args.split, streaming=True)
    eos = resolve_eos_token_id(tokenizer, getattr(args, "eos_token_id", None))
    token_buffer: list[int] = []
    emitted = 0
    for index, row in enumerate(stream):
        # Hold out every 100th source row before token packing.
        if (index % 100 == 0) != validation:
            continue
        text = row_to_text(row)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            continue
        token_buffer.extend(ids)
        if eos is not None:
            token_buffer.append(eos)
        while len(token_buffer) >= args.seq_len + 1:
            block = token_buffer[:args.seq_len + 1]
            del token_buffer[:args.seq_len]
            emitted += 1
            yield torch.tensor(block, dtype=torch.long)
    if emitted == 0:
        raise RuntimeError(f"No {'validation' if validation else 'training'} blocks. Check dataset fields/config.")


def next_batch(iterator, batch_size: int, device: torch.device):
    packed = torch.stack([next(iterator) for _ in range(batch_size)]).to(device, non_blocking=True)
    return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()


def amp_context(device: torch.device, amp: str):
    if device.type == "cuda" and amp == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, kd_weight: float):
    # x contains seq_len inputs and labels contains the corresponding
    # seq_len shifted targets; every output position has a target.
    student = student_logits.float()
    teacher = teacher_logits.float()
    log_student = F.log_softmax(student / temperature, dim=-1)
    teacher_prob = F.softmax(teacher / temperature, dim=-1)
    kd = F.kl_div(log_student, teacher_prob, reduction="none").sum(-1).mean()
    kd = kd * temperature**2
    ce = F.cross_entropy(student.reshape(-1, student.shape[-1]), labels.reshape(-1))
    return kd_weight * kd + (1.0 - kd_weight) * ce, kd.detach(), ce.detach()


@torch.no_grad()
def evaluate(student, teacher, tokenizer, args, device):
    student.eval()
    val_iter = make_blocks(args, tokenizer, validation=True)
    sums = torch.zeros(3, device=device)
    count = 0
    for _ in range(args.eval_batches):
        try:
            x, y = next_batch(val_iter, args.batch_size, device)
        except StopIteration:
            break
        with amp_context(device, args.amp):
            t_logits = teacher(x)
            s_logits = student(x)
        loss, kd, ce = distillation_loss(s_logits, t_logits, y, args.temperature, args.kd_weight)
        sums += torch.stack((loss, kd, ce))
        count += 1
    student.train()
    if count == 0:
        raise RuntimeError("Validation stream produced no batches.")
    return (sums / count).tolist()


def atomic_save(payload: dict[str, Any], path: Path):
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def export_model(student, cfg, tokenizer, folder: Path, metadata: dict[str, Any]):
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    cfg.save(folder / "config.json")
    tokenizer.save_pretrained(folder)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in student.state_dict().items()
        if not (cfg.tie_embeddings and key == "lm_head.weight")
    }
    save_file(state, str(folder / "model.safetensors"))
    (folder / "distillation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if not 0 <= args.kd_weight <= 1:
        raise ValueError("--kd-weight must be between 0 and 1")
    if not 8 <= args.seq_len <= TEACHER_CONTEXT:
        raise ValueError(f"--seq-len must be 8..{TEACHER_CONTEXT}, the teacher's supported context")
    if min(args.batch_size, args.grad_accum, args.steps, args.student_layers) < 1:
        raise ValueError("batch size, accumulation, steps, and student layers must be positive")
    if args.chunk > args.seq_len:
        raise ValueError("--chunk cannot exceed --seq-len")

    device = require_cuda()
    lock = acquire_lock(args.output)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        print(f"[teacher] loading {args.teacher}")
        folder, tokenizer = download_teacher(args.teacher)
        if tokenizer.vocab_size != VOCAB or len(tokenizer) > VOCAB:
            raise RuntimeError(
                f"Tokenizer vocabulary mismatch: vocab_size={tokenizer.vocab_size}, "
                f"len={len(tokenizer)}, expected {VOCAB}"
            )
        eos_id = resolve_eos_token_id(tokenizer, args.eos_token_id)
        if eos_id is None:
            print("[data] tokenizer defines no EOS token; packing documents continuously without an invented separator")
        else:
            print(f"[data] using EOS token ID {eos_id} between documents")
        teacher = build_teacher(folder, device)

        from hypernix.models.hnx0xv3 import HyperNix0xV3Config, HyperNix0xV3Model
        cfg = HyperNix0xV3Config(
            vocab_size=VOCAB, d_model=WIDTH, n_layers=args.student_layers,
            n_heads=8, n_kv_heads=2, d_ff=FF_WIDTH, max_seq_len=args.seq_len,
            rope_theta=100_000.0, window=args.window, chunk=args.chunk,
            n_slots=args.slots, slot_dim=args.slot_dim, norm_eps=1e-5,
            dropout=0.0, tie_embeddings=True,
            precision="fp16" if args.amp == "fp16" else "fp32",
            name="hypernix0x-v3-distilled-3.1-mini",
        )
        student = HyperNix0xV3Model(cfg).to(device)
        transfer = transfer_v2_to_v3(teacher, student)
        print(f"[transfer] copied {transfer['copied_tensors']} tensors "
              f"({transfer['copied_parameters']:,} parameters); "
              f"{len(transfer['new_v3_tensors'])} V3-specific tensors remain trainable")

        optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16")
        start_step, best_val = 0, float("inf")
        checkpoint = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            if checkpoint.get("arch") != "hypernix0x-v3":
                raise RuntimeError("Resume file is not a HyperNix0xV3 training checkpoint.")
            student.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            start_step = int(checkpoint["step"])
            best_val = float(checkpoint.get("best_val", best_val))
            print(f"[resume] step {start_step}")

        def lr_scale(step):
            if step < args.warmup_steps:
                return max(1e-4, (step + 1) / max(1, args.warmup_steps))
            progress = min(1.0, (step - args.warmup_steps) /
                           max(1, args.steps - args.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        if checkpoint and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])

        train_iter = make_blocks(args, tokenizer, validation=False)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        running = torch.zeros(3, device=device)
        for step in range(start_step + 1, args.steps + 1):
            for _ in range(args.grad_accum):
                x, y = next_batch(train_iter, args.batch_size, device)
                with torch.no_grad(), amp_context(device, args.amp):
                    teacher_logits = teacher(x)
                with amp_context(device, args.amp):
                    student_logits = student(x)
                    loss, kd, ce = distillation_loss(
                        student_logits, teacher_logits, y, args.temperature, args.kd_weight
                    )
                    if cfg.n_experts:
                        loss = loss + student.aux_loss
                scaler.scale(loss / args.grad_accum).backward()
                running += torch.stack((loss.detach(), kd, ce)) / args.grad_accum

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 25 == 0 or step == 1:
                divisor = min(25, step)
                vals = (running / divisor).tolist()
                print(f"[train] {step}/{args.steps} loss={vals[0]:.4f} "
                      f"kd={vals[1]:.4f} ce={vals[2]:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"vram={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB")
                running.zero_()

            if step % args.eval_every == 0 or step == args.steps:
                val_loss, val_kd, val_ce = evaluate(student, teacher, tokenizer, args, device)
                print(f"[valid] step={step} loss={val_loss:.4f} kd={val_kd:.4f} ce={val_ce:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    export_model(student, cfg, tokenizer, args.output / "best", {
                        "teacher": args.teacher, "dataset": args.dataset,
                        "dataset_config": args.dataset_config, "step": step,
                        "best_validation_loss": best_val, "transfer": transfer,
                        "objective": "temperature-scaled KL + next-token CE",
                    })
                    print(f"[save] new best: {args.output / 'best'}")

            if step % args.save_every == 0 or step == args.steps:
                atomic_save({
                    "arch": "hypernix0x-v3", "step": step, "best_val": best_val,
                    "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg.to_dict(), "teacher": args.teacher,
                    "dataset": args.dataset, "seed": args.seed,
                }, args.output / "training_state.pt")

        export_model(student, cfg, tokenizer, args.output / "final", {
            "teacher": args.teacher, "dataset": args.dataset,
            "dataset_config": args.dataset_config, "steps": args.steps,
            "best_validation_loss": best_val, "transfer": transfer,
            "objective": "temperature-scaled KL + next-token CE",
        })
        print(f"[done] best={args.output / 'best'} final={args.output / 'final'}")
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA OOM. Retry with --seq-len 256, --batch-size 1, or --grad-accum 4. "
            "The script did not fall back to CPU."
        ) from exc
    finally:
        lock.close()


if __name__ == "__main__":
    main()
