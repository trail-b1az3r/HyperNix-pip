"""hypernix.camouflage — RLHF / RLAF Alignment Module.

Implements a real REINFORCE-style training loop for model alignment.
"""
from __future__ import annotations

import argparse
import sys

from hypernix.security.hfpin import hub_revision

# Separately: torch without transformers can still score a reply, and a
# missing transformers used to make torch look missing too.
try:
    import torch
    from torch.optim import AdamW
except ImportError:
    torch = None
    AdamW = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = PreTrainedModel = None


def _get_reward(
    evaluator_model: PreTrainedModel | None, 
    eval_tokenizer, 
    prompt: str, 
    response: str, 
    sys_prompt: str,
    device: str
) -> float:
    """The evaluator's 1-10 score, or without one (no -Ai) a length/refusal heuristic."""
    if evaluator_model is None or eval_tokenizer is None:
        # Fallback to heuristic length/keyword reward if -Ai is off
        score = 5.0
        if "sorry" in response.lower() or "as an ai" in response.lower():
            score -= 3.0
        score += min(len(response.split()) / 10.0, 3.0)  # slightly reward length
        return score

    # RLAF AI mode
    full_prompt = f"{sys_prompt}\n\nPrompt: {prompt}\nResponse: {response}\n\nScore (1-10):"
    inputs = eval_tokenizer(full_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = evaluator_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    out_text = eval_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # The first number in the reply. This pattern used to be written
    # r"\\d+..." -- a literal backslash then "d" -- so it never matched,
    # every score was the 5.0 fallback, and RLAF trained on nothing.
    import re

    match = re.search(r"\d+(?:\.\d+)?", out_text)
    return min(10.0, max(0.0, float(match.group(0)))) if match else 5.0


DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a polite refusal to a meeting.",
    "How do I sort a list in Python?",
    "What is the capital of France?",
    "Tell me a creative story about a clockwork bird.",
]


def read_prompts(path: str) -> list[str]:
    """Prompts from a text file (one a line) or JSONL ({"prompt": ...})."""
    import json
    from pathlib import Path

    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            line = str(row.get("prompt") or row.get("instruction") or row.get("text") or "")
        if line:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"{path} has no prompts")
    return prompts


def _load(source: str, *, dtype, device_map):
    """A causal LM and its tokenizer from a folder, a repo id, a Hub link or a .gguf."""
    from pathlib import Path

    from hypernix.models.hflinks import LinkError, parse_source

    path = Path(source).expanduser()
    if path.suffix.lower() == ".gguf" and path.is_file():
        kw = {"gguf_file": path.name}
        model = AutoModelForCausalLM.from_pretrained(str(path.parent), torch_dtype=dtype,
                                                     device_map=device_map, **kw)
        return model, AutoTokenizer.from_pretrained(str(path.parent), **kw)
    if not path.exists():
        try:
            parsed = parse_source(source)
        except LinkError:
            parsed = None
        if parsed is not None:
            if parsed.filename and parsed.filename.lower().endswith(".gguf"):
                kw = {"gguf_file": parsed.filename}
                model = AutoModelForCausalLM.from_pretrained(parsed.repo_id, torch_dtype=dtype,
                                                             device_map=device_map, **kw)
                return model, AutoTokenizer.from_pretrained(parsed.repo_id, **kw)
            source = parsed.repo_id
    model = AutoModelForCausalLM.from_pretrained(source, revision=hub_revision(source),
                                                 torch_dtype=dtype, device_map=device_map)
    return model, AutoTokenizer.from_pretrained(source, revision=hub_revision(source))


def run_rlhf(
    local_model: str,
    steps: int,
    use_ai: bool = False,
    evaluator_path: str | None = None,
    sys_prompt: str = "",
    prompts: list[str] | None = None,
    out_dir: str = "camo_aligned_model",
    lr: float = 1e-5,
    max_new_tokens: int = 64,
) -> int:
    if torch is None or AutoModelForCausalLM is None:
        print("[camo] Error: torch and transformers are required to run Camouflage.", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"[camo] Initializing on {device}...")

    print(f"[camo] Loading target policy model: {local_model}")
    try:
        policy, tokenizer = _load(local_model, dtype=torch.float32, device_map=device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"[camo] Failed to load target model: {e}")
        return 1

    optimizer = AdamW(policy.parameters(), lr=lr)

    eval_model = None
    eval_tokenizer = None
    if use_ai:
        if not evaluator_path:
            print("[camo] Error: -Ai specified but no evaluator model (-M) provided.")
            return 1
        print(f"[camo] Loading AI evaluator: {evaluator_path}")
        try:
            # We load the evaluator in 8-bit or half precision to save memory on single-gpu setups
            eval_model, eval_tokenizer = _load(
                evaluator_path, dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device)
            eval_model.eval()
        except Exception as e:
            print(f"[camo] Failed to load evaluator model: {e}")
            return 1

    prompts = prompts or DEFAULT_PROMPTS
    if not use_ai:
        print("[camo] No -Ai evaluator: rewards come from a length/refusal heuristic.")

    print(f"[camo] Starting RLHF/RLAF alignment loop for {steps} steps...")
    policy.train()
    
    baseline = 5.0  # Simple baseline for advantage computation

    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generation phase (inference)
        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                      temperature=0.7, pad_token_id=tokenizer.pad_token_id)
        
        response = tokenizer.decode(gen_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Reward phase
        reward = _get_reward(eval_model, eval_tokenizer, prompt, response, sys_prompt, device)
        advantage = reward - baseline
        baseline = 0.9 * baseline + 0.1 * reward  # EMA baseline
        
        # Training phase (REINFORCE)
        policy.train()
        optimizer.zero_grad()
        
        # The tokens that were sampled, not a re-tokenisation of their
        # text: the log-probability has to be of what was generated.
        labels = gen_out.clone()
        labels[:, :inputs.input_ids.shape[1]] = -100  # Mask the prompt
        if bool((labels != -100).any()):
            outputs = policy(input_ids=gen_out, attention_mask=torch.ones_like(gen_out), labels=labels)
            loss = outputs.loss
        else:  # nothing was generated: nothing to learn from
            loss = torch.zeros((), device=gen_out.device, requires_grad=True)

        # REINFORCE minimises -advantage * log p, and loss is -log p (NLL),
        # so the product is advantage * NLL. It was NLL * -advantage, which
        # made every well-scored answer *less* likely.
        rl_loss = loss * advantage
        
        rl_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        print(f"[camo] Step {step+1}/{steps} | Reward: {reward:.2f} | Adv: {advantage:.2f} | Loss: {rl_loss.item():.4f}")
        
    print("[camo] Alignment complete.")
    print(f"[camo] Saving aligned model to {out_dir}...")
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="hypernix camo", 
        description="Camouflage — RLHF / RLAF model alignment module."
    )
    p.add_argument("-Ai", action="store_true", help="Enable AI-assisted evaluation (RLAF).")
    p.add_argument("-M", dest="eval_model", type=str, help="Local GGUF, PyTorch dir, HF ID, or HF Link for the evaluator AI.")
    p.add_argument("-Sp", dest="sys_prompt", type=str, default="You are an expert alignment evaluator. Score the following response out of 10. Reply only with the number.", help="System prompt for the evaluator AI.")
    p.add_argument("-Lmodel", dest="local_model", type=str, required=True, help="Local model to align (PyTorch dir, HF Link, or HF ID).")
    p.add_argument("-s", dest="steps", type=int, default=100, help="Number of alignment steps.")
    p.add_argument("-P", "--prompts", default=None,
                   help="Prompts to align on: a text file (one a line) or JSONL with \"prompt\".")
    p.add_argument("-o", "--output", default="camo_aligned_model", help="Where the aligned model goes.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-new-tokens", type=int, default=64)

    args = p.parse_args(argv)

    return run_rlhf(
        local_model=args.local_model,
        steps=args.steps,
        use_ai=args.Ai,
        evaluator_path=args.eval_model,
        sys_prompt=args.sys_prompt,
        prompts=read_prompts(args.prompts) if args.prompts else None,
        out_dir=args.output,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
