"""hypernix.camouflage — RLHF / RLAF Alignment Module.

Implements a real REINFORCE-style training loop for model alignment.
"""
from __future__ import annotations

import argparse
import sys

from hypernix.security.hfpin import hub_revision

try:
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
except ImportError:
    torch = None
    AdamW = None


def _get_reward(
    evaluator_model: PreTrainedModel | None, 
    eval_tokenizer, 
    prompt: str, 
    response: str, 
    sys_prompt: str,
    device: str
) -> float:
    """Query the AI evaluator for a reward score, or return a mock heuristic score if not using AI."""
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
    try:
        # Very simple extraction of the first number in the output
        import re
        match = re.search(r"\\d+(\\.\\d+)?", out_text)
        if match:
            return float(match.group(0))
        return 5.0
    except Exception:
        return 5.0


def run_rlhf(
    local_model: str, 
    steps: int, 
    use_ai: bool = False, 
    evaluator_path: str | None = None, 
    sys_prompt: str = ""
) -> int:
    if torch is None:
        print("[camo] Error: torch and transformers are required to run Camouflage.", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"[camo] Initializing on {device}...")

    print(f"[camo] Loading target policy model: {local_model}")
    try:
        policy = AutoModelForCausalLM.from_pretrained(local_model, revision=hub_revision(local_model), torch_dtype=torch.float16 if device != "cpu" else torch.float32, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(local_model, revision=hub_revision(local_model))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"[camo] Failed to load target model: {e}")
        return 1

    optimizer = AdamW(policy.parameters(), lr=1e-5)

    eval_model = None
    eval_tokenizer = None
    if use_ai:
        if not evaluator_path:
            print("[camo] Error: -Ai specified but no evaluator model (-M) provided.")
            return 1
        print(f"[camo] Loading AI evaluator: {evaluator_path}")
        try:
            # We load the evaluator in 8-bit or half precision to save memory on single-gpu setups
            eval_model = AutoModelForCausalLM.from_pretrained(evaluator_path, revision=hub_revision(evaluator_path), torch_dtype=torch.float16, device_map="auto")
            eval_tokenizer = AutoTokenizer.from_pretrained(evaluator_path, revision=hub_revision(evaluator_path))
            eval_model.eval()
        except Exception as e:
            print(f"[camo] Failed to load evaluator model: {e}")
            return 1

    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a polite refusal to a meeting.",
        "How do I sort a list in Python?",
        "What is the capital of France?",
        "Tell me a creative story about a clockwork bird."
    ]

    print(f"[camo] Starting RLHF/RLAF alignment loop for {steps} steps...")
    policy.train()
    
    baseline = 5.0  # Simple baseline for advantage computation

    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generation phase (inference)
        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
        
        response = tokenizer.decode(gen_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Reward phase
        reward = _get_reward(eval_model, eval_tokenizer, prompt, response, sys_prompt, device)
        advantage = reward - baseline
        baseline = 0.9 * baseline + 0.1 * reward  # EMA baseline
        
        # Training phase (REINFORCE)
        policy.train()
        optimizer.zero_grad()
        
        # Re-run forward pass to get gradients on the generated sequence
        train_inputs = tokenizer(prompt + " " + response, return_tensors="pt").to(device)
        labels = train_inputs.input_ids.clone()
        labels[:, :inputs.input_ids.shape[1]] = -100  # Mask the prompt
        
        outputs = policy(**train_inputs, labels=labels)
        loss = outputs.loss
        
        # Weight loss by advantage (negative because PyTorch minimizes, and loss is NLL)
        rl_loss = loss * -advantage
        
        rl_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        print(f"[camo] Step {step+1}/{steps} | Reward: {reward:.2f} | Adv: {advantage:.2f} | Loss: {rl_loss.item():.4f}")
        
    print("[camo] Alignment complete.")
    out_dir = "camo_aligned_model"
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

    args = p.parse_args(argv)

    return run_rlhf(
        local_model=args.local_model,
        steps=args.steps,
        use_ai=args.Ai,
        evaluator_path=args.eval_model,
        sys_prompt=args.sys_prompt
    )

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
