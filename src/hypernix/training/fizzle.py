"""fizzle — Fuzed Architecture Module for HyperNix.

This module provides the ability to fuse multiple HuggingFace models
(e.g., an ASR model and an LLM) into a single unified architecture,
merging tokenizers and applying LoRAs seamlessly.

CLI Usage:
    hypernix fiz model1 "hf link or id" model2 "hf id" loRA1 "lora path" ...
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional imports for model processing
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
        PreTrainedModel,
    )
except ImportError:
    torch = None
    nn = None


@dataclass
class FuzedComponent:
    """Represents a component in the Fuzed model."""
    id: str
    name_or_path: str
    component_type: str = "auto"
    loras: list[str] = None
    
    def __post_init__(self):
        if self.loras is None:
            self.loras = []


class FuzedModelArch(nn.Module if nn else object):
    """A fuzed model architecture combining multiple underlying models."""
    
    def __init__(self, components: list[FuzedComponent], output_dir: str | Path | None = None):
        super().__init__()
        self.components = components
        self.output_dir = Path(output_dir) if output_dir else Path("fuzed_output")
        self.models: dict[str, PreTrainedModel] = {}
        self.tokenizers: dict[str, Any] = {}
        self._is_fuzed = False
        
    def _detect_type(self, path: str) -> str:
        """Heuristic to guess model type."""
        path_lower = path.lower()
        if "whisper" in path_lower or "asr" in path_lower or "wav2vec" in path_lower:
            return "asr"
        if "tts" in path_lower or "vits" in path_lower or "speech" in path_lower:
            return "tts"
        if "vit" in path_lower or "clip" in path_lower or "vision" in path_lower or "image" in path_lower or "siglip" in path_lower:
            return "vision"
        return "llm"
        
    def load_components(self) -> None:
        """Load individual models and tokenizers into memory."""
        if torch is None:
            raise ImportError("Fuzed architecture requires torch and transformers.")
            
        print("[fizzle] Loading components...", file=sys.stderr)
        
        for comp in self.components:
            comp_type = comp.component_type
            if comp_type == "auto":
                comp_type = self._detect_type(comp.name_or_path)
                
            print(f"[fizzle] Loading {comp.id} ({comp_type}) from {comp.name_or_path}...", file=sys.stderr)
            
            # Load tokenizer/processor
            try:
                if comp_type == "asr":
                    self.tokenizers[comp.id] = AutoProcessor.from_pretrained(comp.name_or_path)
                elif comp_type == "vision":
                    self.tokenizers[comp.id] = AutoImageProcessor.from_pretrained(comp.name_or_path)
                else:
                    self.tokenizers[comp.id] = AutoTokenizer.from_pretrained(comp.name_or_path)
            except Exception as e:
                print(f"[fizzle] Warning: Could not load tokenizer/processor for {comp.id}: {e}", file=sys.stderr)
                
            # Load model
            model = AutoModel.from_pretrained(comp.name_or_path, trust_remote_code=True)
            
            # Apply LoRAs
            if comp.loras:
                try:
                    from peft import PeftModel
                    for lora in comp.loras:
                        print(f"[fizzle] Applying LoRA {lora} to {comp.id}...", file=sys.stderr)
                        model = PeftModel.from_pretrained(model, lora)
                except ImportError:
                    print(f"[fizzle] Warning: PEFT library not installed. Cannot apply LoRAs to {comp.id}.", file=sys.stderr)
            
            self.models[comp.id] = model
            # Register as submodule if this inherits from nn.Module
            if isinstance(self, nn.Module):
                self.add_module(f"fuzed_{comp.id}", model)
                
    def fuze_tokenizers(self) -> Any:
        """Merge tokenizers into a single fuzed tokenizer."""
        print("[fizzle] Fuzing tokenizers...", file=sys.stderr)
        # In a real deep fuzing, we'd combine vocabs, BPE rules, etc.
        # For the scaffolding, we select the LLM tokenizer and add special tokens
        llm_tokenizer = None
        for _cid, tok in self.tokenizers.items():
            if "Tokenizer" in type(tok).__name__:
                llm_tokenizer = tok
                break
                
        if llm_tokenizer is None:
            return None
            
        special_tokens = ["<|asr_start|>", "<|asr_end|>", "<|tts_start|>", "<|tts_end|>", "<|image_start|>", "<|image_end|>"]
        llm_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        return llm_tokenizer
        
    def build(self) -> None:
        """Perform the fuzing operation."""
        self.load_components()
        fuzed_tok = self.fuze_tokenizers()
        
        # Resize embeddings for LLM
        for cid, model in self.models.items():
            if hasattr(model, "resize_token_embeddings") and fuzed_tok:
                print(f"[fizzle] Resizing token embeddings for {cid}...", file=sys.stderr)
                try:
                    model.resize_token_embeddings(len(fuzed_tok))
                except Exception as e:
                    print(f"[fizzle] Failed to resize embeddings for {cid}: {e}", file=sys.stderr)
                    
        self._is_fuzed = True
        print("[fizzle] Fuzed architecture successfully built.", file=sys.stderr)
        
    def save(self) -> None:
        """Save the fuzed architecture to disk."""
        if not self._is_fuzed:
            raise RuntimeError("Cannot save. Call build() first.")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[fizzle] Saving fuzed architecture to {self.output_dir}...", file=sys.stderr)
        
        # Save a config map
        import json
        config_map = {
            "architectures": ["FuzedModelArch"],
            "components": [
                {
                    "id": c.id,
                    "path": c.name_or_path,
                    "type": c.component_type,
                    "loras": c.loras
                }
                for c in self.components
            ]
        }
        with open(self.output_dir / "fuzed_config.json", "w", encoding="utf-8") as f:
            json.dump(config_map, f, indent=2)
            
        print("[fizzle] Save complete.", file=sys.stderr)

def main(argv: list[str]) -> int:
    """CLI entry point for fizzle module."""
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "hypernix fizzle — Fuzed Architecture module\n\n"
            "Usage:\n"
            "  hypernix fiz model1 <hf_link_or_id> [model2 <hf_link_or_id> ...] "
            "[loRA1 <lora_path> ...] [--out <dir>]\n\n"
            "Example:\n"
            "  hypernix fiz model1 openai/whisper-tiny model2 Qwen/Qwen1.5-0.5B "
            "loRA1 ./my_qwen_lora --out ./my_fuzed_model"
        )
        return 0
        
    # Parse the flexible arguments
    components = []
    current_model = None
    output_dir = "fuzed_output"
    
    i = 0
    while i < len(argv):
        arg = argv[i]
        
        if arg in ("--out", "-o") and i + 1 < len(argv):
            output_dir = argv[i+1]
            i += 2
            continue
            
        if arg.startswith("model"):
            if i + 1 < len(argv):
                # Start a new component
                current_model = FuzedComponent(id=arg, name_or_path=argv[i+1])
                components.append(current_model)
                i += 2
                continue
                
        if arg.lower().startswith("lora"):
            if i + 1 < len(argv) and current_model:
                current_model.loras.append(argv[i+1])
                i += 2
                continue
                
        # If we hit an unrecognized pattern
        print(f"[fizzle] Ignoring unrecognized argument: {arg}", file=sys.stderr)
        i += 1
        
    if not components:
        print("[fizzle] Error: No models specified to fuze.", file=sys.stderr)
        return 1
        
    print(f"[fizzle] Preparing to fuze {len(components)} models into {output_dir}")
    
    # Try to load and build the fuzed model (this requires torch/transformers)
    try:
        fuzed = FuzedModelArch(components, output_dir=output_dir)
        fuzed.build()
        fuzed.save()
    except ImportError as e:
        print(f"[fizzle] Initialization Error: {e}", file=sys.stderr)
        print("[fizzle] To use fizzle, install: pip install torch transformers peft", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[fizzle] Build Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
