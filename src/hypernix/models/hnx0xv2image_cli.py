"""``hnx image`` — the hnx0xv2image image-to-text model.

::

    hnx image new -o ./eye --preset small            an untrained image model
    hnx image fuse ./eye LiquidAI/LFM2.5-1.2B-Instruct -o ./lfm-eye
    hnx image train ./lfm-eye --data captions.jsonl --stage projector --steps 2000
    hnx image train ./lfm-eye --data chats.jsonl --stage all --steps 500 --lr 2e-5
    hnx image caption ./lfm-eye photo.jpg -p "What is in this picture?"
    hnx image split ./lfm-eye -o ./eye-for-lfm      the image model alone, to fuse again
    hnx image info ./lfm-eye

The text model can be an LFM2.5 (not -VL), a Qwen3.5, or a HyperNix
model; a folder or a Hugging Face repo id. Training ``--stage
projector`` or ``image`` leaves the text model exactly as it was.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["cli_main"]


def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _info(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") == "hnx0xv2image":
        from .hnx0xv2image import ImageConfig

        c = ImageConfig.from_dict(config)
        return {"kind": "image model", "preset": c.name, "image_size": c.image_size,
                "tokens_per_image": c.tokens_per_image, "projects_to": c.text_hidden_size or None}
    return {"kind": "fused", **config}


def cli_main(argv: list[str] | None = None) -> int:
    from . import hnx0xv2image as hi

    parser = argparse.ArgumentParser(prog="hnx image", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="an untrained image model")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--preset", default="small", choices=sorted(hi.PRESETS))
    p.add_argument("--image-size", type=int, default=None)

    p = sub.add_parser("fuse", help="put an image model on a text model")
    p.add_argument("image", help="an image model folder, or 'new' for a fresh one")
    p.add_argument("text", help="LFM2.5 (non-VL), Qwen3.5 or HyperNix: a folder or a repo id")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--preset", default="small", choices=sorted(hi.PRESETS), help="with image 'new'")
    p.add_argument("--link", action="store_true", help="refer to the text model instead of copying it")
    p.add_argument("--allow-other", action="store_true", help="any causal LM, not only the three families")

    p = sub.add_parser("train", help="train one part, the rest frozen")
    p.add_argument("model")
    p.add_argument("--data", required=True, help="JSONL: {image, text} or {image, prompt, response}")
    p.add_argument("--stage", default="projector", choices=list(hi.STAGES))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 for projector, else 2e-5")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "c4"])
    p.add_argument("-o", "--output", default=None, help="default: overwrite MODEL")

    p = sub.add_parser("caption", help="answer about an image")
    p.add_argument("model")
    p.add_argument("images", nargs="+")
    p.add_argument("-p", "--prompt", default="Describe the image.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)

    p = sub.add_parser("split", help="the image model alone, to fuse elsewhere")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True)

    p = sub.add_parser("info")
    p.add_argument("model")

    for name in ("fuse", "train", "caption"):
        sub.choices[name].add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "new":
            overrides = {"image_size": args.image_size} if args.image_size else {}
            model = hi.ImageModel(hi.preset(args.preset, **overrides))
            model.save(args.output)
            print(f"{args.output}: {model.config.name}, {model.num_params():,} parameters, "
                  f"{model.config.tokens_per_image} tokens an image")
        elif args.cmd == "fuse":
            image = hi.ImageModel(hi.preset(args.preset)) if args.image == "new" else args.image
            fused = hi.fuse(image, args.text, device=_device(args.device), allow_other=args.allow_other)
            fused.save(args.output, text=not args.link)
            print(f"{args.output}: {hi.FUSABLE.get(fused.family, fused.family)} + {hi.ARCH}; "
                  f"train it with: hnx image train {args.output} --data DATA --stage projector")
        elif args.cmd == "train":
            fused = hi.load(args.model, device=_device(args.device))
            lr = args.lr if args.lr is not None else (1e-3 if args.stage == "projector" else 2e-5)
            report = hi.train(fused, args.data, stage=args.stage, steps=args.steps, lr=lr,
                              batch_size=args.batch_size, optimizer=args.optimizer,
                              log_every=max(1, args.steps // 20))
            config = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
            linked = config.get("text") != "text"
            fused.save(args.output or args.model, text=not linked or args.stage in ("text", "all"))
            out = report.to_dict()
            print(json.dumps(out) if args.json else
                  f"{args.stage}: {report.steps} steps over {report.trainable:,} parameters, "
                  f"loss {report.first:.4f} -> {report.last:.4f}")
        elif args.cmd == "caption":
            fused = hi.load(args.model, device=_device(args.device))
            images = args.images if len(args.images) > 1 else args.images[0]
            print(fused.generate(images, args.prompt, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature))
        elif args.cmd == "split":
            config = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != hi.FUSED_ARCH:
                raise hi.ImageError(f"{args.model} is not a fused model")
            hi.load_image_model(Path(args.model) / config.get("image", "image")).save(args.output)
            print(f"{args.output}: the image model, projecting to the {config.get('family_name')} width")
        elif args.cmd == "info":
            data = _info(Path(args.model))
            print(json.dumps(data, indent=1) if args.json else
                  "\n".join(f"{k}: {v}" for k, v in data.items()))
    except (hi.ImageError, FileNotFoundError) as exc:
        print(f"hnx image: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
