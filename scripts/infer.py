"""Generate an English caption with a local or Hugging Face BLIP checkpoint."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path


def parse_args():
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    config = json.loads(config_path.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="models/best_model")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-beams", type=int, default=config["num_beams"])
    parser.add_argument("--max-new-tokens", type=int, default=config["max_new_tokens"])
    parser.add_argument("--full-precision", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.num_beams < 1 or args.max_new_tokens < 1:
        parser.error("Beam count and token limit must be positive.")
    if args.model == "models/best_model" and not Path(args.model).is_dir():
        parser.error("Download the checkpoint first; see models/README.md.")
    return args


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu.")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda" and not args.full_precision
        else nullcontext()
    )
    with torch.inference_mode(), context:
        output = model.generate(
            **inputs,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    print(processor.decode(output[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
