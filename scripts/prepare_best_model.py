import torch
from collections import OrderedDict
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.minifasv2.model import MultiFTNet
from src.minifasv2.config import get_kernel


def extract_model_weights(checkpoint_path, output_path, input_size=128, num_classes=2):
    print(f"Loading checkpoint: {checkpoint_path}")

    device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "FTGenerator" in key:
            continue

        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        new_key = new_key.replace("model.prob", "model.logits")
        new_key = new_key.replace(".prob", ".logits")
        new_key = new_key.replace("model.drop", "model.dropout")
        new_key = new_key.replace(".drop", ".dropout")

        clean_state_dict[new_key] = value

    kernel_size = get_kernel(input_size, input_size)
    model = MultiFTNet(
        num_channels=3,
        num_classes=num_classes,
        embedding_size=128,
        conv6_kernel=kernel_size,
    )

    model.load_state_dict(clean_state_dict, strict=False)

    print(f"Saving clean model to: {output_path}")
    torch.save(
        {
            "model_state_dict": clean_state_dict,
            "input_size": input_size,
            "num_classes": num_classes,
            "architecture": "MiniFASNetV2SE",
        },
        output_path,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    reduction = (1 - size_mb / original_size) * 100

    print(f"[OK] Clean model saved: {size_mb:.2f} MB")
    print(f"     Original size: {original_size:.2f} MB")
    print(f"     Size reduction: {reduction:.1f}%")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clean model weights from epoch checkpoint"
    )
    parser.add_argument(
        "epoch_checkpoint",
        type=str,
        help="Path to epoch checkpoint (e.g., epoch_31.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for best model (default: best_model.pth in models/)",
    )
    parser.add_argument(
        "--input_size", type=int, default=128, help="Input image size (default: 128)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of classes (default: 2)"
    )

    args = parser.parse_args()

    assert os.path.isfile(
        args.epoch_checkpoint
    ), f"Checkpoint not found: {args.epoch_checkpoint}"

    if args.output is None:
        os.makedirs("models", exist_ok=True)
        args.output = "models/best_model.pth"

    extract_model_weights(
        args.epoch_checkpoint, args.output, args.input_size, args.num_classes
    )
    print(f"\n[OK] Best model ready: {args.output}")
