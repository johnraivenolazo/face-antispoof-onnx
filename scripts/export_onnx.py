import torch
import onnx
import onnxsim
from collections import OrderedDict
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.minifasv2.model import MultiFTNet
from src.minifasv2.config import get_kernel


def load_model_from_checkpoint(checkpoint_path, device, input_size=128, num_classes=2):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    kernel_size = get_kernel(input_size, input_size)
    model = MultiFTNet(
        num_channels=3,
        num_classes=num_classes,
        embedding_size=128,
        conv6_kernel=kernel_size,
    ).to(device)

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        new_key = new_key.replace("model.prob", "model.logits")
        new_key = new_key.replace(".prob", ".logits")
        new_key = new_key.replace("model.drop", "model.dropout")
        new_key = new_key.replace(".drop", ".dropout")
        new_key = new_key.replace("FTGenerator.ft.", "FTGenerator.fourier_transform.")
        new_key = new_key.replace("FTGenerator.ft", "FTGenerator.fourier_transform")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    return model


def export_to_onnx(model, output_path, input_size=128):
    print("Exporting model to ONNX...")
    print(f"Output path: {output_path}")

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    onnx_model = onnx.load(output_path)
    print("Simplifying ONNX model...")
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, output_path)

    print("[OK] ONNX model exported")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format (regular, non-quantized)"
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument(
        "--input_size", type=int, default=128, help="Input image size (default: 128)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of classes (default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save .onnx (default: replaces .pth with .onnx)",
    )

    args = parser.parse_args()

    assert os.path.isfile(
        args.checkpoint_path
    ), f"Checkpoint not found: {args.checkpoint_path}"

    device = "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading model from {args.checkpoint_path}...")
    model = load_model_from_checkpoint(
        args.checkpoint_path, device, args.input_size, args.num_classes
    )
    print("[OK] Model loaded")

    if args.output is None:
        args.output = args.checkpoint_path.replace(".pth", ".onnx")

    export_to_onnx(model, args.output, args.input_size)

    onnx_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nONNX model size: {onnx_size:.2f} MB")
    print(f"[OK] Done! ONNX model saved: {args.output}")
    print(
        "\nNote: For quantized ONNX, use: python scripts/quantize_onnx.py <checkpoint>"
    )
