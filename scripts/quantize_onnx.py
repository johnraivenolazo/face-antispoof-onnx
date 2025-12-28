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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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


def quantize_onnx_with_ort(onnx_path, output_path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print("\nQuantizing ONNX model with ONNX Runtime...")
        print(f"Input: {onnx_path}")
        print(f"Output: {output_path}")

        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
        )

        print("[OK] Quantized ONNX model created")
        return output_path
    except ImportError:
        print(
            "[ERROR] onnxruntime not installed. Install with: pip install onnxruntime"
        )
        return None
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export model to ONNX and quantize it using ONNX Runtime"
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument(
        "--input_size", type=int, default=128, help="Input image size (default: 128)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of classes (default: 2)"
    )
    parser.add_argument(
        "--output_onnx",
        type=str,
        default=None,
        help="Path to save regular .onnx (default: replaces .pth with .onnx)",
    )
    parser.add_argument(
        "--output_quantized",
        type=str,
        default=None,
        help="Path to save quantized .onnx (default: adds _quantized suffix)",
    )
    parser.add_argument(
        "--skip_regular",
        action="store_true",
        help="Skip exporting regular ONNX if it already exists",
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

    if args.output_onnx is None:
        args.output_onnx = args.checkpoint_path.replace(".pth", ".onnx")

    if not args.skip_regular or not os.path.exists(args.output_onnx):
        export_to_onnx(model, args.output_onnx, args.input_size)
        onnx_size = os.path.getsize(args.output_onnx) / (1024 * 1024)
        print(f"Regular ONNX size: {onnx_size:.2f} MB")
    else:
        print(f"Using existing ONNX: {args.output_onnx}")

    if args.output_quantized is None:
        args.output_quantized = args.checkpoint_path.replace(".pth", "_quantized.onnx")

    result = quantize_onnx_with_ort(args.output_onnx, args.output_quantized)

    if result:
        quantized_size = os.path.getsize(args.output_quantized) / (1024 * 1024)
        onnx_size = os.path.getsize(args.output_onnx) / (1024 * 1024)
        print(f"\nQuantized ONNX size: {quantized_size:.2f} MB")
        print(f"Size reduction: {quantized_size/onnx_size*100:.1f}% of original")
        print(f"\n[OK] Done! Quantized ONNX saved: {args.output_quantized}")
    else:
        print(
            "\n[WARNING] Quantization failed. Regular ONNX is available at:",
            args.output_onnx,
        )
        print(
            "For regular ONNX export only, use: python scripts/export_onnx.py <checkpoint>"
        )
