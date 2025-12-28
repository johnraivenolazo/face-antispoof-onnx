<div align="center">

# Lightweight Face Antispoof (MiniFAS, ONNX)

[![License](https://img.shields.io/badge/License-Apache%202.0-black.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-black)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-black)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-black)](https://onnx.ai/)

![Demo](assets/demo.gif)

</div>

A lightweight face anti-spoofing model that distinguishes real faces from spoofing attempts (printed photos, screen displays, etc.). Used in [**SURI**](https://github.com/johnraivenolazo/suri), an AI attendance system.

This repo contains the training pipeline, pretrained weights, and ONNX export. Standalone repository for training and optimization.

---

## Model

The trained model is a tiny classifier that predicts two classes: **Real** or **Spoof**.

| ONNX | Quantized ONNX | PyTorch | Input | Arch |
|:---------:|:---------------:|:------------:|:-----:|:------------:|
| 1.82 MB | **600 KB** | 1.95 MB | 128×128 RGB | MiniFAS |


### Model Performance

Both models maintain high precision (>99%) at operational thresholds with minimal accuracy loss from quantization:

| Metric | Model | Quantized |
|:-------|:-----:|:---------:|
| **Model Size** | 1.82 MB | **600 KB** |
| **Overall Accuracy** | 97.80% | 97.80% |
| Real Accuracy | 98.16% | 98.14% |
| Spoof Accuracy | 97.50% | 97.52% |
| **ROC-AUC** | **0.9978** | **0.9978** |
| **Average Precision** | **0.9981** | **0.9981** |

> Metrics validated on CelebA Spoof benchmark (70k+ samples). Quantization maintains accuracy with minimal loss.

#### Model Metrics

<div align="center">
  <img src="assets/metrics/conf_matrix.png" width="49%" alt="Confusion Matrix" />
  <img src="assets/metrics/roc_curve.png" width="49%" alt="ROC Curve" />
</div>

<div align="center">
  <img src="assets/metrics/pr_curve.png" width="60%" alt="Precision-Recall Curve" />
</div>

<div align="center">
  <img src="assets/metrics/confidence_dist.png" width="60%" alt="Confidence Distribution" />
</div>

#### Quantized Metrics

<div align="center">
  <img src="assets/metrics_quant/conf_matrix.png" width="49%" alt="Confusion Matrix" />
  <img src="assets/metrics_quant/roc_curve.png" width="49%" alt="ROC Curve" />
</div>

<div align="center">
  <img src="assets/metrics_quant/pr_curve.png" width="60%" alt="Precision-Recall Curve" />
</div>

<div align="center">
  <img src="assets/metrics_quant/confidence_dist.png" width="60%" alt="Confidence Distribution" />
</div>

> The confidence distribution shows strong separation between classes, with minimal overlap in the 0.4–0.6 range. The model maintains high precision (>99%) at operational thresholds, minimizing false biometric access.

---

## Pre-trained

Pre-trained models are available in the `models/` directory:

| Model | Size | Format | Use Case |
|:------|:----:|:------:|:---------|
| `best_model.pth` | 1.95 MB | PyTorch | Training, fine-tuning, PyTorch inference |
| `best_model.onnx` | 1.82 MB | ONNX | General deployment, cross-platform inference |
| `best_model_quantized.onnx` | **600 KB** | ONNX (INT8) | **Production deployment** |

---

## Why MiniFAS?

The first version used MobileNetV4 (still in `src/mobilenetv4` for reference). It worked, but the model was larger than necessary and the training was more complex.

MiniFAS turned out to be a better fit:
- Smaller model, faster inference
- Built specifically for anti-spoofing (not a general-purpose backbone)
- Uses Fourier Transform auxiliary loss which helps the model learn texture patterns that distinguish real skin from printed/displayed images

> The MobileNetV4 code remains for future use and reference, but all active training uses the MiniFAS architecture.

---

## Quick Start

```bash
pip install -r requirements.txt
```

> **Note:** To run on GPU, install `onnxruntime-gpu` instead of `onnxruntime`.

### Run the Demo

**Webcam:**
```bash
python demo.py
```
or

```bash
python demo.py --camera [index]
```

**Single image:**
```bash
python demo.py --image path/to/face.jpg
```

Green bbox = real. 
Red bbox = spoof.

---

## Training

### 1. Prepare the Dataset

The dataset needs:
- Face images (`.jpg` or `.png`)
- Bounding box files: for `image.jpg`, a corresponding `image_BB.txt` with `x y w h`
- Label files: `metas/labels/train_label.json` and `metas/labels/test_label.json`

![Data preparation overview](assets/data_prep.png)

Run the prep script to crop faces:

```bash
python scripts/prepare_data.py \
  --orig_dir /path/to/raw/dataset \
  --crop_dir data \
  --size 128 \
  --bbox_expansion_factor 1.5 \
  --spoof_types 0 1 2 3 7 8 9
```

This reads images, crops faces using the bounding boxes (with some padding), resizes to 128×128, and organizes everything into `train/` and `test/` folders.

→ [**Why these preprocessing choices?**](docs/DATA_PREPARATION.md) (interpolation methods, padding strategy, etc.)

![Data prep result](assets/data_prep2.png)

### 2. Train

```bash
python scripts/train.py \
  --crop_dir data \
  --batch_size 256 \
  --output_dir ./output
```

Checkpoints and TensorBoard logs go to `./output/MINIFAS/`.

**Resume training:**
```bash
python scripts/train.py \
  --crop_dir data \
  --resume ./output/MINIFAS/checkpoint_latest.pth
```

### 3. Prepare Model

Extract clean model weights from checkpoint (removes optimizer state, FTGenerator, DataParallel prefixes):

```bash
python scripts/prepare_best_model.py \
  models/checkpoints/minifasv2\ \(128x128\)/epoch_31.pth \
  --output models/best_model.pth
```

This creates a clean, inference-ready PyTorch model.

### 4. Export to ONNX

**Regular ONNX export:**
```bash
python scripts/export_onnx.py models/best_model.pth --input_size 128
```

**Quantized ONNX:**
```bash
python scripts/quantize_onnx.py models/best_model.pth --input_size 128
```

---

## Repo Structure

The codebase is organized into modular components for clarity and maintainability:

```
├── src/
│   ├── detection/          # Face detection module
│   │   └── face.py         # Face detector (load_detector, detect)
│   │
│   ├── inference/          # Model inference module
│   │   ├── loader.py       # Model loading (load_model)
│   │   ├── inference.py    # Inference functions (infer, process_with_logits)
│   │   ├── preprocess.py   # Image preprocessing (crop, preprocess, preprocess_batch)
│   │   └── system.py       # System information (CPU/GPU info)
│   │
│   ├── minifasv2/          # MiniFAS training code
│   │   ├── config.py       # Training config
│   │   ├── data.py         # Dataset loading
│   │   ├── main.py         # Trainer class
│   │   └── model.py        # MiniFAS architecture
│   │
│   └── mobilenetv4/         # Legacy (kept for reference)
│
├── scripts/
│   ├── prepare_data.py        # Dataset preparation
│   ├── train.py               # Training entrypoint
│   ├── prepare_best_model.py  # Extract clean model weights
│   ├── export_onnx.py         # Regular ONNX export
│   └── quantize_onnx.py       # Quantized ONNX export
│
├── docs/                   # Technical documentation
├── models/                 # Pre-trained models
│   ├── best_model.pth              # Clean PyTorch model
│   ├── best_model.onnx             # Regular ONNX
│   └── best_model_quantized.onnx  # Quantized ONNX
└── demo.py                 # Inference demo
```

---

## Limitations

The model performs best with well-lit, frontal faces and proper preprocessing. For detailed information on limitations, edge cases, and best practices, see [**Limitations & Notes**](docs/LIMITATIONS.md).

---

## Acknowledgment

This project is based on the MiniFAS architecture from the
Silent Face Anti-Spoofing project by Minivision AI, licensed under Apache-2.0.

> This repository provides an independent training pipeline, ONNX export,
quantization, and deployment tooling.

---

## License

Apache-2.0. See `LICENSE`.
