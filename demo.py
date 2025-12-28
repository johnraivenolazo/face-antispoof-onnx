import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
import platform
import sys

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
try:
    import cpuinfo

    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False
try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_MODEL = MODELS_DIR / "detector.onnx"
LIVENESS_MODEL = MODELS_DIR / "best_model_quantized.onnx"


def preprocess(img: np.ndarray, model_img_size: int) -> np.ndarray:
    new_size = model_img_size
    old_size = img.shape[:2]

    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])

    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(
        img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation
    )

    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    return img


def preprocess_batch(face_crops: List[np.ndarray], model_img_size: int) -> np.ndarray:
    if not face_crops:
        raise ValueError("face_crops list cannot be empty")

    batch = np.zeros(
        (len(face_crops), 3, model_img_size, model_img_size), dtype=np.float32
    )
    for i, face_crop in enumerate(face_crops):
        batch[i] = preprocess(face_crop, model_img_size)

    return batch


def crop(img: np.ndarray, bbox: tuple, bbox_expansion_factor: float) -> np.ndarray:
    original_height, original_width = img.shape[:2]
    x, y, w, h = bbox

    w = w - x
    h = h - y

    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox dimensions")

    max_dim = max(w, h)
    center_x = x + w / 2
    center_y = y + h / 2

    x = int(center_x - max_dim * bbox_expansion_factor / 2)
    y = int(center_y - max_dim * bbox_expansion_factor / 2)
    crop_size = int(max_dim * bbox_expansion_factor)

    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(original_width, x + crop_size)
    crop_y2 = min(original_height, y + crop_size)

    top_pad = int(max(0, -y))
    left_pad = int(max(0, -x))
    bottom_pad = int(max(0, (y + crop_size) - original_height))
    right_pad = int(max(0, (x + crop_size) - original_width))

    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        img = np.zeros((0, 0, 3), dtype=img.dtype)

    result = cv2.copyMakeBorder(
        img,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_REFLECT_101,
    )

    if result.shape[0] != crop_size or result.shape[1] != crop_size:
        raise ValueError(
            f"Crop size mismatch: expected {crop_size}x{crop_size}, got {result.shape[0]}x{result.shape[1]}"
        )

    return result


def process_with_logits(raw_logits: np.ndarray, threshold: float) -> Dict:
    real_logit = float(raw_logits[0])
    spoof_logit = float(raw_logits[1])
    logit_diff = real_logit - spoof_logit
    is_real = logit_diff >= threshold
    confidence = abs(logit_diff)

    return {
        "is_real": bool(is_real),
        "status": "real" if is_real else "spoof",
        "logit_diff": float(logit_diff),
        "real_logit": float(real_logit),
        "spoof_logit": float(spoof_logit),
        "confidence": float(confidence),
    }


def infer(
    face_crops: List[np.ndarray],
    ort_session: ort.InferenceSession,
    input_name: str,
    model_img_size: int,
) -> List[np.ndarray]:
    if not face_crops or ort_session is None:
        return []

    try:
        batch_input = preprocess_batch(face_crops, model_img_size)
        logits = ort_session.run([], {input_name: batch_input})[0]

        if logits.shape != (len(face_crops), 2):
            raise ValueError("Model output shape mismatch")

        return [logits[i] for i in range(len(face_crops))]
    except Exception as e:
        print(f"Inference error: {e}", file=sys.stderr)
        return []


def get_cpu_info() -> str:
    cpu_name = None
    cpu_freq_mhz = None
    cpu_cores = None
    cpu_threads = None

    if HAS_CPUINFO:
        try:
            info = cpuinfo.get_cpu_info()
            cpu_name = (
                info.get("brand_raw") or info.get("brand") or info.get("model name")
            )
            if cpu_name:
                cpu_name = cpu_name.strip()
        except Exception:
            pass

    if HAS_PSUTIL:
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            if cpu_freq and cpu_freq.current:
                cpu_freq_mhz = cpu_freq.current
        except Exception:
            pass

    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"

    parts = []
    if cpu_name:
        parts.append(cpu_name)

    if cpu_cores and cpu_threads:
        if cpu_cores == cpu_threads:
            parts.append(f"{cpu_cores} cores")
        else:
            parts.append(f"{cpu_cores}C/{cpu_threads}T")

    if cpu_freq_mhz:
        if cpu_freq_mhz >= 1000:
            parts.append(f"{cpu_freq_mhz/1000:.2f} GHz")
        else:
            parts.append(f"{cpu_freq_mhz:.0f} MHz")

    return " | ".join(parts) if parts else "Unknown CPU"


def get_gpu_info() -> Optional[str]:
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > 0:
                gpu = gpus[0]
                return gpu.name.strip()
        except Exception:
            pass

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return None


def get_execution_provider_name(session: ort.InferenceSession) -> str:
    try:
        providers = session.get_providers()
        if "CUDAExecutionProvider" in providers:
            return "CUDA"
        elif "CPUExecutionProvider" in providers:
            return "CPU"
        elif providers:
            return providers[0].replace("ExecutionProvider", "")
        return "Unknown"
    except Exception:
        return "Unknown"


def load_model(model_path: str) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
    if not Path(model_path).exists():
        return None, None

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available_providers]

        if not providers:
            providers = available_providers

        ort_session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name
    except Exception:
        return None, None


def load_detector(
    model_path: str,
    input_size: tuple,
    confidence_threshold: float = 0.8,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
):
    if not Path(model_path).exists():
        return None

    try:
        return cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            input_size,
            confidence_threshold,
            nms_threshold,
            top_k,
        )
    except Exception:
        return None


def detect(
    image: np.ndarray, detector, min_face_size: int = 60, margin: int = 5
) -> List[Dict]:
    if detector is None or image is None:
        return []

    img_h, img_w = image.shape[:2]
    detector.setInputSize((img_w, img_h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return []

    detections = []
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        conf = float(face[14])

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            continue

        dist_left = x
        dist_right = img_w - (x + w)
        dist_top = y
        dist_bottom = img_h - (y + h)
        if min(dist_left, dist_right, dist_top, dist_bottom) < margin:
            continue

        if w >= min_face_size and h >= min_face_size:
            detections.append(
                {
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                    },
                    "confidence": conf,
                }
            )

    return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (if not provided, uses camera)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index to use (default: 0)"
    )
    parser.add_argument("--model_img_size", type=int, default=128)
    parser.add_argument("--bbox_expansion_factor", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=int, default=5)
    parser.add_argument("--detector_model", type=str, default=str(DETECTOR_MODEL))
    parser.add_argument("--liveness_model", type=str, default=str(LIVENESS_MODEL))
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose error logging"
    )

    args = parser.parse_args()

    p = max(1e-6, min(1 - 1e-6, args.threshold))
    logit_threshold = np.log(p / (1 - p))

    face_detector = load_detector(args.detector_model, (320, 320))
    liveness_session, input_name = load_model(args.liveness_model)

    if liveness_session is None or face_detector is None:
        exit(1)

    if args.image is None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            print("Available cameras:")
            for i in range(10):
                test_cap = cv2.VideoCapture(i)
                if test_cap.isOpened():
                    print(f"  Camera {i}: Available")
                    test_cap.release()
            exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        window_name = "Liveness Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        show_info = True
        fps_history = []

        cpu_info = get_cpu_info()
        gpu_info = get_gpu_info()
        provider_name = get_execution_provider_name(liveness_session)

        print("Controls:")
        print("  'q' - Quit")
        print("  'i' - Toggle info display")

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect(frame_rgb, face_detector, margin=args.margin)

            if faces:
                face_crops = []
                valid_faces = []
                for face in faces:
                    bbox = face["bbox"]
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    try:
                        face_crop = crop(
                            frame_rgb, (x, y, x + w, y + h), args.bbox_expansion_factor
                        )
                        face_crops.append(face_crop)
                        valid_faces.append((face, (int(x), int(y), int(w), int(h))))
                    except Exception as e:
                        if args.verbose:
                            print(
                                f"Warning: Failed to crop face at ({x},{y},{w},{h}): {e}",
                                file=sys.stderr,
                            )
                        continue

                if face_crops:
                    predictions = infer(
                        face_crops, liveness_session, input_name, args.model_img_size
                    )

                    for (face, (x, y, w, h)), pred in zip(valid_faces, predictions):
                        try:
                            result = process_with_logits(pred, logit_threshold)
                        except Exception:
                            continue

                        color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        label = (
                            f"{result['status'].upper()}: {result['logit_diff']:.2f}"
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            frame_time = time.time() - frame_start
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(current_fps)

            if len(fps_history) > 30:
                fps_history.pop(0)

            display_width = 640
            display_height = 480
            display_frame = cv2.resize(
                frame, (display_width, display_height), interpolation=cv2.INTER_AREA
            )

            if show_info:
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

                info_y = 25
                line_height = 20
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                color_white = (255, 255, 255)
                color_cyan = (255, 255, 0)

                cv2.putText(
                    display_frame,
                    f"FPS: {avg_fps:.1f}",
                    (5, info_y),
                    font,
                    font_scale,
                    color_cyan,
                    thickness,
                )
                info_y += line_height

                cpu_lines = []
                max_chars_per_line = 55
                words = cpu_info.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= max_chars_per_line:
                        current_line += " " + word if current_line else word
                    else:
                        if current_line:
                            cpu_lines.append(current_line)
                        current_line = word
                if current_line:
                    cpu_lines.append(current_line)

                for i, cpu_line in enumerate(cpu_lines[:2]):
                    cv2.putText(
                        display_frame,
                        f"CPU: {cpu_line}" if i == 0 else cpu_line,
                        (5, info_y),
                        font,
                        font_scale,
                        color_white,
                        thickness,
                    )
                    info_y += line_height

                if gpu_info:
                    gpu_lines = []
                    words = gpu_info.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= max_chars_per_line:
                            current_line += " " + word if current_line else word
                        else:
                            if current_line:
                                gpu_lines.append(current_line)
                            current_line = word
                    if current_line:
                        gpu_lines.append(current_line)

                    for i, gpu_line in enumerate(gpu_lines[:2]):
                        cv2.putText(
                            display_frame,
                            f"GPU: {gpu_line}" if i == 0 else gpu_line,
                            (5, info_y),
                            font,
                            font_scale,
                            color_white,
                            thickness,
                        )
                        info_y += line_height
                else:
                    cv2.putText(
                        display_frame,
                        "GPU: No GPU detected",
                        (5, info_y),
                        font,
                        font_scale,
                        color_white,
                        thickness,
                    )
                    info_y += line_height

                cv2.putText(
                    display_frame,
                    f"Provider: {provider_name}",
                    (5, info_y),
                    font,
                    font_scale,
                    color_white,
                    thickness,
                )
                info_y += line_height
                cv2.putText(
                    display_frame,
                    "Press 'i' to toggle",
                    (5, info_y),
                    font,
                    0.4,
                    (200, 200, 200),
                    1,
                )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                show_info = not show_info

        cap.release()
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image from '{args.image}'", file=sys.stderr)
            print(
                "Please check that the file exists and is a valid image format.",
                file=sys.stderr,
            )
            exit(1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detect(image_rgb, face_detector, margin=args.margin)

        if not faces:
            exit(0)

        face_crops = []
        valid_faces = []
        for face in faces:
            bbox = face["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            try:
                face_crop = crop(
                    image_rgb, (x, y, x + w, y + h), args.bbox_expansion_factor
                )
                face_crops.append(face_crop)
                valid_faces.append((int(x), int(y), int(w), int(h)))
            except Exception as e:
                if args.verbose:
                    print(
                        f"Warning: Failed to crop face at ({x},{y},{w},{h}): {e}",
                        file=sys.stderr,
                    )
                continue

        if not face_crops:
            exit(0)

        predictions = infer(
            face_crops, liveness_session, input_name, args.model_img_size
        )

        for (x, y, w, h), pred in zip(valid_faces, predictions):
            try:
                result = process_with_logits(pred, logit_threshold)
            except Exception:
                continue

            color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
            cv2.putText(
                image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
