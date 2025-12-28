import cv2
import numpy as np
import sys
import time
import argparse
from pathlib import Path

from src.inference import (
    load_model,
    infer,
    process_with_logits,
    crop,
    get_cpu_info,
    get_gpu_info,
    get_execution_provider_name,
)
from src.detection import load_detector, detect

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_MODEL = MODELS_DIR / "detector.onnx"
LIVENESS_MODEL = MODELS_DIR / "best_model_quantized.onnx"


def draw_info_overlay(display_frame, fps_history, cpu_info, gpu_info, provider_name):
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


def process_camera(args, face_detector, liveness_session, input_name, logit_threshold):
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

                    label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
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
            draw_info_overlay(
                display_frame, fps_history, cpu_info, gpu_info, provider_name
            )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("i"):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()


def process_image(args, face_detector, liveness_session, input_name, logit_threshold):
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

    predictions = infer(face_crops, liveness_session, input_name, args.model_img_size)

    for (x, y, w, h), pred in zip(valid_faces, predictions):
        try:
            result = process_with_logits(pred, logit_threshold)
        except Exception:
            continue

        color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
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
        process_camera(
            args, face_detector, liveness_session, input_name, logit_threshold
        )
    else:
        process_image(
            args, face_detector, liveness_session, input_name, logit_threshold
        )
