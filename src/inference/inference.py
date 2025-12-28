import numpy as np
import onnxruntime as ort
import sys
from typing import List, Dict
from src.inference.preprocess import preprocess_batch


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
