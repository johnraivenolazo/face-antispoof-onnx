"""Pytest fixtures for face-antispoof tests."""

import pytest
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
DETECTOR_MODEL = MODELS_DIR / "detector_quantized.onnx"
LIVENESS_MODEL = MODELS_DIR / "best_model_quantized.onnx"


@pytest.fixture(scope="session")
def liveness_session():
    """Load liveness model once for all tests."""
    from src.inference import load_model

    session, input_name = load_model(str(LIVENESS_MODEL))
    if session is None:
        pytest.skip("Liveness model not found")
    return session, input_name


@pytest.fixture(scope="session")
def face_detector():
    """Load face detector once for all tests."""
    from src.detection import load_detector

    detector = load_detector(str(DETECTOR_MODEL), (640, 480))
    if detector is None:
        pytest.skip("Detector model not found")
    return detector


@pytest.fixture
def dummy_face_crop():
    """Create a dummy 128x128 RGB image."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame():
    """Create a dummy 480x640 RGB frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
