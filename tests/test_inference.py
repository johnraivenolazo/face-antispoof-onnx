"""Tests for inference module."""

import numpy as np
from src.inference import infer, process_with_logits


class TestInference:
    def test_model_loads(self, liveness_session):
        """Model loads without error."""
        session, input_name = liveness_session
        assert session is not None
        assert input_name is not None

    def test_inference_output_shape(self, liveness_session, dummy_face_crop):
        """Output shape is (batch, 2)."""
        session, input_name = liveness_session
        results = infer([dummy_face_crop], session, input_name, 128)

        assert len(results) == 1
        assert results[0].shape == (2,)

    def test_inference_batch(self, liveness_session, dummy_face_crop):
        """Batch inference works."""
        session, input_name = liveness_session
        batch = [dummy_face_crop, dummy_face_crop, dummy_face_crop]
        results = infer(batch, session, input_name, 128)

        assert len(results) == 3

    def test_inference_empty_input(self, liveness_session):
        """Empty input returns empty list."""
        session, input_name = liveness_session
        results = infer([], session, input_name, 128)

        assert results == []


class TestProcessLogits:
    def test_real_classification(self):
        """High real logit → is_real=True."""
        logits = np.array([5.0, -2.0])
        result = process_with_logits(logits, threshold=0.0)

        assert result["is_real"] is True
        assert result["status"] == "real"

    def test_spoof_classification(self):
        """High spoof logit → is_real=False."""
        logits = np.array([-2.0, 5.0])
        result = process_with_logits(logits, threshold=0.0)

        assert result["is_real"] is False
        assert result["status"] == "spoof"

    def test_result_keys(self):
        """Result contains expected keys."""
        logits = np.array([1.0, 0.0])
        result = process_with_logits(logits, threshold=0.0)

        expected_keys = {
            "is_real",
            "status",
            "logit_diff",
            "real_logit",
            "spoof_logit",
            "confidence",
        }
        assert set(result.keys()) == expected_keys
