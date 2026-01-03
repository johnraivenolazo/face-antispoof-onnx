"""Tests for preprocessing module."""

import numpy as np
import pytest
from src.inference.preprocess import preprocess, preprocess_batch, crop


class TestPreprocess:
    def test_output_shape(self, dummy_face_crop):
        """Output shape is (C, H, W)."""
        result = preprocess(dummy_face_crop, 128)
        assert result.shape == (3, 128, 128)

    def test_output_range(self, dummy_face_crop):
        """Output values in [0, 1]."""
        result = preprocess(dummy_face_crop, 128)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self, dummy_face_crop):
        """Output is float32."""
        result = preprocess(dummy_face_crop, 128)
        assert result.dtype == np.float32

    def test_different_input_sizes(self):
        """Handles various input sizes."""
        for size in [64, 100, 200, 256]:
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            result = preprocess(img, 128)
            assert result.shape == (3, 128, 128)


class TestPreprocessBatch:
    def test_batch_shape(self, dummy_face_crop):
        """Batch output shape is (N, C, H, W)."""
        batch = [dummy_face_crop, dummy_face_crop]
        result = preprocess_batch(batch, 128)
        assert result.shape == (2, 3, 128, 128)

    def test_empty_batch_raises(self):
        """Empty batch raises ValueError."""
        with pytest.raises(ValueError):
            preprocess_batch([], 128)


class TestCrop:
    def test_crop_square_output(self):
        """Crop returns square image."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)  # x, y, x2, y2
        result = crop(img, bbox, 1.5)

        assert result.shape[0] == result.shape[1]

    def test_crop_invalid_bbox_raises(self):
        """Invalid bbox raises ValueError."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 100, 100)  # zero width/height

        with pytest.raises(ValueError):
            crop(img, bbox, 1.5)
