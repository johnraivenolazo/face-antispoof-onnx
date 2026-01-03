"""Tests for detection module."""

from src.detection import detect


class TestDetector:
    def test_detector_loads(self, face_detector):
        """Detector loads without error."""
        assert face_detector is not None

    def test_detect_returns_list(self, face_detector, dummy_frame):
        """Detect returns a list."""
        results = detect(dummy_frame, face_detector)
        assert isinstance(results, list)

    def test_detect_none_image(self, face_detector):
        """Detect handles None image."""
        results = detect(None, face_detector)
        assert results == []

    def test_detect_none_detector(self, dummy_frame):
        """Detect handles None detector."""
        results = detect(dummy_frame, None)
        assert results == []

    def test_detection_bbox_keys(self, face_detector, dummy_frame):
        """Detection result has correct structure (if faces found)."""
        results = detect(dummy_frame, face_detector)

        if results:
            detection = results[0]
            assert "bbox" in detection
            assert "confidence" in detection
            bbox = detection["bbox"]
            assert all(k in bbox for k in ["x", "y", "width", "height"])
