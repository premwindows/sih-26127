"""
tests/test_detection.py

CameraSource tests require only OpenCV and run in any environment.
VehicleDetector tests require ultralytics+torch; they SKIP (not fail, not
fake-pass) when those aren't installed, so `pytest` gives an honest signal
about what was actually verified on a given machine.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.capture.camera_source import VideoFileSource


def _make_synthetic_video(path: Path, num_frames: int = 20, size=(320, 240), fps: float = 10.0) -> None:
    """Writes a tiny synthetic video (moving rectangle) purely to test video
    I/O plumbing. This is NOT a substitute for testing on real traffic
    footage — it validates that frames are read correctly, nothing about
    detection accuracy."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    w, h = size
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int((i / num_frames) * (w - 40))
        cv2.rectangle(frame, (x, h // 2 - 20), (x + 40, h // 2 + 20), (0, 200, 0), -1)
        writer.write(frame)
    writer.release()


class TestVideoFileSource:
    def test_reads_expected_frame_count(self, tmp_path):
        video_path = tmp_path / "synthetic.mp4"
        _make_synthetic_video(video_path, num_frames=20)

        source = VideoFileSource(str(video_path), camera_id="CAM_TEST")
        frames_read = list(source.frames())
        source.release()

        assert len(frames_read) == 20
        indices = [idx for idx, _ in frames_read]
        assert indices == list(range(20))

    def test_frame_shape_matches_video(self, tmp_path):
        video_path = tmp_path / "synthetic.mp4"
        _make_synthetic_video(video_path, num_frames=5, size=(320, 240))

        source = VideoFileSource(str(video_path), camera_id="CAM_TEST")
        _, frame = next(source.frames())
        source.release()

        assert frame.shape == (240, 320, 3)

    def test_missing_file_raises_filenotfound(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError):
            VideoFileSource(str(missing_path), camera_id="CAM_TEST")

    def test_context_manager_releases(self, tmp_path):
        video_path = tmp_path / "synthetic.mp4"
        _make_synthetic_video(video_path, num_frames=5)

        with VideoFileSource(str(video_path), camera_id="CAM_TEST") as source:
            frames = list(source.frames())
        assert len(frames) == 5

    def test_fps_reported(self, tmp_path):
        video_path = tmp_path / "synthetic.mp4"
        _make_synthetic_video(video_path, num_frames=5, fps=10.0)

        source = VideoFileSource(str(video_path), camera_id="CAM_TEST")
        fps = source.get_fps()
        source.release()
        assert fps > 0


class TestVehicleDetector:
    """These tests require ultralytics+torch. They SKIP cleanly if those
    aren't installed in this environment (e.g. this sandbox), and RUN for
    real on a machine that followed the Phase 0 setup commands."""

    @pytest.fixture(autouse=True)
    def _require_ultralytics(self):
        pytest.importorskip("ultralytics", reason="ultralytics/torch not installed in this environment")

    def test_import_guard_message(self):
        # Re-verify the detector module imports cleanly when ultralytics IS present.
        from ai.detection.vehicle_detector import VehicleDetector, VehicleDetection  # noqa: F401

    def test_detect_on_synthetic_frame_returns_list(self, tmp_path):
        from ai.detection.vehicle_detector import VehicleDetector

        detector = VehicleDetector(model_path="yolo11n.pt", device="cpu")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert isinstance(detections, list)  # a blank frame should yield zero or more, never crash

    def test_detect_on_empty_frame_returns_empty_list(self):
        from ai.detection.vehicle_detector import VehicleDetector

        detector = VehicleDetector(model_path="yolo11n.pt", device="cpu")
        assert detector.detect(np.array([])) == []


def test_vehicle_detection_import_error_without_ultralytics(monkeypatch):
    """Verify the ImportError path itself is real and has a helpful message
    -- this test runs regardless of whether ultralytics is installed, by
    forcing the import to fail."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Re-import fresh so the patched __import__ takes effect inside __init__
    sys.modules.pop("ai.detection.vehicle_detector", None)
    from ai.detection.vehicle_detector import VehicleDetector

    with pytest.raises(ImportError, match="pip install ultralytics"):
        VehicleDetector()