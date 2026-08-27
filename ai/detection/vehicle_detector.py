"""
vehicle_detector.py

Purpose
-------
Wraps an Ultralytics YOLO model and exposes a single method, detect(), that
takes one video frame and returns structured vehicle detections. Everything
downstream (tracker, plate detector, event builder) depends only on
VehicleDetection — not on Ultralytics' own result objects — so we can swap
the underlying model (e.g. YOLOX) later without touching other files.

Input:
    One BGR video frame (np.ndarray[H, W, 3], as produced by CameraSource).

Output:
    List[VehicleDetection] — zero or more vehicles found in that frame.

Model license note (see docs/MODELS_AND_LICENSES.md):
    Ultralytics YOLO ships under AGPL-3.0 for the open-source tier. That's
    fine for this project because the whole repo is public for SIH. If this
    code is ever reused in a closed-source product, either buy an Ultralytics
    Enterprise license or swap in YOLOX (Apache-2.0) — same detect()
    interface, different model backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO class ids that correspond to road vehicles. YOLO's COCO-pretrained
# weights already know these classes, so no fine-tuning is needed for this
# baseline. Known limitation (see docs/LIMITATIONS.md): COCO has no
# auto-rickshaw / tempo class, which matters for Indian traffic — flagged
# for future fine-tuning, not silently ignored.
VEHICLE_CLASS_MAP = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class VehicleDetection:
    """One detected vehicle in one frame."""
    class_id: int
    vehicle_type: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords


class VehicleDetector:
    """
    Thin, swappable wrapper around an Ultralytics YOLO model.

    Raises:
        ImportError: if ultralytics isn't installed, with the exact install
            command, instead of a bare stack trace.
        FileNotFoundError / ultralytics' own errors: if model_path is a local
            path that doesn't exist. (Using a bare model name like
            "yolo11n.pt" triggers an automatic download on first run — this
            requires internet access once; the weights are then cached
            locally, so run this once *before* the demo/judging.)
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.35,
        classes: List[int] | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed in this environment. Run:\n"
                "    pip install ultralytics --break-system-packages\n"
                "(or without --break-system-packages if you're in a venv, "
                "which you should be)."
            ) from exc

        self.device = device
        self.conf_threshold = conf_threshold
        self.classes = classes if classes is not None else list(VEHICLE_CLASS_MAP.keys())

        logger.info(
            "event=loading_model model_path=%s device=%s conf_threshold=%.2f",
            model_path, device, conf_threshold,
        )
        self._model = YOLO(model_path)
        logger.info("event=model_loaded model_path=%s", model_path)

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Run detection on a single frame. Returns [] on any per-frame
        failure rather than raising, so one bad frame can't crash a whole
        video's processing run (see project error-handling rule)."""
        if frame is None or frame.size == 0:
            logger.warning("event=detect_skipped reason=empty_frame")
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.conf_threshold,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )
        except Exception:
            logger.exception("event=detect_failed")
            return []

        detections: List[VehicleDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    VehicleDetection(
                        class_id=class_id,
                        vehicle_type=VEHICLE_CLASS_MAP.get(class_id, "unknown"),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections