"""
process_video.py — Phase 1 entry point.

Pipeline for this phase only:
    CameraSource (VideoFileSource) -> VehicleDetector -> annotated video + JSON

Usage:
    python scripts/process_video.py --camera_id CAM01 --video data/raw/camera_01/camera_01.mp4

Outputs (written to data/processed/detections/):
    <camera_id>_annotated.mp4   -- input video with bounding boxes drawn
    <camera_id>_detections.json -- per-frame detection list

This script deliberately does NOT do tracking, plates, or OCR yet — those
are Phases 2-4. Keeping this narrow makes it possible to actually verify
Phase 1 in isolation before building on top of it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.capture.camera_source import VideoFileSource
from ai.detection.vehicle_detector import VehicleDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("process_video")

BOX_COLOR = (60, 200, 60)
TEXT_COLOR = (255, 255, 255)


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.vehicle_type} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Phase 1: video -> vehicle detection")
    parser.add_argument("--camera_id", required=True, help="e.g. CAM01")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", default="data/processed/detections")
    parser.add_argument("--model_path", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max_frames", type=int, default=None, help="Cap frames for a quick smoke test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        detector = VehicleDetector(model_path=args.model_path, device=args.device, conf_threshold=args.conf)
    except ImportError as exc:
        logger.error(str(exc))
        sys.exit(1)

    all_frame_detections = []
    writer = None
    start_time = time.time()
    total_detections = 0

    try:
        with VideoFileSource(args.video, args.camera_id) as source:
            annotated_path = output_dir / f"{args.camera_id}_annotated.mp4"

            for frame_idx, frame in source.frames():
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break

                detections = detector.detect(frame)
                total_detections += len(detections)

                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(annotated_path), fourcc, source.get_fps(), (w, h))

                annotated = draw_detections(frame.copy(), detections)
                writer.write(annotated)

                all_frame_detections.append({
                    "frame_index": frame_idx,
                    "camera_id": args.camera_id,
                    "detections": [
                        {
                            "vehicle_type": d.vehicle_type,
                            "confidence": round(d.confidence, 4),
                            "bbox": [round(v, 1) for v in d.bbox],
                        }
                        for d in detections
                    ],
                })

                if frame_idx % 50 == 0:
                    logger.info(
                        "camera=%s event=progress frame=%d vehicles_this_frame=%d",
                        args.camera_id, frame_idx, len(detections),
                    )
    finally:
        if writer is not None:
            writer.release()

    detections_path = output_dir / f"{args.camera_id}_detections.json"
    with open(detections_path, "w") as f:
        json.dump(all_frame_detections, f, indent=2)

    elapsed = time.time() - start_time
    frames_processed = len(all_frame_detections)
    fps_achieved = frames_processed / elapsed if elapsed > 0 else 0.0

    logger.info("event=run_complete frames=%d total_detections=%d elapsed_s=%.2f fps=%.2f",
                frames_processed, total_detections, elapsed, fps_achieved)
    print(f"\nDone. {frames_processed} frames, {total_detections} vehicle detections, "
          f"{fps_achieved:.1f} FPS (device={args.device}).")
    print(f"Annotated video: {annotated_path}")
    print(f"Detections JSON: {detections_path}")


if __name__ == "__main__":
    main()