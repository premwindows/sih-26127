"""
camera_source.py

Purpose
-------
Defines a single abstraction, CameraSource, that every downstream component
(detector, tracker, event writer) talks to. Today the only implementation is
VideoFileSource (reads camera_01.mp4 etc.). Later, RTSPSource will read a
live IP-camera stream. Because both implement the same interface, nothing
above this layer needs to change when we move from demo videos to real
cameras — this is the "video file -> RTSP" migration path required by the
project brief.

Input:
    A video file path (VideoFileSource) or an RTSP URL (RTSPSource, stub).

Output:
    A stream of (frame_index: int, frame: np.ndarray[H, W, 3] BGR) tuples.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraSource(ABC):
    """Abstract base class for any frame-producing camera feed."""

    camera_id: str

    @abstractmethod
    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, frame) tuples in order, starting at 0."""
        raise NotImplementedError

    @abstractmethod
    def get_fps(self) -> float:
        """Return the source's frame rate (frames per second)."""
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release any underlying OS resources (file handles, sockets)."""
        raise NotImplementedError

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class VideoFileSource(CameraSource):
    """
    Reads frames from a local video file (our simulated CCTV feed).

    Raises:
        FileNotFoundError: if OpenCV cannot open the given path. This is
            deliberate — a silently-empty frame stream would look like a
            camera producing zero vehicles, which is misleading. We fail
            loudly instead, per the project's error-handling rule: don't
            crash on *expected* failure modes (bad OCR, missing plate), but
            DO fail loudly on *configuration* mistakes (wrong path).
    """

    def __init__(self, video_path: str, camera_id: str):
        self.video_path = video_path
        self.camera_id = camera_id
        self._cap = cv2.VideoCapture(video_path)

        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"VideoFileSource: could not open '{video_path}' for camera "
                f"'{camera_id}'. Check that the file exists and is a video "
                f"OpenCV can decode (mp4/avi with a standard codec)."
            )

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 25.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "camera=%s event=source_opened path=%s fps=%.2f frame_count=%d",
            self.camera_id, self.video_path, self._fps, self._frame_count,
        )

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                # End of file, or an unreadable frame. Either way, stop
                # cleanly rather than yielding garbage.
                break
            if frame is None or frame.size == 0:
                logger.warning(
                    "camera=%s event=empty_frame_skipped index=%d",
                    self.camera_id, idx,
                )
                idx += 1
                continue
            yield idx, frame
            idx += 1

        logger.info(
            "camera=%s event=source_exhausted frames_yielded=%d",
            self.camera_id, idx,
        )

    def get_fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def release(self) -> None:
        self._cap.release()
        logger.info("camera=%s event=source_released", self.camera_id)


class RTSPSource(CameraSource):
    """
    STUB — intentionally not implemented yet.

    Why not implemented now:
        We have no real IP camera / RTSP stream to test reconnect, timeout,
        or frame-drop behavior against. Writing that logic without something
        real to verify it against means shipping unverified code, which
        contradicts this project's own "don't fake it" rule. Better to ship
        a clean, documented stub than a plausible-looking implementation
        nobody has tested.

    When we do implement it (Phase 7+, or if a real/virtual RTSP source
    becomes available — e.g. re-streaming a file with
    `ffmpeg -re -i camera_01.mp4 -f rtsp rtsp://localhost:8554/cam01`):
        - open with cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        - add a read timeout + reconnect-with-backoff loop
        - add frame-skip/backpressure handling if the consumer is slower
          than the stream
    The public interface (frames/get_fps/release) is already fixed by
    CameraSource, so nothing downstream changes when this lands.
    """

    def __init__(self, rtsp_url: str, camera_id: str):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        raise NotImplementedError(
            "RTSPSource is a documented stub (see class docstring). "
            "Use VideoFileSource for the current prototype."
        )

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        raise NotImplementedError

    def get_fps(self) -> float:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError