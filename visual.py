import cv2
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).parent
_FACE_MODEL_DIR = _ROOT / "models" / "face_detector"
_PROTO = str(_FACE_MODEL_DIR / "deploy.prototxt")
_MODEL = str(_FACE_MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel")
_CONFIDENCE_THRESHOLD = 0.5

_detector = None


def _get_detector() -> cv2.dnn.Net:
    global _detector
    if _detector is None:
        if not Path(_PROTO).exists() or not Path(_MODEL).exists():
            raise FileNotFoundError(
                "Face detector files not found. Expected: "
                f"{_PROTO} and {_MODEL}"
            )
        _detector = cv2.dnn.readNetFromCaffe(_PROTO, _MODEL)
    return _detector


def _largest_face_ratio(frame: np.ndarray) -> float:
    """Return largest face bounding-box area / frame area for one frame."""
    h, w = frame.shape[:2]
    frame_area = h * w
    if frame_area == 0:
        return 0.0

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net = _get_detector()
    net.setInput(blob)
    detections = net.forward()

    max_area = 0.0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < _CONFIDENCE_THRESHOLD:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        face_area = max(0, x2 - x1) * max(0, y2 - y1)
        max_area = max(max_area, face_area)

    return max_area / frame_area


def _grab_frame(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    """Seek to timestamp and return the decoded frame, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    return frame if ret else None


def face_area_ratio(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> float:
    """
    Compute the face_area_ratio feature for a single shot.

    Samples two frames at 1/3 and 2/3 of the shot duration, detects faces
    using OpenCV's DNN SSD detector, and returns the maximum value of
    (largest face bounding-box area / frame area) across both frames.

    Args:
        video_path: Path to the video file.
        start_sec:  Shot start time in seconds.
        end_sec:    Shot end time in seconds.

    Returns:
        float in [0, 1]. 0.0 if no faces detected or frames cannot be read.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return 0.0

    t1 = start_sec + duration / 3
    t2 = start_sec + 2 * duration / 3

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        ratios = []
        for t in (t1, t2):
            frame = _grab_frame(cap, t)
            if frame is not None:
                ratios.append(_largest_face_ratio(frame))
    finally:
        cap.release()

    return max(ratios) if ratios else 0.0
