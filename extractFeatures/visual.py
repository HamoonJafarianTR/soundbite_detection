import cv2
import numpy as np
from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent / "models" / "face_detector"
_YUNET_MODEL = str(_MODEL_DIR / "face_detection_yunet_2023mar.onnx")

# YuNet detection thresholds.
_CONFIDENCE_THRESHOLD = 0.6   # min detection score to accept a face
_NMS_THRESHOLD = 0.3
_TOP_K = 50

# Minimum face area (as a fraction of the frame) for a detection to count as a
# real on-screen face. Tiny spurious detections (e.g. a face-like pattern in
# b-roll occupying <1% of the frame) are below this and are ignored, so they
# can't inflate face_presence. ~0.01 sits well above typical phantom
# detections but below genuine small/field-shot faces; tune as needed.
_MIN_FACE_AREA = 0.01

# Temporal sampling: one frame per second of shot, bounded so that very short
# shots still get a few samples and very long shots stay cheap.
_SAMPLE_FPS = 1.0
_MIN_SAMPLES = 3
_MAX_SAMPLES = 10

_detector = None
_detector_size: tuple[int, int] | None = None


def _get_detector(width: int, height: int) -> "cv2.FaceDetectorYN":
    """Return a cached YuNet detector configured for the given frame size.

    YuNet requires the input size to match the frame it is run on, so the
    detector is reconfigured whenever the frame resolution changes.
    """
    global _detector, _detector_size
    if not Path(_YUNET_MODEL).is_file():
        raise FileNotFoundError(
            f"YuNet face detector model not found at {_YUNET_MODEL}. "
            "Download face_detection_yunet_2023mar.onnx from the OpenCV Zoo "
            "(models/face_detection_yunet) and place it there."
        )
    size = (int(width), int(height))
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(
            _YUNET_MODEL, "", size,
            _CONFIDENCE_THRESHOLD, _NMS_THRESHOLD, _TOP_K,
        )
        _detector_size = size
    elif _detector_size != size:
        _detector.setInputSize(size)
        _detector_size = size
    return _detector


def _largest_face_ratio(frame: np.ndarray) -> float:
    """Return largest face bounding-box area / frame area for one frame.

    Detection runs at the frame's native resolution (no down-squash to
    300x300), so small / off-axis faces survive far better than with the old
    SSD detector.
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    if frame_area == 0:
        return 0.0

    det = _get_detector(w, h)
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return 0.0

    # Each face row is [x, y, w, h, <5 landmark x/y pairs>, score].
    max_area = 0.0
    for f in faces:
        fw = max(0.0, float(f[2]))
        fh = max(0.0, float(f[3]))
        max_area = max(max_area, fw * fh)

    return max_area / frame_area


def _grab_frame(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    """Seek to timestamp and return the decoded frame, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    return frame if ret else None


def _sample_times(start_sec: float, end_sec: float) -> list[float]:
    """Evenly spaced sample timestamps inside a shot (~1 fps, bounded)."""
    duration = end_sec - start_sec
    n = int(round(duration * _SAMPLE_FPS))
    n = max(_MIN_SAMPLES, min(_MAX_SAMPLES, n))
    return [start_sec + duration * (i + 1) / (n + 1) for i in range(n)]


def _collect_ratios(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> list[float]:
    """Sample frames across a shot and return the largest-face area ratio per frame.

    One frame per second of shot (min 3, max 10), detected with YuNet at native
    resolution. Each value is (largest face area / frame area); 0.0 for frames
    with no detected face. Frames that fail to decode are skipped.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        ratios = []
        for t in _sample_times(start_sec, end_sec):
            frame = _grab_frame(cap, t)
            if frame is not None:
                ratios.append(_largest_face_ratio(frame))
    finally:
        cap.release()

    return ratios


def face_presence(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> float:
    """
    Compute the face_presence feature for a single shot.

    face_presence is the fraction of sampled frames that contain an accepted
    face, i.e. a detection larger than _MIN_FACE_AREA of the frame. The size
    gate means a microscopic phantom detection does not count, so this answers
    "how consistently is a real person on screen during the shot?".

    Returns:
        float in [0, 1]; 0.0 when no sufficiently large face is found.
    """
    ratios = _collect_ratios(video_path, start_sec, end_sec)
    if not ratios:
        return 0.0
    accepted = sum(1 for r in ratios if r > _MIN_FACE_AREA)
    return accepted / len(ratios)
