import cv2
import numpy as np
from pathlib import Path

_MODEL_DIR = Path(__file__).parent / "models" / "face_detector"
_YUNET_MODEL = str(_MODEL_DIR / "face_detection_yunet_2023mar.onnx")

# YuNet detection thresholds.
_CONFIDENCE_THRESHOLD = 0.6   # min detection score to accept a face
_NMS_THRESHOLD = 0.3
_TOP_K = 50

# Temporal sampling: exactly one frame per second of shot duration (1 Hz).
_SAMPLE_FPS = 1.0

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

    Uses YuNet's score threshold only; no minimum face-size cutoff. Detection
    runs at the video's native frame resolution.
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
    max_ratio = 0.0
    for f in faces:
        fw = max(0.0, float(f[2]))
        fh = max(0.0, float(f[3]))
        max_ratio = max(max_ratio, (fw * fh) / frame_area)

    return max_ratio


def _grab_frame(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    """Seek to timestamp and return the decoded frame, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    return frame if ret else None


def _sample_times(start_sec: float, end_sec: float) -> list[float]:
    """Return one sample timestamp per second of shot duration (1 Hz, no cap).

    An 82 s shot yields 82 samples at start+0.5 s, start+1.5 s, …, start+81.5 s.
    Sub-second shots still get a single sample at the midpoint.
    """
    duration = end_sec - start_sec
    n = max(1, int(duration * _SAMPLE_FPS))
    return [start_sec + i + 0.5 for i in range(n)]


def _collect_ratios(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> list[float]:
    """Sample frames across a shot and return the largest-face area ratio per frame.

    One frame per second of shot (no upper cap), detected with YuNet at native
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


def face_features(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> tuple[float, float, float, float]:
    """
    Compute visual face features for a single shot.

    Samples one frame per second, runs YuNet on each, and takes the largest
    detected face per frame. Returns:

        face_presence    – mean of per-frame largest-face area ratios
        max_face_ratio   – max of per-frame largest-face area ratios
        face_consistency – fraction of sampled frames in which at least one
                           face was detected (ratio > 0). This distinguishes
                           a consistently visible speaker (soundbite) from an
                           occasional face in B-roll or reporter voiceover.
        face_ratio_std   – std dev of per-frame face area ratios. A fixed
                           tripod shot of a reporter standup yields low std
                           (face stays constant size); interview subjects or
                           slight camera movement yield higher std. Helps
                           separate reporter standups from genuine soundbites
                           even when face_consistency is 1.0 for both.

    All values are non-negative floats.
    """
    ratios = _collect_ratios(video_path, start_sec, end_sec)
    if not ratios:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(ratios, dtype=np.float64)
    n = len(ratios)
    face_presence    = float(arr.mean())
    max_face_ratio   = float(arr.max())
    face_consistency = float((arr > 0).sum() / n)
    face_ratio_std   = float(arr.std())
    return face_presence, max_face_ratio, face_consistency, face_ratio_std


def face_presence(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> float:
    """Return mean largest-face area ratio (see face_features)."""
    return face_features(video_path, start_sec, end_sec)[0]


def face_consistency(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> float:
    """Return fraction of sampled frames with a detected face (see face_features)."""
    return face_features(video_path, start_sec, end_sec)[2]


def face_ratio_std(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> float:
    """Return std dev of per-frame face area ratios (see face_features)."""
    return face_features(video_path, start_sec, end_sec)[3]
