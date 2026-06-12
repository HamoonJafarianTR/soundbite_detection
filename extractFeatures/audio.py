import subprocess
import numpy as np
from pathlib import Path

_SAMPLE_RATE = 16000  # 16 kHz mono is sufficient for RMS analysis

# Framing for RMS analysis.
_FRAME_LENGTH = 2048
_HOP_LENGTH = 512


def _load_audio_segment(
    video_path: str | Path,
    start_sec: float,
    duration: float,
) -> np.ndarray:
    """
    Use ffmpeg to decode a time-sliced audio segment from a video file into
    a raw float32 PCM numpy array, avoiding any intermediate temp files.
    """
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", str(_SAMPLE_RATE),
        "-f", "f32le",
        "-",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    raw = np.frombuffer(result.stdout, dtype=np.float32)
    return raw


def audio_features(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> tuple[float, float]:
    """
    Compute RMS energy statistics for a single shot.

    Slices the audio between start_sec and end_sec directly from the video
    via ffmpeg (no temp files), computes frame-wise RMS energy, and returns
    its mean and standard deviation.

    Args:
        video_path: Path to the video file.
        start_sec:  Shot start time in seconds.
        end_sec:    Shot end time in seconds.

    Returns:
        (rms_mean, rms_std) — both floats. Returns (0.0, 0.0) for silent
        or unreadable segments.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return 0.0, 0.0

    y = _load_audio_segment(video_path, start_sec, duration)

    if len(y) == 0:
        return 0.0, 0.0

    # Compute frame-wise RMS with pure numpy — avoids librosa/numba caching issues
    frames = np.lib.stride_tricks.sliding_window_view(y, _FRAME_LENGTH)[::_HOP_LENGTH]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    return float(np.mean(rms)), float(np.std(rms))
