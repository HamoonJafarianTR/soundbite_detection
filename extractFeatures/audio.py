import subprocess
import numpy as np
from pathlib import Path

_SAMPLE_RATE = 16000  # 16 kHz mono is sufficient for audio analysis

# Framing shared by RMS and ZCR analysis.
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
) -> tuple[float, float, float]:
    """
    Compute audio energy and speech-discriminative features for a single shot.

    Slices the audio between start_sec and end_sec directly from the video
    via ffmpeg (no temp files), then computes frame-wise statistics.

    Features returned:
        rms_std   – std dev of frame-wise RMS energy; captures loudness
                    dynamics (pauses vs speech bursts).
        zcr_mean  – mean frame-wise zero-crossing rate (ZCR), normalised to
                    [0, 1] by dividing by frame length. Speech produces
                    characteristic mid-range ZCR (~0.05–0.15) whereas
                    background noise and music have distinctly different
                    profiles, making this more discriminative than rms_mean
                    for separating voiceover from genuine soundbite speech.
        rms_mean  – mean frame-wise RMS energy; retained for backwards
                    compatibility but has low feature importance and is
                    superseded by zcr_mean for speech discrimination.

    Args:
        video_path: Path to the video file.
        start_sec:  Shot start time in seconds.
        end_sec:    Shot end time in seconds.

    Returns:
        (rms_std, zcr_mean, rms_mean) — all floats. Returns (0.0, 0.0, 0.0)
        for silent or unreadable segments.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return 0.0, 0.0, 0.0

    y = _load_audio_segment(video_path, start_sec, duration)

    if len(y) == 0:
        return 0.0, 0.0, 0.0

    frames = np.lib.stride_tricks.sliding_window_view(y, _FRAME_LENGTH)[::_HOP_LENGTH]

    # RMS energy per frame
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Zero-crossing rate: count sign changes per frame, normalised by frame length
    signs = np.sign(frames)
    signs[signs == 0] = 1  # treat zero samples as positive to avoid double-counting
    zcr = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / (2 * _FRAME_LENGTH)

    return float(np.std(rms)), float(np.mean(zcr)), float(np.mean(rms))
