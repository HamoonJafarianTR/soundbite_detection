"""Shared configuration for the soundbite classifier pipeline."""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Input directories ──────────────────────────────────────────────────────────
VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "transcripts")
DETECTED_SHOTS_DIR = os.path.join(PROJECT_ROOT, "detected_shots")
GT_PATH = os.path.join(PROJECT_ROOT, "shots_gt_data.json")

# Evaluation / hold-out set only (see ``evaluate.py``)
TEST_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "test_videos")
TEST_TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "test_transcriptions")
TEST_DETECTED_SHOTS_DIR = os.path.join(PROJECT_ROOT, "test_detected_shots")
TEST_GT_PATH = os.path.join(PROJECT_ROOT, "test_shots_gt_data.json")

# ── Output directories ─────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames")

# ── Video / timecode ───────────────────────────────────────────────────────────
FPS = 25
AUDIO_SAMPLE_RATE = 16000  # mono WAV extracted for librosa

# ── Labeling thresholds ────────────────────────────────────────────────────────
# A segment is positive if ANY of these holds for some GT soundbite:
#   * IoU(seg, gt) > IOU_THRESHOLD                                 (rough alignment)
#   * intersection / seg_dur > SEG_OVERLAP_RATIO_THRESHOLD         (segment mostly inside a soundbite)
#   * intersection / gt_dur  > GT_OVERLAP_RATIO_THRESHOLD AND
#     intersection / seg_dur > GT_MATCH_MIN_SEG_FRACTION            (segment contains most of a soundbite)
# The third asymmetric check fixes the case of long shots that contain a
# short soundbite plus extra silence/B-roll: previously these were silently
# labeled 0. ``GT_MATCH_MIN_SEG_FRACTION`` guards against the degenerate
# opposite case (a tiny soundbite inside a very long shot) so we do not
# label a segment that is mostly silence as a soundbite.
IOU_THRESHOLD = 0.3
OVERLAP_RATIO_THRESHOLD = 0.5            # legacy alias kept for backward-compat
SEG_OVERLAP_RATIO_THRESHOLD = OVERLAP_RATIO_THRESHOLD
GT_OVERLAP_RATIO_THRESHOLD = 0.5
GT_MATCH_MIN_SEG_FRACTION = 0.2

# ── Transcript analysis ────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.5
FILLER_WORDS = {"uh", "um", "ah", "er", "hmm"}

# ── ML ─────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Feature column names (kept in a canonical order) ───────────────────────────
AUDIO_FEATURES = [
    "rms_mean", "rms_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "snr_estimate",
]

TRANSCRIPT_FEATURES = [
    "word_count", "speech_rate",
    "confidence_mean", "confidence_std", "confidence_min",
    "low_confidence_ratio",
    "filler_word_ratio",
    "speech_coverage",
    "num_audio_segments_overlap",
    "max_silence_gap",
]

TEMPORAL_FEATURES = [
    "segment_duration",
    "relative_position",
]

VISUAL_FEATURES = [
    "face_area_ratio",
]

# Face detection confidence threshold (OpenCV DNN)
FACE_CONFIDENCE_THRESHOLD = 0.5

# Two timestamps per segment: fractions of segment length from seg_start (0–1).
VISUAL_FRAME_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)

FEATURE_COLUMNS = (
    AUDIO_FEATURES + TRANSCRIPT_FEATURES + TEMPORAL_FEATURES + VISUAL_FEATURES
)

# Minimal feature set: top 8 discriminators only
REDUCED_FEATURE_COLUMNS = [
    "word_count",
    "rms_std",
    "rms_mean",
    "confidence_mean",
    "face_area_ratio",
    "segment_duration",
    "speech_coverage",
    "speech_rate",
]

META_COLUMNS = ["video_name", "segment_index", "segment_start", "segment_end"]
LABEL_COLUMN = "label"


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (DATA_DIR, MODELS_DIR, EVAL_DIR, FRAMES_DIR):
        os.makedirs(d, exist_ok=True)
