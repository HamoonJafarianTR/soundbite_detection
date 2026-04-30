"""Feature extraction for the soundbite classifier.

Provides timecode utilities, audio / transcript / temporal feature
extraction, ground-truth labeling, and orchestrator functions that
produce a single DataFrame ready for ML training or inference.
"""

import functools
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import librosa
import numpy as np
import pandas as pd

import config as cfg

logger = logging.getLogger(__name__)

# Reuters videos are keyed by an LOV/RW id like ``LOV395620042026RP1``.
# We use this id as a stable handle that ties together the video file, the
# transcript, the detected-shots JSON, and the ground-truth URI.
_VIDEO_ID_RE = re.compile(r"(?:LOV|RW)\d+RP\d+", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════════
# Timecode utilities
# ═══════════════════════════════════════════════════════════════════════════════

def tc_frames_to_seconds(tc: str, fps: int = cfg.FPS) -> float:
    """Convert *HH:MM:SS:FF* (frame-based) timecode to seconds.

    Kept as a fallback for legacy detected-shots files that only contain
    the frame-based ``timecode`` array (newer files include a precomputed
    ``timecodes_seconds`` array).
    """
    parts = tc.strip().split(":")
    h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    return h * 3600 + m * 60 + s + f / fps


# ═══════════════════════════════════════════════════════════════════════════════
# Segment building from detected shots
# ═══════════════════════════════════════════════════════════════════════════════

def parse_detected_shots(json_path: str, fps: Optional[int] = None) -> List[float]:
    """Return shot-boundary times (seconds) from a detected-shots JSON.

    The JSON has top-level metadata keys (strings/ints) plus one key whose
    value is a dict containing a precomputed ``timecodes_seconds`` array
    (preferred) and a frame-based ``timecode`` array (fallback).

    ``fps`` is only used by the frame-based fallback. When ``None`` (default)
    it defaults to ``cfg.FPS`` (training videos are 25 fps); pass an explicit
    value (e.g. ``30``) for video sets with a different frame rate.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    fps_to_use = cfg.FPS if fps is None else fps

    for value in data.values():
        if not isinstance(value, dict):
            continue
        if "timecodes_seconds" in value:
            return [float(t) for t in value["timecodes_seconds"]]
        if "timecode" in value:
            return [tc_frames_to_seconds(tc, fps=fps_to_use) for tc in value["timecode"]]

    raise ValueError(f"No timecode array found in {json_path}")


def get_video_duration(video_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def build_segments(
    shot_times: List[float], video_duration: float
) -> List[Tuple[float, float]]:
    """Turn an ordered list of shot start-times into (start, end) pairs.

    The final segment extends from the last shot boundary to *video_duration*.
    """
    segments = []
    for i in range(len(shot_times) - 1):
        segments.append((shot_times[i], shot_times[i + 1]))
    if shot_times:
        segments.append((shot_times[-1], video_duration))
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth loading & labeling
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_video_id(name: str) -> Optional[str]:
    """Return the canonical ``LOV<digits>RP<digits>`` id from any Reuters
    filename / URI / S3 key, or ``None`` if not found."""
    if not name:
        return None
    m = _VIDEO_ID_RE.search(name)
    if not m:
        return None
    # Normalise ``RW...`` → ``LOV...`` so video-file ids and URI ids share a key.
    token = m.group(0).upper()
    if token.startswith("RW"):
        token = "LOV" + token[2:]
    return token


@functools.lru_cache(maxsize=1)
def _video_id_to_uri_map() -> Dict[str, str]:
    """Map ``LOV<digits>RP<digits>`` ids → Reuters URI tag.

    Built once from ``english_cleaned_dataset.json``, which is the source of
    truth linking each URI to its S3 video URI.
    """
    src = os.path.join(cfg.PROJECT_ROOT, "english_cleaned_dataset.json")
    if not os.path.exists(src):
        logger.warning(
            "Cannot build URI map: missing %s (ground-truth lookup will fail).",
            src,
        )
        return {}
    with open(src, "r") as f:
        data = json.load(f)

    out: Dict[str, str] = {}
    for uri, item in data.items():
        if not isinstance(item, dict):
            continue
        s3 = item.get("s3_uri") or ""
        vid = _extract_video_id(s3) or _extract_video_id(uri)
        if vid:
            out[vid] = uri
    return out


def load_ground_truth(video_name: str) -> List[Tuple[float, float]]:
    """Return list of (start, end) in seconds for real soundbites.

    The ground-truth JSON (``shots_gt_data.json``) is keyed by Reuters URI tag
    (e.g. ``tag:reuters.com,2026:newsml_RW...``) and stores ``soundbite_start``/
    ``soundbite_end`` as floats (seconds, ms-precision). We resolve the URI by
    extracting the LOV/RW id from *video_name* and looking it up via the
    cleaned-dataset map.
    """
    with open(cfg.GT_PATH, "r") as f:
        gt_data = json.load(f)

    vid = _extract_video_id(video_name)
    uri = _video_id_to_uri_map().get(vid) if vid else None
    if uri is None or uri not in gt_data:
        logger.warning("No ground truth found for '%s'", video_name)
        return []

    ranges: List[Tuple[float, float]] = []
    for entry in gt_data[uri]:
        try:
            start = float(entry["soundbite_start"])
            end = float(entry["soundbite_end"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Malformed GT entry for %s: %r", uri, entry)
            continue
        ranges.append((start, end))
    return ranges


def label_segment(
    seg_start: float,
    seg_end: float,
    gt_ranges: List[Tuple[float, float]],
) -> int:
    """1 if the segment is a real soundbite, 0 otherwise.

    A segment is positive when ANY of the following holds for some GT range:
      * IoU(seg, gt) > ``cfg.IOU_THRESHOLD``                  (rough alignment)
      * intersection / seg_dur > ``cfg.SEG_OVERLAP_RATIO_THRESHOLD``
        — the segment is mostly *inside* a soundbite (long soundbite split
        across multiple shots).
      * intersection / gt_dur > ``cfg.GT_OVERLAP_RATIO_THRESHOLD``
        — the segment *contains* most of a soundbite (long shot with a
        short soundbite + extra silence/B-roll); previously these were
        silently labeled 0.
    """
    seg_dur = seg_end - seg_start
    if seg_dur <= 0:
        return 0

    seg_thr = getattr(cfg, "SEG_OVERLAP_RATIO_THRESHOLD", cfg.OVERLAP_RATIO_THRESHOLD)
    gt_thr = getattr(cfg, "GT_OVERLAP_RATIO_THRESHOLD", cfg.OVERLAP_RATIO_THRESHOLD)
    gt_match_min_seg = getattr(cfg, "GT_MATCH_MIN_SEG_FRACTION", 0.0)

    for gt_start, gt_end in gt_ranges:
        inter_start = max(seg_start, gt_start)
        inter_end = min(seg_end, gt_end)
        intersection = max(0.0, inter_end - inter_start)
        if intersection == 0:
            continue

        gt_dur = max(1e-9, gt_end - gt_start)
        union = seg_dur + gt_dur - intersection
        iou = intersection / union if union > 0 else 0.0
        seg_overlap_ratio = intersection / seg_dur
        gt_overlap_ratio = intersection / gt_dur

        if iou > cfg.IOU_THRESHOLD:
            return 1
        if seg_overlap_ratio > seg_thr:
            return 1
        if (
            gt_overlap_ratio > gt_thr
            and seg_overlap_ratio >= gt_match_min_seg
        ):
            return 1
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Audio feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

_audio_cache: Dict[str, str] = {}


def extract_audio_from_video(video_path: str) -> str:
    """Convert video audio to a temporary mono WAV and cache the path."""
    if video_path in _audio_cache and os.path.exists(_audio_cache[video_path]):
        return _audio_cache[video_path]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(cfg.AUDIO_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        tmp.name,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    _audio_cache[video_path] = tmp.name
    logger.info("Extracted audio → %s", tmp.name)
    return tmp.name


def extract_audio_features(
    audio_path: str, seg_start: float, seg_end: float
) -> Dict[str, float]:
    """Compute audio features for a single segment."""
    duration = seg_end - seg_start
    nan_dict = {feat: np.nan for feat in cfg.AUDIO_FEATURES}

    if duration < 0.1:
        return nan_dict

    try:
        y, sr = librosa.load(
            audio_path,
            sr=cfg.AUDIO_SAMPLE_RATE,
            offset=seg_start,
            duration=duration,
            mono=True,
        )
    except Exception:
        logger.warning("Could not load audio %.2f–%.2f", seg_start, seg_end)
        return nan_dict

    if len(y) < sr * 0.05:
        return nan_dict

    features: Dict[str, float] = {}

    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(cent))
    features["spectral_centroid_std"] = float(np.std(cent))

    # SNR estimate: ratio of overall RMS to RMS of the quietest 10 % of frames
    if len(rms) >= 5:
        sorted_rms = np.sort(rms)
        noise_floor = np.mean(sorted_rms[: max(1, len(sorted_rms) // 10)])
        if noise_floor > 0:
            features["snr_estimate"] = float(
                20 * np.log10(np.mean(rms) / noise_floor)
            )
        else:
            features["snr_estimate"] = 60.0  # very clean
    else:
        features["snr_estimate"] = np.nan

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Transcript feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _load_transcript(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_transcript_features(
    items: list,
    audio_segments: list,
    seg_start: float,
    seg_end: float,
) -> Dict[str, float]:
    """Compute transcript-derived features for one segment."""
    seg_dur = seg_end - seg_start
    defaults = {feat: 0.0 for feat in cfg.TRANSCRIPT_FEATURES}
    defaults["confidence_min"] = np.nan
    defaults["confidence_mean"] = np.nan
    defaults["confidence_std"] = np.nan

    # Filter pronunciation items falling inside the segment
    words_in_seg = []
    for item in items:
        if item["type"] != "pronunciation":
            continue
        t = float(item["start_time"])
        if seg_start <= t < seg_end:
            conf = float(item["alternatives"][0]["confidence"])
            word = item["alternatives"][0]["content"].lower()
            start_t = float(item["start_time"])
            end_t = float(item["end_time"])
            words_in_seg.append((word, conf, start_t, end_t))

    n = len(words_in_seg)
    features: Dict[str, float] = {}
    features["word_count"] = float(n)
    features["speech_rate"] = n / seg_dur if seg_dur > 0 else 0.0

    if n == 0:
        features["confidence_mean"] = np.nan
        features["confidence_std"] = np.nan
        features["confidence_min"] = np.nan
        features["low_confidence_ratio"] = 0.0
        features["filler_word_ratio"] = 0.0
        features["speech_coverage"] = 0.0
        features["max_silence_gap"] = seg_dur
    else:
        confs = [w[1] for w in words_in_seg]
        features["confidence_mean"] = float(np.mean(confs))
        features["confidence_std"] = float(np.std(confs))
        features["confidence_min"] = float(np.min(confs))
        features["low_confidence_ratio"] = (
            sum(1 for c in confs if c < cfg.LOW_CONFIDENCE_THRESHOLD) / n
        )
        features["filler_word_ratio"] = (
            sum(1 for w in words_in_seg if w[0] in cfg.FILLER_WORDS) / n
        )

        total_speech_dur = sum(w[3] - w[2] for w in words_in_seg)
        features["speech_coverage"] = total_speech_dur / seg_dur if seg_dur > 0 else 0.0

        # Max silence gap between consecutive words
        sorted_words = sorted(words_in_seg, key=lambda w: w[2])
        max_gap = 0.0
        for i in range(1, len(sorted_words)):
            gap = sorted_words[i][2] - sorted_words[i - 1][3]
            max_gap = max(max_gap, gap)
        features["max_silence_gap"] = max_gap

    # Count overlapping audio_segments
    overlap_count = 0
    for aseg in audio_segments:
        a_start = float(aseg["start_time"])
        a_end = float(aseg["end_time"])
        if a_start < seg_end and a_end > seg_start:
            overlap_count += 1
    features["num_audio_segments_overlap"] = float(overlap_count)

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal / shot features
# ═══════════════════════════════════════════════════════════════════════════════

def extract_temporal_features(
    seg_start: float,
    seg_end: float,
    video_duration: float,
    seg_index: int,
    total_segments: int,
) -> Dict[str, float]:
    duration = seg_end - seg_start
    midpoint = (seg_start + seg_end) / 2.0
    return {
        "segment_duration": duration,
        "relative_position": midpoint / video_duration if video_duration > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visual feature extraction (face detection via OpenCV DNN)
# ═══════════════════════════════════════════════════════════════════════════════

_face_net = None


def _get_face_detector():
    """Lazy-load the OpenCV DNN face detector (Caffe model shipped with OpenCV)."""
    global _face_net
    if _face_net is not None:
        return _face_net

    proto_path = os.path.join(
        os.path.dirname(cv2.__file__), "data",
        "deploy.prototxt",
    )
    model_path = os.path.join(
        os.path.dirname(cv2.__file__), "data",
        "res10_300x300_ssd_iter_140000.caffemodel",
    )

    if os.path.exists(proto_path) and os.path.exists(model_path):
        _face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        logger.info("Loaded OpenCV DNN face detector")
    else:
        logger.info("DNN model not found, falling back to Haar cascade")
        _face_net = None
    return _face_net


def _extract_frame(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Extract a single frame at *timestamp* seconds using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1", "-q:v", "2",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        os.unlink(tmp.name)
        return None
    frame = cv2.imread(tmp.name)
    os.unlink(tmp.name)
    return frame


def _detect_faces_dnn(frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    """Detect faces using OpenCV DNN. Returns list of (x1, y1, x2, y2, confidence)."""
    net = _get_face_detector()
    if net is None:
        return _detect_faces_haar(frame)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < cfg.FACE_CONFIDENCE_THRESHOLD:
            continue
        x1 = max(0.0, float(detections[0, 0, i, 3]))
        y1 = max(0.0, float(detections[0, 0, i, 4]))
        x2 = min(1.0, float(detections[0, 0, i, 5]))
        y2 = min(1.0, float(detections[0, 0, i, 6]))
        faces.append((x1 * w, y1 * h, x2 * w, y2 * h, conf))
    return faces


def _detect_faces_haar(frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    """Fallback: detect faces using Haar cascade."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    faces = []
    for (x, y, w, h) in rects:
        faces.append((float(x), float(y), float(x + w), float(y + h), 1.0))
    return faces


def _video_stem(video_name: str) -> str:
    """Base name without extension, safe for filesystem."""
    base = os.path.basename(video_name)
    return os.path.splitext(base)[0]


def _face_area_ratio_for_frame(frame: np.ndarray) -> float:
    """Largest face bounding-box area / frame area (0.0 if no face)."""
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return 0.0
    faces = _detect_faces_dnn(frame)
    if not faces:
        return 0.0
    largest_area = max(
        (x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _ in faces
    )
    return largest_area / frame_area


def _segment_visual_timestamps(seg_start: float, seg_end: float) -> Tuple[float, float]:
    """Two sample times within [seg_start, seg_end] using config fractions."""
    dur = seg_end - seg_start
    if dur <= 0:
        return seg_start, seg_start
    f0, f1 = cfg.VISUAL_FRAME_FRACTIONS
    return seg_start + f0 * dur, seg_start + f1 * dur


def extract_visual_features(
    video_path: str,
    seg_start: float,
    seg_end: float,
    video_name: str,
    segment_index: int,
    save_frame: bool = True,
) -> Dict[str, float]:
    """Extract face features from two frames per segment.

    Frames are taken at ``VISUAL_FRAME_FRACTIONS`` along the segment (default: 1/3
    and 2/3). ``face_area_ratio`` is the **maximum** of the per-frame ratios, so a
    face in either frame counts toward the feature.

    When *save_frame* is True, writes JPEGs to
    ``frames/{VIDEO_STEM}_seg_{SEGMENT_INDEX:04d}_f0.jpg`` and ``_f1.jpg``.
    """
    defaults = {feat: 0.0 for feat in cfg.VISUAL_FEATURES}

    t0, t1 = _segment_visual_timestamps(seg_start, seg_end)
    stem = _video_stem(video_name)
    ratios: List[float] = []

    for fi, t in enumerate((t0, t1)):
        frame = _extract_frame(video_path, t)
        if frame is None:
            logger.warning(
                "Could not extract frame at %.2fs (seg %d sample %d)",
                t,
                segment_index,
                fi,
            )
            continue

        if save_frame:
            os.makedirs(cfg.FRAMES_DIR, exist_ok=True)
            out_path = os.path.join(
                cfg.FRAMES_DIR,
                f"{stem}_seg_{segment_index:04d}_f{fi}.jpg",
            )
            if not cv2.imwrite(out_path, frame):
                logger.warning("Could not write frame to %s", out_path)

        ratios.append(_face_area_ratio_for_frame(frame))

    if not ratios:
        return defaults

    return {"face_area_ratio": max(ratios)}


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrators
# ═══════════════════════════════════════════════════════════════════════════════

def _find_in_dir(directory: str, video_id: str, suffix_filter: str = "") -> Optional[str]:
    """Return the first file in *directory* whose name contains *video_id*
    (case-insensitive). Optionally restrict to files ending in *suffix_filter*.
    """
    if not os.path.isdir(directory) or not video_id:
        return None
    needle = video_id.upper()
    for f in os.listdir(directory):
        if suffix_filter and not f.endswith(suffix_filter):
            continue
        if needle in f.upper():
            return os.path.join(directory, f)
    return None


def _resolve_paths(video_name: str, video_path, transcript_path, shots_path):
    """Derive default paths from the video name when not explicitly given.

    File names in the various data folders are not perfectly consistent
    (mixed case, legacy ``.split('.')[0]`` artefacts in detected-shots
    filenames). We resolve everything via the LOV/RW video id whenever
    possible, and fall back to the literal basename if the id cannot be
    extracted.
    """
    vid = _extract_video_id(video_name)
    base = video_name
    for ext in (".MP4", ".mp4"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    if video_path is None:
        cand = _find_in_dir(cfg.VIDEOS_DIR, vid) if vid else None
        if cand is None:
            cand = os.path.join(cfg.VIDEOS_DIR, f"{base}.MP4")
        video_path = cand

    if transcript_path is None:
        cand = (
            _find_in_dir(cfg.TRANSCRIPTS_DIR, vid, "_transcript_raw.json")
            if vid
            else None
        )
        if cand is None:
            cand = os.path.join(
                cfg.TRANSCRIPTS_DIR, f"{base}_transcript_raw.json"
            )
        transcript_path = cand

    if shots_path is None:
        cand = (
            _find_in_dir(
                cfg.DETECTED_SHOTS_DIR,
                vid,
                "_weighted_adaptive_prediction.json",
            )
            if vid
            else None
        )
        if cand is None:
            cand = os.path.join(
                cfg.DETECTED_SHOTS_DIR,
                f"{base}_weighted_adaptive_prediction.json",
            )
        shots_path = cand

    return video_path, transcript_path, shots_path


def extract_features_for_video(
    video_name: str,
    video_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    shots_path: Optional[str] = None,
    gt_ranges: Optional[List[Tuple[float, float]]] = None,
    compute_labels: bool = True,
    save_visual_frames: bool = True,
    fps: Optional[int] = None,
) -> pd.DataFrame:
    """Extract all features for every segment of a single video.

    Returns a DataFrame with one row per segment. ``fps`` is forwarded to
    ``parse_detected_shots`` for video sets whose detected-shots JSON only
    contains frame-based timecodes at a non-default frame rate.
    """
    video_path, transcript_path, shots_path = _resolve_paths(
        video_name, video_path, transcript_path, shots_path
    )

    logger.info("Processing %s", video_name)

    # --- segments -----------------------------------------------------------
    shot_times = parse_detected_shots(shots_path, fps=fps)
    vid_duration = get_video_duration(video_path)
    segments = build_segments(shot_times, vid_duration)
    total_segs = len(segments)
    logger.info("  %d segments (duration %.1fs)", total_segs, vid_duration)

    # --- audio WAV ----------------------------------------------------------
    audio_path = extract_audio_from_video(video_path)

    # --- transcript ---------------------------------------------------------
    if os.path.exists(transcript_path):
        transcript = _load_transcript(transcript_path)
        items = transcript["results"]["items"]
        audio_segments = transcript["results"].get("audio_segments", [])
    else:
        logger.warning("No transcript at %s", transcript_path)
        items = []
        audio_segments = []

    # --- ground truth -------------------------------------------------------
    if compute_labels and gt_ranges is None:
        gt_ranges = load_ground_truth(video_name)

    # --- per-segment feature vectors ----------------------------------------
    rows = []
    for idx, (seg_start, seg_end) in enumerate(segments):
        feat: Dict[str, object] = {}

        feat.update(extract_audio_features(audio_path, seg_start, seg_end))
        feat.update(
            extract_transcript_features(items, audio_segments, seg_start, seg_end)
        )
        feat.update(
            extract_temporal_features(
                seg_start, seg_end, vid_duration, idx, total_segs
            )
        )
        feat.update(
            extract_visual_features(
                video_path,
                seg_start,
                seg_end,
                video_name,
                idx,
                save_frame=save_visual_frames,
            )
        )

        feat["video_name"] = video_name
        feat["segment_index"] = idx
        feat["segment_start"] = seg_start
        feat["segment_end"] = seg_end

        if compute_labels and gt_ranges is not None:
            feat["label"] = label_segment(seg_start, seg_end, gt_ranges)
        else:
            feat["label"] = np.nan

        rows.append(feat)

    df = pd.DataFrame(rows)
    col_order = cfg.META_COLUMNS + cfg.FEATURE_COLUMNS + [cfg.LABEL_COLUMN]
    for c in col_order:
        if c not in df.columns:
            df[c] = np.nan
    df = df[col_order]
    return df


def list_available_video_names() -> List[str]:
    """Return the basenames of every video that has a detected-shots file
    *and* an actual MP4 on disk. This is the canonical list of training videos.
    """
    names: List[str] = []
    shot_files = sorted(
        f
        for f in os.listdir(cfg.DETECTED_SHOTS_DIR)
        if f.endswith("_weighted_adaptive_prediction.json")
    )
    for shot_file in shot_files:
        vid = _extract_video_id(shot_file)
        video_path = _find_in_dir(cfg.VIDEOS_DIR, vid) if vid else None
        if video_path is None or not os.path.exists(video_path):
            continue
        names.append(os.path.basename(video_path))
    return names


def extract_features_for_all_videos(
    skip_video_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Extract features for every video that has a detected-shots file.

    When ``skip_video_names`` is provided, any video whose basename is in the
    set is skipped (used for incremental cache updates).
    """
    skip = {os.path.basename(n) for n in (skip_video_names or [])}

    all_frames: List[pd.DataFrame] = []
    shot_files = sorted(
        f
        for f in os.listdir(cfg.DETECTED_SHOTS_DIR)
        if f.endswith("_weighted_adaptive_prediction.json")
    )
    for shot_file in shot_files:
        shots_path = os.path.join(cfg.DETECTED_SHOTS_DIR, shot_file)

        vid = _extract_video_id(shot_file)
        video_path = _find_in_dir(cfg.VIDEOS_DIR, vid) if vid else None
        if video_path is None or not os.path.exists(video_path):
            logger.warning(
                "Video not found for shot file %s – skipping", shot_file
            )
            continue

        video_name = os.path.basename(video_path)
        if video_name in skip:
            logger.debug("Skipping %s (already in cache)", video_name)
            continue

        transcript_path = (
            _find_in_dir(cfg.TRANSCRIPTS_DIR, vid, "_transcript_raw.json")
            if vid
            else None
        )

        df = extract_features_for_video(
            video_name,
            video_path=video_path,
            transcript_path=transcript_path,
            shots_path=shots_path,
        )
        all_frames.append(df)
        logger.info(
            "  → %d segments (%d soundbite)",
            len(df),
            int(df["label"].sum()),
        )

    if not all_frames:
        if not skip:
            logger.warning("No videos processed; returning empty DataFrame.")
        return pd.DataFrame(
            columns=cfg.META_COLUMNS + cfg.FEATURE_COLUMNS + [cfg.LABEL_COLUMN]
        )
    return pd.concat(all_frames, ignore_index=True)
