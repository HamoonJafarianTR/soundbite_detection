"""
End-to-end inference pipeline for a single video.

  1. Transcribes audio with Whisper  → transcripts/
  2. Detects shots                   → detected_shots/
  3. Extracts features per shot      (no GT label)
  4. Runs the trained model
  5. Writes predictions.json

Run with defaults:

    python inferencePerVideo.py

or pass explicit paths:

    python inferencePerVideo.py --video videos/my_video.mp4 --model models/model_xgboost_v10.joblib
"""

import argparse
import json
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from faster_whisper import WhisperModel

# ── project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepareData.whisper_transcibe import transcribe_video_to_json
from prepareData.shotDetection import weighted_adaptive_threshold_shot_detection
from extractFeatures import visual as visual_module
from extractFeatures.visual import face_features
from extractFeatures.audio import audio_features
from extractFeatures.text import transcript_features

VIDEOS_DIR = ROOT / "videos"
TRANSCRIPTS_DIR = ROOT / "transcripts"
SHOTS_DIR = ROOT / "detected_shots"
OUTPUT_JSON = ROOT / "predictions.json"

DEFAULT_MODEL = ROOT / "models" / "model_xgboost_v10.joblib"
DEFAULT_WHISPER_MODEL = "medium"

# Backward-compatible fallback for face detector model location.
_DEFAULT_YUNET_PATH = ROOT / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"
if not Path(visual_module._YUNET_MODEL).is_file() and _DEFAULT_YUNET_PATH.is_file():
    visual_module._YUNET_MODEL = str(_DEFAULT_YUNET_PATH)

# Model input features (must match the trained model).
FEATURE_COLS = [
    "duration",
    "face_presence",
    "max_face_ratio",
    "face_consistency",
    "face_ratio_std",
    "rms_std",
    "rms_std_relative",
    "zcr_mean",
    "speech_rate",
    "confidence_mean",
    "speech_coverage",
]

OUTPUT_KEYS = [
    "video_id",
    "shot_index",
    "start_sec",
    "end_sec",
    "transcript_text",
    *FEATURE_COLS,
    "predicted_soundbite",
    "soundbite_probability",
]


# ── Path/timecode helpers ──────────────────────────────────────────────────────

def _resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _resolve_whisper_model(spec: str) -> str:
    """Return local model path when it exists, otherwise keep model name."""
    candidate = Path(spec).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    repo_candidate = ROOT / candidate
    if repo_candidate.exists():
        return str(repo_candidate)
    return spec


def _detected_shots_path(video_path: Path, shots_dir: Path) -> Path:
    return shots_dir / (video_path.stem + "_weighted_adaptive_prediction.json")


def _get_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    raw = result.stdout.strip()
    if "/" in raw:
        num, den = raw.split("/")
        den = float(den)
        if den != 0:
            return float(num) / den
        return 25.0
    return float(raw) if raw else 25.0


def _tc_to_sec(tc: str, fps: float) -> float:
    h, m, s, ff = (int(p) for p in tc.split(":"))
    return round(h * 3600 + m * 60 + s + ff / fps, 6)


def _load_shot_timecodes(video_path: Path, shots_dir: Path) -> list[float]:
    shots_file = _detected_shots_path(video_path, shots_dir)
    if not shots_file.exists():
        return []
    with open(shots_file, encoding="utf-8") as f:
        data = json.load(f)
    video_key = video_path.name
    if video_key not in data:
        return []
    entry = data[video_key]
    if "timecodes_seconds" in entry:
        return entry["timecodes_seconds"]
    if "timecode" in entry:
        fps = _get_fps(video_path)
        return [_tc_to_sec(tc, fps) for tc in entry["timecode"]]
    return []


def _get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    raw = result.stdout.strip()
    return float(raw) if raw else 0.0


def _shot_intervals(shot_starts: list[float], video_duration: float) -> list[tuple[float, float]]:
    if not shot_starts:
        return []
    intervals: list[tuple[float, float]] = []
    for i, start_sec in enumerate(shot_starts):
        end_sec = shot_starts[i + 1] if i + 1 < len(shot_starts) else video_duration
        if end_sec > start_sec:
            intervals.append((start_sec, end_sec))
    return intervals


def _overlap_fraction(
    word_start: float,
    word_end: float,
    shot_start: float,
    shot_end: float,
) -> float:
    """Fraction of a word's duration that falls inside the shot window."""
    word_dur = word_end - word_start
    if word_dur <= 0:
        return 0.0
    overlap = max(0.0, min(word_end, shot_end) - max(word_start, shot_start))
    return overlap / word_dur


def _extract_shot_transcript_text(
    transcript_path: Path,
    start_sec: float,
    end_sec: float,
    border_threshold: float = 0.5,
) -> str:
    """Build transcript text for a shot from word-level timestamps."""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    tokens: list[str] = []
    for segment in data.get("segments", []):
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))

        if seg_end <= start_sec or seg_start >= end_sec:
            continue

        for word in segment.get("words", []):
            w_start = float(word.get("start", seg_start))
            w_end = float(word.get("end", seg_end))
            overlap = _overlap_fraction(w_start, w_end, start_sec, end_sec)
            if overlap >= border_threshold:
                token = str(word.get("word", ""))
                if token:
                    tokens.append(token)

    if tokens:
        return "".join(tokens).strip()

    # Fallback when word-level timestamps are not present in transcript.
    segment_texts: list[str] = []
    for segment in data.get("segments", []):
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", 0.0))
        if seg_end <= start_sec or seg_start >= end_sec:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            segment_texts.append(text)
    return " ".join(segment_texts).strip()


# ── Step 1: Transcription ──────────────────────────────────────────────────────

def transcribe_video(video_path: Path, whisper_model: WhisperModel) -> Path:
    out = TRANSCRIPTS_DIR / (video_path.stem + "_transcript.json")
    return transcribe_video_to_json(video_path, whisper_model, out)


# ── Step 2: Shot detection ─────────────────────────────────────────────────────

def detect_shots(video_path: Path) -> Path:
    out = SHOTS_DIR / (video_path.stem + "_weighted_adaptive_prediction.json")
    weighted_adaptive_threshold_shot_detection(
        video_path=str(video_path),
        output_dir=str(SHOTS_DIR),
        adaptive_threshold=2.5,
        min_scene_len=30,
        window_width=4,
        min_content_val=20,
    )
    return out


# ── Step 3: Feature extraction (no GT label) ──────────────────────────────────

def extract_shot_features(
    video_path: Path,
    transcript_path: Path,
    shot_idx: int,
    start_sec: float,
    end_sec: float,
) -> dict:
    feat: dict = {
        "video_id":            video_path.stem,
        "shot_index":          shot_idx,
        "start_sec":           start_sec,
        "end_sec":             end_sec,
        "transcript_text":     "",
        "duration":            end_sec - start_sec,
        "shot_position_ratio": None,   # filled after all shots for the video are collected
        "face_presence":       None,
        "max_face_ratio":      None,
        "face_consistency":    None,
        "face_ratio_std":      None,
        "rms_std":             None,
        "rms_std_relative":    None,   # filled after all shots are collected
        "zcr_mean":            None,
        "speech_rate":         None,
        "confidence_mean":     None,
        "speech_coverage":     None,
    }

    try:
        fp, max_fr, face_cons, face_std = face_features(video_path, start_sec, end_sec)
        feat["face_presence"]    = fp
        feat["max_face_ratio"]   = max_fr
        feat["face_consistency"] = face_cons
        feat["face_ratio_std"]   = face_std
    except Exception:
        print(f"    [WARN] visual failed  shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    try:
        rms_std, zcr_mean, _ = audio_features(video_path, start_sec, end_sec)
        feat["rms_std"]  = rms_std
        feat["zcr_mean"] = zcr_mean
    except Exception:
        print(f"    [WARN] audio failed   shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    if transcript_path.exists():
        try:
            tf = transcript_features(transcript_path, start_sec, end_sec)
            feat["speech_rate"]     = tf.speech_rate
            feat["confidence_mean"] = tf.confidence_mean
            feat["speech_coverage"] = tf.speech_coverage
            feat["n_words"]         = tf.n_words
            feat["transcript_text"] = _extract_shot_transcript_text(
                transcript_path, start_sec, end_sec
            )
        except Exception:
            print(f"    [WARN] text failed    shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")
    else:
        print(f"    [WARN] no transcript: {transcript_path.name}")

    return feat


# ── Step 4 & 5: Predict and write JSON ────────────────────────────────────────

def predict_and_write(rows: list[dict], model, output_json: Path) -> None:
    # Replace None with 0 so the model pipeline doesn't fail on missing features
    X = np.array(
        [[r[c] if r[c] is not None else 0.0 for c in FEATURE_COLS] for r in rows],
        dtype=np.float32,
    )
    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    results = []
    for row, pred, prob in zip(rows, preds, probas):
        entry = {k: row[k] for k in OUTPUT_KEYS if k in row}
        entry["predicted_soundbite"]   = int(pred)
        entry["soundbite_probability"] = round(float(prob), 4)
        results.append(entry)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_soundbite = int(preds.sum())
    print(f"\nDone. {len(rows)} shots — {n_soundbite} predicted soundbite, "
          f"{len(rows) - n_soundbite} non-soundbite")
    print(f"Saved → {output_json}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _add_rms_std_relative(rows: list[dict]) -> None:
    """Compute rms_std_relative in-place: each shot's rms_std divided by the
    75th-percentile rms_std of its own video. Normalises for different recording
    loudness levels so the model can compare shots fairly."""
    video_vals: dict = defaultdict(list)
    for row in rows:
        if row.get("rms_std") is not None:
            video_vals[row["video_id"]].append(row["rms_std"])
    video_p75 = {
        vid: float(np.percentile(vals, 75)) if vals else 0.0
        for vid, vals in video_vals.items()
    }
    for row in rows:
        p75 = video_p75.get(row["video_id"], 0.0)
        rms = row.get("rms_std") or 0.0
        row["rms_std_relative"] = round(rms / p75, 6) if p75 > 0 else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    video_path: Path,
    model_path: Path,
    whisper_model_spec: str,
    output_json: Path,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
) -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load ML model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Loaded model      : {model_path.name}")

    # Load Whisper model (local path or model name, e.g. "medium")
    whisper_model_ref = _resolve_whisper_model(whisper_model_spec)
    whisper_model = WhisperModel(whisper_model_ref, device=device, compute_type=compute_type)
    print(f"Loaded Whisper    : {whisper_model_ref}\n")

    print(f"\n── {video_path.name} ──")

    # Transcribe
    transcript_path = transcribe_video(video_path, whisper_model)

    # Shot detection
    detect_shots(video_path)

    shot_starts = _load_shot_timecodes(video_path, SHOTS_DIR)
    if not shot_starts:
        print("  [SKIP] no shot starts found")
        return

    video_duration = _get_video_duration(video_path)
    if video_duration <= 0:
        print("  [SKIP] could not read video duration")
        return

    shot_ranges = _shot_intervals(shot_starts, video_duration)
    if not shot_ranges:
        print("  [SKIP] no valid shot intervals")
        return

    print(f"  [INFO] {len(shot_ranges)} shots detected")

    n_shots = len(shot_ranges)
    rows: list[dict] = []
    for i, (start_sec, end_sec) in enumerate(shot_ranges):
        row = extract_shot_features(
            video_path, transcript_path,
            i, start_sec, end_sec,
        )
        row["shot_position_ratio"] = round(i / max(n_shots - 1, 1), 6)
        rows.append(row)

    if not rows:
        print("No shots extracted — nothing to predict.")
        return

    _add_rms_std_relative(rows)
    predict_and_write(rows, model, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run soundbite inference on one video.")
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEOS_DIR / "tag_reuters.com,2026_binary_LOV019113052026RP1-STREAM_700_16X9_MP4.mp4",
        help="Input video path (relative paths are resolved from repo root).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained XGBoost model (.joblib).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help='Whisper model path or model name (e.g. "medium").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
        help="Output predictions JSON path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Whisper device, e.g. "cpu" or "cuda".',
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help='Whisper compute type, e.g. "int8", "float16", or "float32".',
    )
    args = parser.parse_args()

    video_path = _resolve_repo_path(args.video)
    model_path = _resolve_repo_path(args.model)
    output_path = _resolve_repo_path(args.output)

    print(f"Video            : {video_path}")
    print(f"Model            : {model_path}")
    print(f"Whisper model    : {args.whisper_model}")
    print(f"Output           : {output_path}\n")

    run(
        video_path=video_path,
        model_path=model_path,
        whisper_model_spec=args.whisper_model,
        output_json=output_path,
        device=args.device,
        compute_type=args.compute_type,
    )
