"""
End-to-end inference pipeline for a single video.

  1. Transcribes audio with Whisper  → transcripts/
  2. Detects shots                   → detected_shots/
  3. Extracts features per shot      (no GT label)
  4. Runs the trained model
  5. Writes predictions.json

Configure VIDEO_PATH, MODEL_PATH, WHISPER_MODEL_PATH, and OUTPUT_JSON
in the __main__ block at the bottom of this file, then run:

    python inferencePerVideo.py
"""

import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
from faster_whisper import WhisperModel

# ── all files live in the same folder as this script ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from whisper_transcibe import (
    extract_audio_mp3,
    transcribe_audio,
    dump_transcript_json,
)
from shotDetection import weighted_adaptive_threshold_shot_detection
from extractFeatures import (
    _transcript_path,
    _detected_shots_path,
    _load_shot_timecodes,
    _shot_intervals,
)
from visual import face_area_ratio
from audio import audio_features
from text import transcript_features

INFERENCE_DIR    = Path(__file__).parent
VIDEOS_DIR       = INFERENCE_DIR / "videos"
TRANSCRIPTS_DIR  = INFERENCE_DIR / "transcripts"
SHOTS_DIR        = INFERENCE_DIR / "detected_shots"
OUTPUT_JSON      = INFERENCE_DIR / "predictions.json"

DEFAULT_MODEL         = ROOT / "model_xgboost.joblib"
DEFAULT_WHISPER_MODEL = ROOT / "whisper_models" / "faster-whisper-medium"

FEATURE_COLS = [
    "face_area_ratio",
    "rms_mean",
    "rms_std",
    "word_count",
    "speech_rate",
    "confidence_mean",
    "speech_coverage",
]

OUTPUT_KEYS = [
    "video_id",
    "shot_index",
    "start_sec",
    "end_sec",
    *FEATURE_COLS,
    "predicted_soundbite",
    "soundbite_probability",
]


# ── Step 1: Transcription ──────────────────────────────────────────────────────

def transcribe_video(video_path: Path, whisper_model: WhisperModel) -> Path:
    out = TRANSCRIPTS_DIR / (video_path.stem + "_transcript.json")
    if out.exists():
        print(f"  [SKIP] transcript already exists: {out.name}")
        return out
    audio_path = extract_audio_mp3(video_path)
    segments, info = transcribe_audio(str(audio_path), whisper_model)
    dump_transcript_json(segments, info, out)
    print(f"  [OK]   transcript saved: {out.name}")
    return out


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
        "video_id":        video_path.stem,
        "shot_index":      shot_idx,
        "start_sec":       start_sec,
        "end_sec":         end_sec,
        "face_area_ratio": None,
        "rms_mean":        None,
        "rms_std":         None,
        "word_count":      None,
        "speech_rate":     None,
        "confidence_mean": None,
        "speech_coverage": None,
    }

    try:
        feat["face_area_ratio"] = face_area_ratio(video_path, start_sec, end_sec)
    except Exception:
        print(f"    [WARN] visual failed  shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    try:
        rms_mean, rms_std = audio_features(video_path, start_sec, end_sec)
        feat["rms_mean"] = rms_mean
        feat["rms_std"]  = rms_std
    except Exception:
        print(f"    [WARN] audio failed   shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    if transcript_path.exists():
        try:
            tf = transcript_features(transcript_path, start_sec, end_sec)
            feat["word_count"]      = tf.word_count
            feat["speech_rate"]     = tf.speech_rate
            feat["confidence_mean"] = tf.confidence_mean
            feat["speech_coverage"] = tf.speech_coverage
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


# ── Main ───────────────────────────────────────────────────────────────────────

def run(video_path: Path, model_path: Path, whisper_model_path: Path, output_json: Path) -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load ML model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Loaded model      : {model_path.name}")

    # Load Whisper model
    if not whisper_model_path.exists():
        raise FileNotFoundError(f"Whisper model not found: {whisper_model_path}")
    whisper_model = WhisperModel(str(whisper_model_path), device="cpu", compute_type="int8")
    print(f"Loaded Whisper    : {whisper_model_path.name}\n")

    print(f"\n── {video_path.name} ──")

    # Transcribe
    transcript_path = transcribe_video(video_path, whisper_model)

    # Shot detection
    detect_shots(video_path)

    # Load shot start times (each timecode is a shot start; last shot ends at video duration)
    shot_starts = _load_shot_timecodes(video_path, SHOTS_DIR)
    if not shot_starts:
        print("  [SKIP] no shot start times found")
        return

    intervals = _shot_intervals(video_path, shot_starts)
    print(f"  [INFO] {len(intervals)} shots detected")

    rows: list[dict] = []
    for i, (start_sec, end_sec) in enumerate(intervals):
        row = extract_shot_features(
            video_path, transcript_path,
            i, start_sec, end_sec,
        )
        rows.append(row)

    if not rows:
        print("No shots extracted — nothing to predict.")
        return

    predict_and_write(rows, model, output_json)


if __name__ == "__main__":
    # ── Configure these paths before running ──────────────────────────────────

    VIDEO_PATH          = VIDEOS_DIR / "380308012026RU1.mp4"
    MODEL_PATH          = DEFAULT_MODEL
    WHISPER_MODEL_PATH  = DEFAULT_WHISPER_MODEL
    OUTPUT_JSON_PATH    = OUTPUT_JSON

    # ─────────────────────────────────────────────────────────────────────────

    run(VIDEO_PATH, MODEL_PATH, WHISPER_MODEL_PATH, OUTPUT_JSON_PATH)
