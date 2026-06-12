"""
End-to-end inference pipeline for a single video.

  1. Transcribes audio with Whisper  → transcripts/
  2. Detects shots                   → detected_shots/
  3. Extracts features per shot
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

# ── project root (same folder as this script) ─────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from prepareData.whisper_transcibe import transcribe_video_to_json
from prepareData.shotDetection import weighted_adaptive_threshold_shot_detection
from extractFeatures.extractFeatures import (
    _transcript_path,
    _detected_shots_path,
    _load_shot_timecodes,
    _get_video_duration,
    _shot_intervals,
)
from extractFeatures.visual import face_presence
from extractFeatures.audio import audio_features
from extractFeatures.text import transcript_features

VIDEOS_DIR       = ROOT / "videos"
TRANSCRIPTS_DIR  = ROOT / "transcripts"
SHOTS_DIR        = ROOT / "detected_shots"
OUTPUT_JSON      = ROOT / "predictions.json"

DEFAULT_MODEL         = ROOT / "models" / "model_xgboost_v08.joblib"
DEFAULT_WHISPER_MODEL = ROOT / "models" / "whisper_models" / "faster-whisper-medium"

FEATURE_COLS = [
    "face_presence",
    "rms_mean",
    "rms_std",
    "speech_rate",
    "confidence_mean",
    "speech_coverage",
    "duration",
]


def feature_value(row: dict, col: str, *, default: float = 0.0) -> float:
    if col == "duration":
        return float(row["end_sec"] - row["start_sec"])
    val = row.get(col)
    return float(val) if val is not None else default

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


# ── Step 3: Feature extraction ──────────────────────────────────────────────────

def extract_shot_features(
    video_path: Path,
    transcript_path: Path,
    shot_idx: int,
    start_sec: float,
    end_sec: float,
) -> dict:
    feat: dict = {
        "video_id":         video_path.stem,
        "shot_index":       shot_idx,
        "start_sec":        start_sec,
        "end_sec":          end_sec,
        "duration":         end_sec - start_sec,
        "face_presence":    None,
        "rms_mean":         None,
        "rms_std":          None,
        "speech_rate":      None,
        "confidence_mean":  None,
        "speech_coverage":  None,
    }

    try:
        feat["face_presence"] = face_presence(video_path, start_sec, end_sec)
    except Exception:
        print(f"    [WARN] visual failed  shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    try:
        rms_mean, rms_std = audio_features(video_path, start_sec, end_sec)
        feat["rms_mean"] = rms_mean
        feat["rms_std"]  = rms_std
    except Exception:
        print(f"    [WARN] audio (rms) failed   shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    if transcript_path.exists():
        try:
            tf = transcript_features(transcript_path, start_sec, end_sec)
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
        [[feature_value(r, c) for c in FEATURE_COLS] for r in rows],
        dtype=np.float32,
    )
    preds  = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    results = []
    for row, pred, prob in zip(rows, preds, probas):
        entry = {k: row[k] for k in ("video_id", "shot_index", "start_sec", "end_sec")}
        for col in FEATURE_COLS:
            entry[col] = feature_value(row, col)
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

    rows: list[dict] = []
    for i, (start_sec, end_sec) in enumerate(shot_ranges):
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
