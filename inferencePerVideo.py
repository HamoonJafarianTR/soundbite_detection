"""
End-to-end inference pipeline for a single video.

  1. Extracts features per shot      (no GT label)
  2. Runs the trained model
  3. Writes predictions.json

Run with explicit paths:

    python inferencePerVideo.py --video videos/my_video.mp4 --transcript transcripts/my_video_transcript.json --shots detected_shots/my_video_shots.json --model models/model_xgboost_v10.joblib --output predictions.json
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

# ── project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractFeatures import visual as visual_module
from extractFeatures.visual import face_features
from extractFeatures.audio import audio_features
from extractFeatures.text import transcript_features

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


def _load_shot_timecodes(video_path: Path, shots_file: Path) -> list[float]:
    if not shots_file.exists():
        return []
    with open(shots_file, encoding="utf-8") as f:
        data = json.load(f)
    video_key = video_path.name
    if video_key not in data:
        # Check if the structure does not have the video_key at the root
        if "timecodes_seconds" in data:
            entry = data
        elif "timecode" in data:
            entry = data
        else:
            return []
    else:
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


# ── Step 1: Feature extraction (no GT label) ──────────────────────────────────

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


# ── Step 2 & 3: Predict and write JSON ────────────────────────────────────────

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
    transcript_path: Path,
    shots_path: Path,
    model_path: Path,
    output_json: Path,
) -> None:

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if not shots_path.exists():
        raise FileNotFoundError(f"Shots JSON not found: {shots_path}")

    # Load ML model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Loaded model      : {model_path.name}")

    print(f"\n── {video_path.name} ──")

    shot_starts = _load_shot_timecodes(video_path, shots_path)
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

    process_start = time.perf_counter()

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

    elapsed = time.perf_counter() - process_start
    per_sec = elapsed / video_duration if video_duration > 0 else 0.0
    print(f"  [TIME] {elapsed:.2f}s processing / {video_duration:.2f}s video "
          f"= {per_sec:.3f}s per video second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run soundbite inference on one video.")
    parser.add_argument(
        "--video",
        type=str,
        help="Input video path (relative paths are resolved from repo root).",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        help="Input transcript JSON path.",
    )
    parser.add_argument(
        "--shots",
        type=str,
        help="Input detected shots JSON path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained XGBoost model (.joblib).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output predictions JSON path.",
    )
    
    args = parser.parse_args()

    video_val = args.video
    if not video_val:
        video_val = input("Please provide the path to the video file: ").strip()
    
    transcript_val = args.transcript
    if not transcript_val:
        transcript_val = input("Please provide the path to the transcript JSON: ").strip()
        
    shots_val = args.shots
    if not shots_val:
        shots_val = input("Please provide the path to the detected shots JSON: ").strip()

    model_val = args.model
    if not model_val:
        model_val = input("Please provide the path to the trained XGBoost model (.joblib): ").strip()
        
    output_val = args.output
    if not output_val:
        output_val = input("Please provide the path to save the output predictions JSON: ").strip()

    if not video_val or not transcript_val or not shots_val or not model_val or not output_val:
        print("Error: video, transcript, shots, model, and output paths are all required.")
        sys.exit(1)

    video_path = _resolve_repo_path(Path(video_val))
    transcript_path = _resolve_repo_path(Path(transcript_val))
    shots_path = _resolve_repo_path(Path(shots_val))
    model_path = _resolve_repo_path(Path(model_val))
    output_path = _resolve_repo_path(Path(output_val))

    print(f"Video            : {video_path}")
    print(f"Transcript       : {transcript_path}")
    print(f"Shots            : {shots_path}")
    print(f"Model            : {model_path}")
    print(f"Output           : {output_path}\n")

    run(
        video_path=video_path,
        transcript_path=transcript_path,
        shots_path=shots_path,
        model_path=model_path,
        output_json=output_path,
    )