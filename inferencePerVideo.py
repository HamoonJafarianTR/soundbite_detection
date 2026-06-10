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
import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv

# ── all files live in the same folder as this script ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
AWS_PROFILE       = os.getenv("AWS_PROFILE")
AWS_REGION        = os.getenv("AWS_REGION", "eu-west-1")
AWS_S3_BUCKET     = os.getenv("AWS_S3_BUCKET", "")
AWS_S3_UPLOAD_PREFIX = os.getenv("AWS_S3_UPLOAD_PREFIX", "soundbite_detection_test/audios")

# Thresholds for switching from Whisper to AWS Transcribe
WHISPER_LANG_PROB_MIN   = 0.5   # fall back if language confidence is below this
WHISPER_NO_SPEECH_MAX   = 0.6   # fall back if mean no-speech probability exceeds this
WHISPER_AVG_LOGPROB_MIN = -1.0  # drop segments whose avg_logprob is below this (likely hallucinations)


INFERENCE_DIR    = Path(__file__).parent
VIDEOS_DIR       = INFERENCE_DIR / "videos"
TRANSCRIPTS_DIR  = INFERENCE_DIR / "transcripts"
DIARIZATION_DIR  = INFERENCE_DIR / "diarization"
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
    "soundbite",
    *FEATURE_COLS,
    "predicted_soundbite",
    "soundbite_probability",
]


# ── Step 1: Transcription ──────────────────────────────────────────────────────

def _aws_raw_to_whisper_format(raw_data: dict) -> dict:
    """
    Convert AWS Transcribe raw JSON into the Whisper-compatible transcript
    format expected by text.py:
      {language, language_probability, duration, segments: [{start, end, words: [{start,end,word,probability}]}]}
    """
    results = raw_data.get("results", {})
    items = results.get("items", [])
    speaker_labels = results.get("speaker_labels", {})

    # Build (start_time, end_time) → speaker map
    speaker_map: dict[tuple[str, str], str] = {}
    for seg in speaker_labels.get("segments", []):
        spk = seg["speaker_label"]
        for item in seg.get("items", []):
            speaker_map[(item.get("start_time", ""), item.get("end_time", ""))] = spk

    segments: list[dict] = []
    current: dict | None = None

    for item in items:
        if item["type"] != "pronunciation":
            continue
        start = item.get("start_time", "")
        end   = item.get("end_time", "")
        alt   = item["alternatives"][0] if item.get("alternatives") else {}
        word_text   = alt.get("content", "")
        confidence  = float(alt.get("confidence", "0.0"))
        speaker     = speaker_map.get((start, end), "unknown")

        word_entry = {
            "start":       float(start) if start else 0.0,
            "end":         float(end)   if end   else 0.0,
            "word":        word_text,
            "probability": confidence,
        }

        if current is None or current["speaker"] != speaker:
            if current:
                segments.append(current)
            current = {
                "start":   float(start) if start else 0.0,
                "end":     float(end)   if end   else 0.0,
                "text":    word_text,
                "speaker": speaker,
                "words":   [word_entry],
            }
        else:
            current["text"] += f" {word_text}"
            current["end"]   = float(end) if end else current["end"]
            current["words"].append(word_entry)

    if current:
        segments.append(current)

    duration = max((s["end"] for s in segments), default=0.0)
    return {
        "language":             results.get("language_code", ""),
        "language_probability": 1.0,
        "duration":             duration,
        "duration_after_vad":   duration,
        "segments":             segments,
    }


def _aws_transcribe_chunk(chunk_path: Path, time_offset: float, speaker: str) -> list[dict]:
    """Transcribe a single diarized WAV chunk with AWS; return Whisper-compatible segments shifted by time_offset."""
    import boto3
    from aws_transcribe import transcribe as _aws_transcribe

    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET not set in .env — required for AWS Transcribe fallback")

    upload_key = f"{AWS_S3_UPLOAD_PREFIX.rstrip('/')}/chunks/{chunk_path.name}"
    s3_uri     = f"s3://{AWS_S3_BUCKET}/{upload_key}"

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    s3 = session.client("s3")
    print(f"      [AWS] uploading chunk {chunk_path.name}")
    s3.upload_file(str(chunk_path), AWS_S3_BUCKET, upload_key)

    raw_dir = TRANSCRIPTS_DIR / "chunk_aws_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / (chunk_path.stem + "_aws_raw.json")

    _aws_transcribe(
        s3_uri=s3_uri,
        output_bucket=AWS_S3_BUCKET,
        profile=AWS_PROFILE,
        region=AWS_REGION,
        save_raw=raw_path,
    )

    raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
    items    = raw_data.get("results", {}).get("items", [])

    words: list[dict] = []
    for item in items:
        if item["type"] != "pronunciation":
            continue
        start = item.get("start_time", "")
        end   = item.get("end_time", "")
        alt   = item["alternatives"][0] if item.get("alternatives") else {}
        words.append({
            "start":       (float(start) if start else 0.0) + time_offset,
            "end":         (float(end)   if end   else 0.0) + time_offset,
            "word":        alt.get("content", ""),
            "probability": float(alt.get("confidence", "0.0")),
        })

    if not words:
        return []

    return [{
        "start":               words[0]["start"],
        "end":                 words[-1]["end"],
        "text":                " ".join(w["word"] for w in words),
        "speaker":             speaker,
        "language":            "",
        "language_probability": 1.0,
        "no_speech_prob":      0.0,
        "service":             "AWS Transcribe",
        "words":               words,
    }]


def transcribe_video(video_path: Path, whisper_model) -> Path:
    from dataclasses import asdict
    from collections import Counter
    from whisper_transcibe import (
        DEFAULT_DIARIZATION_CONFIG,
        extract_audio_mp3,
        _diarize_segments,
        _extract_audio_wav,
        _shift_segment_timestamps,
    )

    out = TRANSCRIPTS_DIR / (video_path.stem + "_transcript.json")
    if out.exists():
        print(f"  [SKIP] transcript already exists: {out.name}")
        return out

    audio_path = extract_audio_mp3(video_path, DIARIZATION_DIR / (video_path.stem + ".mp3"))

    # Step 1: Speaker diarization (cached — skip if already done)
    diar_cache = DIARIZATION_DIR / (audio_path.stem + "_diarization.json")
    if diar_cache.exists():
        print(f"  [SKIP] diarization already exists: {diar_cache.name}")
        diar_segments = json.loads(diar_cache.read_text(encoding="utf-8"))
    else:
        print(f"  [INFO] Running speaker diarization...")
        diar_segments = _diarize_segments(str(audio_path), str(DEFAULT_DIARIZATION_CONFIG))
        if not diar_segments:
            raise RuntimeError("Diarization produced no segments")
        diar_cache.write_text(json.dumps(diar_segments, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK]   diarization saved: {diar_cache.name}")

    work_dir = DIARIZATION_DIR / "speaker_chunks" / audio_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    merged_segments:        list[dict]  = []
    languages:              list[str]   = []
    language_probabilities: list[float] = []
    aws_fallback_count = 0

    # Step 2: Per-chunk Whisper transcription; fall back to AWS Transcribe if thresholds not met
    for idx, seg in enumerate(diar_segments):
        start_sec = seg["start"]
        end_sec   = seg["end"]
        speaker   = seg["speaker"]
        if end_sec <= start_sec:
            continue
        if (end_sec - start_sec) < 1.0:
            print(f"    [SKIP] diarization segment {idx} ({speaker}, {start_sec:.1f}s–{end_sec:.1f}s) too short (<1s) — ignoring")
            continue

        chunk_path = work_dir / f"chunk_{idx:05d}_{speaker}.wav"
        _extract_audio_wav(str(audio_path), start_sec, end_sec, chunk_path)

        whisper_segs, info = whisper_model.transcribe(
            str(chunk_path),
            condition_on_previous_text=False,
            beam_size=5,
            word_timestamps=True,
        )
        seg_list = list(whisper_segs)

        chunk_lang      = info.language                  if hasattr(info, "language")             else ""
        chunk_lang_prob = float(info.language_probability) if hasattr(info, "language_probability") else 0.0
        chunk_no_speech = (
            sum(float(getattr(s, "no_speech_prob", 0.0)) for s in seg_list) / len(seg_list)
            if seg_list else 1.0
        )
        chunk_avg_logprob = (
            sum(float(getattr(s, "avg_logprob", 0.0)) for s in seg_list) / len(seg_list)
            if seg_list else -float("inf")
        )

        if chunk_lang_prob < WHISPER_LANG_PROB_MIN or chunk_no_speech > WHISPER_NO_SPEECH_MAX or chunk_avg_logprob < WHISPER_AVG_LOGPROB_MIN:
            print(
                f"    [WARN] chunk {idx} ({speaker}, {start_sec:.1f}s–{end_sec:.1f}s) "
                f"lang_prob={chunk_lang_prob:.2f}, no_speech={chunk_no_speech:.2f}, "
                f"avg_logprob={chunk_avg_logprob:.2f} → AWS Transcribe"
            )
            aws_fallback_count += 1
            merged_segments.extend(_aws_transcribe_chunk(chunk_path, start_sec, speaker))
        else:
            for s in seg_list:
                shifted = _shift_segment_timestamps(asdict(s), start_sec)
                shifted["speaker"]              = speaker
                shifted["language"]             = chunk_lang
                shifted["language_probability"] = chunk_lang_prob
                shifted["service"]              = "Whisper"
                merged_segments.append(shifted)
            if chunk_lang:
                languages.append(chunk_lang)
            if chunk_lang_prob:
                language_probabilities.append(chunk_lang_prob)

    merged_segments.sort(key=lambda x: float(x.get("start", 0.0)))

    language             = Counter(languages).most_common(1)[0][0] if languages else ""
    language_probability = (sum(language_probabilities) / len(language_probabilities)) if language_probabilities else 0.0
    duration             = max((float(s.get("end", 0.0)) for s in merged_segments), default=0.0)

    # Fill in missing language fields using the overall document language as fallback
    if language:
        for seg in merged_segments:
            if not seg.get("language"):
                seg["language"] = language

    if aws_fallback_count:
        print(f"  [INFO] {aws_fallback_count}/{len(diar_segments)} chunk(s) re-transcribed via AWS Transcribe")

    transcript_data = {
        "language":             language,
        "language_probability": language_probability,
        "duration":             duration,
        "duration_after_vad":   duration,
        "segments":             merged_segments,
    }
    out.write_text(json.dumps(transcript_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK]   transcript saved: {out.name}")
    return out


# ── Step 2: Shot detection ─────────────────────────────────────────────────────

def detect_shots(video_path: Path) -> Path:
    # Import lazily to avoid loading cv2 before xgboost model deserialization.
    from shotDetection import weighted_adaptive_threshold_shot_detection

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
    # Import lazily to avoid cv2/xgboost runtime conflicts during startup.
    from visual import face_area_ratio
    from audio import audio_features
    from text import transcript_features, transcript_text_for_window

    feat: dict = {
        "video_id":        video_path.stem,
        "shot_index":      shot_idx,
        "start_sec":       start_sec,
        "end_sec":         end_sec,
        "soundbite":       None,
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
            feat["soundbite"]      = transcript_text_for_window(
                transcript_path, start_sec, end_sec
            )
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
    soundbite_shot_ids = [
        row["shot_index"] for row, pred in zip(rows, preds) if int(pred) == 1
    ]
    print(
        f"\nDone. soundbite shots {soundbite_shot_ids} — {n_soundbite} predicted soundbite, "
        f"{len(rows) - n_soundbite} non-soundbite"
    )
    print(f"Saved → {output_json}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(video_path: Path, model_path: Path, whisper_model_path: Path, output_json: Path) -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load ML model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Loaded model      : {model_path.name}")

    # Load Whisper model
    from faster_whisper import WhisperModel

    #if not whisper_model_path.exists():
    #    raise FileNotFoundError(f"Whisper model not found: {whisper_model_path}")
    #whisper_model = WhisperModel(str(whisper_model_path), device="cpu", compute_type="int8")
    #print(f"Loaded Whisper    : {whisper_model_path.name}\n")
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
    print(f"Loaded Whisper    : medium (faster-whisper)\n")

    print(f"\n── {video_path.name} ──")

    # Transcribe
    transcript_path = transcribe_video(video_path, whisper_model)

    # Shot detection
    detect_shots(video_path)

    # Load shot timecodes
    from extractFeatures import _load_shot_timecodes
    timecodes = _load_shot_timecodes(video_path, SHOTS_DIR)
    if len(timecodes) < 2:
        print("  [SKIP] not enough shot boundaries")
        return

    n_shots = len(timecodes) - 1
    print(f"  [INFO] {n_shots} shots detected")

    rows: list[dict] = []
    for i in range(n_shots):
        row = extract_shot_features(
            video_path, transcript_path,
            i, timecodes[i], timecodes[i + 1],
        )
        rows.append(row)

    if not rows:
        print("No shots extracted — nothing to predict.")
        return

    predict_and_write(rows, model, output_json)


if __name__ == "__main__":
    # ── Configure these paths before running ──────────────────────────────────

    MODEL_PATH          = DEFAULT_MODEL
    WHISPER_MODEL_PATH  = DEFAULT_WHISPER_MODEL

    # ─────────────────────────────────────────────────────────────────────────

    video_paths = sorted(VIDEOS_DIR.glob("*.mp4")) + sorted(VIDEOS_DIR.glob("*.MP4"))
    if not video_paths:
        print(f"No video files found in {VIDEOS_DIR}")
    else:
        predictions_dir = INFERENCE_DIR / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for video_path in video_paths:
            if video_path.stem == "000420260212RQ1-367503122025RU1":
                output_json_path = predictions_dir / f"{video_path.stem}_predictions.json"
                run(video_path, MODEL_PATH, WHISPER_MODEL_PATH, output_json_path)
                break
            
