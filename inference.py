"""Inference pipeline – predict soundbites in a single video.

Workflow (all driven from a single S3 video URL):
  1. Download the video via download_videos.py  →  videos/<name>
  2. Transcribe the video via transcribe.py      →  transcripts/<name>_transcript_raw.json
  3. Run shot detection via shotDetection_AdaptiveDetector.py
       →  detected_shots/<name>_weighted_adaptive_prediction.json
  4. Extract features and predict soundbites.

Usage:
    python inference.py --video s3://my-bucket/videos/NEW_VIDEO.MP4
    python inference.py --video s3://my-bucket/videos/NEW_VIDEO.MP4 --output-dir /custom/dir
"""

import argparse
import json
import logging
import os
import warnings

import joblib
import numpy as np

import config as cfg
from download_videos import download_video, parse_s3_uri
from feature_extraction import extract_features_for_video
from shotDetection_AdaptiveDetector import weighted_adaptive_threshold_shot_detection
from transcribe import transcribe_video

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def load_model():
    model = joblib.load(os.path.join(cfg.MODELS_DIR, "soundbite_classifier.joblib"))
    scaler = joblib.load(os.path.join(cfg.MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(cfg.MODELS_DIR, "model_meta.json"), "r") as f:
        meta = json.load(f)
    return model, scaler, meta


def merge_adjacent_soundbites(segments):
    """Merge consecutive predicted-soundbite segments into contiguous ranges."""
    ranges = []
    current = None
    for seg in segments:
        if seg["prediction"] == 1:
            if current is None:
                current = {"start": seg["start"], "end": seg["end"],
                           "confidences": [seg["confidence"]]}
            else:
                current["end"] = seg["end"]
                current["confidences"].append(seg["confidence"])
        else:
            if current is not None:
                ranges.append(current)
                current = None
    if current is not None:
        ranges.append(current)

    merged = []
    for r in ranges:
        merged.append({
            "soundbite_start": round(float(r["start"]), 3),
            "soundbite_end": round(float(r["end"]), 3),
            "confidence": float(np.mean(r["confidences"])),
        })
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict soundbites in a video. "
            "Downloads the video from S3, transcribes it, runs shot detection, then predicts."
        )
    )
    parser.add_argument(
        "--video", required=True,
        help="S3 URL of the video file (e.g. s3://bucket/videos/clip.mp4)",
    )
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save the predictions JSON (default: predictions/)")
    parser.add_argument("--no-save-frames", action="store_true",
                        help="Do not write segment JPEGs to the frames/ folder")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        parse_s3_uri(args.video)
    except ValueError as exc:
        parser.error(str(exc))

    # ── Step 1: download video ────────────────────────────────────────────────
    logger.info("Step 1/4 – downloading video from S3")
    video_path = download_video(args.video)
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]

    # ── Step 2: transcribe ───────────────────────────────────────────────────
    logger.info("Step 2/4 – transcribing video")
    transcript_path = transcribe_video(args.video)

    # ── Step 3: shot detection ────────────────────────────────────────────────
    shots_path = os.path.join(
        "detected_shots", f"{video_stem}_weighted_adaptive_prediction.json"
    )
    if os.path.exists(shots_path):
        logger.info("Step 3/4 – shot detection already exists, skipping: %s", shots_path)
    else:
        logger.info("Step 3/4 – running shot detection")
        os.makedirs("detected_shots", exist_ok=True)
        weighted_adaptive_threshold_shot_detection(video_path)

    model, scaler, meta = load_model()
    feature_cols = meta["feature_columns"]
    decision_threshold = float(meta.get("decision_threshold", 0.5))
    logger.info("Using decision_threshold=%.3f for predictions", decision_threshold)

    df = extract_features_for_video(
        video_name,
        video_path=video_path,
        transcript_path=transcript_path,
        shots_path=shots_path,
        compute_labels=False,
        save_visual_frames=False,
    )

    X = df[feature_cols].values
    X = scaler.transform(X)
    X = np.nan_to_num(X, nan=0.0)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        df["confidence"] = proba
        df["predicted"] = (proba >= decision_threshold).astype(int)
    else:
        df["predicted"] = model.predict(X)
        df["confidence"] = np.nan

    # Build per-segment list
    segment_list = []
    for _, row in df.iterrows():
        segment_list.append({
            "index": int(row["segment_index"]),
            "start": float(row["segment_start"]),
            "end": float(row["segment_end"]),
            "prediction": int(row["predicted"]),
            "confidence": round(float(row["confidence"]), 4),
        })

    merged = merge_adjacent_soundbites(segment_list)

    output = {
        "video": video_name,
        "soundbites": merged,
        "segments": segment_list,
    }

    # Determine output path — always inside predictions/ (or a custom dir)
    pred_dir = args.output_dir if args.output_dir else os.path.join(cfg.PROJECT_ROOT, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    base = video_stem
    out_path = os.path.join(pred_dir, f"{base}_soundbites.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Human-readable summary
    print(f"\n{'=' * 60}")
    print(f"SOUNDBITE PREDICTIONS – {video_name}")
    print(f"{'=' * 60}")
    print(f"Total segments : {len(segment_list)}")
    print(f"Soundbite segs : {sum(1 for s in segment_list if s['prediction'] == 1)}")
    print(f"Merged ranges  : {len(merged)}")
    print()
    for i, sb in enumerate(merged, 1):
        print(
            f"  [{i}] {sb['soundbite_start']:>9.3f}s  →  "
            f"{sb['soundbite_end']:>9.3f}s  "
            f"(confidence {sb['confidence']:.2f})"
        )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
