"""
Orchestrator: iterates over all videos that have a GT entry, extracts
visual / audio / transcript features for each shot, and writes the
results to a features.json file.

Usage
-----
# Training data (default):
python -m featureExtraction.extractFeatures

# Tag the output filename with a version → data/features_v08.json:
python -m featureExtraction.extractFeatures --version v09

# Evaluation data:
python -m featureExtraction.extractFeatures --data-dir eval_data --output eval_data/features.json
python -m featureExtraction.extractFeatures --data-dir eval_data --version v08
"""

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from featureExtraction.visual import face_features
from featureExtraction.audio import audio_features
from featureExtraction.text import transcript_features
from featureExtraction.label import soundbite_label_from_gt, _gt_key_for_video

ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _transcript_path(video_path: Path, transcripts_dir: Path) -> Path:
    return transcripts_dir / (video_path.stem + "_transcript.json")


def _detected_shots_path(video_path: Path, shots_dir: Path) -> Path:
    return shots_dir / (video_path.stem + "_weighted_adaptive_prediction.json")


def _get_fps(video_path: Path) -> float:
    """Use ffprobe to read the frame rate of a video file."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    raw = result.stdout.strip()  # e.g. "25/1" or "30000/1001"
    if "/" in raw:
        num, den = raw.split("/")
        return float(num) / float(den)
    return float(raw) if raw else 25.0


def _get_video_duration(video_path: Path) -> float:
    """Use ffprobe to read the duration of a video file in seconds."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    raw = result.stdout.strip()
    return float(raw) if raw else 0.0


def _tc_to_sec(tc: str, fps: float) -> float:
    """Convert a 'HH:MM:SS:FF' timecode string to seconds."""
    h, m, s, ff = (int(p) for p in tc.split(":"))
    return round(h * 3600 + m * 60 + s + ff / fps, 6)


def _load_shot_timecodes(video_path: Path, shots_dir: Path) -> list[float]:
    """Return shot start times in seconds for a video.

    Each entry is the start of a detected shot. The last shot runs from the
    final start time to the end of the video (see ``_shot_intervals``).

    If the shots file already contains 'timecodes_seconds', those are used
    directly. Otherwise the 'timecode' strings ('HH:MM:SS:FF') are converted
    using the video's actual frame rate obtained via ffprobe.
    """
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


def _shot_intervals(
    shot_starts: list[float],
    video_duration: float,
) -> list[tuple[float, float]]:
    """Build (start_sec, end_sec) pairs from shot start times."""
    if not shot_starts:
        return []
    intervals: list[tuple[float, float]] = []
    for i, start_sec in enumerate(shot_starts):
        end_sec = shot_starts[i + 1] if i + 1 < len(shot_starts) else video_duration
        if end_sec > start_sec:
            intervals.append((start_sec, end_sec))
    return intervals


def _extract_shot(
    video_path: Path,
    transcript_path: Path,
    shot_idx: int,
    start_sec: float,
    end_sec: float,
    gt: dict,
) -> dict:
    """Extract all features for a single shot and return as a dict."""
    feat: dict = {
        "id":               None,   # filled after all rows are collected
        "video_id":         video_path.stem,
        "shot_index":       shot_idx,
        "start_sec":        start_sec,
        "end_sec":          end_sec,
        "duration":         end_sec - start_sec,
        "face_presence":    None,
        "max_face_ratio":   None,
        "face_consistency": None,
        "face_ratio_std":   None,
        "rms_std":          None,
        "rms_std_relative": None,   # filled in post-processing pass
        "zcr_mean":         None,
        "rms_mean":         None,
        "speech_rate":      None,
        "confidence_mean":  None,
        "speech_coverage":  None,
        "n_words":          None,
        "soundbite_gt":     None,
    }

    # ── Visual (native video resolution, no downscaling) ────────────────────
    try:
        fp, max_fr, face_cons, face_std = face_features(video_path, start_sec, end_sec)
        feat["face_presence"]    = fp
        feat["max_face_ratio"]   = max_fr
        feat["face_consistency"] = face_cons
        feat["face_ratio_std"]   = face_std
    except Exception:
        print(f"    [WARN] visual failed for shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    # ── Audio ───────────────────────────────────────────────────────────────
    try:
        rms_std, zcr_mean, rms_mean = audio_features(video_path, start_sec, end_sec)
        feat["rms_std"]  = rms_std
        feat["zcr_mean"] = zcr_mean
        feat["rms_mean"] = rms_mean
    except Exception:
        print(f"    [WARN] audio failed for shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")

    # ── Transcript ──────────────────────────────────────────────────────────
    if transcript_path.exists():
        try:
            tf = transcript_features(transcript_path, start_sec, end_sec)
            feat["speech_rate"]     = tf.speech_rate
            feat["confidence_mean"] = tf.confidence_mean
            feat["speech_coverage"] = tf.speech_coverage
            feat["n_words"]         = tf.n_words
        except Exception:
            print(f"    [WARN] transcript failed for shot {shot_idx}: {traceback.format_exc(limit=1).strip()}")
    else:
        print(f"    [WARN] no transcript found: {transcript_path.name}")

    # ── Label ───────────────────────────────────────────────────────────────
    feat["soundbite_gt"] = soundbite_label_from_gt(video_path, gt, start_sec, end_sec)

    return feat


# ── Main ───────────────────────────────────────────────────────────────────────

def extract_all(data_dir: Path, output_path: Path) -> list[dict]:
    videos_dir      = data_dir / "videos"
    transcripts_dir = data_dir / "transcripts"
    shots_dir       = data_dir / "detected_shots"
    gt_path         = data_dir / "shots_gt_data.json"

    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    all_rows: list[dict] = []
    videos = sorted(videos_dir.glob("*.mp4")) + sorted(videos_dir.glob("*.MP4"))

    if not videos:
        print(f"[WARN] No .mp4 files found in {videos_dir}")

    for video_path in videos:
        if _gt_key_for_video(video_path, gt) is None:
            print(f"[SKIP] no GT entry — {video_path.name}")
            continue

        shot_starts = _load_shot_timecodes(video_path, shots_dir)
        if not shot_starts:
            print(f"[SKIP] no shot starts found — {video_path.name}")
            continue

        video_duration = _get_video_duration(video_path)
        if video_duration <= 0:
            print(f"[SKIP] could not read video duration — {video_path.name}")
            continue

        shot_ranges = _shot_intervals(shot_starts, video_duration)
        if not shot_ranges:
            print(f"[SKIP] no valid shot intervals — {video_path.name}")
            continue

        transcript_path = _transcript_path(video_path, transcripts_dir)

        print(f"[INFO] {video_path.stem}  ({len(shot_ranges)} shots)")

        n_shots = len(shot_ranges)
        for i, (start_sec, end_sec) in enumerate(shot_ranges):
            row = _extract_shot(video_path, transcript_path, i, start_sec, end_sec, gt)
            # Position of this shot within the video: 0.0 = first, 1.0 = last.
            row["shot_position_ratio"] = round(i / max(n_shots - 1, 1), 6)
            all_rows.append(row)

    for global_idx, row in enumerate(all_rows):
        row["id"] = global_idx

    # ── Post-processing: video-level rms_std normalisation ──────────────────
    # rms_std_relative = rms_std / 75th-percentile(rms_std) for the same video.
    # This makes quiet-room videos (e.g. press conferences) comparable to
    # louder news packages, so the model can still detect speech energy
    # differences between shots regardless of overall recording volume.
    video_rms_stds: dict = defaultdict(list)
    for row in all_rows:
        if row.get("rms_std") is not None:
            video_rms_stds[row["video_id"]].append(row["rms_std"])

    video_rms_75p: dict = {
        vid: float(np.percentile(vals, 75)) if vals else 0.0
        for vid, vals in video_rms_stds.items()
    }

    for row in all_rows:
        p75 = video_rms_75p.get(row["video_id"], 0.0)
        rms = row.get("rms_std") or 0.0
        row["rms_std_relative"] = round(rms / p75, 6) if p75 > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    n_pos = sum(1 for r in all_rows if r["soundbite_gt"] == 1)
    n_neg = sum(1 for r in all_rows if r["soundbite_gt"] == 0)
    print(f"\nDone. {len(all_rows)} shots total — {n_pos} soundbite, {n_neg} non-soundbite")
    print(f"Saved to {output_path}")

    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract shot features from a data directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Root data directory containing videos/, transcripts/, detected_shots/, shots_gt_data.json "
             "(default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output features.json path. Overrides --version. "
             "Defaults to <data-dir>/features[_<version>].json",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version tag embedded in the default output filename, "
             "e.g. --version v08 → features_v08.json. Ignored when --output is set.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.output is not None:
        output_path = args.output
    elif args.version:
        output_path = data_dir / f"features_{args.version}.json"
    else:
        output_path = data_dir / "features.json"

    print(f"Data dir  → {data_dir}")
    print(f"Output    → {output_path}")

    extract_all(data_dir, output_path)
