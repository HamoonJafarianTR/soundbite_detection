import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TranscriptFeatures:
    speech_rate: float
    confidence_mean: float
    speech_coverage: float
    n_words: int


def _overlap_fraction(
    word_start: float,
    word_end: float,
    shot_start: float,
    shot_end: float,
) -> float:
    """
    Fraction of a word's duration that falls inside the shot window.
    Returns a value in [0.0, 1.0].
    """
    word_dur = word_end - word_start
    if word_dur <= 0:
        return 0.0
    overlap = max(0.0, min(word_end, shot_end) - max(word_start, shot_start))
    return overlap / word_dur


def transcript_features(
    transcript_path: str | Path,
    start_sec: float,
    end_sec: float,
    border_threshold: float = 0.5,
) -> TranscriptFeatures:
    """
    Compute transcript-based features for a single shot.

    Words are matched to the shot by overlap fraction:
      - fully inside  (overlap == 1.0) → always included
      - on the border (0 < overlap < 1) → included if overlap >= border_threshold
      - fully outside (overlap == 0.0) → excluded

    For border words, the word duration counted toward speech_coverage is
    scaled by the overlap fraction, keeping the coverage metric accurate.

    Args:
        transcript_path:  Path to the transcript JSON produced by whisper_transcibe.py.
        start_sec:        Shot start time in seconds.
        end_sec:          Shot end time in seconds.
        border_threshold: Minimum overlap fraction (0–1) for a boundary word
                          to be counted. Default 0.5 means the word must be
                          more than half inside the shot.

    Returns:
        TranscriptFeatures with:
            speech_rate     – words per second (0 if shot duration <= 0)
            confidence_mean – mean word probability (0 if no words found)
            speech_coverage – overlapping word duration / shot duration
            n_words         – absolute word count in the shot window
    """
    shot_dur = end_sec - start_sec
    if shot_dur <= 0:
        return TranscriptFeatures(0.0, 0.0, 0.0)

    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    word_count = 0
    confidence_sum = 0.0
    covered_duration = 0.0

    for segment in data.get("segments", []):
        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", 0.0)

        # skip segments that don't touch the shot window at all
        if seg_end <= start_sec or seg_start >= end_sec:
            continue

        for word in segment.get("words", []):
            w_start = word.get("start", seg_start)
            w_end = word.get("end", seg_end)
            prob = word.get("probability", 0.0)

            overlap = _overlap_fraction(w_start, w_end, start_sec, end_sec)

            if overlap == 0.0:
                continue
            if overlap < border_threshold:
                continue

            word_count += 1
            confidence_sum += prob
            covered_duration += (w_end - w_start) * overlap

    speech_rate = word_count / shot_dur
    confidence_mean = confidence_sum / word_count if word_count > 0 else 0.0
    speech_coverage = min(covered_duration / shot_dur, 1.0)

    return TranscriptFeatures(
        speech_rate=speech_rate,
        confidence_mean=confidence_mean,
        speech_coverage=speech_coverage,
        n_words=word_count,
    )
