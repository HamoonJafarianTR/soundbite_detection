from faster_whisper import WhisperModel
from collections import Counter
import subprocess
import tempfile
from pathlib import Path
import json
from dataclasses import asdict
import shutil
import os
from multiprocessing import get_context
from queue import Empty


DEFAULT_DIARIZATION_CONFIG = Path(__file__).parent / "config.yaml"
# DEFAULT_DIARIZATION_TIMEOUT_SEC = float(os.getenv("DIARIZATION_TIMEOUT_SEC", "90"))


def _diarize_worker(audio_file: str, config_path: str, result_queue) -> None:
    """Run pyannote diarization in a child process so parent can timeout safely."""
    from pyannote.audio import Pipeline
    import time
    import traceback

    try:
        start_load = time.time()
        pipeline = Pipeline.from_pretrained(config_path)
        load_sec = time.time() - start_load

        start_diar = time.time()
        diarization = pipeline(audio_file)
        diar_sec = time.time() - start_diar

        segments: list[dict] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )

        result_queue.put(
            {
                "ok": True,
                "segments": segments,
                "load_sec": load_sec,
                "diar_sec": diar_sec,
            }
        )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc(limit=2),
            }
        )

def extract_audio_mp3(video_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    Extract audio from a video file and save it as MP3.
    Args:
        video_path: Path to input video (mp4, mov, mkv, etc.)
        output_path: Optional output .mp3 path. If omitted, writes to a temp file.
    Returns:
        Path to the extracted MP3 file (for use with model.transcribe()).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Starting audio separation from: {video_path}")

    if output_path is None:
        output_path = Path(tempfile.mkstemp(suffix=".mp3")[1])
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",                  # overwrite output
        "-i", str(video_path), # input video
        "-vn",                 # no video
        "-acodec", "libmp3lame",
        "-q:a", "2",           # good quality (~190 kbps VBR)
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"Finished audio separation. Saved to: {output_path}")
    return output_path

def _shift_segment_timestamps(segment_dict: dict, offset_sec: float) -> dict:
    shifted = dict(segment_dict)
    shifted["start"] = float(shifted.get("start", 0.0)) + offset_sec
    shifted["end"] = float(shifted.get("end", 0.0)) + offset_sec
    if "words" in shifted and isinstance(shifted["words"], list):
        shifted_words = []
        for w in shifted["words"]:
            ww = dict(w)
            ww["start"] = float(ww.get("start", 0.0)) + offset_sec
            ww["end"] = float(ww.get("end", 0.0)) + offset_sec
            shifted_words.append(ww)
        shifted["words"] = shifted_words
    return shifted


def _extract_audio_wav(
    audio_file: str | Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> None:
    duration = max(0.0, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(audio_file),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg chunk extraction failed:\n{result.stderr}")


def _diarize_segments(audio_file: str | Path, config_path: str | Path) -> list[dict]:
    print(f"  [diarization] Loading pre-trained model from: {config_path}")
    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_diarize_worker,
        args=(str(audio_file), str(config_path), result_queue),
        daemon=True,
    )

    proc.start()
    proc.join()

    try:
        payload = result_queue.get_nowait()
    except Empty as exc:
        raise RuntimeError("Diarization worker exited without returning results") from exc

    if not payload.get("ok"):
        raise RuntimeError(f"Diarization worker failed:\n{payload.get('error', '')}")

    print(f"  [diarization] Model loaded in {payload['load_sec']:.2f}s")
    print(f"  [diarization] Diarization completed in {payload['diar_sec']:.2f}s")
    return payload["segments"]


def _transcribe_with_diarization(
    audio_file: str | Path,
    loaded_model,
    config_path: str | Path,
):
    diar_segments = _diarize_segments(audio_file, config_path)
    if not diar_segments:
        raise RuntimeError("Diarization produced no segments")

    audio_file = Path(audio_file)
    work_dir = audio_file.parent / "speaker_chunks" / audio_file.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [diarization] Saving speaker chunks to: {work_dir}")
    merged_segments: list[dict] = []
    languages: list[str] = []
    language_probabilities: list[float] = []

    try:
        for idx, seg in enumerate(diar_segments):
            start_sec = seg["start"]
            end_sec = seg["end"]
            speaker = seg["speaker"]
            if end_sec <= start_sec:
                continue

            chunk_path = work_dir / f"chunk_{idx:05d}_{speaker}.wav"
            _extract_audio_wav(audio_file, start_sec, end_sec, chunk_path)

            segments, info = loaded_model.transcribe(
                str(chunk_path),
                condition_on_previous_text=False,
                beam_size=5,
                word_timestamps=True,
            )
            seg_list = list(segments)

            chunk_language = info.language if hasattr(info, "language") else ""
            chunk_language_prob = float(info.language_probability) if hasattr(info, "language_probability") else 0.0

            if chunk_language:
                languages.append(chunk_language)
            if chunk_language_prob:
                language_probabilities.append(chunk_language_prob)

            for s in seg_list:
                segment_dict = asdict(s)
                shifted = _shift_segment_timestamps(segment_dict, start_sec)
                shifted["speaker"] = speaker
                shifted["language"] = chunk_language
                shifted["language_probability"] = chunk_language_prob
                merged_segments.append(shifted)

        merged_segments.sort(key=lambda x: float(x.get("start", 0.0)))

        language = Counter(languages).most_common(1)[0][0] if languages else ""
        language_probability = (
            float(sum(language_probabilities) / len(language_probabilities))
            if language_probabilities
            else 0.0
        )
        duration = (
            max((float(s.get("end", 0.0)) for s in merged_segments), default=0.0)
            if merged_segments
            else 0.0
        )

        info_dict = {
            "language": language,
            "language_probability": language_probability,
            "duration": duration,
            "duration_after_vad": duration,
        }
        return merged_segments, info_dict
    except Exception:
        raise


def transcribe_audio(audio_file, loaded_model, diarization_config: str | Path | None = None):
    config_path = Path(diarization_config) if diarization_config else DEFAULT_DIARIZATION_CONFIG
    print(f"Using diarization config: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Diarization config not found: {config_path}")

    print(f"Running speaker diarization using: {config_path}")
    segments, info = _transcribe_with_diarization(audio_file, loaded_model, config_path)
    print(
        "Detected language "
        f"{info.get('language', '')} with probability {info.get('language_probability', 0.0):.4f}"
    )
    return segments, info

def transcription_to_dict(segments, info):
    if isinstance(info, dict):
        serialized_segments = [s if isinstance(s, dict) else asdict(s) for s in segments]
        return {
            "language": info.get("language", ""),
            "language_probability": info.get("language_probability", 0.0),
            "duration": info.get("duration", 0.0),
            "duration_after_vad": info.get("duration_after_vad", info.get("duration", 0.0)),
            "segments": serialized_segments,
        }

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "segments": [asdict(segment) for segment in segments],
    }

def dump_transcript_json(segments, info, output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcription_to_dict(segments, info), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # model_path = Path(__file__).parent / "whisper_models" / "faster-whisper-medium"
    # loaded_model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
    loaded_model = WhisperModel("medium", device="cpu", compute_type="int8")
    videos_dir = Path(__file__).parent.parent / "eval_data" / "videos"
    video_paths = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
    )
    transcripts_dir = Path(__file__).parent.parent / "eval_data" / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    for video_path in video_paths:
        print(f"Processing {video_path}")
        audio_path = extract_audio_mp3(video_path)
        segments, info = transcribe_audio(audio_path, loaded_model)
        output_path = transcripts_dir / (video_path.stem + "_transcript.json")
        dump_transcript_json(segments, info, output_path)
        print(f"Saved transcript to {output_path}")

    # video = "data/videos/tag_reuters.com,2026_binary_LOV024206042026RP1-STREAM_700_16X9_MP4.mp4"
    # audio_path = extract_audio_mp3(video)
    # segments, info = transcribe_audio(audio_path, loaded_model)
    # dump_transcript_json(segments, info, "transcript.json")
    # print("Saved transcript to transcript.json")