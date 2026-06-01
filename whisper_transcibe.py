from faster_whisper import WhisperModel
import subprocess
import tempfile
from pathlib import Path
import json
from dataclasses import asdict

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

def transcribe_audio(audio_file, loaded_model):
    segments, info = loaded_model.transcribe(
        audio_file, condition_on_previous_text=False, beam_size=5, word_timestamps=True,
    )
    print(
        f"Detected language {info.language} with probability {info.language_probability:.4f}"
    )
    return segments, info

def transcription_to_dict(segments, info):
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
    model_path = Path(__file__).parent / "whisper_models" / "faster-whisper-medium"
    loaded_model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
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