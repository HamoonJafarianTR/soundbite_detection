"""
AWS Transcribe script with automatic language detection.

Usage:
    python aws_transcribe.py <s3_uri> [--job-name NAME] [--output-bucket BUCKET] [--region REGION]

Example:
    python aws_transcribe.py s3://my-bucket/videos/clip.mp4 --output-bucket my-bucket --region us-east-1

Requirements:
    pip install boto3

AWS credentials must be configured via environment variables, ~/.aws/credentials,
or an IAM role. Required permissions: transcribe:StartTranscriptionJob,
transcribe:GetTranscriptionJob, s3:GetObject (for reading the result).
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_DEFAULT_PROFILE = os.getenv("AWS_PROFILE", None)
_DEFAULT_REGION = os.getenv("AWS_REGION", "eu-west-1")




def start_transcription_job(
    s3_uri: str,
    job_name: str,
    output_bucket: str,
    output_key: str,
    region: str,
    profile: str | None = None,
) -> dict:
    """Start an AWS Transcribe job with automatic language identification."""
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("transcribe")

    params = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": s3_uri},
        "IdentifyMultipleLanguages": False,
        "IdentifyLanguage": True,
        # "LanguageOptions": language_options,
        "OutputBucketName": output_bucket,
        "OutputKey": output_key,
        "Settings": {
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": 10,
        },
    }

    response = client.start_transcription_job(**params)
    return response["TranscriptionJob"]


def wait_for_job(job_name: str, region: str, poll_interval: int = 10, profile: str | None = None) -> dict:
    """Poll until the transcription job completes or fails."""
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("transcribe")

    print(f"Waiting for job '{job_name}' ", end="", flush=True)
    while True:
        response = client.get_transcription_job(TranscriptionJobName=job_name)
        job = response["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]

        if status == "COMPLETED":
            print(" done.")
            return job
        elif status == "FAILED":
            reason = job.get("FailureReason", "Unknown")
            raise RuntimeError(f"Transcription job failed: {reason}")
        else:
            print(".", end="", flush=True)
            time.sleep(poll_interval)


def fetch_transcript(bucket: str, key: str, region: str, profile: str | None = None) -> dict:
    """Download the transcript JSON directly from S3 using boto3 (handles private buckets)."""
    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def parse_transcript(transcript_data: dict) -> list[dict]:
    """
    Convert AWS Transcribe output into a flat list of speaker segments.

    Returns a list of dicts with keys:
        speaker, start_sec, end_sec, transcript
    """
    results = transcript_data.get("results", {})
    items = results.get("items", [])
    speaker_labels = results.get("speaker_labels") or {}

    # Build a map: (start_time, end_time) -> speaker
    speaker_map: dict[tuple[str, str], str] = {}
    for seg in speaker_labels.get("segments", []):
        speaker = seg["speaker_label"]
        for item in seg.get("items", []):
            key = (item.get("start_time", ""), item.get("end_time", ""))
            speaker_map[key] = speaker

    # Group consecutive words by speaker into segments
    segments: list[dict] = []
    current: dict | None = None

    for item in items:
        if item["type"] != "pronunciation":
            # Punctuation – append to current segment text if one exists
            if current and item.get("alternatives"):
                current["transcript"] += item["alternatives"][0].get("content", "")
            continue

        start = item.get("start_time", "")
        end = item.get("end_time", "")
        speaker = speaker_map.get((start, end), "unknown")
        word = item["alternatives"][0].get("content", "") if item.get("alternatives") else ""

        if current is None or current["speaker"] != speaker:
            if current:
                segments.append(current)
            current = {
                "speaker": speaker,
                "start_sec": float(start) if start else 0.0,
                "end_sec": float(end) if end else 0.0,
                "transcript": word,
            }
        else:
            current["transcript"] += f" {word}"
            current["end_sec"] = float(end) if end else current["end_sec"]

    if current:
        segments.append(current)

    return segments


def transcribe(
    s3_uri: str,
    output_bucket: str,
    job_name: str | None = None,
    region: str = "us-east-1",
    profile: str | None = None,
    save_raw: Path | None = None,
    save_segments: Path | None = None,
) -> list[dict]:
    """
    Full pipeline: start job → wait → fetch → parse.

    Returns parsed segments (list of dicts).
    Optionally saves raw AWS JSON to `save_raw` and parsed segments to `save_segments`.
    """
    if job_name is None:
        job_name = f"transcribe-{uuid.uuid4().hex[:8]}"

    output_key = f"transcripts/{job_name}.json"

    print(f"Starting transcription job: {job_name}")
    print(f"  Input : {s3_uri}")
    print(f"  Output: s3://{output_bucket}/{output_key}")
    print(f"  Region: {region}")
    print(f"  Profile: {profile or 'default'}")

    try:
        start_transcription_job(
            s3_uri=s3_uri,
            job_name=job_name,
            output_bucket=output_bucket,
            output_key=output_key,
            region=region,
            profile=profile,
        )
    except ClientError as e:
        raise RuntimeError(f"Failed to start transcription job: {e}") from e

    job = wait_for_job(job_name, region, profile=profile)

    detected_language = job.get("LanguageCode", "unknown")
    confidence = job.get("IdentifiedLanguageScore", None)
    print(f"Detected language: {detected_language}" + (f" (confidence: {confidence:.2f})" if confidence else ""))

    transcript_uri = job["Transcript"]["TranscriptFileUri"]  # noqa: F841 – kept for debugging
    raw_data = fetch_transcript(output_bucket, output_key, region, profile)

    if save_raw:
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2))
        print(f"Raw transcript saved: {save_raw}")

    segments = parse_transcript(raw_data)

    if save_segments:
        save_segments.parent.mkdir(parents=True, exist_ok=True)
        save_segments.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        print(f"Parsed segments saved: {save_segments}")

    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a media file using AWS Transcribe with automatic language detection.")
    parser.add_argument("s3_uri", help="S3 URI of the input media file, e.g. s3://bucket/key.mp4")
    parser.add_argument("--job-name", default=None, help="Transcription job name (auto-generated if omitted)")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket where the transcript JSON will be written")
    parser.add_argument("--profile", default=_DEFAULT_PROFILE, help="AWS CLI profile name (default: AWS_PROFILE from .env)")
    parser.add_argument("--region", default=_DEFAULT_REGION, help="AWS region (default: AWS_REGION from .env)")
    parser.add_argument("--output-dir", default="transcripts", help="Local directory to save output files (default: transcripts/)")
    args = parser.parse_args()

    # Derive a base name from the S3 key for local output files
    s3_key_stem = Path(args.s3_uri.split("/")[-1]).stem
    job_name = args.job_name or f"transcribe-{s3_key_stem}-{uuid.uuid4().hex[:6]}"
    output_dir = Path(args.output_dir)

    segments = transcribe(
        s3_uri=args.s3_uri,
        output_bucket=args.output_bucket,
        job_name=job_name,
        region=args.region,
        profile=args.profile,
        save_raw=output_dir / f"{job_name}_raw.json",
        save_segments=output_dir / f"{job_name}_segments.json",
    )

    print(f"\nTotal segments: {len(segments)}")
    for seg in segments[:5]:
        print(f"  [{seg['start_sec']:.2f}s – {seg['end_sec']:.2f}s] {seg['speaker']}: {seg['transcript'][:80]}")
    if len(segments) > 5:
        print(f"  … ({len(segments) - 5} more)")


if __name__ == "__main__":
    main()
