import argparse
import json
import os
import sys
import time
import uuid
import urllib.request

import boto3

# Credentials and region are read from the environment:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN  (explicit keys)
#   AWS_PROFILE                                                   (named profile)
#   AWS_DEFAULT_REGION                                            (region)
# Or mount ~/.aws into the container (see .env.example).
transcribe_client = boto3.client("transcribe")

OUTPUT_DIR = "transcripts"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return bucket, key


def transcribe_video(s3_uri: str) -> str:
    """Transcribe a video at *s3_uri* and save the result to OUTPUT_DIR.

    Returns the local path to the saved transcript JSON.
    Skips and returns the path immediately if a transcript already exists.
    """
    _, key = parse_s3_uri(s3_uri)
    video_name = os.path.basename(key)
    video_stem = os.path.splitext(video_name)[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{video_stem}_transcript_raw.json")

    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        print(f"Skip (transcript exists): {output_file}")
        return output_file

    job_name = f"transcribe-{uuid.uuid4().hex[:8]}"
    print(f"Starting transcription job: {job_name}")
    print(f"Media: {s3_uri}")
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        MediaFormat="mp4",
        IdentifyLanguage=True,
    )

    while True:
        resp = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
        print(f"  Status: {status}")
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(15)

    if status == "FAILED":
        reason = resp["TranscriptionJob"].get("FailureReason", "unknown")
        sys.exit(f"Transcription failed: {reason}")

    transcript_url = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    print(f"Downloading transcript from {transcript_url}")

    with urllib.request.urlopen(transcript_url) as response:
        transcript_data = json.loads(response.read().decode())

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved to {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a video from S3 and save the transcript to inference/."
    )
    parser.add_argument(
        "s3_url",
        help="S3 URL of the video to transcribe (e.g. s3://bucket/path/video.mp4)",
    )
    args = parser.parse_args()

    try:
        transcribe_video(args.s3_url)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()