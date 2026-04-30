import argparse
import os
import sys

import boto3

# Credentials and region are read from the environment:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN  (explicit keys)
#   AWS_PROFILE                                                   (named profile)
#   AWS_DEFAULT_REGION                                            (region)
# Or mount ~/.aws into the container (see .env.example).
s3 = boto3.client("s3")

OUTPUT_DIR = "videos"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    rest = uri[5:]
    b, _, key = rest.partition("/")
    if not b or not key:
        raise ValueError(f"Invalid s3 URI: {uri!r}")
    return b, key


def remote_size(b: str, key: str) -> int | None:
    """Return ``ContentLength`` of the S3 object, or ``None`` if it can't be read."""
    try:
        return int(s3.head_object(Bucket=b, Key=key)["ContentLength"])
    except Exception as e:
        print(f"Warning: head_object failed for s3://{b}/{key}: {e}", file=sys.stderr)
        return None


def download_video(s3_uri: str) -> str:
    """Download a single video from *s3_uri* into OUTPUT_DIR and return the local path."""
    uri_bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = os.path.basename(key)
    dest = os.path.join(OUTPUT_DIR, filename)

    if os.path.isfile(dest):
        local = os.path.getsize(dest)
        remote = remote_size(uri_bucket, key)
        if remote is None or local == remote:
            print(f"Skip (exists, {local} bytes): {dest}")
            return dest
        print(f"Re-downloading {dest}: local {local} bytes != remote {remote} bytes")

    print(f"Downloading {s3_uri} -> {dest}")
    s3.download_file(uri_bucket, key, dest)
    print(f"Done. Saved to {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a video from S3 into the inference/ folder."
    )
    parser.add_argument(
        "s3_url",
        help="S3 URL of the video to download (e.g. s3://bucket/path/video.mp4)",
    )
    args = parser.parse_args()

    try:
        download_video(args.s3_url)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
