# Soundbite Detector

Automatically predicts soundbite segments in a news video using audio, transcript, shot, and visual features.

Given an S3 URL to a video, the pipeline:
1. Downloads the video from S3
2. Transcribes it with AWS Transcribe
3. Runs shot detection
4. Predicts soundbite timestamps and saves them to `predictions/`

---

## Prerequisites

### 1. Check Docker is installed

```bash
docker --version
```

You should see something like `Docker version 27.x.x`. If not, install Docker Desktop from https://www.docker.com/products/docker-desktop.

Make sure Docker is running before continuing (open the Docker Desktop app).

### 2. Log in with cloud-tool

```bash
cloud-tool login
```

This sets up your AWS credentials automatically. To verify it worked:

```bash
aws sts get-caller-identity --profile tr-reuters-devops-sandbox
```

You should see your account ID and user ARN. If you see an error, your login did not work — try logging in again.

---

## Setup

### Clone the repository

```bash
git clone https://github.com/your-org/soundbite-detector.git
cd soundbite-detector
```

### Build the Docker image

```bash
docker build -t soundbite-inference .
```

This will take a few minutes the first time (downloading base image and installing dependencies). Subsequent builds are faster thanks to Docker layer caching.

---

## Running the pipeline

```bash
docker run \
  -v ~/.aws:/root/.aws:ro \
  -e AWS_PROFILE=tr-reuters-devops-sandbox \
  -v $(pwd)/predictions:/app/predictions \
  soundbite-inference \
  --video s3://your-bucket/path/to/video.mp4
```

**Replace** `s3://your-bucket/path/to/video.mp4` with the S3 URI of your video.

- `-v ~/.aws:/root/.aws:ro` — passes your cloud-tool credentials into the container (read-only)
- `-e AWS_PROFILE=tr-reuters-devops-sandbox` — tells boto3 which profile to use
- `-v $(pwd)/predictions:/app/predictions` — saves the results to a `predictions/` folder on your machine

### Example

```bash
docker run \
  -v ~/.aws:/root/.aws:ro \
  -e AWS_PROFILE=tr-reuters-devops-sandbox \
  -v $(pwd)/predictions:/app/predictions \
  soundbite-inference \
  --video s3://a206709-archive/Hamoon/temp/IRAN-CRISIS-USA-CANADA.MP4
```

---

## Output

Results are saved to `predictions/<video-name>_soundbites.json`:

```json
{
  "video": "IRAN-CRISIS-USA-CANADA.MP4",
  "soundbites": [
    {
      "soundbite_start": 12.480,
      "soundbite_end": 35.160,
      "confidence": 0.87
    }
  ],
  "segments": [ ... ]
}
```

---

## Intermediate files

| Folder | Contents |
|---|---|
| `videos/` | Downloaded video file |
| `transcripts/` | Raw AWS Transcribe JSON |
| `detected_shots/` | Shot boundary JSON |
| `predictions/` | Final soundbite predictions |

All steps are skipped automatically if their output already exists, so re-running on the same video is fast.

---

## Troubleshooting

**`Unable to locate credentials`**
→ Your cloud-tool session has expired. Run `cloud-tool login` and verify with `aws sts get-caller-identity --profile tr-reuters-devops-sandbox`.

**`Access Denied`**
→ You are authenticated but don't have permission to the S3 bucket or AWS Transcribe. Contact your AWS admin.

**`Expected s3:// URI`**
→ You passed an HTTPS URL instead of an S3 URI. Convert it:
- HTTPS: `https://my-bucket.s3.eu-west-1.amazonaws.com/path/video.mp4`
- S3 URI: `s3://my-bucket/path/video.mp4`

**Docker build fails on `apt-get`**
→ Make sure Docker Desktop is running and you have an internet connection.
