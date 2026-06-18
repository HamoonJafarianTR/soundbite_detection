# Soundbite Detection

Run end-to-end soundbite detection on a single video.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install [ffmpeg](https://ffmpeg.org/) (required for audio/video processing):

```bash
brew install ffmpeg   # macOS
```

## Run

1. Put your `.mp4` in `videos/`.
2. Run inference:

```bash
python inferencePerVideo.py --video videos/your_video.mp4
```

Optional flags:

```bash
python inferencePerVideo.py \
  --video videos/your_video.mp4 \
  --model models/model_xgboost_v10.joblib \
  --output predictions.json \
  --device cpu
```

Whisper uses the `medium` model by default (downloaded on first run). Override with `--whisper-model`.

## Output

- `transcripts/<video>_transcript.json` — Whisper transcript
- `detected_shots/<video>_weighted_adaptive_prediction.json` — shot boundaries
- `predictions.json` — per-shot soundbite predictions
