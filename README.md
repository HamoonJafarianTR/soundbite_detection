# Soundbite Detection — Inference

Runs end-to-end soundbite detection on a single video using a pre-trained XGBoost model.

## Folder structure

```
soundbite_detection/
├── videos/                  ← put your input video(s) here
├── transcripts/             ← auto-created: Whisper transcript JSONs
├── detected_shots/          ← auto-created: shot boundary JSONs
├── whisper_models/
│   └── faster-whisper-medium/
├── model_xgboost.joblib     ← pre-trained classifier
├── predictions.json         ← auto-created: inference output
└── inferencePerVideo.py     ← main entry point
```

## Setup

```bash
pip install -r requirements.txt
```

> ffmpeg must also be installed and available on your PATH.
> - macOS: `brew install ffmpeg`
> - Linux: `sudo apt install ffmpeg`

## Usage

1. Drop your `.mp4` file into the `videos/` folder.
2. Open `inferencePerVideo.py` and set `VIDEO_PATH` in the `__main__` block:
   ```python
   VIDEO_PATH = VIDEOS_DIR / "your_video.mp4"
   ```
3. Run:
   ```bash
   python inferencePerVideo.py
   ```

## Pipeline steps

| Step | What happens | Output |
|------|-------------|--------|
| 1 | Audio extracted and transcribed with Whisper | `transcripts/<video>_transcript.json` |
| 2 | Shot boundaries detected (adaptive threshold + flash filter) | `detected_shots/<video>_weighted_adaptive_prediction.json` |
| 3 | Per-shot features extracted (face area, RMS energy, speech rate, etc.) | — |
| 4 | XGBoost classifier predicts soundbite probability for each shot | `predictions.json` |

## Output

`predictions.json` — array of shot objects, each with:

```json
{
  "video_id": "my_video",
  "shot_index": 3,
  "start_sec": 12.4,
  "end_sec": 18.7,
  "face_area_ratio": 0.14,
  "rms_mean": 0.031,
  "predicted_soundbite": 1,
  "soundbite_probability": 0.87
}
```
