# Soundbite Detection

Run soundbite detection inference on a single video. This pipeline evaluates a video on a per-shot basis to extract features and predict soundbites. It relies on a pre-generated video transcript and a list of detected shots.

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install FFmpeg** (required for audio feature extraction and video processing):
   ```bash
   brew install ffmpeg   # macOS
   # OR
   sudo apt install ffmpeg # Ubuntu
   ```

## Input Requirements

To run the pipeline, you need five primary inputs.

### 1. The Video File (`--video`)
The source video file (e.g., `.mp4`) that you want to analyze.

### 2. Transcript JSON (`--transcript`)
A JSON file containing the transcription of the video (Whisper format). The file must contain a `segments` array. Each segment and the words inside it require the following fields for accurate feature extraction:

- `segments`: List of segment objects.
  - `start`: Start time of the segment in seconds.
  - `end`: End time of the segment in seconds.
  - `text`: (Fallback) Full text of the segment, used if `words` are missing.
  - `words`: List of word-level objects (crucial for accurate metrics).
    - `start`: Word start time in seconds.
    - `end`: Word end time in seconds.
    - `probability`: Word-level confidence score (used to calculate `confidence_mean`).
    - `word`: The actual text string of the word.

**Word Overlap Handling (Shot Borders)**  
Because shots (cuts) and spoken words do not always perfectly align, the script calculates an **overlap fraction** for words that fall on the boundary of a shot:
- **Fully Inside:** Included (`overlap == 1.0`).
- **Fully Outside:** Excluded (`overlap == 0.0`).
- **On the Border:** If a word straddles the cut line, it is included in the text and word count only if the overlap fraction is `>= 0.5` (i.e., at least 50% of the spoken word falls inside the shot).

### 3. Detected Shots JSON (`--shots`)
A JSON file containing the detected shot boundaries (cuts). The script looks for a list of timecodes that represent the start times of each shot. The structure can either have the video filename as the top-level key or contain the timecodes directly at the root.

The script expects the timecodes to be provided under one of these two keys:
- `timecodes_seconds`: A list of float values representing shot boundaries in seconds (e.g., `[0.0, 5.2, 12.8]`). **(Preferred Format)**
- `timecode`: A list of formatted timecode strings (e.g., `["00:00:00:00", "00:00:05:05"]`). The script will automatically parse these into seconds based on the video's framerate.

### 4. Trained Model (`--model`)
Path to a custom trained XGBoost model `.joblib` file (e.g., `models/model_xgboost_v10.joblib`).

### 5. Output Path (`--output`)
Path to save the output predictions JSON (e.g., `predictions.json`).

## Running the Pipeline

Run inference by explicitly providing the paths to all your inputs:

```bash
python inferencePerVideo.py \
  --video videos/your_video.mp4 \
  --transcript transcripts/your_video_transcript.json \
  --shots detected_shots/your_video_shots.json \
  --model models/model_xgboost_v10.joblib \
  --output custom_predictions.json
```

*(Note: If you do not provide the paths via command-line arguments, the script will interactively prompt you for them in the terminal.)*

## Output

The script generates a final JSON file (`predictions.json` by default). This file contains a list of every shot in the video alongside:
- The extracted features (speech rate, face presence, rms standard deviation, etc.).
- `predicted_soundbite`: `1` if it is a soundbite, `0` if it is not.
- `soundbite_probability`: The model's confidence score that the shot is a soundbite.