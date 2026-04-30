FROM python:3.12.4-slim

WORKDIR /app

# System dependencies:
#   ffmpeg        – audio/video extraction (librosa + frame extraction)
#   libgl1        – OpenCV GUI/rendering backend
#   libglib2.0-0  – OpenCV runtime
#   libsndfile1   – audio file I/O used by librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model files
COPY *.py .
COPY models/ models/

# Output folders are created at runtime by the scripts themselves.
# AWS credentials must be supplied at runtime via environment variables
# or by mounting ~/.aws (see README).

ENTRYPOINT ["python", "inference.py"]
