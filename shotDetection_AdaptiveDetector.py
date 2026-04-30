from scenedetect import open_video, SceneManager, AdaptiveDetector
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.detectors.threshold_detector import ThresholdDetector
import glob
import json
import os
import cv2
import numpy as np
import time

def shot_detection(video_path, adaptive_threshold, min_scene_len, window_width, min_content_val):
    video = open_video(video_path)
    scene_manager = SceneManager()
    weights = ContentDetector.Components(
        delta_hue=1.0,
        delta_sat=1.0,
        delta_lum=0,
        delta_edges=1,
    )
    scene_manager.add_detector(
    AdaptiveDetector(
        adaptive_threshold,
        min_scene_len,
        window_width,
        min_content_val,
        weights=weights,
    ))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()

def filter_flash_cuts(video_path, scene_list, look_back, look_ahead,
                      similarity_threshold, flash_brightness_jump):
    """Remove false cuts caused by camera flashes.

    Only considers a cut a flash candidate if the frame at the cut point is
    significantly brighter than the frame before it (brightness spike).
    For flash candidates, compares frames before and after the cut — if they
    are visually similar the cut is discarded as a flash artifact.

    Args:
        look_back / look_ahead: how many frames to skip past the flash.
        similarity_threshold: max mean-absolute pixel difference (0-255 scale)
            for two frames to be considered "same scene".
        flash_brightness_jump: minimum increase in mean brightness between the
            frame before the cut and the cut frame to flag it as a flash candidate.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    filtered = []

    for start, end in scene_list:
        cut_frame = start.get_frames()
        if cut_frame == 0:
            filtered.append((start, end))
            continue

        before_idx = max(0, cut_frame - look_back)
        after_idx = min(total_frames - 1, cut_frame + look_ahead)

        cap.set(cv2.CAP_PROP_POS_FRAMES, before_idx)
        ret_b, frame_before = cap.read()

        peak_brightness = -1
        peak_frame_before = None
        for f in [cut_frame - 2, cut_frame - 1, cut_frame, cut_frame + 1, cut_frame + 2]:
            f = max(0, min(total_frames - 1, f))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if ret:
                b = np.mean(frame)
                if b > peak_brightness:
                    peak_brightness = b
                    peak_idx = f

        if peak_brightness < 0 or not ret_b:
            filtered.append((start, end))
            continue

        ref_idx = max(0, peak_idx - look_back)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ret_ref, frame_ref = cap.read()
        if not ret_ref:
            filtered.append((start, end))
            continue

        brightness_before = np.mean(frame_ref)
        brightness_at_cut = peak_brightness
        jump = brightness_at_cut - brightness_before
        tc = frame_to_smpte(start)
        is_flash_candidate = jump > flash_brightness_jump

        if not is_flash_candidate:
            filtered.append((start, end))
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, after_idx)
        ret_a, frame_after = cap.read()

        if ret_a:
            diff = np.mean(np.abs(
                frame_before.astype(np.float32) - frame_after.astype(np.float32)
            ))
            if diff > similarity_threshold:
                filtered.append((start, end))
        else:
            filtered.append((start, end))

    cap.release()
    return filtered


def frame_to_smpte(timecode):
    """Convert a FrameTimecode to HH:MM:SS:FF format."""
    frame_num = timecode.get_frames()
    fps = round(timecode.framerate)
    total_seconds = frame_num // fps
    frames = frame_num % fps
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # print("frame rate: ", fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def smpte_hhmmssff_to_seconds(smpte: str, fps: float) -> float:
    """Convert ``HH:MM:SS:FF`` (same style as ``frame_to_smpte``) to seconds, e.g. ``10.356``, ``117.2``.

    Use the same ``fps`` as when building the timecode string (``round(timecode.framerate)`` from
    ``frame_to_smpte``) so the value matches that border exactly.
    """
    parts = smpte.split(":")
    if len(parts) != 4:
        raise ValueError(f"expected HH:MM:SS:FF, got {smpte!r}")
    h, m, s, ff = (int(p) for p in parts)
    sec = h * 3600.0 + m * 60.0 + s + ff / float(fps)
    # Sub-frame precision not in SMPTE; 6 decimals clears float noise (e.g. 117.2 not 117.1999999)
    return round(sec, 6)


def weighted_adaptive_threshold_shot_detection(
    video_path,
    adaptive_threshold=2.5,
    min_scene_len=30,
    window_width=4,
    min_content_val=20,
    ):
    flash_similarity_threshold = 20
    flash_brightness_jump = 20
    look_back = 2
    look_ahead = 2
    results = {
    "adaptive_threshold": adaptive_threshold,
    "min_scene_len": min_scene_len,
    "window_width": window_width,
    "min_content_val": min_content_val,
    "flash_similarity_threshold" : flash_similarity_threshold,
    "look_back" : look_back,
    "look_ahead" : look_ahead,
    "flash_brightness_jump" : flash_brightness_jump
    }
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    OUTPUT_FILE = f"detected_shots/{video_stem}_weighted_adaptive_prediction.json"
    if os.path.exists(OUTPUT_FILE):
        print(f"{video_name}: shots already detected, skipping ({OUTPUT_FILE})")
        return
    adaptive_scenes = shot_detection(video_path, adaptive_threshold, min_scene_len, window_width, min_content_val)
    adaptive_scenes = filter_flash_cuts(video_path, adaptive_scenes, look_back, look_ahead,flash_similarity_threshold, flash_brightness_jump)
    timecodes = [frame_to_smpte(start) for start, _ in adaptive_scenes]
    fps_smpte = (
        float(round(adaptive_scenes[0][0].framerate))
        if adaptive_scenes
        else 25.0
    )
    if fps_smpte <= 0:
        fps_smpte = 25.0
    timecodes_seconds = [
        smpte_hhmmssff_to_seconds(tc, fps_smpte) for tc in timecodes
    ]
    results[video_name] = {
        "timecode": timecodes,
        "timecodes_seconds": timecodes_seconds,
    }
    print(f"{video_name}: {len(timecodes)} shots")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    videos = glob.glob("videos/*.mp4")
    for video in videos:
        weighted_adaptive_threshold_shot_detection(
            video_path = video,
            adaptive_threshold=2.5,
            min_scene_len=30,
            window_width=4, # farames before and after the shot boundary to be considered as a shot boundary
            min_content_val=20, # Min of difference between two frames to be considered as a shot boundary
        )
        time.sleep(5)
