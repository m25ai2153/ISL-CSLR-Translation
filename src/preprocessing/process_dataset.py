import os
import re
import glob
import json
import cv2
import numpy as np
from src.preprocessing.keypoint_extractor import KeypointExtractor

def resolve_gloss_name(video_path, raw_dir):
    """
    Determines the gloss tag based on folder structure or filename parsing.
    """
    rel_path = os.path.relpath(video_path, raw_dir)
    path_parts = rel_path.split(os.sep)

    # Rule 1: Video is inside a specific gloss subfolder (e.g., raw_dir/Alright/MVI_0037.mp4)
    if len(path_parts) > 1:
        gloss = path_parts[0]
    # Rule 2: Video is a loose file in raw_dir (e.g., raw_dir/rain-1.mp4, raw_dir/Rainbow.mp4)
    else:
        filename_without_ext = os.path.splitext(path_parts[0])[0]
        # Strip trailing digits/dashes/underscores (e.g., "rain-1" -> "rain", "rain-2" -> "rain")
        gloss = re.sub(r'[\-_]?\d+$', '', filename_without_ext)

    # Standardize to clean uppercase string
    gloss_clean = gloss.strip().replace(' ', '_').upper()
    return gloss_clean

def process_all_videos(raw_dir="data/raw_data", output_dir="data/processed_features"):
    """
    Processes all videos, extracts keypoint features, saves .npy files,
    and outputs a metadata mapping json.
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = KeypointExtractor()

    # Search for all supported video extensions recursively
    extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_paths = []
    for ext in extensions:
        video_paths.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

    if not video_paths:
        print(f"No video files found in '{raw_dir}'.")
        extractor.close()
        return

    print(f"Found {len(video_paths)} video(s) to process.")
    metadata = {}

    for idx, video_path in enumerate(video_paths):
        gloss = resolve_gloss_name(video_path, raw_dir)
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Unique output filename combining gloss and original video name
        output_filename = f"{gloss}_{filename}_{idx}.npy"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping existing: {output_filename}")
            metadata[output_filename] = gloss
            continue

        cap = cv2.VideoCapture(video_path)
        frame_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = extractor.extract_landmarks(frame)
            frame_features.append(landmarks)

        cap.release()

        if frame_features:
            feature_array = np.array(frame_features, dtype=np.float32)
            np.save(output_path, feature_array)
            metadata[output_filename] = gloss
            print(f"[{idx+1}/{len(video_paths)}] Saved: {output_filename} | Gloss: '{gloss}' | Shape: {feature_array.shape}")

    extractor.close()

    # Save dataset metadata mapping
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nProcessing complete! Metadata saved to {metadata_path}")

if __name__ == "__main__":
    process_all_videos()