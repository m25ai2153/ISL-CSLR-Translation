import os
import glob
import cv2
import numpy as np
from keypoint_extractor import KeypointExtractor

def process_all_videos(raw_dir="data/raw_data", output_dir="data/processed_features"):
    """
    Iterates through all video files in raw_dir, extracts frame-level keypoints,
    and saves normalized numpy arrays (.npy) into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = KeypointExtractor()

    # Search for MP4 and AVI files
    video_paths = glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True) + \
                  glob.glob(os.path.join(raw_dir, "**", "*.avi"), recursive=True)

    if not video_paths:
        print(f"No video files found in '{raw_dir}'. Place raw video clips there to process.")
        extractor.close()
        return

    print(f"Found {len(video_paths)} video(s) to process.")

    for video_path in video_paths:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.npy")

        if os.path.exists(output_path):
            print(f"Skipping '{filename}', already processed.")
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
            print(f"Saved: {output_path} | Shape: {feature_array.shape}")
        else:
            print(f"Warning: No frames read from {video_path}")

    extractor.close()
    print("Batch feature extraction complete.")

if __name__ == "__main__":
    process_all_videos()