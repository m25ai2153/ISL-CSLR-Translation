import cv2
import numpy as np
import mediapipe as mp

class KeypointExtractor:
    """
    Extracts and normalizes 3D keypoints from video frames using MediaPipe Holistic.
    Applies chest-centric translation and shoulder-width scale normalization.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_landmarks(self, frame):
        """
        Extracts 3D coordinates for Pose, Left Hand, Right Hand, and Face.
        Returns a flattened, normalized feature vector for a single frame.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        # 1. Pose landmarks (33 points * 3 = 99 features)
        pose = np.zeros((33, 3))
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

        # 2. Left Hand landmarks (21 points * 3 = 63 features)
        lh = np.zeros((21, 3))
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

        # 3. Right Hand landmarks (21 points * 3 = 63 features)
        rh = np.zeros((21, 3))
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

        # 4. Normalize coordinates relative to chest position and shoulder width
        normalized_features = self._normalize_landmarks(pose, lh, rh)
        return normalized_features

    def _normalize_landmarks(self, pose, lh, rh):
        """
        Applies Translation (Chest Origin) and Scaling (Shoulder Distance) Invariance.
        """
        # Pose indices: 11 = Left Shoulder, 12 = Right Shoulder
        left_shoulder = pose[11]
        right_shoulder = pose[12]

        # Calculate Chest Midpoint (Origin)
        chest_origin = (left_shoulder + right_shoulder) / 2.0

        # Calculate Shoulder Width (Scale factor)
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_dist == 0:
            shoulder_dist = 1.0  # Prevent division by zero

        # Apply translation and scaling to Pose, Left Hand, and Right Hand
        pose_norm = (pose - chest_origin) / shoulder_dist
        lh_norm = (lh - chest_origin) / shoulder_dist if np.any(lh) else lh
        rh_norm = (rh - chest_origin) / shoulder_dist if np.any(rh) else rh

        # Flatten into a single 1D feature vector for this frame
        return np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])

    def close(self):
        self.holistic.close()