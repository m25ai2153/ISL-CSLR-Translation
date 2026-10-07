import cv2
import numpy as np
import mediapipe as mp

# Access solutions directly from the top-level module attribute
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class KeypointExtractor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        # Pose landmarks (33 points * 3 = 99)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 3)

        # Face landmarks (468 points - taking top 21 points * 3 = 63)
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark[:21]]).flatten()
        else:
            face = np.zeros(21 * 3)

        # Left hand landmarks (21 points * 3 = 63)
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(21 * 3)

        # Right hand landmarks (21 points * 3 = 63)
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            rh = np.zeros(21 * 3)

        # Concatenate into unified feature vector (Dimension: 225)
        return np.concatenate([pose, face, lh, rh])

    def close(self):
        self.holistic.close()