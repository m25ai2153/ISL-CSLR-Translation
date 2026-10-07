import os
import cv2
import json
import torch
import collections
import numpy as np
from src.preprocessing.keypoint_extractor import KeypointExtractor
from src.models.cslr_bilstm import LightweightCSLR
from src.translation.decoder import CTCDecoder

def run_realtime():
    label_map_path = "checkpoints/label_map.json"
    checkpoint_path = "checkpoints/best_isl_model.pth"

    if not os.path.exists(label_map_path) or not os.path.exists(checkpoint_path):
        print("Missing model checkpoint or label map. Please run 'python train.py' first.")
        return

    # Load dynamic index-to-gloss dictionary (JSON keys load as strings, convert to int)
    with open(label_map_path, "r") as f:
        raw_idx_map = json.load(f)
    idx_to_gloss = {int(k): v for k, v in raw_idx_map.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(idx_to_gloss) + 1  # Glosses + CTC Blank
    
    model = LightweightCSLR(input_dim=288, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    extractor = KeypointExtractor()
    decoder = CTCDecoder(idx_to_gloss)
    
    sequence_buffer = collections.deque(maxlen=30)  # Sliding temporal window (~1 sec at 30 fps)
    cap = cv2.VideoCapture(0)

    print("Starting webcam stream... Press 'q' to stop.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features and append to sequence queue
        landmarks = extractor.extract_landmarks(frame)
        sequence_buffer.append(landmarks)

        # Run inference once enough frames have accumulated
        if len(sequence_buffer) >= 15:
            input_tensor = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)[0]
                decoded_glosses = decoder.decode_greedy(logits)
                text_output = " ".join(decoded_glosses)
        else:
            text_output = "Buffering sequence..."

        # Visual UI Overlay
        cv2.putText(frame, f"Translation: {text_output}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("ISL-CSLR Realtime Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == "__main__":
    run_realtime()