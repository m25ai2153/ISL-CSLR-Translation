import os
import torch
from torch.utils.data import DataLoader
from src.preprocessing.dataset import ISLDataset, pad_collate_fn
from src.models.trainer import train_model

def main():
    # Define Gloss Vocabulary Mapping (Blank token reserved at index 0)
    label_map = {
        "HELLO": 1,
        "THANK_YOU": 2,
        "PLEASE": 3,
        "HELP": 4,
        "NAME": 5
    }

    features_dir = "data/processed_features"

    if not os.path.exists(features_dir) or not os.listdir(features_dir):
        print(f"No processed feature arrays found in '{features_dir}'. Run process_dataset.py first.")
        return

    # Dataset & DataLoader instantiation
    dataset = ISLDataset(features_dir=features_dir, label_map=label_map)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)

    # Train BiLSTM sequence model
    train_model(
        train_loader=train_loader,
        val_loader=train_loader,  # Replace with a split validation loader when scaled
        input_dim=225,             # 75 keypoints * 3 coordinates (x, y, z)
        num_classes=len(label_map) + 1,  # Number of classes + CTC blank class
        epochs=20,
        lr=1e-3
    )

if __name__ == "__main__":
    main()