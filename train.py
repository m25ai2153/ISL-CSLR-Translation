import os
import json
import torch
from torch.utils.data import DataLoader
from src.preprocessing.dataset import ISLDataset, pad_collate_fn, split_dataset
from src.models.trainer import train_model

def main():
    features_dir = "data/processed_features"

    if not os.path.exists(features_dir) or not os.listdir(features_dir):
        print(f"No processed feature arrays found in '{features_dir}'. Run process_dataset.py first.")
        return

    # Instantiate Dataset & auto-detect gloss vocabulary
    dataset = ISLDataset(features_dir=features_dir)
    
    # Save active label map to checkpoints directory for real-time inference
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/label_map.json", "w") as f:
        json.dump(dataset.idx_to_gloss, f, indent=4)

    # Perform train/validation split
    train_set, val_set = split_dataset(dataset, val_ratio=0.2)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)

    # Train Model (Number of classes = Unique Glosses + CTC Blank)
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=225,
        num_classes=len(dataset.gloss_to_idx) + 1,
        epochs=30,
        lr=1e-3
    )

if __name__ == "__main__":
    main()